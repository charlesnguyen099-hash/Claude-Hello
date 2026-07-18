"""
Trade Executor — thực thi lệnh và quản lý vị thế
- Dat lenh Market (fill ngay, khong bi cancel), sau do set SL+TP bang set_trading_stop (position-level)
- Level 1 (BREAKEVEN_TRIGGER=60%): lock SL TAI muc 60% TP1 (khong phai ve entry) — dam bao loi toi thieu 60% TP1
- Level 2 (PARTIAL_CLOSE_TRIGGER=75%): dong 50% vi the, cap nhat TP len TP2, xac nhan SL lock 60%
- Level 3: 50% con lai chay den TP2 voi zero downside risk (SL = breakeven)
- Anti-whipsaw: khong force-close vi the < 30 phut vi signal dao chieu
- Tu dong dong lenh khi signal dao chieu (sau 30 phut hoac PnL < -20% margin)
- entry_price cap nhat bang live bid/ask trc khi compute_trade (khong dung gia nen stale)

Defensive layers (execution):
  1. Live bid/ask update — entry_price bang gia real-time thay vi nen 15m cu (14 phut)
  2. Spread check — abort neu spread > nguong (thanh khoan kem, slip lon)
  3. Market order — fill ngay, khong miss vi gia chay di (thay the IOC Limit bi cancel)
  4. SL verification + re-arm — dam bao SL luon active sau khi lenh vao
  5. Partial close race condition fix — chi update flag sau khi close thanh cong
  6. Periodic SL health check — re-arm SL neu bi huy trong khi quan ly vi the
"""

import logging
import math
import time

from client import BybitClient
from risk_manager import RiskManager, TradeParams
from strategies.base import Signal
from bot_logger import BotLogger
import config

logger = logging.getLogger(__name__)


def _fval(d: dict, key: str, default: float = 0.0) -> float:
    """Safe float parse — Bybit tra ve '' (chuoi rong) khi field khong co gia tri."""
    v = d.get(key, default)
    try:
        return float(v) if v != "" else default
    except (TypeError, ValueError):
        return default


class Executor:
    def __init__(self, client: BybitClient, risk_mgr: RiskManager, bot_logger: BotLogger):
        self.client    = client
        self.risk_mgr  = risk_mgr
        self.logger    = bot_logger

        self._partial_closed: dict[str, bool]  = {}
        self._breakeven_set: dict[str, bool]   = {}
        self._atr: dict[str, float]            = {}
        self._tp1_price: dict[str, float]      = {}   # tp1 de detect partial close da xay ra sau restart
        self._tp2_price: dict[str, float]      = {}
        self._sl_price: dict[str, float]       = {}   # sl ban dau de re-arm neu mat
        self._open_time: dict[str, float]      = {}
        self._sl_verified: dict[str, bool]     = {}   # da verify SL sau fill chua
        self._tick_size: dict[str, float]      = {}   # tick size de round be/tp2 dung exchange format

    def execute_signal(
        self,
        symbol: str,
        signal: Signal,
        equity: float,
        open_positions: list[dict],
        is_priority: bool = False,
    ):
        """Xử lý signal mới — vào lệnh nếu đủ điều kiện."""
        if signal.direction == 0:
            return

        existing = [p for p in open_positions if p["symbol"] == symbol]
        if existing:
            pos_side = existing[0]["side"]
            signal_side = "Buy" if signal.direction == 1 else "Sell"
            if pos_side == signal_side:
                return

            # Anti-whipsaw: khong force-close neu position mo < MIN_HOLD_SECONDS
            MIN_HOLD_SECONDS = 1800  # 30 phut = 2 nen 15m
            pos_pnl_pct = 0.0
            try:
                notional = _fval(existing[0], "positionValue", 1) or 1
                lev      = _fval(existing[0], "leverage", 1) or 1
                margin   = notional / lev
                pnl      = _fval(existing[0], "unrealisedPnl")
                pos_pnl_pct = pnl / margin if margin > 0 else 0
            except Exception:
                pass

            held_seconds = 0
            exchange_created_ms = int(existing[0].get("createdTime", 0))
            if exchange_created_ms > 0:
                held_seconds = time.time() - exchange_created_ms / 1000
            else:
                open_ts = self._open_time.get(symbol, 0)
                if open_ts > 0:
                    held_seconds = time.time() - open_ts

            if held_seconds < MIN_HOLD_SECONDS and pos_pnl_pct > -0.20:
                logger.info(
                    f"{symbol}: Signal reversal — SKIP (held={held_seconds:.0f}s "
                    f"< {MIN_HOLD_SECONDS}s, PnL={pos_pnl_pct*100:.1f}%)"
                )
                return

            logger.info(f"{symbol}: Signal reversal -> close {pos_side} (held={held_seconds:.0f}s, "
                        f"PnL={pos_pnl_pct*100:.1f}%) -> enter {signal_side}")
            self._close_position(existing[0])
            time.sleep(0.5)

        # Cap nhat entry_price bang gia live (bid/ask real-time) thay vi gia dong nen 15m (cu toi 14 phut)
        # Day la nguyen nhan chinh khien stale check block het lenh trong trending market
        bid_live, ask_live = self.client.get_bid_ask(symbol)
        if bid_live <= 0 or ask_live <= 0:
            logger.warning(f"{symbol}: SKIP — khong lay duoc bid/ask live, bo qua")
            return
        # Long: entry tai ask (mua ngay gia hien tai); Short: entry tai bid
        signal.entry_price = ask_live if signal.direction == 1 else bid_live

        params = self.risk_mgr.compute_trade(signal, equity, open_positions, is_priority=is_priority)
        if not params:
            return

        params.symbol = symbol
        self._enter_trade(symbol, signal, params)

    def _enter_trade(self, symbol: str, signal: Signal, params: TradeParams):
        """
        Dat lenh Market vao vi the:
        1. Lay tick_size -> round SL/TP
        2. Spread check -> abort neu spread qua rong (thanh khoan kem)
        3. Market order -> fill ngay, khong miss lenh vi gia chay di
        4. Verify SL active sau fill -> re-arm neu thieu
        """
        try:
            self.client.set_leverage(symbol, params.leverage)

            # --- Lay tick_size va bid/ask de check spread ---
            bid, ask = self.client.get_bid_ask(symbol)
            if bid <= 0 or ask <= 0:
                logger.warning(f"{symbol}: ABORT entry — khong lay duoc bid/ask")
                return

            mid_price  = (bid + ask) / 2.0
            spread     = ask - bid
            spread_pct = spread / mid_price if mid_price > 0 else 0

            tick_size = 0.0
            try:
                info      = self.client.get_instrument_info(symbol)
                tick_size = float(info["priceFilter"]["tickSize"])
            except Exception:
                pass

            # Spread check: abort neu spread qua rong (thanh khoan kem, slip lon)
            is_largecap = symbol in {"BTCUSDT", "ETHUSDT"}
            max_spread  = config.MAX_SPREAD_PCT_LARGE if is_largecap else config.MAX_SPREAD_PCT_ALT
            if spread_pct > max_spread:
                logger.warning(
                    f"{symbol}: ABORT — spread={spread_pct*100:.3f}% > {max_spread*100:.3f}%"
                )
                return

            # Round SL/TP theo tick size
            _sl_ceil    = (params.side == "Sell")
            sl_rounded  = self.client.round_to_tick(params.sl_price,  tick_size, ceil=_sl_ceil) if tick_size > 0 else round(params.sl_price,  6)
            tp1_rounded = self.client.round_to_tick(params.tp1_price, tick_size) if tick_size > 0 else round(params.tp1_price, 6)

            # --- Market order: fill ngay (KHONG set SL/TP trong order) ---
            # Ly do tach rieng: Bybit market order voi SL/TP params doi khi tao
            # "order-level conditional orders" thay vi "position-level TP/SL"
            # -> position field stopLoss/takeProfit van rong du lenh co SL/TP
            # Giai phap: dat Market order truoc, sau do set SL+TP bang set_trading_stop
            order = self.client.place_order(
                symbol=symbol,
                side=params.side,
                qty=params.qty,
                order_type="Market",
            )

            order_id = order.get("orderId", "")

            # Update in-memory state
            self._partial_closed[symbol] = False
            self._breakeven_set[symbol]  = False
            self._sl_verified[symbol]    = False
            self._atr[symbol]            = signal.atr
            self._tp1_price[symbol]      = tp1_rounded
            self._tp2_price[symbol]      = params.tp2_price
            self._sl_price[symbol]       = sl_rounded
            self._open_time[symbol]      = time.time()
            self._tick_size[symbol]      = tick_size

            # --- Set SL + TP tren position (position-level) sau khi fill ---
            # Doi 0.5s de Bybit xu ly fill truoc khi set TP/SL
            time.sleep(0.5)
            sl_tp_ok = False
            for _attempt in range(3):
                try:
                    self.client.set_sl_tp(symbol, sl_rounded, tp1_rounded)
                    sl_tp_ok = True
                    logger.info(
                        f"{symbol}: SL={sl_rounded:.6f} TP1={tp1_rounded:.6f} set via set_trading_stop"
                    )
                    break
                except Exception as e:
                    logger.warning(
                        f"{symbol}: set_sl_tp attempt {_attempt+1}/3 failed: "
                        f"{str(e).encode('ascii','replace').decode()}"
                    )
                    time.sleep(0.5 * (2 ** _attempt))

            # --- Verify SL + TP sau set ---
            time.sleep(0.3)
            has_sl, actual_sl, has_tp, actual_tp = self.client.verify_position_tp_sl(symbol)
            if not has_sl or not has_tp:
                logger.warning(
                    f"{symbol}: SL/TP missing (has_sl={has_sl}, has_tp={has_tp}) — retry set_sl_tp"
                )
                try:
                    _sl_fix = sl_rounded if not has_sl else 0.0
                    _tp_fix = tp1_rounded if not has_tp else 0.0
                    self.client.set_sl_tp(symbol, _sl_fix, _tp_fix)
                    logger.info(f"{symbol}: SL/TP re-armed: SL={_sl_fix:.6f} TP={_tp_fix:.6f}")
                except Exception as e2:
                    logger.error(f"{symbol}: CRITICAL — cannot set SL/TP: {str(e2).encode('ascii','replace').decode()}")
            else:
                logger.info(f"{symbol}: Verified SL={actual_sl:.6f} TP={actual_tp:.6f} active on position")
            self._sl_verified[symbol] = True

            self.logger.log_trade({
                "event":     "open",
                "symbol":    symbol,
                "side":      params.side,
                "qty":       params.qty,
                "leverage":  params.leverage,
                "entry":     mid_price,
                "sl":        sl_rounded,
                "tp1":       tp1_rounded,
                "tp2":       params.tp2_price,
                "strategy":  signal.strategy_name,
                "strength":  signal.strength,
                "reason":    signal.reason,
                "notional":  params.notional_usdt,
                "capital":   params.capital_usdt,
                "fee":       params.fee_usdt,
                "order_id":  order_id,
            })

            logger.info(
                f"[OPEN] {symbol} {params.side} | qty={params.qty} | "
                f"lev={params.leverage}x | market~{mid_price:.6f} | "
                f"SL={sl_rounded:.6f} | TP1={tp1_rounded:.6f} | "
                f"strategy={signal.strategy_name} | {signal.reason}"
            )

        except Exception as e:
            logger.error(f"Failed to enter trade {symbol}: {str(e).encode('ascii', 'replace').decode()}")

    def manage_open_positions(self, open_positions: list[dict]):
        """
        Quan ly vi the dang mo theo 3 muc:
        1. BREAKEVEN_TRIGGER (60%): lock SL TAI muc 60% TP1 — dam bao loi toi thieu 60% TP1 du bi stop ra
        2. PARTIAL_CLOSE_TRIGGER (75%): dong 50% reduce-only, cap nhat TP len TP2, xac nhan SL lock
        3. 50% con lai chay den TP2 voi SL da lock 60% TP1 (khong mat von, chi mat 1 phan loi)

        Defensive: kiem tra SL con active khong, re-arm neu mat.
        """
        active_symbols = {pos["symbol"] for pos in open_positions}

        for pos in open_positions:
            symbol     = pos["symbol"]
            entry      = _fval(pos, "avgPrice")
            mark_price = _fval(pos, "markPrice", entry)
            side       = pos["side"]

            # --- State restoration after restart ---
            # Neu bot restart, tat ca in-memory state bi reset. Phuc hoi tu du lieu exchange.
            if symbol not in self._sl_price:
                exchange_sl = _fval(pos, "stopLoss")
                if exchange_sl > 0:
                    self._sl_price[symbol] = exchange_sl
                    logger.info(f"{symbol}: Restored SL={exchange_sl:.6f} from exchange after restart")
                created_ms = int(pos.get("createdTime", 0))
                if created_ms > 0 and symbol not in self._open_time:
                    self._open_time[symbol] = created_ms / 1000
                # Infer breakeven: neu SL da chuyen qua phia loi (LONG: SL > entry; SHORT: SL < entry)
                if exchange_sl > 0:
                    # BE SL cho ca Long va Short: entry + fee_buffer ≈ entry * 1.0011 (0.11% phi)
                    # LONG: initial SL < entry, BE SL > entry -> SL >= entry -> da set BE
                    # SHORT: initial SL = entry + 1.5*ATR >> entry + fee -> phan biet bang tolerance 0.3%
                    #   BE SL: entry <= SL <= entry * 1.003 (chi phi round-trip <= 0.3%)
                    #   Initial SL: entry * 1.01+ (1.5x ATR thuong lon hon 1%)
                    fee_tol = entry * 0.003  # tolerance 0.3% -> phan biet BE vs initial SL
                    if side == "Buy" and exchange_sl >= entry:
                        self._breakeven_set[symbol] = True
                        logger.info(f"{symbol}: Inferred BE set LONG (SL={exchange_sl:.6f} >= entry={entry:.6f})")
                    elif side == "Sell" and entry <= exchange_sl <= entry + fee_tol:
                        self._breakeven_set[symbol] = True
                        logger.info(f"{symbol}: Inferred BE set SHORT (SL={exchange_sl:.6f} in BE range [{entry:.6f}, {entry+fee_tol:.6f}])")
                # Restore _tp1_price: neu chua co (restart), lay tu exchange TP
                # Logic: neu partial chua xay ra, exchange TP chinh la TP1
                # (neu partial da xay ra, exchange TP la TP2 nhung ta khong biet — xem ben duoi)
                exchange_tp = _fval(pos, "takeProfit")
                if exchange_tp > 0 and symbol not in self._tp1_price:
                    # Gia su day la TP1 (neu partial chua xay ra)
                    self._tp1_price[symbol] = exchange_tp
                    logger.info(f"{symbol}: Restored TP1={exchange_tp:.6f} from exchange after restart")
                # Infer partial close: neu TP tren san != _tp1_price ban dau -> partial da xay ra
                # Chi co the detect duoc tren VONG LAP THU 2+ (khi _tp1_price da co gia tri khac exchange TP)
                stored_tp1 = self._tp1_price.get(symbol, 0.0)
                if (exchange_tp > 0 and stored_tp1 > 0
                        and abs(exchange_tp - stored_tp1) > stored_tp1 * 0.001
                        and not self._partial_closed.get(symbol, False)):
                    self._partial_closed[symbol] = True
                    logger.info(
                        f"{symbol}: Inferred partial close already done (exchange TP={exchange_tp:.6f} != stored tp1={stored_tp1:.6f})"
                    )
                # Lay tick_size neu chua co
                if symbol not in self._tick_size:
                    try:
                        info = self.client.get_instrument_info(symbol)
                        self._tick_size[symbol] = float(info["priceFilter"]["tickSize"])
                    except Exception:
                        self._tick_size[symbol] = 0.0

            # --- Periodic SL + TP health check ---
            # Neu SL hoac TP bi huy tren exchange (maintenance, loi API), re-arm ngay
            saved_sl = self._sl_price.get(symbol, 0.0)
            saved_tp = self._tp1_price.get(symbol, 0.0) if not self._partial_closed.get(symbol, False) \
                       else self._tp2_price.get(symbol, 0.0)
            exchange_sl = _fval(pos, "stopLoss")
            exchange_tp = _fval(pos, "takeProfit")
            need_rearm_sl = saved_sl > 0 and exchange_sl <= 0
            need_rearm_tp = saved_tp > 0 and exchange_tp <= 0
            if need_rearm_sl or need_rearm_tp:
                _rearm_sl = saved_sl if need_rearm_sl else 0.0
                _rearm_tp = saved_tp if need_rearm_tp else 0.0
                logger.warning(
                    f"{symbol}: SL/TP missing on exchange "
                    f"(SL={'missing' if need_rearm_sl else 'ok'}, "
                    f"TP={'missing' if need_rearm_tp else 'ok'}) — re-arming"
                )
                try:
                    self.client.set_sl_tp(symbol, _rearm_sl, _rearm_tp)
                except Exception as e:
                    logger.error(f"{symbol}: Failed to re-arm SL/TP: {e}")

            tp1_threshold = _fval(pos, "takeProfit")

            if tp1_threshold <= 0:
                if self.risk_mgr.should_close_position(pos, mark_price):
                    logger.warning(f"{symbol}: Emergency close — excessive loss")
                    self._close_position(pos)
                continue

            # Dung recent high/low tu 3 nen 1m de khong bo lo wick ngan giua 2 poll cycle
            best_price = mark_price
            try:
                df1m = self.client.get_klines(symbol, "1", 4)
                if not df1m.empty and len(df1m) >= 3:
                    if side == "Buy":
                        best_price = df1m["high"].iloc[-3:].max()
                    else:
                        best_price = df1m["low"].iloc[-3:].min()
            except Exception:
                pass

            dist_to_tp1 = abs(tp1_threshold - entry)
            if side == "Buy":
                dist_moved = best_price - entry
            else:
                dist_moved = entry - best_price

            # --- Muc 1: Trail SL khi gia di BREAKEVEN_TRIGGER% (60%) duong den TP1 ---
            # SL dich len nhung LUON duoi entry:
            #   LONG : new_SL = entry - 60% * sl_goc_dist  (van duoi entry, chat hon)
            #   SHORT: new_SL = entry + 60% * sl_goc_dist  (van tren entry, chat hon)
            # Dam bao: neu gia quay dau sau khi trail, van mat toi da 60% buffer goc (khong mat hon)
            if not self._breakeven_set.get(symbol, False) and dist_to_tp1 > 0 and dist_moved > 0:
                if dist_moved >= dist_to_tp1 * config.BREAKEVEN_TRIGGER:
                    try:
                        # Tinh sl_goc_dist tu sl_price da luu
                        saved_sl = self._sl_price.get(symbol, 0.0)
                        if saved_sl <= 0:
                            raise ValueError("no saved SL to trail from")
                        sl_goc_dist = abs(entry - saved_sl)   # khoang cach SL goc tu entry
                        # SL moi = entry - 60% * sl_goc_dist (LONG) / entry + 60% * sl_goc_dist (SHORT)
                        # "it nhat 60%" = giu lai it nhat 60% buffer goc duoi entry
                        lock_ratio = config.BREAKEVEN_TRIGGER   # 0.60
                        if side == "Buy":
                            be_price_raw = entry - lock_ratio * sl_goc_dist
                        else:
                            be_price_raw = entry + lock_ratio * sl_goc_dist
                        ts = self._tick_size.get(symbol, 0.0)
                        _be_ceil = (side == "Sell")
                        be_price = self.client.round_to_tick(be_price_raw, ts, ceil=_be_ceil) if ts > 0 else round(be_price_raw, 6)
                        # Validate: SL moi phai chat hon SL cu (dich ve phia entry)
                        if side == "Buy" and be_price <= saved_sl:
                            raise ValueError(f"trail SL {be_price:.6f} not tighter than original {saved_sl:.6f}")
                        if side == "Sell" and be_price >= saved_sl:
                            raise ValueError(f"trail SL {be_price:.6f} not tighter than original {saved_sl:.6f}")
                        # Validate: SL phai o phia duoi/tren entry (van duoi entry voi LONG)
                        mark = _fval(pos, "markPrice")
                        if mark > 0:
                            if side == "Buy" and be_price >= mark:
                                raise ValueError(f"trail SL {be_price:.6f} >= mark {mark:.6f} LONG — wait")
                            if side == "Sell" and be_price <= mark:
                                raise ValueError(f"trail SL {be_price:.6f} <= mark {mark:.6f} SHORT — wait")
                        self.client.update_stop_loss(symbol, be_price)
                        self._breakeven_set[symbol] = True
                        self._sl_price[symbol] = be_price
                        logger.info(
                            f"{symbol}: SL trailed -> {be_price:.6f} "
                            f"(entry={entry:.6f}, SL goc={saved_sl:.6f}, "
                            f"giu {lock_ratio*100:.0f}% buffer goc duoi entry | "
                            f"moved {dist_moved/dist_to_tp1*100:.0f}% toward TP1)"
                        )
                    except Exception as e:
                        logger.warning(f"{symbol}: Could not trail SL: {e}")

            # --- Muc 2: Partial close tai PARTIAL_CLOSE_TRIGGER% (75%) duong den TP1 ---
            # FIX: _partial_closed chi duoc set True SAU KHI close order thanh cong
            # (truoc day set True truoc -> neu close fail, bot nghi da close nhung khong phai)
            if not self._partial_closed.get(symbol, False) and dist_to_tp1 > 0 and dist_moved > 0:
                if dist_moved >= dist_to_tp1 * config.PARTIAL_CLOSE_TRIGGER:
                    try:
                        # Cap nhat TP tren san tu TP1 sang TP2 (tick-aligned)
                        tp2_raw = self._tp2_price.get(symbol, 0.0)
                        if tp2_raw > 0:
                            ts2 = self._tick_size.get(symbol, 0.0)
                            tp2 = self.client.round_to_tick(tp2_raw, ts2) if ts2 > 0 else round(tp2_raw, 6)
                            self.client.update_take_profit(symbol, tp2)
                            self._tp2_price[symbol] = tp2  # update to tick-aligned value
                            logger.info(f"{symbol}: TP updated TP1={tp1_threshold:.4f} -> TP2={tp2:.6f}")

                        # Dong 50% vi the — align voi qty_step cua instrument
                        pos_qty = _fval(pos, "size")
                        try:
                            info     = self.client.get_instrument_info(symbol)
                            qty_step = float(info["lotSizeFilter"]["qtyStep"])
                        except Exception:
                            qty_step = 0.001
                        partial_qty = math.floor(pos_qty * 0.5 / qty_step) * qty_step
                        partial_qty = round(partial_qty, 8)

                        if partial_qty > 0:
                            close_side = "Sell" if side == "Buy" else "Buy"
                            self.client.place_order(symbol, close_side, partial_qty, reduce_only=True)
                            # FIX: chi set True sau khi place_order thanh cong (khong raise exception)
                            self._partial_closed[symbol] = True
                            logger.info(
                                f"{symbol}: Partial close 50% ({partial_qty}) at "
                                f"{dist_moved/dist_to_tp1*100:.0f}% of TP1 | "
                                f"remaining 50% targets TP2={tp2:.4f}"
                            )
                        else:
                            # qty qua nho de chia doi — danh dau da partial (toan bo chay den TP2)
                            self._partial_closed[symbol] = True

                        # Trail SL khi partial close (neu chua trail o Muc 1)
                        if not self._breakeven_set.get(symbol, False):
                            saved_sl2 = self._sl_price.get(symbol, 0.0)
                            if saved_sl2 > 0:
                                sl_goc_dist2 = abs(entry - saved_sl2)
                                lock_ratio2  = config.BREAKEVEN_TRIGGER
                                if side == "Buy":
                                    be_price_raw2 = entry - lock_ratio2 * sl_goc_dist2
                                else:
                                    be_price_raw2 = entry + lock_ratio2 * sl_goc_dist2
                                ts3 = self._tick_size.get(symbol, 0.0)
                                _be_ceil3 = (side == "Sell")
                                be_price = self.client.round_to_tick(be_price_raw2, ts3, ceil=_be_ceil3) if ts3 > 0 else round(be_price_raw2, 6)
                                mark3 = _fval(pos, "markPrice")
                                ok = True
                                if mark3 > 0:
                                    if (side == "Buy" and be_price >= mark3) or (side == "Sell" and be_price <= mark3):
                                        ok = False
                                if ok:
                                    self.client.update_stop_loss(symbol, be_price)
                                    self._breakeven_set[symbol] = True
                                    self._sl_price[symbol] = be_price
                                    logger.info(f"{symbol}: SL trailed (partial) -> {be_price:.6f} ({lock_ratio2*100:.0f}% buffer duoi entry)")

                    except Exception as e:
                        logger.warning(f"{symbol}: Could not execute partial close: {e}")

            # --- Emergency close ---
            if self.risk_mgr.should_close_position(pos, mark_price):
                logger.warning(f"{symbol}: Emergency close — excessive loss")
                self._close_position(pos)

        # Xoa state cua cac symbol khong con trong active_symbols (dong giua cycle)
        # Tranh state stale khi position dong dot xuat (partial fill, exchange error, v.v.)
        tracked = set(self._sl_price.keys())
        for stale_sym in tracked - active_symbols:
            logger.debug(f"{stale_sym}: position no longer active — clearing stale executor state")
            self.clear_position_state(stale_sym)

    def clear_position_state(self, symbol: str):
        """Xoa toan bo in-memory state cua symbol (goi khi position dong — bot hoac exchange)."""
        self._partial_closed.pop(symbol, None)
        self._breakeven_set.pop(symbol, None)
        self._atr.pop(symbol, None)
        self._tp1_price.pop(symbol, None)
        self._tp2_price.pop(symbol, None)
        self._sl_price.pop(symbol, None)
        self._sl_verified.pop(symbol, None)
        self._open_time.pop(symbol, None)
        self._tick_size.pop(symbol, None)

    def _close_position(self, position: dict):
        symbol = position["symbol"]
        side   = position["side"]
        qty    = _fval(position, "size")
        pnl    = _fval(position, "unrealisedPnl")
        try:
            self.client.close_position(symbol, side, qty)
            self.clear_position_state(symbol)
            self.logger.log_trade({
                "event":  "close",
                "symbol": symbol,
                "side":   side,
                "qty":    qty,
                "reason": "signal_reversal_or_emergency",
            })
            logger.info(f"[CLOSE] {symbol} {side} qty={qty} pnl={pnl:.4f}")
        except Exception as e:
            logger.error(f"Failed to close position {symbol}: {str(e).encode('ascii', 'replace').decode()}")
