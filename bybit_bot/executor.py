"""
Trade Executor — thực thi lệnh và quản lý vị thế
- Dat lenh Market voi SL+TP ngay khi vao
- 1 TP duy nhat: hit la dong toan bo position (Bybit tu dong dong)
- Health check: re-arm SL/TP neu mat, fix SL sai ty le
- Emergency close: dong tay neu loss > 80% margin (SL exchange bi miss)
"""

import logging
import time

from client import BybitClient
from risk_manager import RiskManager, TradeParams
from strategies.base import Signal
from bot_logger import BotLogger
import config

logger = logging.getLogger(__name__)


def _fval(d: dict, key: str, default: float = 0.0) -> float:
    v = d.get(key, default)
    try:
        return float(v) if v != "" else default
    except (TypeError, ValueError):
        return default


class Executor:
    def __init__(self, client: BybitClient, risk_mgr: RiskManager, bot_logger: BotLogger):
        self.client   = client
        self.risk_mgr = risk_mgr
        self.logger   = bot_logger

        self._sl_price:   dict[str, float] = {}   # SL ban dau de re-arm / verify ratio
        self._tp_price:   dict[str, float] = {}   # TP de re-arm
        self._open_time:  dict[str, float] = {}
        self._tick_size:  dict[str, float] = {}
        self._executing:  set               = set()  # symbols dang trong qua trinh execute (lock)

    def execute_signal(
        self,
        symbol: str,
        signal: Signal,
        equity: float,
        open_positions: list[dict],
        is_priority: bool = False,
    ):
        if signal.direction == 0:
            return

        # Execution lock: tuyet doi khong cho 2 luong chay cung luc cho 1 symbol
        if symbol in self._executing:
            logger.warning(f"{symbol}: execute_signal already in progress — skip duplicate call")
            return
        self._executing.add(symbol)
        try:
            self._execute_signal_inner(symbol, signal, equity, open_positions, is_priority)
        finally:
            self._executing.discard(symbol)

    def _execute_signal_inner(
        self,
        symbol: str,
        signal: Signal,
        equity: float,
        open_positions: list[dict],
        is_priority: bool = False,
    ):
        existing = [p for p in open_positions if p["symbol"] == symbol]
        if existing:
            pos_side    = existing[0]["side"]
            signal_side = "Buy" if signal.direction == 1 else "Sell"
            if pos_side == signal_side:
                return

            # Anti-whipsaw: khong force-close neu position mo < 30 phut
            held_seconds = 0
            exchange_created_ms = int(existing[0].get("createdTime", 0))
            if exchange_created_ms > 0:
                held_seconds = time.time() - exchange_created_ms / 1000
            else:
                held_seconds = time.time() - self._open_time.get(symbol, time.time())

            pos_pnl_pct = 0.0
            try:
                notional    = _fval(existing[0], "positionValue", 1) or 1
                lev         = _fval(existing[0], "leverage", 1) or 1
                margin      = notional / lev
                pnl         = _fval(existing[0], "unrealisedPnl")
                pos_pnl_pct = pnl / margin if margin > 0 else 0
            except Exception:
                pass

            if held_seconds < 1800 and pos_pnl_pct > -0.20:
                logger.info(
                    f"{symbol}: Signal reversal — SKIP (held={held_seconds:.0f}s < 1800s, "
                    f"PnL={pos_pnl_pct*100:.1f}%)"
                )
                return

            logger.info(f"{symbol}: Signal reversal -> close {pos_side} -> enter {signal_side}")
            self._close_position(existing[0])
            time.sleep(0.5)

        # Cap nhat entry_price bang gia live
        bid_live, ask_live = self.client.get_bid_ask(symbol)
        if bid_live <= 0 or ask_live <= 0:
            logger.warning(f"{symbol}: SKIP — khong lay duoc bid/ask live")
            return
        signal.entry_price = ask_live if signal.direction == 1 else bid_live

        params = self.risk_mgr.compute_trade(signal, equity, open_positions, is_priority=is_priority)
        if not params:
            return

        params.symbol = symbol
        self._enter_trade(symbol, signal, params)

    def _enter_trade(self, symbol: str, signal: Signal, params: TradeParams):
        try:
            self.client.set_leverage(symbol, params.leverage)

            bid, ask = self.client.get_bid_ask(symbol)
            if bid <= 0 or ask <= 0:
                logger.warning(f"{symbol}: ABORT entry — khong lay duoc bid/ask")
                return

            mid_price  = (bid + ask) / 2.0
            spread_pct = (ask - bid) / mid_price if mid_price > 0 else 0
            is_largecap = symbol in {"BTCUSDT", "ETHUSDT"}
            max_spread  = config.MAX_SPREAD_PCT_LARGE if is_largecap else config.MAX_SPREAD_PCT_ALT
            if spread_pct > max_spread:
                logger.warning(f"{symbol}: ABORT — spread={spread_pct*100:.3f}% > {max_spread*100:.3f}%")
                return

            tick_size = 0.0
            try:
                info      = self.client.get_instrument_info(symbol)
                tick_size = float(info["priceFilter"]["tickSize"])
            except Exception:
                pass

            _sl_ceil   = (params.side == "Sell")
            sl_rounded = self.client.round_to_tick(params.sl_price,  tick_size, ceil=_sl_ceil) if tick_size > 0 else round(params.sl_price,  6)
            tp_rounded = self.client.round_to_tick(params.tp1_price, tick_size) if tick_size > 0 else round(params.tp1_price, 6)

            # Layer 1: Market order voi SL+TP
            order = self.client.place_order(
                symbol=symbol,
                side=params.side,
                qty=params.qty,
                order_type="Market",
                sl=sl_rounded,
                tp=tp_rounded,
                tick_size=tick_size,
            )

            self._sl_price[symbol]  = sl_rounded
            self._tp_price[symbol]  = tp_rounded
            self._open_time[symbol] = time.time()
            self._tick_size[symbol] = tick_size

            # Layer 2: set_trading_stop backup sau khi fill
            time.sleep(1.0)
            try:
                self.client.set_sl_tp(symbol, sl_rounded, tp_rounded)
            except Exception as e:
                logger.error(f"{symbol}: set_sl_tp layer2 FAILED: {str(e).encode('ascii','replace').decode()}")

            # Layer 3: Verify va re-arm (up to 5 attempts)
            for _attempt in range(5):
                time.sleep(1.0)
                has_sl, actual_sl, has_tp, actual_tp = self.client.verify_position_tp_sl(symbol)
                if has_sl and has_tp:
                    # Kiem tra SL co dung ty le 5:1 voi TP
                    _tp_dist = abs(actual_tp - signal.entry_price) if actual_tp > 0 else 0
                    _sl_dist = abs(actual_sl - signal.entry_price) if actual_sl > 0 else 0
                    _ratio_ok = True
                    if _tp_dist > 0 and _sl_dist > 0:
                        _ratio = _sl_dist / _tp_dist
                        if abs(_ratio - config.SL_TP_RATIO) / config.SL_TP_RATIO > 0.15:
                            _ratio_ok = False
                            logger.warning(
                                f"{symbol}: SL ratio wrong ({_ratio:.2f}×TP expected {config.SL_TP_RATIO}×) "
                                f"— re-arm sl={sl_rounded} tp={tp_rounded}"
                            )
                            print(f"[FIX] {symbol} SL ratio {_ratio:.2f}× → forcing {config.SL_TP_RATIO}×", flush=True)
                    if _ratio_ok:
                        logger.info(f"{symbol}: CONFIRMED SL={actual_sl} TP={actual_tp} (attempt={_attempt+1})")
                        print(f"[OK] {symbol} SL={actual_sl} TP={actual_tp} confirmed", flush=True)
                        break
                    try:
                        self.client.set_sl_tp(symbol, sl_rounded, tp_rounded)
                    except Exception as e2:
                        logger.error(f"{symbol}: fix-ratio re-arm FAILED: {str(e2).encode('ascii','replace').decode()}")
                    continue
                logger.error(f"{symbol}: SL/TP MISSING attempt {_attempt+1}/5 — re-arm")
                print(f"[CRITICAL] {symbol} SL/TP MISSING attempt {_attempt+1}/5", flush=True)
                try:
                    self.client.set_sl_tp(symbol, sl_rounded, tp_rounded)
                except Exception as e2:
                    logger.error(f"{symbol}: re-arm FAILED: {str(e2).encode('ascii','replace').decode()}")

            self.logger.log_trade({
                "event":    "open",
                "symbol":   symbol,
                "side":     params.side,
                "qty":      params.qty,
                "leverage": params.leverage,
                "entry":    mid_price,
                "sl":       sl_rounded,
                "tp":       tp_rounded,
                "notional": params.notional_usdt,
                "fee":      params.fee_usdt,
            })
            logger.info(
                f"[ENTRY] {symbol} {params.side} qty={params.qty} lev={params.leverage}x "
                f"entry={mid_price:.6f} SL={sl_rounded:.6f} TP={tp_rounded:.6f}"
            )

        except Exception as e:
            logger.error(f"{symbol}: _enter_trade error: {str(e).encode('ascii','replace').decode()}")

    def manage_positions(self, open_positions: list[dict]):
        """Health check: re-arm SL/TP neu mat, fix SL sai ty le, emergency close."""
        active_symbols = {p["symbol"] for p in open_positions}

        for pos in open_positions:
            symbol     = pos["symbol"]
            entry      = _fval(pos, "avgPrice")
            side       = pos["side"]
            mark_price = _fval(pos, "markPrice", entry)

            # Restore state sau restart
            if symbol not in self._sl_price:
                exchange_sl = _fval(pos, "stopLoss")
                exchange_tp = _fval(pos, "takeProfit")
                if exchange_sl > 0:
                    self._sl_price[symbol] = exchange_sl
                    logger.info(f"{symbol}: Restored SL={exchange_sl:.6f} from exchange")
                if exchange_tp > 0:
                    self._tp_price[symbol] = exchange_tp
                    logger.info(f"{symbol}: Restored TP={exchange_tp:.6f} from exchange")
                created_ms = int(pos.get("createdTime", 0))
                if created_ms > 0:
                    self._open_time[symbol] = created_ms / 1000
                if symbol not in self._tick_size:
                    try:
                        info = self.client.get_instrument_info(symbol)
                        self._tick_size[symbol] = float(info["priceFilter"]["tickSize"])
                    except Exception:
                        self._tick_size[symbol] = 0.0

            saved_sl    = self._sl_price.get(symbol, 0.0)
            saved_tp    = self._tp_price.get(symbol, 0.0)
            exchange_sl = _fval(pos, "stopLoss")
            exchange_tp = _fval(pos, "takeProfit")
            _pos_lev    = max(10.0, _fval(pos, "leverage", 10.0))

            # Fallback SL/TP neu ca saved lan exchange deu khong co (restart + SL mat)
            if saved_sl <= 0 and exchange_sl <= 0 and entry > 0:
                sl_roi   = config.TP_ROI_MIN * config.SL_TP_RATIO
                sl_dist  = sl_roi * entry / _pos_lev
                saved_sl = (entry + sl_dist) if side == "Sell" else (entry - sl_dist)
                self._sl_price[symbol] = saved_sl
                logger.warning(f"{symbol}: Fallback SL={saved_sl:.6f} (ROI={sl_roi*100:.0f}%/{_pos_lev:.0f}x)")

            if saved_tp <= 0 and exchange_tp <= 0 and entry > 0:
                tp_roi   = config.TP_ROI_MIN
                tp_dist  = tp_roi * entry / _pos_lev
                saved_tp = (entry - tp_dist) if side == "Sell" else (entry + tp_dist)
                self._tp_price[symbol] = saved_tp
                logger.warning(f"{symbol}: Fallback TP={saved_tp:.6f} (ROI={tp_roi*100:.0f}%/{_pos_lev:.0f}x)")

            # Fix SL sai ty le 5:1 (co the bi Bybit clamp hoac tu code cu)
            if exchange_sl > 0 and exchange_tp > 0 and entry > 0:
                _tp_dist_hc = abs(exchange_tp - entry)
                _sl_dist_hc = abs(exchange_sl - entry)
                if _tp_dist_hc > 0:
                    _ratio_hc = _sl_dist_hc / _tp_dist_hc
                    if abs(_ratio_hc - config.SL_TP_RATIO) / config.SL_TP_RATIO > 0.15:
                        _correct_sl_dist = _tp_dist_hc * config.SL_TP_RATIO
                        _correct_sl = (entry + _correct_sl_dist) if side == "Sell" else (entry - _correct_sl_dist)
                        logger.warning(
                            f"{symbol}: SL ratio wrong ({_ratio_hc:.2f}×TP expected {config.SL_TP_RATIO}×) "
                            f"SL={exchange_sl:.6f}→{_correct_sl:.6f}"
                        )
                        print(f"[FIX] {symbol} health: SL {_ratio_hc:.2f}×TP → {config.SL_TP_RATIO}×TP (SL={_correct_sl:.6f})", flush=True)
                        try:
                            self.client.set_sl_tp(symbol, _correct_sl, exchange_tp)
                            self._sl_price[symbol] = _correct_sl
                            saved_sl = _correct_sl
                        except Exception as e:
                            logger.error(f"{symbol}: fix-ratio FAILED: {str(e).encode('ascii','replace').decode()}")

            # Re-arm neu SL hoac TP bi mat tren exchange
            need_rearm_sl = saved_sl > 0 and exchange_sl <= 0
            need_rearm_tp = saved_tp > 0 and exchange_tp <= 0
            if need_rearm_sl or need_rearm_tp:
                rearm_sl = saved_sl if need_rearm_sl else exchange_sl
                rearm_tp = saved_tp if need_rearm_tp else exchange_tp
                logger.warning(
                    f"{symbol}: SL/TP missing (SL={'miss' if need_rearm_sl else 'ok'}, "
                    f"TP={'miss' if need_rearm_tp else 'ok'}) — re-arming"
                )
                try:
                    self.client.set_sl_tp(symbol, rearm_sl, rearm_tp)
                    logger.info(f"{symbol}: Re-armed SL={rearm_sl:.6f} TP={rearm_tp:.6f}")
                except Exception as e:
                    logger.error(f"{symbol}: re-arm FAILED: {str(e).encode('ascii','replace').decode()}")

            # Emergency close: chi khi loss > 80% margin va SL exchange bi miss
            if self.risk_mgr.should_close_position(pos, mark_price):
                logger.warning(f"{symbol}: Emergency close — excessive loss")
                self._close_position(pos)

        # Xoa state stale
        for stale_sym in set(self._sl_price.keys()) - active_symbols:
            self.clear_position_state(stale_sym)

    def clear_position_state(self, symbol: str):
        self._sl_price.pop(symbol, None)
        self._tp_price.pop(symbol, None)
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
            logger.error(f"Failed to close {symbol}: {str(e).encode('ascii', 'replace').decode()}")
