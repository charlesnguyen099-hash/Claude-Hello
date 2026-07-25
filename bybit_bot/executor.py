"""
Trade Executor - thuc thi lenh va quan ly vi the
- Dat lenh Market voi SL+TP ngay khi vao
- 1 TP duy nhat: hit la dong toan bo position (Bybit tu dong dong)
- Health check: re-arm SL/TP neu mat, fix SL sai ty le
- Emergency close: dong tay neu loss > 80% margin (SL exchange bi miss)
"""

import logging
import time

from client import BybitClient
from risk_manager import RiskManager, TradeParams, max_safe_sl_roi
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
        self._open_symbols: set             = set()  # symbols co open position theo executor (guard stale list)

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
            logger.warning(f"{symbol}: execute_signal already in progress - skip duplicate call")
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
        # Guard stale open_positions list: neu executor biet co position, khong mo them
        if symbol in self._open_symbols:
            existing_check = [p for p in open_positions if p["symbol"] == symbol]
            if not existing_check:
                # open_positions stale - executor cache noi co position, tin cache
                logger.warning(f"{symbol}: SKIP - executor cache shows open position (open_positions may be stale)")
                return

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

            # B8 fix: PnL condition (> -0.20) was unreachable because min SL ROI = 60%,
            # so position always closes via SL before PnL hits -20%. Time-only check.
            if held_seconds < 300:
                logger.info(
                    f"{symbol}: Signal reversal - SKIP (held={held_seconds:.0f}s < 300s)"
                )
                return

            logger.info(f"{symbol}: Signal reversal -> close {pos_side} -> enter {signal_side}")
            self._close_position(existing[0])
            time.sleep(0.5)

        # Cap nhat entry_price bang gia live
        bid_live, ask_live = self.client.get_bid_ask(symbol)
        if bid_live <= 0 or ask_live <= 0:
            logger.warning(f"{symbol}: SKIP - khong lay duoc bid/ask live")
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
                logger.warning(f"{symbol}: ABORT entry - khong lay duoc bid/ask")
                return

            mid_price  = (bid + ask) / 2.0
            spread_pct = (ask - bid) / mid_price if mid_price > 0 else 0
            is_largecap = symbol in {"BTCUSDT", "ETHUSDT"}
            max_spread  = config.MAX_SPREAD_PCT_LARGE if is_largecap else config.MAX_SPREAD_PCT_ALT
            if spread_pct > max_spread:
                logger.warning(f"{symbol}: ABORT - spread={spread_pct*100:.3f}% > {max_spread*100:.3f}%")
                return

            # KIEM TRA LOI RONG TAI TP TRUOC KHI VAO LENH (pre-entry profitability guard):
            # gross ROI tai TP = (tp_dist / entry) * leverage = tp1_pct/100 * leverage
            # phi ROI = TOTAL_ROUND_TRIP_COST * leverage  (entry+exit+funding, amplified by lev)
            # spread ROI = spread_pct * leverage           (spread an nhu phi, amplified by lev)
            # net ROI tai TP = gross - phi - spread -> phai >= MIN_SAFE_NET_ROI de dam bao co loi that su
            _lev_f         = float(params.leverage)
            _gross_roi_tp  = (params.tp1_pct / 100.0) * _lev_f
            _fee_roi       = config.TOTAL_ROUND_TRIP_COST * _lev_f
            _spread_roi    = spread_pct * _lev_f
            _net_roi_tp    = _gross_roi_tp - _fee_roi - _spread_roi
            _min_net       = getattr(config, "MIN_SAFE_NET_ROI", 0.05)
            if _net_roi_tp < _min_net:
                logger.warning(
                    f"{symbol}: ABORT - pre-entry profitability FAIL: "
                    f"gross_roi={_gross_roi_tp*100:.1f}% - fee={_fee_roi*100:.1f}% - spread={_spread_roi*100:.2f}% "
                    f"= net={_net_roi_tp*100:.1f}% < {_min_net*100:.0f}% (khong du co loi an toan tai TP)"
                )
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
            _order_placed = False
            try:
                order = self.client.place_order(
                    symbol=symbol,
                    side=params.side,
                    qty=params.qty,
                    order_type="Market",
                    sl=sl_rounded,
                    tp=tp_rounded,
                    tick_size=tick_size,
                )
                _order_placed = True
            except Exception as _place_err:
                # place_order co the raise sau khi Bybit da chap nhan lenh (response parse fail / timeout)
                # Kiem tra exchange xem co position thuc su mo khong
                logger.warning(f"{symbol}: place_order raised {_place_err!r} - checking exchange for position")
                time.sleep(1.5)
                try:
                    _positions = self.client.get_positions()
                    _found = [p for p in _positions if p["symbol"] == symbol and _fval(p, "size") > 0]
                    if _found:
                        logger.warning(f"{symbol}: Position confirmed on exchange despite place_order error - proceeding to set SL/TP")
                        _order_placed = True
                    else:
                        logger.error(f"{symbol}: place_order failed and no position found - abort")
                        return
                except Exception as _check_err:
                    logger.error(f"{symbol}: place_order failed + position check failed ({_check_err!r}) - abort")
                    return

            self._sl_price[symbol]  = sl_rounded
            self._tp_price[symbol]  = tp_rounded
            self._open_time[symbol] = time.time()
            self._tick_size[symbol] = tick_size
            self._open_symbols.add(symbol)

            # Layer 2: set_trading_stop backup sau khi fill
            time.sleep(1.0)
            try:
                self.client.set_sl_tp(symbol, sl_rounded, tp_rounded)
            except Exception as e:
                logger.error(f"{symbol}: set_sl_tp layer2 FAILED: {str(e).encode('ascii','replace').decode()}")

            # Layer 3: Verify va re-arm (up to 5 attempts)
            # QUAN TRONG: ty le SL/TP KHONG con co dinh 5:1 - risk_manager nen ty le
            # ve 1:1-3:1 khi liq clamp (TP lon/leverage cao). Verify bang cach so
            # gia thuc te tren exchange voi gia DA DAT (sl_rounded/tp_rounded),
            # KHONG duoc ep ratio 5:1 - ep ratio cu lam re-arm vo han + SL vuot liq.
            _sl_tp_confirmed = False
            for _attempt in range(5):
                time.sleep(1.0)
                has_sl, actual_sl, has_tp, actual_tp = self.client.verify_position_tp_sl(symbol)
                if has_sl and has_tp:
                    _sl_match = abs(actual_sl - sl_rounded) / sl_rounded <= 0.005 if sl_rounded > 0 else False
                    _tp_match = abs(actual_tp - tp_rounded) / tp_rounded <= 0.005 if tp_rounded > 0 else False
                    if _sl_match and _tp_match:
                        logger.info(f"{symbol}: CONFIRMED SL={actual_sl} TP={actual_tp} (attempt={_attempt+1})")
                        print(f"[OK] {symbol} SL={actual_sl} TP={actual_tp} confirmed", flush=True)
                        _sl_tp_confirmed = True
                        break
                    logger.warning(
                        f"{symbol}: SL/TP khac gia da dat (SL {actual_sl} vs {sl_rounded}, "
                        f"TP {actual_tp} vs {tp_rounded}) - re-arm"
                    )
                    try:
                        self.client.set_sl_tp(symbol, sl_rounded, tp_rounded, tick_size=tick_size)
                    except Exception as e2:
                        logger.error(f"{symbol}: re-arm FAILED: {str(e2).encode('ascii','replace').decode()}")
                    continue
                logger.error(f"{symbol}: SL/TP MISSING attempt {_attempt+1}/5 - re-arm")
                print(f"[CRITICAL] {symbol} SL/TP MISSING attempt {_attempt+1}/5", flush=True)
                try:
                    self.client.set_sl_tp(symbol, sl_rounded, tp_rounded, tick_size=tick_size)
                except Exception as e2:
                    logger.error(f"{symbol}: re-arm FAILED: {str(e2).encode('ascii','replace').decode()}")

            # Last resort: neu tat ca 5 attempt deu that bai, thu lai voi gia hien tai
            # Truong hop xay ra khi gia di chuyen qua SL/TP goc trong luc dat lenh
            if not _sl_tp_confirmed:
                logger.error(f"{symbol}: SL/TP not confirmed after 5 attempts - trying fresh prices or closing")
                print(f"[CRITICAL] {symbol}: SL/TP UNSET after 5 tries - emergency recovery", flush=True)
                _recovered = self._recover_sl_tp(
                    symbol=symbol,
                    side=params.side,
                    entry=signal.entry_price,
                    leverage=params.leverage,
                    orig_sl=sl_rounded,
                    orig_tp=tp_rounded,
                    tick_size=tick_size,
                )

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

    def _recover_sl_tp(
        self,
        symbol: str,
        side: str,
        entry: float,
        leverage: int,
        orig_sl: float,
        orig_tp: float,
        tick_size: float,
    ) -> bool:
        """Last-resort SL/TP recovery khi gia di chuyen sau khi dat lenh.
        Thu dat gia goc, neu khong duoc thi tinh lai tu gia hien tai.
        Neu khong the dat ca hai -> dong lenh ngay (khong co SL = rui ro khong kiem soat duoc)."""
        try:
            bid, ask = self.client.get_bid_ask(symbol)
            mark = (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0
        except Exception:
            mark = 0.0

        _dir = 1 if side == "Buy" else -1

        # Kiem tra xem gia co cross qua SL/TP goc chua
        # Long: SL phai < mark, TP phai > mark
        # Short: SL phai > mark, TP phai < mark
        orig_sl_valid = mark <= 0 or (_dir == 1 and orig_sl < mark) or (_dir == -1 and orig_sl > mark)
        orig_tp_valid = mark <= 0 or (_dir == 1 and orig_tp > mark) or (_dir == -1 and orig_tp < mark)

        if orig_sl_valid and orig_tp_valid:
            # Gia hop le - thu dat lai lan cuoi
            try:
                self.client.set_sl_tp(symbol, orig_sl, orig_tp, tick_size=tick_size)
                time.sleep(0.5)
                has_sl, _, has_tp, _ = self.client.verify_position_tp_sl(symbol)
                if has_sl and has_tp:
                    logger.info(f"{symbol}: Recovery OK with original SL={orig_sl} TP={orig_tp}")
                    print(f"[RECOVERED] {symbol} SL/TP set OK on last try", flush=True)
                    return True
            except Exception as e:
                logger.error(f"{symbol}: Recovery attempt FAILED: {str(e).encode('ascii','replace').decode()}")

        # Gia goc khong hop le hoac thu lai van that bai -> tinh lai tu gia hien tai
        if mark > 0 and entry > 0:
            lev = max(leverage, 1)
            tp_roi = config.TP_ROI_MIN
            # CLAMP theo vung an toan thanh ly: 5:1 tho tai leverage cao cho SL vuot
            # gia liq -> Bybit reject -> khong the dat SL -> position khong duoc bao ve
            sl_roi = min(tp_roi * config.SL_TP_RATIO, max_safe_sl_roi(lev))
            tp_dist = tp_roi * entry / lev
            sl_dist = sl_roi * entry / lev
            fresh_sl = entry - _dir * sl_dist
            fresh_tp = entry + _dir * tp_dist

            fresh_sl_valid = (_dir == 1 and fresh_sl < mark) or (_dir == -1 and fresh_sl > mark)
            fresh_tp_valid = (_dir == 1 and fresh_tp > mark) or (_dir == -1 and fresh_tp < mark)

            if fresh_sl_valid and fresh_tp_valid:
                _sl_c = tick_size > 0 and side == "Sell"
                fresh_sl_r = self.client.round_to_tick(fresh_sl, tick_size, ceil=_sl_c) if tick_size > 0 else round(fresh_sl, 6)
                fresh_tp_r = self.client.round_to_tick(fresh_tp, tick_size) if tick_size > 0 else round(fresh_tp, 6)
                try:
                    self.client.set_sl_tp(symbol, fresh_sl_r, fresh_tp_r, tick_size=tick_size)
                    time.sleep(0.5)
                    has_sl, _, has_tp, _ = self.client.verify_position_tp_sl(symbol)
                    if has_sl and has_tp:
                        self._sl_price[symbol] = fresh_sl_r
                        self._tp_price[symbol] = fresh_tp_r
                        logger.info(f"{symbol}: Recovery OK with fresh SL={fresh_sl_r} TP={fresh_tp_r} (mark={mark:.6f})")
                        print(f"[RECOVERED] {symbol} fresh SL={fresh_sl_r} TP={fresh_tp_r}", flush=True)
                        return True
                except Exception as e:
                    logger.error(f"{symbol}: Fresh SL/TP FAILED: {str(e).encode('ascii','replace').decode()}")

        # Khong the dat SL/TP -> dong lenh ngay de tranh rui ro khong kiem soat
        logger.error(f"{symbol}: Cannot set SL/TP -> EMERGENCY CLOSE to protect account")
        print(f"[EMERGENCY] {symbol}: SL/TP unset - closing position for safety", flush=True)
        try:
            positions = self.client.get_positions()
            for pos in positions:
                if pos["symbol"] == symbol:
                    self._close_position(pos)
                    return False
        except Exception as e:
            logger.error(f"{symbol}: Emergency close FAILED: {str(e).encode('ascii','replace').decode()}")
        return False

    def manage_positions(self, open_positions: list[dict]):
        """Health check: re-arm SL/TP neu mat, fix SL sai ty le, emergency close."""
        active_symbols = {p["symbol"] for p in open_positions}
        # Sync open_symbols cache voi thuc te exchange
        self._open_symbols = active_symbols.copy()

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
                # Re-set SL/TP neu co du ca hai: dam bao trigger type = LastPrice (fix lech cu MarkPrice)
                if exchange_sl > 0 and exchange_tp > 0:
                    try:
                        _rt_tick = self._tick_size.get(symbol, 0.0)
                        self.client.set_sl_tp(symbol, exchange_sl, exchange_tp, tick_size=_rt_tick)
                        logger.info(f"{symbol}: Re-applied SL/TP with LastPrice trigger on restore")
                    except Exception as _re:
                        logger.warning(f"{symbol}: Re-apply SL/TP on restore failed: {_re!r}")
                # Dat _open_time = now (thoi diem phat hien, khong phai thoi diem tao lenh).
                # Neu dung created_ms (gio thuc), held_seconds sau restart = nhieu gio ->
                # anti-whipsaw 300s pass -> signal nguoc chieu dong lenh ngay khi restart.
                # Dung now: lenh duoc bao ve 300s tu khi bot bat dau chay (khong dong khi restart).
                self._open_time[symbol] = time.time()
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
            _pos_lev    = max(1.0, _fval(pos, "leverage", 10.0))

            # Fallback SL/TP neu ca saved lan exchange deu khong co (restart + SL mat)
            # CLAMP liq-safe: 5:1 tho o leverage cao (vd 100x -> SL dist 0.6% > liq 0.45%)
            # bi Bybit reject -> re-arm that bai vinh vien -> position khong co SL
            if saved_sl <= 0 and exchange_sl <= 0 and entry > 0:
                sl_roi   = min(config.TP_ROI_MIN * config.SL_TP_RATIO, max_safe_sl_roi(int(_pos_lev)))
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

            # Health check: exchange SL/TP lech khoi gia DA LUU -> re-arm gia da luu
            # KHONG ep ty le 5:1 nua - ty le la DONG (risk_manager nen ve 1:1-3:1 khi
            # liq clamp). Ep 5:1 cu tinh ra SL vuot gia thanh ly -> Bybit reject moi cycle.
            if exchange_sl > 0 and exchange_tp > 0 and saved_sl > 0 and saved_tp > 0:
                _sl_drift = abs(exchange_sl - saved_sl) / saved_sl
                _tp_drift = abs(exchange_tp - saved_tp) / saved_tp
                if _sl_drift > 0.005 or _tp_drift > 0.005:
                    logger.warning(
                        f"{symbol}: SL/TP drift khoi gia da luu "
                        f"(SL {exchange_sl:.6f} vs {saved_sl:.6f}, TP {exchange_tp:.6f} vs {saved_tp:.6f}) - re-arm"
                    )
                    try:
                        _hc_tick0 = self._tick_size.get(symbol, 0.0)
                        self.client.set_sl_tp(symbol, saved_sl, saved_tp, tick_size=_hc_tick0)
                    except Exception as e:
                        logger.error(f"{symbol}: drift re-arm FAILED: {str(e).encode('ascii','replace').decode()}")

            # Re-arm neu SL hoac TP bi mat tren exchange
            need_rearm_sl = saved_sl > 0 and exchange_sl <= 0
            need_rearm_tp = saved_tp > 0 and exchange_tp <= 0
            if need_rearm_sl or need_rearm_tp:
                rearm_sl = saved_sl if need_rearm_sl else exchange_sl
                rearm_tp = saved_tp if need_rearm_tp else exchange_tp
                logger.warning(
                    f"{symbol}: SL/TP missing (SL={'miss' if need_rearm_sl else 'ok'}, "
                    f"TP={'miss' if need_rearm_tp else 'ok'}) - re-arming"
                )
                rearm_ok = False
                try:
                    self.client.set_sl_tp(symbol, rearm_sl, rearm_tp)
                    logger.info(f"{symbol}: Re-armed SL={rearm_sl:.6f} TP={rearm_tp:.6f}")
                    rearm_ok = True
                except Exception as e:
                    logger.error(f"{symbol}: re-arm FAILED: {str(e).encode('ascii','replace').decode()}")

                if not rearm_ok:
                    # Gia co the di chuyen qua SL/TP goc -> thu recover hoac close
                    _hc_tick = self._tick_size.get(symbol, 0.0)
                    _hc_lev  = max(int(_pos_lev), 1)
                    self._recover_sl_tp(
                        symbol=symbol,
                        side=side,
                        entry=entry,
                        leverage=_hc_lev,
                        orig_sl=rearm_sl,
                        orig_tp=rearm_tp,
                        tick_size=_hc_tick,
                    )

            # Max hold-time exit: dong lenh neu ngam von qua lau ma dang lo
            # Tranh tinh huong: lenh sai chieu, gia khong hit SL, von bi giu hang gio
            # Nguong: 3h hold + lo > 20% ROI (chua hit SL nhung chac chan khong phuc hoi)
            _open_ts  = self._open_time.get(symbol, 0)
            _hold_sec = time.time() - _open_ts if _open_ts > 0 else 0
            if _hold_sec > 3 * 3600:  # 3 gio
                _pos_val = max(_fval(pos, "positionValue", 0), 1.0)
                _lev_h   = max(1.0, _fval(pos, "leverage", 10.0))
                _margin  = _pos_val / _lev_h
                _upnl    = _fval(pos, "unrealisedPnl")
                _pnl_roi = _upnl / _margin if _margin > 0 else 0
                if _pnl_roi < -0.20:  # lo > 20% ROI
                    logger.warning(
                        f"{symbol}: max-hold {_hold_sec/3600:.1f}h exceeded, "
                        f"PnL_ROI={_pnl_roi*100:.0f}% < -20% -> force close (save capital)"
                    )
                    self._close_position(pos)
                    continue
                # Stagnation exit: > 6h ma dang LO (roi < 0) -> dong, xoay vong von.
                # KHONG dong lenh dang LOI chi vi het gio (do la dumb timer) - dynamic-exit
                # se lo viec chot loi khi market quay dau. Lenh con xanh + trend con thuan -> giu.
                if _hold_sec > 6 * 3600 and _pnl_roi < 0:
                    logger.warning(
                        f"{symbol}: stagnation {_hold_sec/3600:.1f}h, PnL_ROI={_pnl_roi*100:.0f}% < 0 "
                        f"-> close to rotate capital"
                    )
                    self._close_position(pos)
                    continue

            # Emergency close: chi khi loss > 80% margin va SL exchange bi miss
            if self.risk_mgr.should_close_position(pos, mark_price):
                logger.warning(f"{symbol}: Emergency close - excessive loss")
                self._close_position(pos)

        # Xoa state stale
        for stale_sym in set(self._sl_price.keys()) - active_symbols:
            self.clear_position_state(stale_sym)

    def clear_position_state(self, symbol: str):
        self._sl_price.pop(symbol, None)
        self._tp_price.pop(symbol, None)
        self._open_time.pop(symbol, None)
        self._open_symbols.discard(symbol)
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
