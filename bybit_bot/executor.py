"""
Trade Executor — thực thi lệnh và quản lý vị thế
- Đặt lệnh với SL + TP1 (safety net trên sàn)
- Level 1 (BREAKEVEN_TRIGGER=20%): chuyển SL về break-even sớm
- Level 2 (PARTIAL_CLOSE_TRIGGER=75%): đóng 50% vị thế, cập nhật TP lên TP2, xác nhận SL break-even
- Level 3: 50% còn lại chạy đến TP2 với zero downside risk
- Anti-whipsaw: khong force-close vi the < 30 phut vi signal dao chieu
- Tự động đóng lệnh khi signal đảo chiều (sau 30 phut hoac PnL < -20%)

Defensive layers (execution):
  1. Stale signal check (0.3%) trong execute_signal
  2. Spread check — abort neu spread > nguong hoac SL dist < 2x spread
  3. IOC Limit order — tranh market order slippage trong dump/pump nhanh
  4. Fill verification — xac nhan IOC duoc fill truoc khi update state
  5. SL verification + re-arm — dam bao SL luon active sau khi lenh vao
  6. Partial close race condition fix — chi update flag sau khi close thanh cong
  7. Periodic SL check — re-arm SL neu bi huy trong khi quan ly vi the
"""

import logging
import math
import time
from typing import Callable, Optional

from client import BybitClient
from risk_manager import RiskManager, TradeParams
from strategies.base import Signal
from bot_logger import BotLogger
import config

logger = logging.getLogger(__name__)


class Executor:
    def __init__(self, client: BybitClient, risk_mgr: RiskManager, bot_logger: BotLogger):
        self.client    = client
        self.risk_mgr  = risk_mgr
        self.logger    = bot_logger

        self._partial_closed: dict[str, bool]  = {}
        self._breakeven_set: dict[str, bool]   = {}
        self._atr: dict[str, float]            = {}
        self._tp2_price: dict[str, float]      = {}
        self._sl_price: dict[str, float]       = {}   # sl ban dau de re-arm neu mat
        self._open_time: dict[str, float]      = {}
        self._sl_verified: dict[str, bool]     = {}   # da verify SL sau fill chua
        # Callback duoc goi khi dong lenh lo — (symbol: str, side: str) -> None
        self.on_loss_callback: Optional[Callable[..., None]] = None

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
                notional = float(existing[0].get("positionValue", 1)) or 1
                lev      = float(existing[0].get("leverage", 1)) or 1
                margin   = notional / lev
                pnl      = float(existing[0].get("unrealisedPnl", 0))
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

            logger.info(f"{symbol}: Signal reversal → close {pos_side} (held={held_seconds:.0f}s, "
                        f"PnL={pos_pnl_pct*100:.1f}%) → enter {signal_side}")
            self._close_position(existing[0])
            time.sleep(0.5)

        # STALE SIGNAL CHECK (0.3%): bat drift xay ra giua analysis va dat lenh
        live_price = self.client.get_current_price(symbol)
        if live_price <= 0:
            logger.warning(f"{symbol}: SKIP — khong lay duoc live price, bo qua de tranh stale entry")
            return
        if signal.entry_price > 0:
            price_drift = abs(live_price - signal.entry_price) / signal.entry_price
            if price_drift > 0.003:
                logger.warning(
                    f"{symbol}: STALE SIGNAL — live={live_price:.6f} vs entry={signal.entry_price:.6f} "
                    f"drift={price_drift*100:.2f}% > 0.3% → skip"
                )
                return

        params = self.risk_mgr.compute_trade(signal, equity, open_positions, is_priority=is_priority)
        if not params:
            return

        params.symbol = symbol
        self._enter_trade(symbol, signal, params)

    def _enter_trade(self, symbol: str, signal: Signal, params: TradeParams):
        """
        Dat lenh vao vi the voi day du protection layers:
        1. Lay bid/ask → spread check → SL-vs-spread check
        2. Pre-order drift check (0.5%) → abort neu gia di xa tu stale check
        3. IOC Limit order tai bid/ask → fill ngay hoac huy (khong slip)
        4. Verify fill → neu IOC khong fill → skip (thi truong da di xa)
        5. Verify SL active → re-arm neu SL khong duoc dat sau fill
        """
        try:
            self.client.set_leverage(symbol, params.leverage)

            # --- Lay bid/ask va tick_size ---
            bid, ask = self.client.get_bid_ask(symbol)
            if bid <= 0 or ask <= 0:
                logger.warning(f"{symbol}: ABORT entry — khong lay duoc bid/ask")
                return

            mid_price = (bid + ask) / 2.0
            spread    = ask - bid
            spread_pct = spread / mid_price if mid_price > 0 else 0

            # Get tick_size (can thiet cho round SL/TP va limit price)
            tick_size = 0.0
            try:
                info      = self.client.get_instrument_info(symbol)
                tick_size = float(info["priceFilter"]["tickSize"])
            except Exception:
                pass

            # --- Check 1: Spread qua rong → thanh khoan kem, khong trade ---
            is_largecap = symbol in {"BTCUSDT", "ETHUSDT"}
            max_spread  = config.MAX_SPREAD_PCT_LARGE if is_largecap else config.MAX_SPREAD_PCT_ALT
            if spread_pct > max_spread:
                logger.warning(
                    f"{symbol}: ABORT — spread={spread_pct*100:.3f}% > {max_spread*100:.3f}% "
                    f"(low liquidity, spread={spread:.6f})"
                )
                return

            # --- Check 2: SL distance vs spread ---
            # Neu SL qua gan entry so voi spread, price noise co the hit SL ngay
            sl_dist = abs(params.sl_price - signal.entry_price)
            if spread > 0 and sl_dist < spread * 2.5:
                logger.warning(
                    f"{symbol}: ABORT — SL dist={sl_dist:.6f} < 2.5x spread={spread*2.5:.6f} "
                    f"(SL too tight for current spread)"
                )
                return

            # --- Check 3: Pre-order drift 0.5% (sau stale check 0.3% trong execute_signal) ---
            # Bat gia di chuyen giua 2 lan check (stale check → set_leverage → get_bid_ask)
            if signal.entry_price > 0:
                pre_drift = abs(mid_price - signal.entry_price) / signal.entry_price
                if pre_drift > 0.005:
                    logger.warning(
                        f"{symbol}: ABORT — pre-order drift {pre_drift*100:.2f}% > 0.5% "
                        f"(entry={signal.entry_price:.6f}, now={mid_price:.6f})"
                    )
                    return

            # --- Dat IOC Limit order ---
            # SHORT (Sell): limit tai bid — fill neu thi truong van o day hoac cao hon
            #               Neu gia dump xuong (bid giam manh), IOC huy → tranh vao SHORT giua dump
            # LONG  (Buy):  limit tai ask — fill neu thi truong van o day hoac thap hon
            #               Neu gia pump len (ask tang manh), IOC huy → tranh vao LONG giua pump
            if params.side == "Sell":
                limit_price = self.client.round_to_tick(bid, tick_size) if tick_size > 0 else round(bid, 6)
            else:
                limit_price = self.client.round_to_tick(ask, tick_size) if tick_size > 0 else round(ask, 6)

            # Round SL/TP theo tick size de dam bao Bybit chap nhan
            sl_rounded  = self.client.round_to_tick(params.sl_price,  tick_size) if tick_size > 0 else round(params.sl_price,  6)
            tp1_rounded = self.client.round_to_tick(params.tp1_price, tick_size) if tick_size > 0 else round(params.tp1_price, 6)

            order = self.client.place_order(
                symbol=symbol,
                side=params.side,
                qty=params.qty,
                order_type="Limit",
                sl=sl_rounded,
                tp=tp1_rounded,
                limit_price=limit_price,
                tick_size=tick_size,
            )

            order_id = order.get("orderId", "")

            # --- Verify IOC fill ---
            # IOC Limit: fill ngay hoac huy — khong bao gio treo
            # Neu huy: gia da di xa khoi limit → dung mo lenh (tranh chase)
            time.sleep(0.4)
            status = self.client.get_order_status(symbol, order_id)
            if status not in ("Filled", "PartiallyFilled"):
                logger.warning(
                    f"{symbol}: IOC Limit NOT filled (status={status}) — "
                    f"market moved away from limit={limit_price:.6f}, trade skipped"
                )
                return

            # Update in-memory state sau khi xac nhan fill
            self._partial_closed[symbol] = False
            self._breakeven_set[symbol]  = False
            self._sl_verified[symbol]    = False
            self._atr[symbol]            = signal.atr
            self._tp2_price[symbol]      = params.tp2_price
            self._sl_price[symbol]       = sl_rounded
            self._open_time[symbol]      = time.time()

            # --- Verify SL active sau fill ---
            # Bybit doi khi khong attach SL/TP vao Limit order ngay lap tuc
            # → check va force-set neu thieu
            time.sleep(0.5)
            has_sl, actual_sl = self.client.verify_position_sl(symbol)
            if not has_sl:
                logger.warning(
                    f"{symbol}: SL missing after fill — force setting SL={sl_rounded:.6f}"
                )
                self.client.update_stop_loss(symbol, sl_rounded)
            self._sl_verified[symbol] = True

            self.logger.log_trade({
                "event":     "open",
                "symbol":    symbol,
                "side":      params.side,
                "qty":       params.qty,
                "leverage":  params.leverage,
                "entry":     signal.entry_price,
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
                f"lev={params.leverage}x | limit={limit_price:.6f} | "
                f"SL={sl_rounded:.6f} | TP1={tp1_rounded:.6f} | "
                f"strategy={signal.strategy_name} | {signal.reason}"
            )

        except Exception as e:
            logger.error(f"Failed to enter trade {symbol}: {str(e).encode('ascii', 'replace').decode()}")

    def manage_open_positions(self, open_positions: list[dict]):
        """
        Quan ly vi the dang mo theo 3 muc:
        1. BREAKEVEN_TRIGGER: doi SL ve entry + phi som
        2. PARTIAL_CLOSE_TRIGGER: dong 50% reduce-only, cap nhat TP len TP2, xac nhan breakeven SL
        3. 50% con lai chay den TP2 voi zero downside risk (SL = breakeven)

        Defensive: kiem tra SL con active khong, re-arm neu mat.
        """
        for pos in open_positions:
            symbol     = pos["symbol"]
            entry      = float(pos["avgPrice"])
            mark_price = float(pos.get("markPrice", entry))
            side       = pos["side"]

            # --- Periodic SL health check ---
            # Neu SL bi huy tren exchange (maintenance, loi API, v.v.), re-arm ngay
            # Chi check cho cac vi the ma bot nay da mo (co _sl_price)
            saved_sl = self._sl_price.get(symbol, 0.0)
            if saved_sl > 0:
                exchange_sl = float(pos.get("stopLoss", 0))
                if exchange_sl <= 0:
                    logger.warning(
                        f"{symbol}: SL missing on exchange — re-arming SL={saved_sl:.6f}"
                    )
                    try:
                        self.client.update_stop_loss(symbol, saved_sl)
                    except Exception as e:
                        logger.error(f"{symbol}: Failed to re-arm SL: {e}")

            tp1_threshold = float(pos.get("takeProfit", 0))

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

            # --- Muc 1: Break-even SL tai BREAKEVEN_TRIGGER% (20%) duong den TP1 ---
            if not self._breakeven_set.get(symbol, False) and dist_to_tp1 > 0 and dist_moved > 0:
                if dist_moved >= dist_to_tp1 * config.BREAKEVEN_TRIGGER:
                    try:
                        fee_buffer = entry * config.ROUND_TRIP_FEE
                        be_price   = entry + fee_buffer if side == "Buy" else entry - fee_buffer
                        self.client.update_stop_loss(symbol, round(be_price, 6))
                        self._breakeven_set[symbol] = True
                        # Cap nhat saved_sl voi gia moi
                        self._sl_price[symbol] = round(be_price, 6)
                        logger.info(
                            f"{symbol}: Break-even SL -> {be_price:.4f} "
                            f"(moved {dist_moved:.4f}/{dist_to_tp1:.4f} = "
                            f"{dist_moved/dist_to_tp1*100:.0f}% toward TP1)"
                        )
                    except Exception as e:
                        logger.warning(f"{symbol}: Could not set break-even SL: {e}")

            # --- Muc 2: Partial close tai PARTIAL_CLOSE_TRIGGER% (75%) duong den TP1 ---
            # FIX: _partial_closed chi duoc set True SAU KHI close order thanh cong
            # (truoc day set True truoc → neu close fail, bot nghi da close nhung khong phai)
            if not self._partial_closed.get(symbol, False) and dist_to_tp1 > 0 and dist_moved > 0:
                if dist_moved >= dist_to_tp1 * config.PARTIAL_CLOSE_TRIGGER:
                    try:
                        # Cap nhat TP tren san tu TP1 sang TP2
                        tp2 = self._tp2_price.get(symbol, 0.0)
                        if tp2 > 0:
                            self.client.update_take_profit(symbol, tp2)
                            logger.info(f"{symbol}: TP updated TP1={tp1_threshold:.4f} -> TP2={tp2:.4f}")

                        # Dong 50% vi the — align voi qty_step cua instrument
                        pos_qty = float(pos["size"])
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

                        # Xac nhan breakeven SL neu chua set
                        if not self._breakeven_set.get(symbol, False):
                            fee_buffer = entry * config.ROUND_TRIP_FEE
                            be_price   = entry + fee_buffer if side == "Buy" else entry - fee_buffer
                            self.client.update_stop_loss(symbol, round(be_price, 6))
                            self._breakeven_set[symbol] = True
                            self._sl_price[symbol] = round(be_price, 6)
                            logger.info(f"{symbol}: Breakeven SL confirmed -> {be_price:.4f}")

                    except Exception as e:
                        logger.warning(f"{symbol}: Could not execute partial close: {e}")

            # --- Emergency close ---
            if self.risk_mgr.should_close_position(pos, mark_price):
                logger.warning(f"{symbol}: Emergency close — excessive loss")
                self._close_position(pos)

    def _close_position(self, position: dict):
        symbol = position["symbol"]
        side   = position["side"]
        qty    = float(position["size"])
        pnl    = float(position.get("unrealisedPnl", 0))
        try:
            self.client.close_position(symbol, side, qty)
            # Clear in-memory state cho symbol nay
            self._partial_closed.pop(symbol, None)
            self._breakeven_set.pop(symbol, None)
            self._atr.pop(symbol, None)
            self._tp2_price.pop(symbol, None)
            self._sl_price.pop(symbol, None)
            self._sl_verified.pop(symbol, None)
            self._open_time.pop(symbol, None)
            self.logger.log_trade({
                "event":  "close",
                "symbol": symbol,
                "side":   side,
                "qty":    qty,
                "reason": "signal_reversal_or_emergency",
            })
            logger.info(f"[CLOSE] {symbol} {side} qty={qty} pnl={pnl:.4f}")
            if pnl < 0 and self.on_loss_callback:
                self.on_loss_callback(symbol, side)
        except Exception as e:
            logger.error(f"Failed to close position {symbol}: {str(e).encode('ascii', 'replace').decode()}")
