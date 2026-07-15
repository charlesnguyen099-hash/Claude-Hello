"""
Trade Executor — thực thi lệnh và quản lý vị thế
- Đặt lệnh với SL + TP1 (safety net trên sàn)
- Level 1 (BREAKEVEN_TRIGGER=30%): chuyển SL về break-even sớm
- Level 2 (PARTIAL_CLOSE_TRIGGER=75%): đóng 50% vị thế, cập nhật TP lên TP2, xác nhận SL break-even
- Level 3: 50% còn lại chạy đến TP2 với zero downside risk
- Tự động đóng lệnh khi signal đảo chiều
"""

import logging
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

        self._partial_closed: dict[str, bool] = {}
        self._breakeven_set: dict[str, bool]  = {}
        self._atr: dict[str, float]           = {}
        self._tp2_price: dict[str, float]     = {}
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

            logger.info(f"{symbol}: Signal reversal — closing {pos_side} before entering {signal_side}")
            self._close_position(existing[0])
            time.sleep(0.5)

        params = self.risk_mgr.compute_trade(signal, equity, open_positions, is_priority=is_priority)
        if not params:
            return

        params.symbol = symbol
        self._enter_trade(symbol, signal, params)

    def _enter_trade(self, symbol: str, signal: Signal, params: TradeParams):
        try:
            self.client.set_leverage(symbol, params.leverage)

            # TP1 dat tren san lam safety net — partial close se cap nhat len TP2 khi dat 75% TP1
            order = self.client.place_order(
                symbol=symbol,
                side=params.side,
                qty=params.qty,
                sl=params.sl_price,
                tp=params.tp1_price,
            )

            self._partial_closed[symbol] = False
            self._breakeven_set[symbol]  = False
            self._atr[symbol]            = signal.atr
            self._tp2_price[symbol]      = params.tp2_price

            self.logger.log_trade({
                "event":     "open",
                "symbol":    symbol,
                "side":      params.side,
                "qty":       params.qty,
                "leverage":  params.leverage,
                "entry":     signal.entry_price,
                "sl":        params.sl_price,
                "tp1":       params.tp1_price,
                "tp2":       params.tp2_price,
                "strategy":  signal.strategy_name,
                "strength":  signal.strength,
                "reason":    signal.reason,
                "notional":  params.notional_usdt,
                "capital":   params.capital_usdt,
                "fee":       params.fee_usdt,
                "order_id":  order.get("orderId", ""),
            })

            logger.info(
                f"[OPEN] {symbol} {params.side} | qty={params.qty} | "
                f"lev={params.leverage}x | SL={params.sl_price:.4f} | "
                f"TP1={params.tp1_price:.4f} | strategy={signal.strategy_name} | "
                f"{signal.reason}"
            )

        except Exception as e:
            logger.error(f"Failed to enter trade {symbol}: {str(e).encode('ascii', 'replace').decode()}")

    def manage_open_positions(self, open_positions: list[dict]):
        """
        Quan ly vi the dang mo theo 3 muc:
        1. BREAKEVEN_TRIGGER (30%): doi SL ve entry + phi som
        2. PARTIAL_CLOSE_TRIGGER (75%): dong 50% reduce-only, cap nhat TP len TP2, xac nhan breakeven SL
        3. 50% con lai chay den TP2 voi zero downside risk (SL = breakeven)
        """
        for pos in open_positions:
            symbol     = pos["symbol"]
            entry      = float(pos["avgPrice"])
            mark_price = float(pos.get("markPrice", entry))
            side       = pos["side"]

            tp1_threshold = float(pos.get("takeProfit", 0))

            if tp1_threshold <= 0:
                if self.risk_mgr.should_close_position(pos, mark_price):
                    logger.warning(f"{symbol}: Emergency close — excessive loss")
                    self._close_position(pos)
                continue

            dist_to_tp1 = abs(tp1_threshold - entry)
            if side == "Buy":
                dist_moved = mark_price - entry
            else:
                dist_moved = entry - mark_price

            # --- Muc 1: Break-even SL tai BREAKEVEN_TRIGGER% (30%) duong den TP1 ---
            if not self._breakeven_set.get(symbol, False) and dist_to_tp1 > 0 and dist_moved > 0:
                if dist_moved >= dist_to_tp1 * config.BREAKEVEN_TRIGGER:
                    try:
                        fee_buffer = entry * config.ROUND_TRIP_FEE
                        be_price   = entry + fee_buffer if side == "Buy" else entry - fee_buffer
                        self.client.update_stop_loss(symbol, round(be_price, 6))
                        self._breakeven_set[symbol] = True
                        logger.info(
                            f"{symbol}: Break-even SL -> {be_price:.4f} "
                            f"(moved {dist_moved:.4f}/{dist_to_tp1:.4f} = "
                            f"{dist_moved/dist_to_tp1*100:.0f}% toward TP1)"
                        )
                    except Exception as e:
                        logger.warning(f"{symbol}: Could not set break-even SL: {e}")

            # --- Muc 2: Partial close tai PARTIAL_CLOSE_TRIGGER% (75%) duong den TP1 ---
            # Dong 50% vi the, cap nhat TP tu TP1 sang TP2, xac nhan SL = breakeven
            if not self._partial_closed.get(symbol, False) and dist_to_tp1 > 0 and dist_moved > 0:
                if dist_moved >= dist_to_tp1 * config.PARTIAL_CLOSE_TRIGGER:
                    self._partial_closed[symbol] = True
                    try:
                        # Cap nhat TP tren san tu TP1 sang TP2
                        tp2 = self._tp2_price.get(symbol, 0.0)
                        if tp2 > 0:
                            self.client.update_take_profit(symbol, tp2)
                            logger.info(f"{symbol}: TP updated TP1={tp1_threshold:.4f} -> TP2={tp2:.4f}")

                        # Dong 50% vi the
                        pos_qty = float(pos["size"])
                        partial_qty = round(pos_qty * 0.5, 8)
                        if partial_qty > 0:
                            close_side = "Sell" if side == "Buy" else "Buy"
                            self.client.place_order(symbol, close_side, partial_qty, reduce_only=True)
                            logger.info(
                                f"{symbol}: Partial close 50% ({partial_qty}) at "
                                f"{dist_moved/dist_to_tp1*100:.0f}% of TP1 | "
                                f"remaining 50% targets TP2={tp2:.4f}"
                            )

                        # Xac nhan breakeven SL neu chua set
                        if not self._breakeven_set.get(symbol, False):
                            fee_buffer = entry * config.ROUND_TRIP_FEE
                            be_price   = entry + fee_buffer if side == "Buy" else entry - fee_buffer
                            self.client.update_stop_loss(symbol, round(be_price, 6))
                            self._breakeven_set[symbol] = True
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
