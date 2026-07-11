"""
Trade Executor — thực thi lệnh và quản lý vị thế
- Đặt lệnh với SL + TP1 ngay khi vào
- Theo dõi TP2 bằng trailing stop
- Tự động đóng lệnh khi signal đảo chiều
"""

import logging
import time
from typing import Optional

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

        # Theo dõi TP1 đã hit chưa: {symbol: bool}
        self._tp1_hit: dict[str, bool] = {}

    def execute_signal(
        self,
        symbol: str,
        signal: Signal,
        equity: float,
        open_positions: list[dict],
    ):
        """Xử lý signal mới — vào lệnh nếu đủ điều kiện."""
        if signal.direction == 0:
            return

        # Kiểm tra nếu đã có vị thế cùng chiều cho symbol này
        existing = [p for p in open_positions if p["symbol"] == symbol]
        if existing:
            pos_side = existing[0]["side"]
            signal_side = "Buy" if signal.direction == 1 else "Sell"
            if pos_side == signal_side:
                return  # Không pyramid cùng chiều

            # Signal ngược chiều → đóng vị thế cũ
            logger.info(f"{symbol}: Signal reversal — closing {pos_side} before entering {signal_side}")
            self._close_position(existing[0])
            time.sleep(0.5)

        params = self.risk_mgr.compute_trade(signal, equity, open_positions)
        if not params:
            return

        params.symbol = symbol
        self._enter_trade(symbol, signal, params)

    def _enter_trade(self, symbol: str, signal: Signal, params: TradeParams):
        try:
            # Set leverage
            self.client.set_leverage(symbol, params.leverage)

            # Vào lệnh với SL + TP1 (đóng 100% tại TP1, sau đó trailing)
            order = self.client.place_order(
                symbol=symbol,
                side=params.side,
                qty=params.qty,
                sl=params.sl_price,
                tp=params.tp1_price,
            )

            self._tp1_hit[symbol] = False

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
                "order_id":  order.get("orderId", ""),
            })

            logger.info(
                f"[OPEN] {symbol} {params.side} | qty={params.qty} | "
                f"lev={params.leverage}x | SL={params.sl_price:.4f} | "
                f"TP1={params.tp1_price:.4f} | strategy={signal.strategy_name} | "
                f"{signal.reason}"
            )

        except Exception as e:
            logger.error(f"Failed to enter trade {symbol}: {e}")

    def manage_open_positions(self, open_positions: list[dict]):
        """Kiểm tra trailing stop + TP2 logic cho mỗi vị thế đang mở."""
        for pos in open_positions:
            symbol        = pos["symbol"]
            size          = float(pos["size"])
            entry         = float(pos["avgPrice"])
            mark_price    = float(pos.get("markPrice", entry))
            side          = pos["side"]
            unrealised    = float(pos.get("unrealisedPnl", 0))

            # Kích hoạt trailing stop khi đạt TP1
            if not self._tp1_hit.get(symbol, False):
                tp1_threshold = float(pos.get("takeProfit", 0))
                if tp1_threshold > 0:
                    if (side == "Buy"  and mark_price >= tp1_threshold) or \
                       (side == "Sell" and mark_price <= tp1_threshold):
                        self._tp1_hit[symbol] = True
                        # Đặt trailing stop để bảo vệ phần còn lại
                        try:
                            self.client.set_trading_stop(symbol, side, float(pos.get("trailingStop", 0)) or
                                                          abs(mark_price - entry) * 0.5)
                            logger.info(f"{symbol}: TP1 hit — trailing stop activated")
                        except Exception as e:
                            logger.warning(f"{symbol}: Could not set trailing stop: {e}")

            # Emergency close nếu vượt ngưỡng rủi ro
            if self.risk_mgr.should_close_position(pos, mark_price):
                logger.warning(f"{symbol}: Emergency close — excessive loss")
                self._close_position(pos)

    def _close_position(self, position: dict):
        symbol = position["symbol"]
        side   = position["side"]
        qty    = float(position["size"])
        try:
            self.client.close_position(symbol, side, qty)
            self.logger.log_trade({
                "event":  "close",
                "symbol": symbol,
                "side":   side,
                "qty":    qty,
                "reason": "signal_reversal_or_emergency",
            })
            logger.info(f"[CLOSE] {symbol} {side} qty={qty}")
        except Exception as e:
            logger.error(f"Failed to close position {symbol}: {e}")
