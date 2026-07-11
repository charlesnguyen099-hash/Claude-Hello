"""
Risk Manager — Quản lý vốn và rủi ro thông minh
- Position sizing theo Kelly Criterion + ATR
- SL/TP động theo ATR
- Trailing stop tự động
- Portfolio exposure control
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional

import config
from client import BybitClient
from strategies.base import Signal

logger = logging.getLogger(__name__)


@dataclass
class TradeParams:
    symbol: str
    side: str           # "Buy" | "Sell"
    qty: float
    leverage: int
    sl_price: float
    tp1_price: float    # Đóng 50% tại TP1
    tp2_price: float    # Đóng 50% còn lại tại TP2
    trailing_stop: float
    notional_usdt: float


class RiskManager:
    def __init__(self, client: BybitClient):
        self.client = client

    def compute_trade(
        self,
        signal: Signal,
        equity: float,
        open_positions: list[dict],
    ) -> Optional[TradeParams]:
        """
        Tính toán tham số lệnh tối ưu.
        Trả về None nếu không đủ điều kiện vào lệnh.
        """
        # Kiểm tra giới hạn vị thế
        if len(open_positions) >= config.MAX_OPEN_POSITIONS:
            logger.debug(f"Max positions reached ({config.MAX_OPEN_POSITIONS})")
            return None

        sides    = [p["side"] for p in open_positions]
        n_long   = sides.count("Buy")
        n_short  = sides.count("Sell")

        if signal.direction == 1 and n_long >= config.MAX_POSITIONS_PER_SIDE:
            return None
        if signal.direction == -1 and n_short >= config.MAX_POSITIONS_PER_SIDE:
            return None

        if signal.atr <= 0 or signal.entry_price <= 0:
            return None

        side = "Buy" if signal.direction == 1 else "Sell"

        # ── Leverage tối ưu theo volatility ──────────────────────────────────
        # Volatility = ATR / price (%) → leverage tỷ lệ nghịch
        atr_pct   = signal.atr / signal.entry_price
        leverage  = min(
            config.MAX_LEVERAGE,
            max(2, int(0.05 / atr_pct))  # target 5% move = 1× leverage unit
        )

        # ── Position sizing theo Risk per trade ───────────────────────────────
        # Risk amount = equity × 1%
        # SL distance = 1.5 × ATR
        risk_usdt  = equity * config.ACCOUNT_RISK_PCT
        sl_dist    = config.SL_ATR_MULTIPLIER * signal.atr
        # qty = risk_usdt / sl_dist (tính theo contract size)
        qty_raw    = risk_usdt / sl_dist
        # Nhân leverage để tính notional
        notional   = qty_raw * signal.entry_price

        qty = self._round_qty(qty_raw, signal.entry_price, signal.symbol)
        if qty <= 0:
            return None

        # ── SL / TP prices ────────────────────────────────────────────────────
        d      = signal.direction
        sl     = signal.entry_price - d * config.SL_ATR_MULTIPLIER  * signal.atr
        tp1    = signal.entry_price + d * config.TP1_ATR_MULTIPLIER * signal.atr
        tp2    = signal.entry_price + d * config.TP2_ATR_MULTIPLIER * signal.atr
        trail  = config.TRAILING_STOP_ATR * signal.atr

        return TradeParams(
            symbol=signal.strategy_name,  # được ghi đè bởi executor
            side=side,
            qty=qty,
            leverage=leverage,
            sl_price=round(sl, 6),
            tp1_price=round(tp1, 6),
            tp2_price=round(tp2, 6),
            trailing_stop=round(trail, 6),
            notional_usdt=round(qty * signal.entry_price, 2),
        )

    def _round_qty(self, qty: float, price: float, symbol: str) -> float:
        """Làm tròn qty theo bước tối thiểu của Bybit."""
        try:
            info     = self.client.get_instrument_info(symbol)
            lot_step = float(info["lotSizeFilter"]["qtyStep"])
            min_qty  = float(info["lotSizeFilter"]["minOrderQty"])
            qty      = math.floor(qty / lot_step) * lot_step
            qty      = round(qty, 10)
            return qty if qty >= min_qty else 0.0
        except Exception as e:
            logger.warning(f"Could not get instrument info for {symbol}: {e}")
            # Fallback: làm tròn 3 chữ số thập phân
            return round(qty, 3) if qty > 0 else 0.0

    def should_close_position(self, position: dict, current_price: float) -> bool:
        """Kiểm tra xem có cần đóng vị thế sớm không (trailing stop logic)."""
        # Bybit tự xử lý trailing stop khi đặt lệnh — hàm này dự phòng
        unrealised_pnl_pct = float(position.get("unrealisedPnl", 0)) / (
            float(position.get("positionValue", 1)) or 1
        )
        # Cắt lỗ bổ sung nếu thua > 5% (on top of normal SL)
        if unrealised_pnl_pct < -0.05:
            return True
        return False
