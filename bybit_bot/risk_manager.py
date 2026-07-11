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

        # ── Lấy leverage tối đa từ Bybit cho symbol này ───────────────────────
        if config.USE_MAX_LEVERAGE:
            leverage = self.client.get_max_leverage(signal.symbol)
        else:
            atr_pct  = signal.atr / signal.entry_price
            leverage = min(config.MAX_LEVERAGE, max(2, int(0.05 / atr_pct)))

        # ── Position sizing theo Risk per trade ───────────────────────────────
        # Risk amount = equity × ACCOUNT_RISK_PCT
        # SL distance = SL_ATR_MULTIPLIER × ATR
        # qty = risk_usdt / sl_dist
        risk_usdt = equity * config.ACCOUNT_RISK_PCT
        sl_dist   = config.SL_ATR_MULTIPLIER * signal.atr
        qty_raw   = risk_usdt / sl_dist

        qty = self._round_qty(qty_raw, signal.entry_price, signal.symbol)
        if qty <= 0:
            logger.debug(f"{signal.symbol}: qty=0 sau khi làm tròn (equity quá thấp?)")
            return None

        notional = qty * signal.entry_price

        # ── Kiểm tra notional >= min order của Bybit ──────────────────────────
        min_notional = self.client.get_min_order_usdt(signal.symbol)
        if notional < min_notional:
            # Tự động tăng qty lên đủ min order
            min_qty_info = self.client.get_instrument_info(signal.symbol)
            min_qty      = float(min_qty_info["lotSizeFilter"]["minOrderQty"])
            qty          = min_qty
            notional     = qty * signal.entry_price
            logger.debug(
                f"{signal.symbol}: notional {notional:.2f} < min {min_notional:.2f} USDT "
                f"→ dùng min qty={qty}"
            )

        # ── SL / TP prices ────────────────────────────────────────────────────
        d     = signal.direction
        sl    = signal.entry_price - d * config.SL_ATR_MULTIPLIER  * signal.atr
        tp1   = signal.entry_price + d * config.TP1_ATR_MULTIPLIER * signal.atr
        tp2   = signal.entry_price + d * config.TP2_ATR_MULTIPLIER * signal.atr
        trail = config.TRAILING_STOP_ATR * signal.atr

        logger.debug(
            f"{signal.symbol}: leverage={leverage}x | qty={qty} | "
            f"notional={notional:.2f} USDT | SL={sl:.4f} | TP1={tp1:.4f}"
        )

        return TradeParams(
            symbol=signal.strategy_name,  # được ghi đè bởi executor
            side=side,
            qty=qty,
            leverage=leverage,
            sl_price=round(sl, 6),
            tp1_price=round(tp1, 6),
            tp2_price=round(tp2, 6),
            trailing_stop=round(trail, 6),
            notional_usdt=round(notional, 2),
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
