"""
Risk Manager
- SL co dinh: khoang cach gia de mat dung 30% capital bo vao lenh
- TP thuc te theo ATR thi truong — co the < SL, khong ep R:R
- Phi 0.11% tich hop vao ca SL lan TP
- Von moi lenh = 10% equity, khong all-in
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
    side: str
    qty: float
    leverage: int
    sl_price: float
    tp1_price: float
    tp2_price: float
    trailing_stop: float
    notional_usdt: float
    fee_usdt: float
    capital_usdt: float
    sl_pct: float
    tp1_pct: float
    tp2_pct: float


class RiskManager:
    def __init__(self, client: BybitClient):
        self.client = client

    def compute_trade(
        self,
        signal: Signal,
        equity: float,
        open_positions: list[dict],
    ) -> Optional[TradeParams]:

        if signal.entry_price <= 0 or signal.atr <= 0:
            return None

        side = "Buy" if signal.direction == 1 else "Sell"

        # Lay leverage toi da cua cap nay
        leverage = self.client.get_max_leverage(signal.symbol) if config.USE_MAX_LEVERAGE \
                   else config.DEFAULT_LEVERAGE

        # Qty = min order quantity cua Bybit voi max leverage
        # Khong can tinh % von — cu dung muc toi thieu de trade duoc la vao
        try:
            info     = self.client.get_instrument_info(signal.symbol)
            min_qty  = float(info["lotSizeFilter"]["minOrderQty"])
            qty_step = float(info["lotSizeFilter"]["qtyStep"])
        except Exception as e:
            logger.warning(f"{signal.symbol}: cannot get instrument info: {e}")
            return None

        # Scale qty theo consensus va strength: nhieu strategies dong thuan + signal manh -> vao nhieu hon
        # scale = consensus x (strength / 0.6), cap tai QTY_SCALE_CAP
        consensus    = getattr(signal, 'consensus', 1)
        scale_factor = min(consensus * (signal.strength / config.MIN_SIGNAL_STRENGTH), config.QTY_SCALE_CAP)
        scale_factor = max(1.0, scale_factor)

        qty      = math.ceil(min_qty * scale_factor / qty_step) * qty_step
        notional = qty * signal.entry_price

        # Bybit minimum order value is 5 USDT — scale qty up if needed
        MIN_NOTIONAL = 5.0
        if notional < MIN_NOTIONAL:
            qty      = math.ceil(MIN_NOTIONAL / signal.entry_price / qty_step) * qty_step
            notional = qty * signal.entry_price

        capital_used = notional / leverage

        # Phi round-trip
        fee_usdt  = notional * config.ROUND_TRIP_FEE
        fee_price = signal.entry_price * config.ROUND_TRIP_FEE

        # SL co dinh: tinh khoang cach gia de mat dung 30% capital_used
        # loss = qty x sl_dist => sl_dist = (capital_used x 30%) / qty
        sl_dist = (capital_used * config.SL_MAX_LOSS_PCT) / qty
        # SL phai lon hon phi toi thieu (khong the dat SL sat phi)
        sl_dist = max(sl_dist, fee_price * 3)

        # TP toi da: gross profit = phi + 50% capital
        # => net profit sau phi = 50% capital
        # tp_dist = (fee_usdt + capital_used * 0.50) / qty
        max_tp_dist = (fee_usdt + capital_used * 0.25) / qty

        # TP theo ATR thi truong, cap o muc toi da
        tp1_dist = min(config.TP1_ATR_MULT * signal.atr, max_tp_dist)
        tp2_dist = min(config.TP2_ATR_MULT * signal.atr, max_tp_dist)
        trail    = config.TRAILING_STOP_ATR * signal.atr

        d   = signal.direction
        sl  = signal.entry_price - d * (sl_dist + fee_price)
        tp1 = signal.entry_price + d * tp1_dist
        tp2 = signal.entry_price + d * tp2_dist

        sl_pct  = sl_dist  / signal.entry_price * 100
        tp1_pct = tp1_dist / signal.entry_price * 100
        tp2_pct = tp2_dist / signal.entry_price * 100

        # Log ro rang de biet RR thuc te
        rr1 = tp1_pct / sl_pct if sl_pct > 0 else 0
        rr2 = tp2_pct / sl_pct if sl_pct > 0 else 0
        logger.info(
            f"{signal.symbol}: {side} lev={leverage}x | "
            f"qty={qty} | notional={notional:.2f}$ | "
            f"capital={capital_used:.2f}$ ({capital_used/equity*100:.1f}% eq) | "
            f"fee={fee_usdt:.4f}$ | "
            f"SL=-{sl_pct:.2f}% | TP1=+{tp1_pct:.2f}% (RR={rr1:.2f}) | "
            f"TP2=+{tp2_pct:.2f}% (RR={rr2:.2f})"
        )

        return TradeParams(
            symbol=signal.strategy_name,
            side=side,
            qty=qty,
            leverage=leverage,
            sl_price=round(sl, 6),
            tp1_price=round(tp1, 6),
            tp2_price=round(tp2, 6),
            trailing_stop=round(trail, 6),
            notional_usdt=round(notional, 2),
            fee_usdt=round(fee_usdt, 6),
            capital_usdt=round(capital_used, 4),
            sl_pct=round(sl_pct, 4),
            tp1_pct=round(tp1_pct, 4),
            tp2_pct=round(tp2_pct, 4),
        )

    def _round_qty(self, qty: float, price: float, symbol: str) -> float:
        try:
            info     = self.client.get_instrument_info(symbol)
            lot_step = float(info["lotSizeFilter"]["qtyStep"])
            min_qty  = float(info["lotSizeFilter"]["minOrderQty"])
            qty      = math.floor(qty / lot_step) * lot_step
            qty      = round(qty, 10)
            return qty if qty >= min_qty else 0.0
        except Exception as e:
            logger.warning(f"Instrument info error {symbol}: {e}")
            return round(qty, 3) if qty > 0 else 0.0

    def should_close_position(self, position: dict, current_price: float) -> bool:
        unrealised_pnl_pct = float(position.get("unrealisedPnl", 0)) / (
            float(position.get("positionValue", 1)) or 1
        )
        if unrealised_pnl_pct < -0.30:
            return True
        return False
