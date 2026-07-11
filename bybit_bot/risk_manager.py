"""
Risk Manager
- SL co dinh: dat o muc mat toi da SL_MAX_LOSS_PCT (30%) capital bo vao lenh
- TP dong: tinh theo Risk:Reward ratio (1.5x va 3x SL distance) sau phi
- Von moi lenh: CAPITAL_PER_TRADE_PCT (10%) equity, khong all-in
- Phi giao dich 0.11% round-trip tich hop vao SL/TP
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
    sl_pct: float        # SL distance % so voi entry
    tp1_pct: float       # TP1 distance %
    tp2_pct: float       # TP2 distance %


class RiskManager:
    def __init__(self, client: BybitClient):
        self.client = client

    def compute_trade(
        self,
        signal: Signal,
        equity: float,
        open_positions: list[dict],
    ) -> Optional[TradeParams]:

        # Kiem tra gioi han vi the
        if len(open_positions) >= config.MAX_OPEN_POSITIONS:
            return None

        sides   = [p["side"] for p in open_positions]
        n_long  = sides.count("Buy")
        n_short = sides.count("Sell")
        if signal.direction == 1  and n_long  >= config.MAX_POSITIONS_PER_SIDE:
            return None
        if signal.direction == -1 and n_short >= config.MAX_POSITIONS_PER_SIDE:
            return None

        if signal.entry_price <= 0:
            return None

        side = "Buy" if signal.direction == 1 else "Sell"

        # Lay leverage toi da cua cap nay tren Bybit
        if config.USE_MAX_LEVERAGE:
            leverage = self.client.get_max_leverage(signal.symbol)
        else:
            leverage = config.DEFAULT_LEVERAGE

        # Von thuc bo vao lenh = 10% equity
        capital = equity * config.CAPITAL_PER_TRADE_PCT

        # Notional = capital x leverage
        notional_target = capital * leverage
        qty_raw         = notional_target / signal.entry_price

        qty = self._round_qty(qty_raw, signal.entry_price, signal.symbol)
        if qty <= 0:
            logger.debug(f"{signal.symbol}: qty=0")
            return None

        notional     = qty * signal.entry_price
        capital_used = notional / leverage

        # Kiem tra min order Bybit
        min_notional = self.client.get_min_order_usdt(signal.symbol)
        if notional < min_notional:
            try:
                info    = self.client.get_instrument_info(signal.symbol)
                min_qty = float(info["lotSizeFilter"]["minOrderQty"])
                qty     = min_qty
                notional     = qty * signal.entry_price
                capital_used = notional / leverage
            except Exception:
                pass

        # Phi round-trip tinh theo don vi gia
        fee_usdt  = notional * config.ROUND_TRIP_FEE
        fee_price = signal.entry_price * config.ROUND_TRIP_FEE

        # SL co dinh: mat toi da SL_MAX_LOSS_PCT (30%) capital bo vao lenh
        # loss_usdt = capital x 30%
        # sl_dist   = loss_usdt / qty
        max_loss_usdt = capital_used * config.SL_MAX_LOSS_PCT
        sl_dist       = max_loss_usdt / qty if qty > 0 else signal.atr * 1.5

        # Dam bao SL khong qua gian (toi thieu = phi)
        sl_dist = max(sl_dist, fee_price * 2)

        # TP tinh theo Risk:Reward (TP = SL x RR ratio) + phi
        # TP1 = 1.5x SL distance, TP2 = 3.0x SL distance
        tp1_dist = sl_dist * config.TP1_RR
        tp2_dist = sl_dist * config.TP2_RR
        trail    = sl_dist * 0.5  # trailing stop = 50% SL distance

        d   = signal.direction
        sl  = signal.entry_price - d * (sl_dist  + fee_price)
        tp1 = signal.entry_price + d * (tp1_dist + fee_price)
        tp2 = signal.entry_price + d * (tp2_dist + fee_price)

        sl_pct  = sl_dist  / signal.entry_price * 100
        tp1_pct = tp1_dist / signal.entry_price * 100
        tp2_pct = tp2_dist / signal.entry_price * 100

        logger.info(
            f"{signal.symbol}: {side} lev={leverage}x | "
            f"qty={qty} | notional={notional:.2f}$ | "
            f"capital={capital_used:.2f}$ ({capital_used/equity*100:.1f}% eq) | "
            f"fee={fee_usdt:.4f}$ | "
            f"SL={sl:.5f}(-{sl_pct:.2f}%) | "
            f"TP1={tp1:.5f}(+{tp1_pct:.2f}%) | "
            f"TP2={tp2:.5f}(+{tp2_pct:.2f}%)"
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
