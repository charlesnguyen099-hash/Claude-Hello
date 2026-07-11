"""
Risk Manager
- Position sizing: 10% von thuc moi lenh, khong all-in
- SL/TP tinh NET sau phi giao dich Bybit (0.055% taker moi chieu)
- Max loss 30% von moi lenh (dung leverage)
- Kiem tra min order Bybit truoc khi dat lenh
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
    capital_usdt: float     # von thuc bo ra (notional / leverage)


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

        sides  = [p["side"] for p in open_positions]
        n_long = sides.count("Buy")
        n_short = sides.count("Sell")
        if signal.direction == 1  and n_long  >= config.MAX_POSITIONS_PER_SIDE:
            return None
        if signal.direction == -1 and n_short >= config.MAX_POSITIONS_PER_SIDE:
            return None

        if signal.atr <= 0 or signal.entry_price <= 0:
            return None

        side = "Buy" if signal.direction == 1 else "Sell"

        # Lay leverage toi da cua tung cap tren Bybit
        if config.USE_MAX_LEVERAGE:
            leverage = self.client.get_max_leverage(signal.symbol)
        else:
            atr_pct  = signal.atr / signal.entry_price
            leverage = min(config.MAX_LEVERAGE, max(2, int(0.05 / atr_pct)))

        # Von thuc moi lenh = CAPITAL_PER_TRADE_PCT x equity (khong all-in)
        # Vi du: equity=100 USDT, 10% -> 10 USDT von thuc
        # Notional = von thuc x leverage -> 10 x 20 = 200 USDT notional
        capital_per_trade = equity * config.CAPITAL_PER_TRADE_PCT
        notional_from_cap = capital_per_trade * leverage

        # Gioi han tu risk 30%: neu SL hit -> thua toi da 30% equity
        # max_loss = equity x 30%
        # sl_dist_pct = SL_ATR x ATR / entry_price
        # max_notional = max_loss / sl_dist_pct
        sl_dist_pct       = config.SL_ATR_MULTIPLIER * signal.atr / signal.entry_price
        max_loss_usdt     = equity * config.ACCOUNT_RISK_PCT
        notional_from_risk = max_loss_usdt / sl_dist_pct if sl_dist_pct > 0 else notional_from_cap

        # Chon notional nho hon de an toan
        notional_target = min(notional_from_cap, notional_from_risk)
        qty_raw         = notional_target / signal.entry_price

        qty = self._round_qty(qty_raw, signal.entry_price, signal.symbol)
        if qty <= 0:
            logger.debug(f"{signal.symbol}: qty=0 sau lam tron")
            return None

        notional = qty * signal.entry_price

        # Kiem tra notional >= min order Bybit
        min_notional = self.client.get_min_order_usdt(signal.symbol)
        if notional < min_notional:
            try:
                info    = self.client.get_instrument_info(signal.symbol)
                min_qty = float(info["lotSizeFilter"]["minOrderQty"])
                qty     = min_qty
                notional = qty * signal.entry_price
            except Exception:
                pass

        capital_used = notional / leverage

        # Phi giao dich ca 2 chieu (vao + ra)
        fee_usdt  = notional * config.ROUND_TRIP_FEE
        fee_price = signal.entry_price * config.ROUND_TRIP_FEE  # phi tinh theo don vi gia

        # SL/TP tinh NET sau phi:
        # SL dat xa hon phi de loss thuc = ATR x multiplier
        # TP dat xa hon phi de profit thuc = ATR x multiplier
        d   = signal.direction
        sl  = signal.entry_price - d * (config.SL_ATR_MULTIPLIER  * signal.atr + fee_price)
        tp1 = signal.entry_price + d * (config.TP1_ATR_MULTIPLIER * signal.atr + fee_price)
        tp2 = signal.entry_price + d * (config.TP2_ATR_MULTIPLIER * signal.atr + fee_price)
        trail = config.TRAILING_STOP_ATR * signal.atr

        logger.info(
            f"{signal.symbol}: {side} | lev={leverage}x | "
            f"qty={qty} | notional={notional:.2f}$ | "
            f"capital={capital_used:.2f}$ ({capital_used/equity*100:.1f}% equity) | "
            f"fee~{fee_usdt:.4f}$ | SL={sl:.5f} | TP1={tp1:.5f} | TP2={tp2:.5f}"
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
        if unrealised_pnl_pct < -0.30:  # dong som neu thua qua 30% notional
            return True
        return False
