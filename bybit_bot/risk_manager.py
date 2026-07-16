"""
Risk Manager
- SL dong: khoang cach = ATR * SL_MULTIPLIER (config), tinh tu entry
- TP1 = entry +/- ATR * TP1_MULTIPLIER; TP2 = entry +/- ATR * TP2_MULTIPLIER
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
        is_priority: bool = False,
    ) -> Optional[TradeParams]:

        if signal.entry_price <= 0 or signal.atr <= 0 or equity <= 0:
            return None

        side = "Buy" if signal.direction == 1 else "Sell"

        try:
            info     = self.client.get_instrument_info(signal.symbol)
            min_qty  = float(info["lotSizeFilter"]["minOrderQty"])
            qty_step = float(info["lotSizeFilter"]["qtyStep"])
        except Exception as e:
            logger.warning(f"{signal.symbol}: cannot get instrument info: {e}")
            return None

        # Consensus scale: 1=0.5x, 2=0.7x, 3=1.0x, 4=1.3x, 5=1.6x, 6=2.0x, 7=2.5x
        # Scale nho hon: tranh over-size khi consensus cao nhung thi truong khong ro
        consensus = getattr(signal, 'consensus', 1)
        CONSENSUS_SCALE = {1: 0.5, 2: 0.7, 3: 1.0, 4: 1.3, 5: 1.6, 6: 2.0, 7: 2.5}
        scale_factor = CONSENSUS_SCALE.get(consensus, 1.0)

        # Phi round-trip (tinh tren entry price de co trong sl/tp calc)
        fee_price = signal.entry_price * config.ROUND_TRIP_FEE

        # SL/TP dua tren ATR — RR >= 1.3 sau phi
        sl_dist  = config.SL_ATR_MULT  * signal.atr + fee_price
        tp1_dist = config.TP1_ATR_MULT * signal.atr - fee_price   # 2.0x ATR - phi
        tp2_dist = config.TP2_ATR_MULT * signal.atr - fee_price   # 4.0x ATR - phi
        # Dam bao TP1 >= SL (RR >= 1)
        tp1_dist = max(tp1_dist, sl_dist)
        tp2_dist = max(tp2_dist, sl_dist * 2.0)
        # Dam bao SL/TP duong
        sl_dist  = max(sl_dist,  signal.entry_price * 0.002)
        tp1_dist = max(tp1_dist, signal.entry_price * 0.003)
        tp2_dist = max(tp2_dist, signal.entry_price * 0.006)

        # RISK-BASED POSITION SIZING:
        # Muc tieu: neu SL hit thi mat dung RISK_PER_TRADE_PCT% equity (x scale_factor)
        # qty = risk_amount / sl_dist
        risk_amount = equity * config.RISK_PER_TRADE_PCT * scale_factor
        qty_by_risk = risk_amount / sl_dist

        # Round xuong de khong over-risk
        qty = math.floor(qty_by_risk / qty_step) * qty_step
        # Phai >= min_qty cua exchange
        qty = max(qty, min_qty)
        # Dam bao notional >= $5 (Bybit minimum)
        MIN_NOTIONAL = 5.0
        if qty * signal.entry_price < MIN_NOTIONAL:
            qty = math.ceil(MIN_NOTIONAL / signal.entry_price / qty_step) * qty_step

        notional = qty * signal.entry_price

        # Leverage: dung leverage de chi can margin = MAX_CAPITAL_PCT * equity
        # Nhung cap tai MAX_LEVERAGE de tranh liquidation risk
        max_capital = equity * config.MAX_CAPITAL_PCT
        leverage = math.ceil(notional / max_capital)  # can bao nhieu leverage de margin <= 10% equity
        leverage = min(leverage, config.MAX_LEVERAGE)  # cap tai 20x
        leverage = max(leverage, 1)
        # Neu leverage theo exchange thap hon → dung leverage exchange
        exchange_max_lev = self.client.get_max_leverage(signal.symbol) if config.USE_MAX_LEVERAGE \
                           else config.DEFAULT_LEVERAGE
        leverage = min(leverage, exchange_max_lev)

        capital_used = notional / leverage
        fee_usdt     = notional * config.ROUND_TRIP_FEE

        d   = signal.direction
        sl  = signal.entry_price - d * sl_dist
        tp1 = signal.entry_price + d * tp1_dist
        tp2 = signal.entry_price + d * tp2_dist

        sl_pct  = sl_dist  / signal.entry_price * 100
        tp1_pct = tp1_dist / signal.entry_price * 100
        tp2_pct = tp2_dist / signal.entry_price * 100

        rr1 = tp1_pct / sl_pct if sl_pct > 0 else 0
        rr2 = tp2_pct / sl_pct if sl_pct > 0 else 0

        logger.info(
            f"{signal.symbol}: {side} lev={leverage}x | consensus={consensus}({scale_factor}x) | "
            f"qty={qty} | notional={notional:.2f}$ | capital={capital_used:.2f}$ | "
            f"fee={fee_usdt:.4f}$ | "
            f"SL=-{sl_pct:.2f}% | TP1=+{tp1_pct:.2f}% (RR={rr1:.2f}) | "
            f"TP2=+{tp2_pct:.2f}% (RR={rr2:.2f})"
        )

        return TradeParams(
            symbol=signal.symbol,
            side=side,
            qty=qty,
            leverage=leverage,
            sl_price=round(sl, 6),
            tp1_price=round(tp1, 6),
            tp2_price=round(tp2, 6),
            notional_usdt=round(notional, 2),
            fee_usdt=round(fee_usdt, 6),
            capital_usdt=round(capital_used, 4),
            sl_pct=round(sl_pct, 4),
            tp1_pct=round(tp1_pct, 4),
            tp2_pct=round(tp2_pct, 4),
        )

    def should_close_position(self, position: dict, current_price: float) -> bool:
        # So sanh PnL voi margin (von bo vao lenh), khong phai notional
        # positionValue = notional (qty x price), margin = notional / leverage
        notional = float(position.get("positionValue", 1)) or 1
        leverage = float(position.get("leverage", 1)) or 1
        margin   = notional / leverage
        unrealised_pnl = float(position.get("unrealisedPnl", 0))
        unrealised_pnl_pct = unrealised_pnl / margin if margin > 0 else 0
        if unrealised_pnl_pct < -0.30:
            return True
        return False
