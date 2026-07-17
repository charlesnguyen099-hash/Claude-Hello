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

        # SL/TP dua tren ATR
        # SL  = 1.5x ATR + phi (entry - sl_dist)
        # TP1 = 1.5x ATR - phi -> max(tp1_dist, sl_dist) lam cho TP1 = sl_dist -> RR ~1:1 sau phi
        # TP2 = 3.0x ATR - phi -> max(tp2_dist, sl_dist*2) -> TP2 = 2×SL (RR 2:1)
        # Partial close 50% tai 75% cua TP1 = 1.125x ATR, con lai chay den TP2
        sl_dist  = config.SL_ATR_MULT  * signal.atr + fee_price
        tp1_dist = config.TP1_ATR_MULT * signal.atr - fee_price
        tp2_dist = config.TP2_ATR_MULT * signal.atr - fee_price
        # TP1_ATR_MULT == SL_ATR_MULT nen tp1_dist < sl_dist (fee offset nguoc chieu)
        # -> max dam bao TP1 >= SL distance -> RR >= 1
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

        # Neu risk-based qty < min_qty (exchange minimum), cap qty tai min_qty
        # NHUNG: scale sl_dist down de giu risk_amount khong doi (tranh over-risk)
        # Neu khong the dieu chinh (min_qty * sl_dist > risk_amount * 3), thi bao log va skip
        if qty < min_qty:
            qty = min_qty
            actual_risk = qty * sl_dist
            if actual_risk > risk_amount * 5:
                logger.warning(
                    f"{signal.symbol}: min_qty risk too high — "
                    f"actual_risk={actual_risk:.4f} > 5x intended={risk_amount:.4f} -> skip"
                )
                return None

        # Dam bao notional >= $5 (Bybit minimum)
        MIN_NOTIONAL = 5.0
        if qty * signal.entry_price < MIN_NOTIONAL:
            qty = math.ceil(MIN_NOTIONAL / signal.entry_price / qty_step) * qty_step

        notional = qty * signal.entry_price

        # Leverage: tinh leverage can thiet de margin = MAX_CAPITAL_PCT * equity
        # Sau do cap vao min(MAX_LEVERAGE, exchange_max_lev)
        # Neu leverage bi cap thap hon muc can thiet -> capital_used tang > MAX_CAPITAL_PCT
        # -> giam qty de dam bao capital_used <= MAX_CAPITAL_PCT * equity
        exchange_max_lev = self.client.get_max_leverage(signal.symbol) if config.USE_MAX_LEVERAGE \
                           else config.DEFAULT_LEVERAGE
        max_capital = equity * config.MAX_CAPITAL_PCT
        leverage_needed = math.ceil(notional / max_capital)
        leverage = min(leverage_needed, config.MAX_LEVERAGE, exchange_max_lev)
        leverage = max(leverage, 1)

        # Neu leverage bi cap thap hon muc can (vi exchange gioi han) -> giam qty de giu capital cap
        capital_used = notional / leverage
        if capital_used > max_capital * 1.05:  # 5% tolerance
            # Giam qty sao cho capital_used <= max_capital
            max_notional  = max_capital * leverage
            qty = math.floor(max_notional / signal.entry_price / qty_step) * qty_step
            qty = max(qty, min_qty)
            notional = qty * signal.entry_price
            capital_used = notional / leverage

        # Check lai $5 minimum SAU capital adjustment — capital cap co the day notional xuong duoi $5
        # Neu van thieu $5: thu dung min qty de dat notional=$5, mien la margin can thiet <= equity
        # (voi max leverage, $5 notional chi can $5/lev margin — hoan toan kha thi voi tai khoan nho)
        if notional < MIN_NOTIONAL:
            min_qty_for_notional = math.ceil(MIN_NOTIONAL / signal.entry_price / qty_step) * qty_step
            min_margin_needed = (min_qty_for_notional * signal.entry_price) / leverage
            if min_margin_needed <= equity:
                qty = min_qty_for_notional
                notional = qty * signal.entry_price
                capital_used = notional / leverage
                logger.info(
                    f"{signal.symbol}: min-notional override — notional={notional:.2f}$ "
                    f"margin={capital_used:.4f}$ lev={leverage}x"
                )
            else:
                logger.warning(
                    f"{signal.symbol}: notional={notional:.2f}$ < $5, margin needed={min_margin_needed:.4f}$ "
                    f"> equity={equity:.2f}$ -> skip"
                )
                return None

        fee_usdt = notional * config.ROUND_TRIP_FEE

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
