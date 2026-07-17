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

        # CAPITAL SCALE THEO DO TIEM NANG LENH:
        # Consensus (so strategies dong thuan) x Signal Strength (0.0-1.0)
        # Cang nhieu strategy dong thuan + strength cao = lenh cang tiem nang = capital lon hon
        #
        # Cong thuc:
        #   potential = (consensus/7) * 0.6 + strength * 0.4   (trong so: consensus quan trong hon)
        #   scale = 0.5 + potential * 2.0   -> range [0.5x, 2.5x]
        #     potential=0.0 (consensus=1,strength=0): scale=0.5x  (lenh yeu, bet nho)
        #     potential=0.5 (consensus=3-4,str~0.7): scale=1.5x  (lenh trung binh)
        #     potential=1.0 (consensus=7,strength=1): scale=2.5x (lenh manh nhat, all-in)
        consensus = getattr(signal, 'consensus', 1)
        strength  = getattr(signal, 'strength',  0.5)
        potential = (consensus / 7) * 0.6 + strength * 0.4
        potential = max(0.0, min(1.0, potential))
        scale_factor = 0.5 + potential * 2.0   # [0.5x, 2.5x]

        # Phi round-trip (tinh tren entry price de co trong sl/tp calc)
        fee_price = signal.entry_price * config.ROUND_TRIP_FEE

        # SL/TP — Swing-based voi ATR clamp
        # Uu tien dat SL tai swing high/low 15m (co y nghia cau truc hon ATR thuan tuy)
        # Clamp trong [1.5x, 3.0x] ATR: khong qua chat (noise hit) va khong qua rong (mat nhieu)
        # TP1/TP2 tu dong scale theo sl_dist de giu RR
        _atr = signal.atr
        _swing_sl = getattr(signal, 'swing_sl', 0.0)
        if _swing_sl > 0 and signal.entry_price > 0:
            _swing_dist = abs(signal.entry_price - _swing_sl)
            _sl_floor   = 1.5 * _atr  # toi thieu: khong chat hon 1.5x ATR
            _sl_cap     = 3.0 * _atr  # toi da:    khong rong hon 3.0x ATR
            sl_dist = max(_sl_floor, min(_sl_cap, _swing_dist)) + fee_price
            logger.debug(
                f"{signal.symbol}: swing SL dist={_swing_dist:.4f} "
                f"clamped [{_sl_floor:.4f}, {_sl_cap:.4f}] -> {sl_dist:.4f}"
            )
        else:
            sl_dist = config.SL_ATR_MULT * _atr + fee_price

        tp1_dist = config.TP1_ATR_MULT * _atr - fee_price
        tp2_dist = config.TP2_ATR_MULT * _atr - fee_price
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

        # Helper: round qty theo so chu so thap phan cua qty_step (tranh float artifact)
        _qty_decimals = len(str(qty_step).rstrip("0").split(".")[-1]) if "." in str(qty_step) else 0

        def _round_qty(q: float) -> float:
            return round(math.floor(q / qty_step) * qty_step, _qty_decimals)

        def _ceil_qty(q: float) -> float:
            return round(math.ceil(q / qty_step) * qty_step, _qty_decimals)

        # Round xuong de khong over-risk
        qty = _round_qty(qty_by_risk)

        # Neu risk-based qty < min_qty (exchange minimum), cap qty tai min_qty
        if qty < min_qty:
            qty = min_qty
            actual_risk = qty * sl_dist
            if actual_risk > risk_amount * 10:
                logger.warning(
                    f"{signal.symbol}: min_qty risk too high — "
                    f"actual_risk={actual_risk:.4f} > 10x intended={risk_amount:.4f} -> skip"
                )
                return None

        # Dam bao notional >= $5 (Bybit minimum)
        MIN_NOTIONAL = 5.0
        if qty * signal.entry_price < MIN_NOTIONAL:
            qty = _ceil_qty(MIN_NOTIONAL / signal.entry_price)

        notional = qty * signal.entry_price

        # Leverage: luon dung leverage cao nhat exchange cho phep (toi da MAX_LEVERAGE)
        # Margin thap nhat = notional / leverage_max -> von bo vao it nhat, giu room
        exchange_max_lev = self.client.get_max_leverage(signal.symbol) if config.USE_MAX_LEVERAGE \
                           else config.DEFAULT_LEVERAGE
        leverage = min(config.MAX_LEVERAGE, exchange_max_lev)
        leverage = max(leverage, 1)

        capital_used = notional / leverage

        # Check lai $5 minimum SAU capital adjustment — capital cap co the day notional xuong duoi $5
        # Neu van thieu $5: thu dung min qty de dat notional=$5, mien la margin can thiet <= equity
        # (voi max leverage, $5 notional chi can $5/lev margin — hoan toan kha thi voi tai khoan nho)
        if notional < MIN_NOTIONAL:
            min_qty_for_notional = _ceil_qty(MIN_NOTIONAL / signal.entry_price)
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
