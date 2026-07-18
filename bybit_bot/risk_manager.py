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
    notional_usdt: float
    fee_usdt: float
    capital_usdt: float
    sl_pct: float
    tp1_pct: float


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
        #   scale = (0.5 + potential * 2.0) * 3.0   -> range [1.5x, 7.5x]
        #     potential=0.0 (consensus=1,strength=0): scale=1.5x  (lenh yeu)
        #     potential=0.5 (consensus=3-4,str~0.7): scale=4.5x  (lenh trung binh)
        #     potential=1.0 (consensus=7,strength=1): scale=7.5x (lenh manh nhat)
        consensus = getattr(signal, 'consensus', 1)
        strength  = getattr(signal, 'strength',  0.5)
        potential = (consensus / 7) * 0.6 + strength * 0.4
        potential = max(0.0, min(1.0, potential))
        scale_factor = (0.5 + potential * 2.0) * 3.0   # [1.5x, 7.5x] — 3x capital boost

        # Lay leverage truoc de tinh SL/TP theo ROI
        exchange_max_lev = self.client.get_max_leverage(signal.symbol) if config.USE_MAX_LEVERAGE \
                           else config.DEFAULT_LEVERAGE
        leverage = min(config.MAX_LEVERAGE, exchange_max_lev)
        leverage = max(leverage, 1)

        entry = signal.entry_price

        # SL/TP TINH THEO ROI (% tren margin), KHONG PHAI % GIA:
        #   ROI = (price_dist / entry) * leverage
        #   price_dist = ROI * entry / leverage
        #
        # TP ROI: scale theo potential [12%, 50%]
        #   potential=0 -> TP ROI=12%, potential=1 -> TP ROI=50%
        # SL ROI = SL_TP_RATIO x TP ROI (hien tai 5x)
        #   -> SL ROI range: [60%, 250%]
        #   -> SL toi thieu 60% ROI (khi TP=12%), SL toi da 250% ROI (khi TP=50%)
        tp_roi  = config.TP_ROI_MIN + potential * (config.TP_ROI_MAX - config.TP_ROI_MIN)
        sl_roi  = tp_roi * config.SL_TP_RATIO   # SL = SL_TP_RATIO x TP

        # Clamp SL/TP de dam bao SL luon nam TREN gia thanh ly (liquidation price)
        # Voi leverage cao (50-100x), sl_dist = 5*tp_dist co the xuong duoi liq price
        # → Bybit tu dong cap SL lai → pha ty le 5:1
        #
        # liq_price (Long) ≈ entry * (1 - 1/L + maint_rate)
        # sl phai > liq_price → sl_roi < (1 - maint_rate * L)
        # Dung 0.5% maint rate (pho bien tren Bybit) + 10% buffer de tranh bi clamp
        MAINT_RATE_EST = 0.005   # 0.5% maintenance margin (Bybit typical)
        LIQ_BUFFER     = 0.10    # 10% safety buffer
        max_sl_roi = max(0.20, (1.0 - MAINT_RATE_EST * leverage) * (1.0 - LIQ_BUFFER))
        if sl_roi > max_sl_roi:
            # Scale ca TP lan SL xuong de GIU TY LE 5:1 va SL khong bi Bybit clamp
            liq_scale = max_sl_roi / sl_roi
            tp_roi    = tp_roi * liq_scale
            sl_roi    = max_sl_roi   # = tp_roi * SL_TP_RATIO (ty le van la 5:1)
            logger.info(
                f"{signal.symbol}: SL/TP scaled to fit liq constraint at {leverage}x "
                f"→ TP_ROI={tp_roi*100:.0f}% SL_ROI={sl_roi*100:.0f}% (ratio={config.SL_TP_RATIO:.0f}:1 maintained)"
            )

        tp_dist = tp_roi * entry / leverage
        sl_dist = sl_roi * entry / leverage

        logger.info(
            f"{signal.symbol}: lev={leverage}x | "
            f"TP_ROI={tp_roi*100:.0f}% SL_ROI={sl_roi*100:.0f}% (={sl_roi/tp_roi:.0f}xTP) | "
            f"tp_dist={tp_dist:.6f} sl_dist={sl_dist:.6f}"
        )

        # RISK-BASED POSITION SIZING:
        # Muc tieu: neu SL hit thi mat dung RISK_PER_TRADE_PCT% equity (x scale_factor)
        # qty = risk_amount / sl_dist
        risk_amount = equity * config.RISK_PER_TRADE_PCT * scale_factor
        qty_by_risk = risk_amount / (sl_dist + 1e-12)

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

        # leverage da tinh o tren (dung lai, khong goi API lan 2)
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

        d  = signal.direction
        sl = entry - d * sl_dist
        tp = entry + d * tp_dist

        logger.info(
            f"{signal.symbol}: {side} lev={leverage}x | consensus={consensus}({scale_factor:.1f}x) | "
            f"qty={qty} | notional={notional:.2f}$ | capital={capital_used:.2f}$ | "
            f"TP_ROI=+{tp_roi*100:.0f}% SL_ROI=-{sl_roi*100:.0f}% (SL={config.SL_TP_RATIO:.0f}xTP)"
        )

        return TradeParams(
            symbol=signal.symbol,
            side=side,
            qty=qty,
            leverage=leverage,
            sl_price=round(sl, 6),
            tp1_price=round(tp, 6),
            notional_usdt=round(notional, 2),
            fee_usdt=round(fee_usdt, 6),
            capital_usdt=round(capital_used, 4),
            sl_pct=round(sl_dist / entry * 100, 4),
            tp1_pct=round(tp_dist / entry * 100, 4),
        )

    def should_close_position(self, position: dict, current_price: float) -> bool:
        # Emergency close: chi trigger khi SL exchange KHONG hoat dong (SL bi missed/huy)
        # Nguong -0.80 (80% margin loss) dam bao:
        #   - SL min = 60% ROI: exchange close TRUOC emergency (60% < 80%) — khong can thiep
        #   - SL max = 250% ROI: emergency close truoc de tranh liquidation
        # Truoc day -0.30 fire TRUOC exchange SL (30% < 60% min SL) -> force-close qua som,
        # cat lenh o -30% du price se phuc hoi — nguyen nhan mat lenh loi.
        def _f(d, k, default=0.0):
            v = d.get(k, default)
            try:
                return float(v) if v != "" else default
            except (TypeError, ValueError):
                return default
        notional = _f(position, "positionValue", 1) or 1
        leverage = _f(position, "leverage", 1) or 1
        margin   = notional / leverage
        unrealised_pnl = _f(position, "unrealisedPnl")
        unrealised_pnl_pct = unrealised_pnl / margin if margin > 0 else 0
        if unrealised_pnl_pct < -0.80:
            return True
        return False
