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
        # TP ROI: scale theo potential [20%, 50%]
        #   potential=0 -> TP ROI=20%, potential=1 -> TP ROI=50%
        # SL ROI = 3 x TP ROI (luon gap 3 lan TP)
        #   -> SL ROI range: [60%, 150%]
        #   -> SL toi thieu 60% ROI (khi TP=20%), SL toi da 150% ROI (khi TP=50%)
        tp_roi  = config.TP_ROI_MIN + potential * (config.TP_ROI_MAX - config.TP_ROI_MIN)
        sl_roi  = tp_roi * config.SL_TP_RATIO   # SL = SL_TP_RATIO x TP

        # CLAMP sl_roi: SL price phai o tren gia liquidation
        # Liq price (Long) ≈ entry * (1 - 1/leverage) → sl_roi = 1.0 (100% ROI) = liq price
        # Bybit tu choi bat ky SL nao tai hoac duoi liq price → KHONG BAO GIO set duoc SL
        # Clamp sl_roi tai MAX_SL_ROI (80%) → SL luon cach liq it nhat 20% khoang cach entry-to-liq
        MAX_SL_ROI = 0.80
        if sl_roi > MAX_SL_ROI:
            logger.debug(f"{signal.symbol}: sl_roi={sl_roi*100:.0f}% clamped to {MAX_SL_ROI*100:.0f}% (tranh SL duoi liq price)")
            sl_roi = MAX_SL_ROI

        tp1_dist = tp_roi * entry / leverage
        tp2_dist = tp_roi * config.TP2_SCALE * entry / leverage  # TP2 = TP1 * TP2_SCALE
        sl_dist  = sl_roi * entry / leverage

        fee_price = entry * config.ROUND_TRIP_FEE

        logger.info(
            f"{signal.symbol}: lev={leverage}x | "
            f"TP_ROI={tp_roi*100:.0f}% SL_ROI={sl_roi*100:.0f}% (cap 80% ROI) | "
            f"tp1_dist={tp1_dist:.6f} sl_dist={sl_dist:.6f}"
        )

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

        d   = signal.direction
        sl  = entry - d * sl_dist
        tp1 = entry + d * tp1_dist
        tp2 = entry + d * tp2_dist

        sl_roi_pct  = sl_roi  * 100
        tp1_roi_pct = tp_roi  * 100
        tp2_roi_pct = tp_roi * config.TP2_SCALE * 100

        logger.info(
            f"{signal.symbol}: {side} lev={leverage}x | consensus={consensus}({scale_factor}x) | "
            f"qty={qty} | notional={notional:.2f}$ | capital={capital_used:.2f}$ | "
            f"fee={fee_usdt:.4f}$ | "
            f"TP_ROI=+{tp1_roi_pct:.0f}% | SL_ROI=-{sl_roi_pct:.0f}% (SL={config.SL_TP_RATIO:.0f}xTP) | "
            f"TP2_ROI=+{tp2_roi_pct:.0f}%"
        )

        sl_pct  = sl_dist  / entry * 100
        tp1_pct = tp1_dist / entry * 100
        tp2_pct = tp2_dist / entry * 100

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
        if unrealised_pnl_pct < -0.30:
            return True
        return False
