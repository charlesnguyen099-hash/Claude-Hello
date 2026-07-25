"""
Risk Manager
- SL/TP theo ROI (% tren margin): TP scale theo potential [12%, 60%]
- SL = min(SL_TP_RATIO x TP, tran an toan thanh ly) - TP khong bi scale xuong
- Leverage chon cao nhat thoa: SL >= TP, SL trong vung an toan liq, TP loi rong >= 5% sau phi
- Von moi lenh theo potential: [5%, 25%] cua max(equity, EQUITY_FLOOR)
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional

import config
from client import BybitClient
from strategies.base import Signal

logger = logging.getLogger(__name__)

# Hang so vung an toan thanh ly - dung chung cho compute_trade va executor fallback
MAINT_RATE_EST = 0.005   # 0.5% maintenance margin (Bybit typical)
LIQ_BUFFER     = 0.10    # 10% safety buffer truoc gia thanh ly


def max_safe_sl_roi(leverage: int) -> float:
    """SL ROI cao nhat con AN TOAN tai leverage nay:
    - Nam tren gia thanh ly (kem buffer 10%) - SL ngoai liq = vo nghia, chay margin truoc
    - Tran cung 0.75: emergency close (-0.80) phai fire SAU exchange SL
    Moi noi tinh SL fallback PHAI dung ham nay - SL tinh tu ty le 5:1 tho co the
    vuot gia thanh ly o leverage cao va bi Bybit reject -> position khong co SL."""
    _L = max(int(leverage), 1)
    return max(0.20, min(0.75, (1.0 - MAINT_RATE_EST * _L) * (1.0 - LIQ_BUFFER)))


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

        # DO TIEM NANG LENH (potential [0,1]):
        # Consensus (so strategies dong thuan) x Signal Strength (0.0-1.0)
        #   potential = (consensus/7) * 0.6 + strength * 0.4   (trong so: consensus quan trong hon)
        # Dung cho CA HAI: TP ROI scale [12%, 60%] va VON scale [5%, 25%] equity
        consensus = getattr(signal, 'consensus', 1)
        strength  = getattr(signal, 'strength',  0.5)
        potential = (consensus / 7) * 0.6 + strength * 0.4
        potential = max(0.0, min(1.0, potential))

        # Lay leverage truoc de tinh SL/TP theo ROI
        exchange_max_lev = self.client.get_max_leverage(signal.symbol) if config.USE_MAX_LEVERAGE \
                           else config.DEFAULT_LEVERAGE
        leverage = min(config.MAX_LEVERAGE, exchange_max_lev)
        leverage = max(int(leverage), 1)   # int: dung cho range() trong fee+liq viability loop

        entry = signal.entry_price

        # SL/TP TINH THEO ROI (% tren margin), KHONG PHAI % GIA:
        #   ROI = (price_dist / entry) * leverage
        #   price_dist = ROI * entry / leverage
        #
        # TP ROI: scale theo potential [12%, 60%] - khong con tran 30%
        #   potential=0 -> TP ROI=12%, potential=1 -> TP ROI=60%
        # SL ROI muc tieu = SL_TP_RATIO x TP (5x), nhung bi clamp boi vung an toan thanh ly
        # trong vong chon leverage ben duoi - TP giu nguyen, chi SL bi gioi han
        _tp_roi_override = getattr(signal, 'tp_roi_override', 0.0)
        if _tp_roi_override > 0:
            tp_roi = _tp_roi_override
        else:
            tp_roi = config.TP_ROI_MIN + potential * (config.TP_ROI_MAX - config.TP_ROI_MIN)

        # CHON LEVERAGE - TP GIU NGUYEN GIA TRI DAY DU, chi SL bi clamp (khong bao gio skip):
        #   1. LIQ:  sl_roi = min(5*tp_roi, max_sl(L)) voi max_sl(L) = min(0.75, (1-0.005L)*0.9)
        #      SL phai nam TREN gia thanh ly. Tran cung 0.75 de emergency close (-0.80) luon
        #      fire SAU exchange SL - tranh conflict logic voi should_close_position.
        #   2. SL >= TP: neu clamp ep SL hep hon TP -> leverage qua cao, giam L xuong
        #      (lenh can khoang tho it nhat bang TP; L thap hon -> max_sl rong hon)
        #   3. FEE: tp_roi >= ROUND_TRIP_FEE*L + 5% -> TP hit luon LOI RONG >=5% margin sau phi
        # Duyet L giam dan -> lay L CAO NHAT thoa ca 3.
        # Khac ban cu: TP KHONG bi scale xuong theo liq clamp nua (bo tran TP 30% ->
        # TP 60% kha thi: L=66, SL clamp ~60%, ty le nen tu 5:1 ve ~1:1 cho lenh manh).
        # Luon ton tai L hop le: tai L=1 max_sl=0.75 >= tp (tp da clamp <=0.70), fee=0.11%.
        _min_net_roi = 0.01    # loi rong toi thieu 1% margin sau phi — "miễn không lỗ sau phí"

        tp_roi = min(tp_roi, 0.70)   # tran cung: dam bao SL >= TP ton tai o L=1 (max_sl=0.75)

        _chosen_lev = 0
        _sl_roi_eff = 0.0
        for _L in range(leverage, 0, -1):
            _max_sl = max_safe_sl_roi(_L)
            _sl_eff = min(tp_roi * config.SL_TP_RATIO, _max_sl)
            if _sl_eff < tp_roi:                     # SL hep hon TP -> can L thap hon
                continue
            # TP phai phu TAT CA phi (vao+ra+funding) tinh theo ROI (= cost_notional * L)
            # RONG hon net toi thieu -> TP hit la LOI RONG that su sau MOI loai phi.
            if tp_roi >= config.TOTAL_ROUND_TRIP_COST * _L + _min_net_roi:
                _chosen_lev = _L
                _sl_roi_eff = _sl_eff
                break

        if _chosen_lev < 1:
            logger.warning(f"{signal.symbol}: SKIP - TP_ROI={tp_roi*100:.0f}% khong tim duoc leverage hop le")
            return None

        if _chosen_lev != leverage:
            logger.info(
                f"{signal.symbol}: lev {leverage}x->{_chosen_lev}x (fee+liq viability) | "
                f"TP={tp_roi*100:.0f}% SL={_sl_roi_eff*100:.0f}% (SL/TP={_sl_roi_eff/tp_roi:.1f})"
            )
        leverage = _chosen_lev
        sl_roi   = _sl_roi_eff

        tp_dist = tp_roi * entry / leverage
        sl_dist = sl_roi * entry / leverage

        # TP THICH UNG VOLATILITY - TP phai NAM TRONG TAM VOI cua coin (khong dat qua xa range).
        # Loi 'ngu set TP': TP ROI co dinh -> khoang cach gia co the vuot bien do dao dong binh
        # thuong cua coin -> gia dao chieu TRUOC khi cham TP -> lo. Cap TP <= 2 x ATR gan nhat
        # (dat duoc trong vai nen), nhung van >= phi + buffer de con LOI sau phi.
        # Neu TP-bu-phi VUOT tam-voi (coin volatility qua thap) -> KHONG the lai sau phi -> SKIP.
        _atr = signal.atr if signal.atr > 0 else 0.0
        if _atr > 0:
            _tp_cap   = 2.0 * _atr                                 # tran: trong tam voi (~2 nen)
            # san TP (price dist): phi khu hoi + loi rong toi thieu (_min_net_roi) -> hit TP luon LOI RONG.
            # Phi (theo price dist): TOTAL_ROUND_TRIP_COST * entry (khong phu thuoc leverage vi phi = % notional)
            # Loi rong toi thieu (theo price dist): _min_net_roi * entry / leverage
            # -> TP floor chinh xac theo tung lenh (leverage cao -> floor nho hon theo % gia)
            _tp_floor = config.TOTAL_ROUND_TRIP_COST * entry + _min_net_roi * entry / leverage
            if _tp_floor > _tp_cap:
                logger.info(
                    f"{signal.symbol}: SKIP - volatility qua thap (ATR={_atr:.6f}), TP bu phi "
                    f"({_tp_floor:.6f}) vuot tam voi (2xATR={_tp_cap:.6f}) -> khong lai sau phi"
                )
                return None
            _tp_target = max(_tp_floor, min(tp_dist, _tp_cap))
            if abs(_tp_target - tp_dist) > 1e-12:
                tp_dist = _tp_target
                tp_roi  = tp_dist * leverage / entry               # dong bo ROI theo dist moi
                sl_roi  = min(tp_roi * config.SL_TP_RATIO, max_safe_sl_roi(leverage))
                sl_dist = sl_roi * entry / leverage

        logger.info(
            f"{signal.symbol}: lev={leverage}x | "
            f"TP_ROI={tp_roi*100:.0f}% SL_ROI={sl_roi*100:.0f}% (SL/TP={sl_roi/tp_roi:.1f}) | "
            f"tp_dist={tp_dist:.6f} sl_dist={sl_dist:.6f} atr={_atr:.6f}"
        )

        # Helper: round qty theo so chu so thap phan cua qty_step (tranh float artifact)
        _qty_decimals = len(str(qty_step).rstrip("0").split(".")[-1]) if "." in str(qty_step) else 0

        def _round_qty(q: float) -> float:
            return round(math.floor(q / qty_step) * qty_step, _qty_decimals)

        def _ceil_qty(q: float) -> float:
            return round(math.ceil(q / qty_step) * qty_step, _qty_decimals)

        # VON THEO GATE SCORE (dong) - TREN EQUITY THAT:
        #   gate_score (0-100) tu scenario gate -> map truc tiep sang % von:
        #     score=50 (threshold) -> 5% equity (lenh qua gate vua du)
        #     score=75             -> 47% equity
        #     score=90             -> 80% equity
        #     score=95+            -> 90-95% equity (gan all-in)
        #   Neu khong co gate_score (lenh khong qua gate) -> dung potential-based cu
        # So lenh KHONG bi chan cung - tu dieu tiet qua free margin:
        #   1. capital <= equity * MAX_CAPITAL_PCT (tran mem 95%)
        #   2. FREE MARGIN: capital <= free_margin * MAX_FREE_MARGIN_FRAC (95% free con lai)
        #   3. capital <= equity
        MIN_NOTIONAL = 5.0   # Bybit min order value

        # Margin dang bi chiem boi cac position dang mo (de tinh free margin)
        _used_margin = 0.0
        for _p in (open_positions or []):
            try:
                _pv  = float(_p.get("positionValue", 0) or 0)
                _plv = max(1.0, float(_p.get("leverage", 1) or 1))
                _used_margin += _pv / _plv
            except (TypeError, ValueError):
                continue
        _free_margin = max(0.0, equity - _used_margin)

        gate_score = getattr(signal, 'gate_score', 0.0)
        if gate_score >= 50:
            # Score-to-capital mapping truc tiep: score 50->5%, score 100->95%
            _score_frac = (gate_score - 50.0) / 50.0   # 0.0 at score=50, 1.0 at score=100
            _cap_pct = config.CAPITAL_PCT_MIN + _score_frac * (config.CAPITAL_PCT_MAX - config.CAPITAL_PCT_MIN)
        else:
            # Fallback: potential-based (lenh khong qua gate hoac gate_score chua set)
            _cap_pct = config.CAPITAL_PCT_MIN + potential * (config.CAPITAL_PCT_MAX - config.CAPITAL_PCT_MIN)

        _cap_target = equity * _cap_pct
        _cap_target = min(_cap_target, equity * config.MAX_CAPITAL_PCT)          # (1) tran mem 90%
        _cap_target = min(_cap_target, _free_margin * config.MAX_FREE_MARGIN_FRAC)  # (2) chua free
        _cap_target = min(_cap_target, equity)                                    # (3) tran tong von

        _notional_target = _cap_target * leverage

        qty = _round_qty(_notional_target / entry)
        if qty < min_qty:
            qty = min_qty

        # Bybit min notional $5: nang qty len neu can
        if qty * entry < MIN_NOTIONAL:
            qty = max(_ceil_qty(MIN_NOTIONAL / entry), min_qty)

        notional     = qty * entry
        capital_used = notional / leverage

        # Neu min-notional/min-qty ep margin vuot free margin -> khong con cho, skip
        # (giu von cho lenh khac thay vi don het vao lenh min-size nay)
        if capital_used > equity:
            logger.warning(
                f"{signal.symbol}: margin {capital_used:.2f}$ > equity {equity:.2f}$ "
                f"(min_qty/min_notional qua lon cho equity) -> skip"
            )
            return None
        if capital_used > _free_margin and _free_margin > 0 and _used_margin > 0:
            logger.info(
                f"{signal.symbol}: margin {capital_used:.2f}$ > free {_free_margin:.2f}$ "
                f"(da co {_used_margin:.2f}$ dang trade) -> skip, giu von cho lenh dang mo"
            )
            return None

        fee_usdt = notional * config.TOTAL_ROUND_TRIP_COST   # vao+ra+funding buffer

        d  = signal.direction
        sl = entry - d * sl_dist
        tp = entry + d * tp_dist

        logger.info(
            f"{signal.symbol}: {side} lev={leverage}x | consensus={consensus} str={strength:.2f} "
            f"potential={potential:.2f} | cap_pct={_cap_pct*100:.0f}% free={_free_margin:.1f}$ | "
            f"qty={qty} | notional={notional:.2f}$ | capital={capital_used:.2f}$ | "
            f"TP_ROI=+{tp_roi*100:.0f}% SL_ROI=-{sl_roi*100:.0f}% (SL/TP={sl_roi/tp_roi:.1f})"
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
        #   - SL luon <= 75% ROI (max_safe_sl_roi tran 0.75): exchange SL fire TRUOC emergency
        #   - Emergency chi la lop bao ve cuoi khi SL exchange bi mat/khong trigger
        # Truoc day -0.30 fire TRUOC exchange SL -> force-close qua som,
        # cat lenh o -30% du price se phuc hoi - nguyen nhan mat lenh loi.
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
