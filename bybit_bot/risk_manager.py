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
        _tp_roi_override = getattr(signal, 'tp_roi_override', 0.0)
        if _tp_roi_override > 0:
            tp_roi = _tp_roi_override
        else:
            tp_roi = config.TP_ROI_MIN + potential * (config.TP_ROI_MAX - config.TP_ROI_MIN)
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

        # Fee break-even check: TP phai LON HON phi giao dich + buffer toi thieu
        # Phi round-trip tinh theo % margin = ROUND_TRIP_FEE * leverage
        # Vi du 100x: phi = 0.11% * 100 = 11% margin. TP_ROI < 11% = lo dam bao du TP hit chinh xac
        _fee_as_roi    = config.ROUND_TRIP_FEE * leverage   # phi tinh theo % margin
        _min_net_roi   = 0.05                               # buffer toi thieu 5% margin sau phi
        _min_tp_needed = _fee_as_roi + _min_net_roi
        if tp_roi < _min_tp_needed:
            logger.warning(
                f"{signal.symbol}: SKIP — TP_ROI={tp_roi*100:.0f}% < fee_breakeven "
                f"(fee={_fee_as_roi*100:.0f}% + buffer=5% = {_min_tp_needed*100:.0f}%) at {leverage}x "
                f"→ guaranteed loss even on TP hit"
            )
            return None

        tp_dist = tp_roi * entry / leverage
        sl_dist = sl_roi * entry / leverage

        logger.info(
            f"{signal.symbol}: lev={leverage}x | "
            f"TP_ROI={tp_roi*100:.0f}% SL_ROI={sl_roi*100:.0f}% (={sl_roi/tp_roi:.0f}xTP) | "
            f"tp_dist={tp_dist:.6f} sl_dist={sl_dist:.6f}"
        )

        # Helper: round qty theo so chu so thap phan cua qty_step (tranh float artifact)
        _qty_decimals = len(str(qty_step).rstrip("0").split(".")[-1]) if "." in str(qty_step) else 0

        def _round_qty(q: float) -> float:
            return round(math.floor(q / qty_step) * qty_step, _qty_decimals)

        def _ceil_qty(q: float) -> float:
            return round(math.ceil(q / qty_step) * qty_step, _qty_decimals)

        # MIN-QTY BASED POSITION SIZING:
        # Qty = min_qty * base_mult * consensus_boost
        # base_mult: 10x-15x min_qty (theo consensus), consensus_boost: 1.0-2.0x (theo strength)
        # Neu equity khong du → ha dan multiplier xuong cho den khi vua von
        # Muc dich: size lon hon, bat duoc profit on dinh, khong phu thuoc % equity (qua nho)
        #
        #   base_mult: consensus=1 → 10x, consensus=7 → 15x (scale tuyen tinh)
        #   consensus_boost: strength=0 → 1.0x, strength=1 → 2.0x
        #   final_mult = base_mult * consensus_boost → range [10x, 30x]
        base_mult = 30 + int((consensus / 7) * 20)  # 30 → 50 theo consensus
        base_mult = max(30, min(50, base_mult))
        consensus_boost = 1.0 + strength             # 1.0 → 2.0 theo strength
        final_mult = base_mult * consensus_boost     # 30x → 100x

        MIN_NOTIONAL = 5.0

        # Thu lan luot tu final_mult xuong den 1x (min_qty), chon mult vua equity
        qty = 0.0
        _used_mult = 0.0
        for _try_mult in [final_mult, final_mult * 0.7, final_mult * 0.5,
                          base_mult, 30.0, 20.0, 15.0, 10.0, 5.0, 3.0, 1.5, 1.0]:
            _q = _round_qty(min_qty * _try_mult)
            if _q < min_qty:
                _q = min_qty
            _notional_try = _q * signal.entry_price
            _cap_try      = _notional_try / leverage
            if _cap_try <= equity and _notional_try >= MIN_NOTIONAL:
                qty = _q
                _used_mult = _try_mult
                break

        if qty <= 0:
            # Last resort: min_qty neu notional >= $5, margin <= equity
            _q = _ceil_qty(MIN_NOTIONAL / signal.entry_price)
            _q = max(_q, min_qty)
            if _q * signal.entry_price / leverage <= equity:
                qty = _q
                _used_mult = qty / min_qty
            else:
                logger.warning(f"{signal.symbol}: even min_qty notional exceeds equity={equity:.2f}$ -> skip")
                return None

        notional    = qty * signal.entry_price
        capital_used = notional / leverage

        fee_usdt = notional * config.ROUND_TRIP_FEE

        d  = signal.direction
        sl = entry - d * sl_dist
        tp = entry + d * tp_dist

        logger.info(
            f"{signal.symbol}: {side} lev={leverage}x | consensus={consensus} str={strength:.2f} | "
            f"mult={_used_mult:.1f}x({base_mult}base×{consensus_boost:.1f}boost) | "
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
