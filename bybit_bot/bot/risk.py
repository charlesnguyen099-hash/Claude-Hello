"""Position sizing, leverage and safety limits.

No signal is ever treated as "guaranteed" — confidence tiers below cap
the fraction of equity risked even for the strongest setup. Defaults are
intentionally conservative; see README for how to change them and why
that's risky.
"""
from __future__ import annotations

from dataclasses import dataclass


# Confidence (0-100) -> (equity_pct_to_risk, max_leverage)
# equity_pct_to_risk is the fraction of *equity* allowed to be lost if the
# stop-loss is hit (not the notional size of the position).
TIERS = [
    (80.0, 0.20, 20),   # very high confidence
    (65.0, 0.12, 15),   # high
    (50.0, 0.06, 10),   # medium
]
MIN_CONFIDENCE_TO_TRADE = 50.0

# Hard safety ceilings regardless of tier/config above.
ABSOLUTE_MAX_LEVERAGE = 25
ABSOLUTE_MAX_EQUITY_RISK_PCT = 0.25
MAX_CONCURRENT_POSITIONS = 4
DAILY_LOSS_LIMIT_PCT = 0.15  # circuit breaker: stop opening new trades


@dataclass(frozen=True)
class PositionPlan:
    equity_risk_pct: float
    leverage: int
    qty: float
    notional: float
    margin_used: float


def tier_for_confidence(confidence: float) -> tuple[float, int] | None:
    if confidence < MIN_CONFIDENCE_TO_TRADE:
        return None
    for threshold, equity_pct, lev in TIERS:
        if confidence >= threshold:
            return equity_pct, lev
    return None


def dynamic_leverage_cap(atr_pct: float, tier_leverage: int) -> int:
    """Volatility-adjusted leverage: even a "very high confidence" tier
    gets its leverage cut down hard when the symbol's own volatility
    (ATR as % of price) is high, since higher leverage + higher volatility
    is what causes forced liquidation on a single wick.
    """
    if atr_pct >= 0.015:
        cap = 5
    elif atr_pct >= 0.008:
        cap = 10
    elif atr_pct >= 0.004:
        cap = 15
    else:
        cap = ABSOLUTE_MAX_LEVERAGE
    return max(1, min(tier_leverage, cap, ABSOLUTE_MAX_LEVERAGE))


def plan_position(
    equity: float,
    entry: float,
    stop: float,
    confidence: float,
    atr_pct: float,
) -> PositionPlan | None:
    tier = tier_for_confidence(confidence)
    if tier is None or equity <= 0:
        return None
    equity_pct, tier_leverage = tier
    equity_pct = min(equity_pct, ABSOLUTE_MAX_EQUITY_RISK_PCT)

    stop_distance = abs(entry - stop)
    if stop_distance <= 0:
        return None
    stop_pct_of_entry = stop_distance / entry

    leverage = dynamic_leverage_cap(atr_pct, tier_leverage)
    if stop_pct_of_entry > 0 and (1.0 / stop_pct_of_entry) < leverage:
        # stop distance narrower than 1/leverage means normal noise could
        # brush the liquidation price before the stop even fills; derate.
        leverage = max(1, int(1.0 / stop_pct_of_entry * 0.5))

    equity_to_risk = equity * equity_pct
    # qty such that (qty * stop_distance) == equity_to_risk  -> risk-based sizing
    qty = equity_to_risk / stop_distance
    notional = qty * entry
    margin_required = notional / leverage

    # Never let margin for a single trade exceed the equity fraction cap
    # even if a huge stop_distance would imply a bigger notional, and never
    # after the leverage derating above either — this check runs last using
    # the FINAL leverage so the cap always holds.
    max_margin = equity * equity_pct
    if margin_required > max_margin:
        scale = max_margin / margin_required
        qty *= scale
        notional *= scale
        margin_required = max_margin

    return PositionPlan(
        equity_risk_pct=equity_pct,
        leverage=leverage,
        qty=qty,
        notional=notional,
        margin_used=margin_required,
    )
