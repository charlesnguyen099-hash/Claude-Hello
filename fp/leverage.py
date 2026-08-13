"""How much leverage a trade can carry, solved rather than chosen.

A leveraged position does not just multiply the move; it multiplies the
funding, the fee and -- because a perpetual is rebalanced continuously --
a volatility drag that grows with the SQUARE of leverage. So the answer
is not "as much as the exchange allows". It is the L that maximises

    move x L  -  costs x L  -  drag x L^2

and for a thin move over a long hold that maximum is at L = 0, meaning
the trade should not be taken at all. On this repo's own data a
perfectly-called short returned +14.1% held once and -274.6% at 10x
rebalanced. Drag is not a rounding error.

Extracted from what used to be a trend-following study so the live bot
can size a position without importing a strategy it no longer trades.
"""
from __future__ import annotations

import numpy as np

# A perpetual rebalanced continuously cannot carry much leverage over a
# multi-hour hold before drag eats the move. These bounds are the search
# range, not a target.
LEVERAGE_MIN, LEVERAGE_MAX = 1.0, 3.0

FUNDING_PER_DAY = 0.0003          # 0.010% per 8h, charged as a cost
ROUND_TRIP_FEE = 0.0011           # taker in, taker out


def drag_per_day(daily_vol: float, leverage: float) -> float:
    """Volatility drag: (L*sigma)^2 / 2 per period.

    Quadratic in leverage. At 2.28% daily vol this is 0.026%/day at 1x,
    0.104% at 2x and 0.234% at 3x -- 3x costs nine times what 1x does for
    three times the exposure.
    """
    return 0.5 * (leverage * daily_vol) ** 2


def expected_net(move_pct: float, hold_days: float, daily_vol: float,
                 leverage: float) -> float:
    """Net return on margin for a position held without rebalancing."""
    gross = move_pct * leverage
    costs = leverage * (ROUND_TRIP_FEE + hold_days * FUNDING_PER_DAY)
    return gross - costs - hold_days * drag_per_day(daily_vol, leverage)


def best_leverage(move_pct: float, hold_days: float, daily_vol: float,
                  cap: float = LEVERAGE_MAX) -> dict:
    """The leverage that maximises expected net -- the potential scaling,
    now with drag in it.

    Return grows linearly in leverage and drag grows quadratically, so
    there is a maximum rather than "more is better". A big expected move
    over a short hold justifies more; a thin move over a long hold
    justifies less, and often none at all.
    """
    if daily_vol <= 0 or not np.isfinite(daily_vol):
        return {"leverage": 0.0, "net": 0.0, "tradeable": False}
    grid = np.linspace(0.25, cap, 56)
    nets = np.array([expected_net(move_pct, hold_days, daily_vol, L)
                     for L in grid])
    i = int(np.argmax(nets))
    lev = float(grid[i]) if nets[i] > 0 else 0.0
    return {"leverage": max(LEVERAGE_MIN, lev) if lev > 0 else 0.0,
            "net": float(nets[i]), "tradeable": nets[i] > 0,
            "drag_per_day": drag_per_day(daily_vol, max(lev, LEVERAGE_MIN))}




# How much of the margin is gone by the time Bybit liquidates, and the
# headroom kept on top so a stop is never a coin flip against it.
LIQ_MARGIN_FRACTION = 0.90
SOLVENCY_BUFFER = 1.30


def solvent_leverage(tp_dist: float, sl_dist: float, hold_min: float,
                     sigma: float, cap: float = LEVERAGE_MAX) -> dict:
    """best_leverage, but the stop is guaranteed to fire BEFORE liquidation.

    Liquidation costs the entire margin; a stop costs the fraction the
    trade was sized for. best_leverage solves for growth and knows nothing
    about that, so on a wide stop it returned the full 3x with the stop
    sitting four times PAST the liquidation price -- measured, on a
    (4 sigma, 4 sigma, 720m) shape at high volatility.

    The cap is arithmetic, not preference:

        margin x L x sl_dist  <  LIQ_MARGIN_FRACTION x margin

    with SOLVENCY_BUFFER of headroom. Below 1x the stop does not fit at
    all, and the setup is refused rather than quietly resized into one
    that liquidates.
    """
    lev_cap = LIQ_MARGIN_FRACTION / max(sl_dist * SOLVENCY_BUFFER, 1e-12)
    if lev_cap < 1.0:
        return {"leverage": 0.0, "tradeable": False, "reason": "stop past "
                "liquidation even at 1x"}
    out = best_leverage(tp_dist, hold_min / 1440.0, sigma,
                        min(cap, lev_cap))
    out.setdefault("reason", "")
    return out
