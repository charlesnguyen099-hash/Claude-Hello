"""FINAL_Logic_PotentialScaledLeverage — leverage scaled by trade potential.

22,533 rows, every one profitable (final_net +0.000% to +8.17%, no
negative row). What this file adds over the previous one is the leverage
chain, and all of it was recovered from the data rather than guessed:

    potential_score        0.035 .. 0.990, median 0.498
    potential_multiplier   = potential_score + 0.500      (exact, resid 5e-4)
    lev_base               17 .. 100, median 85.5
    leverage_potential     = min(lev_base x multiplier, lev_base)   96.4%
    net_leveraged_potential = final_net x leverage_potential        100.0%

WHAT potential_score ACTUALLY IS

It correlates +0.951 with the percentile rank of atr14_pct. It is a
volatility ranking: the more the instrument is moving, the higher the
"potential" of the setup.

That matters for how much weight to put on it. potential_score correlates
+0.739 with final_net, which looks like it predicts the payoff — but
atr14_pct alone correlates +0.891 with the same column, higher. The
relationship is mechanical, not predictive: the exits are ATR multiples,
so a high-ATR setup that reaches TP3.0 necessarily books a larger percent
than a low-ATR one. The score tells you how big a winner would be, not
how likely the trade is to win. The file cannot show the difference
because it contains only winners.

THE TWO ATR EFFECTS PULL OPPOSITE WAYS

lev_base falls as volatility rises (correlation -0.953 with atr14_pct):
a wider stop needs less leverage to keep it inside the margin. The
potential multiplier rises with volatility. And because the product is
capped at lev_base, the multiplier can only ever cut leverage, never add
it. So in practice: volatile setups get the full (already low) base, calm
setups get a fraction of their (high) base.

WHAT A BOT CAN AND CANNOT TAKE FROM THE FILE

final_direction is not the methods' verdict. On the 19,717 rows where the
twelve methods reached a consensus, it follows that consensus 50.4% of
the time and reverses it 49.6% — a coin flip, so the column carries no
information the methods produced. It is whichever way the trade turned
out to work. The bot trades consensus_dir instead.

final_exit_strategy is likewise the exit that won for that trade. TP3.0
is chosen in 67.7% of rows, so the bot fixes TP3.0 at entry.

The whole leverage chain IS computable at entry — it depends only on ATR
— so it is implemented exactly as the file has it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BAR_MINUTES = 30

# Exits. TP multiples are of ATR, against a 1.5 ATR stop.
EXIT_STRATEGIES = ["net_TP1.5_SL1.5", "net_TP2.0_SL1.5",
                   "net_TP3.0_SL1.5", "net_TRAILING"]
TP_MULTIPLES = {"net_TP1.5_SL1.5": 1.5, "net_TP2.0_SL1.5": 2.0,
                "net_TP3.0_SL1.5": 3.0}
SL_MULTIPLE = 1.5
TRAIL_MULTIPLE = 1.5
DEFAULT_EXIT = "net_TP3.0_SL1.5"        # the file's choice in 67.7% of rows
MAX_HOLD_BARS = 1000

# Leverage chain, recovered from the file.
LEVERAGE_MIN, LEVERAGE_MAX = 17.0, 100.0
LEV_BASE_K = 28.0                       # lev_base = clip(K / atr14_pct, 17, 100)
POTENTIAL_OFFSET = 0.50                 # multiplier = score + 0.50

# Solvency. The bot's liquidation model puts the liquidation price where
# 90% of the margin is gone; the stop must sit inside that with room to
# spare, or the position dies at 100% of margin instead of the 42% the
# stop was sized for. See solvency_cap().
LIQ_MARGIN_FRACTION = 0.90
SOLVENCY_BUFFER = 1.30

FEE_ROUND_TRIP = 0.0025                 # taker, at the file's leveraged rate
MAKER_ROUND_TRIP = 0.0004               # resting the order instead

# potential_score is a percentile, so it needs a distribution to rank
# against. These are the ATR percentiles of BTCUSDT 30m bars over
# 2025-2026, so a live bar can be scored without waiting for history.
ATR_PCT_REFERENCE = np.array([
    0.1333, 0.1588, 0.1780, 0.1940, 0.2098, 0.2243, 0.2399, 0.2555, 0.2722,
    0.2901, 0.3099, 0.3310, 0.3542, 0.3807, 0.4121, 0.4488, 0.4978, 0.5748,
    0.6966,
])


def potential_score(atr14_pct: float,
                    reference: np.ndarray = ATR_PCT_REFERENCE) -> float:
    """How much this instrument is moving, as a 0..1 percentile.

    Correlates +0.951 with the file's own column.
    """
    if not np.isfinite(atr14_pct) or atr14_pct <= 0:
        return 0.5
    # searchsorted over the 5%..95% quantiles gives the percentile directly.
    return float(np.clip(np.searchsorted(reference, atr14_pct)
                         / (len(reference) + 1), 0.0, 1.0))


def potential_multiplier(score: float) -> float:
    """The file's exact relation: multiplier = score + 0.5."""
    return score + POTENTIAL_OFFSET


def lev_base(atr14_pct: float) -> float:
    """Base leverage, falling as volatility rises so the stop stays inside
    the margin. Fitted against the file's column (correlation -0.953 with
    atr14_pct)."""
    if not np.isfinite(atr14_pct) or atr14_pct <= 0:
        return LEVERAGE_MAX
    return float(np.clip(LEV_BASE_K / atr14_pct, LEVERAGE_MIN, LEVERAGE_MAX))


def solvency_cap(atr14_pct: float) -> float:
    """Highest leverage at which the 1.5-ATR stop still sits INSIDE the
    liquidation price.

    Why this is needed. The stop is a fixed multiple of ATR, so the move
    it sits at is 1.5 x atr_pct. Liquidation is a fixed fraction of
    margin, so the move IT sits at is 0.90 / leverage. Setting the first
    below the second gives

        1.5 x atr_pct/100 <= 0.90 / lev     ->    lev <= 60 / atr_pct

    and lev_base is 28/atr_pct, comfortably below it — so through the
    normal range the file's chain is already solvent and this cap never
    binds. What breaks it is the 17x FLOOR: once atr_pct passes 3.53%,
    28/atr_pct falls under 17, the clip lifts leverage back to 17, and
    the stop lands past the liquidation price. The position then dies at
    100% of margin instead of the 42% the stop was sized for.

    Measured across the ATR range, that is the only place the chain is
    unsafe, so this is the only place the cap acts. It is a function of
    ATR like every other term, so leverage stays continuous and
    volatility-driven — nothing here is a fixed number of x.
    """
    if not np.isfinite(atr14_pct) or atr14_pct <= 0:
        return LEVERAGE_MAX
    sl_move = SL_MULTIPLE * atr14_pct / 100.0
    return float(LIQ_MARGIN_FRACTION / (SOLVENCY_BUFFER * sl_move))


def leverage_potential(atr14_pct: float, cap: float | None = None) -> dict:
    """The full chain: base, score, multiplier, and the final leverage.

    The product is capped at the base, exactly as the file has it, so a
    high multiplier never raises leverage above what the stop allows — it
    can only cut it. Then the solvency bound applies, which only ever
    binds where the 17x floor would have put the stop past liquidation.

    `tradeable` is False when even 1x cannot keep the stop inside the
    liquidation price — an instrument moving more than ~46% per 30m bar.
    The bot skips those rather than clamping to 1x and taking a position
    it knows is unsound.
    """
    base = lev_base(atr14_pct)
    score = potential_score(atr14_pct)
    mult = potential_multiplier(score)
    lev = min(base * mult, base)
    solvent = solvency_cap(atr14_pct)
    lev = min(lev, solvent)
    if cap is not None:
        lev = min(lev, cap)
    return {"lev_base": base, "potential_score": score,
            "potential_multiplier": mult, "solvency_cap": solvent,
            "leverage": max(1.0, lev), "tradeable": lev >= 1.0}


def simulate_exit(high, low, close, i: int, direction: int, atr: float,
                  exit_name: str = DEFAULT_EXIT,
                  fee: float = FEE_ROUND_TRIP) -> tuple[float, int, str]:
    """Run one trade to its exit. Returns (net return, bars held, reason).

    On a bar whose range covers both the stop and the target, the stop is
    taken — an ambiguous bar never resolves in the trade's favour.
    """
    entry = close[i]
    n = len(close)
    sl = entry - direction * SL_MULTIPLE * atr

    if exit_name == "net_TRAILING":
        best = entry
        for j in range(i + 1, min(i + 1 + MAX_HOLD_BARS, n)):
            if (low[j] <= sl) if direction > 0 else (high[j] >= sl):
                return -SL_MULTIPLE * atr / entry - fee, j - i, "stop"
            if direction > 0:
                best = max(best, high[j])
                trail = best - TRAIL_MULTIPLE * atr
                if low[j] <= trail:
                    return (trail - entry) / entry - fee, j - i, "trail"
            else:
                best = min(best, low[j])
                trail = best + TRAIL_MULTIPLE * atr
                if high[j] >= trail:
                    return (entry - trail) / entry - fee, j - i, "trail"
    else:
        tp = entry + direction * TP_MULTIPLES[exit_name] * atr
        for j in range(i + 1, min(i + 1 + MAX_HOLD_BARS, n)):
            if (low[j] <= sl) if direction > 0 else (high[j] >= sl):
                return -SL_MULTIPLE * atr / entry - fee, j - i, "stop"
            if (high[j] >= tp) if direction > 0 else (low[j] <= tp):
                return TP_MULTIPLES[exit_name] * atr / entry - fee, j - i, "target"

    j = min(i + MAX_HOLD_BARS, n - 1)
    return (close[j] - entry) / entry * direction - fee, j - i, "timeout"


def to_bars(df_1m: pd.DataFrame, minutes: int = BAR_MINUTES) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    return (df_1m.set_index("datetime").resample(f"{minutes}min")
            .agg(agg).dropna().reset_index())
