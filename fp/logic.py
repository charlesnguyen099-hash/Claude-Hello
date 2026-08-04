"""The trading logic of FINAL_Profitable_Logic_Only, implemented for live use.

WHAT THE FILE CONTAINS

22,533 rows, every one profitable: final_net runs from +0.0001% to +8.17%
with no negative row. Each carries the twelve method votes, the vote
tally, 106 features, a chosen direction, a chosen exit, and a flexible
leverage between 17x and 100x (median 85.6x).

    final_direction        SHORT 11,521 / LONG 11,012
    final_exit_strategy    TP3.0 15,256 | TP2.0 4,253 | TP1.5 2,922 | TRAIL 102
    final_net              median +0.4898%, minimum +0.0001%, none negative
    leverage_flexible      17-100x, median 85.6x

THE PART A BOT CAN RUN, AND THE PART IT CANNOT

Three of those columns are decisions, and they are not the same kind of
decision.

`final_direction` is not the methods' verdict. Of the 19,717 rows where
the twelve methods did reach a consensus, final_direction agrees with it
9,930 times and reverses it 9,787 — 50.4% against 49.6%, a coin flip.
It is whichever way the trade turned out to work, so it cannot be
computed before the trade. The bot therefore trades `consensus_dir`,
which is the methods' actual output.

`final_exit_strategy` has the same problem in weaker form: it is the exit
that won for that trade. TP3.0 is chosen 67.7% of the time, so the bot
fixes TP3.0 at entry as the single best standing guess.

`leverage_flexible` is computable — it is sized from the setup's own stop
distance, which is known at entry. That one is used as the file has it.

So what runs live is: twelve methods vote, consensus sets direction, the
stop distance sets leverage, TP3.0/SL1.5 closes the trade.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BAR_MINUTES = 30

# The file's exits. TP multiples are of ATR, against a 1.5 ATR stop.
EXIT_STRATEGIES = ["net_TP1.5_SL1.5", "net_TP2.0_SL1.5",
                   "net_TP3.0_SL1.5", "net_TRAILING"]
TP_MULTIPLES = {"net_TP1.5_SL1.5": 1.5, "net_TP2.0_SL1.5": 2.0,
                "net_TP3.0_SL1.5": 3.0}
SL_MULTIPLE = 1.5
TRAIL_MULTIPLE = 1.5
DEFAULT_EXIT = "net_TP3.0_SL1.5"      # the file's choice in 67.7% of rows
MAX_HOLD_BARS = 1000

# The file's leverage band.
LEVERAGE_MIN, LEVERAGE_MAX = 17.0, 100.0
FEE_ROUND_TRIP = 0.0025               # taker at the file's leveraged rate
MAKER_ROUND_TRIP = 0.0004             # resting the order instead


def leverage_flexible(stop_distance_pct: float) -> float:
    """The file's leverage_flexible column: 17x to 100x, median 85.6x.

    Sized so the stop sits inside the margin rather than beyond it, then
    clamped to the range the file actually used.
    """
    if stop_distance_pct <= 0:
        return LEVERAGE_MAX
    return float(np.clip(90.0 / stop_distance_pct, LEVERAGE_MIN, LEVERAGE_MAX))


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


def features_to_bars(df_1m: pd.DataFrame, minutes: int = BAR_MINUTES) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    return (df_1m.set_index("datetime").resample(f"{minutes}min")
            .agg(agg).dropna().reset_index())
