"""The five exit strategies of FINAL_Wide_Logic_AllMethods_AllFactors.

The file scores every setup under all five and records the net result of
each, so a bar carries five parallel answers to "what would this trade
have returned":

    net_TP1.0_SL1.5     take profit at 1.0 ATR, stop at 1.5 ATR
    net_TP1.5_SL1.5     1.5 / 1.5
    net_TP2.0_SL1.5     2.0 / 1.5
    net_TP3.0_SL1.5     3.0 / 1.5
    net_TRAILING        no fixed target; trail behind the best price

It also records best_exit_strategy and best_exit_net — the winner among
those five for that particular trade. Those two columns are the reason
the file looks profitable overall (best_exit_net averages +0.1343 while
every individual strategy averages between -0.2191 and -0.2563), and they
are also the two a live bot cannot use: which exit turns out best is only
known after the trade is over.

So both are implemented. `simulate_all` produces the file's five columns
plus the best-of, and the backtest reports them separately, because the
gap between "best fixed exit" and "best exit chosen per trade" is exactly
the value of information the bot does not have.
"""
from __future__ import annotations

import numpy as np

STRATEGIES = ["net_TP1.0_SL1.5", "net_TP1.5_SL1.5", "net_TP2.0_SL1.5",
              "net_TP3.0_SL1.5", "net_TRAILING"]

TP_MULTIPLES = {"net_TP1.0_SL1.5": 1.0, "net_TP1.5_SL1.5": 1.5,
                "net_TP2.0_SL1.5": 2.0, "net_TP3.0_SL1.5": 3.0}
SL_MULTIPLE = 1.5
TRAIL_MULTIPLE = 1.5
MAX_HOLD_BARS = 1000
FEE_ROUND_TRIP = 0.0025      # matches the file's leveraged fee at 100x


def _fixed(high, low, close, i, direction, atr, tp_mult):
    """Take-profit / stop-loss exit. Stop is checked first on any bar whose
    range covers both, so an ambiguous bar never resolves in our favour."""
    entry = close[i]
    tp = entry + direction * tp_mult * atr
    sl = entry - direction * SL_MULTIPLE * atr
    n = len(close)
    for j in range(i + 1, min(i + 1 + MAX_HOLD_BARS, n)):
        hit_sl = (low[j] <= sl) if direction > 0 else (high[j] >= sl)
        hit_tp = (high[j] >= tp) if direction > 0 else (low[j] <= tp)
        if hit_sl:
            return -SL_MULTIPLE * atr / entry - FEE_ROUND_TRIP, j - i
        if hit_tp:
            return tp_mult * atr / entry - FEE_ROUND_TRIP, j - i
    j = min(i + MAX_HOLD_BARS, n - 1)
    return (close[j] - entry) / entry * direction - FEE_ROUND_TRIP, j - i


def _trailing(high, low, close, i, direction, atr):
    """No target: ride the move, exit when it gives back TRAIL_MULTIPLE ATR
    from its best point."""
    entry = close[i]
    n = len(close)
    best = entry
    for j in range(i + 1, min(i + 1 + MAX_HOLD_BARS, n)):
        if direction > 0:
            best = max(best, high[j])
            if low[j] <= best - TRAIL_MULTIPLE * atr:
                return (best - TRAIL_MULTIPLE * atr - entry) / entry - FEE_ROUND_TRIP, j - i
        else:
            best = min(best, low[j])
            if high[j] >= best + TRAIL_MULTIPLE * atr:
                return (entry - best - TRAIL_MULTIPLE * atr) / entry - FEE_ROUND_TRIP, j - i
    j = min(i + MAX_HOLD_BARS, n - 1)
    return (close[j] - entry) / entry * direction - FEE_ROUND_TRIP, j - i


def simulate_all(high, low, close, i: int, direction: int, atr: float) -> dict:
    """All five exits for one entry, plus the best of them."""
    out = {}
    for name, mult in TP_MULTIPLES.items():
        net, dur = _fixed(high, low, close, i, direction, atr, mult)
        out[name] = net
        out["dur_" + name[4:]] = dur
    net, dur = _trailing(high, low, close, i, direction, atr)
    out["net_TRAILING"] = net
    out["dur_TRAILING"] = dur

    best = max(STRATEGIES, key=lambda s: out[s])
    out["best_exit_strategy"] = best
    out["best_exit_net"] = out[best]
    out["any_strategy_profitable"] = out[best] > 0
    return out


def leverage_flexible(mae_pct: float, cap: float = 100.0,
                      floor: float = 17.0) -> float:
    """The file's leverage_flexible column: 17x to 100x, median 86.7x.

    Sized so the expected adverse excursion stays inside the margin, and
    clamped to the range the file actually used.
    """
    if mae_pct <= 0:
        return cap
    return float(np.clip(90.0 / mae_pct, floor, cap))
