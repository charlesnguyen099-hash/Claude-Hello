"""The barrier arithmetic, and nothing else.

sigma_at and barrier_outcomes are the two functions the whole system
stands on: one says how volatile a bar is, the other says what a trade
with a target, a stop and a time limit actually returned. They used to
live in fp/exits.py, a research script that imported four other research
modules to run its own command line -- so importing the barrier walk
dragged in the entire dead search with it.

Nothing here imports anything from fp. That is the point: the live bot
and the training pipeline both need this arithmetic, and neither should
have to load a study to get it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sigma_at(close: np.ndarray, window: int = 50) -> np.ndarray:
    """Trailing volatility per bar, past-only, in return units."""
    r = pd.Series(close).pct_change()
    s = r.rolling(window, min_periods=window // 2).std().shift(1)
    return s.bfill().fillna(0.0).values


def barrier_outcomes(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                     sigma: np.ndarray, direction: int, tp: float, sl: float,
                     max_h: int) -> tuple[np.ndarray, np.ndarray]:
    """Return and bars-held for a trade opened at every bar, vectorised.

    Walks forward one bar at a time keeping a mask of trades still open,
    so the FIRST touch wins -- which is the whole point of a barrier and
    the thing a close-only test gets wrong.
    """
    n = len(close)
    entry = close
    up = entry * (1.0 + direction * tp * sigma) if direction > 0 else \
        entry * (1.0 - tp * sigma)
    dn = entry * (1.0 - sl * sigma) if direction > 0 else \
        entry * (1.0 + sl * sigma)

    out = np.full(n, np.nan)
    held = np.zeros(n, dtype=int)
    live = np.ones(n, dtype=bool)
    live[np.maximum(n - max_h - 1, 0):] = False        # cannot complete
    live &= sigma > 0

    for k in range(1, max_h + 1):
        idx = np.flatnonzero(live)
        if len(idx) == 0:
            break
        j = idx + k
        hi, lo = high[j], low[j]
        if direction > 0:
            hit_sl = lo <= dn[idx]           # pessimistic: stop checked first
            hit_tp = hi >= up[idx]
        else:
            hit_sl = hi >= dn[idx]
            hit_tp = lo <= up[idx]
        done_sl = hit_sl
        done_tp = hit_tp & ~hit_sl
        for mask, level in ((done_sl, dn), (done_tp, up)):
            w = idx[mask]
            if len(w):
                out[w] = direction * (level[w] - entry[w]) / entry[w]
                held[w] = k
                live[w] = False

    # never touched: close at the time limit
    rest = np.flatnonzero(live)
    if len(rest):
        j = np.minimum(rest + max_h, n - 1)
        out[rest] = direction * (close[j] - entry[rest]) / entry[rest]
        held[rest] = max_h
    return out, held
