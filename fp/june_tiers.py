"""Three tiers of logic on the June block: a month, not a day.

The August study could only run at 1m/3m/5m -- 34 hours does not give
136 fifteen-minute bars enough history for the library to build. A month
gives 2,582, so this runs at 5m, 15m, 30m and 1h: the timeframes the
book actually uses, and the ones where the round trip is a bearable
share of the move.

Tier 1  per coin   -- a signal that pays on one symbol, on its own
Tier 2  per group  -- the same signal paying across a correlated block
Tier 3  all coins  -- the same signal paying across uncorrelated blocks

Every number is net of taker fees both sides and funding on the hours
actually held. Four controls, all of them mandatory:

  time split      positive in the FIRST half and the SECOND, measured
                  separately. A rule that only worked while one move was
                  running shows up as one good half and one dead one.
  Bonferroni      against every combination attempted, not the handful
                  that survived. Correcting against the survivors prices
                  one lottery ticket and ignores the rest of the roll.
  leave-one-out   any cross-symbol claim is re-scored with its best
                  symbol removed; the worst case is the number kept.
  rotation null   the WHOLE selection re-run on data whose logic series
                  have been rolled by a random offset. That keeps drift,
                  volatility and long/short balance and destroys only
                  the alignment, so whatever the real run beats it by is
                  what the signal is worth -- and whatever it does not
                  is the search finding itself.

The August run failed the last control outright: rolled data produced
MORE survivors than the real thing. This one is longer by a factor of
twenty, which is the only thing that changed.

RESULT: NOTHING SURVIVES HERE EITHER.

1,458,624 cells across 5m, 15m, 30m and 1h, counting only independent
trades. Bonferroni bar |t| > 5.52. Then the whole selection re-run five
times per timeframe on rolled data:

       tf         tier1            tier2            tier3
                real/null        real/null        real/null
       5m          0 / 0.2        134 / 176          29 / 58
      15m          2 / 0.6       2503 / 2304        586 / 683
      30m          4 / 2.2       3623 / 4086       1112 / 1162
       1h          4 / 8.6       1407 / 1737        541 / 415
      ALL        10 / 11.6       7667 / 8303       2268 / 2317

      ratio    tier1 0.86x      tier2 0.92x      tier3 0.98x

Every ratio sits at or below one. Destroying the alignment entirely --
keeping drift, volatility and the long/short balance, breaking only the
timing -- produces as many survivors as the real data, and at 1h it
produces more than twice as many tier-1 rules. The ten that passed are
what a search of this size hands out for free; nine of them rest on
twelve to twenty independent trades.

So the month answers the question the 34-hour window could not: it is
not the length that was missing. A month of ten symbols at four
timeframes, with overlapping trades removed and every control applied,
contains no per-coin, group or all-coin logic that beats its own null.
"""
from __future__ import annotations

import collections
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from fp import june as J
from fp import ensemble as E

OUT = Path(__file__).resolve().parent / "june_tiers.json"

TFS = [("5m", 5), ("15m", 15), ("30m", 30), ("1h", 60)]

# Target and stop in units of sigma at entry; the limit in bars. Every
# extra cell is another test the correction has to pay for, so the grid
# is the smallest one that still spans "tight target" to "wide target".
BARRIERS = [(2.0, 1.0), (3.0, 1.5), (4.0, 3.0), (2.0, 2.0)]
HMAX = [24, 96]

# Counted in INDEPENDENT trades -- non-overlapping barrier windows --
# so these are lower than they look against a raw firing count.
MIN_TRADES = 12          # per cell, over the whole month
MIN_HALF = 4             # per half


def outcomes(d: pd.DataFrame, minutes: int) -> dict:
    """Net outcome AND bars held per bar, for every (side, tp, sl, hmax).

    Held is needed as well as the outcome: two entries whose barrier
    windows overlap are not two observations, and only the exit bar says
    where one window ends. See scan().
    """
    from fp.exits import barrier_outcomes, sigma_at
    c = d["close"].values.astype(float)
    h = d["high"].values.astype(float)
    lo = d["low"].values.astype(float)
    sg = sigma_at(c)
    out = {}
    for side in (1, -1):
        for tp, sl in BARRIERS:
            for hm in HMAX:
                o, held = barrier_outcomes(h, lo, c, sg, side, tp, sl, hm)
                net = o - J.FEE - held * minutes / 60.0 / 8.0 * J.FUNDING_PER_8H
                out[(side, tp, sl, hm)] = (net, held)
    return out


def independent(idx: np.ndarray, net: np.ndarray,
                held: np.ndarray) -> np.ndarray:
    """Keep only entries whose barrier windows do not overlap.

    A rule firing on consecutive bars with a 96-bar limit opens trades
    that watch almost the same four days. Their outcomes are nearly the
    same number, so the standard error is far too small and t is far too
    large. Measured on the June block, SNDK 1h volr55|rankfade90_0.85
    read t = 7.81 across 34 overlapping trades and t = 2.26 across the 10
    independent ones -- the difference between clearing a Bonferroni bar
    of 5.51 and not coming close.

    The next entry is taken only once the previous one has exited.
    """
    keep = []
    free = -1
    for i in idx:
        if i <= free or not np.isfinite(net[i]):
            continue
        keep.append(i)
        free = i + int(held[i])
    return np.array(keep, dtype=int)


def scan(logics: dict, outs: dict, n: int) -> dict:
    """Every (logic, side, barrier) cell on one symbol, split in half."""
    mid = n // 2
    res = {}
    for name, ser in logics.items():
        v = np.asarray(ser, dtype=float)
        for side in (1, -1):
            on = np.zeros(n, dtype=bool)
            on[1:] = (v[1:] == side) & (v[:-1] != side)
            idx = np.flatnonzero(on)
            if len(idx) < MIN_TRADES:
                continue
            for (s2, tp, sl, hm), (net, held) in outs.items():
                if s2 != side:
                    continue
                i2 = independent(idx, net, held)
                if len(i2) < MIN_TRADES:
                    continue
                x2 = net[i2]
                h1, h2 = x2[i2 < mid], x2[i2 >= mid]
                if len(h1) < MIN_HALF or len(h2) < MIN_HALF:
                    continue
                res[(name, side, tp, sl, hm)] = {
                    "n": len(x2), "mean": float(x2.mean()),
                    "h1": float(h1.mean()), "h2": float(h2.mean()),
                    "sd": float(x2.std(ddof=1)),
                    "wins": int((x2 > 0).sum()),
                }
    return res


def tstat(c: dict) -> float:
    return c["mean"] / (c["sd"] / math.sqrt(c["n"])) if c["sd"] > 0 else 0.0


def bonferroni_t(attempted: int, alpha: float = 0.05) -> float:
    p = alpha / max(attempted, 1)
    z = 3.0
    for _ in range(80):
        z = math.sqrt(max(2.0 * math.log(2.0 / (p * z * math.sqrt(2 * math.pi))),
                          1e-9))
    return z


def select(scans: dict, attempted: int) -> tuple[list, list, list]:
    """The three tiers, from one set of per-symbol scans."""
    tcrit = bonferroni_t(attempted)
    t1 = []
    agg = collections.defaultdict(dict)
    for sym, cells in scans.items():
        for k, c in cells.items():
            agg[k][sym] = c
            if c["h1"] > 0 and c["h2"] > 0 and tstat(c) >= tcrit:
                t1.append({"sym": sym, "cell": k, "t": tstat(c), **c})
    t2, t3 = [], []
    for k, hits in agg.items():
        paid = [s for s, c in hits.items() if c["h1"] > 0 and c["h2"] > 0]
        if len(paid) < 3:
            continue
        means = np.array([hits[s]["mean"] for s in paid])
        ns = np.array([hits[s]["n"] for s in paid])
        # Drop the BEST symbol, not a random one: a cross-symbol claim
        # carried by one name is one observation in a crowd's clothing.
        loo = float(np.delete(means, int(np.argmax(means))).mean())
        if loo <= 0:
            continue
        blocks = {J.BLOCK.get(s, s) for s in paid}
        rec = {"cell": k, "symbols": sorted(paid), "blocks": sorted(blocks),
               "mean": float(np.average(means, weights=ns)),
               "loo_worst": loo, "n": int(ns.sum())}
        (t3 if len(blocks) >= 3 else t2).append(rec)
    return t1, t2, t3
