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

MIN_TRADES = 20          # per cell, over the whole month
MIN_HALF = 8             # per half


def outcomes(d: pd.DataFrame, minutes: int) -> dict:
    """Net outcome per bar for every (side, tp, sl, hmax). Computed once
    per symbol: the barrier does not care which signal opened the trade."""
    return {(side, tp, sl, hm): J.net(d, minutes, side, tp, sl, hm)
            for side in (1, -1) for tp, sl in BARRIERS for hm in HMAX}


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
            for (s2, tp, sl, hm), net in outs.items():
                if s2 != side:
                    continue
                x = net[idx]
                ok = np.isfinite(x)
                x2, i2 = x[ok], idx[ok]
                if len(x2) < MIN_TRADES:
                    continue
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
