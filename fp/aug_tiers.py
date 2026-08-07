"""Three tiers of logic on the 34-hour window of 2026-08-05..08-07.

Tier 1  per coin      -- independent signals, one symbol at a time
Tier 2  per group     -- the same signal paying across a correlated block
Tier 3  all coins     -- the same signal paying across uncorrelated blocks

The window is short. At the timeframes the book uses -- 15m and slower --
136 bars is not enough history for most of the logic library to build at
all, so this study runs at 1m, 3m and 5m, which is where 2,040 minutes
of data can actually support a lookback. That is the short end, where
the round trip is the largest share of the move, and the results say so.

Every number is net: taker in, taker out, and funding on the real hours
held. The controls are the ones the earlier studies established and are
not optional here:

  time split      a rule must pay in the first half AND the second,
                  measured separately, before it is looked at again
  rotation null   the rule's own position series rolled by a random
                  offset, which keeps drift, volatility and long/short
                  balance and destroys only the alignment
  Bonferroni      against EVERY combination attempted, not the handful
                  that survived the gates
  leave-one-out   for any cross-symbol claim, the worst symbol removed
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fp import aug as A
from fp import ensemble as E
from fp.exits import barrier_outcomes, sigma_at
from fp.logic import FEE_ROUND_TRIP as FEE

OUT = Path(__file__).resolve().parent / "aug_tiers.json"

TFS = [("1m", 1), ("3m", 3), ("5m", 5)]

# Targets and stops in units of sigma at entry, with the time limit in
# bars. Kept small on purpose: every extra cell is another test the
# Bonferroni correction has to pay for.
BARRIERS = [(2.0, 1.0), (3.0, 1.5), (4.0, 3.0), (2.0, 2.0)]
HMAX = [24, 96]

# Correlated blocks. Counting symbols is not counting evidence: three
# semiconductor names moving together are one bet, not three.
BLOCK = {"SKHYNIXUSDT": "semi", "SNDKUSDT": "semi", "SOXLUSDT": "semi",
         "ETHUSDT": "crypto", "SOLUSDT": "crypto", "BTCUSDT": "crypto",
         "XRPUSDT": "crypto", "HYPEUSDT": "crypto",
         "BLESSUSDT": "bless", "XAUUSDT": "gold"}

FUNDING_PER_8H = 1e-4


def outcomes(d: pd.DataFrame, minutes: int) -> dict:
    """Net barrier outcome per bar for every (side, tp, sl, hmax) cell.

    Computed once per symbol and reused by every logic, because the
    barrier does not care which signal opened the trade.
    """
    c = d["close"].values.astype(float)
    h = d["high"].values.astype(float)
    lo = d["low"].values.astype(float)
    sg = sigma_at(c)
    out = {}
    for side in (1, -1):
        for tp, sl in BARRIERS:
            for hm in HMAX:
                o, held = barrier_outcomes(h, lo, c, sg, side, tp, sl, hm)
                fund = held * minutes / 60.0 / 8.0 * FUNDING_PER_8H
                out[(side, tp, sl, hm)] = o - FEE - fund
    return out


def entries(logics: dict) -> dict:
    """Bars where each logic turns on, per side."""
    out = {}
    for name, s in logics.items():
        v = s.values.astype(float)
        for side in (1, -1):
            on = np.zeros(len(v), dtype=bool)
            on[1:] = (v[1:] == side) & (v[:-1] != side)
            if on.sum() >= 6:
                out[(name, side)] = on
    return out


def scan(d: pd.DataFrame, minutes: int, min_trades: int = 8) -> dict:
    """Every (logic, side, barrier) cell on one symbol, split in half.

    The split is by TIME, not by trade count: a rule that only worked
    while one move was running shows up as one good half and one dead
    one, which is the thing the split exists to catch.
    """
    L = E.build_logics(d)
    if not L:
        return {}
    ent = entries(L)
    outs = outcomes(d, minutes)
    n = len(d)
    mid = n // 2
    res = {}
    for (name, side), on in ent.items():
        idx = np.flatnonzero(on)
        for (s2, tp, sl, hm), net in outs.items():
            if s2 != side:
                continue
            v = net[idx]
            ok = np.isfinite(v)
            v2, i2 = v[ok], idx[ok]
            if len(v2) < min_trades:
                continue
            h1 = v2[i2 < mid]
            h2 = v2[i2 >= mid]
            if len(h1) < 3 or len(h2) < 3:
                continue
            res[(name, side, tp, sl, hm)] = {
                "n": len(v2), "mean": float(v2.mean()),
                "h1": float(h1.mean()), "h2": float(h2.mean()),
                "n1": len(h1), "n2": len(h2),
                "wins": int((v2 > 0).sum()),
                "sd": float(v2.std(ddof=1)) if len(v2) > 1 else 0.0,
            }
    return res


def rotation_null(d: pd.DataFrame, minutes: int, cells: list,
                  rounds: int = 40, seed: int = 0) -> float:
    """What the same selection finds when alignment is destroyed.

    The position series is rolled by a random offset, so drift,
    volatility and the long/short balance all survive and only the
    timing is broken. Whatever the real selection beats this by is what
    the signal is worth; whatever it does not is the search itself.
    """
    rng = np.random.default_rng(seed)
    L = E.build_logics(d)
    if not L:
        return float("nan")
    outs = outcomes(d, minutes)
    n = len(d)
    best = []
    for _ in range(rounds):
        hits = []
        for name, side, tp, sl, hm in cells:
            s = L.get(name)
            if s is None:
                continue
            v = np.roll(s.values.astype(float), int(rng.integers(10, n - 10)))
            on = np.zeros(n, dtype=bool)
            on[1:] = (v[1:] == side) & (v[:-1] != side)
            idx = np.flatnonzero(on)
            if len(idx) < 8:
                continue
            net = outs[(side, tp, sl, hm)][idx]
            net = net[np.isfinite(net)]
            if len(net) >= 8:
                hits.append(net.mean())
        if hits:
            best.append(max(hits))
    return float(np.mean(best)) if best else float("nan")


def run() -> dict:
    D = A.load()
    report = {"window": "2026-08-05T17:00..2026-08-07T02:59 (34h)",
              "symbols": sorted(D), "timeframes": [t for t, _ in TFS],
              "tiers": {}}
    per_tf = {}
    for label, m in TFS:
        frames = {s: (A.resample(d, m) if m > 1 else d) for s, d in D.items()}
        scans = {}
        attempted = 0
        for s, d in frames.items():
            r = scan(d, m)
            scans[s] = r
            attempted += len(r)
        per_tf[label] = {"frames": frames, "scans": scans,
                         "attempted": attempted, "minutes": m}
        print(f"  {label}: {len(frames)} symbols, {attempted:,} cells tested",
              flush=True)
    report["_per_tf"] = per_tf
    return report


if __name__ == "__main__":
    run()


# ------------------------------------------------------------------ tiers

def tstat(mean: float, sd: float, n: int) -> float:
    return mean / (sd / np.sqrt(n)) if (sd > 0 and n > 1) else 0.0


def bonferroni_t(attempted: int, alpha: float = 0.05) -> float:
    """Two-sided t threshold once every combination attempted is paid for.

    Attempted, not survived. Correcting against the handful that passed
    the gates prices one lottery ticket and ignores the rest of the roll.
    """
    from math import sqrt, log
    p = alpha / max(attempted, 1)
    # Normal tail approximation, good enough at these thresholds.
    z = sqrt(2.0 * log(2.0 / p)) if p < 1 else 0.0
    for _ in range(60):
        z = sqrt(max(2.0 * log(2.0 / (p * z * sqrt(2 * np.pi))), 1e-9))
    return z


def paid_both_halves(cell: dict) -> bool:
    return cell["h1"] > 0 and cell["h2"] > 0


def analyse(per_tf: dict) -> dict:
    total_attempted = sum(v["attempted"] for v in per_tf.values())
    tcrit = bonferroni_t(total_attempted)
    out = {"attempted": total_attempted, "t_bonferroni": tcrit, "tf": {}}

    for label, blob in per_tf.items():
        scans, frames, m = blob["scans"], blob["frames"], blob["minutes"]

        # ---- tier 1: one coin, its own signal ----------------------
        t1 = []
        for sym, cells in scans.items():
            for k, c in cells.items():
                if not paid_both_halves(c):
                    continue
                t = tstat(c["mean"], c["sd"], c["n"])
                if t >= tcrit:
                    t1.append({"sym": sym, "cell": k, **c, "t": t})

        # ---- tier 2 and 3: the same cell across symbols ------------
        agg = collections.defaultdict(dict)
        for sym, cells in scans.items():
            for k, c in cells.items():
                agg[k][sym] = c
        t2, t3 = [], []
        for k, hits in agg.items():
            paid = [s for s, c in hits.items() if paid_both_halves(c)]
            if len(paid) < 3:
                continue
            blocks = {BLOCK.get(s, s) for s in paid}
            means = np.array([hits[s]["mean"] for s in paid])
            ns = np.array([hits[s]["n"] for s in paid])
            # Leave-one-out: the worst symbol removed is the number that
            # matters. A "cross-symbol" rule carried by one name is one
            # observation wearing a crowd's clothes.
            loo = min(float(np.delete(means, i).mean())
                      for i in range(len(means)))
            rec = {"cell": k, "symbols": sorted(paid), "blocks": sorted(blocks),
                   "mean": float(np.average(means, weights=ns)),
                   "loo_worst": loo, "n": int(ns.sum())}
            if loo <= 0:
                continue
            if len(blocks) >= 3:
                t3.append(rec)
            elif len(paid) >= 3:
                t2.append(rec)

        out["tf"][label] = {
            "cells": blob["attempted"],
            "tier1": sorted(t1, key=lambda r: -r["t"])[:40],
            "tier1_count": len(t1),
            "tier2": sorted(t2, key=lambda r: -r["loo_worst"])[:40],
            "tier2_count": len(t2),
            "tier3": sorted(t3, key=lambda r: -r["loo_worst"])[:40],
            "tier3_count": len(t3),
        }
    return out
