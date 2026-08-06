"""Rank the coins against each other: a logic that needs no market view.

    python -m fp.crosscoin
    python -m fp.crosscoin --tf 15m --legs 2

THE FAMILY THIS OPENS, WHICH ONE SYMBOL COULD NOT

Every logic in this repo so far asks one question: is this coin going up
or down. That question needs a market view, and a market view is what
none of 2.1 million tested rules could produce on BTC.

Nine symbols allow a different question, one that has no direction in it
at all: which of these is strongest RELATIVE to the others. Long the top
of the ranking, short the bottom, equal money each side. If everything
rises together the position makes nothing and loses nothing; the P&L is
the spread between what was picked long and what was picked short.

That is why it is worth running even on five days of data. It does not
need the market to go anywhere.

HOW IT WORKS

At each bar every coin gets a factor value -- momentum over k bars,
distance from its own moving average, realised volatility, the position
in its own range -- computed from that coin's own past only. The coins
are then RANKED against each other on that value, and the book is:

    long    the top --legs coins
    short   the bottom --legs coins

Every factor is normalised by the coin's own history before ranking, so
a coin that moves 29% a day and one that moves 1% a day are compared on
how unusual their move is for THEM, not on the raw size. Without that
the ranking is just a volatility sort and BLESSUSDT wins every bar.

COSTS ARE CHARGED ON TURNOVER, WHICH IS THE HARD PART

A cross-sectional book rebalances, and every change of membership is a
round trip on two legs. So the fee is charged on the actual change in
weights each bar, not per position, and --hold sets how many bars a
ranking is kept before it is allowed to change. A book that reshuffles
every bar pays 0.11% on most of its notional every bar and cannot win;
the sweep over --hold is where that shows up.

WHAT THE NULL IS

The same ranking with each coin's factor series rotated by a random
offset. That preserves every coin's own drift and volatility, and the
book's long/short structure, and destroys only which coin is picked
when. A book that merely held BLESSUSDT long all week scores the same
rotated as it does real.

WHAT IT MEASURED, and the diagnostic that decided it

Twenty books were positive after fees and beat the rotation null at
p<=0.10; thirteen reached p<=0.05. Then leave-one-out:

    tf   factor      hold  dir       total   without its worst symbol
    5m   pos5          48  follow   +34.33%   +12.08%   (BLESSUSDT)
    15m  volr30        12  follow   +49.80%   +11.98%
    15m  pos5          48  follow   +82.70%    +8.36%
    15m  volr30        48  follow   +86.05%    +4.48%
    5m   madist240     48  follow  +149.04%    +3.43%
    15m  mom240        48  reverse +128.57%    +1.25%
    15m  mom240        12  reverse  +93.15%    +0.72%
    15m  mom5          48  follow   +96.07%    +0.20%
    5m   volr30        48  follow  +101.90%    -7.45%
    5m   madist240     12  follow  +131.55%    -6.81%
    15m  mom60          1  follow  +135.02%    -6.44%

The symbol removed is BLESSUSDT every time, and it is the same story in
every row: BLESSUSDT rose 223.91% in 5.7 days, and a momentum ranking
holds whatever is running. Decomposed by symbol, BLESSUSDT contributed
+82.35% of the +81.8% gross on volr30, +94.05% of +99.5% on madist240,
+89.81% of +97.1% on mom240 -- the other eight symbols together are
noise around zero.

So most of these are not books. They are one coin wearing nine names,
and their effective sample size is ONE event, not 1,638 bars. Sharpe 27
on a single parabolic move is arithmetic, not evidence.

WHAT SURVIVES, AND HOW MUCH IT IS WORTH

pos5 held 48 bars is the only rule positive at both timeframes with its
best symbol removed: +12.08% at 5m and +8.36% at 15m. It is long the
coins sitting near the top of their own recent range and short those
near the bottom -- short-horizon cross-sectional momentum.

It still does not clear a multiple-testing bar. Twelve factors x four
holds x two directions x two timeframes is 192 tests; Bonferroni at 0.05
needs p < 0.00026 and the best pos5 p is 0.008. Eight survivors out of
twenty is also close to what half-noise would give.

So: one candidate worth watching, no confirmed logic, and the honest
resolution needs the same nine symbols over months rather than days --
enough that a BLESSUSDT-style run is one event among many instead of
the whole sample.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from fp.coins import load_coins
from fp.horizon import FEE_ROUND_TRIP, FUNDING_PER_8H, resample

TF = {"5m": 5, "15m": 15, "30m": 30, "60m": 60}
HOLDS = (1, 4, 12, 48)


def panel(coins: dict[str, pd.DataFrame], minutes: int) -> pd.DataFrame:
    """Closing prices of every symbol on one shared time grid."""
    cols = {}
    for sym, d in coins.items():
        b = resample(d, minutes)
        cols[sym] = b["close"]
    p = pd.DataFrame(cols).dropna()
    return p


def factors(p: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Cross-sectionally comparable scores, one frame per factor.

    Each is standardised by the coin's OWN trailing behaviour before any
    comparison, so the ranking is about how unusual a move is for that
    coin rather than about which coin is most volatile.
    """
    r = p.pct_change()
    out: dict[str, pd.DataFrame] = {}
    for k in (5, 15, 60, 240):
        mom = p.pct_change(k)
        sd = r.rolling(max(k, 30)).std() * np.sqrt(k)
        out[f"mom{k}"] = (mom / sd.replace(0, np.nan))
        ma = p.rolling(k).mean()
        out[f"madist{k}"] = ((p / ma - 1) / sd.replace(0, np.nan))
        hi, lo = p.rolling(k).max(), p.rolling(k).min()
        out[f"pos{k}"] = ((p - lo) / (hi - lo).replace(0, np.nan))
    for k in (30, 120):
        v = r.rolling(k).std()
        out[f"volr{k}"] = v / v.rolling(k * 2, min_periods=k).mean()
    return {k: v.replace([np.inf, -np.inf], np.nan) for k, v in out.items()}


def run_book(p: pd.DataFrame, score: pd.DataFrame, legs: int, hold: int,
             minutes: int, reverse: bool = False) -> np.ndarray:
    """Per-bar net return of the long/short book, fees on real turnover."""
    r = p.pct_change().shift(-1)          # the bar the position is held for
    s = score if not reverse else -score
    ranks = s.rank(axis=1, ascending=False)
    n = ranks.shape[1]
    w = pd.DataFrame(0.0, index=p.index, columns=p.columns)
    w[ranks <= legs] = 1.0 / legs
    w[ranks > n - legs] = -1.0 / legs
    w = w.where(s.notna().sum(axis=1) >= 2 * legs, 0.0)
    # a ranking is kept for `hold` bars before it may change
    keep = np.zeros(len(w), dtype=bool)
    keep[::hold] = True
    w = w.where(pd.Series(keep, index=w.index), np.nan).ffill().fillna(0.0)

    turnover = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    gross = (w * r).sum(axis=1)
    fund = FUNDING_PER_8H * (minutes / 480.0) * w.abs().sum(axis=1)
    return (gross - turnover * (FEE_ROUND_TRIP / 2.0) - fund).values[:-1]


def leave_one_out(p: pd.DataFrame, F_of, legs: int, hold: int, minutes: int,
                  fname: str, rev: bool) -> tuple[str, float]:
    """Re-run the book with each symbol removed. The worst result is the
    number that matters.

    A cross-sectional book is supposed to be a book. If dropping one
    symbol collapses it, then the other symbols were decoration and the
    effective sample size is one trade, not one per bar -- however many
    bars there were.

    This is not hypothetical. Every positive book found on this week
    died here: volr30 went +101.9% to -7.5%, madist240 +149.0% to +3.4%
    and mom240 +130.6% to -7.9% when BLESSUSDT was removed, while
    dropping any OTHER symbol changed almost nothing.
    """
    worst_sym, worst = "", float("inf")
    for drop in p.columns:
        q = p.drop(columns=[drop])
        if q.shape[1] < 2 * legs + 1:
            continue
        t = stat(run_book(q, F_of(q)[fname], legs, hold, minutes, rev),
                 minutes)[0]
        if t < worst:
            worst, worst_sym = t, drop
    return worst_sym, worst


def stat(x: np.ndarray, minutes: int) -> tuple[float, float, float]:
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return 0.0, 0.0, 0.0
    per_year = 365 * 24 * 60 / minutes
    total = float(np.prod(1 + x) - 1)
    sh = float(x.mean() / x.std() * np.sqrt(per_year)) if x.std() > 0 else 0.0
    return total, sh, float(x.mean())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tf", default="5m,15m,30m")
    ap.add_argument("--legs", type=int, default=2)
    ap.add_argument("--null-runs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=41)
    a = ap.parse_args(argv)

    coins = load_coins()
    rng = np.random.default_rng(a.seed)
    print(f"{len(coins)} symbols, long the top {a.legs} and short the "
          f"bottom {a.legs}, equal money each side")
    print("no market view is needed: the P&L is the spread between the "
          "legs\n")

    best = []
    for label in a.tf.split(","):
        label = label.strip()
        if label not in TF:
            continue
        minutes = TF[label]
        p = panel(coins, minutes)
        if len(p) < 300:
            print(f"{label}: only {len(p)} shared bars, skipped")
            continue
        F = factors(p)
        print("=" * 78)
        print(f"{label}: {len(p)} shared bars across {p.shape[1]} symbols")
        print("=" * 78)
        print(f"{'factor':>10} {'hold':>6} {'dir':>8} {'total':>9} "
              f"{'Sharpe':>8} {'per bar':>9} {'null p':>8} {'w/o worst':>10} "
              f"{'that coin':>13}")
        for fname, S in F.items():
            for hold in HOLDS:
                for rev in (False, True):
                    x = run_book(p, S, a.legs, hold, minutes, rev)
                    total, sh, mean = stat(x, minutes)
                    if total <= 0:
                        continue
                    # rotate each coin's score independently
                    null = []
                    for _ in range(a.null_runs):
                        R = S.copy()
                        for c in R.columns:
                            R[c] = np.roll(R[c].values,
                                           int(rng.integers(1, len(R))))
                        null.append(stat(run_book(p, R, a.legs, hold,
                                                  minutes, rev), minutes)[0])
                    null = np.array(null)
                    pv = float((null >= total).mean())
                    if pv > 0.10:
                        continue
                    sym, worst = leave_one_out(p, factors, a.legs, hold,
                                               minutes, fname, rev)
                    print(f"{fname:>10} {hold:>6} "
                          f"{'reverse' if rev else 'follow':>8} "
                          f"{100*total:>8.2f}% {sh:>8.2f} "
                          f"{100*mean:>8.4f}% {pv:>8.3f} "
                          f"{100*worst:>9.2f}% {sym:>13}")
                    best.append({"tf": label, "factor": fname,
                                 "hold": hold, "reverse": rev,
                                 "total": total, "sharpe": sh, "p": pv,
                                 "worst_loo": worst, "worst_sym": sym})
        if not best:
            print("  nothing positive after fees at this timeframe")
        print()

    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    if best:
        strong = [b for b in best if b["p"] <= 0.05]
        survive = [b for b in best if b["worst_loo"] > 0]
        print(f"  {len(best)} books were positive after fees and beat the "
              f"rotation null at p<=0.10")
        print(f"  {len(strong)} of those reached p<=0.05")
        print(f"  {len(survive)} of those still made money with their worst "
              f"single symbol removed")
        for b in sorted(survive, key=lambda x: -x["worst_loo"])[:10]:
            print(f"    {b['tf']:>4} {b['factor']:<9} hold {b['hold']:>3} "
                  f"{'reverse' if b['reverse'] else 'follow':>8} "
                  f"{100*b['total']:+.2f}% -> {100*b['worst_loo']:+.2f}% "
                  f"without {b['worst_sym']}")
        if not survive:
            print("\n  None of them survived leave-one-out. Every positive")
            print("  book on this week was one symbol wearing nine names --")
            print("  the effective sample size is 1, not 1,638 bars.")
        print("\n  Five days is five days. This says the ranking carried")
        print("  information in THIS week, not that it will next week.")
    else:
        print("  No cross-sectional book cleared its own fees.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
