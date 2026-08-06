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
              f"{'Sharpe':>8} {'per bar':>9} {'null p':>8}")
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
                    if pv <= 0.10:
                        print(f"{fname:>10} {hold:>6} "
                              f"{'reverse' if rev else 'follow':>8} "
                              f"{100*total:>8.2f}% {sh:>8.2f} "
                              f"{100*mean:>8.4f}% {pv:>8.3f}")
                        best.append({"tf": label, "factor": fname,
                                     "hold": hold, "reverse": rev,
                                     "total": total, "sharpe": sh, "p": pv})
        if not best:
            print("  nothing positive after fees at this timeframe")
        print()

    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    if best:
        strong = [b for b in best if b["p"] <= 0.05]
        print(f"  {len(best)} factor/hold/direction books were positive after "
              f"fees, {len(strong)} of them beat the rotation null at p<=0.05")
        for b in sorted(strong, key=lambda x: -x["total"])[:10]:
            print(f"    {b['tf']:>4} {b['factor']:<9} hold {b['hold']:>3} "
                  f"{'reverse' if b['reverse'] else 'follow':>8} "
                  f"{100*b['total']:+.2f}% Sharpe {b['sharpe']:.2f} "
                  f"p={b['p']:.3f}")
        print("\n  Five days is five days. This says the ranking carried")
        print("  information in THIS week, not that it will next week.")
    else:
        print("  No cross-sectional book cleared its own fees.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
