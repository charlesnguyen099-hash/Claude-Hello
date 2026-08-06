"""Pick the logic by what the market is DOING, not by what recently paid.

    python -m fp.regime
    python -m fp.regime --states vol,trend --top 15

THE CORRECTION THIS MAKES

fp/ensemble.py scored 2,602 logics by their trailing P&L and traded the
leaders. It lost in 15 of 16 settings. But trailing P&L is a lagging and
noisy proxy for the thing that actually decides whether a logic works:
the state the market is in. A mean-reversion logic pays in a choppy
regime and bleeds in a trending one, and by the time its P&L reflects
that, the regime has often turned.

So this conditions directly on the state instead. At each bar the market
is classified -- volatility band, trend strength, and how far price sits
in its own range -- and each logic is scored ONLY on the past days that
shared the current state. The logic that has historically worked in THIS
state gets traded now, whatever it did last month.

That is the difference between "what worked recently" and "what works
here", and it is the version of the idea worth testing.

HOW THE STATES ARE BUILT

Each dimension is cut at rolling percentiles of its own past, so a state
label means the same thing in 2025 and 2026 and nothing is standardised
using data from the future:

    vol      realised volatility over 30d, low / mid / high
    trend    |price / MA50 - 1|, weak / strong
    pos      where price sits in its 90d range, low / mid / high
    dir      price above or below its MA50

Crossed, that is up to 36 states. A state with too little history is not
traded rather than guessed at.

WHAT IT MEASURED, AND THE CORRECTION THAT REPLACED IT

This module once reported the only out-of-sample-positive result in the
project: `trend` top5 at +89.5%, Sharpe 2.72, chosen twice independently
by nested validation, +78.1% and +40.1% forward. All of it came from a
one-bar misalignment.

states() labels bar i from bar i's own close. The return the backtest
credited to bar i is close[i]/close[i-1] - 1, driven by that same close.
So conditioning the choice of logic on the unlagged label let the choice
see part of the outcome it was about to collect. With 2,602 logics to
choose from, that sliver was enough to manufacture the entire result.

Lagging the label one bar -- which is what the live bot does anyway,
since it reads the last CLOSED bar and holds through the next one --
turns every cell of the sweep negative:

    states            top5    top15    top50   top200      before (top5)
    vol             -14.2%    -9.2%   -16.1%   -15.9%              +4.6%
    trend           -21.5%   -18.9%   -22.4%   -19.8%             +89.5%
    dir             -19.7%   -23.6%   -21.6%   -16.9%             +41.9%
    vol,trend       -17.4%   -10.3%   -15.0%   -13.6%             +37.6%
    vol,dir         -15.0%   -21.2%   -20.2%   -20.4%              -5.8%
    trend,dir       -14.8%   -15.9%   -15.4%    -7.2%             +43.0%
    vol,trend,dir   -23.6%   -16.8%   -18.5%   -14.6%              +4.4%
    vol,pos,dir     -13.1%   -15.0%   -18.5%   -17.2%              +8.5%
    vol,trend,pos   -20.3%   -14.4%   -15.8%   -12.1%              +5.0%

36 cells, 36 negative. Sharpe runs from -0.30 to -5.97. The granularity
sweep and the mirror study built on top of this result inherit the same
correction; their numbers are void, and the shape of the conclusion they
drew -- that coverage bounds how many logics the data can carry, and
that a large library always finds an in-sample winner -- survives, since
neither depended on the sign.

Every measurement now goes through lagged_states() so this cannot recur
in one caller and not another, and fp/test_bot.py fails if the lag is
removed.

WHAT REMAINS TRUE

State conditioning is still a better idea than trailing P&L -- it just
does not carry any edge here either. Both are now measured and both are
negative, which is the same answer arrived at twice.

EVERYTHING ELSE IS AS ESTABLISHED

Daily bars, positions held until the logic flips, returns measured entry
to exit without rebalancing, leverage solved with volatility drag in it.
No logic is ever scored on a day it later trades.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from fp.ensemble import FUND_DAY, build_logics, daily_net, load_daily


def states(d: pd.DataFrame, which: list[str]) -> pd.Series:
    """A label per bar, from rolling percentiles of the bar's own past."""
    c = d["close"]
    r = c.pct_change()
    parts = []
    if "vol" in which:
        v = r.rolling(30).std()
        q = v.rolling(250, min_periods=120).rank(pct=True)
        parts.append(pd.cut(q, [-.01, .33, .66, 1.01], labels=["v0", "v1", "v2"]))
    if "trend" in which:
        t = (c / c.rolling(50).mean() - 1).abs()
        q = t.rolling(250, min_periods=120).rank(pct=True)
        parts.append(pd.cut(q, [-.01, .5, 1.01], labels=["t0", "t1"]))
    if "pos" in which:
        hi = d["high"].rolling(90).max()
        lo = d["low"].rolling(90).min()
        p = ((c - lo) / (hi - lo)).where(hi > lo, 0.5)
        parts.append(pd.cut(p, [-.01, .33, .66, 1.01], labels=["p0", "p1", "p2"]))
    if "dir" in which:
        parts.append(pd.Series(np.where(c > c.rolling(50).mean(), "d1", "d0"),
                               index=c.index))
    if not parts:
        return pd.Series("all", index=c.index)
    out = parts[0].astype(str)
    for p in parts[1:]:
        out = out + "_" + p.astype(str)
    return out.where(~out.str.contains("nan"), np.nan)


def lagged_states(d: pd.DataFrame, which: list[str]) -> pd.Series:
    """The only state label a backtest may condition on.

    states() labels each bar from that bar's own close. The return a
    backtest credits to bar i is driven by that same close, so choosing
    a logic inside the unlagged label lets the choice see part of the
    outcome it is about to collect. That single-bar misalignment was
    worth the entire result this module once reported: `trend` top5 read
    +89.5% with it and -21.5% without.

    Live there is no such shift to make -- the bot reads the last CLOSED
    bar and holds through the next one, which is what this reproduces.
    Every measurement goes through here so the mistake cannot recur in
    one caller and not another.
    """
    return states(d, which).shift(1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--states", default="vol,trend,dir")
    ap.add_argument("--min-days", type=int, default=40,
                    help="history a state needs before it is traded")
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args(argv)

    d = load_daily()
    close = d["close"]
    logics = build_logics(d)
    N = pd.DataFrame({k: daily_net(close, v) for k, v in logics.items()})
    print(f"{len(d):,} daily bars, {N.shape[1]:,} logics")

    combos = ([a.states] if not a.sweep else
              ["vol", "trend", "dir", "vol,trend", "vol,dir", "trend,dir",
               "vol,trend,dir", "vol,pos,dir", "vol,trend,pos,dir"])
    tops = [a.top] if not a.sweep else [5, 15, 50, 200]

    print(f"\n{'states':>22} {'n':>4} {'top':>5} {'traded':>7} {'total':>9} "
          f"{'Sharpe':>8} {'win days':>9}")
    for combo in combos:
        S = lagged_states(d, combo.split(","))
        for top in tops:
            eq, rets, traded = 1.0, [], 0
            for i in range(a.warmup, len(close) - 1):
                st = S.iloc[i]
                if not isinstance(st, str):
                    continue
                # only PAST bars, and only those sharing today's state
                past = S.iloc[:i]
                same = np.flatnonzero((past == st).values)
                if len(same) < a.min_days:
                    continue
                hist = N.iloc[same]
                score = hist.mean() / hist.std().replace(0, np.nan)
                score = score.dropna()
                if score.empty:
                    continue
                pick = score.nlargest(top).index
                step = float(N.iloc[i][pick].mean())
                if np.isfinite(step):
                    eq *= (1 + step)
                    rets.append(step)
                    traded += 1
            if not rets:
                print(f"{combo:>22} {len(S.dropna().unique()):>4} {top:>5} "
                      f"{'--':>7}")
                continue
            r = np.array(rets)
            sh = r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else 0.0
            print(f"{combo:>22} {len(S.dropna().unique()):>4} {top:>5} "
                  f"{traded:>7} {100*(eq-1):>8.1f}% {sh:>8.2f} "
                  f"{100*(r > 0).mean():>8.1f}%")

    print("\n" + "=" * 74)
    print("Positive here means the state label predicts which logic pays,")
    print("which trailing P&L did not. Negative means the state does not")
    print("carry that information either.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
