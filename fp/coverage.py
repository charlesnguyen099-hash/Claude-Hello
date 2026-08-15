"""Does the logic cover every profitable chance? Yes. It does not help.

The operator's instruction was to separate two things that had been
conflated, and the separation is right:

  BUILD TIME     the logic must cover EVERY trade that clears +1% net
  TRADE TIME     the bot holds one position per coin

So this measures coverage at build time, with the execution rule set
aside. A ">1% chance" is defined by the market, not by any strategy: at
bar i, on side s, does SOME exit within the next day return more than +1%
after taker in, taker out and funding? That is a property of the price
path and it does not care what indicators exist.

    RESULT, 1,763,304 bars across 10 coins:

      >1% chances      2,067,284
      covered          2,067,267
      COVERAGE            100.0%

Coverage is already total. Asking for more is asking for something the
library reached before the question was posed.

AND IT IS WORTH NOTHING, which is the point of this file. Coverage is
total because at 100.0% of bars SOME strategy is long and SOME strategy
is short. A library that always says both always covers whatever happens.
The number that matters is not whether a covering signal exists, it is
whether the covering signals LEAN the right way:

      when a >1% LONG existed,  49.7% of firing strategies said long
      when a >1% SHORT existed, 50.5% of firing strategies said short
      a coin flip is             50.0%
      measured discrimination    +0.13 points

WHAT WOULD BE NEEDED. On a roughly symmetric +/-1% outcome paying a
0.110% round trip, break-even direction accuracy is

      p * 1% - (1-p) * 1% - 0.110% > 0   ->   p > 55.50%

So the gap is +5.50 points and the measurement is +0.13. Not a shortfall
of degree -- a shortfall of about forty times, across 47 methods, 623
combinations and nineteen months of BTC.

This is why "catch all 2.5 million" cannot be delivered by adding logic.
Every one of those chances is already covered. What is missing is any
way to tell, at the moment of entry, which of the two directions being
signalled is the one that pays.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from fp import data as D
from fp import strategies as SG
from fp.data import FEE, FUNDING_PER_8H

# A chance is judged over a day. Holds are unbounded in this system, so a
# longer window only makes coverage look better; a shorter one is the
# harsher test and still comes out at 100%.
HORIZON = 1440
WIN = 0.01


def chances(close: np.ndarray, horizon: int = HORIZON, win: float = WIN):
    """(long_ok, short_ok): where a >win net move was available."""
    s = pd.Series(close)
    fmax = s[::-1].rolling(horizon, min_periods=1).max()[::-1].values
    fmin = s[::-1].rolling(horizon, min_periods=1).min()[::-1].values
    cost = FEE + horizon / 60.0 / 8.0 * FUNDING_PER_8H
    return ((fmax / close - 1.0) - cost) > win, \
           ((1.0 - fmin / close) - cost) > win


def main():
    print("=" * 78)
    print("  COVERAGE: does the logic reach every >1% chance in the market?")
    print("=" * 78)
    P = D.load()
    print(f"  {'coin':<12} {'bars':>9} {'>1% chances':>12} {'covered':>10} "
          f"{'cover':>7} {'lean':>7}")
    to = tc = tb = 0
    leans = []
    for sym, d in sorted(P.items()):
        c = d["close"].values.astype("float64")
        n = len(c)
        ol, os_ = chances(c)
        S = SG.all_strategies(d, P, sym)
        A = np.vstack(list(S.values()))
        on_l = (A == 1).any(axis=0)
        on_s = (A == -1).any(axis=0)
        nL = (A == 1).sum(axis=0)
        nS = (A == -1).sum(axis=0)
        tot = np.maximum(nL + nS, 1)
        lean = ((nL / tot)[ol].mean() + (nS / tot)[os_].mean()) / 2 - 0.5
        cov = int((ol & on_l).sum() + (os_ & on_s).sum())
        opp = int(ol.sum() + os_.sum())
        to += opp; tc += cov; tb += n
        leans.append(lean)
        print(f"  {sym:<12} {n:>9,} {opp:>12,} {cov:>10,} "
              f"{100*cov/max(opp,1):>6.1f}% {100*lean:>+6.2f}%")
    lean = float(np.mean(leans))
    print(f"\n  {'TOTAL':<12} {tb:>9,} {to:>12,} {tc:>10,} "
          f"{100*tc/max(to,1):>6.1f}% {100*lean:>+6.2f}%")
    need = 0.5 + FEE / 0.02          # p*1% - (1-p)*1% - FEE > 0
    print(f"\n  COVERAGE   : {100*tc/max(to,1):.1f}% -- essentially total, and")
    print(f"               essentially meaningless: both directions are")
    print(f"               signalled at almost every bar, so a covering")
    print(f"               signal always exists whatever happens next.")
    print(f"  LEAN       : {100*lean:+.2f} points above a coin flip.")
    print(f"  NEEDED     : {100*(need-0.5):+.2f} points, to clear a "
          f"{100*FEE:.3f}% round trip")
    print(f"               on a +/-1% outcome ({100*need:.2f}% accuracy).")
    print(f"  VERDICT    : the gap is direction, not coverage. Adding logic")
    print(f"               cannot close it -- every chance is already covered.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
