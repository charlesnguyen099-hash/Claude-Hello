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

WHAT IT MEASURED

Sweeping nine state definitions x four basket sizes, two are positive at
EVERY basket size -- parameter stability, not a lone spike:

    states        top5    top15    top50   top200
    vol,trend   +37.6%   +23.3%   +20.0%   +24.9%
    trend,dir   +43.0%   +53.9%   +35.9%   +22.0%
    vol,dir      -5.8%   -16.5%   -10.8%    -9.1%

Choosing a state definition after seeing that table is a leak, so it was
redone with the definition itself picked from past data only. Both splits
independently chose `trend`, and both were positive forward:

    picked on            chose         traded on            result  Sharpe
    2025-10..2025-12     trend top5    2025-12..2026-08     +78.1%    3.00
    2025-10..2026-03     trend top5    2026-03..2026-08     +40.1%    4.18

First and second half separately, `trend` is the only definition positive
in both (+4.5%, +69.2%).

AND THE LIMIT OF THAT EVIDENCE, WHICH IS THE IMPORTANT PART

Broken out by period, it pays when the market falls and does nothing when
it rises:

    2025 Q4    BTC -22.4%    strategy  +7.1%   Sharpe  1.59
    2026 Q1    BTC -23.2%    strategy +26.6%   Sharpe  2.38
    2026 Q3    BTC  +7.8%    strategy  -0.3%   Sharpe -0.16

That is trend-following working in a trend, which is its known property
rather than a discovery. The warm-up leaves usable history only from late
2025, so essentially all of the evidence comes from ONE sustained decline.
The single rising quarter available is 36 days long and returns nothing.

So: the first out-of-sample-positive result in this project, produced by
one regime, and untested against a rising or choppy market because the
data barely contains one. Treat the size of the number as a property of
that decline, not as an expectation.

AND THEN THE RISING MARKET WAS TESTED ANYWAY -- see fp/symmetry.py

The data contains no sustained rise, so one was built: every daily log
return reflected, r -> -r, which turns the 31.6% fall into a 46.2% rise
while preserving volatility, clustering and range structure exactly.
Run unchanged on that series the same procedure returns +78.7% at
Sharpe 2.47, and it gets there by being LONG 121 days rather than short
122. The weight moves to the other side on its own, because the factors
change sign and the state labels follow them.

On the real series both sides already paid -- short 122 days +55.1%,
long 65 days +20.1% at Sharpe 5.67, the best bucket in the table, taken
during the year BTC fell a third. This is not a short with extra steps.

It also reaches short swings: median hold is TWO days and 65% of
positions last one to three, so a two-day drop is inside what it trades
rather than something it waits out.

HOW MANY LOGICS THE DATA CAN ACTUALLY CARRY

The natural next step is to slice the state space finer -- many narrow
states, each with its own specialist logic. That was measured, sweeping
from two states to a hundred and forty:

    state dimensions            states   days traded    total   Sharpe
    trend (k=2)                      2           274    +89.5%     2.72
    trend (k=3)                      3           247   +123.6%     3.71
    trend,dir (k=3)                  8           136    +27.4%     2.01
    vol,trend,dir (k=3)             24            50     +5.2%     3.27
    vol,trend,dir,mom (k=3)         46            21     +4.7%     6.55
    +pos (k=3)                      70             7     -0.7%    -4.90
    +volr (k=3)                    140             0        --       --

The binding constraint is not how much code can be written. It is that
there are 583 daily bars. Cut them into 140 states and each holds about
four days of history -- below any threshold at which a logic could be
judged -- so nothing trades at all. At 70 states, seven days qualify in
nineteen months.

More logics do not cover more signals here; finer states cover fewer,
because coverage is bounded by samples per state and the samples are
fixed. The returns above shrink monotonically with granularity for that
reason, not because the narrow logics are worse.

AND THE STATES WHERE NOTHING WORKS DO NOT EXIST

The other half of the idea -- where no logic is profitable, invert the
losing ones instead of dropping the state -- was implemented and never
fired. Across every configuration above, zero inversions.

The reason is worth stating: with 2,602 logics in hand, the five best by
in-state Sharpe have a positive in-state mean in EVERY state, always.
There is no state without an apparent winner. That is not evidence the
states are all tradeable; it is the overfitting warning in its clearest
form -- a large enough logic library guarantees an in-sample winner
inside any slice you draw, including slices that are pure noise.

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
        S = states(d, combo.split(","))
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
