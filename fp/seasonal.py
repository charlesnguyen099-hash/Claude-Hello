"""The clock as a SIGNAL -- the one input no logic here has ever seen.

    python -m fp.seasonal
    python -m fp.seasonal --horizons 15,60,240

THE GAP THIS FILLS

Every one of the 34,000 logics tested so far reads price, volume and
their derivatives. Not one reads the calendar. A factor library cannot
find an effect it has no column for, so if BTCUSDT has an edge that
lives at a particular hour, every study in this repo would have missed
it completely -- and "missed" is different from "tested and rejected".

There are concrete reasons to look. Perpetual funding settles at 00:00,
08:00 and 16:00 UTC, and positions are opened and closed around those
settlements for reasons that have nothing to do with direction. The
Asian, European and US sessions hand over at fixed hours. Weekend
liquidity is thinner than weekday liquidity. None of that is a theory
about where price is going; it is structure in when people trade, and
structure in when is exactly what a calendar column can see.

WHAT IS TESTED

Forward returns over several horizons, bucketed three ways:

    hour        the 24 hours of the UTC day
    weekday     the 7 days of the week
    funding     hours until the next 8-hourly settlement, 0 through 7

Each bucket is a candidate trade: be long in it, or short in it, for the
horizon in question. The bar is the same as everywhere else -- net of
0.055% each way plus funding, an in-sample half that must show a
positive lower bound, an out-of-sample half that must clear a Bonferroni
threshold for every bucket x horizon x side tested.

Overlapping windows are the trap here: sampling every minute makes 1,440
observations a day that are almost the same trade, which inflates any
t-stat by roughly the square root of the overlap. So returns are sampled
NON-OVERLAPPING at each horizon -- one observation per horizon-length
block -- and the count of independent observations is reported next to
every result.

WHAT IT MEASURED

204 bucket x horizon x side candidates, Bonferroni bar |t| > 3.67. One
cleared the in-sample bar; zero cleared out of sample.

    horizon   bucket  value   side      n   IS mean  IS lower  OOS mean   OOS t
       1440  weekday      2   long     42   0.6283%   0.0350%  -0.2940%   -0.84
        240  weekday      2   long    252   0.0129%  -0.0872%  -0.1397%   -2.20
         60  weekday      2   long   1008  -0.0794%  -0.1034%  -0.1174%   -7.43
         15  weekday      2   long   4032  -0.1023%  -0.1086%  -0.1118%  -27.71
         15     hour     17   long   1164  -0.0960%  -0.1106%  -0.1172%  -13.81
         15  funding      7   long   3492  -0.1036%  -0.1108%  -0.1120%  -25.95

Read the 15-minute rows. Every bucket -- every hour, every weekday,
every funding offset, long or short -- returns about -0.11% per trade.
That IS the round trip. Over fifteen minutes the calendar carries
exactly nothing, and the whole number is the fee being paid.

The single in-sample qualifier is Wednesday held for a day: 42 non-
overlapping observations, +0.628% in sample and -0.294% out of it. Forty
-two observations is what a calendar effect on nineteen months of data
can offer, and it is not enough to establish anything.

So the calendar is not an untested hypothesis any more. It was the one
column no logic in this repo could read, it has been read, and it is
empty.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from fp.horizon import FEE_ROUND_TRIP, FUNDING_PER_8H, load_1m
from fp.survivors import POTENTIAL_Z, bonferroni_t, tstat

HORIZONS = (15, 60, 240, 1440)          # minutes


def buckets(idx: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """The three calendar labels, none of which any factor logic sees."""
    return {
        "hour": idx.hour.values,
        "weekday": idx.dayofweek.values,
        # hours until 00:00, 08:00 or 16:00 UTC, the funding settlements
        "funding": (8 - (idx.hour.values % 8)) % 8,
    }


def sample(close: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Non-overlapping forward returns, and the index each one starts at.

    Overlapping windows would give 1,440 near-identical observations a
    day and inflate every t-stat by about sqrt(horizon). One block per
    horizon length is the honest sample size.
    """
    starts = np.arange(0, len(close) - horizon, horizon)
    r = close[starts + horizon] / close[starts] - 1.0
    return r, starts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--horizons", default=",".join(str(h) for h in HORIZONS))
    ap.add_argument("--min-obs", type=int, default=30)
    a = ap.parse_args(argv)

    d = load_1m()
    close = d["close"].values.astype(float)
    B = buckets(d.index)
    print(f"{len(d):,} one-minute bars of BTCUSDT, "
          f"{d.index[0]} .. {d.index[-1]}")
    print("returns sampled NON-OVERLAPPING, so the observation count is real\n")

    rows = []
    for horizon in (int(x) for x in a.horizons.split(",")):
        r, starts = sample(close, horizon)
        cost = FEE_ROUND_TRIP + FUNDING_PER_8H * (horizon / 480.0)
        split = len(r) // 2
        for bname, labels in B.items():
            lab = labels[starts]
            for value in np.unique(lab):
                m = lab == value
                for side in (1, -1):
                    net = side * r[m] - cost
                    a_, b_ = net[:int(m[:len(r)].cumsum()[split - 1])], None
                    k = int(m[:split].sum())
                    a_, b_ = net[:k], net[k:]
                    if len(a_) < a.min_obs or len(b_) < a.min_obs:
                        continue
                    lo = (a_.mean() - POTENTIAL_Z * a_.std(ddof=1)
                          / np.sqrt(len(a_)))
                    rows.append({
                        "horizon": horizon, "bucket": bname, "value": int(value),
                        "side": "long" if side > 0 else "short",
                        "is_n": len(a_), "is_mean": float(a_.mean()),
                        "is_lo": float(lo),
                        "oos_n": len(b_), "oos_mean": float(b_.mean()),
                        "oos_t": tstat(b_),
                        "all_mean": float(net.mean()), "all_t": tstat(net),
                    })
    R = pd.DataFrame(rows)
    if R.empty:
        print("no bucket had enough non-overlapping observations")
        return 0

    thr = bonferroni_t(len(R))
    print(f"{len(R):,} bucket x horizon x side candidates tested, "
          f"Bonferroni bar |t| > {thr:.2f}\n")

    print("STRONGEST CALENDAR EFFECTS, in sample")
    print(f"{'horizon':>8} {'bucket':>9} {'value':>6} {'side':>6} {'n':>6} "
          f"{'IS mean':>9} {'IS lower':>9} {'OOS mean':>9} {'OOS t':>7}")
    for _, x in R.sort_values("is_lo", ascending=False).head(12).iterrows():
        print(f"{x.horizon:>8} {x.bucket:>9} {x.value:>6} {x.side:>6} "
              f"{x.is_n:>6} {100*x.is_mean:>8.4f}% {100*x.is_lo:>8.4f}% "
              f"{100*x.oos_mean:>8.4f}% {x.oos_t:>7.2f}")

    gated = R[R["is_lo"] > 0]
    surv = gated[gated["oos_t"] > thr]
    print(f"\n  {len(gated)} cleared the in-sample bar, "
          f"{len(surv)} then cleared |t| > {thr:.2f} out of sample")
    if len(surv):
        for _, x in surv.sort_values("oos_t", ascending=False).iterrows():
            print(f"    {x.bucket}={x.value} {x.side} over {x.horizon}m: "
                  f"{100*x.oos_mean:+.4f}%/trade, t={x.oos_t:.2f}, "
                  f"n={x.oos_n}")
    else:
        best = R.loc[R["oos_t"].idxmax()]
        print(f"    none. best out of sample anywhere: {best.bucket}="
              f"{best.value} {best.side} over {best.horizon}m at "
              f"t={best.oos_t:.2f}")

    print("\n" + "=" * 74)
    print("The calendar was the one column no logic in this repo could read.")
    print("It has now been read.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
