"""Does the state logic trade BOTH directions, and does it catch short swings?

    python -m fp.symmetry

THE TWO CLAIMS BEING TESTED

fp/regime.py earned +123.6% over a period in which BTC fell 31.6%. Two
things have to be true before that is a strategy rather than a short:

    1. it must go LONG and be paid for it, not only short
    2. an up-market must work the same way with the factors reversed

Both are measurable on the data in hand, and the second is measurable
even though the data contains no sustained rise -- by mirroring it.

HOW THE MIRROR WORKS

Reflect every daily log return, r -> -r, and rebuild the bars from it.
High and low swap (in log space the mirror of the high IS the low), so
the synthetic series has the same volatility, the same clustering, the
same range structure and the same autocorrelation -- and rises where the
real one fell. A 31.6% fall becomes a 46.2% rise.

If the logic is symmetric, it must earn on the mirror too. If it earns
only on the real series, it is a short dressed as a strategy, and the
mirror is the only way to find that out before a bull market does.

Fees and funding are charged the same on both sides here, which is the
conservative choice: in reality a short in a positive-funding market
collects funding rather than paying it, so the real series is being
handicapped relative to the mirror, not flattered.

One thing the reflection cannot preserve exactly: log volatility is
identical by construction, but simple-return volatility is not, because
exp(-x)-1 is not the negative of exp(x)-1. At daily volatility the gap
is a few tenths of a percent, relative -- small enough not to move any
conclusion below, and stated rather than glossed.

WHAT THE HOLDING-PERIOD TEST ANSWERS

A logic that flips every thirty days cannot take a three-day drop. The
run-length distribution of the consensus position says directly which
swings are reachable, and the swing table says what was actually
collected from the short ones.

WHAT IT MEASURED

Both sides are traded, and both sides are paid:

                buy & hold   strategy   Sharpe    LONG days      SHORT days
    real            -31.6%     +89.5%     2.72   65  +20.1%   122  +55.1%
    mirror          +46.2%     +78.7%     2.47  121  +61.5%    49   +6.5%

Long alone on the real series returns +20.1% at Sharpe 5.67 over 65 days
-- the highest Sharpe of any bucket in the table, inside the year BTC
fell a third. The strategy is not a short with extra steps.

The mirror settles the up-market question the data could not: run on a
market that rises 46.2%, the same procedure returns +78.7% and does it
by being long 121 days instead of short 122. The weight moves to the
other side by itself, because the factors reverse sign and the state
labels follow them. That is the symmetry claim, confirmed.

It reaches short swings. Median hold is TWO days, and 65% of positions
last one to three days:

    hold        1-3d    4-7d   8-20d    21+d
    positions    24       7       4       2      (real, 37 positions)

SO A FEW DOWN DAYS IS EXACTLY WHAT IT TRADES, and the return breakdown
by how long the move had already run before entry says the same:

    run before entry    real          mirror
    1 day              +43.9%        +37.6%
    2-3 days           +62.7%        +57.8%
    4-6 days           -14.1%        -15.2%
    7+ days             -5.2%         -2.4%

Fresh and few-day-old moves carry everything; moves already four days
old lose on both series. Note this table is lagged by a day on purpose
-- labelling a day by its own return would use the outcome to pick the
trade, and the unlagged version says the opposite, which is exactly how
that mistake looks from the inside.

THE FILTER THAT LOOKS OBVIOUS AND IS NOT WIRED IN

Dropping entries into moves older than three days lifts the real series
to +132.6% at Sharpe 4.09 and the mirror to +115.7% at 3.66. Half by
half, it does not hold:

                                   first half   second half
    unfiltered                          33.4%         42.0%
    skip stale moves (run > 3d)         63.6%         42.1%

All of the gain is in the first half; in the second it is worth nothing.
That is 34 excluded days carrying the whole effect, which is a fit, and
the mirror agreeing does not make it independent evidence -- the mirror
is the same 583 days reflected, not a new sample. So it stays measured
and unwired until a period of fresh data either repeats it or does not.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from fp.ensemble import build_logics, daily_net, load_daily
from fp.regime import states


# ------------------------------------------------------------------ mirror

def mirror(d: pd.DataFrame) -> pd.DataFrame:
    """Same bars with every return reflected: falls become rises.

    In log space the reflection of a bar's high is its low, so the ratios
    are swapped rather than the levels negated. Volatility, range width
    and clustering are preserved exactly; only the sign of the drift and
    of every return changes.
    """
    c = d["close"].astype(float)
    lr = np.log(c).diff().fillna(0.0)
    cm = float(c.iloc[0]) * np.exp((-lr).cumsum())
    return pd.DataFrame({
        "open": cm * (c / d["open"]),
        "high": cm * (c / d["low"]),      # the high of the mirror is the low
        "low": cm * (c / d["high"]),
        "close": cm,
        "volume": d["volume"],
    }, index=d.index)


# ---------------------------------------------------------------- backtest

def run(d: pd.DataFrame, state_dims=("trend",), top=5, min_days=40,
        warmup=300) -> pd.DataFrame:
    """The fp.regime procedure, returning the per-day record it produced.

    Identical selection rule -- score every logic on the past days that
    shared today's state, hold the best `top`. The extra column is the
    consensus direction, which is what the bot actually trades and what
    the symmetry question is about.
    """
    close = d["close"]
    logics = build_logics(d)
    keys = list(logics)
    N = np.column_stack([daily_net(close, logics[k]).values for k in keys])
    P = np.column_stack([logics[k].values for k in keys])
    S = states(d, list(state_dims))
    sv = S.values
    mkt = close.pct_change().values

    rows = []
    for i in range(warmup, len(close) - 1):
        st = sv[i]
        if not isinstance(st, str):
            continue
        same = np.flatnonzero(sv[:i] == st)
        if len(same) < min_days:
            continue
        h = N[same]
        sd = h.std(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            score = np.where(sd > 0, h.mean(axis=0) / sd, np.nan)
        if not np.isfinite(score).any():
            continue
        pick = np.argsort(np.where(np.isfinite(score), score, -np.inf))[-top:]
        step = float(np.nanmean(N[i, pick]))
        # the position that earned it was set at the close of i-1
        vote = float(np.nanmean(P[i - 1, pick]))
        if not np.isfinite(step):
            continue
        rows.append({"date": close.index[i], "net": step, "vote": vote,
                     "dir": 1 if vote > 0.2 else (-1 if vote < -0.2 else 0),
                     "mkt": float(mkt[i]) if np.isfinite(mkt[i]) else 0.0})
    return pd.DataFrame(rows).set_index("date")


def stat(r: np.ndarray) -> tuple[float, float]:
    if len(r) == 0:
        return 0.0, 0.0
    tot = float(np.prod(1 + r) - 1)
    sh = float(r.mean() / r.std() * np.sqrt(365)) if r.std() > 0 else 0.0
    return tot, sh


def runs(dirs: np.ndarray) -> list[tuple[int, int]]:
    """(direction, length) for each unbroken stretch of the same position."""
    out, i = [], 0
    while i < len(dirs):
        j = i
        while j + 1 < len(dirs) and dirs[j + 1] == dirs[i]:
            j += 1
        out.append((int(dirs[i]), j - i + 1))
        i = j + 1
    return out


# -------------------------------------------------------------------- report

def by_direction(name: str, R: pd.DataFrame, bh: float) -> None:
    tot, sh = stat(R["net"].values)
    print(f"\n{name}   buy & hold {100*bh:+.1f}%   strategy {100*tot:+.1f}% "
          f"Sharpe {sh:.2f}   {len(R)} days traded")
    print(f"  {'side':>6} {'days':>6} {'total':>9} {'Sharpe':>8} "
          f"{'win days':>9} {'mean/day':>9}")
    for d_, label in ((1, "LONG"), (-1, "SHORT"), (0, "flat")):
        seg = R[R["dir"] == d_]["net"].values
        if len(seg) == 0:
            print(f"  {label:>6} {0:>6}")
            continue
        t, s = stat(seg)
        print(f"  {label:>6} {len(seg):>6} {100*t:>8.1f}% {s:>8.2f} "
              f"{100*(seg > 0).mean():>8.1f}% {100*seg.mean():>8.2f}%")


def swing_age(mkt: np.ndarray) -> np.ndarray:
    """How many days the market's move had ALREADY run as of yesterday.

    Lagged on purpose. Labelling a day by its own return would use the
    outcome to choose the trade, so the table would look like a filter
    and not be one. Shifted by a day it is knowable at entry and can be
    traded on directly.
    """
    sign = np.sign(mkt)
    age = np.zeros(len(mkt), dtype=int)
    for i in range(len(mkt)):
        age[i] = age[i - 1] + 1 if i and sign[i] == sign[i - 1] else 1
    return np.concatenate([[0], age[:-1]])


def by_swing(name: str, R: pd.DataFrame) -> None:
    """What the logic collected from short moves versus long ones.

    Each traded day is labelled by how long the market's direction had
    persisted up to the previous close, so a three-day dip and a
    thirty-day slide are scored separately -- and the split is one the
    bot could actually act on.
    """
    age = swing_age(R["mkt"].values)
    net = R["net"].values
    print(f"\n{name}: return by how long the move had run BEFORE entry")
    print(f"  {'run so far':>12} {'days':>6} {'total':>9} {'mean/day':>9} "
          f"{'hit rate':>9}")
    for lo, hi, lab in ((1, 1, "1 day"), (2, 3, "2-3 days"),
                        (4, 6, "4-6 days"), (7, 99, "7+ days")):
        seg = net[(age >= lo) & (age <= hi)]
        if len(seg) == 0:
            continue
        t, _ = stat(seg)
        print(f"  {lab:>12} {len(seg):>6} {100*t:>8.1f}% {100*seg.mean():>8.2f}% "
              f"{100*(seg > 0).mean():>8.1f}%")
    tall, sall = stat(net)
    print(f"  {'unfiltered':>34} {100*tall:>+8.1f}% Sharpe {sall:>5.2f}")
    for lab, mask in (("skip fresh moves (run < 2d)", age >= 2),
                      ("skip stale moves (run > 3d)", age <= 3)):
        t, s = stat(net[mask])
        print(f"  {lab:>34} {100*t:>+8.1f}% Sharpe {s:>5.2f}")

    # Half by half, because a filter chosen after seeing the whole sample
    # is a fit. If it only pays in one half it is one.
    h = len(net) // 2
    print(f"  {'':>34} {'first half':>16} {'second half':>16}")
    for lab, mask in (("unfiltered", np.ones(len(net), bool)),
                      ("skip stale moves (run > 3d)", age <= 3)):
        a1, _ = stat(net[:h][mask[:h]])
        a2, _ = stat(net[h:][mask[h:]])
        print(f"  {lab:>34} {100*a1:>15.1f}% {100*a2:>15.1f}%")


def by_hold(name: str, R: pd.DataFrame) -> None:
    rr = runs(R["dir"].values)
    traded = [(d_, n) for d_, n in rr if d_ != 0]
    if not traded:
        print(f"\n{name}: never took a position")
        return
    lens = np.array([n for _, n in traded])
    print(f"\n{name}: {len(traded)} positions, "
          f"median hold {np.median(lens):.0f}d, "
          f"shortest {lens.min()}d, longest {lens.max()}d")
    print(f"  {'hold':>10} {'positions':>10} {'share':>8}")
    for lo, hi, lab in ((1, 3, "1-3 days"), (4, 7, "4-7 days"),
                        (8, 20, "8-20 days"), (21, 999, "21+ days")):
        n = int(((lens >= lo) & (lens <= hi)).sum())
        print(f"  {lab:>10} {n:>10} {100*n/len(lens):>7.0f}%")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--states", default="trend")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=300)
    a = ap.parse_args(argv)

    d = load_daily()
    dims = tuple(a.states.split(","))
    print(f"{len(d):,} daily bars, {d.index[0].date()} .. {d.index[-1].date()}")
    print(f"states={a.states} top={a.top}")

    m = mirror(d)
    real_bh = float(d['close'].iloc[-1] / d['close'].iloc[0] - 1)
    mir_bh = float(m['close'].iloc[-1] / m['close'].iloc[0] - 1)

    print("\n" + "=" * 74)
    print("1. REAL SERIES -- does it trade both sides?")
    print("=" * 74)
    R = run(d, dims, a.top, warmup=a.warmup)
    by_direction("real", R, real_bh)
    by_hold("real", R)
    by_swing("real", R)

    print("\n" + "=" * 74)
    print("2. MIRRORED SERIES -- the same market rising instead of falling")
    print("=" * 74)
    M = run(m, dims, a.top, warmup=a.warmup)
    by_direction("mirror", M, mir_bh)
    by_hold("mirror", M)
    by_swing("mirror", M)

    rt, _ = stat(R["net"].values)
    mt, _ = stat(M["net"].values)
    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    if mt > 0 and rt > 0:
        print(f"  Both directions pay: real {100*rt:+.1f}%, "
              f"mirror {100*mt:+.1f}%. The logic is symmetric -- it follows")
        print("  the trend it is given rather than being short by habit.")
    elif rt > 0:
        print(f"  Real {100*rt:+.1f}% but mirror {100*mt:+.1f}%. The result")
        print("  belongs to the fall, not to the method. In a rising market")
        print("  this loses, and the data contains no rise to have shown it.")
    else:
        print(f"  real {100*rt:+.1f}%, mirror {100*mt:+.1f}%")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
