"""Test every logic individually and keep only the ones that pay.

    python -m fp.survivors                 # the full sweep, writes the list
    python -m fp.survivors --tf 1d,2d,3d

WHAT THIS DOES DIFFERENTLY

Everything before this scored logics as a basket -- rank them, hold the
top five, measure the basket. That answers "does the ranking work", and
the answer was no, twice. It never answers the question actually being
asked, which is whether ANY single logic has an edge that survives.

So this tests them one at a time, all of them, and applies the bar that
testing thousands of things requires.

NO HOLD FLOOR, AND WHY THE ONE THAT WAS HERE WAS WRONG

An earlier version of this file refused any logic whose median hold fell
under an hour, reasoning from fp/horizon.py: the average one-minute move
is 0.040% and the round trip costs 0.110%, so a one-minute position
cannot pay.

That argument is sound about a RANDOM one-minute position and says
nothing about a SELECTED one. It compares the fee to the mean of the
move distribution and quietly discards the distribution. The same table
it came from carries the correction in its last column:

    hold        E|move|      cost    share of moves ABOVE the cost
    1 min        0.040%    0.110%                            6.8%
    10 min       0.127%    0.110%                           39.1%
    15 min       0.155%    0.110%                           46.6%
    30 min       0.218%    0.111%                           58.9%

Thirty-nine percent of ten-minute moves already clear the round trip. A
logic that picks direction inside that group is profitable at ten
minutes, and with leverage a 0.3% move is a 30% return on margin. The
floor threw every such logic away untested -- which is exactly the
hard-coded constant this project keeps being told not to introduce.

So there is no floor. A logic qualifies on its own economics:

    lower 95% bound of its in-sample mean net return per trade > 0

where the return is already net of 0.055% each way and of funding over
the hold it actually ran. A ten-minute logic and a ten-day logic meet
the identical bar. Hold length is reported in every table below so the
answer can be read off the data instead of assumed.

Sizing follows the same principle: leverage is solved over the hold each
logic was measured at, and a short hold supports MORE leverage, not less
-- volatility drag is paid per period held, so a ten-minute position
pays a fraction of what a ten-day one does for the same exposure.

THE BAR A LOGIC HAS TO CLEAR

Each surviving logic is measured on trades, entry to exit, never
rebalanced inside, net of 0.055% each way and funding per bar held. The
sample is split in half by time:

    first half     used only to rank
    second half    used only to judge

Two things are then reported, because they answer different questions:

    1. Does any logic have a real edge?   Its own second-half t-stat,
       against a Bonferroni threshold for the number of logics tested.
       At ~2,600 logics per timeframe, p < 0.05 requires |t| > 4.4 --
       a logic at t = 2.5 is what a couple of thousand coin flips
       produce and means nothing on its own.

    2. Does picking the good ones work?   Rank on the first half, hold
       the top ten through the second, and report what they did. This
       is the procedure a live system would run, so its result is the
       one that matters operationally.

AND THE NULL THAT DECIDES IT

Alongside runs a rotation null: each logic's position series is rolled by
a random offset. That preserves the market path, its drift, and that
logic's own long/short balance and hold-length distribution, and destroys
only whether the position lines up with the move. So a logic that merely
sat short through a 31.6% fall scores just as well rotated as it does
real, and only genuine timing shows up as a gap. A sign-flip null cannot
make that distinction and would certify every short-biased logic.

A logic reaches the whitelist only if BOTH bars are cleared: its own
out-of-sample t against Bonferroni, AND its timeframe's picking procedure
against the rotation null. The second gate exists because of what
happened without it -- see below.

WHAT IT MEASURED

    tf   built  dropped  tested  t bar  survivors  null  best t  best OOS
    4h    3066        0    2966   4.30          0     2    2.07     38.3%
    8h    3024        0    2460   4.26          0     1    2.98     54.0%
    1d    2602        0     342   3.80          0     0    1.82     31.5%
    2d     906        0      12   2.87          0     0    0.70     21.4%

Zero. Out of 5,780 logics tested individually across four timeframes, not
one clears its own significance bar, and the best t-stat anywhere (2.98)
is what 2,460 coin flips produce.

The picking procedure is worse than that:

    tf   top   mean/trade   short%   null mean   null sd       p
    4h    10     -0.2639%      43%     0.6833%   0.4030%   1.000
    8h    10     -0.8516%      41%     0.9442%   0.4523%   1.000
    1d    10      0.0534%      42%     0.2915%   0.4766%   0.665
    2d    10     -0.3024%      50%    -0.3880%   0.4276%   0.415

Look at 4h and 8h. Randomly-rotated positions earn +0.68% and +0.94% per
trade; the logics actually chosen earn -0.26% and -0.85%. Selecting on
first-half performance does not merely fail to help, it lands reliably
BELOW random timing -- p = 1.000 in both. Whatever the top logics learned
from the first half, applying it to the second is worse than not knowing
anything.

THE VERSION OF THIS TABLE THAT WAS WRONG, AND WHY IT IS WORTH KEEPING

Before the exit was corrected -- trades were closed at the last bar of a
run rather than the first bar on which the flip was knowable, skipping
exactly the bar that caused the flip -- the same code produced:

                            with the look-ahead    corrected
    4h survivors                          176              0
    4h best t                            7.61           2.07
    4h best OOS                       1471.9%          38.3%
    1d picking, mean/trade            2.1433%        0.0534%
    1d picking, p vs null               0.000          0.665

One bar. That is the entire difference between a hundred and seventy-six
"significant" logics and none.

The 176 were also already being caught by the second gate, before the
exit bug was found: their timeframe scored p = 0.405 against the rotation
null, meaning those individually-significant logics were collectively
indistinguishable from randomly-timed positions with the same tilt. Two
independent checks, each of which would have refused them.

SO THE WHITELIST IS EMPTY

fp/survivors.json holds zero logics and `run_bot.py --signals survivors`
opens nothing. That is the honest output, not a failure to find the
setting that works: across 5,780 logics, four timeframes, every hold from
one hour to two days, on 838,112 real one-minute bars, nothing in this
library has an edge that survives its own fees and an honest test.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fp.ensemble import build_logics
from fp.horizon import (FEE_ROUND_TRIP, FUNDING_PER_8H, TIMEFRAMES, load_1m,
                        resample)

OUT = Path(__file__).resolve().parent / "survivors.json"

EXTRA_TF = {"2d": 2880, "3d": 4320, "5d": 7200}

# How sure the in-sample edge has to be before a logic is worth judging
# out of sample. This is the potential gate, and it is deliberately not a
# time: a logic qualifies by what its own trades earn after real fees,
# whether it holds them for ten minutes or ten days.
POTENTIAL_Z = 1.64                      # one-sided 95%


def trades_of(close: np.ndarray, pos: np.ndarray, minutes: int
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Net return, entry bar and hold length for every completed trade.

    A position that survives twenty bars is ONE trade paying one round
    trip and twenty bars of funding -- not twenty positions each paying
    the fee again. That distinction is the difference between a logic
    that clears its costs and one that cannot.
    """
    n = len(pos)
    if n == 0:
        return np.empty(0), np.empty(0, int), np.empty(0, int)
    p = np.where(np.isfinite(pos), pos, 0.0)
    # Segment boundaries, by run-length arithmetic rather than a loop
    # because the rotation null re-does this a few hundred thousand times.
    edges = np.flatnonzero(np.diff(p) != 0) + 1
    starts_all = np.concatenate([[0], edges])
    last_all = np.concatenate([edges - 1, [n - 1]])
    d = p[starts_all]
    keep = d != 0
    starts, last, d = starts_all[keep], last_all[keep], d[keep]
    # THE EXIT IS ONE BAR AFTER THE RUN ENDS, and this is not a detail.
    # The signal is still `d` at bar `last`; it changes at `last + 1`, and
    # `last + 1` is when that change becomes knowable. Exiting at
    # close[last] would mean closing on the strength of a flip nobody has
    # seen yet -- and the bar it skips is precisely the bar that caused
    # the flip, which is usually the one that ran against the position.
    # Dropping it inflates every logic in the library.
    exits = last + 1
    ok = exits < n
    starts, exits, d = starts[ok], exits[ok], d[ok]     # the last open
    if len(starts) == 0:                                # trade is unclosed
        return np.empty(0), np.empty(0, int), np.empty(0, int)
    p0, p1 = close[starts], close[exits]
    fin = (p0 > 0) & np.isfinite(p1)
    starts, exits, d, p0, p1 = (starts[fin], exits[fin], d[fin], p0[fin],
                                p1[fin])
    holds = exits - starts
    fund = FUNDING_PER_8H * (minutes / 480.0)
    rets = d * (p1 - p0) / p0 - FEE_ROUND_TRIP - holds * fund
    return rets, starts, holds


def tstat(r: np.ndarray) -> float:
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r))))


def bonferroni_t(n_tests: int, alpha: float = 0.05) -> float:
    """Two-sided t threshold once n_tests things have been looked at.

    Normal approximation, which is close enough at these sample sizes and
    errs on the strict side.
    """
    from math import erf, sqrt
    lo, hi = 0.0, 12.0
    target = 1.0 - alpha / (2.0 * max(n_tests, 1))
    for _ in range(80):
        mid = (lo + hi) / 2
        if 0.5 * (1 + erf(mid / sqrt(2))) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def pick_and_score(recs: list[dict], top: int) -> float:
    """Rank on the first half, hold the best through the second."""
    ranked = sorted(recs, key=lambda x: -x["is_t"])[:top]
    return float(np.mean([x["oos_mean"] for x in ranked])) if ranked else 0.0


def shift_null(logics: dict, close: np.ndarray, minutes: int, split: int,
               min_trades: int, top: int, runs: int,
               rng: np.random.Generator, thr: float = 99.0
               ) -> tuple[np.ndarray, np.ndarray]:
    """The same picking procedure on logics whose timing has been broken.

    Each position series is rotated by a random offset. That keeps the
    market path, the drift, and each logic's own long/short balance and
    hold-length distribution exactly as they are -- and destroys only
    whether the position happens to line up with the move.

    So a short-biased logic still collects the fall in the null, which is
    the point: it separates "this logic predicts" from "this logic was
    short in a market that dropped 31.6%". A sign-flip null cannot make
    that distinction and would call every short-biased logic significant.
    """
    out, counts = [], []
    keys = list(logics)
    arrs = {k: logics[k].values.astype(float) for k in keys}
    n = len(close)
    for _ in range(runs):
        recs, n_surv = [], 0
        for k in keys:
            p = np.roll(arrs[k], int(rng.integers(1, n)))
            r, s, h = trades_of(close, p, minutes)
            if len(r) == 0:
                continue
            a, b = r[s < split], r[s >= split]
            if len(a) < min_trades or len(b) < min_trades:
                continue
            if not (a.mean() - POTENTIAL_Z * a.std(ddof=1)
                    / np.sqrt(len(a)) > 0):        # same gate as the real
                continue
            ot = tstat(b)
            n_surv += ot > thr
            recs.append({"is_t": tstat(a), "oos_mean": float(b.mean())})
        if recs:
            out.append(pick_and_score(recs, top))
            counts.append(n_surv)
    return np.array(out), np.array(counts)


def study_tf(d1m: pd.DataFrame, label: str, minutes: int,
             min_trades: int, top: int, rng: np.random.Generator,
             null_runs: int = 0, max_bars: int = 60000) -> dict:
    d = resample(d1m, minutes)
    if len(d) > max_bars:
        d = d.iloc[-max_bars:]
    if len(d) < 150:
        return {"tf": label, "note": "too few bars"}
    close = d["close"].values.astype(float)
    # The rolling-rank methods are O(n*window) and unusable at minute
    # resolution; below an hour the faster family is used and the count
    # built is reported so the difference is visible rather than hidden.
    logics = build_logics(d, fast=minutes < 60)
    split = len(d) // 2

    rows, dropped_hold, dropped_few = [], 0, 0
    for name, pos in logics.items():
        r, s, h = trades_of(close, pos.values.astype(float), minutes)
        if len(r) == 0:
            continue
        a, b = r[s < split], r[s >= split]
        if len(a) < min_trades or len(b) < min_trades:
            dropped_few += 1
            continue
        # THE POTENTIAL GATE. Not "is the hold long enough" but "does this
        # logic's own edge clear its own costs" -- a is already net of the
        # round trip and of funding over the hold it actually ran, so a
        # lower confidence bound above zero says the trade is worth taking
        # on its own economics. A ten-minute logic and a ten-day logic
        # meet exactly the same bar, and the hold is reported rather than
        # required.
        lo = a.mean() - POTENTIAL_Z * a.std(ddof=1) / np.sqrt(len(a))
        if not (lo > 0):
            dropped_hold += 1
            continue
        pv = pos.values.astype(float)
        rows.append({"name": name, "tf": label,
                     "short_share": float((pv < 0).sum()
                                          / max((pv != 0).sum(), 1)),
                     "is_t": tstat(a), "is_mean": float(a.mean()),
                     "is_n": int(len(a)),
                     "oos_t": tstat(b), "oos_mean": float(b.mean()),
                     "oos_n": int(len(b)),
                     "oos_total": float(np.prod(1 + b) - 1),
                     "hold_min": float(np.median(h)) * minutes,
                     # sign-flipped copy: same trades, no edge by
                     # construction, so it says what this test finds in
                     # a library that certainly has nothing in it
                     "null_t": tstat(b * rng.choice([-1.0, 1.0], len(b)))})
    if not rows:
        return {"tf": label, "note": "no logic's edge cleared its costs",
                "dropped_hold": dropped_hold, "dropped_few": dropped_few,
                "logics": len(logics)}

    thr = bonferroni_t(len(rows))
    surv = [x for x in rows if x["oos_t"] > thr]
    null_surv = [x for x in rows if x["null_t"] > thr]

    ranked = sorted(rows, key=lambda x: -x["is_t"])[:top]
    picked = pick_and_score(rows, top)
    # How much of the picked set is simply short, and how much of the
    # market's own fall it would collect by standing still.
    short_share = float(np.mean([x["short_share"] for x in ranked])) if ranked else 0.0

    null, null_counts = (shift_null(logics, close, minutes, split,
                                    min_trades, top, null_runs, rng, thr)
                         if null_runs else (np.array([]), np.array([])))
    pval = (float((null >= picked).mean()) if len(null)
            else float("nan"))

    return {"tf": label, "logics": len(logics), "tested": len(rows),
            "dropped_hold": dropped_hold, "dropped_few": dropped_few,
            "threshold": thr, "survivors": surv,
            "flip_survivors": len(null_surv),
            "rot_survivors": float(null_counts.mean()) if len(null_counts) else float("nan"),
            "rot_survivors_max": int(null_counts.max()) if len(null_counts) else 0,
            "best": max(rows, key=lambda x: x["oos_t"]),
            "picked_mean": picked, "picked_short_share": short_share,
            "null_mean": float(null.mean()) if len(null) else float("nan"),
            "null_sd": float(null.std()) if len(null) else float("nan"),
            "null_p": pval, "null_runs": len(null),
            "picked": [x["name"] for x in ranked]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tf", default="1h,4h,8h,1d,2d,3d")
    ap.add_argument("--max-bars", type=int, default=60000,
                    help="memory cap per timeframe; the fine ones cover a "
                         "shorter span and the span is reported")
    ap.add_argument("--min-trades", type=int, default=20)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--null-alpha", type=float, default=0.05,
                    help="a timeframe must beat the rotation null at this "
                         "level before ANY of its logics may be traded")
    ap.add_argument("--null-runs", type=int, default=200,
                    help="rotation-null repeats for the picking procedure")
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args(argv)

    rng = np.random.default_rng(a.seed)
    tfs = dict(TIMEFRAMES)
    tfs.update(EXTRA_TF)
    tfs["8h"] = 480

    d1m = load_1m()
    print(f"{len(d1m):,} one-minute bars, {d1m.index[0]} .. {d1m.index[-1]}")
    print("no hold floor: a logic qualifies by whether its own trades clear "
          "their own\nfees, at any horizon -- the hold is reported, never "
          "required")

    print("\n" + "=" * 78)
    print("PER-TIMEFRAME: every logic tested on its own, out of sample")
    print("=" * 78)
    print(f"{'tf':>4} {'built':>7} {'no edge':>8} {'tested':>7} {'t bar':>6} "
          f"{'survivors':>10} {'null':>5} {'best t':>7} {'best hold':>10}")

    results = []
    for label in a.tf.split(","):
        label = label.strip()
        if label not in tfs:
            continue
        res = study_tf(d1m, label, tfs[label], a.min_trades,
                       a.top, rng, a.null_runs, a.max_bars)
        results.append(res)
        if "note" in res:
            print(f"{res['tf']:>4} {res.get('logics', 0):>7} "
                  f"{res.get('dropped_hold', 0):>8} {'--':>7}  {res['note']}")
            continue
        b = res["best"]
        hm = b["hold_min"]
        hs = (f"{hm:.0f}m" if hm < 120 else f"{hm/60:.1f}h" if hm < 2880
              else f"{hm/1440:.1f}d")
        print(f"{res['tf']:>4} {res['logics']:>7} {res['dropped_hold']:>8} "
              f"{res['tested']:>7} {res['threshold']:>6.2f} "
              f"{len(res['survivors']):>10} {res['rot_survivors']:>5.0f} "
              f"{b['oos_t']:>7.2f} {hs:>10}")

    print("\n" + "=" * 78)
    print("PICKING THE GOOD ONES: rank on the first half, hold through the second")
    print("=" * 78)
    print("  Against a null that rotates each logic's timing at random --")
    print("  same market, same drift, same long/short balance, alignment")
    print("  destroyed. A logic that only collects the fall scores the same")
    print("  in the null as it does for real.")
    print(f"\n{'tf':>4} {'top':>5} {'mean/trade':>12} {'short%':>8} "
          f"{'null mean':>11} {'null sd':>9} {'p':>7}")
    for res in results:
        if "note" in res:
            continue
        p = res["null_p"]
        ps = "--" if p != p else f"{p:.3f}"
        print(f"{res['tf']:>4} {a.top:>5} {100*res['picked_mean']:>11.4f}% "
              f"{100*res['picked_short_share']:>7.0f}% "
              f"{100*res['null_mean']:>10.4f}% {100*res['null_sd']:>8.4f}% "
              f"{ps:>7}")

    # A logic gets in only if BOTH bars are cleared: its own out-of-sample
    # t against the Bonferroni threshold, and its timeframe's picking
    # procedure against the rotation null. The second is what stops a
    # timeframe whose apparent winners are pure directional tilt from
    # exporting a hundred and seventy of them into the live bot.
    keep, rejected = [], []
    for res in results:
        if "note" in res:
            continue
        p = res["null_p"]
        if not (p == p and p <= a.null_alpha):
            if res["survivors"]:
                rejected.append((res["tf"], len(res["survivors"]), p))
            continue
        for s in res["survivors"]:
            if s["is_t"] > 0:            # must also have worked before
                keep.append(s)

    OUT.write_text(json.dumps(
        {"potential_z": POTENTIAL_Z,
         "generated_from": f"{len(d1m)} 1m bars ending {d1m.index[-1]}",
         "logics": keep}, indent=2))

    print("\n" + "=" * 78)
    print("WHAT THE BOT MAY TRADE")
    print("=" * 78)
    for tf, n, p in rejected:
        print(f"  {tf}: {n} logics cleared their own significance bar and are")
        print(f"  REFUSED anyway -- that timeframe's picking procedure scores")
        print(f"  p={p:.3f} against the rotation null, so its winners are not")
        print(f"  distinguishable from randomly-timed positions with the same")
        print(f"  long/short tilt. Individually significant, collectively noise.\n")
    if keep:
        for s in keep:
            print(f"  {s['tf']:>4} {s['name']:<34} OOS t={s['oos_t']:.2f} "
                  f"mean={100*s['oos_mean']:+.3f}%/trade over {s['oos_n']} trades")
        print(f"\n  {len(keep)} logics written to {OUT.name}.")
    else:
        tot_null = sum(r.get("flip_survivors", 0) for r in results
                       if "note" not in r)
        print("  None. Every logic whose own edge cleared its own fees failed the")
        print("  significance bar for the number of logics tested, and the")
        print(f"  sign-flipped null produced {tot_null} 'survivors' by chance")
        print("  under the same test -- so the best real numbers are not")
        print("  distinguishable from having no edge at all.")
        print(f"\n  {OUT.name} written empty: the bot opens nothing until a")
        print("  logic earns its place in it.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
