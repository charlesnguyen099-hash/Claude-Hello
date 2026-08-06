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

THE HORIZON FLOOR, applied before anything is tested

fp/horizon.py established by arithmetic that below ten minutes no logic
can be profitable at Bybit's fees: the average one-minute move is 0.040%
and the round trip costs 0.110%, so a forecast right every single time
still loses. A one-to-two-minute position is not a hard trade, it is a
closed one, and it is also where reversal risk is highest.

So any logic whose median hold falls under --min-hold minutes is dropped
before it is scored. Not down-weighted -- dropped. The count of what the
floor removed is reported, because a filter that silently removes most
of the library is worth seeing.

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

And a null is run alongside: the same test on sign-flipped trades, which
has no edge by construction. If the real library produces no more
survivors than the null does, the survivors are noise regardless of how
good their numbers look.

WHAT COMES OUT

fp/survivors.json -- the logics that cleared the bar, with the timeframe
and the statistics that justified them. The bot reads it and trades only
what is in it. An empty file means the honest answer was "none", and the
bot then opens nothing, which is what "only trade profitable logics"
means when no logic is profitable.
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

# Below this, fp/horizon.py showed the fee exceeds the move even with a
# perfect forecast. Ten minutes is where break-even first becomes possible
# at all; sixty is where it stops requiring a hit rate nobody has.
MIN_HOLD_MINUTES = 60

EXTRA_TF = {"2d": 2880, "3d": 4320, "5d": 7200}


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
    # Segment boundaries: a trade runs from where the position changes to
    # the bar before it changes again. Done with run-length arithmetic
    # rather than a loop, because the rotation null re-does this a few
    # hundred thousand times.
    edges = np.flatnonzero(np.diff(p) != 0) + 1
    starts_all = np.concatenate([[0], edges])
    ends_all = np.concatenate([edges - 1, [n - 1]])
    d = p[starts_all]
    keep = d != 0
    starts, ends, d = starts_all[keep], ends_all[keep], d[keep]
    if len(starts) == 0:
        return np.empty(0), np.empty(0, int), np.empty(0, int)
    p0, p1 = close[starts], close[ends]
    ok = (p0 > 0) & np.isfinite(p1)
    starts, ends, d, p0, p1 = (starts[ok], ends[ok], d[ok], p0[ok], p1[ok])
    holds = ends - starts + 1
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
               min_hold: int, min_trades: int, top: int, runs: int,
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
            if len(r) == 0 or float(np.median(h)) * minutes < min_hold:
                continue
            a, b = r[s < split], r[s >= split]
            if len(a) < min_trades or len(b) < min_trades:
                continue
            ot = tstat(b)
            n_surv += ot > thr
            recs.append({"is_t": tstat(a), "oos_mean": float(b.mean())})
        if recs:
            out.append(pick_and_score(recs, top))
            counts.append(n_surv)
    return np.array(out), np.array(counts)


def study_tf(d1m: pd.DataFrame, label: str, minutes: int, min_hold: int,
             min_trades: int, top: int, rng: np.random.Generator,
             null_runs: int = 0) -> dict:
    d = resample(d1m, minutes)
    if len(d) < 150:
        return {"tf": label, "note": "too few bars"}
    close = d["close"].values.astype(float)
    logics = build_logics(d)
    split = len(d) // 2

    rows, dropped_hold, dropped_few = [], 0, 0
    for name, pos in logics.items():
        r, s, h = trades_of(close, pos.values.astype(float), minutes)
        if len(r) == 0:
            continue
        if float(np.median(h)) * minutes < min_hold:
            dropped_hold += 1
            continue
        a, b = r[s < split], r[s >= split]
        if len(a) < min_trades or len(b) < min_trades:
            dropped_few += 1
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
        return {"tf": label, "note": "nothing passed the floor",
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

    null, null_counts = (shift_null(logics, close, minutes, split, min_hold,
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
    ap.add_argument("--min-hold", type=int, default=MIN_HOLD_MINUTES,
                    help="drop any logic whose median hold is shorter, in "
                         "minutes -- the frontier says these cannot pay")
    ap.add_argument("--min-trades", type=int, default=20)
    ap.add_argument("--top", type=int, default=10)
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
    print(f"horizon floor: any logic holding under {a.min_hold} minutes is "
          f"dropped before scoring")

    print("\n" + "=" * 78)
    print("PER-TIMEFRAME: every logic tested on its own, out of sample")
    print("=" * 78)
    print(f"{'tf':>4} {'built':>7} {'dropped':>8} {'tested':>7} {'t bar':>6} "
          f"{'survivors':>10} {'null':>5} {'best t':>7} {'best OOS':>10}")

    results = []
    for label in a.tf.split(","):
        label = label.strip()
        if label not in tfs:
            continue
        res = study_tf(d1m, label, tfs[label], a.min_hold, a.min_trades,
                       a.top, rng, a.null_runs)
        results.append(res)
        if "note" in res:
            print(f"{res['tf']:>4} {res.get('logics', 0):>7} "
                  f"{res.get('dropped_hold', 0):>8} {'--':>7}  {res['note']}")
            continue
        b = res["best"]
        print(f"{res['tf']:>4} {res['logics']:>7} {res['dropped_hold']:>8} "
              f"{res['tested']:>7} {res['threshold']:>6.2f} "
              f"{len(res['survivors']):>10} {res['rot_survivors']:>5.0f} "
              f"{b['oos_t']:>7.2f} {100*b['oos_total']:>9.1f}%")

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

    keep = []
    for res in results:
        if "note" in res:
            continue
        for s in res["survivors"]:
            if s["is_t"] > 0:            # must also have worked before
                keep.append(s)

    OUT.write_text(json.dumps(
        {"min_hold_minutes": a.min_hold,
         "generated_from": f"{len(d1m)} 1m bars ending {d1m.index[-1]}",
         "logics": keep}, indent=2))

    print("\n" + "=" * 78)
    print("WHAT THE BOT MAY TRADE")
    print("=" * 78)
    if keep:
        for s in keep:
            print(f"  {s['tf']:>4} {s['name']:<34} OOS t={s['oos_t']:.2f} "
                  f"mean={100*s['oos_mean']:+.3f}%/trade over {s['oos_n']} trades")
        print(f"\n  {len(keep)} logics written to {OUT.name}.")
    else:
        tot_null = sum(r.get("flip_survivors", 0) for r in results
                       if "note" not in r)
        print("  None. Every logic that cleared the horizon floor failed the")
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
