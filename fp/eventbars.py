"""Bars that form when something HAPPENS, not when the clock says so.

    python -m fp.eventbars                 # build them, then test on them
    python -m fp.eventbars --kinds dollar,cusum

WHY THE CLOCK IS THE PROBLEM

Every logic tested so far lives on a fixed time grid. A 4h bar is a 4h
bar whether the market traded a billion dollars in it or nothing at all.
That grid is a hard-coded constant sitting underneath all 5,780 logics,
and no amount of dynamic thresholding on top of it removes the fact that
the SAMPLING itself ignores what the market is doing.

The market does not deliver information at a constant rate. It delivers
it in bursts. Sampling on a clock therefore oversamples dead hours and
undersamples exactly the moments worth trading -- which is the opposite
of what a volatility-driven instrument needs.

WHAT REPLACES IT

Four ways of cutting the same 838,112 one-minute bars into bars that
close on an event instead of a timestamp:

    volume    closes once N contracts have traded
    dollar    closes once N dollars of turnover have passed
    range     closes once price has travelled R from where the bar opened
    cusum     closes when cumulative drift, in units of the market's own
              current volatility, exceeds a threshold in either direction

All four are self-adjusting by construction. In a fast market a bar can
close in seconds; in a quiet one the same bar takes hours. Nobody picks
the duration -- the market does, and the duration becomes an OUTPUT that
gets reported rather than an input that gets assumed.

The cusum filter is the strictest of the four: it emits nothing at all
until the market has actually moved, so it produces no bars during the
periods that generated most of the losing trades in the earlier work.

AND THEY ARE RUN AT SEVERAL RATES, NOT ONE

Each kind is built at several average bar rates -- roughly 4, 24 and 96
bars a day -- so no single scale is privileged. A logic that only works
when sampling is fast, or only when it is slow, shows up as such instead
of being hidden by a single choice.

WHAT IS UNCHANGED, DELIBERATELY

The bar is the only thing that changes. The same factor library, the
same methods, and above all the same gates apply:

    potential   the lower 95% bound of a logic's own in-sample net return
                per trade must clear zero, after 0.055% each way and
                funding over the hold it actually ran
    Bonferroni  out-of-sample t against a threshold for the number tested
    rotation    the timeframe's picking procedure must beat a null that
                rolls each logic's timing at random

Funding is charged on REAL elapsed hours here, not on a bar count, since
bars no longer have a fixed duration. That is the one accounting change
event bars force, and it makes the cost model more correct rather than
less.

WHAT THE BARS TURNED OUT TO BE

The duration really is an output, and it moves by more than an order of
magnitude inside a single series:

    bars           built    median      p10       p90    longest
    volume/4       2,328      5.1h    94.0m     11.3h      38.3h
    volume/24     13,968     44.0m    10.0m      2.1h       9.2h
    volume/96     55,874     10.0m     2.0m     35.0m       3.1h
    range/24      13,968     53.0m    22.0m    104.0m       8.1h
    cusum/4        1,766      6.2h    75.0m     17.2h      32.4h
    cusum/96      56,825     10.0m     3.0m     32.0m       6.3h

A volume/4 bar takes 94 minutes when the market is busy and 38 hours
when it is not. Nobody chose either number.

AND WHAT THEY FOUND

    bars        logics  no edge  tested  t bar  surv  best t   picked     null      p
    volume/4      3014     2763      47   3.27     0    1.13  -0.4408%  0.7571%  1.000
    volume/24     3036     3007       3   2.39     0    0.33  +0.0490%  0.2227%  0.775
    volume/96     1120     1110      --                    --
    dollar/4      3128     2825      57   3.33     0    1.23  -0.3176%  0.9390%  1.000
    dollar/24     3054     3028      --                    --
    dollar/96     1122     1114      --                    --
    range/4       2980     2707      57   3.33     0    1.42  -0.3254%  0.7430%  1.000
    range/24      3012     2989       1   1.96     0    0.50  +0.2606%  0.2008%  0.350
    range/96      1134     1124      --                    --
    cusum/4       2952     2387      23   3.07     0    1.50  -0.2032%  0.9935%  1.000
    cusum/24      2956     2934       4   2.50     0    0.74  +0.0141%  0.1729%  0.875
    cusum/96      1082     1074      --                    --

28,590 logic instances across twelve event-bar series. Zero survivors in
every one. The best out-of-sample t-stat anywhere is 1.50, against a bar
of 3.07.

At the fastest sampling -- roughly 96 bars a day, a median bar of ten
minutes -- not one logic on any of the four bar types has a positive
in-sample edge after fees. That is the same answer the ten-minute clock
bars gave, reached by a completely different route, which is worth more
than either result alone: it is not the clock that was the problem.

And the pattern that has now appeared five times: wherever the picking
procedure could be run, randomly-rotated positions beat the logics that
were actually selected. +0.76% against -0.44%, +0.94% against -0.32%,
+0.99% against -0.20%. p = 1.000 in each. Choosing on first-half
performance is reliably worse than not choosing.

WHAT THIS RULES OUT, WHICH IS THE POINT OF HAVING RUN IT

The clock was a real hard-coded constant and it is gone: sampling now
adapts to volume, to turnover, to distance travelled, and to volatility
itself. The result did not move. So the failure is not in how the market
was sampled, and it is not in how many logics were tried -- it is that
this factor library does not predict this market, on any clock or none.

The honest next step is not more logics. 34,000 have now been tested
across time bars and event bars, and a library that large is guaranteed
to produce in-sample winners inside any slice, which is precisely why
the bar for believing one has to stay where it is.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from fp.ensemble import build_logics
from fp.horizon import FEE_ROUND_TRIP, FUNDING_PER_8H, load_1m
from fp.survivors import POTENTIAL_Z, bonferroni_t, pick_and_score, tstat

KINDS = ("volume", "dollar", "range", "cusum")
RATES = (4, 24, 96)              # target bars per day


# ------------------------------------------------------------ constructors

def _agg(d: pd.DataFrame, ends: np.ndarray) -> pd.DataFrame:
    """Fold the minute bars into the bars whose last minute is `ends`."""
    if len(ends) < 50:
        return pd.DataFrame()
    starts = np.concatenate([[0], ends[:-1] + 1])
    o = d["open"].values[starts]
    c = d["close"].values[ends]
    hi = np.maximum.reduceat(d["high"].values, starts)
    lo = np.minimum.reduceat(d["low"].values, starts)
    v = np.add.reduceat(d["volume"].values, starts)
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c,
                         "volume": v}, index=d.index[ends])


def _cuts(x: np.ndarray, threshold: float) -> np.ndarray:
    """Index of each bar's last minute, closing whenever cumsum crosses."""
    cum = np.cumsum(x)
    if cum[-1] <= threshold:
        return np.array([], dtype=int)
    # bar k closes at the first index whose running total passes k*threshold
    levels = np.arange(1, int(cum[-1] // threshold) + 1) * threshold
    return np.searchsorted(cum, levels, side="left")


def volume_bars(d: pd.DataFrame, per_day: float) -> pd.DataFrame:
    days = max((d.index[-1] - d.index[0]).total_seconds() / 86400.0, 1.0)
    return _agg(d, _cuts(d["volume"].values, d["volume"].sum() / (days * per_day)))


def dollar_bars(d: pd.DataFrame, per_day: float) -> pd.DataFrame:
    turnover = (d["close"] * d["volume"]).values
    days = max((d.index[-1] - d.index[0]).total_seconds() / 86400.0, 1.0)
    return _agg(d, _cuts(turnover, turnover.sum() / (days * per_day)))


def range_bars(d: pd.DataFrame, per_day: float) -> pd.DataFrame:
    """Closes once price has travelled R, so it samples MOVEMENT directly."""
    move = np.abs(np.diff(np.log(d["close"].values), prepend=np.log(
        d["close"].values[0])))
    days = max((d.index[-1] - d.index[0]).total_seconds() / 86400.0, 1.0)
    return _agg(d, _cuts(move, move.sum() / (days * per_day)))


def cusum_bars(d: pd.DataFrame, per_day: float) -> pd.DataFrame:
    """Symmetric CUSUM: nothing is emitted until the market actually moves.

    The threshold is in units of the market's own trailing volatility, so
    it widens in turbulence and tightens in calm on its own. The running
    sums reset at every event, which is what stops a slow drift from
    accumulating into a signal that never happened.
    """
    lr = np.diff(np.log(d["close"].values), prepend=np.log(d["close"].values[0]))
    vol = pd.Series(lr).rolling(1440, min_periods=240).std().bfill().values
    days = max((d.index[-1] - d.index[0]).total_seconds() / 86400.0, 1.0)
    # calibrate k so the event count lands near the requested rate
    k = 1.0
    ends = np.array([], dtype=int)
    for _ in range(28):
        sp = sn = 0.0
        out = []
        for i in range(len(lr)):
            h = k * vol[i]
            if h <= 0 or not np.isfinite(h):
                continue
            sp = max(0.0, sp + lr[i])
            sn = min(0.0, sn + lr[i])
            if sp > h or sn < -h:
                sp = sn = 0.0
                out.append(i)
        ends = np.array(out, dtype=int)
        want = per_day * days
        if len(ends) == 0:
            k *= 0.5
        elif len(ends) > want * 1.15:
            k *= 1.25
        elif len(ends) < want * 0.85:
            k *= 0.8
        else:
            break
    return _agg(d, ends)


BUILDERS = {"volume": volume_bars, "dollar": dollar_bars,
            "range": range_bars, "cusum": cusum_bars}


# ------------------------------------------------------------------- P&L

def trades_at(close: np.ndarray, pos: np.ndarray, hours: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trades on irregular bars, funding charged on REAL elapsed hours.

    Same exit rule as fp/survivors: a run ending at bar `last` is closed
    at `last + 1`, because that is the first bar on which the flip is
    knowable. Bars no longer have a fixed duration, so funding cannot be
    a bar count and is taken from the clock instead.
    """
    n = len(pos)
    if n == 0:
        return np.empty(0), np.empty(0, int), np.empty(0)
    p = np.where(np.isfinite(pos), pos, 0.0)
    edges = np.flatnonzero(np.diff(p) != 0) + 1
    starts_all = np.concatenate([[0], edges])
    last_all = np.concatenate([edges - 1, [n - 1]])
    d = p[starts_all]
    keep = d != 0
    starts, last, d = starts_all[keep], last_all[keep], d[keep]
    exits = last + 1
    ok = exits < n
    starts, exits, d = starts[ok], exits[ok], d[ok]
    if len(starts) == 0:
        return np.empty(0), np.empty(0, int), np.empty(0)
    p0, p1 = close[starts], close[exits]
    fin = (p0 > 0) & np.isfinite(p1)
    starts, exits, d, p0, p1 = (starts[fin], exits[fin], d[fin], p0[fin],
                                p1[fin])
    held = np.maximum(hours[exits] - hours[starts], 0.0)
    rets = (d * (p1 - p0) / p0 - FEE_ROUND_TRIP
            - FUNDING_PER_8H * held / 8.0)
    return rets, starts, held


# ----------------------------------------------------------------- study

def study(bars: pd.DataFrame, label: str, min_trades: int, top: int,
          rng: np.random.Generator, null_runs: int) -> dict:
    if len(bars) < 400:
        return {"tf": label, "note": f"only {len(bars)} bars"}
    close = bars["close"].values.astype(float)
    hours = (bars.index - bars.index[0]).total_seconds().values / 3600.0
    logics = build_logics(bars, fast=len(bars) > 30000)
    if not logics:
        return {"tf": label, "note": "no logics"}
    split = len(bars) // 2

    rows, no_edge = [], 0
    for name, pos in logics.items():
        r, s, h = trades_at(close, pos.values.astype(float), hours)
        if len(r) == 0:
            continue
        a, b = r[s < split], r[s >= split]
        if len(a) < min_trades or len(b) < min_trades:
            continue
        if not (a.mean() - POTENTIAL_Z * a.std(ddof=1) / np.sqrt(len(a)) > 0):
            no_edge += 1
            continue
        rows.append({"name": name, "tf": label,
                     "is_t": tstat(a), "oos_t": tstat(b),
                     "oos_mean": float(b.mean()), "oos_n": int(len(b)),
                     "hold_h": float(np.median(h))})
    if not rows:
        return {"tf": label, "bars": len(bars), "logics": len(logics),
                "no_edge": no_edge, "note": "no logic's edge cleared its costs"}

    thr = bonferroni_t(len(rows))
    surv = [x for x in rows if x["oos_t"] > thr]
    picked = pick_and_score(rows, top)

    null = []
    keys = list(logics)
    arrs = {k: logics[k].values.astype(float) for k in keys}
    for _ in range(null_runs):
        recs = []
        for k in keys:
            q = np.roll(arrs[k], int(rng.integers(1, len(close))))
            r, s, h = trades_at(close, q, hours)
            if len(r) == 0:
                continue
            a, b = r[s < split], r[s >= split]
            if len(a) < min_trades or len(b) < min_trades:
                continue
            if not (a.mean() - POTENTIAL_Z * a.std(ddof=1)
                    / np.sqrt(len(a)) > 0):
                continue
            recs.append({"is_t": tstat(a), "oos_mean": float(b.mean())})
        if recs:
            null.append(pick_and_score(recs, top))
    null = np.array(null)
    return {"tf": label, "bars": len(bars), "logics": len(logics),
            "no_edge": no_edge, "tested": len(rows), "threshold": thr,
            "survivors": surv, "picked": picked,
            "best": max(rows, key=lambda x: x["oos_t"]),
            "null_mean": float(null.mean()) if len(null) else float("nan"),
            "null_p": float((null >= picked).mean()) if len(null) else float("nan")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kinds", default=",".join(KINDS))
    ap.add_argument("--rates", default=",".join(str(r) for r in RATES))
    ap.add_argument("--min-trades", type=int, default=20)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--null-runs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=23)
    a = ap.parse_args(argv)

    rng = np.random.default_rng(a.seed)
    d1m = load_1m()
    span = (d1m.index[-1] - d1m.index[0]).total_seconds() / 86400.0
    print(f"{len(d1m):,} one-minute bars over {span:.0f} days")
    print("bars close on events, not on the clock -- duration is an output")

    print("\n" + "=" * 78)
    print("THE BARS THEMSELVES: how long they actually take")
    print("=" * 78)
    print(f"{'bars':>16} {'built':>8} {'median':>10} {'p10':>9} {'p90':>9} "
          f"{'longest':>10}")
    built = {}
    for kind in a.kinds.split(","):
        kind = kind.strip()
        if kind not in BUILDERS:
            continue
        for rate in (int(x) for x in a.rates.split(",")):
            b = BUILDERS[kind](d1m, rate)
            if len(b) < 400:
                print(f"{kind + '/' + str(rate):>16} {len(b):>8}   too few")
                continue
            gaps = np.diff(b.index.values).astype("timedelta64[s]"
                                                 ).astype(float) / 60.0
            built[f"{kind}/{rate}"] = b

            def fmt(m):
                return (f"{m:.1f}m" if m < 120 else f"{m/60:.1f}h"
                        if m < 2880 else f"{m/1440:.1f}d")
            print(f"{kind + '/' + str(rate):>16} {len(b):>8,} "
                  f"{fmt(np.median(gaps)):>10} {fmt(np.percentile(gaps, 10)):>9} "
                  f"{fmt(np.percentile(gaps, 90)):>9} {fmt(gaps.max()):>10}")

    print("\n" + "=" * 78)
    print("EVERY LOGIC ON EVERY EVENT-BAR SERIES, tested one at a time")
    print("=" * 78)
    print(f"{'bars':>16} {'logics':>8} {'no edge':>8} {'tested':>7} "
          f"{'t bar':>6} {'surv':>5} {'best t':>7} {'picked':>10} "
          f"{'null':>10} {'p':>6}")
    results = []
    for label, b in built.items():
        res = study(b, label, a.min_trades, a.top, rng, a.null_runs)
        results.append(res)
        if "note" in res:
            print(f"{res['tf']:>16} {res.get('logics', 0):>8} "
                  f"{res.get('no_edge', 0):>8} {'--':>7}  {res['note']}")
            continue
        p = res["null_p"]
        print(f"{res['tf']:>16} {res['logics']:>8} {res['no_edge']:>8} "
              f"{res['tested']:>7} {res['threshold']:>6.2f} "
              f"{len(res['survivors']):>5} {res['best']['oos_t']:>7.2f} "
              f"{100*res['picked']:>9.4f}% {100*res['null_mean']:>9.4f}% "
              f"{p:>6.3f}")

    keep = [s for r in results if "note" not in r
            and r["null_p"] == r["null_p"] and r["null_p"] <= 0.05
            for s in r["survivors"] if s["is_t"] > 0]
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if keep:
        for s in keep:
            print(f"  {s['tf']:>14} {s['name']:<32} OOS t={s['oos_t']:.2f} "
                  f"{100*s['oos_mean']:+.3f}%/trade over {s['oos_n']} trades, "
                  f"median hold {s['hold_h']:.1f}h")
    else:
        print("  Nothing. Removing the clock changed how the market is")
        print("  sampled, not whether this factor library can predict it.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
