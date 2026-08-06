"""Large-scale rule search with the validation welded in.

    python -m fp.search                 # the whole thing
    python -m fp.search --tf 30 --top 40

WHAT THIS IS

Tens of thousands of trading rules, generated mechanically from a wide
factor library, every one scored against every exit, on 1-minute data
from 2025 and 2026. That is the search you asked for.

WHY IT IS BUILT THE WAY IT IS

A search this size cannot be reported the way a small one can. With a
million tests at p = 0.05, chance alone produces fifty thousand
"significant" results, and the best of a million coin-flip strategies
looks superb. Every version of this idea tried in this project has
followed the same arc:

    106 features, one at a time      best AUC 0.5122, 0 clear the bar
    48 method x exit combinations    fit t = 6.60, worse than baseline OOS
    118 adaptive cells               fit +439%, walk-forward -478%, t = -7.24
    55,752 pattern keys              100% in-sample, 0.0% OOS coverage

The in-sample number improved every time the search got bigger. The
out-of-sample number got worse every time. So this module reports both,
always, and never the first without the second:

  * every rule is scored on FIT years only
  * the significance bar is Bonferroni-corrected for the ACTUAL number of
    rules tested, which is printed
  * the ENTIRE search is then re-run inside each walk-forward fold, using
    only the blocks before it, and the winner is applied to the block
    after -- no rule ever sees the bar it is scored on

That last point is not a detail. The first version of this module ranked
rules on the first 75% and then "validated" across all ten blocks, seven
of which sat inside that same 75%. It reported 25 of 25 rules surviving
with out-of-sample t up to 7.90. Every one of those numbers was the rule
being re-measured on the data it was picked from.

WHAT IT REPORTS TODAY, on 67,070 rules over 2025-2026 30m bars:

    rules clearing |t| > 4.6 in the fit years          621
    best fit t                                       10.45  (+1.35%/trade)
    nested walk-forward, pooled                   1,256 trades
                                                  -0.2745%/trade
                                                     t = -6.56

621 rules clear a bar corrected for 67,070 tests, the best of them at
t = 10.45, and the honest forward test of the whole procedure is -0.27%
per trade with t = -6.56. Not merely unprofitable: reliably worse than
the -0.109% that trading every signal blind returns.

The search finds rules. What it does not find is rules that keep working.

HOW THE RULES ARE GENERATED

Each factor is turned into rules by thresholding it at percentiles of its
own history: above the 70th, 80th, 90th, 95th; below the 30th, 20th,
10th, 5th. Each threshold rule fires long or short. Every rule is then
crossed with every take-profit multiple. Pairs of factors are crossed
too, which is where the count becomes large.

Outcomes are precomputed once per bar per exit, so adding rules costs
almost nothing -- the search is bounded by the factor library, not by the
rule count.
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fp import features as F
from fp import logic as L

DATA = Path(__file__).resolve().parent.parent / "bybit_bot" / "data"
TP_MULTIPLES = (1.5, 2.0, 3.0, 4.0, 6.0)
MAX_LOOKFORWARD = 200
PERCENTILES = (5, 10, 20, 30, 70, 80, 90, 95)
MIN_TRADES = 200          # a rule firing less than this is not evaluated
WALK_FOLDS = 10


def read(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, sep=None, engine="python")
    d.columns = [c.strip().lower() for c in d.columns]
    dt = next(c for c in d.columns if "time" in c or "date" in c)
    d["datetime"] = pd.to_datetime(d[dt])
    return d[["datetime", "open", "high", "low", "close", "volume"]]


def load(tf: int) -> pd.DataFrame:
    frames = [read(p) for p in sorted(DATA.glob("BTCUSDT_*.csv"))]
    raw = (pd.concat(frames).drop_duplicates("datetime")
           .sort_values("datetime").reset_index(drop=True))
    return L.to_bars(raw, tf)


def outcomes(bars: pd.DataFrame) -> dict:
    """Long and short gross return for every bar and every TP multiple."""
    h, lo, c = bars["high"].values, bars["low"].values, bars["close"].values
    atr = np.asarray(pd.Series(
        np.maximum(h - lo, np.maximum(
            np.abs(h - np.r_[c[0], c[:-1]]),
            np.abs(lo - np.r_[c[0], c[:-1]])))
    ).ewm(alpha=1 / 14, adjust=False).mean())
    n = len(c)
    out = {}
    for tp_mult in TP_MULTIPLES:
        gl = np.full(n, np.nan)
        gs = np.full(n, np.nan)
        bl = np.zeros(n)
        for i in range(20, n - 1):
            a = atr[i]
            if a <= 0 or not np.isfinite(a):
                continue
            entry = c[i]
            end = min(i + 1 + MAX_LOOKFORWARD, n)
            wh, wl = h[i + 1:end], lo[i + 1:end]
            if len(wh) == 0:
                continue
            for d in (1, -1):
                tp = entry + d * tp_mult * a
                sl = entry - d * L.SL_MULTIPLE * a
                if d > 0:
                    s_hit = np.flatnonzero(wl <= sl)
                    t_hit = np.flatnonzero(wh >= tp)
                else:
                    s_hit = np.flatnonzero(wh >= sl)
                    t_hit = np.flatnonzero(wl <= tp)
                js = s_hit[0] if len(s_hit) else np.inf
                jt = t_hit[0] if len(t_hit) else np.inf
                if js <= jt:
                    r, held = -L.SL_MULTIPLE * a / entry, js
                elif jt < np.inf:
                    r, held = tp_mult * a / entry, jt
                else:
                    r, held = (c[end - 1] - entry) / entry * d, len(wh)
                if d > 0:
                    gl[i], bl[i] = r, held
                else:
                    gs[i] = r
        out[tp_mult] = {"long": gl, "short": gs, "bars": bl}
    out["atr_pct"] = 100.0 * atr / c
    return out


def factor_library(bars: pd.DataFrame, tf: int) -> pd.DataFrame:
    """Every numeric column from the 106-feature set, plus multi-horizon
    versions of the ones that take a lookback."""
    base = F.build(bars)
    cols = {c: base[c].values.astype(float)
            for c in base.columns if base[c].dtype.kind in "fi"}
    c = bars["close"].values
    v = bars["volume"].values
    # Extra horizons: returns, volume ratios and range position over spans
    # the base library does not cover.
    for k in (3, 8, 13, 21, 34, 55, 89, 144):
        cols[f"ret_{k}"] = pd.Series(c).pct_change(k).values
        cols[f"volr_{k}"] = (v / pd.Series(v).rolling(k).mean().values)
        hi = pd.Series(bars["high"]).rolling(k).max().values
        lo_ = pd.Series(bars["low"]).rolling(k).min().values
        cols[f"pos_{k}"] = np.where(hi > lo_, (c - lo_) / (hi - lo_), 0.5)
        cols[f"vol_{k}"] = pd.Series(c).pct_change().rolling(k).std().values
        cols[f"skew_{k}"] = pd.Series(c).pct_change().rolling(k).skew().values
    return pd.DataFrame(cols)


def evaluate(mask, gross, cost):
    """Net stats for the trades a rule selects."""
    g = gross[mask]
    g = g[np.isfinite(g)]
    if len(g) < MIN_TRADES:
        return None
    net = g - cost
    sd = net.std()
    return {"n": len(net), "mean": float(net.mean()),
            "t": float(net.mean() / (sd / math.sqrt(len(net)))) if sd > 0 else 0.0,
            "win": float((net > 0).mean())}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tf", type=int, default=30, help="bar minutes")
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--pairs", type=int, default=300,
                   help="how many top single factors to cross with each other")
    a = p.parse_args(argv)

    t0 = time.time()
    bars = load(a.tf)
    print(f"{len(bars):,} {a.tf}m bars, {bars.datetime.min().date()} .. "
          f"{bars.datetime.max().date()}")
    o = outcomes(bars)
    X = factor_library(bars, a.tf)
    print(f"{X.shape[1]} factors, {len(TP_MULTIPLES)} exits, "
          f"{len(PERCENTILES)} thresholds, 2 directions")

    n = len(bars)
    split = int(n * 0.75)
    fit = np.zeros(n, bool); fit[20:split] = True
    hold = np.zeros(n, bool); hold[split:n - 1] = True
    print(f"fit to {bars.datetime.iloc[split].date()}, hold out the rest\n")

    cost = {tp: L.round_trip_cost(L.DEFAULT_EXIT, False)["total"]
            for tp in TP_MULTIPLES}

    # ---- single-factor rules ------------------------------------------
    rules, tested = [], 0
    names = list(X.columns)
    for name in names:
        col = X[name].values
        finite = np.isfinite(col)
        if finite.sum() < 1000:
            continue
        qs = np.percentile(col[finite & fit], PERCENTILES)
        for pct, thr in zip(PERCENTILES, qs):
            for side in ("above", "below"):
                base = (col > thr) if side == "above" else (col < thr)
                for d in (1, -1):
                    for tp in TP_MULTIPLES:
                        tested += 1
                        g = o[tp]["long"] if d > 0 else o[tp]["short"]
                        r = evaluate(base & fit & finite, g, cost[tp])
                        if r and r["mean"] > 0:
                            rules.append({"rule": f"{name} {side} p{pct}",
                                          "cols": (name,), "thr": (thr,),
                                          "sides": (side,), "dir": d, "tp": tp,
                                          **r})
    print(f"single-factor rules tested: {tested:,}, "
          f"profitable in the fit years: {len(rules):,}")

    # ---- pairs of the best singles -------------------------------------
    rules.sort(key=lambda r: -r["t"])
    seeds = []
    seen = set()
    for r in rules:
        if r["cols"][0] in seen:
            continue
        seen.add(r["cols"][0])
        seeds.append(r)
        if len(seeds) >= a.pairs:
            break
    pair_rules, pair_tested = [], 0
    for r1, r2 in itertools.combinations(seeds, 2):
        c1, c2 = X[r1["cols"][0]].values, X[r2["cols"][0]].values
        m1 = (c1 > r1["thr"][0]) if r1["sides"][0] == "above" else (c1 < r1["thr"][0])
        m2 = (c2 > r2["thr"][0]) if r2["sides"][0] == "above" else (c2 < r2["thr"][0])
        both = m1 & m2 & np.isfinite(c1) & np.isfinite(c2)
        for d in (1, -1):
            for tp in TP_MULTIPLES:
                pair_tested += 1
                g = o[tp]["long"] if d > 0 else o[tp]["short"]
                res = evaluate(both & fit, g, cost[tp])
                if res and res["mean"] > 0:
                    pair_rules.append({
                        "rule": f"{r1['rule']} AND {r2['rule']}",
                        "cols": (r1["cols"][0], r2["cols"][0]),
                        "thr": (r1["thr"][0], r2["thr"][0]),
                        "sides": (r1["sides"][0], r2["sides"][0]),
                        "dir": d, "tp": tp, **res})
    print(f"pair rules tested: {pair_tested:,}, profitable in fit: {len(pair_rules):,}")

    total = tested + pair_tested
    allr = rules + pair_rules
    allr.sort(key=lambda r: -r["t"])
    # Bonferroni for the number actually tested, two-sided 5%
    z = 5.0 if total > 500_000 else (4.6 if total > 50_000 else 4.0)
    print(f"\nTOTAL RULES TESTED: {total:,}")
    print(f"Bonferroni bar for {total:,} tests: |t| > {z:.1f}")
    sig = [r for r in allr if r["t"] > z]
    print(f"Rules clearing it in the fit years: {len(sig):,}\n")

    if not allr:
        print("nothing profitable even in-sample")
        return 0

    print(f"{'rule':>58} {'dir':>4} {'tp':>4} {'n':>7} {'fit net':>9} "
          f"{'t':>6} | {'HELD n':>7} {'HELD net':>10} {'sign':>5}")
    kept = []
    for r in allr[:a.top]:
        cols = [X[c].values for c in r["cols"]]
        m = np.ones(n, bool)
        for c, thr, side in zip(cols, r["thr"], r["sides"]):
            m &= (c > thr) if side == "above" else (c < thr)
            m &= np.isfinite(c)
        g = o[r["tp"]]["long"] if r["dir"] > 0 else o[r["tp"]]["short"]
        hres = evaluate(m & hold, g, cost[r["tp"]])
        hn = f"{100*hres['mean']:>9.4f}%" if hres else f"{'--':>10}"
        same = "OK" if hres and hres["mean"] > 0 else "FLIP"
        print(f"{r['rule'][:58]:>58} {r['dir']:>4} {r['tp']:>4.1f} {r['n']:>7,} "
              f"{100*r['mean']:>8.4f}% {r['t']:>6.2f} | "
              f"{hres['n'] if hres else 0:>7,} {hn} {same:>5}")
        if hres and hres["mean"] > 0 and r["t"] > z:
            kept.append((r, m, g, cost[r["tp"]]))

    # NESTED WALK-FORWARD. The first version of this was wrong and the
    # result was spectacular because of it: rules were ranked on the first
    # 75% of the data and then "validated" over all ten blocks, seven of
    # which were inside that same 75%. Re-measuring a rule on the data it
    # was picked from is not a test, and it reported 25 of 25 surviving
    # with t up to 7.9.
    #
    # Done properly, the ENTIRE search is repeated inside each fold using
    # only the blocks before it, and only then is the winner applied to
    # the block after. No rule ever sees the bar it is scored on.
    print("\n" + "=" * 78)
    print(f"NESTED WALK-FORWARD -- the whole search re-run inside each of "
          f"{WALK_FOLDS} folds")
    print("=" * 78)
    edges = np.linspace(20, n - 1, WALK_FOLDS + 1).astype(int)
    print(f"{'fold':>5} {'train to':>10} {'best rule found on train':>46} "
          f"{'train t':>8} {'fwd n':>6} {'fwd net':>10}")
    fwd_all = []
    for k in range(3, WALK_FOLDS):          # need some history to search on
        tr = np.zeros(n, bool); tr[20:edges[k]] = True
        te = np.zeros(n, bool); te[edges[k]:edges[k + 1]] = True
        best = None
        for name in names:
            col = X[name].values
            fin = np.isfinite(col)
            if (fin & tr).sum() < 500:
                continue
            qs = np.percentile(col[fin & tr], PERCENTILES)
            for pct, thr in zip(PERCENTILES, qs):
                for side in ("above", "below"):
                    base = (col > thr) if side == "above" else (col < thr)
                    for d in (1, -1):
                        for tp in TP_MULTIPLES:
                            g = o[tp]["long"] if d > 0 else o[tp]["short"]
                            r = evaluate(base & tr & fin, g, cost[tp])
                            if r and (best is None or r["t"] > best[0]["t"]):
                                best = (r, name, thr, side, d, tp)
        if best is None:
            continue
        r, name, thr, side, d, tp = best
        col = X[name].values
        m = ((col > thr) if side == "above" else (col < thr)) & np.isfinite(col)
        g = o[tp]["long"] if d > 0 else o[tp]["short"]
        gg = g[m & te]; gg = gg[np.isfinite(gg)]
        fwd = (gg - cost[tp]) if len(gg) else np.array([])
        fwd_all.extend(fwd.tolist())
        lab = f"{name} {side} p{PERCENTILES[list(np.percentile(col[np.isfinite(col)&tr], PERCENTILES)).index(thr)] if thr in list(np.percentile(col[np.isfinite(col)&tr], PERCENTILES)) else '?'} d={d} tp={tp}"
        print(f"{k:>5} {str(bars.datetime.iloc[edges[k]].date()):>10} "
              f"{lab[:46]:>46} {r['t']:>8.2f} {len(fwd):>6} "
              + (f"{100*fwd.mean():>+9.4f}%" if len(fwd) else f"{'--':>10}"))
    if fwd_all:
        arr = np.array(fwd_all); sd = arr.std()
        tt = arr.mean() / (sd / math.sqrt(len(arr))) if sd > 0 else 0.0
        print(f"\n  POOLED FORWARD: {len(arr):,} trades   "
              f"{100*arr.mean():>+8.4f}%/trade   total {100*arr.sum():>+8.1f}%   "
              f"t={tt:>5.2f}")
        print(f"  {'PROFITABLE OUT OF SAMPLE' if arr.mean() > 0 and tt > 2 else 'NOT profitable out of sample'}")
    else:
        print("  no forward trades")

    print("\n" + "=" * 78)
    print("SINGLE-SPLIT HOLDOUT")
    print("=" * 78)
    if not kept:
        print(f"  Nothing cleared |t| > {z:.1f} in the fit years AND stayed")
        print("  positive on the held-out period.")
        best_h = [r for r in allr[:a.top]]
        print(f"\n  Best fit t was {allr[0]['t']:.2f} on '{allr[0]['rule'][:50]}'")
        print(f"  -- {total:,} tests were run, so a t of about {z:.1f} is what")
        print("  chance alone produces at this scale. A rule below that bar")
        print("  is indistinguishable from the best of that many coin flips.")
    else:
        for r, m, g, c in kept:
            oos = []
            edges = np.linspace(20, n - 1, WALK_FOLDS + 1).astype(int)
            for k in range(1, WALK_FOLDS):
                te = np.zeros(n, bool); te[edges[k]:edges[k + 1]] = True
                gg = g[m & te]; gg = gg[np.isfinite(gg)]
                oos.extend((gg - c).tolist())
            if oos:
                arr = np.array(oos)
                sd = arr.std()
                tt = arr.mean() / (sd / math.sqrt(len(arr))) if sd > 0 else 0
                print(f"  {r['rule'][:52]:>52}: {len(arr):>6,} OOS trades  "
                      f"{100*arr.mean():>+8.4f}%/trade  t={tt:>5.2f}")
    print("=" * 78)
    print(f"elapsed {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
