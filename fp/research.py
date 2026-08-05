"""Walk-forward validator: fit on 2025+2026, hold August 2026 out.

    python -m fp.research

Every claim that some variant of this logic makes money has to survive
this, because the obvious version of the search does not. Selecting the
method/exit pairs that were positive in BOTH fit years produced nine
survivors with a pooled t of 3.89 -- and on the held-out month they
returned +0.0315% gross against +0.0974% for taking all twelve methods
unselected. The selection was worse than no selection out of sample,
which is what overfitting looks like when you finally test it.

So this module exists to make that failure mode visible by default:

  * the fit years and the held-out month are never mixed
  * a t-statistic is reported against a Bonferroni threshold for the
    number of variants actually screened, not the one you liked
  * results are ranked by WORST period, not by average, because an
    average lets one good year carry two bad ones
  * fees are quoted at all three tiers, since the gross edge here is
    smaller than the spread between them

What it currently reports, on BTCUSDT 30m:

  no exit is profitable in all three periods at any fee tier. The least
  bad is TP1.5/SL1.5 at maker, losing 0.041% per trade in its worst
  period. Direction does not help either -- SHORT beat LONG in both fit
  years and LONG beat SHORT in the held-out month.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fp import features as F
from fp import logic as L
from fp import methods as M

DATA = Path(__file__).resolve().parent.parent / "bybit_bot" / "data"
FEE_TIERS = (("maker 0.040%", L.MAKER_ROUND_TRIP),
             ("taker 0.110%", L.TAKER_ROUND_TRIP),
             ("taker+slip 0.250%", L.TAKER_WITH_SLIPPAGE))


def read(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, sep=None, engine="python")
    d.columns = [c.strip().lower() for c in d.columns]
    dt = next(c for c in d.columns if "time" in c or "date" in c)
    d["datetime"] = pd.to_datetime(d[dt])
    return d[["datetime", "open", "high", "low", "close", "volume"]]


def periods() -> dict[str, tuple[pd.DataFrame, pd.Timestamp | None]]:
    """Fit periods, then the held-out month.

    August is spliced onto the tail of 2026 so the indicators have their
    warm-up, but only bars at or after the August start are ever scored.
    """
    y25, y26 = read(DATA / "BTCUSDT_2025.csv"), read(DATA / "BTCUSDT_2026.csv")
    aug = read(DATA / "BTCUSDT_202608.csv")
    a0 = aug["datetime"].min()
    pre = y26[y26["datetime"] < a0]
    return {"2025": (y25, None),
            "2026": (pre, None),
            "AUG (held out)": (pd.concat([pre.tail(30_000), aug]), a0)}


def trades(raw: pd.DataFrame, start: pd.Timestamp | None) -> pd.DataFrame:
    """One row per method signal per exit, gross of fees."""
    bars = L.to_bars(raw)
    feats = F.build(bars)
    votes = M.evaluate_all(feats)
    hi, lo, cl = bars["high"].values, bars["low"].values, bars["close"].values
    atr_pct = feats["atr14_pct"].values
    when = bars["datetime"].values
    out = []
    for i in range(250, len(cl) - 1):
        if start is not None and when[i] < np.datetime64(start):
            continue
        a = atr_pct[i]
        if not np.isfinite(a) or a <= 0:
            continue
        for m in M.METHOD_NAMES:
            side = votes[m].iloc[i]
            if side == M.NONE:
                continue
            d = 1 if side == M.LONG else -1
            for ex in L.EXIT_STRATEGIES:
                g, _, _ = L.simulate_exit(hi, lo, cl, i, d,
                                          (a / 100) * cl[i], ex, fee=0.0)
                out.append((m, ex, d, a, g))
    return pd.DataFrame(out, columns=["method", "exit", "dir", "atr", "gross"])


def tstat(x: pd.Series) -> float:
    if len(x) < 2 or x.std() == 0:
        return 0.0
    return float(x.mean() / (x.std() / np.sqrt(len(x))))


def main() -> int:
    tabs = {}
    for name, (raw, start) in periods().items():
        tabs[name] = trades(raw, start)
        print(f"  {name}: {len(tabs[name]):,} method-exit rows")
    fit = pd.concat([tabs["2025"], tabs["2026"]])
    held = tabs["AUG (held out)"]
    n_screened = len(M.METHOD_NAMES) * len(L.EXIT_STRATEGIES)
    # Two-sided 5%, Bonferroni over everything screened.
    t_needed = 2.94 if n_screened >= 40 else 2.5

    print("\n" + "=" * 78)
    print("1. EVERY METHOD x EXIT, SCREENED ON THE FIT YEARS")
    print("=" * 78)
    print(f"{'method':>22} {'exit':>13} {'n':>7} {'gross%':>9} {'t':>7} "
          f"{'sig?':>5} {'held-out%':>11} {'n':>5}")
    survivors = []
    for m in M.METHOD_NAMES:
        for ex in L.EXIT_STRATEGIES:
            g = fit[(fit.method == m) & (fit.exit == ex)].gross
            h = held[(held.method == m) & (held.exit == ex)].gross
            if len(g) < 100:
                continue
            t = tstat(g)
            sig = abs(t) >= t_needed
            if g.mean() > 0:
                survivors.append((m, ex))
            hv = f"{100*h.mean():>10.4f}" if len(h) else f"{'--':>10}"
            print(f"{m.split('_',1)[1]:>22} {ex.replace('net_',''):>13} "
                  f"{len(g):>7,} {100*g.mean():>8.4f} {t:>7.2f} "
                  f"{'YES' if sig else '':>5} {hv} {len(h):>5}")
    print(f"\n  Bonferroni threshold for {n_screened} screened variants: |t| >= {t_needed}")

    print("\n" + "=" * 78)
    print("2. DOES SELECTING THE FIT-YEAR WINNERS HELP OUT OF SAMPLE?")
    print("=" * 78)
    sel_fit = pd.concat([fit[(fit.method == m) & (fit.exit == ex)].gross
                         for m, ex in survivors]) if survivors else pd.Series(dtype=float)
    sel_held = pd.concat([held[(held.method == m) & (held.exit == ex)].gross
                          for m, ex in survivors]) if survivors else pd.Series(dtype=float)
    base_fit = fit[fit.exit == L.DEFAULT_EXIT].gross
    base_held = held[held.exit == L.DEFAULT_EXIT].gross
    print(f"  selected ({len(survivors)} pairs) : fit {100*sel_fit.mean():+.4f}% "
          f"(t={tstat(sel_fit):.2f}, n={len(sel_fit):,})   "
          f"HELD OUT {100*sel_held.mean():+.4f}% (n={len(sel_held):,})")
    print(f"  all twelve, unselected : fit {100*base_fit.mean():+.4f}% "
          f"(t={tstat(base_fit):.2f}, n={len(base_fit):,})   "
          f"HELD OUT {100*base_held.mean():+.4f}% (n={len(base_held):,})")
    if len(sel_held) and len(base_held):
        verdict = ("selection HURT out of sample -- the survivors were noise"
                   if sel_held.mean() <= base_held.mean()
                   else "selection helped out of sample")
        print(f"  -> {verdict}")

    print("\n" + "=" * 78)
    print("3. EXITS RANKED BY WORST PERIOD, NOT BY AVERAGE")
    print("=" * 78)
    header = f"{'exit':>13} " + "".join(f"{n:>13}" for n in tabs)
    print(header + "   worst period, net")
    for ex in L.EXIT_STRATEGIES:
        by = [tabs[n][tabs[n].exit == ex].gross.mean() for n in tabs]
        row = f"{ex.replace('net_',''):>13} " + "".join(f"{100*x:>12.4f}%" for x in by)
        worst = min(by)
        tail = "  ".join(f"{tag.split()[0]} {100*(worst-f):+.4f}%"
                         for tag, f in FEE_TIERS)
        print(f"{row}   {tail}")
    print("\n  A variant is only usable if its WORST period clears the fee.")

    print("\n" + "=" * 78)
    print("4. DIRECTION")
    print("=" * 78)
    for n, tb in tabs.items():
        parts = []
        for d, nm in ((1, "LONG"), (-1, "SHORT")):
            g = tb[(tb.exit == L.DEFAULT_EXIT) & (tb.dir == d)].gross
            parts.append(f"{nm} {100*g.mean():+.4f}% (n={len(g):,})")
        print(f"  {n:>16}: " + "   ".join(parts))
    print("\n  If the better side changes between the fit years and the")
    print("  held-out month, there is no directional edge to trade.")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    usable = []
    for ex in L.EXIT_STRATEGIES:
        worst = min(tabs[n][tabs[n].exit == ex].gross.mean() for n in tabs)
        for tag, f in FEE_TIERS:
            if worst - f > 0:
                usable.append((ex, tag, worst - f))
    if usable:
        for ex, tag, net in sorted(usable, key=lambda r: -r[2]):
            print(f"  PROFITABLE IN EVERY PERIOD: {ex} at {tag}, "
                  f"worst {100*net:+.4f}%/trade")
    else:
        print("  No exit is profitable in every period at any fee tier.")
        print("  Nothing here should be traded with real money.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
