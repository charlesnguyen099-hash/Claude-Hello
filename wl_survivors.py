"""How many of these logics actually make money, and how much?

A "logic" here is one tradeable rule: a trigger plus a fixed exit chosen
in advance. Three levels of trigger are counted, because "logic" could
reasonably mean any of them:

  L1  a single method                    12 x 5 exits =  60 logics
  L2  a pair of methods firing together  66 x 5 exits = 330 logics
  L3  a vote count (1..7 methods fired)   7 x 5 exits =  35 logics

Each is scored on 2025 and on 2026 separately, and reported three ways:
profitable on 2025, profitable on 2026, and profitable on BOTH.

Only the third column answers the question. A logic that pays on one year
and not the other is not a logic that makes money — it is the year that
suited it, and there is no way to know in advance which year you are in.
Exits are fixed at entry throughout; picking the best exit per trade is
what the file's own best_exit column does, and no bot can do it.

Run:  python3 wl_survivors.py
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from wl import engine as E
from wl import exits as X
from wl import methods as M

DATASETS = {"2025": "bybit_bot/data/BTCUSDT_2025.csv",
            "2026": "bybit_bot/data/BTCUSDT_2026.csv"}
MIN_TRADES = 30


def score(t25: pd.DataFrame, t26: pd.DataFrame, mask25, mask26,
          label: str) -> list[dict]:
    a, b = t25[mask25], t26[mask26]
    if len(a) < MIN_TRADES or len(b) < MIN_TRADES:
        return []
    out = []
    for s in X.STRATEGIES:
        m25, m26 = float(a[s].mean()), float(b[s].mean())
        out.append({
            "logic": f"{label} + {s.replace('net_', '')}",
            "n25": len(a), "n26": len(b),
            "mean25": 100 * m25, "mean26": 100 * m26,
            "total25": 100 * float(a[s].sum()), "total26": 100 * float(b[s].sum()),
            "pos25": m25 > 0, "pos26": m26 > 0, "both": m25 > 0 and m26 > 0,
        })
    return out


def main() -> None:
    print("Scoring every logic on 2025 and 2026 separately "
          f"(minimum {MIN_TRADES} trades per year)...\n")
    t = {y: E.run(E.load(p)) for y, p in DATASETS.items()}
    t25, t26 = t["2025"], t["2026"]
    print(f"  signals: 2025 {len(t25):,}   2026 {len(t26):,}\n")

    rows: list[dict] = []

    # L1 -- single methods
    for m in M.METHOD_NAMES:
        rows += score(t25, t26, t25[m] != "-", t26[m] != "-",
                      m.split("_", 1)[1])

    # L2 -- pairs firing together
    for m1, m2 in itertools.combinations(M.METHOD_NAMES, 2):
        rows += score(t25, t26,
                      (t25[m1] != "-") & (t25[m2] != "-"),
                      (t26[m1] != "-") & (t26[m2] != "-"),
                      f"{m1.split('_',1)[1]}+{m2.split('_',1)[1]}")

    # L3 -- vote counts
    for k in sorted(set(t25["n_methods_fired"]) | set(t26["n_methods_fired"])):
        rows += score(t25, t26, t25["n_methods_fired"] == k,
                      t26["n_methods_fired"] == k, f"{k}-methods-fired")

    df = pd.DataFrame(rows)
    total = len(df)
    print("=" * 92)
    print("HOW MANY LOGICS MAKE MONEY")
    print("=" * 92)
    print(f"  logics tested (trigger x fixed exit, >={MIN_TRADES} trades/year) : {total:,}")
    print(f"  profitable on 2025                                        : "
          f"{int(df['pos25'].sum()):,}  ({100*df['pos25'].mean():.1f}%)")
    print(f"  profitable on 2026                                        : "
          f"{int(df['pos26'].sum()):,}  ({100*df['pos26'].mean():.1f}%)")
    print(f"  profitable on BOTH                                        : "
          f"{int(df['both'].sum()):,}  ({100*df['both'].mean():.1f}%)")

    both = df[df["both"]].sort_values("mean25", ascending=False)
    print()
    print("=" * 92)
    if both.empty:
        print("No logic is profitable on both years.")
        print()
        best = df.assign(worst=df[["mean25", "mean26"]].min(axis=1)) \
                 .sort_values("worst", ascending=False).head(10)
        print("Closest ten, ranked by their weaker year:")
        print(f"  {'logic':52s} {'n25':>5s} {'2025%':>9s} {'n26':>5s} {'2026%':>9s}")
        print("  " + "-" * 84)
        for _, r in best.iterrows():
            print(f"  {r['logic']:52s} {r['n25']:5.0f} {r['mean25']:9.4f} "
                  f"{r['n26']:5.0f} {r['mean26']:9.4f}")
    else:
        print(f"THE {len(both)} LOGICS THAT MAKE MONEY ON BOTH YEARS")
        print("=" * 92)
        print(f"  {'logic':52s} {'n25':>5s} {'2025%':>9s} {'n26':>5s} {'2026%':>9s}")
        print("  " + "-" * 84)
        for _, r in both.iterrows():
            print(f"  {r['logic']:52s} {r['n25']:5.0f} {r['mean25']:9.4f} "
                  f"{r['n26']:5.0f} {r['mean26']:9.4f}")
        print()
        print(f"  combined per-trade edge, weaker year : "
              f"{both[['mean25','mean26']].min(axis=1).mean():+.4f}%")
        print(f"  total return if all were traded      : "
              f"2025 {both['total25'].sum():+.1f}%, 2026 {both['total26'].sum():+.1f}%")
        print()
        print("  Sanity check before believing this: with "
              f"{total:,} logics tested, roughly {total//4:,} would clear both")
        print("  years on coin flips alone. Compare that against the count above.")


if __name__ == "__main__":
    main()
