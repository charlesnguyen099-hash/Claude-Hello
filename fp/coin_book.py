"""A book for the nine symbols, built the way the BTC book was built.

    python -m fp.coin_book
    python -m fp.coin_book --tf 1m,5m,15m --min-coins 4

WHAT REPLACES "PROFITABLE IN BOTH YEARS"

fp/btc_book.py could demand that a rule pay in 2025 and again in 2026,
because it had nineteen months. Here there are 5.7 days, so that test
does not exist. What does exist is nine symbols, and agreement across
them is a stronger requirement than agreement across two halves of one
week -- a rule cannot fit itself to ETHUSDT and XAUUSDT and SOXLUSDT at
once by accident as easily as it can fit itself to one series twice.

So a rule enters this book only if:

    coins        profitable on at least --min-coins of the symbols it
                 traded on, each with at least --min-trades trades
    halves       profitable in the first half of the week AND the second,
                 aggregated across coins
    expectancy   Wilson lower bound on the win rate, priced against the
                 stop's DESIGNED size, still clears the round trip
    leave-one-out
                 still profitable with its BEST symbol removed

The last one is not optional and it is the lesson of fp/crosscoin.py.
BLESSUSDT rose 223.91% in this week, and every cross-sectional book that
looked significant turned out to be that one coin: +82% of +82% gross,
+94% of +99%, +90% of +97%. A rule that dies when its best symbol is
removed has an effective sample size of one, however many bars it
touched.

TIMEFRAMES, INCLUDING THE FAST ONES

Runs at 1m, 5m and 15m -- 8,190, 1,638 and 546 bars. Nothing slower fits
in 5.7 days with 200-bar factors. The fast end is deliberately included:
the concern that an hourly frame hides minute-scale opportunities is a
fair one, and the way to settle it is to look, not to argue.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fp.coins import load_coins
from fp.ensemble import build_logics
from fp.exits import barrier_outcomes, sigma_at, wilson_lower
from fp.horizon import FEE_ROUND_TRIP, FUNDING_PER_8H, resample

OUT = Path(__file__).resolve().parent / "coin_book.json"
TF = {"1m": 1, "5m": 5, "15m": 15}
# trimmed from the BTC grid: 5.7 days cannot support 90 exits per entry
TPS = (1.0, 2.0, 3.0, 4.0)
SLS = (1.0, 2.0, 3.0)
HOLDS = (12, 48, 192)


def per_coin(d: pd.DataFrame, minutes: int, min_trades: int) -> dict:
    """Every entry x exit x side on one symbol: trades, halves, stats."""
    b = resample(d, minutes)
    if len(b) < 320:
        return {}
    close = b["close"].values.astype(float)
    high = b["high"].values.astype(float)
    low = b["low"].values.astype(float)
    sigma = sigma_at(close)
    fund = FUNDING_PER_8H * (minutes / 480.0)
    split = len(b) // 2
    logics = build_logics(b, fast=True)

    entries: dict[tuple[str, int], np.ndarray] = {}
    for name, pos in logics.items():
        p = pos.values.astype(float)
        prev = np.concatenate([[0.0], p[:-1]])
        for side in (1, -1):
            e = np.flatnonzero((p == side) & (prev != side))
            if len(e) >= min_trades:
                entries[(name, side)] = e

    out = {}
    for hmax in HOLDS:
        for tp in TPS:
            for sl in SLS:
                for side in (1, -1):
                    o, held = barrier_outcomes(high, low, close, sigma, side,
                                               tp, sl, hmax)
                    net = o - FEE_ROUND_TRIP - held * fund
                    for (name, s2), e in entries.items():
                        if s2 != side:
                            continue
                        v = net[e]
                        ok = np.isfinite(v)
                        v, e2 = v[ok], e[ok]
                        if len(v) < min_trades:
                            continue
                        out[(name, side, tp, sl, hmax)] = {
                            "mean": float(v.mean()), "n": len(v),
                            "wins": int((v > 0).sum()),
                            "avg_win": float(v[v > 0].mean()) if (v > 0).any()
                            else 0.0,
                            "h1": float(v[e2 < split].sum()),
                            "h2": float(v[e2 >= split].sum()),
                            "hold": float(np.median(held[e2])) * minutes,
                        }
    return out


def build(coins: dict[str, pd.DataFrame], label: str, minutes: int,
          min_trades: int, min_coins: int) -> pd.DataFrame:
    per = {sym: per_coin(d, minutes, min_trades) for sym, d in coins.items()}
    per = {k: v for k, v in per.items() if v}
    if not per:
        return pd.DataFrame()
    keys = set()
    for v in per.values():
        keys |= set(v)

    rows = []
    for k in keys:
        name, side, tp, sl, hmax = k
        hits = {s: v[k] for s, v in per.items() if k in v}
        if len(hits) < min_coins:
            continue
        pos_coins = [s for s, r in hits.items() if r["mean"] > 0]
        if len(pos_coins) < min_coins:
            continue
        # halves, aggregated across every coin that traded it
        h1 = sum(r["h1"] for r in hits.values())
        h2 = sum(r["h2"] for r in hits.values())
        if not (h1 > 0 and h2 > 0):
            continue
        # expectancy with the stop priced at its designed size
        n = sum(r["n"] for r in hits.values())
        w = sum(r["wins"] for r in hits.values())
        aw = float(np.mean([r["avg_win"] for r in hits.values()
                            if r["avg_win"] > 0]) or 0.0)
        pl = wilson_lower(w, n)
        if not (pl * aw + (1 - pl) * (-(sl / tp) * aw) > 0):
            continue
        # leave-one-out on the BEST symbol
        means = {s: r["mean"] for s, r in hits.items()}
        best = max(means, key=means.get)
        without = [m for s, m in means.items() if s != best]
        if not without or float(np.mean(without)) <= 0:
            continue
        rows.append({
            "tf": label, "logic": name,
            "side": "long" if side > 0 else "short", "sidenum": side,
            "tp": tp, "sl": sl, "hmax": hmax,
            "n_coins": len(pos_coins), "tested_on": len(hits),
            "coins": ",".join(sorted(pos_coins)),
            "mean": float(np.mean(list(means.values()))),
            "mean_wo_best": float(np.mean(without)), "dropped": best,
            "trades": n, "win": w / max(n, 1),
            "hold": float(np.median([r["hold"] for r in hits.values()])),
            "family": name.split("|")[0].rstrip("0123456789_"),
        })
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tf", default="1m,5m,15m")
    ap.add_argument("--min-trades", type=int, default=10)
    ap.add_argument("--min-coins", type=int, default=4)
    ap.add_argument("--size", type=int, default=30)
    ap.add_argument("--per-family", type=int, default=3)
    a = ap.parse_args(argv)

    coins = load_coins()
    print(f"{len(coins)} symbols, {len(next(iter(coins.values()))):,} 1m bars "
          f"each (5.7 days)")
    print("agreement across symbols replaces the two-year test\n")

    frames = []
    for label in a.tf.split(","):
        label = label.strip()
        if label not in TF:
            continue
        R = build(coins, label, TF[label], a.min_trades, a.min_coins)
        print(f"  {label}: {len(R):,} rules paid on {a.min_coins}+ symbols, "
              f"in both halves, and survived leave-one-out")
        if not R.empty:
            frames.append(R)
    if not frames:
        print("\nNothing qualified.")
        return 0

    R = pd.concat(frames, ignore_index=True)
    R = R.sort_values("mean_wo_best", ascending=False)
    picked, fam, seen = [], {}, set()
    for _, r in R.iterrows():
        if r.logic in seen or fam.get(r.family, 0) >= a.per_family:
            continue
        picked.append(r)
        seen.add(r.logic)
        fam[r.family] = fam.get(r.family, 0) + 1
        if len(picked) >= a.size:
            break
    book = pd.DataFrame(picked)

    print(f"\nselected {len(book)} rules, at most {a.per_family} per family\n")
    print(f"{'tf':>4} {'entry':<26} {'side':>5} {'exit':>16} {'coins':>7} "
          f"{'trades':>7} {'win':>6} {'mean':>8} {'w/o best':>9} {'hold':>8}")
    for _, r in book.iterrows():
        h = r.hold
        hs = f"{h:.0f}m" if h < 120 else f"{h/60:.1f}h" if h < 2880 else f"{h/1440:.1f}d"
        print(f"{r.tf:>4} {r.logic:<26} {r.side:>5} "
              f"{f'tp{r.tp}/sl{r.sl}/{r.hmax}b':>16} "
              f"{r.n_coins}/{r.tested_on:>2} {r.trades:>7} {100*r.win:>5.1f}% "
              f"{100*r['mean']:>7.3f}% {100*r.mean_wo_best:>8.3f}% {hs:>8}")

    print("\n" + "=" * 78)
    print("THE BOOK")
    print("=" * 78)
    print(f"  rules                    : {len(book)}")
    print(f"  trades                   : {int(book['trades'].sum()):,}")
    print(f"  mean net per trade       : {100*book['mean'].mean():+.4f}%")
    print(f"  same, best symbol removed: {100*book['mean_wo_best'].mean():+.4f}%")
    print(f"  median hold              : {book['hold'].median():.0f} minutes")
    by_tf = book.groupby("tf")["mean_wo_best"].agg(["size", "mean"])
    for tf, r in by_tf.iterrows():
        print(f"  {tf:>4}: {int(r['size']):>2} rules, "
              f"{100*r['mean']:+.4f}%/trade without the best symbol")
    drops = book["dropped"].value_counts()
    print(f"\n  symbol most often the best one, and therefore removed:")
    for s, c in drops.head(4).items():
        print(f"    {s:<14} {c} of {len(book)} rules")

    OUT.write_text(json.dumps({
        "fitted_on": "9 symbols, 2026-07-31..2026-08-06, 1m bars (5.7 days)",
        "in_sample": True,
        "note": "Selected by agreement across symbols and survival of "
                "leave-one-out, not by a time split. 5.7 days is short: "
                "treat the coin count as the evidence.",
        "logics": [{"tf": r.tf, "name": r.logic, "side": r.side,
                    "tp": float(r.tp), "sl": float(r.sl), "hmax": int(r.hmax),
                    "coins": r.coins, "n_coins": int(r.n_coins),
                    "hold_min": float(r.hold), "mean": float(r["mean"]),
                    "mean_wo_best": float(r.mean_wo_best)}
                   for _, r in book.iterrows()]}, indent=2))
    print(f"\n  written to {OUT.name}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
