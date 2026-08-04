"""Backtest the twelve-method logic on 2025 and 2026 separately.

Run:  python3 wl_backtest.py
"""
from __future__ import annotations

from wl import engine as E
from wl import exits as X
from wl import methods as M

DATASETS = {"2025": "bybit_bot/data/BTCUSDT_2025.csv",
            "2026": "bybit_bot/data/BTCUSDT_2026.csv"}


def main() -> None:
    print("=" * 96)
    print("TWELVE-METHOD VOTING LOGIC — 30m bars, all five exits scored")
    print("=" * 96)
    print(f"  stop {X.SL_MULTIPLE} ATR, trail {X.TRAIL_MULTIPLE} ATR, "
          f"fee {X.FEE_ROUND_TRIP*100:.3f}% round trip\n")

    results = {}
    for year, path in DATASETS.items():
        t = E.run(E.load(path))
        results[year] = t
        print(f"  {year}: {len(t):,} signals")

    print()
    hdr = (f"{'exit strategy':20s} | " + " | ".join(
        f"{y} {'mean%':>9s} {'win%':>7s} {'total%':>10s}" for y in DATASETS))
    print(hdr)
    print("-" * len(hdr))
    for s in X.STRATEGIES:
        cells = []
        for y in DATASETS:
            v = results[y][s]
            cells.append(f"{y} {100*v.mean():9.4f} {100*(v>0).mean():7.2f} "
                         f"{100*v.sum():10.1f}")
        print(f"{s:20s} | " + " | ".join(cells))

    cells = []
    for y in DATASETS:
        v = results[y]["best_exit_net"]
        cells.append(f"{y} {100*v.mean():9.4f} {100*(v>0).mean():7.2f} "
                     f"{100*v.sum():10.1f}")
    print(f"{'best_exit (hindsight)':20s} | " + " | ".join(cells))

    print()
    print("=" * 96)
    print("PER-METHOD, best fixed exit on each year")
    print("=" * 96)
    print(f"{'method':26s} | " + " | ".join(
        f"{y} {'n':>6s} {'best exit':>16s} {'mean%':>9s}" for y in DATASETS))
    print("-" * 96)
    survivors = []
    for m in M.METHOD_NAMES:
        cells, means = [], {}
        for y in DATASETS:
            sub = results[y][results[y][m] != "-"]
            if sub.empty:
                cells.append(f"{y} {0:6d} {'-':>16s} {'-':>9s}")
                continue
            best = max(X.STRATEGIES, key=lambda s: sub[s].mean())
            mv = 100 * sub[best].mean()
            means[y] = (best, mv)
            cells.append(f"{y} {len(sub):6d} {best.replace('net_',''):>16s} {mv:9.4f}")
        print(f"{m:26s} | " + " | ".join(cells))
        if len(means) == 2:
            (b25, v25), (b26, v26) = means["2025"], means["2026"]
            if v25 > 0 and v26 > 0 and b25 == b26:
                survivors.append((m, b25, v25, v26))

    print()
    print("=" * 96)
    if survivors:
        print("Methods profitable on BOTH years with the SAME fixed exit:")
        for m, b, v25, v26 in survivors:
            print(f"  {m:26s} {b:18s} {v25:+.4f}% / {v26:+.4f}%")
    else:
        print("No method is profitable on both years with the same fixed exit.")
        print()
        print("Compare the best_exit row against the five above it. That row is")
        print("the only positive one, and it is produced by choosing the winning")
        print("exit per trade after seeing how each turned out.")


if __name__ == "__main__":
    main()
