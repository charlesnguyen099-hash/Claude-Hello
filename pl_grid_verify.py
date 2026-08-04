"""Verify the grid result with explicit dollar accounting.

pl_grid.py measured returns in units of "one position", which is fine for
ranking configurations and wrong for answering "what happened to $10". This
re-runs the two best configurations with real money: a fixed starting
balance, each grid level taking a fixed slice of it, margin actually
reserved when inventory is held, and equity marked to market every bar.

It also stress-tests the one assumption the result depends on. The grid
fills a resting order whenever a bar's range touches its level. A real
resting order at a level price only grazes may not fill at all, and the
ones that do fill in a fast move are exactly the ones that fill against
you. FILL_RATIO models that: only that fraction of touched levels are
treated as filled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pl import strategy as S

DATASETS = {"2025": "bybit_bot/data/BTCUSDT_2025.csv",
            "2026": "bybit_bot/data/BTCUSDT_2026.csv"}
MAKER_FEE = S.MAKER_FEE_PCT
RNG = np.random.default_rng(11)


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def run(df: pd.DataFrame, spacing: float, max_inv: int, recenter: float,
        equity0: float = 10.0, leverage: int = 1,
        fill_ratio: float = 1.0) -> dict:
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    close = df["close"].to_numpy(float)

    # Each level gets an equal slice of the deployable notional.
    notional_total = equity0 * leverage
    per_level = notional_total / max_inv

    equity = equity0
    centre = close[0]
    inv: list[tuple[float, float]] = []      # (entry price, qty)
    pairs = 0
    recentres = 0
    curve = []

    for i in range(1, len(close)):
        h, l, c = high[i], low[i], close[i]

        if inv and (centre - c) / centre > recenter:
            for entry, qty in inv:
                equity += (c - entry) * qty - MAKER_FEE * qty * c
            inv = []
            centre = c
            recentres += 1

        if inv:
            kept = []
            for entry, qty in inv:
                target = entry * (1 + spacing)
                if h >= target and (fill_ratio >= 1.0 or RNG.random() < fill_ratio):
                    equity += (target - entry) * qty - MAKER_FEE * qty * target
                    pairs += 1
                else:
                    kept.append((entry, qty))
            inv = kept

        while len(inv) < max_inv:
            base = min(e for e, _ in inv) if inv else centre
            level = base * (1 - spacing)
            if l <= level and (fill_ratio >= 1.0 or RNG.random() < fill_ratio):
                qty = per_level / level
                equity -= MAKER_FEE * qty * level
                inv.append((level, qty))
            else:
                break

        mtm = equity + sum((c - e) * q for e, q in inv)
        curve.append(mtm)
        if mtm <= 0:
            return {"pairs": pairs, "final": 0.0, "return_pct": -100.0,
                    "max_dd_pct": -100.0, "blown": True, "recentres": recentres}

    final = equity + sum((close[-1] - e) * q for e, q in inv)
    arr = np.array(curve)
    peak = np.maximum.accumulate(arr)
    dd = float(((arr - peak) / peak).min() * 100)
    return {
        "pairs": pairs, "final": final,
        "return_pct": 100 * (final / equity0 - 1),
        "max_dd_pct": dd, "blown": False, "recentres": recentres,
        "open_inv": len(inv),
    }


def main() -> None:
    data = {y: load(p) for y, p in DATASETS.items()}
    configs = [(0.005, 20, 0.05), (0.01, 20, 0.05)]

    print("=" * 96)
    print("GRID, WITH REAL MONEY — $10 start, equity marked to market each bar")
    print("=" * 96)
    print(f"{'spacing':>8s} {'lev':>4s} {'fill':>5s} | " + " | ".join(
        f"{y} {'pairs':>6s} {'final$':>9s} {'return%':>9s} {'maxDD%':>8s}"
        for y in DATASETS))
    print("-" * 96)

    for spacing, max_inv, rec in configs:
        for lev in (1, 3, 5):
            for fill in (1.0, 0.5, 0.25):
                cells, rets, blown = [], [], False
                for y, df in data.items():
                    r = run(df, spacing, max_inv, rec, 10.0, lev, fill)
                    rets.append(r["return_pct"])
                    blown |= r["blown"]
                    cells.append(f"{y} {r['pairs']:6d} {r['final']:9.2f} "
                                 f"{r['return_pct']:9.2f} {r['max_dd_pct']:8.2f}")
                ok = all(x > 0 for x in rets) and not blown
                print(f"{spacing*100:7.2f}% {lev:4d} {fill:5.2f} | "
                      + " | ".join(cells) + ("  <== both +" if ok else ""))
        print()

    print("=" * 96)
    print("fill is the fraction of touched levels assumed to actually fill. 1.00")
    print("is the optimistic case pl_grid.py used; a resting order at a level")
    print("price only grazes often does not fill, and the fills you do get in a")
    print("fast move are the adverse ones. Watch how the result moves with it.")


if __name__ == "__main__":
    main()
