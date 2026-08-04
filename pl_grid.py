"""Earn from oscillation instead of direction: a maker grid on the same data.

Everything tested so far tried to predict direction, and every one of them
measured a gross edge of approximately zero. That is the wall. This does
not try to get over it — it goes around it.

A grid rests buy orders below the market and sell orders above it. When
price falls to a buy level it fills; when it rises one level, that
inventory is sold. Each completed pair earns

    grid spacing - maker round trip

and it does not matter whether price was going up or down when it
happened. The edge is not a forecast, it is the fact that price oscillates
and a resting order is paid the maker rate for supplying liquidity to
that oscillation.

What can go wrong is not a wrong forecast, it is a trend: price walks away
from the grid, inventory piles up on one side, and the position is
underwater with nothing to sell into. That is real, so it is measured
rather than assumed away — inventory is capped, unrealised P&L on the open
inventory is carried, and both the realised profit and the worst inventory
drawdown are reported.

Fills are conservative. A level counts as filled only if the bar's range
strictly crosses it, and both sides of a pair pay the maker fee.

Run:  python3 pl_grid.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pl import strategy as S

DATASETS = {"2025": "bybit_bot/data/BTCUSDT_2025.csv",
            "2026": "bybit_bot/data/BTCUSDT_2026.csv"}
MAKER_FEE = S.MAKER_FEE_PCT          # per side
ROUND_TRIP = 2 * MAKER_FEE           # 0.040%


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def run_grid(df: pd.DataFrame, spacing: float, max_inventory: int,
             recenter_after: float | None = None) -> dict:
    """Walk the bars, filling grid levels as price crosses them.

    spacing         distance between levels, as a fraction of price
    max_inventory   how many unsold buys may accumulate
    recenter_after  if price leaves the grid by this fraction, abandon the
                    inventory at market and rebuild around the new price
                    (None = never recenter, let it ride)
    """
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    close = df["close"].to_numpy(float)

    centre = close[0]
    inventory: list[float] = []          # entry prices of unsold buys
    realised = 0.0
    pairs = 0
    recentres = 0
    worst_unrealised = 0.0
    equity_curve = []

    for i in range(1, len(close)):
        h, l, c = high[i], low[i], close[i]

        if recenter_after is not None and inventory:
            drift = (centre - c) / centre
            if drift > recenter_after:
                # Price walked away downward; close the stranded inventory.
                for entry in inventory:
                    realised += (c - entry) / entry - ROUND_TRIP
                inventory = []
                centre = c
                recentres += 1

        # Sell any inventory that this bar's high reached one level above.
        if inventory:
            kept = []
            for entry in inventory:
                target = entry * (1 + spacing)
                if h >= target:
                    realised += spacing - ROUND_TRIP
                    pairs += 1
                else:
                    kept.append(entry)
            inventory = kept

        # Buy at the next level below, if the bar's low reached it.
        while len(inventory) < max_inventory:
            level = (min(inventory) if inventory else centre) * (1 - spacing)
            if l <= level:
                inventory.append(level)
            else:
                break

        unreal = sum((c - e) / e for e in inventory)
        worst_unrealised = min(worst_unrealised, unreal)
        equity_curve.append(realised + unreal)

    final_unreal = sum((close[-1] - e) / e for e in inventory)
    curve = np.array(equity_curve)
    peak = np.maximum.accumulate(curve)
    drawdown = float((curve - peak).min()) if curve.size else 0.0

    return {
        "pairs": pairs,
        "realised_pct": round(100 * realised, 2),
        "open_inventory": len(inventory),
        "unrealised_pct": round(100 * final_unreal, 2),
        "total_pct": round(100 * (realised + final_unreal), 2),
        "worst_unrealised_pct": round(100 * worst_unrealised, 2),
        "max_drawdown_pct": round(100 * drawdown, 2),
        "recentres": recentres,
        "profit_per_pair_pct": round(100 * (spacing - ROUND_TRIP), 4),
    }


def main() -> None:
    print("=" * 104)
    print("MAKER GRID — profit from oscillation, no direction forecast")
    print("=" * 104)
    print(f"  maker fee {MAKER_FEE*100:.3f}% per side, {ROUND_TRIP*100:.3f}% "
          f"per completed pair")
    print("  a pair earns (spacing - round trip), regardless of trend direction")
    print("  the risk is not a wrong call, it is inventory stranded by a trend\n")

    data = {y: load(p) for y, p in DATASETS.items()}

    hdr = (f"{'spacing':>8s} {'maxInv':>7s} {'recentre':>9s} | " + " | ".join(
        f"{y} {'pairs':>7s} {'real%':>9s} {'openInv':>8s} {'unreal%':>9s} "
        f"{'TOTAL%':>9s} {'maxDD%':>9s}" for y in DATASETS))
    print(hdr)
    print("-" * len(hdr))

    results = []
    for spacing in (0.001, 0.002, 0.005, 0.01):
        for max_inv in (20, 100):
            for rec in (None, 0.05):
                cells, totals = [], []
                for y, df in data.items():
                    r = run_grid(df, spacing, max_inv, rec)
                    totals.append(r["total_pct"])
                    cells.append(
                        f"{y} {r['pairs']:7d} {r['realised_pct']:9.2f} "
                        f"{r['open_inventory']:8d} {r['unrealised_pct']:9.2f} "
                        f"{r['total_pct']:9.2f} {r['max_drawdown_pct']:9.2f}")
                tag = "never" if rec is None else f"{rec*100:.0f}%"
                ok = all(t > 0 for t in totals)
                if ok:
                    results.append((spacing, max_inv, tag, totals))
                print(f"{spacing*100:7.2f}% {max_inv:7d} {tag:>9s} | "
                      + " | ".join(cells) + ("  <== both +" if ok else ""))

    print()
    print("=" * 104)
    if results:
        print("Configurations profitable on BOTH years:")
        for spacing, max_inv, tag, totals in sorted(
                results, key=lambda r: -sum(r[3])):
            print(f"  spacing {spacing*100:.2f}%, max inventory {max_inv}, "
                  f"recentre {tag}: "
                  + ", ".join(f"{t:+.2f}%" for t in totals))
        print()
        print("These are unleveraged returns on the capital the grid ties up.")
        print("Check maxDD before sizing: the drawdown is inventory held")
        print("through a trend, and that is what decides survivable leverage.")
    else:
        print("No grid configuration was profitable on both years.")


if __name__ == "__main__":
    main()
