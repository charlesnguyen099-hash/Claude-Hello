"""How deep should the entry pull back before we take the trade?

research/two_stage_entry.py compared entry modes on a simplified,
equal-sized, fixed-target model and found waiting for a 0.5-2 ATR
pullback more than halved the loss on both years. Applying the same idea
inside the REAL engine -- with confidence-scaled sizing, dynamic
leverage, partial TP, breakeven moves, chandelier trailing and the stale
exit -- did not reproduce that: 2026 improved sharply (-13.34% ->
-2.35%) while 2025 got worse (+4.76% -> -5.68%).

That gap is the whole reason for this script. A conclusion from a
simplified model is a hypothesis, not a result; the engine that actually
places the trades is the one that decides. So sweep PULLBACK_ATR_MULT
through the full engine on both years and look at the pair of outcomes
together, rather than trusting either year alone.

Run:  python3 -m research.pullback_sweep
"""
from __future__ import annotations

import pandas as pd

from backtest.engine import run_backtest_prepared
from bot import strategy

DATASETS = {"2026": "data/BTCUSDT_2026.csv", "2025": "data/BTCUSDT_2025.csv"}
PULLBACKS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
WAIT_BARS = [60, 120, 240]


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def main() -> None:
    prepared = {y: strategy.prepare(load(p)) for y, p in DATASETS.items()}

    header = (f"{'pullback':>8s} {'wait':>5s} | "
              f"{'2026 trades':>11s} {'win%':>6s} {'PF':>6s} {'ret%':>8s} {'maxDD%':>8s} | "
              f"{'2025 trades':>11s} {'win%':>6s} {'PF':>6s} {'ret%':>8s} {'maxDD%':>8s} | both+")
    print(header)
    print("-" * len(header))

    rows = []
    orig_pb, orig_wait = strategy.PULLBACK_ATR_MULT, strategy.PENDING_MAX_BARS
    try:
        for pb in PULLBACKS:
            for wait in (WAIT_BARS if pb > 0 else [orig_wait]):
                strategy.PULLBACK_ATR_MULT = pb
                strategy.PENDING_MAX_BARS = wait
                res = {y: run_backtest_prepared(df).summary() for y, df in prepared.items()}
                a, b = res["2026"], res["2025"]
                if a.get("trades", 0) == 0 or b.get("trades", 0) == 0:
                    print(f"{pb:8.2f} {wait:5d} | (no trades on at least one year)")
                    continue
                both = a["return_pct"] > 0 and b["return_pct"] > 0
                rows.append((pb, wait, a, b, both))
                print(f"{pb:8.2f} {wait:5d} | "
                      f"{a['trades']:11d} {a['win_rate_pct']:6.1f} {a['profit_factor']:6.2f} "
                      f"{a['return_pct']:8.2f} {a['max_drawdown_pct']:8.2f} | "
                      f"{b['trades']:11d} {b['win_rate_pct']:6.1f} {b['profit_factor']:6.2f} "
                      f"{b['return_pct']:8.2f} {b['max_drawdown_pct']:8.2f} | "
                      f"{'YES' if both else ''}")
    finally:
        strategy.PULLBACK_ATR_MULT, strategy.PENDING_MAX_BARS = orig_pb, orig_wait

    print("\n" + "=" * 74)
    winners = [r for r in rows if r[4]]
    if winners:
        print(f"{len(winners)} setting(s) profitable on BOTH years:")
        for pb, wait, a, b, _ in sorted(winners,
                                        key=lambda r: -(r[2]["return_pct"] + r[3]["return_pct"])):
            print(f"  pullback={pb:.2f} ATR wait={wait}m  "
                  f"2026 {a['return_pct']:+7.2f}% ({a['trades']} trades)  "
                  f"2025 {b['return_pct']:+7.2f}% ({b['trades']} trades)")
    else:
        print("No pullback setting was profitable on BOTH years.")
        best = sorted(rows, key=lambda r: -(r[2]["return_pct"] + r[3]["return_pct"]))[:5]
        print("\nLeast-bad by combined return (still not a profitable system):")
        for pb, wait, a, b, _ in best:
            print(f"  pullback={pb:.2f} ATR wait={wait:>3d}m  "
                  f"2026 {a['return_pct']:+7.2f}% / 2025 {b['return_pct']:+7.2f}%  "
                  f"= {a['return_pct'] + b['return_pct']:+7.2f}% combined, "
                  f"worst DD {min(a['max_drawdown_pct'], b['max_drawdown_pct']):.2f}%")


if __name__ == "__main__":
    main()
