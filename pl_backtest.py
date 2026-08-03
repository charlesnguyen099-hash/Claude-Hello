"""Backtest the PureLogic rule on 2025-2026, at several leverage settings.

Run:  python3 pl_backtest.py
"""
from __future__ import annotations

import numpy as np

from pl import backtest as B
from pl import features as F
from pl import strategy as S

TABLE = ("/root/.claude/uploads/2499e73f-5145-5c6f-b255-816732633901/"
         "70587060-Sheet16_PureLogic_MaxLeverage_MaxFee_24590.txt")
DATASETS = {"2025": "bybit_bot/data/BTCUSDT_2025.csv",
            "2026": "bybit_bot/data/BTCUSDT_2026.csv"}


def main() -> None:
    logic = S.PureLogic(TABLE)
    print("=" * 96)
    print("THE RULE, COMPILED FROM THE TABLE")
    print("=" * 96)
    print(" ", logic.describe())
    print(f"  features: {', '.join(logic.keys)}")
    print(f"  bars: {S.BAR_MINUTES}m for indicators, 1m for entries and exits")
    print(f"  safe leverage for a {S.LEVERAGE_SAFETY_MAE*100:.2f}% expected "
          f"adverse move: {S.safe_leverage()}x")

    data = {y: B.load(p) for y, p in DATASETS.items()}
    cells = {}
    for y, df in data.items():
        feats = S.features_30m_on_1m(df, F.build)
        cells[y] = logic.cells_for(feats)

    print()
    print("=" * 96)
    print("RESULTS — direction and hold from the table, prices from BTCUSDT")
    print("=" * 96)
    print("avg_gross is the edge before costs. It has to clear the "
          f"{S.ROUND_TRIP_PCT*100:.3f}% round trip")
    print("before any leverage helps; leverage multiplies whatever sign it has.\n")

    hdr = (f"{'leverage':>9s} | " + " | ".join(
        f"{y} {'trades':>7s} {'win%':>6s} {'gross%':>8s} {'net%':>8s} "
        f"{'lev.net%':>9s} {'liq':>5s} {'equity':>10s}" for y in DATASETS))
    print(hdr)
    print("-" * len(hdr))

    for lev in (1, 5, 10, S.safe_leverage(), 25, 50, 100):
        cellsout = []
        for y in DATASETS:
            r = B.run(data[y], logic, cells[y], lev)
            if r["trades"] == 0:
                cellsout.append(f"{y} {0:7d} {'-':>6s} {'-':>8s} {'-':>8s} "
                                f"{'-':>9s} {'-':>5s} {'-':>10s}")
                continue
            eq = r["equity_x"]
            eqs = "wiped" if eq <= -0.999 else f"{1+eq:9.3g}x"
            cellsout.append(
                f"{y} {r['trades']:7d} {r['win_rate_pct']:6.2f} "
                f"{r['avg_gross_pct']:8.4f} {r['avg_net_unlev_pct']:8.4f} "
                f"{r['avg_net_lev_pct']:9.3f} {r['liquidations']:5d} {eqs:>10s}")
        print(f"{lev:8d}x | " + " | ".join(cellsout))

    print()
    print("=" * 96)
    print("WHY THE LOSING TRADES LOST")
    print("=" * 96)
    for y in DATASETS:
        r = B.run(data[y], logic, cells[y], S.safe_leverage())
        c = r["loss_causes"]
        print(f"\n  {y}: {r['losses']:,} losers of {r['trades']:,} "
              f"({100*r['losses']/r['trades']:.1f}%), "
              f"median MAE {r['median_mae_pct']:.4f}%")
        print(f"    liquidated                    : {c['liquidated']:,}")
        print(f"    price went the wrong way      : {c['wrong_direction']:,}")
        print(f"    right way, fee was bigger     : {c['fee_ate_it']:,}")
        print(f"    barely moved at all           : {c['flat']:,}")


if __name__ == "__main__":
    main()
