"""CLI: run the strategy backtest against a 1m OHLCV CSV file and print a
report. This is how the strategy in bot/strategy.py was validated before
being wired into the live bot — it is not a marketing claim, it is the
actual trade log.

Usage:
    python -m backtest.run_backtest data/BTCUSDT_202607.csv
    python -m backtest.run_backtest data/BTCUSDT_202607.csv --equity 10000
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from backtest.engine import run_backtest


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"datetime", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the trend-pullback strategy")
    parser.add_argument("csv_path")
    parser.add_argument("--equity", type=float, default=10_000.0)
    args = parser.parse_args()

    df = load_csv(args.csv_path)
    result = run_backtest(df, starting_equity=args.equity)
    summary = result.summary()

    print(f"\n=== Backtest report: {args.csv_path} ===")
    if summary.get("trades", 0) == 0:
        print("No trades were triggered on this dataset.")
        sys.exit(0)

    for k, v in summary.items():
        print(f"{k:>18}: {v}")

    print("\n--- Trade log (first 20) ---")
    for t in result.trades[:20]:
        print(
            f"{t.entry_time} {t.side.upper():5s} entry={t.entry:.2f} stop={t.stop:.2f} "
            f"lev={t.leverage}x conf={t.confidence:.0f} risk%={t.equity_risk_pct*100:.1f} "
            f"tp1_hit={t.tp1_hit} -> exit {t.exit_time} @{t.exit_price:.2f} ({t.exit_reason}) "
            f"pnl=${t.pnl_usd:.2f} R={t.r_multiple:.2f}"
        )
    if len(result.trades) > 20:
        print(f"... and {len(result.trades) - 20} more trades")


if __name__ == "__main__":
    main()
