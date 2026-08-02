"""Replay historical candles through the live paper-trading broker.

This drives the exact same PaperBroker used by `bot.paper_trading`
against a CSV instead of Bybit's live API, one 1m candle at a time, in
chronological order and with no lookahead: at each step the broker only
sees candles up to `now`, exactly as it would live.

Two reasons this exists:
  1. It lets you see the end-of-session summary format immediately,
     without waiting hours for live trades to open and close.
  2. It's the only way to exercise the live loop in an environment
     where Bybit's API is unreachable (firewall / egress policy).

This is NOT a replacement for backtest/run_backtest.py -- that one is
faster and covers whole years. This one prioritises running the *live
code path* faithfully.

Run:  python3 -m bot.replay data/BTCUSDT_2026.csv --equity 10 --candles 3000
"""
from __future__ import annotations

import argparse

import pandas as pd

from bot.config import Config
from bot.paper_trading import PaperBroker, print_summary

SIGNAL_EVERY_N_CANDLES = 1  # a new 1m candle is a new signal opportunity


class ReplayExchange:
    """Serves historical candles as if they were arriving live."""

    def __init__(self, df_1m: pd.DataFrame, df_1h: pd.DataFrame):
        self.df_1m = df_1m
        self.df_1h = df_1h
        self.now: pd.Timestamp | None = None

    def get_klines(self, symbol: str, timeframe: str, limit: int = 300) -> pd.DataFrame:
        src = self.df_1m if timeframe == "1m" else self.df_1h
        return src[src["datetime"] <= self.now].tail(limit).reset_index(drop=True)

    def get_last_price(self, symbol: str) -> float:
        row = self.df_1m[self.df_1m["datetime"] <= self.now].iloc[-1]
        return float(row["close"])


def load_frames(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["datetime"])
    df_1m = df.sort_values("datetime").reset_index(drop=True)
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    df_1h = (df_1m.set_index("datetime").resample("1h").agg(agg)
             .dropna().reset_index())
    return df_1m, df_1h


def run_replay(csv_path: str, symbol: str, equity: float, candles: int,
               warmup: int = 2000, trades_csv: str | None = None) -> dict:
    df_1m, df_1h = load_frames(csv_path)
    if len(df_1m) < warmup + 10:
        raise SystemExit(f"Need at least {warmup + 10} candles, CSV has {len(df_1m)}")

    exchange = ReplayExchange(df_1m, df_1h)
    broker = PaperBroker(exchange, Config(symbols=[symbol]), starting_equity=equity)

    start = warmup
    end = min(len(df_1m), warmup + candles)
    print(f"Replaying {end - start:,} candles of {symbol} from {csv_path}")
    print(f"  {df_1m['datetime'].iloc[start]}  ->  {df_1m['datetime'].iloc[end - 1]}")
    print(f"  starting virtual equity: ${equity:.2f}\n")

    for i in range(start, end):
        exchange.now = df_1m["datetime"].iloc[i]
        if symbol in broker.open_positions:
            broker.manage_with_candles(symbol)
        if symbol in broker.open_positions:
            if i % SIGNAL_EVERY_N_CANDLES == 0:
                broker.refresh_trend_and_trail(symbol)
        elif i % SIGNAL_EVERY_N_CANDLES == 0:
            broker.try_open(symbol)

    summary = broker.summary()
    print_summary(summary)
    if trades_csv:
        broker.export_trades_csv(trades_csv)
        print(f"\nTrade log written to {trades_csv}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv_path")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--equity", type=float, default=10.0)
    p.add_argument("--candles", type=int, default=3000,
                   help="how many 1m candles to replay after warmup")
    p.add_argument("--warmup", type=int, default=2000,
                   help="candles reserved for indicator convergence")
    p.add_argument("--trades-csv", default=None)
    args = p.parse_args()
    run_replay(args.csv_path, args.symbol, args.equity, args.candles,
               args.warmup, args.trades_csv)


if __name__ == "__main__":
    main()
