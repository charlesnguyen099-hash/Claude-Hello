"""Run the real paper-trading bot end to end, with $10, minus the network hop.

bot.paper_trading.run() is executed exactly as it is in production: the
same poll loop, the same signal evaluation, the same TP/SL management,
the same SIGINT shutdown path, the same session summary and CSV export.
The only substitution is the HTTP layer -- BybitExchange is swapped for a
stand-in that serves real historical BTCUSDT candles in the same shape
the live wrapper returns.

This exists because Bybit is unreachable from the build environment: the
egress policy answers 403 to CONNECT for api.bybit.com, api-testnet.
bybit.com, api.bytick.com and stream.bybit.com alike. Rather than leave
the live path untested, everything above the socket is exercised here.

The clock advances one minute per poll, so a session covering hours of
market action finishes in seconds. Stop is delivered as a real SIGINT,
so the shutdown path being tested is the one Ctrl+C uses.

Run:  python3 -m research.live_loop_dryrun --equity 10 --minutes 600
"""
from __future__ import annotations

import argparse
import os
import signal
import threading

import pandas as pd

from bot import paper_trading
from bot.config import CONFIG


class ReplayHTTP:
    """Serves real historical candles in BybitExchange's output format."""

    def __init__(self, csv_path: str, start_index: int):
        df = pd.read_csv(csv_path, sep=None, engine="python")
        df.columns = [c.strip().lower() for c in df.columns]
        df["datetime"] = pd.to_datetime(df["datetime"])
        self.df_1m = df.sort_values("datetime").reset_index(drop=True)
        agg = {"open": "first", "high": "max", "low": "min",
               "close": "last", "volume": "sum"}
        self.df_1h = (self.df_1m.set_index("datetime").resample("1h")
                      .agg(agg).dropna().reset_index())
        self.cursor = start_index
        self.calls = 0
        self.exhausted = False

    @property
    def now(self) -> pd.Timestamp:
        return self.df_1m["datetime"].iloc[self.cursor]

    def get_klines(self, symbol, timeframe, limit=300):
        self.calls += 1
        src = self.df_1m if timeframe == "1m" else self.df_1h
        window = src[src["datetime"] <= self.now].tail(limit).reset_index(drop=True)
        # Market time advances in lockstep with the bot's own polling, one
        # minute per 1m-kline fetch, so the feed can never outrun the loop.
        if timeframe == "1m":
            self.cursor += 1
            if self.cursor >= len(self.df_1m) - 1:
                self.exhausted = True
        return window

    def get_last_price(self, symbol):
        return float(self.df_1m["close"].iloc[self.cursor])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/BTCUSDT_2026.csv")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--equity", type=float, default=10.0)
    p.add_argument("--minutes", type=int, default=600,
                   help="how many minutes of market action to walk through")
    p.add_argument("--start", type=int, default=27000,
                   help="candle index to start from (needs indicator warmup)")
    p.add_argument("--trades-csv", default=None)
    args = p.parse_args()

    # The replay feed serves one instrument, so scope the bot to it rather
    # than letting all configured symbols trade the same candles. Config is
    # a frozen dataclass, so swap in a copy rather than mutating it.
    import dataclasses
    scoped = dataclasses.replace(CONFIG, symbols=[args.symbol])
    paper_trading.CONFIG = scoped

    feed = ReplayHTTP(args.csv, args.start)
    print(f"Running the real bot loop on {args.csv}")
    print(f"  virtual equity : ${args.equity:.2f}")
    print(f"  market time    : {feed.now} onward, {args.minutes} minutes")
    print(f"  symbols        : {scoped.symbols}")
    print("  (BybitExchange replaced by a replay feed -- Bybit is unreachable")
    print("   from this environment; every other code path is the real one)\n")

    # Swap the exchange and collapse the poll delays; the loop is unchanged.
    paper_trading.BybitExchange = lambda config: feed
    paper_trading.SIGNAL_POLL_SECONDS = 0

    target_cursor = args.start + args.minutes

    def watcher():
        """Stop the bot once it has walked the requested stretch of market."""
        import time
        while feed.cursor < target_cursor and not feed.exhausted:
            time.sleep(0.05)
        os.kill(os.getpid(), signal.SIGINT)

    threading.Thread(target=watcher, daemon=True).start()
    paper_trading.run(starting_equity=args.equity, poll_seconds=0,
                      trades_csv=args.trades_csv)
    print(f"\n(replay feed served {feed.calls:,} kline requests, "
          f"ended at market time {feed.now})")


if __name__ == "__main__":
    main()
