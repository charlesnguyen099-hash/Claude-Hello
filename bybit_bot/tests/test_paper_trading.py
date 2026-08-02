"""Tests for bot/paper_trading.py against a fake exchange (no network),
fed from the real BTCUSDT 2026 dataset. Verifies the simulated fill/fee
accounting and the end-of-session summary math, without ever touching
Bybit.
"""
from __future__ import annotations

import pandas as pd

from bot.config import Config
from bot.paper_trading import PaperBroker


def _load_resampled(csv_path="data/BTCUSDT_2026.csv"):
    df = pd.read_csv(csv_path, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    df_15m = df.resample("15min").agg(agg).dropna().reset_index()
    df_1h = df.resample("1h").agg(agg).dropna().reset_index()
    return df_15m, df_1h


class FakeExchange:
    def __init__(self, df_15m, df_1h):
        self.df_15m = df_15m
        self.df_1h = df_1h
        self.now: pd.Timestamp | None = None
        self.price_override: float | None = None

    def get_klines(self, symbol, timeframe, limit=300):
        src = self.df_15m if timeframe == "15m" else self.df_1h
        return src[src["datetime"] <= self.now].tail(limit).reset_index(drop=True)

    def get_last_price(self, symbol):
        if self.price_override is not None:
            return self.price_override
        row = self.df_15m[self.df_15m["datetime"] <= self.now].iloc[-1]
        return float(row["close"])


def test_paper_broker_opens_position_matching_backtest_signal():
    df_15m, df_1h = _load_resampled()
    fake = FakeExchange(df_15m, df_1h)
    config = Config(symbols=["BTCUSDT"])
    broker = PaperBroker(fake, config, starting_equity=10_000.0)

    # Same SHORT entry the backtest found at 2026-01-21 16:45:00.
    fake.now = pd.Timestamp("2026-01-21 17:00:00")
    broker.try_open("BTCUSDT")

    assert "BTCUSDT" in broker.open_positions
    pos = broker.open_positions["BTCUSDT"]
    assert pos.side == "short"
    assert broker.equity < 10_000.0  # entry fee deducted


def test_paper_broker_no_open_when_no_signal():
    df_15m, df_1h = _load_resampled()
    fake = FakeExchange(df_15m, df_1h)
    config = Config(symbols=["BTCUSDT"])
    broker = PaperBroker(fake, config, starting_equity=10_000.0)

    fake.now = pd.Timestamp("2026-01-03 00:15:00")
    broker.try_open("BTCUSDT")

    assert "BTCUSDT" not in broker.open_positions
    assert broker.equity == 10_000.0


def test_paper_broker_stop_loss_closes_and_deducts_correctly():
    df_15m, df_1h = _load_resampled()
    fake = FakeExchange(df_15m, df_1h)
    config = Config(symbols=["BTCUSDT"])
    broker = PaperBroker(fake, config, starting_equity=10_000.0)

    fake.now = pd.Timestamp("2026-01-21 17:00:00")
    broker.try_open("BTCUSDT")
    pos = broker.open_positions["BTCUSDT"]
    equity_after_entry = broker.equity

    # Drive price to the stop (this is a SHORT, so stop is above entry).
    broker.manage_with_price("BTCUSDT", pos.stop)

    assert "BTCUSDT" not in broker.open_positions
    assert len(broker.closed_trades) == 1
    closed = broker.closed_trades[0]
    assert closed.exit_reason == "stop_loss"
    assert closed.pnl_usd < 0
    assert broker.equity < equity_after_entry


def test_paper_broker_tp1_then_trailing_and_summary():
    df_15m, df_1h = _load_resampled()
    fake = FakeExchange(df_15m, df_1h)
    config = Config(symbols=["BTCUSDT"])
    broker = PaperBroker(fake, config, starting_equity=10_000.0)

    fake.now = pd.Timestamp("2026-01-21 17:00:00")
    broker.try_open("BTCUSDT")
    pos = broker.open_positions["BTCUSDT"]

    # Push price down (favorable for this SHORT) to TP1.
    broker.manage_with_price("BTCUSDT", pos.tp1)
    assert broker.open_positions["BTCUSDT"].tp1_hit
    assert broker.open_positions["BTCUSDT"].stop == broker.open_positions["BTCUSDT"].entry

    # Summary should count this as one open, currently-winning position.
    fake.price_override = pos.tp1
    s = broker.summary()
    assert s["open_positions"] == 1
    assert s["open_currently_winning"] == 1
    assert s["closed_trades"] == 0
    assert s["unrealized_pnl_usd"] > 0

    # Now close it out via a further favorable move to tp2.
    fake.price_override = None
    broker.manage_with_price("BTCUSDT", broker.open_positions["BTCUSDT"].tp2)
    assert "BTCUSDT" not in broker.open_positions
    final = broker.summary()
    assert final["closed_trades"] == 1
    assert final["closed_wins"] == 1
    assert final["realized_pnl_usd"] > 0
