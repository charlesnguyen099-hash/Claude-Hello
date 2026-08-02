"""Tests for bot/paper_trading.py against a fake exchange (no network),
fed from the real BTCUSDT 2026 dataset. Verifies the simulated fill/fee
accounting and the end-of-session summary math, without ever touching
Bybit.
"""
from __future__ import annotations

import pandas as pd

from bot.config import Config
from bot.paper_trading import PaperBroker


def _load_1m_and_1h(csv_path="data/BTCUSDT_2026.csv"):
    df = pd.read_csv(csv_path, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["datetime"])
    df_1m = df.reset_index(drop=True)
    dfi = df.set_index("datetime")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    df_1h = dfi.resample("1h").agg(agg).dropna().reset_index()
    return df_1m, df_1h


class FakeExchange:
    def __init__(self, df_1m, df_1h):
        self.df_1m = df_1m
        self.df_1h = df_1h
        self.now: pd.Timestamp | None = None
        self.price_override: float | None = None
        self.klines_override: pd.DataFrame | None = None

    def get_klines(self, symbol, timeframe, limit=300):
        if self.klines_override is not None and timeframe == "1m":
            return self.klines_override.tail(limit).reset_index(drop=True)
        src = self.df_1m if timeframe == "1m" else self.df_1h
        return src[src["datetime"] <= self.now].tail(limit).reset_index(drop=True)

    def get_last_price(self, symbol):
        if self.price_override is not None:
            return self.price_override
        row = self.df_1m[self.df_1m["datetime"] <= self.now].iloc[-1]
        return float(row["close"])


def test_paper_broker_opens_position_matching_backtest_signal():
    df_1m, df_1h = _load_1m_and_1h()
    fake = FakeExchange(df_1m, df_1h)
    config = Config(symbols=["BTCUSDT"])
    broker = PaperBroker(fake, config, starting_equity=10_000.0)

    # Same SHORT entry the backtest found at 2026-01-21 16:56:00.
    fake.now = pd.Timestamp("2026-01-21 16:57:00")
    broker.try_open("BTCUSDT")

    assert "BTCUSDT" in broker.open_positions
    pos = broker.open_positions["BTCUSDT"]
    assert pos.side == "short"
    assert broker.equity < 10_000.0  # entry fee deducted


def test_paper_broker_no_open_when_no_signal():
    df_1m, df_1h = _load_1m_and_1h()
    fake = FakeExchange(df_1m, df_1h)
    config = Config(symbols=["BTCUSDT"])
    broker = PaperBroker(fake, config, starting_equity=10_000.0)

    fake.now = pd.Timestamp("2026-01-03 00:15:00")
    broker.try_open("BTCUSDT")

    assert "BTCUSDT" not in broker.open_positions
    assert broker.equity == 10_000.0


def test_paper_broker_stop_loss_closes_and_deducts_correctly():
    df_1m, df_1h = _load_1m_and_1h()
    fake = FakeExchange(df_1m, df_1h)
    config = Config(symbols=["BTCUSDT"])
    broker = PaperBroker(fake, config, starting_equity=10_000.0)

    fake.now = pd.Timestamp("2026-01-21 16:57:00")
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
    df_1m, df_1h = _load_1m_and_1h()
    fake = FakeExchange(df_1m, df_1h)
    config = Config(symbols=["BTCUSDT"])
    broker = PaperBroker(fake, config, starting_equity=10_000.0)

    fake.now = pd.Timestamp("2026-01-21 16:57:00")
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


def test_intrabar_wick_triggers_stop_even_if_candle_closes_safe():
    """A wick that touches the stop and retraces must still stop us out.

    Polling `get_last_price` every N seconds can step right over such a
    wick; a real stop order on the exchange would not. manage_with_candles
    scans candle high/low, so it catches it.
    """
    df_1m, df_1h = _load_1m_and_1h()
    fake = FakeExchange(df_1m, df_1h)
    config = Config(symbols=["BTCUSDT"])
    broker = PaperBroker(fake, config, starting_equity=10_000.0)

    fake.now = pd.Timestamp("2026-01-21 16:57:00")
    broker.try_open("BTCUSDT")
    pos = broker.open_positions["BTCUSDT"]
    assert pos.side == "short"  # stop sits ABOVE entry

    # One candle whose high pierces the stop but whose close is back at
    # entry -- i.e. exactly the case a last-price poll would miss.
    spike = pd.DataFrame([
        {"datetime": pd.Timestamp("2026-01-21 16:58:00"), "open": pos.entry,
         "high": pos.stop + 1.0, "low": pos.entry, "close": pos.entry, "volume": 1.0},
        {"datetime": pd.Timestamp("2026-01-21 16:59:00"), "open": pos.entry,
         "high": pos.entry, "low": pos.entry, "close": pos.entry, "volume": 1.0},
    ])
    fake.klines_override = spike

    # Sanity: the "current price" never leaves entry, so a price-only
    # check sees nothing wrong.
    broker.manage_with_price("BTCUSDT", pos.entry)
    assert "BTCUSDT" in broker.open_positions

    broker.manage_with_candles("BTCUSDT")
    assert "BTCUSDT" not in broker.open_positions
    assert broker.closed_trades[0].exit_reason == "stop_loss"


def test_summary_combines_closed_and_open_counts():
    df_1m, df_1h = _load_1m_and_1h()
    fake = FakeExchange(df_1m, df_1h)
    config = Config(symbols=["BTCUSDT"])
    broker = PaperBroker(fake, config, starting_equity=10.0)

    fake.now = pd.Timestamp("2026-01-21 16:57:00")
    broker.try_open("BTCUSDT")
    pos = broker.open_positions["BTCUSDT"]

    # Close one trade at a loss.
    broker.manage_with_price("BTCUSDT", pos.stop)
    assert len(broker.closed_trades) == 1

    # Re-open and leave it open, in profit (SHORT -> price below entry).
    fake.now = pd.Timestamp("2026-01-21 16:57:00")
    broker.try_open("BTCUSDT")
    open_pos = broker.open_positions["BTCUSDT"]
    fake.price_override = open_pos.entry * 0.99

    s = broker.summary()
    assert s["closed_trades"] == 1 and s["closed_losses"] == 1
    assert s["open_positions"] == 1 and s["open_currently_winning"] == 1
    # Combined view spans both.
    assert s["total_trades"] == 2
    assert s["total_winning"] == 1
    assert s["total_losing"] == 1
    assert s["gross_loss_usd"] < 0
    assert s["open_profit_usd"] > 0
    assert s["total_profit_usd"] == s["gross_profit_usd"] + s["open_profit_usd"]


def test_export_trades_csv_includes_open_positions(tmp_path):
    df_1m, df_1h = _load_1m_and_1h()
    fake = FakeExchange(df_1m, df_1h)
    config = Config(symbols=["BTCUSDT"])
    broker = PaperBroker(fake, config, starting_equity=10.0)

    fake.now = pd.Timestamp("2026-01-21 16:57:00")
    broker.try_open("BTCUSDT")
    pos = broker.open_positions["BTCUSDT"]
    broker.manage_with_price("BTCUSDT", pos.stop)
    fake.now = pd.Timestamp("2026-01-21 16:57:00")
    broker.try_open("BTCUSDT")

    out = tmp_path / "trades.csv"
    broker.export_trades_csv(str(out))
    rows = pd.read_csv(out)
    assert set(rows["status"]) == {"closed", "open_at_shutdown"}
    assert len(rows) == 2
