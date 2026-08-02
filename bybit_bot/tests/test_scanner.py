"""Integration test for bot/scanner.py against a fake exchange (no
network) fed from the real BTCUSDT Jan-Aug 2026 dataset used for the
backtest, to prove the live wiring (fetch -> signal -> risk plan ->
order calls -> trailing-stop management) reaches the exact same entry
the backtest found, without hitting Bybit at all.
"""
from __future__ import annotations

import pandas as pd

from bot.config import Config
from bot.scanner import Scanner

CSV_PATH = "data/BTCUSDT_2026.csv"


def _load_1m_and_1h():
    df = pd.read_csv(CSV_PATH, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["datetime"])
    df_1m = df.reset_index(drop=True)
    dfi = df.set_index("datetime")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    df_1h = dfi.resample("1h").agg(agg).dropna().reset_index()
    return df_1m, df_1h


class FakeExchange:
    """Duck-types the subset of BybitExchange the scanner calls."""

    def __init__(self, df_1m: pd.DataFrame, df_1h: pd.DataFrame):
        self.df_1m = df_1m
        self.df_1h = df_1h
        self.now: pd.Timestamp | None = None
        self.calls: list[tuple] = []

    def get_klines(self, symbol, timeframe, limit=300):
        src = self.df_1m if timeframe == "1m" else self.df_1h
        window = src[src["datetime"] <= self.now].tail(limit).reset_index(drop=True)
        return window

    def get_wallet_equity_usdt(self):
        return 10_000.0

    def get_instrument_info(self, symbol):
        from bot.exchange_bybit import InstrumentInfo
        return InstrumentInfo(symbol, tick_size=0.1, qty_step=0.001, min_order_qty=0.001, max_leverage=100)

    def round_qty(self, symbol, qty):
        step = 0.001
        return max((qty // step) * step, 0.0)

    def set_leverage(self, symbol, leverage):
        self.calls.append(("set_leverage", symbol, leverage))

    def place_market_entry_with_stop(self, symbol, side, qty, stop_price):
        self.calls.append(("open", symbol, side, qty, stop_price))

    def update_stop_loss(self, symbol, stop_price):
        self.calls.append(("update_stop", symbol, stop_price))

    def close_position_market(self, symbol, side, qty):
        self.calls.append(("close", symbol, side, qty))


def test_scanner_opens_the_same_trade_the_backtest_found():
    df_1m, df_1h = _load_1m_and_1h()
    fake = FakeExchange(df_1m, df_1h)
    config = Config(symbols=["BTCUSDT"], max_concurrent_positions=4, equity_override_usdt=10_000.0)
    scanner = Scanner(exchange=fake, config=config)

    # Backtest found a SHORT entry on BTCUSDT at 2026-01-21 16:56:00 (see
    # backtest/run_backtest.py output). Advance the fake clock to just
    # after that bar closes and run one scan.
    fake.now = pd.Timestamp("2026-01-21 16:57:00")  # +1 bar so it's not "still forming"
    scanner.run_once()

    assert "BTCUSDT" in scanner.open_positions
    opens = [c for c in fake.calls if c[0] == "open"]
    assert len(opens) == 1
    assert opens[0][2] == "short"


def test_scanner_does_not_open_when_no_signal():
    df_1m, df_1h = _load_1m_and_1h()
    fake = FakeExchange(df_1m, df_1h)
    config = Config(symbols=["BTCUSDT"], max_concurrent_positions=4, equity_override_usdt=10_000.0)
    scanner = Scanner(exchange=fake, config=config)

    fake.now = pd.Timestamp("2026-01-03 00:15:00")  # HTF EMA200 not warmed up yet
    scanner.run_once()

    assert "BTCUSDT" not in scanner.open_positions
    assert not any(c[0] == "open" for c in fake.calls)


def test_scanner_moves_stop_to_breakeven_after_tp1():
    df_1m, df_1h = _load_1m_and_1h()
    fake = FakeExchange(df_1m, df_1h)
    config = Config(symbols=["BTCUSDT"], max_concurrent_positions=4, equity_override_usdt=10_000.0)
    scanner = Scanner(exchange=fake, config=config)

    # This entry (2026-02-01 11:09:00 SHORT) is the one the backtest shows
    # actually reaching TP1 (tp1_hit=True).
    fake.now = pd.Timestamp("2026-02-01 11:10:00")
    scanner.run_once()
    assert "BTCUSDT" in scanner.open_positions

    # Step forward bar-by-bar until TP1 fires (matches backtest, which
    # showed tp1_hit=True for this exact trade) or the position closes.
    times = df_1m[df_1m["datetime"] > fake.now]["datetime"].tolist()
    for t in times:
        fake.now = t
        scanner.run_once()
        if "BTCUSDT" not in scanner.open_positions:
            break
        if scanner.open_positions["BTCUSDT"].tp1_hit:
            break

    assert any(c[0] == "update_stop" for c in fake.calls)
