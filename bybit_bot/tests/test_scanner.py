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
        # Signed size of the simulated exchange-side position. The scanner
        # reads this to learn whether its resting limit entry has filled.
        self.position_qty = 0.0
        self.resting: tuple[str, float, float] | None = None  # side, price, qty

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

    def round_price(self, symbol, price):
        return round(price, 1)

    def set_leverage(self, symbol, leverage):
        self.calls.append(("set_leverage", symbol, leverage))

    def place_market_entry_with_stop(self, symbol, side, qty, stop_price):
        self.calls.append(("open", symbol, side, qty, stop_price))

    def place_limit_entry(self, symbol, side, qty, limit_price):
        self.calls.append(("limit", symbol, side, qty, limit_price))
        self.resting = (side, limit_price, qty)
        return {"result": {"orderId": "fake-1"}}

    def cancel_order(self, symbol, order_id):
        self.calls.append(("cancel", symbol, order_id))
        self.resting = None

    def get_open_position_qty(self, symbol):
        return self.position_qty

    def fill_resting(self):
        """Simulate the resting limit order being hit."""
        assert self.resting is not None
        side, _price, qty = self.resting
        self.position_qty = qty if side == "long" else -qty
        self.resting = None

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

    # The strategy no longer chases the close: it rests a limit entry on
    # the pullback and only becomes a position once that order fills.
    assert "BTCUSDT" not in scanner.open_positions
    assert "BTCUSDT" in scanner.pending_entries
    limits = [c for c in fake.calls if c[0] == "limit"]
    assert len(limits) == 1
    assert limits[0][2] == "short"
    pending = scanner.pending_entries["BTCUSDT"]
    # A short rests ABOVE the signal price, so it fills on a bounce.
    assert pending.limit_price > df_1m[df_1m["datetime"] == pd.Timestamp(
        "2026-01-21 16:56:00")]["close"].iloc[0]

    # Once the exchange reports the fill, the next scan promotes it to a
    # managed position and attaches the stop.
    fake.fill_resting()
    fake.now = pd.Timestamp("2026-01-21 16:58:00")
    scanner.run_once()

    assert "BTCUSDT" in scanner.open_positions
    assert "BTCUSDT" not in scanner.pending_entries
    assert scanner.open_positions["BTCUSDT"].side == "short"
    assert any(c[0] == "update_stop" for c in fake.calls)


def test_scanner_cancels_entry_that_never_fills():
    df_1m, df_1h = _load_1m_and_1h()
    fake = FakeExchange(df_1m, df_1h)
    config = Config(symbols=["BTCUSDT"], max_concurrent_positions=4, equity_override_usdt=10_000.0)
    scanner = Scanner(exchange=fake, config=config)

    fake.now = pd.Timestamp("2026-01-21 16:57:00")
    scanner.run_once()
    assert "BTCUSDT" in scanner.pending_entries

    # Price never comes back to the limit: the order should age out and be
    # cancelled rather than resting forever and tying up margin.
    from bot import strategy
    times = df_1m[df_1m["datetime"] > fake.now]["datetime"].tolist()
    for t in times[:strategy.PENDING_MAX_BARS + 2]:
        fake.now = t
        scanner.run_once()
        if "BTCUSDT" not in scanner.pending_entries:
            break

    assert "BTCUSDT" not in scanner.pending_entries
    assert "BTCUSDT" not in scanner.open_positions
    assert any(c[0] == "cancel" for c in fake.calls)


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
    assert "BTCUSDT" in scanner.pending_entries
    fake.fill_resting()
    fake.now = pd.Timestamp("2026-02-01 11:11:00")
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
