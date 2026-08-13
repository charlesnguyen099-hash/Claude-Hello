"""The bar cache and the exchange bill. Two things, no opinions.

Everything else in this package is a hypothesis about the market. This
file is not: it loads one-minute bars and states what Bybit charges. If
either number here is wrong, every result downstream is wrong in the same
direction, so they are stated once and imported everywhere rather than
retyped.

THE BILL. Bybit VIP0 linear perpetuals: 0.055% taker per side. A target
and a stop are both market orders when they fire, so a barrier trade pays
taker twice and cannot earn the maker rate on the way out. Funding is
charged every eight hours on notional; the live bot reads each symbol's
actual rate off the ticker, and this average is what the studies use.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "data" / "all_1m.csv.gz"

TAKER_PER_SIDE = 0.00055
FEE = 2 * TAKER_PER_SIDE          # 0.110% round trip, both sides taker
FUNDING_PER_8H = 1e-4

_MEM: dict = {}


def load(start: str | None = None, end: str | None = None,
         symbols: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """One-minute OHLCV per symbol, ascending, in [start, end).

    The whole cache is parsed once per process and sliced from memory
    after that: a walk-forward reads the same file six times, and parsing
    a million rows six times is most of its runtime.
    """
    if "all" not in _MEM:
        d = pd.read_csv(CACHE)
        d["ts"] = pd.to_datetime(d["ts"], utc=True)
        _MEM["all"] = d.sort_values(["symbol", "ts"])
    d = _MEM["all"]
    if start is not None:
        d = d[d.ts >= start]
    if end is not None:
        d = d[d.ts < end]
    if symbols:
        d = d[d.symbol.isin(symbols)]
    out = {}
    for s, g in d.groupby("symbol", sort=True):
        g = g.set_index("ts")[["open", "high", "low", "close", "volume"]]
        out[s] = g.astype("float64")
    return out


def resample(d: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Right-closed, right-labelled: a bar is stamped when it CLOSES, so a
    signal on bar i can be acted on at i's close and not one second
    before."""
    if minutes <= 1:
        return d
    return d.resample(f"{minutes}min", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last", "volume": "sum"}).dropna()


def coverage() -> pd.DataFrame:
    """What the cache actually holds, per day. Gaps are not obvious from
    a row count -- this repo once walked forward across a month that was
    not in the file and read 'no data' as 'no signal'."""
    d = _MEM.get("all")
    if d is None:
        load()
        d = _MEM["all"]
    g = d.assign(day=d.ts.dt.date).groupby("day").symbol.nunique()
    return g
