"""Replay the shipped book against the bars it actually traded on.

The 34 hours in data/aug0507_1m.csv.gz cover the live session that lost
0.39%. Every earlier study measured rules on data they were selected
from; this is the first time the book's own rules can be fired on bars
that include the ones the bot was live for, with the same barrier code
the bot uses, and the result compared with what the bot reported.

Two jobs:

  diagnose   fire every book rule here and see which ones are broken,
             which are merely unlucky, and which never fire at all
  tiers      search this window for per-coin, per-group and all-coin
             logic, under the multiple-testing control the earlier
             studies established

The window is short -- 136 fifteen-minute bars per symbol -- and it
OVERLAPS the fitting window of every book except btc_book. Both facts
bound what can honestly be claimed from it and are reported with every
number rather than mentioned once.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fp.exits import barrier_outcomes, sigma_at
from fp.logic import FEE_ROUND_TRIP

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "aug0507_1m.csv.gz"
BOOK = HERE / "book.json"

# Funding is charged every 8h; the window is 34h, so it is not optional.
FUNDING_PER_8H = 0.0001

TF_MIN = {"15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}


def load() -> dict[str, pd.DataFrame]:
    """One OHLCV frame per symbol, 1-minute bars, ascending."""
    d = pd.read_csv(DATA)
    d["ts"] = pd.to_datetime(d["ts"], utc=True)
    out = {}
    for s, g in d.groupby("symbol"):
        g = g.sort_values("ts").set_index("ts")
        out[s] = g[["open", "high", "low", "close", "volume"]].astype(float)
    return out


def resample(d: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Down-sample to a coarser bar. Right-closed, right-labelled, so a
    bar is stamped when it CLOSES -- the same convention every earlier
    study used and the only one under which a signal on bar i can be
    acted on at bar i's close."""
    r = d.resample(f"{minutes}min", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last", "volume": "sum"}).dropna()
    return r


def net_outcomes(d: pd.DataFrame, side: int, tp: float, sl: float,
                 hmax: int, minutes: int) -> tuple[np.ndarray, np.ndarray]:
    """Barrier outcome per bar, net of the round trip and of funding.

    Identical to what the books were measured with: sigma at entry sets
    the barrier distances, the first touch on HIGH/LOW decides, and a
    bar touching both is booked as the loss.
    """
    close = d["close"].values.astype(float)
    high = d["high"].values.astype(float)
    low = d["low"].values.astype(float)
    sigma = sigma_at(close)
    o, held = barrier_outcomes(high, low, close, sigma, side, tp, sl, hmax)
    hours = held * minutes / 60.0
    return o - FEE_ROUND_TRIP - hours / 8.0 * FUNDING_PER_8H, held


def book_rules() -> list[dict]:
    return json.loads(BOOK.read_text())["logics"]


def fire(logics: dict, name: str, side: int) -> np.ndarray | None:
    """Bars where this logic TURNS ON in this direction -- the same
    definition the bot enters on, so the replay trades what the bot
    would have traded rather than everything the logic ever said."""
    p = logics.get(name)
    if p is None:
        return None
    v = p.values.astype(float)
    on = np.zeros(len(v), dtype=bool)
    on[1:] = (v[1:] == side) & (v[:-1] != side)
    return on
