"""The June block: one month of 1-minute bars on all ten symbols.

2026-05-31..06-29, 387,360 rows. Long enough that the logic library
builds at the timeframes the book actually uses -- 2,582 fifteen-minute
bars per symbol against the 136 the August window could offer -- and it
ENDS the day the earliest book's fitting window begins, so for every
rule in fp/book.json this is out-of-sample.

Two questions, in order of how much they cost to answer:

  validate   do the 104 rules already shipped survive here? That is 104
             pre-specified tests, not a search, so the correction is
             cheap and the answer is worth having first.
  discover   what per-coin, group and all-coin logic does this month
             contain, under the same controls every earlier study used.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fp.exits import barrier_outcomes, sigma_at
from fp.logic import FEE_ROUND_TRIP as FEE

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "all_1m.csv.gz"
BOOK = HERE / "book.json"

FUNDING_PER_8H = 1e-4
TF_MIN = {"5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}

# The block that predates every book's fitting window.
JUNE = ("2026-05-31", "2026-06-30")

BLOCK = {"SKHYNIXUSDT": "semi", "SNDKUSDT": "semi", "SOXLUSDT": "semi",
         "ETHUSDT": "crypto", "SOLUSDT": "crypto", "BTCUSDT": "crypto",
         "XRPUSDT": "crypto", "HYPEUSDT": "crypto",
         "BLESSUSDT": "bless", "XAUUSDT": "gold"}


def load(start: str = JUNE[0], end: str = JUNE[1]) -> dict[str, pd.DataFrame]:
    d = pd.read_csv(DATA)
    d["ts"] = pd.to_datetime(d["ts"], utc=True)
    d = d[(d.ts >= start) & (d.ts < end)]
    out = {}
    for s, g in d.groupby("symbol"):
        g = g.sort_values("ts").set_index("ts")
        out[s] = g[["open", "high", "low", "close", "volume"]].astype(float)
    return out


def resample(d: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Right-closed, right-labelled: a bar is stamped when it CLOSES, so a
    signal on bar i can be acted on at bar i's close and not before."""
    if minutes <= 1:
        return d
    return d.resample(f"{minutes}min", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last", "volume": "sum"}).dropna()


def net(d: pd.DataFrame, minutes: int, side: int, tp: float, sl: float,
        hmax: int) -> np.ndarray:
    """Net barrier outcome per bar: taker both sides, funding on real hours."""
    c = d["close"].values.astype(float)
    h = d["high"].values.astype(float)
    lo = d["low"].values.astype(float)
    o, held = barrier_outcomes(h, lo, c, sigma_at(c), side, tp, sl, hmax)
    return o - FEE - held * minutes / 60.0 / 8.0 * FUNDING_PER_8H


def turns_on(series: pd.Series, side: int) -> np.ndarray:
    """Bars where a logic TURNS ON in this direction -- the same entry the
    bot takes, so a replay trades what the bot would have traded."""
    v = series.values.astype(float)
    on = np.zeros(len(v), dtype=bool)
    on[1:] = (v[1:] == side) & (v[:-1] != side)
    return on


def book_rules() -> list[dict]:
    return json.loads(BOOK.read_text())["logics"]
