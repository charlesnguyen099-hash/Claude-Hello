"""Run the multi-timeframe model against a live exchange.

The one thing that must not go wrong here: the feature matrix the bot
builds from live bars has to be the SAME matrix the model was trained
on. Same views, same lookbacks, same resampling convention, same shift,
same column order. A single column out of place and the model is reading
volume where it expects momentum, silently, with real money behind it.

So the live path calls the SAME view_features() the training path
called, on 1-minute bars fetched from the exchange, and then asserts the
column order against what the trained model recorded. There is no second
implementation to drift.

HOW MUCH HISTORY. The deepest feature is the 60-minute view's 55-bar
lookback: 55 hours. Bybit serves 1000 bars a call, so 1-minute history
is paged backwards until there is enough, four calls per symbol, cached
until the entry bar rolls. Fetching each view at its own native interval
would be one call instead of four, but native 60m bars and 60m bars
resampled from 1m do not agree exactly at the edges, and "close enough"
is how a feature drifts away from its training distribution.
"""
from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fp import june as J
from fp import mtf as M

logger = logging.getLogger("fp.mtf_live")

HERE = Path(__file__).resolve().parent
PKL = HERE / "mtf_model.pkl"
META = HERE / "mtf_model.json"

# The deepest feature is NOT the 55-bar lookback it looks like. slope55 is
# rolling(55).mean().diff(55), so it needs 110 bars, and at the 60-minute
# view that is 110 hours = 6,600 one-minute bars. Sized at 7,200 (five
# days) so the newest row is finite with margin.
#
# This was wrong at 4,200 and the live bot would have produced NO signal
# at all, silently: features() returns None on any non-finite value, and
# v60_slope55 was the one column that never filled.
NEED_1M_BARS = 7200


class MTFModel:
    """The trained model plus everything needed to feed it correctly."""

    def __init__(self, pkl: Path = PKL, meta: Path = META):
        self.ok = pkl.exists() and meta.exists()
        if not self.ok:
            logger.warning("%s / %s missing -- run python -m fp.mtf_train",
                           pkl.name, meta.name)
            self.meta, self.models, self.columns = {}, {}, []
            return
        self.meta = json.loads(meta.read_text())
        self.models = pickle.loads(pkl.read_bytes())
        self.columns = list(self.meta["columns"])
        self.entry_min = int(self.meta["entry_minutes"])
        self.shapes = [tuple(s) for s in self.meta["shapes"]]
        self.gates = self.meta["gates"]
        # Views live in fp.mtf; assert rather than duplicate, so the two
        # cannot drift apart without something failing loudly.
        if list(self.meta["views"]) != list(M.VIEWS):
            raise ValueError(
                f"model was trained on views {self.meta['views']} but "
                f"fp.mtf.VIEWS is {M.VIEWS} -- retrain or revert")

    def realized_for(self, pred: float) -> float:
        """What the walk-forward measured for a prediction this size.

        The gate table is a step function measured on real trades, not a
        formula. A prediction below the lowest gate returns the lowest
        band's realized number, which is near zero -- that is the honest
        answer for a setup the study never saw pay.
        """
        best = self.gates[0]["realized_pct"]
        for g in self.gates:
            if pred >= g["gate"]:
                best = g["realized_pct"]
        return best

    def features(self, d1: pd.DataFrame) -> pd.DataFrame | None:
        """The model's feature row for the most recently closed entry bar.

        d1 is 1-minute OHLCV, ascending, newest bar already dropped by
        the caller if it is still forming.
        """
        de = J.resample(d1, self.entry_min) if self.entry_min > 1 else d1
        if len(de) < 4:
            return None
        X = pd.concat([M.view_features(d1, m, de.index, f"v{m}")
                       for m in M.VIEWS], axis=1)
        if list(X.columns) != self.columns:
            raise ValueError("live feature columns do not match the trained "
                             "model -- refusing to predict")
        tail = X.iloc[[-1]]
        if not np.isfinite(tail.values).all():
            return None
        return tail

    def predict(self, d1: pd.DataFrame):
        """Best (side, shape, predicted net) for this symbol right now.

        Every shape and both sides are scored; the highest predicted net
        wins. Potential picks the direction AND the exit, which is what
        makes the barriers dynamic rather than chosen.
        """
        if not self.ok:
            return None
        X = self.features(d1)
        if X is None:
            return None
        best = None
        for tp, sl, hm in self.shapes:
            for side, tag in ((1, "long"), (-1, "short")):
                key = f"{tp}_{sl}_{hm}_{tag}"
                m = self.models.get(key)
                if m is None:
                    continue
                p = float(m.predict(X)[0])
                if best is None or p > best["pred"]:
                    best = {"pred": p, "side": side, "tp": tp, "sl": sl,
                            "hmax": int(hm),
                            "hold_min": int(hm) * self.entry_min}
        return best


def fetch_1m(client, symbol: str, need: int = NEED_1M_BARS,
             counter=None) -> pd.DataFrame | None:
    """Page 1-minute bars backwards until there is enough history.

    Bybit returns newest-first and caps a call at 1000, so each page asks
    for bars ending just before the oldest one already held.
    """
    rows: list[list] = []
    end = None
    for _ in range(10):
        kw = dict(category="linear", symbol=symbol, interval="1", limit=1000)
        if end is not None:
            kw["end"] = end
        try:
            got = client.get_kline(**kw)["result"]["list"]
        except Exception:
            logger.debug("kline failed %s", symbol, exc_info=True)
            break
        if counter is not None:
            counter()
        if not got:
            break
        rows.extend(got)
        oldest = min(int(r[0]) for r in got)
        if len(rows) >= need:
            break
        end = oldest - 1
    if len(rows) < 400:
        return None
    d = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close",
                                    "volume", "turnover"])
    d["ts"] = pd.to_datetime(d["ts"].astype("int64"), unit="ms", utc=True)
    for c in ("open", "high", "low", "close", "volume"):
        d[c] = pd.to_numeric(d[c])
    d = (d.drop_duplicates(subset="ts").sort_values("ts")
         .set_index("ts")[["open", "high", "low", "close", "volume"]])
    # The newest 1-minute bar is still forming; the model was trained on
    # closed bars only.
    return d.iloc[:-1]
