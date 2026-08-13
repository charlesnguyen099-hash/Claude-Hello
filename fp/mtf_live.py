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
BANDS = HERE / "mtf_bands.json"

# A band is tradeable only if the walk-forward measured it POSITIVE and
# did so with more than two standard errors behind it. Both conditions
# are read from the file, not chosen here.
MIN_BAND_T = 2.0

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

    def __init__(self, pkl: Path = PKL, meta: Path = META,
                 gate: str = "band"):
        # "band"  -- trade only bands the study measured profitable.
        #            With the out-of-sample table that is NOTHING, which
        #            is the honest reading and also a bot that does not
        #            move.
        # "probe" -- trade every shape and side at a fixed small stake and
        #            let each one earn or lose its way out of probing on
        #            its OWN live record. No in-sample number is used.
        self.gate = gate
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
        self.gates = self.meta.get("gates", [])
        # MARGINAL bands: what a trade landing IN a band is worth. The
        # first shipped table was CUMULATIVE and used as if it were
        # per-band, so the bot traded two bands the study had measured as
        # losing -- which is exactly what the first live session did, at
        # -0.101%/trade in the bottom band.
        try:
            self.bands = json.loads(BANDS.read_text())
        except Exception:
            self.bands = []
        self.paying = [b for b in self.bands
                       if b.get("edge_over_b", 0) > 0 and b["t"] >= MIN_BAND_T]
        if not self.paying and self.gate == "band":
            logger.warning("no band in %s measured positive at t >= %.1f -- "
                           "nothing will be traded", BANDS.name, MIN_BAND_T)
        # Views live in fp.mtf; assert rather than duplicate, so the two
        # cannot drift apart without something failing loudly.
        if list(self.meta["views"]) != list(M.VIEWS):
            raise ValueError(
                f"model was trained on views {self.meta['views']} but "
                f"fp.mtf.VIEWS is {M.VIEWS} -- retrain or revert")

    def band_for(self, pred: float):
        """The MARGINAL band this prediction falls in, or None.

        None means the band was not measured profitable, and a setup
        whose band was not measured profitable is not a trade -- however
        confident the raw prediction looks. Six bands were measured and
        one pays; the other five include the two the first live session
        spent its money in.
        """
        for b in self.paying:
            if b["lo"] <= pred < b["hi"]:
                return b
        return None

    def edge_over_b(self, pred: float) -> float:
        """What a trade in this band earned as a SHARE of what its own win
        pays. Dimensionless, so it applies to any barrier width.

        A mean net cannot: +1.5%/trade averaged across barrier widths is
        impossible for a target only 0.16% wide, and the potential score
        correctly refuses it -- which is how the first version of this
        gate blocked every setup it was supposed to size.
        """
        b = self.band_for(pred)
        return b["edge_over_b"] if b else 0.0

    def realized_for(self, pred: float) -> float:
        """Kept for the dashboard: the band's share, as a percent."""
        return 100.0 * self.edge_over_b(pred)

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

    def predict_all(self, d1: pd.DataFrame) -> list[dict]:
        """EVERY (shape, side) whose band was measured profitable.

        Returning only the single best collapsed a symbol to one position
        at a time. With holds up to eight hours on five-minute bars, ten
        symbols then produced about fourteen trades in thirteen hours --
        not because the market was quiet but because each symbol was
        occupied. Each qualifying combination now gets its own slot, so a
        symbol can carry up to six, and none of them is admitted on a
        lower standard than the others.
        """
        if not self.ok:
            return []
        X = self.features(d1)
        if X is None:
            return []
        out = []
        for tp, sl, hm in self.shapes:
            for side, tag in ((1, "long"), (-1, "short")):
                m = self.models.get(f"{tp}_{sl}_{hm}_{tag}")
                if m is None:
                    continue
                p = float(m.predict(X)[0])
                band = self.band_for(p)
                if self.gate == "band":
                    if band is None:
                        continue
                    eob, lab = band["edge_over_b"], \
                        f"{band['lo']:.3f}-{band['hi']:.3f}"
                    bt, bn = band["t"], band["trades"]
                elif band is None:
                    # Probing. The prediction still ORDERS the candidates,
                    # because something has to, but it makes no claim
                    # about what the trade is worth -- out of sample its
                    # rank correlation with the outcome was -0.0096, and a
                    # stake sized off that would be sized off noise.
                    eob, lab, bt, bn = 0.0, "probe", 0.0, 0
                else:
                    eob, lab = band["edge_over_b"], \
                        f"{band['lo']:.3f}-{band['hi']:.3f}"
                    bt, bn = band["t"], band["trades"]
                out.append({"pred": p, "side": side, "tp": tp, "sl": sl,
                            "hmax": int(hm),
                            "hold_min": int(hm) * self.entry_min,
                            "edge_over_b": eob, "band": lab,
                            "band_t": bt, "band_n": bn})
        return sorted(out, key=lambda r: -r["pred"])

    def predict(self, d1: pd.DataFrame):
        """The single best qualifying setup, or None."""
        got = self.predict_all(d1)
        return got[0] if got else None


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
