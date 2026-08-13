"""Run the shipped logic against a live exchange.

ONE THING MUST NOT GO WRONG: the factor matrix built from live bars has
to BE the matrix the model was fitted on. Same lookbacks, same windows,
same volatility yardstick, same column order. So the live path calls the
SAME fp.factors.build() the training path called and then asserts the
column order against what the model recorded. There is no second
implementation that could drift.

WHY THE WHOLE BOARD IS FETCHED AT ONCE. A third of the factors are
cross-sectional -- this coin's rank among the ten, the board's median
move, the dispersion, the residual. They cannot be computed one symbol at
a time. So the live path pulls every scanned symbol, aligns them on a
common index and builds the panel together, exactly as training did. A
symbol missing from that pull changes the rank of every other symbol,
which is why a short pull is refused outright rather than filled in.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from fp import engine as E
from fp import factors as F

logger = logging.getLogger("fp.live_engine")

HERE = Path(__file__).resolve().parent
PKL = HERE / "engine_model.pkl"
META = HERE / "engine_meta.json"


class Engine:
    """The trained logic plus everything needed to feed it correctly."""

    def __init__(self, pkl: Path = PKL, meta: Path = META):
        self.ok = pkl.exists() and meta.exists()
        self.models, self.columns, self.shapes, self.meta = {}, [], [], {}
        self.gate = 20.0
        self.need_bars = 1500
        if not self.ok:
            logger.warning("%s / %s missing -- run python -m fp.run_engine "
                           "then python -m fp.train_engine", pkl.name,
                           meta.name)
            return
        self.meta = json.loads(meta.read_text())
        raw = pickle.loads(pkl.read_bytes())
        self.models = {eval(k) if isinstance(k, str) else k: v
                       for k, v in raw.items()}
        self.columns = list(self.meta.get("columns", []))
        self.shapes = [tuple(s) for s in self.meta.get("shapes", [])]
        self.gate = float(self.meta.get("gate") or 20.0)
        self.need_bars = int(self.meta.get("need_1m_bars", 1500))
        if not self.models:
            logger.warning("engine_model.pkl is EMPTY -- no shape survived "
                           "the walk-forward, so nothing will be traded")
        # The factor library is the single source of truth for what the
        # lookbacks are; assert rather than duplicate, so the two cannot
        # drift apart without something failing loudly.
        if (list(self.meta.get("lookbacks", F.LOOKBACKS)) != list(F.LOOKBACKS)
                or list(self.meta.get("state_windows", F.STATE_WINDOWS))
                != list(F.STATE_WINDOWS)):
            raise ValueError(
                "the model was trained with different factor lookbacks than "
                "fp.factors defines -- retrain or revert")

    # -- feature side ---------------------------------------------------

    def panel(self, bars: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Factors for every symbol, cross-section included."""
        good = {s: d for s, d in bars.items()
                if d is not None and len(d) >= 400}
        if len(good) < 2:
            return {}
        X = F.build(good)
        for s, x in X.items():
            if list(x.columns) != self.columns:
                raise ValueError(
                    "live factor columns do not match the trained model -- "
                    "refusing to predict")
        return X

    def candidates(self, X: pd.DataFrame) -> list[dict]:
        """Every shape and side worth trading on this symbol's newest bar.

        Returns one entry per shape/side whose lower-bounded potential
        clears the gate the walk-forward fixed. The score IS the stake:
        that is the whole point of putting every shape on one 0-100 scale.
        """
        if not self.models or X is None or len(X) == 0:
            return []
        row = X.iloc[[-1]]
        v = row.values
        if not np.isfinite(v).all():
            return []
        sg = float(row["sigma"].iloc[0])
        if not np.isfinite(sg) or sg <= 0:
            return []
        out = []
        for key, fit in self.models.items():
            tp, sl, hold, side = key
            p = float(E.p_hat(fit, v)[0])
            p_be, a, b = E.break_even(tp, sl, hold, np.array([sg]))
            p_be = float(p_be[0])
            if b[0] <= 0:
                continue
            score = float(E.potential(np.array([p]), np.array([p_be]))[0])
            if score < self.gate:
                continue
            out.append({
                "tp": tp, "sl": sl, "hold_min": int(hold), "side": int(side),
                "p": p, "p_be": p_be, "score": score,
                "tp_dist": float(tp * sg * np.sqrt(hold)),
                "sl_dist": float(sl * sg * np.sqrt(hold)),
                "sigma": sg,
                "rule": f"eng:{tp}/{sl}/{hold}:"
                        f"{'long' if side > 0 else 'short'}",
            })
        return sorted(out, key=lambda r: -r["score"])


def fetch_1m(client, symbol: str, need: int, counter=None):
    """One-minute bars, newest closed bar last.

    Bybit caps a call at 1000 and returns newest-first, so this pages
    backwards until there is enough. The forming bar is dropped: the
    model was fitted on closed bars only.
    """
    rows: list[list] = []
    end = None
    for _ in range(4):
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
        if len(rows) >= need:
            break
        end = min(int(r[0]) for r in got) - 1
    if len(rows) < 400:
        return None
    d = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close",
                                    "volume", "turnover"])
    d["ts"] = pd.to_datetime(d["ts"].astype("int64"), unit="ms", utc=True)
    for c in ("open", "high", "low", "close", "volume"):
        d[c] = pd.to_numeric(d[c])
    d = (d.drop_duplicates(subset="ts").sort_values("ts")
         .set_index("ts")[["open", "high", "low", "close", "volume"]])
    return d.iloc[:-1]
