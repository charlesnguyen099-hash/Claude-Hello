"""Multi-timeframe signal recognition.

The concept, and it is a good one: before a trade that pays after fees,
there is usually something visible at several horizons at once -- one
minute, five, fifteen, thirty. A rule that looks at one timeframe sees
one slice of that and misses the confluence.

Everything tested in this repo so far has been single-timeframe rule
enumeration, which failed its rotation null three times. This is a
different question and deserves its own answer:

  LABEL      on the entry timeframe, mark every bar with what a trade
             opened there actually earned, net of both taker fees and
             funding. That is the ground truth -- "the profitable trade"
             -- and it is only knowable in hindsight, which is exactly
             why it is the target and never a feature.
  FEATURES   the state of the market at that bar as seen from SEVERAL
             timeframes at once, every one of them built from bars that
             had already closed. A 30-minute feature at 09:07 uses the
             30-minute bar that closed at 09:00, never the one forming.
  LEARN      fit a model from the multi-timeframe state to the label.
  TEST       on a later period the model has never seen, and price the
             result at full cost.

What separates this from the searches that failed: a search asks a
million yes/no questions and keeps whichever happened to look good. A
model asks one question -- is the label predictable from this state --
and is scored once, out of sample. The multiple-testing burden that
killed the rule enumeration does not apply, because there is nothing
being selected after the fact.

If the out-of-sample edge does not clear the round trip, this dies the
same way the others did, and it says so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fp import june as J
from fp.exits import barrier_outcomes, sigma_at

# Horizons the state is read from. The entry bar is the finest one; the
# rest are what "confluence" means -- the same instant described by
# progressively slower observers.
VIEWS = [1, 5, 15, 30, 60]

FEE = J.FEE
FUNDING_PER_8H = J.FUNDING_PER_8H


def view_features(d1: pd.DataFrame, minutes: int, index: pd.DatetimeIndex,
                  tag: str) -> pd.DataFrame:
    """State at one horizon, aligned onto the entry bar's clock.

    Two rules make this honest and both are easy to get wrong:

      * the coarse bar is resampled label='right', closed='right', so it
        is stamped at the moment it CLOSES;
      * it is then shifted one bar before being forward-filled onto the
        fine index, so the value standing at 09:07 is the 30-minute bar
        that closed at 09:00 -- not the one that will close at 09:30.

    Without the shift a 30-minute feature at 09:07 knows how the next
    twenty-three minutes went, which is the whole trade.
    """
    d = J.resample(d1, minutes) if minutes > 1 else d1
    c = d["close"].astype(float)
    h = d["high"].astype(float)
    lo = d["low"].astype(float)
    v = d["volume"].astype(float)
    r = np.log(c).diff()

    f = pd.DataFrame(index=d.index)
    # Direction and strength at several lookbacks within this horizon.
    for w in (3, 8, 21, 55):
        f[f"ret{w}"] = np.log(c / c.shift(w))
        f[f"z{w}"] = ((c - c.rolling(w).mean()) /
                      c.rolling(w).std().replace(0, np.nan))
        f[f"slope{w}"] = c.rolling(w).mean().diff(w) / c
    # Volatility, and whether it is rising or falling.
    for w in (8, 21, 55):
        f[f"sd{w}"] = r.rolling(w).std()
        f[f"rng{w}"] = ((h.rolling(w).max() - lo.rolling(w).min()) / c)
    f["volratio"] = (r.rolling(8).std() /
                     r.rolling(55).std().replace(0, np.nan))
    # Where price sits inside its own recent range.
    for w in (21, 55):
        hi, low = h.rolling(w).max(), lo.rolling(w).min()
        f[f"pos{w}"] = (c - low) / (hi - low).replace(0, np.nan)
    # Participation.
    f["vz21"] = (v - v.rolling(21).mean()) / v.rolling(21).std().replace(0, np.nan)
    f["vtrend"] = v.rolling(8).mean() / v.rolling(55).mean().replace(0, np.nan)
    # Shape of the last bar at this horizon.
    body = (c - d["open"].astype(float)).abs()
    span = (h - lo).replace(0, np.nan)
    f["body"] = body / span
    f["upwick"] = (h - np.maximum(c, d["open"].astype(float))) / span
    f["dnwick"] = (np.minimum(c, d["open"].astype(float)) - lo) / span
    # Persistence: how many of the last bars went the same way.
    f["updens21"] = (r > 0).rolling(21).mean()

    f = f.add_prefix(f"{tag}_")
    # THE SHIFT. Everything above is known only once the bar closes.
    return f.shift(1).reindex(index, method="ffill")


def build_panel(d1: pd.DataFrame, entry_min: int, tp: float, sl: float,
                hmax: int):
    """Features and label for one symbol at one entry timeframe.

    Returns (X, net_long, net_short, held_long, held_short). The holds
    come back too: two entries whose barrier windows overlap are not two
    observations, and only the exit bar says where a window ends. An
    earlier study in this repo read t = 7.81 on overlapping trades and
    t = 2.26 on the independent ones from the same rule.
    """
    de = J.resample(d1, entry_min) if entry_min > 1 else d1
    idx = de.index
    X = pd.concat([view_features(d1, m, idx, f"v{m}") for m in VIEWS], axis=1)

    c = de["close"].values.astype(float)
    h = de["high"].values.astype(float)
    lo = de["low"].values.astype(float)
    sg = sigma_at(c)
    out, hold = {}, {}
    for side in (1, -1):
        o, held = barrier_outcomes(h, lo, c, sg, side, tp, sl, hmax)
        out[side] = o - FEE - held * entry_min / 60.0 / 8.0 * FUNDING_PER_8H
        hold[side] = held
    return X, out[1], out[-1], hold[1], hold[-1]
