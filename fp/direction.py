"""Learn the direction from the trades that actually paid.

The operator's instruction, and it is the one framing that had not been
tried: the 2.5 million profitable trades already tell us BOTH things --
what the signals looked like at entry, and which way the price then went.
So stop searching for a rule and learn the mapping directly.

    features at bar i  ->  which side reached +1% FIRST

THE LABEL. A clean symmetric barrier: from bar i, does the price reach
+1% net on the long side before it reaches +1% net on the short side,
within HORIZON minutes? First touch wins, both barriers net of taker in,
taker out and funding. Three outcomes:

    +1   a long would have made +1% before a short could
    -1   the reverse
     0   neither, inside the horizon -- not a trade

This is exactly "kết quả đi theo hướng nào" as a number, and it is what a
bot must predict to trade at all.

THE FEATURES. Everything the system already computes:

    47   method states -- every classic technique's current reading
   153   factors -- price, flow, state and cross-section
   200   total

So the model sees the same evidence a discretionary trader would, and the
label is the answer that evidence is supposed to imply.

WHAT IT HAS TO BEAT. A +1% barrier against a 0.110% round trip pays
+0.89% and costs -1.11%, so break-even accuracy is

    1.11 / (1.11 + 0.89) = 55.5%

Below that, more trades lose more money, however many of the 2.5 million
they cover. The raw strategy vote measures 50.13%. This file asks whether
a model trained on the answers does better.

POTENTIAL DRIVES BOTH STAKE AND LEVERAGE. The model's calibrated
probability p becomes

    POTENTIAL = 100 x (p - p_be) / (1 - p_be)

0 at break-even, 100 at certainty -- and the operator's rule is that this
one number scales capital AND leverage together, each from its own floor
to its own ceiling. A setup the model is barely sure of takes the minimum
of both; one it is certain of takes the maximum of both.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fp import factors as F
from fp import methods as M
from fp.data import FEE, FUNDING_PER_8H

# How long a +1% move has to appear in.
HORIZON = 1440
WIN = 0.01

# Leverage moves with potential, exactly as the stake does.
LEV_MIN, LEV_MAX = 1.0, 10.0

MODEL = dict(max_depth=4, min_samples_leaf=2000, max_iter=150,
             l2_regularization=10.0, learning_rate=0.06,
             early_stopping=True, validation_fraction=0.12)


def label(close: np.ndarray, horizon: int = HORIZON, win: float = WIN):
    """Which side reaches +win net FIRST, per bar. +1, -1 or 0.

    STAGED, because the obvious version does not finish. Scanning up to
    `horizon` bars ahead for every bar is 846,640 x 1440 element
    comparisons on BTC alone, in a Python loop -- it ran for over an hour
    without clearing one coin. Almost every bar resolves within the first
    few minutes, so the window is opened in blocks: a rolling max/min
    says whether ANY bar in the block crosses, and only the handful that
    do get scanned precisely. Same answer, a fraction of the work.
    """
    n = len(close)
    out = np.zeros(n, dtype="int8")
    if n < 2:
        return out
    up = close * (1.0 + win + FEE)
    dn = close * (1.0 - (win + FEE))
    s = pd.Series(close)
    pending = np.ones(n, dtype=bool)
    pending[-1] = False

    lo_off = 1
    for block in (15, 60, 240, 720, horizon):
        hi_off = min(block, horizon)
        if hi_off < lo_off:
            continue
        w = hi_off - lo_off + 1
        # Max and min over close[i+lo_off : i+hi_off], for every i at once.
        fwd_max = s[::-1].rolling(w, min_periods=1).max()[::-1] \
            .shift(-lo_off).values
        fwd_min = s[::-1].rolling(w, min_periods=1).min()[::-1] \
            .shift(-lo_off).values
        hits = pending & (((fwd_max >= up) & np.isfinite(fwd_max))
                          | ((fwd_min <= dn) & np.isfinite(fwd_min)))
        for i in np.flatnonzero(hits):
            j0, j1 = i + lo_off, min(i + hi_off, n - 1)
            if j1 < j0:
                continue
            seg = close[j0:j1 + 1]
            u = np.flatnonzero(seg >= up[i])
            dd = np.flatnonzero(seg <= dn[i])
            fu = u[0] if len(u) else n + 1
            fd = dd[0] if len(dd) else n + 1
            if fu == fd:
                continue
            out[i] = 1 if fu < fd else -1
            pending[i] = False
        lo_off = hi_off + 1
        if lo_off > horizon or not pending.any():
            break
    return out


# What a cross-sectional factor means when there is no board to compare
# against: nothing. Rank sits in the middle, the market has not moved,
# dispersion is zero, the residual is zero, breadth is even.
XS_NEUTRAL = {"rank": 0.5, "breadth": 0.5, "mkt": 0.0, "disp": 0.0,
              "resid": 0.0}


def features(d: pd.DataFrame, panel: dict, sym: str) -> pd.DataFrame:
    """The 200 columns: every method's state plus every factor.

    WHERE THE BOARD DOES NOT EXIST. BTC has bars from January 2025; the
    other nine coins start in May 2026. For those first sixteen months
    there is nothing to rank BTC against, so every cross-sectional column
    is NaN -- and dropping rows with a NaN threw away 88% of the longest
    history in the cache, which is the exact opposite of "each coin on
    the data it actually has".

    So the cross-section is filled with its NEUTRAL value there: rank in
    the middle, no market move, no dispersion, no residual. That says
    "the board is silent", which is true, instead of "this row is
    unusable", which is not.
    """
    st = M.states(d, panel, sym).add_prefix("m_")
    fx = F.build({sym: d} if len(panel) < 2 else panel)[sym]
    X = pd.concat([st.astype("float32"), fx], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        base = col.rstrip("0123456789")
        if base in XS_NEUTRAL:
            X[col] = X[col].fillna(XS_NEUTRAL[base])
    return X


def break_even(win: float = WIN, fee: float = FEE) -> float:
    """Accuracy needed for a symmetric +/-win barrier to pay the fee."""
    gain, loss = win - fee, win + fee
    return loss / (loss + gain)


def potential(p: np.ndarray, p_be: float) -> np.ndarray:
    """0 at break-even, 100 at certainty. Never negative."""
    return np.where(p > p_be, 100.0 * (p - p_be) / (1.0 - p_be), 0.0)


def leverage_for(score, lo: float = LEV_MIN, hi: float = LEV_MAX):
    """Leverage scaled by potential, floor to ceiling.

    The operator's rule: leverage is not a constant and not a separate
    decision. It rides the same number the stake does, so a setup the
    model is barely sure of takes the minimum of both and one it is
    certain of takes the maximum of both.
    """
    return lo + (hi - lo) * np.clip(np.asarray(score, dtype=float), 0, 100) / 100.0


def fit(X, y, seed: int = 0):
    """A three-way classifier plus a calibration on unseen rows."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression
    n = len(X)
    cut = int(n * 0.8)
    if cut < 500 or n - cut < 500:
        return None
    m = HistGradientBoostingClassifier(random_state=seed, **MODEL)
    m.fit(X[:cut], y[:cut])
    # Calibrate P(long | not flat) on rows the trees never saw.
    raw = directional_p(m, X[cut:])
    yy = y[cut:]
    keep = yy != 0
    if keep.sum() < 200 or len(np.unique(yy[keep])) < 2:
        return {"model": m, "iso": None, "n_calib": int(n - cut)}
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw[keep], (yy[keep] > 0).astype(float))
    return {"model": m, "iso": iso, "n_calib": int(keep.sum())}


def directional_p(model, X):
    """P(long wins) conditional on one of the two barriers being hit."""
    proba = model.predict_proba(X)
    cls = list(model.classes_)
    pl = proba[:, cls.index(1)] if 1 in cls else np.zeros(len(X))
    ps = proba[:, cls.index(-1)] if -1 in cls else np.zeros(len(X))
    tot = pl + ps
    return np.where(tot > 1e-9, pl / np.maximum(tot, 1e-9), 0.5)


def predict(fitted, X):
    """Calibrated P(long wins first), and the confidence either way."""
    p = directional_p(fitted["model"], X)
    if fitted.get("iso") is not None:
        p = fitted["iso"].predict(p)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    side = np.where(p >= 0.5, 1, -1)
    conf = np.where(p >= 0.5, p, 1.0 - p)
    return p, side, conf
