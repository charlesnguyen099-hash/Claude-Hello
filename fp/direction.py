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

    Walks the horizon once, keeping the first touch. A close-only test
    would call a move that never happened a win, so both barriers are
    checked bar by bar and whichever is hit first ends the race.
    """
    n = len(close)
    out = np.zeros(n, dtype="int8")
    # The cost is charged at the barrier, so the price has to travel
    # win + fee before the trade is worth +win NET.
    up = 1.0 + win + FEE
    dn = 1.0 - (win + FEE)
    hi = pd.Series(close)[::-1].rolling(horizon, min_periods=1).max()[::-1].values
    lo = pd.Series(close)[::-1].rolling(horizon, min_periods=1).min()[::-1].values
    # Cheap pre-filter: bars where neither barrier is reachable at all.
    maybe = np.flatnonzero((hi >= close * up) | (lo <= close * dn))
    for i in maybe:
        j_end = min(i + horizon, n - 1)
        seg = close[i + 1:j_end + 1]
        if len(seg) == 0:
            continue
        u = np.flatnonzero(seg >= close[i] * up)
        d = np.flatnonzero(seg <= close[i] * dn)
        fu = u[0] if len(u) else n + 1
        fd = d[0] if len(d) else n + 1
        if fu < fd:
            out[i] = 1
        elif fd < fu:
            out[i] = -1
    return out


def features(d: pd.DataFrame, panel: dict, sym: str) -> pd.DataFrame:
    """The 200 columns: every method's state plus every factor."""
    st = M.states(d, panel, sym).add_prefix("m_")
    fx = F.build({sym: d} if len(panel) < 2 else panel)[sym]
    X = pd.concat([st.astype("float32"), fx], axis=1)
    return X.replace([np.inf, -np.inf], np.nan)


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
