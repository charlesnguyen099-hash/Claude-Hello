"""Turn the answer sheet into logic, and price the logic honestly.

The chain, end to end:

  fp/factors.py  153 numbers describing the bar -- price, flow, state and
                 what the other nine coins are doing at that instant.
  fp/labels.py   every triple-barrier trade that profits after the whole
                 exchange bill, with the target scaled by sqrt(horizon)
                 so the break-even win rate is near a coin flip instead
                 of 94%.
  HERE           one classifier per (shape, side) predicting P(win) from
                 the factors, ONE model across all ten coins, then the
                 potential score that turns that probability into a
                 stake.

THE POTENTIAL SCORE, and why it is a probability question.

A shape paying +b against -a needs

    p_be = a / (a + b)

just to stand still. The model supplies p_hat. The score is where the
estimate sits between break-even and certainty:

    POTENTIAL = 100 * (p_hat - p_be) / (1 - p_be)

0 means break-even, 100 means it cannot lose. Same scale for every shape,
every coin and every horizon, so a 60-minute scalp and a 12-hour swing
are directly comparable and the stake reads straight off it. That is the
operator's "logic nao tiem nang cao thi trade nhieu von".

p_hat is NOT the raw classifier output. A gradient booster's probability
is a ranking, not a frequency, so it is isotonically calibrated on a
slice of the training window that the trees never fit -- and the score
uses the LOWER end of the calibration's own confidence interval, because
sizing off the point estimate is how a backtest becomes a margin call.

HOW IT IS JUDGED. Walk-forward: train on the past, trade the future,
never the reverse. Every number reported is on INDEPENDENT trades --
overlapping barrier windows watch the same price path and counting both
inflates t, which has bitten this repo three times. And every result is
put against a rotation null: roll the prediction series, which keeps its
drift, its volatility and its long/short balance and destroys only its
alignment. A logic that cannot beat a rolled copy of itself has found
nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fp import factors as F
from fp import data as DATA
from fp import labels as LB

# Train on every Nth bar. Consecutive one-minute entries overlap almost
# completely, so the 5th bar carries nearly all the information of the
# five and the fit runs five times faster. Evaluation uses every bar and
# then applies the independence mask, so nothing is hidden by this.
# Every 8th bar. At a 60-720 minute horizon, eight consecutive
# one-minute entries are the same trade eight times over, so the 8th bar
# carries essentially all of the information and the fit runs eight times
# faster. Two full walk-forwards were killed for memory before finishing,
# which is worse than a slightly coarser sample.
TRAIN_STRIDE = 8
# Evaluate every 3rd bar. The independence mask already collapses
# overlapping entries -- with holds of 60 to 720 minutes, consecutive
# one-minute entries are the same trade seen three times -- so this costs
# almost no information and cuts peak memory threefold. The first version
# held 32 combinations x 200k rows of PYTHON STRINGS for the symbol
# column and was killed by the OOM reaper halfway through fold 2.
EVAL_STRIDE = 3

# Held out from the END of each training window, never fitted, used only
# to turn the classifier's ranking into a frequency.
CALIB_FRAC = 0.20

MODEL = dict(max_depth=3, min_samples_leaf=500, max_iter=120,
             l2_regularization=5.0, learning_rate=0.05,
             early_stopping=True, validation_fraction=0.12)


def load_panel(window):
    """Factors, labels and bookkeeping for every symbol in a window."""
    D = DATA.load(*window)
    D = {s: d for s, d in D.items() if len(d) > 2000}
    X = F.build(D)
    out = {}
    for s, d in D.items():
        L, sg = LB.label_all(d)
        out[s] = {"X": X[s], "labels": L, "sigma": sg, "index": d.index}
    return out


def stack(panel, key, stride=1):
    """One (shape, side) across every coin, as flat arrays.

    This is where "logic chung cua nhieu coin" becomes literal: the rows
    from all ten symbols go into one matrix and one model is fitted to
    all of them. The factors are scale-free, so BTC and BLESS contribute
    comparable rows.
    """
    Xs, y, net, held, sym, pos = [], [], [], [], [], []
    order = sorted(panel)
    for s, P in sorted(panel.items()):
        L = P["labels"][key]
        ok = L["tradeable"] & np.isfinite(P["X"].values).all(axis=1)
        idx = np.flatnonzero(ok)
        if stride > 1:
            idx = idx[::stride]
        if len(idx) < 50:
            continue
        Xs.append(P["X"].values[idx])
        y.append(L["win"][idx])
        net.append(L["net"][idx].astype("float32"))
        held.append(L["held"][idx].astype("int32"))
        # Symbols as small integer CODES, not strings. A string column
        # over a few million rows is a few million Python objects, and
        # that alone was most of the memory that got an earlier run
        # killed. The order is the sorted panel order, so the code is
        # stable within a fold, which is all the independence mask needs.
        sym.append(np.full(len(idx), order.index(s), dtype="int16"))
        pos.append(idx.astype("int32"))
    if not Xs:
        return None
    return (np.vstack(Xs), np.concatenate(y), np.concatenate(net),
            np.concatenate(held), np.concatenate(sym), np.concatenate(pos))


def fit_one(X, y, seed=0):
    """Classifier plus an isotonic calibration fitted on unseen rows."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression
    n = len(X)
    cut = int(n * (1 - CALIB_FRAC))
    if cut < 200 or n - cut < 200 or len(np.unique(y[:cut])) < 2:
        return None
    m = HistGradientBoostingClassifier(random_state=seed, **MODEL)
    m.fit(X[:cut], y[:cut])
    raw = m.predict_proba(X[cut:])[:, 1]
    if len(np.unique(y[cut:])) < 2:
        return None
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw, y[cut:].astype(float))
    # How many calibration rows sit behind each probability, so the score
    # can use a LOWER bound rather than the point estimate.
    return {"model": m, "iso": iso, "n_calib": int(n - cut)}


def p_hat(fit, X, lower=True):
    """Calibrated win probability, optionally lower-bounded.

    The lower bound is a Wilson-style haircut using the calibration
    sample size: a probability estimated from 800 rows is worth less than
    the same number from 80,000, and the stake should know that.
    """
    p = fit["iso"].predict(fit["model"].predict_proba(X)[:, 1])
    p = np.clip(p, 1e-6, 1 - 1e-6)
    if not lower:
        return p
    n = max(fit["n_calib"], 1)
    se = np.sqrt(p * (1 - p) / n)
    return np.clip(p - 1.96 * se, 1e-6, 1 - 1e-6)


def break_even(tp, sl, hold, sg, fee=LB.FEE):
    """p_be, and the payouts behind it, at each row's own volatility."""
    sgh = sg * np.sqrt(hold)
    b = tp * sgh - fee
    a = sl * sgh + fee
    return a / (a + b), a, b


def potential(p, p_be):
    """0 = break-even, 100 = cannot lose. Negative is clipped to 0."""
    return np.where(p > p_be, 100.0 * (p - p_be) / (1.0 - p_be), 0.0)


def independent(sym, pos, held, take):
    """Keep only entries whose barrier window is clear of the last one."""
    keep = np.zeros(len(take), dtype=bool)
    order = np.lexsort((pos, sym))
    free: dict = {}
    for i in order:
        if not take[i]:
            continue
        s = sym[i]
        if pos[i] <= free.get(s, -1):
            continue
        keep[i] = True
        free[s] = pos[i] + int(held[i])
    return keep


def stats(v):
    n = len(v)
    if n < 2:
        return dict(n=n, mean=np.nan, t=np.nan, win=np.nan, tot=0.0)
    se = v.std(ddof=1) / np.sqrt(n)
    return dict(n=n, mean=float(v.mean()),
                t=float(v.mean() / se) if se > 0 else 0.0,
                win=float((v > 0).mean()), tot=float(v.sum()))


def rotation_null(pred, net, sym, pos, held, gate, rounds=200, seed=0):
    """Same predictions, same market, alignment destroyed."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(rounds):
        p = np.roll(pred, int(rng.integers(1, max(len(pred) - 1, 2))))
        k = independent(sym, pos, held, gate(p))
        out.append(net[k].mean() if k.sum() >= 2 else 0.0)
    return np.array(out)
