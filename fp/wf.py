"""Walk-forward: train, trade, retrain -- and three things measured at once.

This is the concept the operator described: the model learns from what
has happened, trades forward, then learns again with the new results
included. Each fold trains on everything before a cut and is scored only
on what comes after it, so no fold ever sees its own test window.

Three numbers come out of every fold, and they only mean something
together:

  CEILING   what PERFECT selection would have earned on the same bars --
            take every trade whose realized net is positive, skip the
            rest. This is "trading past data is 100% profitable" as an
            actual number. It is real, it is large, and it is not
            reachable, because it is computed from the answer.
  MODEL     what the past-only model earns on the same bars, priced at
            full cost and counted on independent trades only.
  NULL      the same model, same features, trained on SHUFFLED labels.
            Anything the model scores above this is signal; anything it
            does not is the fitting procedure flattering itself.

The gap between CEILING and MODEL is the whole problem in one line: the
first says what is there, the second says how much of it a signal built
from earlier bars can actually reach.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fp import june as J
from fp import mtf as M
from fp import mtf_run as R

# The most heavily regularised configuration from the complexity sweep.
# Depth 6 read t = 22 on train against t = -1.75 on validation, which is
# memorisation; depth 2 was the only shape whose validation curve rose
# with the gate instead of falling.
MODEL = dict(max_depth=2, min_samples_leaf=3000, max_iter=80,
             l2_regularization=50.0, learning_rate=0.05)


def fit(X, y, seed=0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        early_stopping=True, validation_fraction=0.15,
        random_state=seed, **MODEL).fit(X, y)


def ceiling(real, sym, pos, held):
    """What perfect selection earns: take every winner, skip every loser.

    Counted on independent trades so it is comparable with the model's
    number rather than inflated relative to it.
    """
    take = real > 0
    keep = R.independent_mask(sym, pos, held, take)
    v = real[keep]
    return (len(v), 100 * v.mean() if len(v) else np.nan,
            100 * v.sum() if len(v) else 0.0)


def fold(train_win, test_win, shuffle=False, seed=0):
    """One walk-forward step. Returns (ceiling, model rows, n_train)."""
    tr, te = {}, {}
    for sh in R.SHAPES:
        p = R.panel(train_win, sh)
        q = R.panel(test_win, sh)
        if p is not None:
            tr[sh] = p
        if q is not None:
            te[sh] = q
    if not tr or not te:
        return None

    rng = np.random.default_rng(seed)
    models = {}
    ntr = 0
    for sh, (X, yl, ys, _, _, _, _, _) in tr.items():
        ntr += len(X)
        a, b = yl, ys
        if shuffle:
            # Break the link between state and outcome, keep everything
            # else -- the distribution of labels, the feature matrix, the
            # fitting procedure and its capacity to overfit.
            a, b = rng.permutation(yl), rng.permutation(ys)
        models[(sh, 1)] = fit(X, a, seed)
        models[(sh, -1)] = fit(X, b, seed)

    bp = br = bh = bs = bx = bb = None
    for sh, (X, nl, ns, kl, ks, sy, po, sg) in te.items():
        tp, sl, hm = sh
        # What a win pays on THIS trade, after the round trip. The band
        # table records net/b rather than net, because a mean net mixed
        # across barrier widths cannot be applied to one setup: +1.5% is
        # impossible for a target only 0.16% wide, and the potential
        # score correctly refuses it.
        bwin = tp * sg - J.FEE
        for side, real, hh in ((1, nl, kl), (-1, ns, ks)):
            p = models[(sh, side)].predict(X)
            if bp is None:
                bp, br, bh, bs, bx, bb = (p.copy(), real.copy(), hh.copy(),
                                          sy.copy(), po.copy(), bwin.copy())
            else:
                n = min(len(p), len(bp))
                bp, br, bh, bs, bx, bb = (bp[:n], br[:n], bh[:n], bs[:n],
                                          bx[:n], bb[:n])
                sw = p[:n] > bp
                bp[sw] = p[:n][sw]
                br[sw] = real[:n][sw]
                bh[sw] = hh[:n][sw]
                bb[sw] = bwin[:n][sw]
    return bp, br, bs, bx, bh, ntr, bb
