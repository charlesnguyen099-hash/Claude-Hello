"""Can the logic reproduce the past it was shown? It must, or it is broken.

The operator's check, and it is the right one to demand: before believing
anything about the future, prove the pipeline can learn the data it was
handed. If a high-capacity model, fitted on rows and scored on THOSE SAME
rows, cannot approach 100% accuracy, something is wrong in the plumbing --
features misaligned with labels, a shift in the wrong direction, rows
dropped unevenly - and every out-of-sample number that follows is
measuring the bug, not the market.

So this runs the deliberately unfair test:

  FIT      a deep, unregularised model on a block of bars
  SCORE    the very same bars
  EXPECT   near 100%

Passing it proves the pipeline is sound. It proves NOTHING about the
future -- a model with enough capacity can memorise noise, which is
exactly what it is doing here. The value is diagnostic: it separates "the
market is hard" from "the code is wrong", and those need different
responses.

Reported next to it, on the same rows, is the same model at the
regularisation the live pipeline actually uses, plus the out-of-sample
number. The three together say where the loss lives:

  memorise (in-sample, deep)     ~100%  -> the plumbing is sound
  shipped model (in-sample)        ...  -> how much it chooses to fit
  shipped model (out-of-sample)    ...  -> what survives contact
"""
from __future__ import annotations

import sys

import numpy as np

from fp import costs as C
from fp import data as D
from fp import direction as DIR

# Deliberately over-powered: no depth limit worth the name, tiny leaves,
# no early stopping. This is a memorisation test, not a model.
MEMORISE = dict(max_depth=None, min_samples_leaf=1, max_iter=400,
                l2_regularization=0.0, learning_rate=0.3,
                early_stopping=False)


def fit_raw(X, y, params):
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(random_state=0, **params)
    m.fit(X, y)
    return m


def accuracy(m, X, y):
    """Directional accuracy on rows where a barrier was actually hit."""
    p = DIR.directional_p(m, X)
    side = np.where(p >= 0.5, 1, -1)
    res = y != 0
    return float((side[res] == y[res]).mean()) if res.any() else np.nan


def main():
    syms = sys.argv[1:] or ["SOXLUSDT", "BTCUSDT"]
    P = D.load()
    print("=" * 78)
    print("  IN-SAMPLE FIT: can the pipeline learn the data it was given?")
    print("  A deep model scored on its OWN training rows must approach")
    print("  100%. If it cannot, the plumbing is broken, not the market.")
    print("=" * 78, flush=True)

    for sym in syms:
        d = P.get(sym)
        if d is None:
            continue
        cost = float(np.nanmedian(C.round_trip(d)))
        X = DIR.features(d, P, sym).values.astype("float32")
        y = DIR.label(d["close"].values.astype("float64"))
        ok = np.isfinite(X).all(axis=1)
        idx = np.flatnonzero(ok)
        # One block, split in half: fit on the first, score both.
        half = len(idx) // 2
        tr = idx[:half][::max(1, half // 40000)]
        te = idx[half:][::max(1, (len(idx) - half) // 40000)]
        res_tr = int((y[tr] != 0).sum())
        print(f"\n--- {sym}  train {len(tr):,} rows ({res_tr:,} resolved)  "
              f"test {len(te):,}   cost {100*cost:.4f}%", flush=True)

        deep = fit_raw(X[tr], y[tr], MEMORISE)
        a_mem_in = accuracy(deep, X[tr], y[tr])
        a_mem_out = accuracy(deep, X[te], y[te])
        del deep

        ship = fit_raw(X[tr], y[tr], DIR.MODEL)
        a_shp_in = accuracy(ship, X[tr], y[tr])
        a_shp_out = accuracy(ship, X[te], y[te])
        del ship

        p_be = float(C.break_even(DIR.WIN, cost))
        print(f"    memorising model, ON ITS OWN ROWS : {100*a_mem_in:6.2f}%"
              f"   <- must approach 100")
        print(f"    memorising model, unseen rows     : {100*a_mem_out:6.2f}%")
        print(f"    shipped model,    ON ITS OWN ROWS : {100*a_shp_in:6.2f}%")
        print(f"    shipped model,    unseen rows     : {100*a_shp_out:6.2f}%")
        print(f"    break-even needed                 : {100*p_be:6.2f}%")
        if a_mem_in > 0.95:
            print(f"    PLUMBING OK: the features and labels line up, and a")
            print(f"    model with enough capacity reproduces the past.")
        else:
            print(f"    PLUMBING SUSPECT: a model that cannot memorise its")
            print(f"    own rows has a wiring problem, not a market problem.")
        gap = a_mem_in - a_mem_out
        print(f"    memorised {100*gap:.1f} points that did NOT transfer -- "
              f"that gap is noise, not signal.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
