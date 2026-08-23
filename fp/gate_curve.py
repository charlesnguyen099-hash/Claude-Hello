"""The full confidence -> accuracy curve, not just the single gate point.

fp/full.py's oof_gate() reports ONE number: the lowest confidence that
clears break-even, or 1.0 if none does. That number hides the shape of
the curve behind it -- exactly what the operator is asking to see:
if the gate came down, how many more trades would that admit, and
what would their actual measured accuracy be? This is the same
walk-forward, pooled, isotonic-calibrated measurement oof_gate() uses,
printed at every decile of confidence instead of collapsed to one
threshold.
"""
from __future__ import annotations

import gc
import sys

import numpy as np

from fp import costs as C
from fp import data as D
from fp import full as FU

N_FOLDS = 4
SPAN = 0.5


def walk_folds(X, side):
    from sklearn.ensemble import HistGradientBoostingClassifier
    n = len(X)
    start = int(n * (1.0 - SPAN))
    if n - start < 4000:
        return None
    cuts = np.linspace(start, n - 1, N_FOLDS + 1)[:-1].astype(int)
    step = max((n - start) // N_FOLDS, 1000)
    confs, oks = [], []
    for cut in cuts:
        stop = min(cut + step, n)
        if stop - cut < 500 or cut < 2000:
            continue
        m = HistGradientBoostingClassifier(random_state=0, **FU.BASE,
                                           **FU.LADDER[0])
        m.fit(X[:cut], side[:cut])
        proba = m.predict_proba(X[cut:stop])
        cls = list(m.classes_)
        w = stop - cut
        pl = proba[:, cls.index(1)] if 1 in cls else np.zeros(w)
        ps = proba[:, cls.index(-1)] if -1 in cls else np.zeros(w)
        p0 = proba[:, cls.index(0)] if 0 in cls else np.zeros(w)
        pred = np.where(pl >= ps, 1, -1).astype("int8")
        conf = np.maximum(pl, ps)
        pred = np.where(conf > p0, pred, 0).astype("int8")
        truth = side[cut:stop]
        took = pred != 0
        confs.append(conf[took])
        oks.append((pred[took] == truth[took]).astype(float))
        del m
        gc.collect()
    if not confs or sum(len(a) for a in confs) < 300:
        return None
    return np.concatenate(confs), np.concatenate(oks)


def wilson_lower(k, n, z=1.96):
    """95% lower confidence bound on a binomial proportion (Wilson
    score interval) -- not just the point estimate, whether the TRUE
    accuracy could plausibly still clear break-even given how few
    held-out calls back the number up."""
    if n == 0:
        return float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    adj = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - adj) / denom


def main():
    P = D.load()
    fitted = sorted({f.name.split(".")[0] for f in FU.MODELS.glob("*.pkl*")})
    print("=" * 100)
    print("  THE FULL CONFIDENCE -> ACCURACY CURVE (not just the gate crossing)")
    print("  At each threshold: calls admitted, point-estimate accuracy,")
    print("  and the 95%-confidence LOWER BOUND on that accuracy (Wilson).")
    print("=" * 100, flush=True)

    for sym in fitted:
        d = P.get(sym)
        if d is None:
            continue
        cost = C.round_trip(d)
        cost = np.where(np.isfinite(cost), cost, np.nanmedian(cost))
        side, profit, mae, _ = FU.opportunities(d, cost)
        X = FU.features_for(d, P, sym)
        ok = np.isfinite(X).all(axis=1)
        Xg = np.ascontiguousarray(X[ok])
        sideg = side[ok]
        costg = cost[ok]
        p_be = float(C.break_even(FU.FLOOR, float(np.nanmedian(costg))))

        out = walk_folds(Xg, sideg)
        if out is None:
            print(f"\n--- {sym}: not enough held-out bars, skipping")
            continue
        conf, ok_arr = out
        print(f"\n--- {sym}   need {100*p_be:.2f}% to break even   "
              f"({len(conf):,} held-out calls total)")
        print(f"    {'threshold':>10}{'n calls':>10}{'accuracy':>10}"
              f"{'95% low':>10}   status")
        for q in (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99):
            thresh = float(np.percentile(conf, q))
            mask = conf >= thresh
            n = int(mask.sum())
            if n < 30:
                continue
            k = int(ok_arr[mask].sum())
            acc = k / n
            low = wilson_lower(k, n)
            status = ("CLEARS (point est.)" if acc >= p_be else "")
            status2 = ("CLEARS EVEN AT 95% LOW BOUND"
                      if low >= p_be else status)
            print(f"    {thresh:>10.4f}{n:>10,}{100*acc:>9.2f}%"
                  f"{100*low:>9.2f}%   {status2}")


if __name__ == "__main__":
    sys.exit(main())
