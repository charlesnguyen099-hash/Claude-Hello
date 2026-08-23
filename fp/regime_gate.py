"""Does a volatility regime hide an edge the pooled gate misses?

An earlier version of this file tested exactly this question and found
0/3 coins cleared in any regime -- but that version ran BEFORE the
201-point grid bug in oof_gate was found and fixed (see fp/full.py's
oof_gate() docstring): the same coarse search that hid SKHYNIXUSDT's and
SNDKUSDT's pooled edge would just as easily have hidden a regime-only
edge for any other coin. This rebuilds the same test on the FIXED
exact-breakpoint + Wilson-lower-bound gate search (gate_from_isotonic),
so a real answer is possible this time.

THE REGIME. Trailing 60-minute realized volatility of returns, divided
by its own trailing 24h (1440-minute) rolling median -- "calm" below
that ratio's own recent normal, "volatile" at or above it. Computed
directly off OHLCV, independent of the 200-column feature cache, so
this stays cheap: no feature rebuild, just one rolling std per coin.

THE TEST. The same walk-forward fold-walk oof_gate uses (throwaway
LADDER[0] model per fold, scored strictly after the cut it trained on),
but every held-out call also carries the regime label of ITS OWN bar.
Pool (confidence, correct) pairs separately within calm and volatile,
then run the exact same gate_from_isotonic + Wilson-95%-lower-bound
search independently on each pool. A regime edge is real only if it
clears the same bar the pooled search already requires -- nothing here
is loosened for either regime.
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
VOL_WINDOW = 60
VOL_REF_WINDOW = 1440


def volatility_regime(d):
    """calm/volatile label per bar: trailing 60m vol vs its own 24h median."""
    r1 = d["close"].astype("float64").pct_change()
    vol = r1.rolling(VOL_WINDOW, min_periods=VOL_WINDOW // 2).std()
    ref = vol.rolling(VOL_REF_WINDOW, min_periods=200).median()
    ratio = (vol / ref).values
    return ratio  # >= 1 volatile, < 1 calm, NaN where undefined


def walk_folds_by_regime(X, side, regime):
    from sklearn.ensemble import HistGradientBoostingClassifier
    n = len(X)
    start = int(n * (1.0 - SPAN))
    if n - start < 4000:
        return None
    cuts = np.linspace(start, n - 1, N_FOLDS + 1)[:-1].astype(int)
    step = max((n - start) // N_FOLDS, 1000)
    confs, oks, regs = [], [], []
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
        regs.append(regime[cut:stop][took])
        del m
        gc.collect()
    if not confs or sum(len(a) for a in confs) < 300:
        return None
    return (np.concatenate(confs), np.concatenate(oks),
            np.concatenate(regs))


def main():
    P = D.load()
    fitted = sorted({f.name.split(".")[0] for f in FU.MODELS.glob("*.pkl*")})
    print("=" * 100)
    print("  VOLATILITY-REGIME GATE  (rebuilt on the fixed exact-breakpoint")
    print("  + Wilson-lower-bound search, not the retired grid search)")
    print("=" * 100, flush=True)

    for sym in fitted:
        d = P.get(sym)
        if d is None:
            continue
        cost = C.round_trip(d)
        cost = np.where(np.isfinite(cost), cost, np.nanmedian(cost))
        side, profit, mae, _ = FU.opportunities(d, cost)
        X = FU.features_for(d, P, sym)
        regime = volatility_regime(d)
        ok = np.isfinite(X).all(axis=1) & np.isfinite(regime)
        Xg = np.ascontiguousarray(X[ok])
        sideg = side[ok]
        costg = cost[ok]
        regimeg = regime[ok]
        p_be = float(C.break_even(FU.FLOOR, float(np.nanmedian(costg))))

        out = walk_folds_by_regime(Xg, sideg, regimeg)
        if out is None:
            print(f"\n--- {sym}: not enough held-out bars, skipping")
            continue
        conf, ok_arr, reg = out
        n_calm = int((reg < 1.0).sum())
        n_vol = int((reg >= 1.0).sum())
        print(f"\n--- {sym}   need {100*p_be:.2f}% to break even   "
              f"({len(conf):,} held-out calls: {n_calm:,} calm, "
              f"{n_vol:,} volatile)")

        from sklearn.isotonic import IsotonicRegression
        pooled_iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0,
                                        y_max=1.0)
        pooled_iso.fit(conf, ok_arr)
        pooled_gate = FU.gate_from_isotonic(pooled_iso, conf, ok_arr, p_be)
        print(f"    pooled (no regime split)  gate {pooled_gate:.4f}")

        for label, mask in (("calm", reg < 1.0), ("volatile", reg >= 1.0)):
            c, o = conf[mask], ok_arr[mask]
            if len(c) < 300:
                print(f"    {label:<9}                 too few calls "
                      f"({len(c):,}), skipping")
                continue
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0,
                                     y_max=1.0)
            iso.fit(c, o)
            gate = FU.gate_from_isotonic(iso, c, o, p_be)
            status = "CLEARS" if gate < 1.0 else ""
            improved = (" (found only within this regime -- pooled "
                       "search missed it)"
                       if gate < 1.0 and pooled_gate >= 1.0 else "")
            print(f"    {label:<9}  {len(c):>8,} calls   gate {gate:.4f}"
                  f"   {status}{improved}")


if __name__ == "__main__":
    sys.exit(main())
