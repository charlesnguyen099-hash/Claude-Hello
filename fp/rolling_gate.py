"""Walk forward in 15-day windows across the WHOLE history, not 4 folds
over the back half -- does the edge hold steady window by window, or
is a coin's pooled pass/fail hiding a trend?

oof_gate's 4-fold walk already trains-then-tests strictly forward in
time, but it only covers the back half of history in 4 wide folds and
POOLS them before judging pass/fail. A coin whose edge is fading (real
in early folds, gone in late ones) or just appearing (the reverse) reads
identically to a coin with steady edge once everything is averaged
together. This walks forward in much narrower, chained windows across
ALL available history, expanding the training set each step (never
re-using a bar for both train and test), and reports each window
separately -- so a trend is visible instead of averaged away.

Two numbers per window:
  all signals   every predicted call in that window, whatever its
                confidence -- the model's raw discriminating power,
                unfiltered, tracked over time.
  at live_gate  only the calls that would have cleared the coin's
                ACTUAL shipped live_gate -- does the threshold already
                chosen still hold up 15 days at a time, or does it
                decay/improve as time moves forward?
"""
from __future__ import annotations

import gc
import sys

import numpy as np

from fp import costs as C
from fp import data as D
from fp import full as FU

WINDOW_DAYS = 15
WINDOW_BARS = WINDOW_DAYS * 1440
MIN_TRAIN_BARS = 30 * 1440   # 30 days minimum before the first test window


def walk_windows(X, side, window_bars=WINDOW_BARS,
                 min_train=MIN_TRAIN_BARS):
    """(cut, stop, conf, ok) per chained, non-overlapping window."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    n = len(X)
    out = []
    cut = min_train
    while cut < n - 500:
        stop = min(cut + window_bars, n)
        if stop - cut < 500:
            break
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
        out.append((cut, stop, conf[took],
                    (pred[took] == truth[took]).astype(float)))
        del m
        gc.collect()
        cut = stop
    return out


def wilson_lower(k, n, z=1.96):
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
    print(f"  ROLLING {WINDOW_DAYS}-DAY WALK-FORWARD  (whole history, "
          f"expanding train set, never averaged away)")
    print("=" * 100, flush=True)

    for sym in fitted:
        d = P.get(sym)
        if d is None:
            continue
        blob = FU.load_models(sym)
        live_gate = float((blob or {}).get("meta", {}).get("live_gate", 1.0))
        cost = C.round_trip(d)
        cost = np.where(np.isfinite(cost), cost, np.nanmedian(cost))
        side, profit, mae, _ = FU.opportunities(d, cost)
        X = FU.features_for(d, P, sym)
        ok = np.isfinite(X).all(axis=1)
        Xg = np.ascontiguousarray(X[ok])
        sideg = side[ok]
        costg = cost[ok]
        idxg = d.index[ok]
        p_be = float(C.break_even(FU.FLOOR, float(np.nanmedian(costg))))

        windows = walk_windows(Xg, sideg)
        if not windows:
            print(f"\n--- {sym}: not enough history for even one window")
            continue
        print(f"\n--- {sym}   need {100*p_be:.2f}% to break even   "
              f"shipped live_gate {live_gate:.4f}   "
              f"({len(windows)} windows of {WINDOW_DAYS}d)")
        print(f"    {'window start':<12}{'window end':<12}{'n (all)':>9}"
              f"{'acc (all)':>11}{'n (>=gate)':>12}{'acc (>=gate)':>13}"
              f"{'95% low':>10}   status")
        for cut, stop, conf, ok_arr in windows:
            d0 = str(idxg[min(cut, len(idxg) - 1)].date())
            d1 = str(idxg[min(stop - 1, len(idxg) - 1)].date())
            n_all = len(conf)
            acc_all = float(ok_arr.mean()) if n_all else float("nan")
            mask_g = conf >= live_gate
            n_g = int(mask_g.sum())
            if n_g > 0:
                k_g = int(ok_arr[mask_g].sum())
                acc_g = k_g / n_g
                low_g = wilson_lower(k_g, n_g)
                status = ("HOLDS" if low_g >= p_be else
                          ("below" if acc_g < p_be else ""))
                g_str = f"{n_g:>12,}{100*acc_g:>12.2f}%{100*low_g:>9.2f}%   {status}"
            else:
                g_str = f"{0:>12,}{'--':>13}{'--':>10}   "
            print(f"    {d0:<12}{d1:<12}{n_all:>9,}{100*acc_all:>10.2f}%"
                  f"{g_str}")


if __name__ == "__main__":
    sys.exit(main())
