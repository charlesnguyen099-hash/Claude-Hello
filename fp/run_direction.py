"""Walk the direction model forward and price it, leverage included.

Train on the past, trade the future, per coin, on whatever history that
coin has. Every number is out of sample.

  ACCURACY   how often the model calls the winning side, on trades it
             actually takes. The bar is 55.5% -- a +1% barrier pays
             +0.89% and costs -1.11% after a 0.110% round trip.
  NET        what the account does, with the stake AND the leverage both
             scaled by potential from floor to ceiling.
  NULL       the same model against a rotation of its own predictions.

The gate is potential > 0, which is the same thing as p > p_be: the model
must believe the setup clears the fee before a cent is committed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from fp import data as D
from fp import direction as DIR

HERE = Path(__file__).resolve().parent
OUT = HERE / "direction_model.json"

TRAIN_FRAC = 0.55
FOLD_DAYS = 70
MIN_TEST_DAYS = 7
STRIDE = 5            # every 5th bar; neighbours are near-duplicates

MIN_STAKE, MAX_STAKE = 0.05, 1.00


def folds_for(index):
    start, end = index[0], index[-1]
    total = (end - start).days
    if total < 3 * MIN_TEST_DAYS:
        return []
    train_days = max(int(total * TRAIN_FRAC), MIN_TEST_DAYS)
    block = min(FOLD_DAYS, max((total - train_days) // 2, MIN_TEST_DAYS))
    out, cut = [], start + np.timedelta64(train_days, "D")
    while cut < end - np.timedelta64(MIN_TEST_DAYS - 1, "D"):
        stop = min(cut + np.timedelta64(block, "D"), end)
        out.append((start, cut, stop))
        cut = stop
    return out


def main():
    p_be = DIR.break_even()
    print("=" * 84)
    print("  DIRECTION FROM THE TRADES THAT PAID")
    print(f"  label : which side reaches +{100*DIR.WIN:.0f}% net first, "
          f"within {DIR.HORIZON} minutes")
    print(f"  needs : {100*p_be:.1f}% accuracy to clear a "
          f"{100*DIR.FEE:.3f}% round trip")
    print(f"  stake : {100*MIN_STAKE:.0f}%..{100*MAX_STAKE:.0f}% of equity, "
          f"leverage {DIR.LEV_MIN:.0f}x..{DIR.LEV_MAX:.0f}x -- BOTH scaled "
          f"by potential")
    print("=" * 84, flush=True)

    P = D.load()
    rows, keepers = [], {}
    for sym in sorted(P):
        d = P[sym]
        idx = d.index
        fl = folds_for(idx)
        if not fl:
            continue
        print(f"\n--- {sym}   {len(d):,} bars   {len(fl)} folds", flush=True)
        X = DIR.features(d, P, sym)
        y = DIR.label(d["close"].values.astype("float64"))
        ok = np.isfinite(X.values).all(axis=1)
        pos = {t: i for i, t in enumerate(idx)}
        fold_net = []
        for fi, (a, cut, stop) in enumerate(fl, 1):
            ia = pos[idx[idx >= a][0]]
            ic = pos[idx[idx >= cut][0]]
            ib = pos[idx[idx < stop][-1]]
            tr = np.flatnonzero(ok[ia:ic]) + ia
            te = np.flatnonzero(ok[ic:ib]) + ic
            tr = tr[::STRIDE]
            if len(tr) < 2000 or len(te) < 200:
                print(f"    fold {fi}: too few usable rows")
                continue
            fit = DIR.fit(X.values[tr], y[tr])
            if fit is None:
                print(f"    fold {fi}: could not fit")
                continue
            pr, side, conf = DIR.predict(fit, X.values[te])
            score = DIR.potential(conf, p_be)
            take = score > 0
            yt = y[te]
            # Accuracy only counts bars where a barrier was actually hit;
            # a bar that never resolved is not a wrong call, it is no call.
            res = take & (yt != 0)
            acc = float((side[res] == yt[res]).mean()) if res.any() else np.nan
            # The account: stake and leverage BOTH ride the potential.
            stake = MIN_STAKE + (MAX_STAKE - MIN_STAKE) * score[res] / 100.0
            lev = DIR.leverage_for(score[res])
            won = side[res] == yt[res]
            per = np.where(won, DIR.WIN - DIR.FEE, -(DIR.WIN + DIR.FEE))
            net = float((stake * lev * per).sum())
            fold_net.append(net)
            rng = np.random.default_rng(fi)
            nulls = []
            for _ in range(60):
                sh = np.roll(side, int(rng.integers(1, max(len(side) - 1, 2))))
                w2 = sh[res] == yt[res]
                p2 = np.where(w2, DIR.WIN - DIR.FEE, -(DIR.WIN + DIR.FEE))
                nulls.append(float((stake * lev * p2).sum()))
            pv = float((np.array(nulls) >= net).mean())
            print(f"    fold {fi}: took {int(res.sum()):>6} resolved calls  "
                  f"accuracy {100*acc:>5.2f}% (needs {100*p_be:.1f}%)  "
                  f"net {100*net:>+9.1f}%  p(null) {pv:.3f}", flush=True)
            rows.append(dict(sym=sym, fold=fi, n=int(res.sum()), acc=acc,
                             net=net, p=pv))
        if fold_net and all(v > 0 for v in fold_net):
            keepers[sym] = [float(v) for v in fold_net]

    print("\n" + "=" * 84)
    if rows:
        accs = np.array([r["acc"] for r in rows if np.isfinite(r["acc"])])
        w = np.array([r["n"] for r in rows if np.isfinite(r["acc"])])
        wa = float((accs * w).sum() / w.sum())
        print(f"  WEIGHTED ACCURACY : {100*wa:.2f}%   "
              f"(break-even {100*p_be:.1f}%)")
        print(f"  EDGE              : {100*(wa - p_be):+.2f} points")
        print(f"  TOTAL NET         : {100*sum(r['net'] for r in rows):+.1f}%")
        print(f"  FOLDS POSITIVE    : "
              f"{sum(1 for r in rows if r['net'] > 0)}/{len(rows)}")
        print(f"  BEAT THEIR NULL   : "
              f"{sum(1 for r in rows if r['p'] < 0.05)}/{len(rows)}")
    print(f"  COINS SHIPPING    : {len(keepers)} "
          f"({', '.join(keepers) or 'none'})")
    OUT.write_text(json.dumps(
        {"break_even": p_be, "win": DIR.WIN, "horizon": DIR.HORIZON,
         "lev_min": DIR.LEV_MIN, "lev_max": DIR.LEV_MAX,
         "min_stake": MIN_STAKE, "max_stake": MAX_STAKE,
         "folds": rows, "ships": keepers}, indent=1))
    print(f"  wrote {OUT.name}")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    sys.exit(main())
