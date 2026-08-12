"""Train the multi-timeframe model and price it honestly.

One model for all ten symbols -- that is what "logic chung cho nhiều
coin" means here, and it is only possible because every feature is
scale-free (ratios, z-scores, positions in range), so BTC at $100k and
BLESS at $0.02 present the same numbers.

Three splits, and the gap between them is the whole point:

  TRAIN  June 1..20      the model sees this
  VALID  June 21..29     tuning and the go/no-go read
  TEST   Aug 5..7        never seen, a month later, and the exact bars
                         the live bot traded

Barriers are not fixed. Several (target, stop, limit) shapes are trained
side by side and, at each bar, the model picks the side and the shape
with the highest predicted net. Potential decides the trade AND its
exit, which is what a dynamic system means.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fp import june as J
from fp import mtf as M

OUT = Path(__file__).resolve().parent / "mtf_model.json"

# Barrier shapes offered to the model. Not a search over these -- all of
# them are trained, and the model chooses per bar.
SHAPES = [(2.0, 1.5, 24), (3.0, 2.0, 48), (4.0, 3.0, 96)]
ENTRY_MIN = 5

TRAIN = ("2026-05-31", "2026-06-21")
VALID = ("2026-06-21", "2026-06-30")
TEST = ("2026-08-05", "2026-08-08")


def panel(window, shape):
    tp, sl, hm = shape
    D = J.load(*window)
    Xs, yl, ys, hl, hs, syms, pos, sgs = [], [], [], [], [], [], [], []
    for sym, d1 in sorted(D.items()):
        if len(d1) < 1500:
            continue
        from fp.exits import sigma_at as _sig
        _de = J.resample(d1, ENTRY_MIN) if ENTRY_MIN > 1 else d1
        _sg = _sig(_de["close"].values.astype(float))
        X, nl, ns, kl, ks = M.build_panel(d1, ENTRY_MIN, tp, sl, hm)
        ok = np.isfinite(nl) & np.isfinite(ns) & X.notna().all(axis=1).values
        if ok.sum() < 100:
            continue
        Xs.append(X[ok])
        yl.append(nl[ok]); ys.append(ns[ok])
        hl.append(kl[ok]); hs.append(ks[ok])
        syms.append(np.full(int(ok.sum()), sym))
        pos.append(np.flatnonzero(ok))
        sgs.append(_sg[ok])
    if not Xs:
        return None
    return (pd.concat(Xs), np.concatenate(yl), np.concatenate(ys),
            np.concatenate(hl), np.concatenate(hs),
            np.concatenate(syms), np.concatenate(pos),
            np.concatenate(sgs))


def fit_one(Xtr, ytr):
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_depth=6,
        min_samples_leaf=200, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15,
        random_state=0)
    m.fit(Xtr, ytr)
    return m


def independent_mask(sym, pos, held, take):
    """Drop entries whose barrier window overlaps one already taken.

    Within each symbol, walking forward in bar order: a candidate is
    kept only once the previously kept trade on that symbol has exited.
    Two overlapping windows watch nearly the same price path, so their
    outcomes are nearly the same number -- counting both understates the
    standard error and inflates t. This is the correction that took an
    earlier rule in this repo from t = 7.81 to t = 2.26.
    """
    keep = np.zeros(len(take), dtype=bool)
    order = np.lexsort((pos, sym))
    free = {}
    for i in order:
        if not take[i]:
            continue
        s = sym[i]
        if pos[i] <= free.get(s, -1):
            continue
        keep[i] = True
        free[s] = pos[i] + int(held[i])
    return keep


def evaluate(pred_net, real_net, sym, pos, held, name,
             thresholds=(0.0, 0.001, 0.002, 0.005, 0.01)):
    """What trading the model's own prediction actually earns.

    Reported on INDEPENDENT trades only -- see independent_mask.
    """
    rows = []
    for th in thresholds:
        take = pred_net > th
        keep = independent_mask(sym, pos, held, take)
        n = int(keep.sum())
        if n < 2:
            rows.append({"gate": th, "raw": int(take.sum()), "indep": n,
                         "mean_pct": np.nan, "total_pct": 0.0,
                         "win_pct": np.nan, "t": np.nan})
            continue
        v = real_net[keep]
        se = v.std(ddof=1) / np.sqrt(n)
        rows.append({"gate": th, "raw": int(take.sum()), "indep": n,
                     "mean_pct": 100 * v.mean(), "total_pct": 100 * v.sum(),
                     "win_pct": 100 * (v > 0).mean(),
                     "t": v.mean() / se if se > 0 else 0.0})
    df = pd.DataFrame(rows)
    print(f"\n--- {name} ---")
    print(df.to_string(index=False))
    return df


def main():
    print("building panels...", flush=True)
    data = {}
    for tag, win in (("train", TRAIN), ("valid", VALID), ("test", TEST)):
        for shape in SHAPES:
            p = panel(win, shape)
            if p is None:
                print(f"  {tag} {shape}: no data")
                continue
            data[(tag, shape)] = p
            print(f"  {tag} {shape}: {len(p[0]):,} rows, {p[0].shape[1]} features",
                  flush=True)

    models = {}
    for shape in SHAPES:
        key = ("train", shape)
        if key not in data:
            continue
        Xtr, yl, ys = data[key][0], data[key][1], data[key][2]
        models[(shape, 1)] = fit_one(Xtr, yl)
        models[(shape, -1)] = fit_one(Xtr, ys)
        print(f"  fitted {shape}", flush=True)

    for tag in ("train", "valid", "test"):
        # At each bar the model picks the best (side, shape) by its own
        # predicted net -- potential chooses the trade and the exit.
        bp = br = bh = bs = bx = None
        for shape in SHAPES:
            if (tag, shape) not in data:
                continue
            X, nl, ns, kl, ks, sy, po = data[(tag, shape)][:7]
            for side, real, hh in ((1, nl, kl), (-1, ns, ks)):
                p = models[(shape, side)].predict(X)
                if bp is None:
                    bp, br, bh, bs, bx = (p.copy(), real.copy(), hh.copy(),
                                          sy.copy(), po.copy())
                else:
                    n = min(len(p), len(bp))
                    bp, br, bh, bs, bx = bp[:n], br[:n], bh[:n], bs[:n], bx[:n]
                    sw = p[:n] > bp
                    bp[sw] = p[:n][sw]
                    br[sw] = real[:n][sw]
                    bh[sw] = hh[:n][sw]
        if bp is not None:
            evaluate(bp, br, bs, bx, bh, tag.upper())
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
