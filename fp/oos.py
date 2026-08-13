"""Score the SHIPPED model on bars it has never seen.

The model in mtf_model.pkl was trained on everything up to 2026-08-07
03:00 -- the end of the cache at the time. The upload of 2026-08-06..13
extends the cache six days past that line. Those six days are the first
genuinely out-of-sample test this model has ever had: not a walk-forward
fold carved out of the same month, not a held-out slice chosen after the
fact, but bars that did not exist when the model was fitted.

What is measured here:

  1. the band gate as shipped -- does the one band the walk-forward
     called profitable still pay?
  2. the full marginal decomposition -- did the SHAPE of the
     prediction/outcome relationship survive, or was it noise?
  3. a rotation null -- roll the prediction series by a random offset,
     which preserves its drift, its volatility and its long/short
     balance and destroys only its alignment with the market. If the
     real result does not clear its own null, there is nothing there.

Everything is reported on INDEPENDENT trades. Overlapping barrier
windows watch the same price path and counting both inflates t; this
repo has been burned by that three times.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fp import june as J
from fp import mtf as M
from fp import mtf_run as R

HERE = Path(__file__).resolve().parent

# Warm-up has to cover the deepest feature: the 60-minute view's
# slope55 is rolling(55).mean().diff(55) = 110 hours. Load from well
# before the cut so every OOS row has finite features, then keep only
# the rows on the far side of it.
WARM_FROM = "2026-07-20"
CUT = "2026-08-07T03:00:00Z"
END = "2026-08-14"


def oos_panel(shape):
    """(X, net_long, net_short, held_long, held_short, sym, pos, ts) for
    entry bars strictly after the training cut."""
    tp, sl, hm = shape
    D = J.load(WARM_FROM, END)
    Xs, yl, ys, hl, hs, syms, pos, tss = [], [], [], [], [], [], [], []
    cut = pd.Timestamp(CUT)
    for sym, d1 in sorted(D.items()):
        if len(d1) < 8000:
            continue
        de = J.resample(d1, R.ENTRY_MIN)
        X, nl, ns, kl, ks = M.build_panel(d1, R.ENTRY_MIN, tp, sl, hm)
        ok = (np.isfinite(nl) & np.isfinite(ns)
              & X.notna().all(axis=1).values
              & (de.index[:len(X)] >= cut))
        if ok.sum() < 50:
            continue
        Xs.append(X[ok])
        yl.append(nl[ok]); ys.append(ns[ok])
        hl.append(kl[ok]); hs.append(ks[ok])
        syms.append(np.full(int(ok.sum()), sym))
        pos.append(np.flatnonzero(ok))
        tss.append(de.index[:len(X)][ok])
    if not Xs:
        return None
    return (pd.concat(Xs), np.concatenate(yl), np.concatenate(ys),
            np.concatenate(hl), np.concatenate(hs),
            np.concatenate(syms), np.concatenate(pos),
            np.concatenate([np.asarray(t) for t in tss]))


def stats(v):
    n = len(v)
    if n < 2:
        return dict(n=n, mean=np.nan, t=np.nan, win=np.nan, tot=0.0)
    se = v.std(ddof=1) / np.sqrt(n)
    return dict(n=n, mean=v.mean(), t=(v.mean() / se if se > 0 else 0.0),
                win=(v > 0).mean(), tot=v.sum())


def main():
    models = pickle.loads((HERE / "mtf_model.pkl").read_bytes())
    meta = json.loads((HERE / "mtf_model.json").read_text())
    bands = json.loads((HERE / "mtf_bands.json").read_text())
    cols = list(meta["columns"])

    print(f"model trained on {meta['train_window'][0]}..{meta['train_window'][1]}"
          f" ({meta['train_rows']:,} rows)")
    print(f"scoring entry bars from {CUT} to {END}\n", flush=True)

    # Every (shape, side) scored on every OOS bar, kept flat so the band
    # decomposition and the gate both read the same rows.
    P, Rn, Sy, Po, Hd, Sg, Bv = [], [], [], [], [], [], []
    for shape in R.SHAPES:
        p = oos_panel(shape)
        if p is None:
            print(f"  {shape}: no OOS rows")
            continue
        X, nl, ns, kl, ks, sy, po, ts = p
        if list(X.columns) != cols:
            raise SystemExit("column mismatch -- retrain before scoring")
        tp, sl, hm = shape
        for side, tag, real, held in ((1, "long", nl, kl),
                                      (-1, "short", ns, ks)):
            m = models.get(f"{tp}_{sl}_{hm}_{tag}")
            if m is None:
                continue
            pr = m.predict(X)
            P.append(pr); Rn.append(real); Sy.append(sy); Po.append(po)
            Hd.append(held)
            Sg.append(np.full(len(pr), f"{tp}_{sl}_{hm}_{tag}"))
            # b = what a win pays, net of fee, at this row's own sigma.
            # Recovered from the barrier definition so net/b is
            # comparable across shapes.
            Bv.append(np.full(len(pr), np.nan))
        print(f"  {shape}: {len(X):,} OOS entry bars", flush=True)

    if not P:
        raise SystemExit("no OOS rows at all")

    P = np.concatenate(P); Rn = np.concatenate(Rn)
    Sy = np.concatenate(Sy); Po = np.concatenate(Po)
    Hd = np.concatenate(Hd); Sg = np.concatenate(Sg)

    print(f"\ntotal scored rows: {len(P):,}")

    # ---- 1. the gate as shipped -------------------------------------
    paying = [b for b in bands if b["edge_over_b"] > 0 and b["t"] >= 2.0]
    take = np.zeros(len(P), dtype=bool)
    for b in paying:
        take |= (P >= b["lo"]) & (P < b["hi"])
    keep = R.independent_mask(Sy, Po, Hd, take)
    s = stats(Rn[keep])
    print("\n=== THE SHIPPED GATE, OUT OF SAMPLE " + "=" * 30)
    for b in paying:
        print(f"  band {b['lo']:.3f}..{b['hi']:.3f}  "
              f"(in-sample {b['edge_over_b']:+.3f} of a win, t={b['t']:+.2f})")
    print(f"  rows in band      : {int(take.sum()):,}")
    print(f"  independent trades: {s['n']}")
    if s["n"] >= 2:
        print(f"  mean net/trade    : {100*s['mean']:+.4f}%")
        print(f"  total             : {100*s['tot']:+.3f}%")
        print(f"  win rate          : {100*s['win']:.1f}%")
        print(f"  t                 : {s['t']:+.2f}")

    # ---- 2. the full marginal decomposition -------------------------
    print("\n=== EVERY BAND, OUT OF SAMPLE " + "=" * 36)
    edges = [-9e9] + [b["hi"] for b in bands[:-1]] + [9e9]
    print(f"  {'band':<16} {'indep':>6} {'mean%':>9} {'t':>7} {'win%':>6}")
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        t_ = (P >= lo) & (P < hi)
        k_ = R.independent_mask(Sy, Po, Hd, t_)
        st = stats(Rn[k_])
        lab = f"{'-inf' if lo < -1 else f'{lo:.3f}'}..{'+inf' if hi > 1 else f'{hi:.3f}'}"
        if st["n"] < 2:
            print(f"  {lab:<16} {st['n']:>6}        --      --     --")
        else:
            print(f"  {lab:<16} {st['n']:>6} {100*st['mean']:>+9.4f} "
                  f"{st['t']:>+7.2f} {100*st['win']:>6.1f}")

    # ---- 3. rotation null -------------------------------------------
    # Roll the PREDICTION within each (shape,side) block by a random
    # offset. Same predictions, same market, alignment destroyed.
    print("\n=== ROTATION NULL " + "=" * 48)
    rng = np.random.default_rng(0)
    nulls = []
    for _ in range(200):
        Pn = P.copy()
        for g in np.unique(Sg):
            idx = np.flatnonzero(Sg == g)
            Pn[idx] = np.roll(P[idx], int(rng.integers(1, len(idx))))
        tk = np.zeros(len(Pn), dtype=bool)
        for b in paying:
            tk |= (Pn >= b["lo"]) & (Pn < b["hi"])
        kp = R.independent_mask(Sy, Po, Hd, tk)
        nulls.append(Rn[kp].mean() if kp.sum() >= 2 else 0.0)
    nulls = np.array(nulls)
    real = s["mean"] if s["n"] >= 2 else 0.0
    pval = float((nulls >= real).mean())
    print(f"  real mean net/trade : {100*real:+.4f}%")
    print(f"  null mean (200 rolls): {100*nulls.mean():+.4f}%  "
          f"sd {100*nulls.std():.4f}%")
    print(f"  p-value (null >= real): {pval:.3f}")
    print(f"  VERDICT: " + ("the gate beats its own null"
                            if pval < 0.05 else
                            "the gate does NOT beat its own null"))

    # ---- 4. is prediction related to outcome AT ALL? ----------------
    print("\n=== RANK CORRELATION, OUT OF SAMPLE " + "=" * 30)
    from scipy.stats import spearmanr
    for g in sorted(np.unique(Sg)):
        idx = np.flatnonzero(Sg == g)
        kp = R.independent_mask(Sy[idx], Po[idx], Hd[idx],
                                np.ones(len(idx), dtype=bool))
        ii = idx[kp]
        if len(ii) < 30:
            continue
        rho, pv = spearmanr(P[ii], Rn[ii])
        print(f"  {g:<18} n={len(ii):>4}  rho={rho:+.4f}  p={pv:.3f}")


if __name__ == "__main__":
    sys.exit(main())
