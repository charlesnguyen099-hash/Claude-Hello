"""One clean split, all the data, and a straight answer.

fp/oos.py scored the SHIPPED model on the six days that arrived after it
was trained: 23 trades in the one band the walk-forward called
profitable, -0.678% each, worse than a rotation of its own predictions.
Six days and 23 trades is a small sample, and a small sample can be
unlucky.

So this refits from scratch on an early block and scores a long test
block -- 24 days instead of 6, roughly 4x the trades -- and asks the one
question that does not depend on any gate, threshold or band:

    OUT OF SAMPLE, IS THE PREDICTION RELATED TO THE OUTCOME AT ALL?

Rank correlation answers that with no tuning knob to hide behind. If
rho is zero, every band table built on top of it is a picture of noise,
and no amount of re-cutting the bands will change it. If rho is
positive and holds up, the gate can be rebuilt on the new data and the
bot has something to trade.

Reported alongside it, on the same rows:

  CEILING  what perfect selection earns on these bars -- the size of
           the prize, and the reason the operator is right that past
           data is 100% profitable in hindsight.
  DECILES  outcome by predicted decile. A model with real power makes
           this monotone. A model with none makes it flat and noisy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pathlib import Path

from fp import june as J
from fp import mtf_run as R
from fp import wf

HERE = Path(__file__).resolve().parent

TRAIN = ("2026-05-31", "2026-07-20")
TEST = ("2026-07-20", "2026-08-14")

# The band edges the shipped gate used, kept so the OUT-OF-SAMPLE table
# is directly comparable with the in-sample one it replaces.
BAND_EDGES = [-9e9, 0.000, 0.002, 0.004, 0.006, 0.009, 9e9]


def main():
    print(f"TRAIN {TRAIN[0]}..{TRAIN[1]}   TEST {TEST[0]}..{TEST[1]}")
    print("refitting from scratch -- the shipped model saw part of TEST\n",
          flush=True)

    P, Rn, Sy, Po, Hd, Sg, Bp = [], [], [], [], [], [], []
    ntr = 0
    for sh in R.SHAPES:
        tr = R.panel(TRAIN, sh)
        te = R.panel(TEST, sh)
        if tr is None or te is None:
            print(f"  {sh}: missing panel")
            continue
        Xtr, ytl, yts = tr[0], tr[1], tr[2]
        Xte, nl, ns, kl, ks, sy, po, sg = te
        ntr += len(Xtr)
        # b = what a win pays at this row's own volatility, net of the
        # round trip. Recorded so a band can be expressed as net/b, which
        # is comparable across barrier widths -- a mean percent is not.
        # NOT clamped. A row whose target does not clear the round trip
        # has b <= 0 and is not a trade -- potential() refuses it live.
        # Clamping it to 1e-9 instead turned a normal loss into net/b of
        # -83,310 and swamped the whole band average.
        b = sh[0] * sg - J.FEE
        for side, tag, ytr, real, held in ((1, "long", ytl, nl, kl),
                                           (-1, "short", yts, ns, ks)):
            m = wf.fit(Xtr, ytr)
            pr = m.predict(Xte)
            P.append(pr); Rn.append(real); Sy.append(sy); Po.append(po)
            Hd.append(held); Bp.append(b)
            Sg.append(np.full(len(pr), f"{sh[0]}_{sh[1]}_{sh[2]}_{tag}"))
        print(f"  {sh}: train {len(Xtr):,} -> test {len(Xte):,}", flush=True)

    P = np.concatenate(P); Rn = np.concatenate(Rn)
    Sy = np.concatenate(Sy); Po = np.concatenate(Po)
    Hd = np.concatenate(Hd); Sg = np.concatenate(Sg)
    Bp = np.concatenate(Bp)
    np.savez_compressed(HERE.parent / "data" / "verdict_scores.npz",
                        P=P, Rn=Rn, Sy=Sy, Po=Po, Hd=Hd, Sg=Sg, Bp=Bp)
    print(f"\ntrain rows {ntr:,}   scored rows {len(P):,}")

    # ---- the prize -------------------------------------------------
    print("\n=== CEILING (perfect selection, same bars) " + "=" * 23)
    for g in sorted(np.unique(Sg)):
        i = np.flatnonzero(Sg == g)
        n, mean, tot = wf.ceiling(Rn[i], Sy[i], Po[i], Hd[i])
        print(f"  {g:<18} {n:>5} trades  {mean:+.3f}%/trade  {tot:+.0f}% total")

    # ---- the question ----------------------------------------------
    print("\n=== IS PREDICTION RELATED TO OUTCOME? " + "=" * 28)
    print(f"  {'shape/side':<18} {'indep n':>8} {'rho':>9} {'p':>8}")
    rhos = []
    for g in sorted(np.unique(Sg)):
        i = np.flatnonzero(Sg == g)
        kp = R.independent_mask(Sy[i], Po[i], Hd[i],
                                np.ones(len(i), dtype=bool))
        ii = i[kp]
        rho, pv = spearmanr(P[ii], Rn[ii])
        rhos.append(rho)
        print(f"  {g:<18} {len(ii):>8} {rho:>+9.4f} {pv:>8.3f}")
    print(f"  {'MEAN rho':<18} {'':>8} {np.mean(rhos):>+9.4f}")

    # ---- deciles ----------------------------------------------------
    print("\n=== OUTCOME BY PREDICTED DECILE " + "=" * 34)
    keep = R.independent_mask(Sy, Po, Hd, np.ones(len(P), dtype=bool))
    p, r = P[keep], Rn[keep]
    q = pd.qcut(p, 10, labels=False, duplicates="drop")
    print(f"  {'decile':>6} {'n':>6} {'pred%':>9} {'real%':>9} {'t':>7}")
    for d in range(int(q.max()) + 1):
        m = q == d
        v = r[m]
        se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
        print(f"  {d:>6} {len(v):>6} {100*p[m].mean():>+9.4f} "
              f"{100*v.mean():>+9.4f} {v.mean()/se if se else 0:>+7.2f}")
    print(f"\n  ALL rows: n={len(r)}  mean={100*r.mean():+.4f}%  "
          f"-- this is the baseline any gate has to beat")

    # ---- top decile vs baseline -------------------------------------
    top = r[q == q.max()]
    diff = top.mean() - r.mean()
    se = np.sqrt(top.var(ddof=1) / len(top) + r.var(ddof=1) / len(r))
    print(f"  top decile beats baseline by {100*diff:+.4f}%  "
          f"(t={diff/se:+.2f})")

    # ---- rewrite the gate from the OUT-OF-SAMPLE measurement --------
    # The shipped mtf_bands.json was measured in-sample and called one
    # band profitable. That band is what the live bot traded. Measured
    # here, on bars the model never saw, the same edges give a different
    # answer -- and THAT is the number the bot has a right to size on.
    import json
    out = []
    print("\n=== GATE REWRITTEN FROM OUT-OF-SAMPLE BARS " + "=" * 23)
    print(f"  {'band':<16} {'indep':>6} {'net/b':>9} {'t':>7}")
    for i in range(len(BAND_EDGES) - 1):
        lo, hi = BAND_EDGES[i], BAND_EDGES[i + 1]
        t_ = (P >= lo) & (P < hi) & (Bp > 0)
        k_ = R.independent_mask(Sy, Po, Hd, t_)
        v = Rn[k_] / Bp[k_]
        if len(v) < 2:
            eob, tt = 0.0, 0.0
        else:
            se = v.std(ddof=1) / np.sqrt(len(v))
            eob = float(v.mean())
            tt = float(v.mean() / se) if se > 0 else 0.0
        out.append({"lo": float(lo), "hi": float(hi),
                    "trades": int(k_.sum()),
                    "edge_over_b": eob, "t": tt})
        lab = f"{'-inf' if lo < -1 else f'{lo:.3f}'}..{'+inf' if hi > 1 else f'{hi:.3f}'}"
        print(f"  {lab:<16} {int(k_.sum()):>6} {eob:>+9.3f} {tt:>+7.2f}")
    (HERE / "mtf_bands.json").write_text(json.dumps(out, indent=1))
    live = [b for b in out if b["edge_over_b"] > 0 and b["t"] >= 2.0]
    print(f"  wrote mtf_bands.json -- {len(live)} band(s) tradeable "
          f"(positive at t >= 2.0)")

    print("\n" + "=" * 66)
    if np.mean(rhos) > 0.02 and diff > 0:
        print("  There is something. Rebuild the gate on the new data.")
    else:
        print("  No relationship out of sample. Every band table built on")
        print("  these predictions is a picture of noise, and re-cutting")
        print("  the bands cannot fix it.")
    print("=" * 66)


if __name__ == "__main__":
    main()
