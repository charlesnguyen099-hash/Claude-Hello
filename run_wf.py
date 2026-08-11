"""Walk-forward over June: ceiling, model, and the shuffled-label null."""
import numpy as np, pandas as pd
from fp import wf, mtf_run as R

FOLDS = [
    (("2026-05-31", "2026-06-11"), ("2026-06-11", "2026-06-17")),
    (("2026-05-31", "2026-06-17"), ("2026-06-17", "2026-06-23")),
    (("2026-05-31", "2026-06-23"), ("2026-06-23", "2026-06-30")),
]
GATES = (0.0, 0.001, 0.002, 0.005)

def score(bp, br, bs, bx, bh, gate):
    take = bp > gate
    keep = R.independent_mask(bs, bx, bh, take)
    v = br[keep]
    if len(v) < 2:
        return len(v), np.nan, 0.0, np.nan
    se = v.std(ddof=1) / np.sqrt(len(v))
    return (len(v), 100*v.mean(), 100*v.sum(),
            v.mean()/se if se > 0 else 0.0)

rows, nulls = [], []
for k, (trw, tew) in enumerate(FOLDS, 1):
    got = wf.fold(trw, tew)
    if got is None:
        print(f"fold {k}: no data"); continue
    bp, br, bs, bx, bh, ntr = got
    cn, cm, ct = wf.ceiling(br, bs, bx, bh)
    print(f"\n=== FOLD {k}  train {trw[0]}..{trw[1]}  test {tew[0]}..{tew[1]} "
          f"({ntr:,} train rows) ===")
    print(f"  CEILING (perfect selection): {cn} trades  "
          f"{cm:+.3f}%/trade  total {ct:+.1f}%")
    for g in GATES:
        n, m, t_, tt = score(bp, br, bs, bx, bh, g)
        rows.append({"fold": k, "gate": g, "trades": n, "mean_pct": m,
                     "total_pct": t_, "t": tt})
        print(f"  MODEL gate {g:<6}: {n:>4} trades  {m:+.3f}%/trade  "
              f"total {t_:+.1f}%  t={tt:+.2f}")
    ng = wf.fold(trw, tew, shuffle=True, seed=100+k)
    if ng is not None:
        p2, r2, s2, x2, h2, _ = ng
        for g in GATES:
            n, m, t_, tt = score(p2, r2, s2, x2, h2, g)
            nulls.append({"fold": k, "gate": g, "trades": n, "mean_pct": m})
        nn, nm, _, _ = score(p2, r2, s2, x2, h2, 0.002)
        print(f"  NULL  gate 0.002 : {nn:>4} trades  {nm:+.3f}%/trade "
              f"(labels shuffled)")

D = pd.DataFrame(rows); N = pd.DataFrame(nulls)
print("\n=== POOLED ACROSS FOLDS ===")
for g in GATES:
    d = D[D.gate == g]; nu = N[N.gate == g]
    tot = d.trades.sum()
    if tot == 0: continue
    mm = np.average(d.mean_pct.fillna(0), weights=d.trades.clip(lower=1))
    nmm = (np.average(nu.mean_pct.fillna(0), weights=nu.trades.clip(lower=1))
           if len(nu) else np.nan)
    print(f"  gate {g:<6}: {tot:>4} trades   model {mm:+.3f}%/trade   "
          f"null {nmm:+.3f}%/trade   edge {mm-nmm:+.3f}")
print("\nDONE", flush=True)
