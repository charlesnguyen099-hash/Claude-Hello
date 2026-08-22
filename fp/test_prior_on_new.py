"""Does the CURRENTLY SHIPPED model hold up on data it has never seen?

This drop extends the cache BACKWARD again: one more week
(2025-12-31 to 2026-01-07) that sits BEFORE five coins' shipped fit
window begins (2026-01-07). That is a genuinely clean out-of-sample
slice -- a past the model was simply never shown -- and scoring the
model already on disk against exactly that slice, with no refitting,
is the most direct "does the old logic generalise" test available.

This is NOT fp/full.py's oof_gate(): that fits a fresh throwaway
classifier per fold to measure what a model SHAPED like this tends to
do. This loads the ACTUAL shipped model and asks what IT does, on the
one slice of real market it has not memorised.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from fp import costs as C
from fp import data as D
from fp import full as FU

# The exact boundary each coin's shipped model was fit up to, read
# from the report BEFORE this session's merge. A coin not listed here
# gained no new history in this drop and has nothing to test.
PRIOR_START = {
    "BLESSUSDT": "2026-01-07T17:00:00Z",
    "ETHUSDT": "2026-01-07T17:00:00Z",
    "HYPEUSDT": "2026-01-07T17:00:00Z",
    "SOLUSDT": "2026-01-07T17:00:00Z",
    "XRPUSDT": "2026-01-07T17:00:00Z",
}


def main():
    P = D.load()
    print("=" * 92)
    print("  DOES THE SHIPPED MODEL GENERALISE TO DATA IT HAS NEVER SEEN?")
    print("  Scoring the CURRENT fp/models/*.pkl.gz -- no refitting -- on")
    print("  the slice of each coin's history that sits BEFORE its fit window.")
    print("=" * 92, flush=True)

    rows = []
    for sym, start in sorted(PRIOR_START.items()):
        d = P.get(sym)
        if d is None:
            continue
        blob = FU.load_models(sym)
        if blob is None:
            print(f"\n--- {sym}: no shipped model, skipping")
            continue
        models, meta = blob["models"], blob["meta"]

        new_slice = d.loc[:start].iloc[:-1]      # strictly before the old start
        n_new = len(new_slice)
        if n_new < 2000:
            print(f"\n--- {sym}: only {n_new} bars in the new "
                  f"slice, skipping")
            continue

        cost = C.round_trip(d)
        cost = np.where(np.isfinite(cost), cost, np.nanmedian(cost))
        side_full, profit_full, mae_full, _ = FU.opportunities(d, cost)

        X = FU.features_for(d, P, sym)
        ok = np.isfinite(X).all(axis=1)
        idx_new = np.arange(n_new)
        ok_new = ok[idx_new]
        side_new = side_full[idx_new]
        cost_new = cost[idx_new]

        live = ok_new & (side_new != 0)
        if live.sum() < 200:
            print(f"\n--- {sym}: only {int(live.sum())} usable resolved "
                  f"bars in the new slice, skipping")
            continue

        Xn = np.ascontiguousarray(X[idx_new][live])
        truth = side_new[live]
        p_be = float(C.break_even(FU.FLOOR, float(np.nanmedian(cost_new))))

        pred, conf, profit_pred, mae_pred = FU.call(models, Xn)
        called = pred != 0
        acc = (float((pred[called] == truth[called]).mean())
              if called.any() else float("nan"))
        live_gate = meta.get("live_gate", 1.0)
        trusted = called & (conf > live_gate)
        acc_trusted = (float((pred[trusted] == truth[trusted]).mean())
                      if trusted.any() else float("nan"))

        print(f"\n--- {sym}   new slice: {n_new:,} bars "
              f"({str(d.index[0])[:10]} .. {start[:10]})")
        print(f"    resolved bars in slice     : {int(live.sum()):,}")
        print(f"    called (any confidence)    : {int(called.sum()):,}   "
              f"accuracy {100*acc:.2f}%   (need {100*p_be:.2f}%)")
        print(f"    called ABOVE live_gate      : {int(trusted.sum()):,}   "
              f"accuracy {100*acc_trusted:.2f}%"
              if trusted.any() else
              f"    called above live_gate     : 0 (gate {live_gate:.4f} "
              f"admits nothing)", flush=True)
        rows.append(dict(sym=sym, n=int(live.sum()), called=int(called.sum()),
                         acc=acc, trusted=int(trusted.sum()),
                         acc_trusted=acc_trusted, p_be=p_be))

    print("\n" + "=" * 92)
    if rows:
        beat = [r for r in rows if np.isfinite(r["acc_trusted"])
               and r["acc_trusted"] >= r["p_be"] and r["trusted"] >= 30]
        print(f"  coins tested            : {len(rows)}")
        print(f"  clear break-even at the live gate, on genuinely prior "
              f"data : {len(beat)}/{len(rows)}"
              + (f" ({', '.join(r['sym'] for r in beat)})" if beat else ""))
    print("=" * 92)
    return 0


if __name__ == "__main__":
    sys.exit(main())
