"""Marginal band table from the walk-forward, at fine resolution.

The shipped gate was cumulative and used as if it were per-band, so the
bot traded two bands the study had measured as LOSING. This re-derives
what a trade landing IN each band is worth, which is the only number a
gate may be built from.
"""
import numpy as np, pandas as pd
from fp import wf, mtf_run as R

FOLDS = [
    (("2026-05-31", "2026-06-11"), ("2026-06-11", "2026-06-17")),
    (("2026-05-31", "2026-06-17"), ("2026-06-17", "2026-06-23")),
    (("2026-05-31", "2026-06-23"), ("2026-06-23", "2026-06-30")),
]
EDGES = [-1e9, 0.0, 0.002, 0.004, 0.006, 0.009, 0.013, 1e9]

acc = {i: [0, 0.0, 0.0, 0] for i in range(len(EDGES) - 1)}
for k, (trw, tew) in enumerate(FOLDS, 1):
    got = wf.fold(trw, tew)
    if got is None:
        continue
    bp, br, bs, bx, bh, ntr, bb = got
    # Independence must be resolved per band: a trade only blocks the
    # window it actually occupies.
    for i in range(len(EDGES) - 1):
        take = (bp >= EDGES[i]) & (bp < EDGES[i + 1])
        keep = R.independent_mask(bs, bx, bh, take)
        # net / b: what the trade earned as a SHARE of what its own win
        # pays. Dimensionless, so a band measured across many barrier
        # widths can be applied to one setup without claiming a win rate
        # its target cannot support.
        ok = bb[keep] > 0
        v = (br[keep] / bb[keep])[ok]
        if len(v) == 0:
            continue
        acc[i][0] += len(v); acc[i][1] += v.sum(); acc[i][2] += (v*v).sum()
        acc[i][3] += int((v > 0).sum())
    print(f"fold {k} done ({ntr:,} train rows)", flush=True)

print(f"\n{'band':>18}{'trades':>8}{'net/b':>10}{'win%':>8}{'t':>8}")
rows = []
for i in range(len(EDGES) - 1):
    n, s1, s2, w = acc[i]
    if n < 2:
        continue
    m = s1 / n
    sd = np.sqrt(max(s2/n - m*m, 0.0))
    t = m / (sd/np.sqrt(n)) if sd > 0 else 0.0
    lo = "-inf" if EDGES[i] < -1 else f"{EDGES[i]:.3f}"
    hi = "+inf" if EDGES[i+1] > 1 else f"{EDGES[i+1]:.3f}"
    flag = "  <-- pays" if m > 0 and t > 1 else ("  LOSES" if m < 0 else "")
    print(f"{lo+'..'+hi:>18}{n:>8}{m:>10.4f}{100*w/n:>7.1f}%{t:>8.2f}{flag}")
    rows.append({"lo": EDGES[i], "hi": EDGES[i+1], "trades": n,
                 "edge_over_b": m, "win_pct": 100*w/n, "t": t})
pd.DataFrame(rows).to_json("/home/user/Claude-Hello/fp/mtf_bands.json",
                           orient="records", indent=1)
print("\nwrote fp/mtf_bands.json")
print("DONE", flush=True)
