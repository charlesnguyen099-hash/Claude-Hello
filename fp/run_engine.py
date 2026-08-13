"""Walk-forward the whole thing and say plainly whether it works.

Three folds. Each trains on everything before a cut and trades only what
comes after, so no fold ever sees its own test bars. Within a fold every
(shape, side) is fitted across all ten coins at once, the potential score
is computed at each test bar, and a trade is taken where the score clears
a gate fixed BEFORE the numbers are read.

What is printed:

  CEILING     what perfect selection earns on the test bars. This is the
              operator's "trade tren data qua khu la loi 100%" as a
              number. It is real and it is large.
  MODEL       what the past-only logic earns on the same bars, at full
              cost, on independent trades.
  NULL        the same logic against a rotation of its own predictions.

MODEL above NULL is the only result that counts. MODEL above zero with
NULL above zero too means the market drifted, not that the logic worked.
"""
from __future__ import annotations

import sys

import numpy as np

from fp import engine as E
from fp import labels as LB

# Fixed before any number is read. A setup trades when the lower-bounded
# potential clears this; 20/100 means the estimate must sit a fifth of
# the way from break-even to certainty.
GATE = 20.0

FOLDS = (
    (("2026-05-31", "2026-07-01"), ("2026-07-01", "2026-07-15")),
    (("2026-05-31", "2026-07-15"), ("2026-07-15", "2026-07-30")),
    (("2026-05-31", "2026-07-30"), ("2026-07-30", "2026-08-14")),
)


def run_fold(train_win, test_win, gate=GATE, verbose=True):
    tr = E.load_panel(train_win)
    te = E.load_panel(test_win)
    keys = sorted(LB.SHAPES)
    rows = []
    P_all, N_all, S_all, X_all, H_all, K_all = [], [], [], [], [], []

    for tp, sl, hold in keys:
        for side in (1, -1):
            k = (tp, sl, hold, side)
            a = E.stack(tr, k, E.TRAIN_STRIDE)
            b = E.stack(te, k, E.EVAL_STRIDE)
            if a is None or b is None:
                continue
            Xtr, ytr = a[0], a[1]
            fit = E.fit_one(Xtr, ytr)
            if fit is None:
                continue
            Xte, yte, net, held, sym, pos = b
            p = E.p_hat(fit, Xte)
            # sigma is a factor column, so each test row carries its own.
            sg = Xte[:, list(tr[next(iter(tr))]["X"].columns).index("sigma")]
            p_be, _, _ = E.break_even(tp, sl, hold, sg)
            score = E.potential(p, p_be)
            P_all.append(score); N_all.append(net); S_all.append(sym)
            X_all.append(pos); H_all.append(held)
            K_all.append(np.full(len(p), f"{tp}/{sl}/{hold}/{side}"))
            take = score >= gate
            keep = E.independent(sym, pos, held, take)
            st = E.stats(net[keep])
            rows.append((f"{tp}/{sl}/{hold}", side, int(take.sum()),
                         st["n"], st["mean"], st["t"], st["win"]))
    if not P_all:
        return None
    return (np.concatenate(P_all), np.concatenate(N_all),
            np.concatenate(S_all), np.concatenate(X_all),
            np.concatenate(H_all), np.concatenate(K_all), rows)


def main():
    print("=" * 74)
    print("  WALK-FORWARD: train on the past, trade the future")
    print(f"  gate: potential >= {GATE:.0f}/100, fixed before reading anything")
    print("=" * 74, flush=True)

    grand = []
    for i, (trw, tew) in enumerate(FOLDS, 1):
        print(f"\n--- FOLD {i}   train {trw[0]}..{trw[1]}   "
              f"test {tew[0]}..{tew[1]}", flush=True)
        got = run_fold(trw, tew)
        if got is None:
            print("  no data")
            continue
        score, net, sym, pos, held, key, rows = got

        allkeep = E.independent(sym, pos, held, np.ones(len(net), bool))
        base = E.stats(net[allkeep])
        ceil_take = net > 0
        ck = E.independent(sym, pos, held, ceil_take)
        ceil = E.stats(net[ck])

        take = score >= GATE
        keep = E.independent(sym, pos, held, take)
        st = E.stats(net[keep])
        null = E.rotation_null(score, net, sym, pos, held,
                               lambda p: p >= GATE)
        pval = float((null >= st["mean"]).mean()) if st["n"] >= 2 else 1.0

        print(f"  CEILING (perfect selection) : {ceil['n']:>6} trades  "
              f"{100*ceil['mean']:+.4f}%/trade  {100*ceil['tot']:+.0f}% total")
        print(f"  BASELINE (take everything)  : {base['n']:>6} trades  "
              f"{100*base['mean']:+.4f}%/trade")
        print(f"  MODEL   (potential >= {GATE:.0f})    : {st['n']:>6} trades  "
              f"{100*st['mean']:+.4f}%/trade  t={st['t']:+.2f}  "
              f"win {100*st['win']:.1f}%  total {100*st['tot']:+.2f}%")
        print(f"  NULL    (rotated, 200x)     : "
              f"{100*null.mean():+.4f}%/trade  sd {100*null.std():.4f}   "
              f"p(null >= model) = {pval:.3f}")
        grand.append((st, base, ceil, pval))

        good = [r for r in rows if r[3] >= 10 and r[4] > 0]
        good.sort(key=lambda r: -r[5])
        if good:
            print(f"  best shapes this fold ({len(good)} of {len(rows)} "
                  f"positive):")
            for sh, sd, raw, n, mn, t, w in good[:6]:
                print(f"      {sh:<14} {'long' if sd>0 else 'short':<6} "
                      f"n={n:>4} {100*mn:+.4f}%  t={t:+.2f}  "
                      f"win {100*w:.0f}%")

    if not grand:
        return 1
    print("\n" + "=" * 74)
    tot_n = sum(g[0]["n"] for g in grand)
    tot_sum = sum(g[0]["mean"] * g[0]["n"] for g in grand
                  if g[0]["n"] >= 2)
    beat = sum(1 for g in grand if g[3] < 0.05)
    print(f"  ACROSS {len(grand)} FOLDS: {tot_n} independent trades, "
          f"{100*tot_sum/max(tot_n,1):+.4f}%/trade")
    print(f"  folds beating their own rotation null at p<0.05: "
          f"{beat}/{len(grand)}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
