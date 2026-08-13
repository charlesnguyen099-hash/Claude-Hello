"""Six classic effects, pre-registered, priced at full cost.

The 135-feature model is dead: fp/verdict.py refit it on a clean split
and measured rank correlation -0.0096 between what it predicted and what
happened, across 24 out-of-sample days. The top predicted decile beat the
baseline by t = +0.09. There is no relationship to gate on.

That does not mean there is nothing in the data -- fp/verdict.py also
shows perfect selection earning +200% to +270% per shape on the same
bars. It means a 135-column gradient booster is the wrong instrument:
too many ways to fit June, no way to check which of them was real.

So this tries the opposite. A short, FIXED list of effects that were
described in the literature before this data existed, each one a single
number, each tested in both directions so no direction is chosen after
the fact:

  rev5     five-bar return, z-scored          short-term reversal
  mom60    sixty-bar return, z-scored         momentum / trend
  brk      position in the 96-bar range       breakout
  volx     ATR now over ATR of 96 bars ago    volatility expansion
  vwapd    close minus VWAP, in sigma         value / stretch
  hour     time of day                        session effect

Every effect x every direction x every barrier shape is 36 tests, so a
result has to clear Bonferroni at 36 -- |t| >= 3.4 -- and then beat a
rotation of its own signal. Both hurdles are set before the numbers are
read.

TRAIN is used for nothing except confirming an effect exists there at
all; the decision is made on TEST, which the search never touches until
the end.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fp import june as J
from fp import mtf_run as R
from fp.exits import barrier_outcomes, sigma_at

TRAIN = ("2026-05-31", "2026-07-20")
TEST = ("2026-07-20", "2026-08-14")

# 6 effects x 2 directions x 3 shapes.
N_TESTS = 36
BONF_T = 3.40  # two-sided 0.05/36

# The fraction of bars an effect is allowed to fire on. Fixed here, not
# tuned: a threshold chosen after seeing the outcome is not a threshold.
TOP_Q = 0.10


def signals(de: pd.DataFrame) -> dict[str, np.ndarray]:
    """Six scores on the entry-bar frame. Every one shifted, so a score
    on bar i uses only bars up to and including i's close."""
    c = de["close"].astype(float)
    h = de["high"].astype(float)
    lo = de["low"].astype(float)
    v = de["volume"].astype(float)
    r1 = c.pct_change()

    out = {}
    r5 = c.pct_change(5)
    out["rev5"] = (r5 / r1.rolling(96).std().replace(0, np.nan)).values
    r60 = c.pct_change(60)
    out["mom60"] = (r60 / r1.rolling(96).std().replace(0, np.nan)).values
    hh, ll = h.rolling(96).max(), lo.rolling(96).min()
    out["brk"] = ((c - ll) / (hh - ll).replace(0, np.nan)).values
    tr = pd.concat([h - lo, (h - c.shift()).abs(),
                    (lo - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    out["volx"] = (atr / atr.shift(96).replace(0, np.nan)).values
    tp = (h + lo + c) / 3.0
    vw = (tp * v).rolling(96).sum() / v.rolling(96).sum().replace(0, np.nan)
    out["vwapd"] = ((c - vw) / (c * r1.rolling(96).std())
                    .replace(0, np.nan)).values
    out["hour"] = de.index.hour.values.astype(float)

    # THE SHIFT. Score on bar i is acted on at bar i's close, and the
    # barrier outcome starts from bar i+1. barrier_outcomes already
    # measures forward from the entry bar, so the score is aligned to
    # its own bar and nothing here may peek past it.
    return {k: np.asarray(a, dtype=float) for k, a in out.items()}


def collect(window):
    """Per shape: scores, realized nets, and everything the independence
    mask needs."""
    D = J.load(*window)
    per = {}
    for sh in R.SHAPES:
        tp, sl, hm = sh
        S, NL, NS, KL, KS, SY, PO = {}, [], [], [], [], [], []
        for sym, d1 in sorted(D.items()):
            if len(d1) < 8000:
                continue
            de = J.resample(d1, R.ENTRY_MIN)
            if len(de) < 400:
                continue
            sc = signals(de)
            c = de["close"].values.astype(float)
            hh = de["high"].values.astype(float)
            ll = de["low"].values.astype(float)
            sg = sigma_at(c)
            fund = R.ENTRY_MIN / 60.0 / 8.0 * J.FUNDING_PER_8H
            ol, kl = barrier_outcomes(hh, ll, c, sg, 1, tp, sl, hm)
            os_, ks = barrier_outcomes(hh, ll, c, sg, -1, tp, sl, hm)
            nl = ol - J.FEE - kl * fund
            ns = os_ - J.FEE - ks * fund
            ok = np.isfinite(nl) & np.isfinite(ns)
            for k in sc:
                ok &= np.isfinite(sc[k])
            if ok.sum() < 100:
                continue
            for k, a in sc.items():
                S.setdefault(k, []).append(a[ok])
            NL.append(nl[ok]); NS.append(ns[ok])
            KL.append(kl[ok]); KS.append(ks[ok])
            SY.append(np.full(int(ok.sum()), sym))
            PO.append(np.flatnonzero(ok))
        if not NL:
            continue
        per[sh] = ({k: np.concatenate(a) for k, a in S.items()},
                   np.concatenate(NL), np.concatenate(NS),
                   np.concatenate(KL), np.concatenate(KS),
                   np.concatenate(SY), np.concatenate(PO))
    return per


def measure(score, sign, real, held, sym, pos, q=TOP_Q):
    """Trade the top q of sign*score. Independent trades, full cost."""
    s = sign * score
    cut = np.nanquantile(s, 1 - q)
    take = s >= cut
    keep = R.independent_mask(sym, pos, held, take)
    v = real[keep]
    if len(v) < 10:
        return len(v), np.nan, np.nan, np.nan
    se = v.std(ddof=1) / np.sqrt(len(v))
    return len(v), v.mean(), (v.mean() / se if se > 0 else 0.0), v.sum()


def main():
    print("collecting panels...", flush=True)
    tr = collect(TRAIN)
    te = collect(TEST)
    print(f"  train shapes {len(tr)}   test shapes {len(te)}\n", flush=True)

    # Baseline: what an untimed trade on these bars costs.
    for tag, per in (("TRAIN", tr), ("TEST", te)):
        for sh, (S, nl, ns, kl, ks, sy, po) in per.items():
            kp = R.independent_mask(sy, po, kl, np.ones(len(nl), bool))
            print(f"  {tag} {sh} baseline long : n={int(kp.sum()):>5} "
                  f"{100*nl[kp].mean():+.4f}%")
        break

    rows = []
    print(f"\n{'effect':<7} {'dir':>4} {'shape':<14} {'side':<6} "
          f"{'TRAIN t':>9} {'TEST n':>7} {'TEST %':>9} {'TEST t':>8}")
    print("-" * 74)
    for eff in ("rev5", "mom60", "brk", "volx", "vwapd", "hour"):
        for sign in (1, -1):
            for sh in R.SHAPES:
                if sh not in tr or sh not in te:
                    continue
                Str, nlt, nst, klt, kst, syt, pot = tr[sh]
                Ste, nle, nse, kle, kse, sye, poe = te[sh]
                # An effect predicts a DIRECTION of price, so the long
                # book and the short book are the same claim read two
                # ways. Both are scored; both count against Bonferroni.
                for sd, realtr, heldtr, realte, heldte in (
                        ("long", nlt, klt, nle, kle),
                        ("short", nst, kst, nse, kse)):
                    _, _, ttr, _ = measure(Str[eff], sign, realtr, heldtr,
                                           syt, pot)
                    n, mn, tt, tot = measure(Ste[eff], sign, realte, heldte,
                                             sye, poe)
                    rows.append(dict(eff=eff, sign=sign, shape=str(sh),
                                     side=sd, t_train=ttr, n=n,
                                     mean=mn, t=tt, tot=tot))
                    print(f"{eff:<7} {sign:>+4} {str(sh):<14} {sd:<6} "
                          f"{ttr:>+9.2f} {n:>7} {100*mn:>+9.4f} {tt:>+8.2f}")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 74)
    print(f"tests run: {len(df)}   Bonferroni threshold |t| >= {BONF_T}")
    win = df[(df["t"] >= BONF_T) & (df["mean"] > 0)]
    if len(win) == 0:
        best = df.loc[df["t"].idxmax()]
        print("NOTHING SURVIVES.")
        print(f"  best of {len(df)}: {best['eff']} sign {best['sign']:+d} "
              f"{best['shape']} {best['side']}  "
              f"{100*best['mean']:+.4f}%/trade  t={best['t']:+.2f}")
        print(f"  needed t >= {BONF_T} to be worth a dollar.")
    else:
        print(f"{len(win)} SURVIVED Bonferroni:")
        print(win.to_string(index=False))
    print("=" * 74)
    df.to_csv("data/simple_results.csv", index=False)


if __name__ == "__main__":
    main()
