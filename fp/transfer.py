"""Does logic learned on one coin work on another?

The operator's requirement, stated plainly: *"logic không bị ràng buộc ...
đặc biệt là ở coin nào luôn vì đôi khi một số logic bạn đúc kết ở 1 coin
có thể apply cho coin khác"*. Nothing in the feature set forbids it --
all 200 columns are scale-free by construction, so a BTC row and a BLESS
row are directly comparable numbers. The question is whether the mapping
they feed is shared or per-coin, and that is measurable.

Three models, scored on exactly the same held-out rows of one coin:

  OWN     trained on that coin's own earlier bars.
          The baseline the shipped per-coin pipeline gives.

  POOLED  trained on every coin's earlier bars, target included.
          If this beats OWN, the coins reinforce each other and one
          shared model is the right object.

  FOREIGN trained on every coin EXCEPT the target, earlier bars only.
          This is the operator's claim under test. The model has never
          seen a single bar of the coin it is being asked to trade. If
          it still calls direction above that coin's break-even, the
          logic genuinely transfers. If it lands at 50%, the logic was
          per-coin all along and pooling only dilutes it.

TWO THINGS THIS TEST HAS TO GET RIGHT.

Time. Coins move together -- a BTC drop is a BLESS drop the same minute.
Training on other coins over the SAME minutes the target is tested on
would let the model read the answer off a correlated coin, and it would
score well while knowing nothing. So there is ONE cut, and every training
row from every coin comes from before it, every test row after.

Where that cut goes is not obvious, and the first version put it in the
wrong place. Splitting the FULL span 70/30 landed it in February 2026 --
before nine of the ten coins have a single bar. Every "pooled" fit was
silently BTC-only, "foreign" for a non-BTC coin was BTC-only too (hence
the two columns matching to the decimal), and "own" was empty for
everything but BTC. The test looked like it ran and measured nothing.

The cut has to sit inside the window all ten coins share, so it is
placed at 70% of the OVERLAP -- the span from the latest coin's first bar
to the earliest coin's last. BTC still trains on everything it has back
to January 2025; the newer coins train on what they have. That is "coin
nào có data bao nhiêu thì tính trên bấy nhiêu đó", with a cut every coin
can actually see.

Cost. The bar to clear is the target coin's own break-even, from its own
estimated spread -- 55.5% for BTC, 62.0% for BLESS. A pooled model that
hits 57% has an edge on BTC and no edge at all on BLESS.
"""
from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import numpy as np

from fp import costs as C
from fp import data as D
from fp import direction as DIR

# Per coin, per side of the cut. Keeps the pooled fit to a size that
# survives the box: ten coins x 12k is 120k rows x 200 float32 columns.
MAX_PER_COIN = 12_000
MAX_TEST = 8_000
TRAIN_FRAC = 0.70

# Building the 200 columns for 1.76M bars takes ~20 minutes, and this
# test wants re-running as the cut and the caps move. The matrices are a
# pure function of the cached bars, so they go on disk.
CACHE = Path(os.environ.get("FP_CACHE", "/tmp/fp-features"))


def prepare(P, horizon: int = DIR.HORIZON, use_cache: bool = True):
    """Features, labels and timestamps for every coin, once.

    Building the 200 columns is the expensive part and each coin needs
    them for three different fits, so it happens exactly once here -- and
    is memmapped back from disk on any later run. The features do not
    depend on the horizon; only the labels do, so they are keyed by it.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    out = {}
    for sym in sorted(P):
        d = P[sym]
        fx = CACHE / f"{sym}.X.npy"
        fy = CACHE / f"{sym}.y{horizon}.npy"
        hit = use_cache and fx.exists() and fy.exists()
        if hit:
            X = np.load(fx, mmap_mode="r")
            y = np.load(fy)
            hit = len(X) == len(d) and len(y) == len(d)
        if not hit:
            if fx.exists() and len(np.load(fx, mmap_mode="r")) == len(d):
                X = np.load(fx, mmap_mode="r")
            else:
                X = DIR.features(d, P, sym).values.astype("float32")
                np.save(fx, X)
            y = DIR.label(d["close"].values.astype("float64"),
                          horizon=horizon)
            np.save(fy, y)
        ok = np.isfinite(np.asarray(X)).all(axis=1)
        t = d.index.values.astype("datetime64[ns]")
        cost = float(np.nanmedian(C.round_trip(d)))
        out[sym] = dict(X=X, y=y, ok=ok, t=t, cost=cost,
                        p_be=float(C.break_even(DIR.WIN, cost)))
        print(f"    {sym:<12} {len(d):>9,} bars  "
              f"{int(ok.sum()):>9,} usable  cost {100*cost:.4f}%"
              f"   {'cached' if hit else 'built'}", flush=True)
        gc.collect()
    return out


def thin(idx, cap):
    """Even stride down to a cap. Neighbouring minutes are near-copies."""
    if len(idx) <= cap:
        return idx
    return idx[::int(np.ceil(len(idx) / cap))]


def rows(store, sym, lo, hi, cap):
    """Usable row indices for one coin inside a time window."""
    s = store[sym]
    m = s["ok"] & (s["t"] >= lo) & (s["t"] < hi)
    return thin(np.flatnonzero(m), cap)


def stack(store, syms, lo, hi, cap):
    """Training matrix pooled across coins, coin identity NOT included.

    Deliberately: if the model could read which coin a row came from it
    would learn ten private mappings under one roof and the foreign test
    would be meaningless. It sees only the 200 scale-free columns.
    """
    Xs, ys = [], []
    for sym in syms:
        i = rows(store, sym, lo, hi, cap)
        if len(i) < 500:
            continue
        Xs.append(store[sym]["X"][i])
        ys.append(store[sym]["y"][i])
    if not Xs:
        return None, None
    return np.vstack(Xs), np.concatenate(ys)


def independent(take_idx, spacing: int):
    """Thin a set of taken bars down to non-overlapping trades.

    THIS IS THE NUMBER THAT MATTERS, and leaving it out is how the first
    version of this table read as a result. The label looks 1440 minutes
    ahead, so bar i and bar i+1 are answering almost the same question
    about almost the same stretch of price. Counting 2,979 such rows as
    2,979 observations understates the standard error by a factor of
    ~sqrt(1440) = 38, and turns a coin flip into a five-sigma discovery.

    A greedy left-to-right pass keeps only bars whose horizons do not
    overlap. Those are real, separate bets.
    """
    out, last = [], -(10 ** 9)
    for i in take_idx:
        if i - last >= spacing:
            out.append(i)
            last = i
    return np.asarray(out, dtype=int)


def binom_p(k: int, n: int, p0: float) -> float:
    """P(at least k of n correct | true accuracy is exactly break-even)."""
    if n <= 0:
        return np.nan
    from math import comb
    return float(sum(comb(n, j) * p0 ** j * (1 - p0) ** (n - j)
                     for j in range(k, n + 1)))


def score(fitted, X, y, p_be, pos=None, horizon: int = DIR.HORIZON):
    """Accuracy on the rows it wants to trade -- naive AND independent.

    Returns (all-resolved accuracy, taken accuracy, n taken, independent
    accuracy, n independent, one-sided p against break-even).
    """
    if fitted is None:
        return np.nan, np.nan, 0, np.nan, 0, np.nan
    _, side, conf = DIR.predict(fitted, np.asarray(X))
    res = y != 0
    acc = float((side[res] == y[res]).mean()) if res.any() else np.nan
    take = res & (DIR.potential(conf, p_be) > 0)
    tacc = float((side[take] == y[take]).mean()) if take.any() else np.nan
    ti = np.flatnonzero(take)
    # Space by bar position in the original series, not by row order:
    # the test rows are strided, so consecutive rows are already minutes
    # apart by a variable amount.
    keep = independent(pos[ti] if pos is not None else ti, horizon)
    sel = np.isin(pos[ti] if pos is not None else ti, keep)
    ii = ti[sel]
    iacc = float((side[ii] == y[ii]).mean()) if len(ii) else np.nan
    pv = (binom_p(int((side[ii] == y[ii]).sum()), len(ii), p_be)
          if len(ii) else np.nan)
    return acc, tacc, int(take.sum()), iacc, int(len(ii)), pv


def main():
    horizon = int(sys.argv[1]) if len(sys.argv) > 1 else DIR.HORIZON
    P = D.load()
    print("=" * 88)
    print("  CROSS-COIN TRANSFER: is the logic shared, or per coin?")
    print("  OWN     = trained on the target coin's own earlier bars")
    print("  POOLED  = trained on all coins' earlier bars")
    print("  FOREIGN = trained on every OTHER coin, never the target")
    print("  All three scored on the SAME later bars of the target coin.")
    print(f"  horizon = {horizon} minutes for a +{100*DIR.WIN:.0f}% net move")
    # The horizon is the sample-size knob, not just a modelling choice.
    # Overlapping labels do not count as separate evidence, so a window
    # of W minutes can never yield more than W/horizon real bets per
    # coin. At 1440 that is ~20 per coin over this data, which cannot
    # resolve a 6-point question no matter how the model is built.
    print("=" * 88, flush=True)

    print("\n  building features")
    store = prepare(P, horizon)
    syms = sorted(store)

    # ONE cut, placed inside the window every coin shares. Splitting the
    # full span instead puts it before nine coins exist, and then every
    # "pooled" fit is quietly BTC-only.
    lo = min(s["t"][0] for s in store.values())
    hi = max(s["t"][-1] for s in store.values())
    ov_lo = max(s["t"][0] for s in store.values())
    ov_hi = min(s["t"][-1] for s in store.values())
    cut = ov_lo + (ov_hi - ov_lo) * TRAIN_FRAC
    print(f"\n  overlap  {str(ov_lo)[:16]} .. {str(ov_hi)[:16]}")
    print(f"  time cut {str(cut)[:16]}   train < cut < test", flush=True)
    # Every coin must contribute training rows, or the comparison below
    # is between models that differ by nothing.
    have = {q: len(rows(store, q, lo, cut, MAX_PER_COIN)) for q in syms}
    thin_coins = [q for q, n in have.items() if n < 500]
    if thin_coins:
        print(f"  WARNING: no usable training rows for "
              f"{', '.join(thin_coins)} -- pooled and foreign will coincide")

    mix = ", ".join("%s:%dk" % (q[:-4], n // 1000)
                    for q, n in have.items() if n >= 500)
    Xp_all, yp_all = stack(store, syms, lo, cut, MAX_PER_COIN)
    pooled = DIR.fit(Xp_all, yp_all) if Xp_all is not None else None
    print(f"  pooled model: {0 if Xp_all is None else len(Xp_all):,} rows "
          f"from {sum(1 for n in have.values() if n >= 500)} coins ({mix})",
          flush=True)
    del Xp_all, yp_all
    gc.collect()

    # Two accuracies per model. The first counts every taken bar, the
    # second only bars whose 1440-minute horizons do not overlap -- and
    # only the second one can carry a p-value.
    print(f"\n  {'coin':<12}{'need':>7}{'own':>8}{'pool':>8}{'foreign':>8}"
          f"  |{'indep foreign':>16}{'n':>5}{'p':>8}   verdict")
    print("  " + "-" * 86)
    table = []
    for sym in syms:
        te = rows(store, sym, cut, hi + np.timedelta64(1, "D"), MAX_TEST)
        if len(te) < 300:
            print(f"  {sym:<12} too few test rows after the cut")
            continue
        s = store[sym]
        Xte, yte, p_be = s["X"][te], s["y"][te], s["p_be"]

        tr = rows(store, sym, lo, cut, MAX_PER_COIN * 3)
        own = DIR.fit(s["X"][tr], s["y"][tr]) if len(tr) >= 2000 else None

        others = [q for q in syms if q != sym]
        Xf, yf = stack(store, others, lo, cut, MAX_PER_COIN)
        foreign = DIR.fit(Xf, yf) if Xf is not None else None
        del Xf, yf
        gc.collect()

        _, t_own, _, i_own, n_own, p_own = score(own, Xte, yte, p_be, te, horizon)
        _, t_pool, _, i_pool, n_pool, p_pool = score(pooled, Xte, yte, p_be, te, horizon)
        _, t_for, _, i_for, n_for, p_for = score(foreign, Xte, yte, p_be, te, horizon)
        del own, foreign
        gc.collect()

        # A verdict needs BOTH: above the coin's break-even on
        # independent trades, and unlikely to be luck.
        if np.isfinite(i_for) and i_for >= p_be and np.isfinite(p_for) \
                and p_for < 0.05:
            verdict = "TRANSFERS"
        elif np.isfinite(i_own) and i_own >= p_be and np.isfinite(p_own) \
                and p_own < 0.05:
            verdict = "own only"
        else:
            verdict = "not proven"
        print(f"  {sym:<12}{100*p_be:>6.1f}%{100*t_own:>7.1f}%"
              f"{100*t_pool:>7.1f}%{100*t_for:>7.1f}%  |"
              f"{100*i_for:>15.1f}%{n_for:>5}{p_for:>8.3f}   {verdict}",
              flush=True)
        table.append(dict(sym=sym, p_be=p_be, own=t_own, pooled=t_pool,
                          foreign=t_for, i_own=i_own, i_pooled=i_pool,
                          i_foreign=i_for, n_indep=n_for, p=p_for))

    print("  " + "-" * 86)
    if table:
        for name, key in (("own", "i_own"), ("pooled", "i_pooled"),
                          ("foreign", "i_foreign")):
            v = np.array([r[key] - r["p_be"] for r in table
                          if np.isfinite(r[key])])
            if len(v):
                print(f"  {name:<8} edge on INDEPENDENT trades, over each "
                      f"coin's own break-even: {100*v.mean():+.2f} pts, "
                      f"positive on {int((v > 0).sum())}/{len(v)}")
        tot = sum(r["n_indep"] for r in table)
        w = [r for r in table if np.isfinite(r["i_foreign"])
             and r["i_foreign"] >= r["p_be"] and r["p"] < 0.05]
        print(f"\n  independent trades in the whole test window: {tot}")
        print(f"  coins where a FOREIGN model beats the fee, and it is not "
              f"luck: {len(w)}/{len(table)}"
              f"{' -- ' + ', '.join(r['sym'] for r in w) if w else ''}")
        if tot < 200:
            print("\n  READ THIS BEFORE BELIEVING THE TABLE ABOVE.")
            print("  A 1440-minute label inside a test window this short")
            print("  leaves only a couple of dozen genuinely separate bets")
            print("  per coin. At that sample size the per-coin accuracies")
            print("  swing 20 points on noise alone -- which is exactly what")
            print("  the spread between the 'own', 'pool' and 'foreign'")
            print("  columns is showing. Nothing here is decidable yet.")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    sys.exit(main())
