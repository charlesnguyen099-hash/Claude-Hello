"""Does a trade's POTENTIAL SCORE predict what the trade earns?

Every study so far asked "does this rule have an edge?", and paid for
1.4 million tests to ask it. All of them failed their own null.

This asks a different question, and a much better posed one: forget
which rule fired. At the moment of entry, score the setup from
past-only information, then look at what trades in each score band
actually went on to earn. If high scores earn more than low scores,
sizing by score works -- and it works WITHOUT any single rule needing
to be significant, because the claim is about the mapping, not about
any one logic.

That is a handful of hypotheses instead of a million, so the evidence
bar is one a month of data can actually clear.

THE SCORE USES ONLY THE PAST. At trade k of a given (logic, symbol,
timeframe, target, stop, limit) cell, the score is built from trades
1..k-1 of that same cell and from the bar's own volatility. Nothing
from trade k or after it enters. The expanding mean is shifted by one,
which is the whole ballgame -- the fitted books in this repo all died
of exactly this mistake in one form or another.

    edge_hat = trailing mean net, shrunk toward zero by how few
               trades stand behind it
    score    = 100 x edge_hat / b,  b = what a win pays after the round
               trip. Same scale the bot already sizes on: 0 is
               break-even, 100 is "never loses".

Then trades are bucketed by score and the realized net of each bucket
is measured. Monotone and positive at the top means the score is worth
sizing on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fp import june as J
from fp import june_tiers as T


def trailing_score(net: np.ndarray, b: float, a: float) -> np.ndarray:
    """Score at each trade, from that cell's OWN earlier trades only.

    net is the realized outcome of each independent trade in order. The
    estimate available before trade k is the mean of trades 1..k-1,
    shrunk toward zero by n/(n+n0): with two trades behind it a cell
    claims almost nothing, with fifty it claims most of what it has
    shown. n0 is the sample size at which a two-sigma distinction from
    break-even becomes possible, computed from the barriers alone -- so
    it is derived, not chosen, and it cannot collapse the way a sample
    variance can when the early trades happen to agree.
    """
    n = len(net)
    csum = np.concatenate([[0.0], np.cumsum(net)])
    k = np.arange(n, dtype=float)                 # trades available before k
    with np.errstate(invalid="ignore", divide="ignore"):
        trail = np.where(k > 0, csum[:-1] / np.maximum(k, 1), 0.0)
    p_be = a / (a + b)
    sd0 = np.sqrt(max(p_be * b * b + (1 - p_be) * a * a, 1e-12))
    # n0 from a two-sigma distinction against break-even, using the
    # trailing mean itself as the effect size being tested.
    with np.errstate(invalid="ignore", divide="ignore"):
        n0 = np.where(np.abs(trail) > 0, (2.0 * sd0 / np.abs(trail)) ** 2, 1e9)
    w = k / (k + np.maximum(n0, 1.0))
    edge = trail * w
    return 100.0 * edge / b


# Score bands. Wide at the bottom because most trades score near zero,
# and open-ended at the top because the point is to find out whether the
# top band is worth anything.
EDGES = np.array([0.0, 2.0, 5.0, 10.0, 20.0, 35.0, 50.0, np.inf])


def accumulate(tfs=None, symbols=None, halves: bool = True) -> pd.DataFrame:
    """Bucket every independent trade by its past-only score, on the fly.

    Storing one row per trade means fifteen million tuples and an OOM
    before the DataFrame is even built. Nothing here needs the rows --
    only per-band counts, sums and sums of squares, which is all a mean,
    a t and a win rate require.
    """
    from fp import ensemble as E
    from fp.exits import sigma_at

    D = J.load()
    if symbols:
        D = {s: v for s, v in D.items() if s in symbols}
    tfs = tfs or T.TFS
    nb = len(EDGES) - 1
    acc = {}                       # (tf, half) -> [n, sum, sumsq, wins] per band

    def add(key, band, x):
        a = acc.setdefault(key, np.zeros((nb, 4)))
        np.add.at(a, (band, 0), 1.0)
        np.add.at(a, (band, 1), x)
        np.add.at(a, (band, 2), x * x)
        np.add.at(a, (band, 3), (x > 0).astype(float))

    for label, m in tfs:
        for sym, d1 in D.items():
            d = J.resample(d1, m)
            if len(d) < 320:
                continue
            L = E.build_logics(d)
            outs = T.outcomes(d, m)
            n = len(d)
            mid = n // 2
            sg = float(np.nanmedian(sigma_at(d["close"].values.astype(float))))
            for name, ser in L.items():
                v = np.asarray(ser.values, dtype=float)
                for side in (1, -1):
                    on = np.zeros(n, dtype=bool)
                    on[1:] = (v[1:] == side) & (v[:-1] != side)
                    idx = np.flatnonzero(on)
                    if len(idx) < 8:
                        continue
                    for (s2, tp, sl, hm), (net, held) in outs.items():
                        if s2 != side:
                            continue
                        i2 = T.independent(idx, net, held)
                        if len(i2) < 8:
                            continue
                        x = net[i2]
                        bb, aa = tp * sg - J.FEE, sl * sg + J.FEE
                        if bb <= 0:
                            continue
                        band = np.clip(
                            np.searchsorted(EDGES, trailing_score(x, bb, aa),
                                            side="right") - 1, 0, nb - 1)
                        add((label, "all"), band, x)
                        if halves:
                            late = i2 >= mid
                            if late.any():
                                add((label, "late"), band[late], x[late])
                            if (~late).any():
                                add((label, "early"), band[~late], x[~late])
            del L, outs
        print(f"  {label} done", flush=True)

    rows = []
    for (tf, half), a in acc.items():
        for i in range(nb):
            nn, s1, s2, w = a[i]
            if nn < 1:
                continue
            mean = s1 / nn
            var = max(s2 / nn - mean * mean, 0.0)
            se = np.sqrt(var / nn) if nn > 1 else np.inf
            rows.append({"tf": tf, "half": half,
                         "band": f"{EDGES[i]:g}-{EDGES[i+1]:g}",
                         "trades": int(nn), "mean_pct": 100 * mean,
                         "win_pct": 100 * w / nn,
                         "t": mean / se if se > 0 else 0.0})
    return pd.DataFrame(rows)


def collect(tfs=None, symbols=None) -> pd.DataFrame:
    """Every independent trade in the library, with its past-only score.

    Returns one row per trade: the score available before it opened and
    the net it actually earned. Rule identity is kept only so the result
    can be split by timeframe and symbol, never to select on.
    """
    from fp import ensemble as E

    D = J.load()
    if symbols:
        D = {s: v for s, v in D.items() if s in symbols}
    tfs = tfs or T.TFS
    rows = []
    for label, m in tfs:
        for sym, d1 in D.items():
            d = J.resample(d1, m)
            if len(d) < 320:
                continue
            L = E.build_logics(d)
            outs = T.outcomes(d, m)
            n = len(d)
            for name, ser in L.items():
                v = np.asarray(ser.values, dtype=float)
                for side in (1, -1):
                    on = np.zeros(n, dtype=bool)
                    on[1:] = (v[1:] == side) & (v[:-1] != side)
                    idx = np.flatnonzero(on)
                    if len(idx) < 8:
                        continue
                    for (s2, tp, sl, hm), (net, held) in outs.items():
                        if s2 != side:
                            continue
                        i2 = T.independent(idx, net, held)
                        if len(i2) < 8:
                            continue
                        x = net[i2]
                        # Barrier sizes in the same units the score wants:
                        # what a win pays and what a loss costs, after the
                        # round trip, using this cell's own sigma scale.
                        from fp.exits import sigma_at
                        sg = float(np.nanmedian(
                            sigma_at(d["close"].values.astype(float))))
                        bb, aa = tp * sg - J.FEE, sl * sg + J.FEE
                        if bb <= 0:
                            continue
                        sc = trailing_score(x, bb, aa)
                        for j in range(len(x)):
                            rows.append((label, sym, sc[j], x[j], i2[j], n))
            del L, outs
    return pd.DataFrame(rows, columns=["tf", "sym", "score", "net",
                                       "bar", "nbars"])


def buckets(T_: pd.DataFrame, edges=(0, 5, 10, 20, 35, 50, 1e9)) -> pd.DataFrame:
    """Realized net per score band. The question in one table."""
    lab = pd.cut(T_.score, bins=list(edges), right=False)
    g = T_.groupby(lab, observed=True)["net"]
    out = pd.DataFrame({
        "trades": g.size(),
        "mean_pct": 100 * g.mean(),
        "median_pct": 100 * g.median(),
        "win_pct": 100 * T_.assign(w=T_.net > 0).groupby(lab, observed=True)["w"].mean(),
        "t": g.mean() / (g.std(ddof=1) / np.sqrt(g.size())),
    })
    return out
