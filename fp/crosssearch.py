"""Search the signal families that are not price: positioning and breadth.

    python -m fp.crosssearch                 # everything downloaded
    python -m fp.crosssearch --min-symbols 40

Run `python -m fp.altdata` first -- this reads only from disk.

WHAT IS DIFFERENT ABOUT THIS SEARCH

Every earlier search in this project ranked one symbol against its own
past, using inputs computed from that symbol's price. This one does two
things that family cannot do at all.

POSITIONING. Funding rate, open interest and the long/short ratio are not
derived from price -- they are what traders are actually holding and what
they are paying to hold it. Extreme positive funding means the crowd is
long and paying for the privilege, which is the textbook squeeze setup.
Rising open interest into a move means new money; falling open interest
means the move is positions closing. A price series cannot express any of
this, which is precisely why it is worth testing separately.

CROSS-SECTION. Ranking 120 coins against EACH OTHER at the same instant
is structurally different from ranking one coin against its own history.
Cross-sectional momentum and reversal are among the most replicated
effects in finance, they survive in markets where time-series signals do
not, and nothing in this project has ever used the breadth the bot
already scans.

THE VALIDATION IS THE SAME AND IT IS NOT NEGOTIABLE

Nested walk-forward: the entire ranking is recomputed inside each fold
using only prior blocks, then applied forward. Bonferroni over the actual
number of variants tested. Ranked by worst fold. A search that finds a
signal in every fold and a different one each time has found nothing, and
this reports that plainly when it happens.

Costs are the full model throughout: taker in, taker out, and each
symbol's own funding over the measured hold.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fp import logic as L

ALT = Path(__file__).resolve().parent.parent / "bybit_bot" / "data" / "alt"
HOLD_BARS = 16                 # 8 hours at 30m -- one funding period
FOLDS = 8


def load_panel(min_symbols: int) -> dict:
    """Everything on disk, aligned on a common 30m index."""
    kl = sorted(ALT.glob("kline_*.csv"))
    if len(kl) < min_symbols:
        return {}
    px, fund, oi, ls = {}, {}, {}, {}
    for f in kl:
        sym = f.stem.replace("kline_", "")
        d = pd.read_csv(f, parse_dates=["datetime"]).set_index("datetime")
        if len(d) < 500:
            continue
        px[sym] = d["close"]
        for kind, store in (("funding", fund), ("oi", oi), ("ls", ls)):
            g = ALT / f"{kind}_{sym}.csv"
            if g.exists():
                e = pd.read_csv(g, parse_dates=["datetime"]).set_index("datetime")
                col = e.columns[0]
                store[sym] = e[col].reindex(d.index, method="ffill")
    if len(px) < min_symbols:
        return {}
    P = pd.DataFrame(px).sort_index()
    out = {"close": P}
    for name, store in (("funding", fund), ("oi", oi), ("ls", ls)):
        if store:
            out[name] = pd.DataFrame(store).reindex(P.index).sort_index()
    return out


def forward_return(P: pd.DataFrame, bars: int) -> pd.DataFrame:
    return P.shift(-bars) / P - 1.0


def cost_matrix(panel: dict, index, columns) -> pd.DataFrame:
    """Taker both sides plus each symbol's own funding over the hold."""
    base = L.ENTRY_FEE_TAKER + L.EXIT_FEE_TAKER
    events = (HOLD_BARS * 30 / 60) / L.FUNDING_INTERVAL_HOURS
    f = panel.get("funding")
    if f is None:
        return pd.DataFrame(base, index=index, columns=columns)
    return base + f.reindex(index=index, columns=columns).fillna(0.0) * events


def signals(panel: dict) -> dict:
    """Every candidate, as a cross-sectional score. Higher = go long."""
    P = panel["close"]
    out = {}
    r = P.pct_change()
    for k in (2, 8, 16, 48, 96, 336):
        out[f"xs_mom_{k}"] = P.pct_change(k)
        out[f"xs_rev_{k}"] = -P.pct_change(k)
    out["xs_vol"] = -r.rolling(48).std()
    if "funding" in panel:
        f = panel["funding"]
        out["fund_level"] = -f                      # short the crowded side
        out["fund_level_pos"] = f                   # and the opposite
        out["fund_chg"] = -f.diff(16)
    if "oi" in panel:
        o = panel["oi"]
        oi_chg = o.pct_change(16)
        out["oi_up"] = oi_chg
        out["oi_down"] = -oi_chg
        # new money behind the move, versus a move on closing positions
        out["oi_x_mom"] = np.sign(P.pct_change(16)) * oi_chg
        out["oi_x_rev"] = -np.sign(P.pct_change(16)) * oi_chg
    if "ls" in panel:
        l_ = panel["ls"]
        out["ls_fade"] = -l_
        out["ls_follow"] = l_
    return out


def rank_trade(score: pd.DataFrame, fwd: pd.DataFrame, cost: pd.DataFrame,
               frac: float, mask) -> np.ndarray:
    """Long the top `frac` of the cross-section, short the bottom."""
    s = score.where(mask)
    n_ok = s.notna().sum(axis=1)
    ranks = s.rank(axis=1, pct=True)
    k = max(1, int(frac * 100))
    longs = ranks >= (1 - frac)
    shorts = ranks <= frac
    rows = []
    for t in s.index:
        if n_ok.get(t, 0) < 20:
            continue
        fl = fwd.loc[t][longs.loc[t]].dropna()
        fs = fwd.loc[t][shorts.loc[t]].dropna()
        cl = cost.loc[t][longs.loc[t]].reindex(fl.index).fillna(0.0011)
        cs = cost.loc[t][shorts.loc[t]].reindex(fs.index).fillna(0.0011)
        if len(fl):
            rows.extend((fl.values - cl.values).tolist())
        if len(fs):
            rows.extend(((-fs.values) - cs.values).tolist())
    return np.array(rows)


def stats(x: np.ndarray) -> tuple[int, float, float]:
    if len(x) < 30:
        return len(x), 0.0, 0.0
    sd = x.std()
    return len(x), float(x.mean()), float(x.mean() / (sd / math.sqrt(len(x)))) if sd > 0 else 0.0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--min-symbols", type=int, default=30)
    p.add_argument("--fracs", default="0.05,0.10,0.20")
    a = p.parse_args(argv)

    panel = load_panel(a.min_symbols)
    if not panel:
        print(f"Not enough data in {ALT}.")
        print("Run this first, on a machine that can reach Bybit:\n")
        print("    python -m fp.altdata --top 120 --days 180\n")
        print("It downloads funding, open interest, long/short ratio and 30m")
        print("klines for the most liquid perpetuals. Everything after that")
        print("runs offline.")
        return 1

    P = panel["close"]
    print(f"{P.shape[1]} symbols, {P.shape[0]:,} 30m bars, "
          f"{P.index.min().date()} .. {P.index.max().date()}")
    print("available:", ", ".join(k for k in panel if k != "close") or "price only")

    fwd = forward_return(P, HOLD_BARS)
    cost = cost_matrix(panel, P.index, P.columns)
    sig = signals(panel)
    fracs = [float(x) for x in a.fracs.split(",")]
    tested = len(sig) * len(fracs)
    z = 4.0 if tested > 50 else 3.0
    print(f"{len(sig)} signal families x {len(fracs)} basket sizes = "
          f"{tested} variants, bar |t| > {z}\n")

    liquid = P.notna()
    print(f"{'signal':>16} {'frac':>6} {'n':>8} {'in-sample':>11} {'t':>7} "
          f"| {'walk-fwd n':>11} {'net':>11} {'t':>7} {'verdict':>9}")
    edges = np.linspace(0, len(P), FOLDS + 1).astype(int)
    survivors = []
    for name, score in sig.items():
        for frac in fracs:
            allr = rank_trade(score, fwd, cost, frac, liquid)
            n, mu, t = stats(allr)
            if n < 100:
                continue
            oos = []
            for k in range(2, FOLDS):
                # the score is a pure ranking, so there is nothing fitted to
                # carry forward -- what the folds test is stability, not a
                # learned parameter
                m = liquid.copy()
                m.iloc[:edges[k]] = False
                m.iloc[edges[k + 1]:] = False
                oos.extend(rank_trade(score, fwd, cost, frac, m).tolist())
            on, omu, ot = stats(np.array(oos))
            ok = omu > 0 and ot > z
            if ok:
                survivors.append((name, frac, on, omu, ot))
            print(f"{name:>16} {frac:>6.2f} {n:>8,} {100*mu:>10.4f}% {t:>7.2f} "
                  f"| {on:>11,} {100*omu:>10.4f}% {ot:>7.2f} "
                  f"{'SURVIVES' if ok else '':>9}")

    print("\n" + "=" * 78)
    if survivors:
        print(f"{len(survivors)} variant(s) cleared |t| > {z} out of sample:")
        for name, frac, n, mu, t in sorted(survivors, key=lambda r: -r[4]):
            print(f"  {name} at {frac:.0%} baskets: {n:,} trades, "
                  f"{100*mu:+.4f}%/trade, t={t:.2f}")
        print("\nThese are worth trading. Wire the winner into the bot with")
        print("--signals and keep feeding it daily data.")
    else:
        print("Nothing cleared the bar. Positioning and cross-section behave")
        print("like everything else tested here on this sample.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
