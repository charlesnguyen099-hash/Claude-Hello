"""The exit has never been tested. It is the largest unexplored lever left.

    python -m fp.exits                     # entries x exits, on BTCUSDT
    python -m fp.exits --tf 1h --max-hold 96

WHAT HAS BEEN HELD FIXED THIS WHOLE TIME

Across 34,000 logics -- time bars, event bars, seven timeframes, four
bar constructors -- every single one used the same exit: hold until the
signal flips. Nothing else was ever tried.

That is not a neutral choice. Holding to the flip means giving back the
entire move every time the signal turns slowly, which is most of the
time, because the same smoothing that makes an entry reliable makes the
exit late. A logic with a good entry and a late exit measures as a bad
logic, and would have been discarded as one in every study so far.

So this separates them. The entry stays exactly what it was; the exit
becomes a variable.

THE EXITS, ALL SCALED BY THE MARKET'S OWN VOLATILITY

Barriers are never a fixed percentage. Each is a multiple of the
trailing volatility measured AT THE MOMENT OF ENTRY, so the same rule
takes a wide target in a violent market and a tight one in a calm one
without anybody adjusting it:

    take profit    tp x sigma above entry, in the trade's direction
    stop loss      sl x sigma against it
    time limit     if neither is touched within h bars, close at market

Which is also the answer to sizing a target by potential: a setup in a
market moving 3% a day gets a target three times the one in a market
moving 1%, from the same rule.

Touch is detected on the bar's HIGH and LOW, not its close, because a
target sitting inside the bar's range was reached whether or not the
bar closed there. Where both barriers fall inside the same bar the loss
is assumed -- the pessimistic assignment, since the path inside a bar
is unknown and assuming the win would manufacture edge.

LONG AND SHORT ARE SCORED SEPARATELY

Every logic so far was symmetric by construction: the same rule with the
sign flipped. But crypto is not symmetric -- falls are faster and deeper
than rises -- so a rule can be genuinely good in one direction and
useless in the other, and averaging the two hides both. Each side gets
its own line.

THE GATES ARE UNCHANGED

    potential   lower 95% bound of the in-sample net per trade > 0,
                after 0.055% each way and funding on real elapsed hours
    Bonferroni  out-of-sample t against a threshold for the number of
                entry x exit combinations tested, which is large
    honesty     the exit grid is searched on the first half only; the
                second half is never consulted while choosing
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from fp.ensemble import build_logics
from fp.horizon import (FEE_ROUND_TRIP, FUNDING_PER_8H, TIMEFRAMES, load_1m,
                        resample)
from fp.survivors import POTENTIAL_Z, bonferroni_t, tstat

TP_MULTS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
SL_MULTS = (0.5, 1.0, 1.5, 2.0, 3.0)
HOLDS = (6, 24, 96)


def sigma_at(close: np.ndarray, window: int = 50) -> np.ndarray:
    """Trailing volatility per bar, past-only, in return units."""
    r = pd.Series(close).pct_change()
    s = r.rolling(window, min_periods=window // 2).std().shift(1)
    return s.bfill().fillna(0.0).values


def barrier_outcomes(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                     sigma: np.ndarray, direction: int, tp: float, sl: float,
                     max_h: int) -> tuple[np.ndarray, np.ndarray]:
    """Return and bars-held for a trade opened at every bar, vectorised.

    Walks forward one bar at a time keeping a mask of trades still open,
    so the FIRST touch wins -- which is the whole point of a barrier and
    the thing a close-only test gets wrong.
    """
    n = len(close)
    entry = close
    up = entry * (1.0 + direction * tp * sigma) if direction > 0 else \
        entry * (1.0 - tp * sigma)
    dn = entry * (1.0 - sl * sigma) if direction > 0 else \
        entry * (1.0 + sl * sigma)

    out = np.full(n, np.nan)
    held = np.zeros(n, dtype=int)
    live = np.ones(n, dtype=bool)
    live[np.maximum(n - max_h - 1, 0):] = False        # cannot complete
    live &= sigma > 0

    for k in range(1, max_h + 1):
        idx = np.flatnonzero(live)
        if len(idx) == 0:
            break
        j = idx + k
        hi, lo = high[j], low[j]
        if direction > 0:
            hit_sl = lo <= dn[idx]           # pessimistic: stop checked first
            hit_tp = hi >= up[idx]
        else:
            hit_sl = hi >= dn[idx]
            hit_tp = lo <= up[idx]
        done_sl = hit_sl
        done_tp = hit_tp & ~hit_sl
        for mask, level in ((done_sl, dn), (done_tp, up)):
            w = idx[mask]
            if len(w):
                out[w] = direction * (level[w] - entry[w]) / entry[w]
                held[w] = k
                live[w] = False

    # never touched: close at the time limit
    rest = np.flatnonzero(live)
    if len(rest):
        j = np.minimum(rest + max_h, n - 1)
        out[rest] = direction * (close[j] - entry[rest]) / entry[rest]
        held[rest] = max_h
    return out, held


def evaluate(d: pd.DataFrame, minutes: int, max_hold: int, min_trades: int
             ) -> pd.DataFrame:
    """Every entry logic crossed with every exit, long and short apart."""
    close = d["close"].values.astype(float)
    high = d["high"].values.astype(float)
    low = d["low"].values.astype(float)
    sigma = sigma_at(close)
    n = len(close)
    split = n // 2
    fund_per_bar = FUNDING_PER_8H * (minutes / 480.0)

    logics = build_logics(d, fast=n > 30000)
    print(f"  {len(logics):,} entry logics x "
          f"{len(TP_MULTS) * len(SL_MULTS) * len(HOLDS)} exits x 2 sides")

    # entry bars per logic per side: where the position turns ON in that
    # direction, which is the moment a real system would open
    entries: dict[tuple[str, int], np.ndarray] = {}
    for name, pos in logics.items():
        p = pos.values.astype(float)
        prev = np.concatenate([[0.0], p[:-1]])
        for side in (1, -1):
            e = np.flatnonzero((p == side) & (prev != side))
            if len(e) >= 2 * min_trades:
                entries[(name, side)] = e

    rows = []
    for hmax in HOLDS:
        if hmax > max_hold:
            continue
        for tp in TP_MULTS:
            for sl in SL_MULTS:
                for side in (1, -1):
                    o, held = barrier_outcomes(high, low, close, sigma,
                                               side, tp, sl, hmax)
                    net = o - FEE_ROUND_TRIP - held * fund_per_bar
                    for (name, s2), e in entries.items():
                        if s2 != side:
                            continue
                        v = net[e]
                        ok = np.isfinite(v)
                        e2, v = e[ok], v[ok]
                        a, b = v[e2 < split], v[e2 >= split]
                        if len(a) < min_trades or len(b) < min_trades:
                            continue
                        lo_b = (a.mean() - POTENTIAL_Z * a.std(ddof=1)
                                / np.sqrt(len(a)))
                        if not (lo_b > 0):
                            continue
                        rows.append({
                            "logic": name, "side": "long" if side > 0 else "short",
                            "tp": tp, "sl": sl, "hmax": hmax,
                            "is_n": len(a), "is_mean": float(a.mean()),
                            "is_t": tstat(a),
                            "oos_n": len(b), "oos_mean": float(b.mean()),
                            "oos_t": tstat(b),
                            "hold": float(np.median(held[e2])) * minutes,
                        })
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tf", default="1h,4h")
    ap.add_argument("--max-hold", type=int, default=96)
    ap.add_argument("--min-trades", type=int, default=25)
    ap.add_argument("--max-bars", type=int, default=60000)
    a = ap.parse_args(argv)

    d1m = load_1m()
    print(f"{len(d1m):,} one-minute bars of BTCUSDT")
    print("the entry is unchanged; the EXIT is the variable\n")

    for label in a.tf.split(","):
        label = label.strip()
        if label not in TIMEFRAMES:
            continue
        d = resample(d1m, TIMEFRAMES[label])
        if len(d) > a.max_bars:
            d = d.iloc[-a.max_bars:]
        print(f"{label}: {len(d):,} bars")
        R = evaluate(d, TIMEFRAMES[label], a.max_hold, a.min_trades)
        if R.empty:
            print("  no entry x exit combination has a positive in-sample "
                  "edge after fees\n")
            continue
        thr = bonferroni_t(len(R))
        surv = R[R["oos_t"] > thr]
        print(f"  {len(R):,} combinations cleared the potential gate, "
              f"Bonferroni bar |t| > {thr:.2f}")

        # what the exit grid itself says, before any significance test
        print(f"\n  {'exit':>22} {'combos':>7} {'IS mean':>9} {'OOS mean':>9} "
              f"{'OOS t':>7}")
        g = R.groupby(["tp", "sl", "hmax"]).agg(
            combos=("oos_t", "size"), ismean=("is_mean", "mean"),
            oosmean=("oos_mean", "mean"), oost=("oos_t", "mean"))
        for (tp, sl, hm), r in g.sort_values("oosmean", ascending=False).head(8).iterrows():
            print(f"  {f'tp{tp}/sl{sl}/{hm}b':>22} {int(r.combos):>7} "
                  f"{100*r.ismean:>8.3f}% {100*r.oosmean:>8.3f}% {r.oost:>7.2f}")

        print(f"\n  {'side':>6} {'combos':>7} {'IS mean':>9} {'OOS mean':>9}")
        for side, r in R.groupby("side").agg(
                n=("oos_t", "size"), ism=("is_mean", "mean"),
                oosm=("oos_mean", "mean")).iterrows():
            print(f"  {side:>6} {int(r.n):>7} {100*r.ism:>8.3f}% "
                  f"{100*r.oosm:>8.3f}%")

        print(f"\n  survivors: {len(surv)}")
        for _, r in surv.sort_values("oos_t", ascending=False).head(15).iterrows():
            print(f"    {r.logic:<28} {r.side:>5} tp{r.tp}/sl{r.sl}/{r.hmax}b "
                  f"OOS t={r.oos_t:.2f} {100*r.oos_mean:+.3f}%/trade "
                  f"n={r.oos_n} hold={r.hold:.0f}m")
        if surv.empty:
            best = R.loc[R["oos_t"].idxmax()]
            print(f"    none. best out of sample: {best.logic} {best.side} "
                  f"tp{best.tp}/sl{best.sl}/{best.hmax}b at t={best.oos_t:.2f}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
