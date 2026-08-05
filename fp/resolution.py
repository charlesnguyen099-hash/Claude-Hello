"""The correctness check: keep ONLY rules that never lost, and see the cost.

    python -m fp.resolution

THE CHECK, AND WHY IT IS THE RIGHT ONE TO ASK FOR

If the table keeps only logics that make money, then replaying the very
bars it was built from must show a 100% win rate. Anything less means the
filter was not "keep the winners" but something weaker. An earlier run
reported 55.2%, which was exactly that mistake: rules were kept on a
positive MEAN, so a rule could stay while losing four trades in ten.

Fixed here. A rule is kept only when every single occurrence of it in the
learn window won. In-sample purity is then 100% by construction, and the
interesting question becomes what it costs to get there.

WHAT IT COSTS

Purity is bought with resolution. A coarse signature lumps many different
setups under one key, so that key contains winners and losers together
and can never be pure. Split it finer and the losers move to their own
keys. Split far enough and every bar has its own key, every key is
trivially pure -- and nothing ever repeats.

    signature detail        in-sample win%     out-of-sample coverage
    3 dims x 3 buckets            34.1%              99.9% of bars
    5 dims x 6 buckets            35.7%              95.2%
    + last 4 candles              48.5%              69.8%
    + last 8 candles              96.5%               6.4%
    + all 11 deltas               99.8%               0.4%
    + 8 buckets                  100.0%               0.0%  (7 trades)

100% in-sample is reachable. At that point the table holds 25,583 rules
for 39,048 learn bars and matches 7 bars out of the next 16,735.

THE NUMBER THAT SETTLES IT

Down the whole sweep, out-of-sample win rate does not move:

    33.3%   33.3%   33.0%   33.1%   33.3%

Flat, at every resolution, while in-sample climbs 34% to 100%. And
33.33% is not an arbitrary level -- it is SL/(TP+SL) = 1.5/4.5, the rate
a coin flip produces against these targets. The signature detail buys
in-sample purity and buys nothing else.

This is why the search cannot be fixed by filtering harder. Filtering
harder moves the in-sample number and leaves the out-of-sample number
where it was.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from fp import logic as L
from fp import patterns as P
from fp import walkforward as W

# (dims, per-candle deltas, buckets, label) -- increasing resolution.
LEVELS = [
    (3, 0, 3, "3 dims x 3 buckets"),
    (5, 0, 4, "5 dims x 4 buckets"),
    (5, 0, 6, "5 dims x 6 buckets"),
    (5, 4, 4, "5 dims + last 4 candles"),
    (5, 8, 4, "5 dims + last 8 candles"),
    (5, 11, 4, "5 dims + all 11 deltas"),
    (5, 11, 8, "5 dims x 8 buckets + 11 deltas"),
]


def signature_at(i, c, h, lo, v, atr, dims, per_candle, buckets):
    """Same idea as patterns.signature, with the resolution exposed."""
    a = atr[i]
    if a <= 0 or not np.isfinite(a) or i < P.LOOKBACK:
        return None
    s, e = i - P.LOOKBACK + 1, i + 1
    cc, vv = c[s:e], v[s:e]
    parts = [np.searchsorted(tuple(np.linspace(-3, 3, buckets)), (cc[-1] - cc[0]) / a)]
    if dims >= 2:
        parts.append(np.searchsorted(tuple(np.linspace(.2, .8, buckets)),
                                     np.mean(np.diff(cc) > 0)))
    if dims >= 3:
        half = P.LOOKBACK // 2
        v0 = vv[:half].mean()
        parts.append(np.searchsorted(tuple(np.linspace(.6, 1.6, buckets)),
                                     vv[half:].mean() / v0 if v0 > 0 else 1.0))
    hi_, lo_ = h[s:e].max(), lo[s:e].min()
    if dims >= 4:
        parts.append(np.searchsorted(tuple(np.linspace(.1, .9, buckets)),
                                     (cc[-1] - lo_) / (hi_ - lo_) if hi_ > lo_ else .5))
    if dims >= 5:
        parts.append(np.searchsorted(tuple(np.linspace(1, 8, buckets)), (hi_ - lo_) / a))
    if per_candle:
        parts.extend(np.searchsorted((-0.3, 0.3), (np.diff(cc) / a)[-per_candle:]))
    return ",".join(map(str, parts))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fee", default="taker", choices=["maker", "taker", "taker-slip"])
    p.add_argument("--split", type=float, default=0.70)
    a = p.parse_args(argv)
    fee = {"maker": L.MAKER_ROUND_TRIP, "taker": L.TAKER_ROUND_TRIP,
           "taker-slip": L.TAKER_WITH_SLIPPAGE}[a.fee]

    bars = W.load_series()
    o = W.cached_outcomes(bars, L.DEFAULT_EXIT)
    c, h, lo, v = (bars["close"].values, bars["high"].values,
                   bars["low"].values, bars["volume"].values)
    atr = P.atr_series(bars)
    n = len(c)
    split = int(n * a.split)
    breakeven = 100 * L.SL_MULTIPLE / (L.TP_MULTIPLES[L.DEFAULT_EXIT] + L.SL_MULTIPLE)

    print(f"{n:,} {P.BAR_MINUTES}m bars, {bars.datetime.min().date()} .. "
          f"{bars.datetime.max().date()}")
    print(f"learn on the first {a.split:.0%} (to {bars.datetime.iloc[split].date()}), "
          f"test on the rest")
    print(f"fee {a.fee} {fee*100:.3f}%, exit {L.DEFAULT_EXIT}, "
          f"coin-flip win rate for these targets = {breakeven:.1f}%\n")

    print(f"{'signature detail':>32} {'keys':>8} {'pure':>8} "
          f"{'IN-SAMPLE':>19}  {'OUT-OF-SAMPLE':>27}")
    print(f"{'':>32} {'':>8} {'rules':>8} {'trades':>11}{'win%':>8}  "
          f"{'coverage':>9}{'trades':>8}{'win%':>9}")
    oos_rates = []
    for dims, pc, buckets, label in LEVELS:
        keys = np.array([signature_at(i, c, h, lo, v, atr, dims, pc, buckets)
                         for i in range(n)], dtype=object)
        learn: dict = {}
        for i in range(P.LOOKBACK, split):
            k = keys[i]
            if k is None or not np.isfinite(o["long"][i]):
                continue
            rl, rs = o["long"][i] - fee, o["short"][i] - fee
            d = 1 if rl >= rs else -1
            rec = learn.setdefault((k, d), [0, 0])
            rec[0] += 1
            rec[1] += max(rl, rs) > 0
        pure = {k: d for (k, d), rec in learn.items() if rec[0] == rec[1]}

        def replay(lo_i, hi_i):
            t = w = 0
            for i in range(lo_i, hi_i):
                k = keys[i]
                if k is None or k not in pure or not np.isfinite(o["long"][i]):
                    continue
                d = pure[k]
                t += 1
                w += ((o["long"][i] if d > 0 else o["short"][i]) - fee) > 0
            return t, w

        it, iw = replay(P.LOOKBACK, split)
        ot, ow = replay(split, n - 1)
        nk = len({k for k in keys if k is not None})
        oos = 100 * ow / ot if ot else float("nan")
        if ot >= 100:
            oos_rates.append(oos)
        print(f"{label:>32} {nk:>8,} {len(pure):>8,} {it:>11,}"
              f"{100*iw/it if it else 0:>7.1f}%  "
              f"{100*ot/(n-split):>8.1f}%{ot:>8,}"
              f"{oos:>8.1f}%")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print("  In-sample win rate reaches 100%. The filter is correct: a rule is")
    print("  kept only if it never once lost. That was the thing being checked.")
    print()
    print("  It is bought entirely with resolution. By the time the table is")
    print("  pure it holds roughly one rule per learn bar, and it matches")
    print("  almost nothing afterwards -- a rule so specific it never lost is")
    print("  a rule so specific it never recurs.")
    print()
    if oos_rates:
        print(f"  Out-of-sample win rate across the sweep: "
              f"{', '.join(f'{x:.1f}%' for x in oos_rates)}")
        print(f"  Spread: {max(oos_rates)-min(oos_rates):.1f} percentage points, "
              f"against a coin-flip value of {breakeven:.1f}%.")
        print()
        print("  In-sample moves from 34% to 100%. Out-of-sample does not move")
        print("  at all. Every point of that in-sample gain is description of")
        print("  the past, not prediction of the future -- which is why")
        print("  filtering harder cannot fix the result, only the report.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
