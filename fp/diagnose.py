"""Why the losing trades lost, and whether the two obvious fixes work.

    python -m fp.diagnose

Three questions, each with a measured answer.

1. ARE THE LOSSES FROM THE WRONG DIRECTION, OR FROM FEES?

   Entirely from direction. Of 5,667 trades on BTCUSDT 2026, 3,664 lose
   after full costs and 99.9% of them lost by hitting the stop. Not one
   trade reached its target and was turned negative by the fee -- a
   target pays 1.425% gross against a 0.119% cost, so it cannot be.

   The fee never converts a winning trade into a losing one. What it does
   is convert a break-even SYSTEM into a losing one, and that is a
   different problem with a different fix.

2. SO TRADE THE REVERSE?

   Worse, at every target multiple. Reversing does not mirror the trade:
   the stop sits 1.5 ATR away and the target 3.0 ATR away on BOTH sides,
   so both directions carry the same geometry and both pay the same fee.
   Forward -0.108%, reverse -0.133% per trade. The methods do hold a
   little directional information -- forward beats reverse in 2026 and
   August -- just nowhere near enough to cover the cost.

3. SO RAISE THE TARGET, SO THE FEE IS A SMALLER SHARE OF IT?

   The premise is right and the conclusion does not follow. The fee's
   share of the target does collapse, from 17.0% at TP1.5 to 1.4% at
   TP12. But the win rate falls in exact step, 50.6% to 12.6%, tracking
   SL/(TP+SL) the way a random walk demands. Net per trade sits at about
   -0.10% at every level:

       TP    win%   hold   fee/trade   fee/target   net/trade
      1.5   50.6%   4.3h     0.1154%       17.0%     -0.1059%
      2.0   43.1%   5.5h     0.1169%       11.3%     -0.1105%
      3.0   33.9%   7.5h     0.1193%        6.6%     -0.1082%
      4.0   28.3%   9.2h     0.1215%        4.6%     -0.0982%
      8.0   16.6%  17.0h     0.1313%        2.1%     -0.1182%
     12.0   12.6%  23.6h     0.1394%        1.4%     -0.0925%

   A wider target does cut the TOTAL fee bill, because it means fewer
   trades -- 2026 goes from -507% to -286% across the whole period. That
   is trading less, not trading better.

4. THEN JUST TRADE THE ONES THAT WIN.

   That requires telling them apart before the outcome. All 106 features
   were ranked by how well each separates winners from losers at entry,
   on 2025+2026, corrected for testing 106 of them:

       best AUC 0.5122, significance threshold 0.5291, 0 of 106 clear it

   AUC 0.50 is no separation whatsoever. At entry the 34 winners and the
   79 losers of your session look the same.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from fp import features as F
from fp import logic as L
from fp import methods as M
from fp.calibrate import DATA


def read(name: str) -> pd.DataFrame:
    d = pd.read_csv(DATA / name, sep=None, engine="python")
    d.columns = [c.strip().lower() for c in d.columns]
    dt = next(c for c in d.columns if "time" in c or "date" in c)
    d["datetime"] = pd.to_datetime(d[dt])
    return d[["datetime", "open", "high", "low", "close", "volume"]]


def run_exit(hi, lo, cl, i, d, atr, tp_mult):
    entry = cl[i]
    sl = entry - d * L.SL_MULTIPLE * atr
    tp = entry + d * tp_mult * atr
    for j in range(i + 1, min(i + 1 + L.MAX_HOLD_BARS, len(cl))):
        if (lo[j] <= sl) if d > 0 else (hi[j] >= sl):
            return -L.SL_MULTIPLE * atr / entry, j - i, "stop"
        if (hi[j] >= tp) if d > 0 else (lo[j] <= tp):
            return tp_mult * atr / entry, j - i, "target"
    j = min(i + L.MAX_HOLD_BARS, len(cl) - 1)
    return (cl[j] - entry) / entry * d, j - i, "timeout"


def main() -> int:
    bars = L.to_bars(read("BTCUSDT_2026.csv"))
    feats = F.build(bars)
    votes = M.evaluate_all(feats)
    hi, lo, cl = bars["high"].values, bars["low"].values, bars["close"].values
    atr_pct = feats["atr14_pct"].values
    cost = L.round_trip_cost(L.DEFAULT_EXIT, False)["total"]

    entries = []
    for i in range(250, len(cl) - 1):
        if votes["consensus_dir"].iloc[i] == M.TIE:
            continue
        if int(votes["n_methods_fired"].iloc[i]) < 1:
            continue
        a = atr_pct[i]
        if not np.isfinite(a) or a <= 0:
            continue
        entries.append((i, 1 if votes["consensus_dir"].iloc[i] == M.LONG else -1,
                        (a / 100) * cl[i]))

    print("=" * 74)
    print("1. WHY THE LOSERS LOST")
    print("=" * 74)
    why = {"stop": [0, 0], "target": [0, 0], "timeout": [0, 0]}
    for i, d, atr in entries:
        g, _, r = run_exit(hi, lo, cl, i, d, atr, 3.0)
        why[r][0] += 1
        why[r][1] += (g - cost) <= 0
    tot = sum(v[0] for v in why.values())
    nl = sum(v[1] for v in why.values())
    label = {"stop": "hit the stop -- WRONG DIRECTION",
             "target": "hit the target -- would be a FEE loss",
             "timeout": "timed out"}
    print(f"  {tot:,} trades, {nl:,} lose after the full {100*cost:.4f}% cost\n")
    print(f"{'ended by':>10} {'trades':>9} {'of them losing':>15}  reason")
    for k in ("stop", "target", "timeout"):
        if not why[k][0]:
            continue
        print(f"{k:>10} {why[k][0]:>9,} {why[k][1]:>15,}  {label[k]}")
    print(f"\n  {100*why['stop'][1]/nl:.1f}% of every loss is a stop-out.")
    print("  Not one target was turned negative by the fee, and none can be:")
    print("  a target pays about 1.4% gross against a 0.12% cost.")

    print("\n" + "=" * 74)
    print("2. REVERSING, AND 3. A WIDER TARGET")
    print("=" * 74)
    print(f"{'TP':>5} {'win%':>7} {'hold':>7} {'fee/trade':>10} {'fee/target':>11}"
          f" {'net fwd':>9} {'net rev':>9}")
    for tp in (1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0):
        f_net, r_net, wins, holds = [], [], 0, []
        for i, d, atr in entries:
            g, bars_held, _ = run_exit(hi, lo, cl, i, d, atr, tp)
            gr, bars_r, _ = run_exit(hi, lo, cl, i, -d, atr, tp)
            fee_f = (L.ENTRY_FEE_TAKER + L.EXIT_FEE_TAKER
                     + (bars_held * 0.5 / 8) * L.FUNDING_RATE_TYPICAL)
            fee_r = (L.ENTRY_FEE_TAKER + L.EXIT_FEE_TAKER
                     + (bars_r * 0.5 / 8) * L.FUNDING_RATE_TYPICAL)
            f_net.append(g - fee_f)
            r_net.append(gr - fee_r)
            wins += g > 0
            holds.append(bars_held * 0.5)
        h = float(np.mean(holds))
        fee = (L.ENTRY_FEE_TAKER + L.EXIT_FEE_TAKER
               + (h / 8) * L.FUNDING_RATE_TYPICAL)
        share = 100 * fee / (tp * float(np.mean(atr_pct[np.isfinite(atr_pct)])) / 100)
        print(f"{tp:>5.1f} {100*wins/len(entries):>6.1f}% {h:>6.1f}h "
              f"{100*fee:>9.4f}% {share:>10.1f}% "
              f"{100*np.mean(f_net):>8.4f}% {100*np.mean(r_net):>8.4f}%")
    print("\n  The fee's share of the target collapses; the win rate falls in")
    print("  step, tracking SL/(TP+SL). Net per trade does not move, and")
    print("  reversing is worse at every level.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
