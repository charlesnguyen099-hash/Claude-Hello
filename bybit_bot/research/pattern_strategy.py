"""Turn the pattern-matching signal into actual trades and see if it pays.

research/pattern_matching.py found something real: the shape of the 40
candles before a move carries genuine, cross-validated information about
what happens next. Out-of-sample lift ran from 1.2x to 7.9x, and the two
directions (2025->2026 and 2026->2025) agreed closely, which is what
rules out a fluke.

Lift is not profit, though. A lift of 4.8x on a 0.11% base rate means
0.54% of matched setups reach the target -- the other 99.46% do not, and
those still pay fees and can hit a stop. Whether the edge survives is a
question about expected value, not about lift, so this script places the
trades:

  - build the pattern library from ONE year's profitable setups
  - walk the OTHER year bar by bar, entering when the current shape is
    close enough to something in the library
  - real stop loss, real target, max holding time, fees and slippage on
    every trade, one position at a time
  - resolve stop-vs-target ambiguity inside a bar in favour of the stop

Both directions are run (2025 library -> 2026 trades, and the reverse),
because a rule that only works one way round is a coincidence.

Run:  python3 -m research.pattern_strategy
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bot import risk
from research.pattern_matching import build_patterns, label, load

MAX_LEVERAGE = risk.ABSOLUTE_MAX_LEVERAGE
FEE_ROUNDTRIP = 2 * (risk.TAKER_FEE_PCT + risk.ASSUMED_SLIPPAGE_PCT)

DATASETS = {"2026": "data/BTCUSDT_2026.csv", "2025": "data/BTCUSDT_2025.csv"}

WINDOW = 40
CONFIGS = [
    # (horizon minutes, leveraged target %, stop as fraction of target move)
    (15, 10.0, 1.0),
    (30, 10.0, 1.0),
    (60, 10.0, 1.0),
    (60, 20.0, 1.0),
    (240, 10.0, 1.0),
    (30, 10.0, 0.5),   # tighter stop: cut losers faster
    (60, 20.0, 0.5),
]
# Trade only the closest X% of bars, so "similar enough" is a controlled
# knob rather than an arbitrary distance.
SELECTIVITY = [0.5, 2.0, 5.0]
MAX_LIBRARY = 600
RNG = np.random.default_rng(0)


def nearest_distance(library: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Distance from every query pattern to its closest library pattern."""
    lib_sq = (library ** 2).sum(axis=1)[None, :]
    out = np.empty(len(queries), dtype=np.float32)
    CH = 20_000
    for s in range(0, len(queries), CH):
        chunk = queries[s:s + CH]
        d = ((chunk ** 2).sum(axis=1)[:, None] + lib_sq
             - 2.0 * chunk @ library.T)
        out[s:s + CH] = np.sqrt(np.maximum(d.min(axis=1), 0.0))
    return out


def simulate(df: pd.DataFrame, bars: np.ndarray, enter: np.ndarray, side: str,
             horizon: int, target_spot: float, stop_frac: float) -> dict:
    """Walk the signals in time order, one position at a time, with a real
    stop, a real target and a hard time exit.
    """
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    close = df["close"].to_numpy(float)
    n = len(close)

    stop_spot = target_spot * stop_frac
    long = side == "long"

    pnl = []
    busy_until = -1
    for i in bars[enter]:
        if i <= busy_until or i + horizon >= n:
            continue
        entry = close[i]
        if long:
            tp, sl = entry * (1 + target_spot), entry * (1 - stop_spot)
        else:
            tp, sl = entry * (1 - target_spot), entry * (1 + stop_spot)

        outcome = None
        for j in range(i + 1, min(i + 1 + horizon, n)):
            hit_sl = (low[j] <= sl) if long else (high[j] >= sl)
            hit_tp = (high[j] >= tp) if long else (low[j] <= tp)
            if hit_sl:            # conservative: stop first on an ambiguous bar
                outcome = -stop_spot
                break
            if hit_tp:
                outcome = target_spot
                break
        if outcome is None:       # time exit at whatever the price is
            j = min(i + horizon, n - 1)
            outcome = (close[j] - entry) / entry if long else (entry - close[j]) / entry
        pnl.append(outcome - FEE_ROUNDTRIP)
        busy_until = j

    if not pnl:
        return {"trades": 0}
    arr = np.array(pnl)
    wins = arr[arr > 0]
    losses = arr[arr <= 0]
    return {
        "trades": len(arr),
        "win_rate_pct": round(100 * float((arr > 0).mean()), 2),
        "avg_net_pct": round(100 * float(arr.mean()), 4),
        "total_spot_pct": round(100 * float(arr.sum()), 2),
        "total_leveraged_pct": round(100 * float(arr.sum()) * MAX_LEVERAGE, 1),
        "profit_factor": (round(float(wins.sum() / -losses.sum()), 3)
                          if len(losses) and losses.sum() < 0 else float("inf")),
    }


def main() -> None:
    print(f"Pattern window {WINDOW} candles | max leverage {MAX_LEVERAGE}x | "
          f"round trip {FEE_ROUNDTRIP*100:.3f}%")
    print("Library built on one year, traded on the other. Both directions shown.\n")

    data = {y: load(p) for y, p in DATASETS.items()}
    pats = {y: build_patterns(df, WINDOW) for y, df in data.items()}

    header = (f"{'horiz':>5s} {'targ':>5s} {'stop':>5s} {'sel%':>5s} {'dir':>5s} "
              f"{'lib->test':>12s} | {'trades':>6s} {'win%':>6s} {'avgNet%':>8s} "
              f"{'PF':>6s} {'total@25x%':>11s}")
    print(header)
    print("-" * len(header))

    all_rows = []
    for horizon, target_lev, stop_frac in CONFIGS:
        target_spot = target_lev / 100.0 / MAX_LEVERAGE + FEE_ROUNDTRIP

        labels = {}
        for y, df in data.items():
            f, bars = pats[y]
            s, l = label(df, bars, horizon, target_lev)
            labels[y] = {"feats": f, "bars": bars, "short": s, "long": l}

        for direction in ("short", "long"):
            for lib_y in DATASETS:
                test_y = next(y for y in DATASETS if y != lib_y)
                opp = np.flatnonzero(labels[lib_y][direction])
                if len(opp) < 50:
                    continue
                if len(opp) > MAX_LIBRARY:
                    opp = RNG.choice(opp, MAX_LIBRARY, replace=False)
                library = labels[lib_y]["feats"][opp]

                test = labels[test_y]
                dist = nearest_distance(library, test["feats"])

                for sel in SELECTIVITY:
                    thresh = np.percentile(dist, sel)
                    enter = dist <= thresh
                    r = simulate(data[test_y], test["bars"], enter, direction,
                                 horizon, target_spot, stop_frac)
                    if r["trades"] == 0:
                        continue
                    all_rows.append((horizon, target_lev, stop_frac, sel,
                                     direction, lib_y, test_y, r))
                    print(f"{horizon:5d} {target_lev:5.0f} {stop_frac:5.1f} {sel:5.1f} "
                          f"{direction:>5s} {lib_y+'->'+test_y:>12s} | "
                          f"{r['trades']:6d} {r['win_rate_pct']:6.2f} "
                          f"{r['avg_net_pct']:8.4f} {r['profit_factor']:6.2f} "
                          f"{r['total_leveraged_pct']:11.1f}")

    print("\n" + "=" * 78)
    # A setup only counts if BOTH library directions made money -- one
    # year working alone is exactly what overfitting looks like.
    by_setup: dict[tuple, list] = {}
    for horizon, targ, stop_frac, sel, direction, lib_y, test_y, r in all_rows:
        by_setup.setdefault((horizon, targ, stop_frac, sel, direction), []).append(r)

    robust = [(k, v) for k, v in by_setup.items()
              if len(v) == 2 and all(x["avg_net_pct"] > 0 for x in v)]
    if robust:
        print("Setups profitable in BOTH cross-year directions:")
        for (horizon, targ, stop_frac, sel, direction), v in sorted(
                robust, key=lambda kv: -sum(x["avg_net_pct"] for x in kv[1])):
            tot = sum(x["total_leveraged_pct"] for x in v)
            print(f"  horizon={horizon:>3d}m target={targ:.0f}% stop={stop_frac:.1f}x "
                  f"sel={sel:.1f}% {direction:5s}: "
                  f"avg net {v[0]['avg_net_pct']:+.4f}% / {v[1]['avg_net_pct']:+.4f}% "
                  f"per trade, {v[0]['trades']+v[1]['trades']} trades, "
                  f"{tot:+.1f}% combined at {MAX_LEVERAGE}x")
    else:
        print("No setup was profitable in both cross-year directions.")
        best = sorted(all_rows, key=lambda r: -r[7]["avg_net_pct"])[:8]
        print("\nBest single runs (not confirmed in the other direction — treat as noise):")
        for horizon, targ, stop_frac, sel, direction, lib_y, test_y, r in best:
            print(f"  horizon={horizon:>3d}m target={targ:.0f}% stop={stop_frac:.1f}x "
                  f"sel={sel:.1f}% {direction:5s} {lib_y}->{test_y}: "
                  f"avg net {r['avg_net_pct']:+.4f}% over {r['trades']} trades")


if __name__ == "__main__":
    main()
