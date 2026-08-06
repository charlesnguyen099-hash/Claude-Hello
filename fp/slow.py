"""Slow trend logic: few trades, held for weeks, leverage bounded by drag.

    python -m fp.slow                  # the measurement this is built on

WHAT THE DATA ACTUALLY SAID

BTCUSDT over 2025 and the first eight months of 2026 fell 31.6%, with a
53% maximum drawdown. Perfect directional knowledge was available in
hindsight: be short the whole way. Two ways of holding that same correct
short, over that same fall:

    held as ONE position, never rebalanced   +31.6% gross   +14.1% net
    rebalanced to constant notional daily     +8.1% gross    -9.2% net

The 23.5-point difference is volatility drag. Daily returns have a 2.28%
standard deviation, and a position rebalanced through that path compounds
sigma^2/2 against itself every period. Nothing about the forecast changed
between those two rows -- only how often the position was reset.

AND LEVERAGE MULTIPLIES DRAG BY THE SQUARE

Same perfectly-correct short, same period, net of funding:

     1x   -9.3%
     2x  -48.6%
     3x -102.1%
     5x -181.3%
    10x -274.6%   account gone

That is with the direction right every single day. The old bot opened and
closed every ~7 hours at 17-100x leverage, which is the worst available
combination of both effects, and it would have lost money on this data
holding a position that was correct throughout.

This is arithmetic, not statistics. It does not depend on whether a
signal predicts anything, and no amount of signal quality repairs it.

WHAT FOLLOWS FOR THE DESIGN

  hold for weeks     drag is paid per turnover, so turn over rarely
  leverage near 1x   drag scales with L^2 while return scales with L
  few trades         each round trip is a fixed 0.11% of notional
  slow direction     a signal that changes daily forces daily turnover

Measured on the same data with positions held from one signal change to
the next and never rebalanced inside a position:

    MA50 cross      38 trades, 14 days each   +26.1% at 1x
                                              (+20.5% 2025, +4.6% 2026)
    momentum 120d   10 trades, 46 days each   +15.4% at 1x

WHAT IS NOT ESTABLISHED, STATED PLAINLY

The signal does not survive walk-forward parameter selection: choosing
the best trend length on prior data every 60 days returns -8.5% over 35
out-of-sample trades, t = -0.12. Statistically that is zero, not a loss,
but it is not an edge either. The parameter surface is rough -- across 45
momentum lengths only 38% are profitable, median -10.8% -- so MA50 is a
lucky value rather than a robust one.

So the structure below is justified by the drag arithmetic, which is
certain. The direction rule is not justified by anything stronger than
"it is the least-bad of the slow options tested", and the leverage chain
is what keeps that honest: Kelly on a measured win rate sizes an unproven
signal at nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Bar size the direction is decided on. Daily, because a signal that can
# change every 30 minutes forces 48 turnovers a day and pays drag on all
# of them.
TREND_BAR_MINUTES = 1440
# Trend lookback, in those bars.
TREND_LOOKBACK = 50
# A position is held until the trend flips. Nothing inside it is
# rebalanced, resized or re-entered -- that is the entire point.
MIN_HOLD_BARS = 3

# Leverage. The old chain ran 17-100x, which the drag arithmetic above
# rules out completely. What survives is a band near 1x, and the ceiling
# is set where drag over a typical hold stays a small share of the move.
LEVERAGE_MIN, LEVERAGE_MAX = 1.0, 3.0

FUNDING_PER_DAY = 0.0003          # 0.010% per 8h, charged as a cost
ROUND_TRIP_FEE = 0.0011           # taker in, taker out


def drag_per_day(daily_vol: float, leverage: float) -> float:
    """Volatility drag: (L*sigma)^2 / 2 per period.

    Quadratic in leverage. At 2.28% daily vol this is 0.026%/day at 1x,
    0.104% at 2x and 0.234% at 3x -- 3x costs nine times what 1x does for
    three times the exposure.
    """
    return 0.5 * (leverage * daily_vol) ** 2


def expected_net(move_pct: float, hold_days: float, daily_vol: float,
                 leverage: float) -> float:
    """Net return on margin for a position held without rebalancing."""
    gross = move_pct * leverage
    costs = leverage * (ROUND_TRIP_FEE + hold_days * FUNDING_PER_DAY)
    return gross - costs - hold_days * drag_per_day(daily_vol, leverage)


def best_leverage(move_pct: float, hold_days: float, daily_vol: float,
                  cap: float = LEVERAGE_MAX) -> dict:
    """The leverage that maximises expected net -- the potential scaling,
    now with drag in it.

    Return grows linearly in leverage and drag grows quadratically, so
    there is a maximum rather than "more is better". A big expected move
    over a short hold justifies more; a thin move over a long hold
    justifies less, and often none at all.
    """
    if daily_vol <= 0 or not np.isfinite(daily_vol):
        return {"leverage": 0.0, "net": 0.0, "tradeable": False}
    grid = np.linspace(0.25, cap, 56)
    nets = np.array([expected_net(move_pct, hold_days, daily_vol, L)
                     for L in grid])
    i = int(np.argmax(nets))
    lev = float(grid[i]) if nets[i] > 0 else 0.0
    return {"leverage": max(LEVERAGE_MIN, lev) if lev > 0 else 0.0,
            "net": float(nets[i]), "tradeable": nets[i] > 0,
            "drag_per_day": drag_per_day(daily_vol, max(lev, LEVERAGE_MIN))}


def trend_direction(closes: np.ndarray, lookback: int = TREND_LOOKBACK) -> int:
    """+1, -1 or 0 from a slow moving average. Past-only."""
    if len(closes) < lookback + 1:
        return 0
    ma = float(np.mean(closes[-lookback:]))
    last = float(closes[-1])
    if not np.isfinite(ma) or ma <= 0:
        return 0
    return 1 if last > ma else -1


def daily_volatility(closes: np.ndarray, lookback: int = 30) -> float:
    if len(closes) < lookback + 2:
        return float("nan")
    r = np.diff(closes[-(lookback + 1):]) / closes[-(lookback + 1):-1]
    return float(np.std(r))


# ---------------------------------------------------------------- evidence

def _demo() -> int:
    D = Path(__file__).resolve().parent.parent / "bybit_bot" / "data"

    def rd(p):
        d = pd.read_csv(p, sep=None, engine="python")
        d.columns = [c.strip().lower() for c in d.columns]
        d["datetime"] = pd.to_datetime(
            d[[c for c in d.columns if "time" in c or "date" in c][0]])
        return d[["datetime", "open", "high", "low", "close", "volume"]]

    files = sorted(D.glob("BTCUSDT_*.csv"))
    if not files:
        print(f"no data in {D}")
        return 1
    raw = (pd.concat([rd(f) for f in files]).drop_duplicates("datetime")
           .sort_values("datetime").set_index("datetime"))
    px = raw["close"].resample("1D").last().dropna()
    r = px.pct_change().dropna()
    vol = float(r.std())
    days = len(px) - 1
    move = (px.iloc[0] - px.iloc[-1]) / px.iloc[0]

    print("=" * 74)
    print("THE MEASUREMENT THIS MODULE IS BUILT ON")
    print("=" * 74)
    print(f"  {px.index[0].date()} .. {px.index[-1].date()}  ({days} days)")
    print(f"  {px.iloc[0]:,.0f} -> {px.iloc[-1]:,.0f}   "
          f"{100*(px.iloc[-1]/px.iloc[0]-1):+.1f}%   daily vol {100*vol:.2f}%")
    print(f"\n  A perfectly correct short, held two different ways:")
    held = move - ROUND_TRIP_FEE - days * FUNDING_PER_DAY
    reb = float((1 - r).prod() - 1) - days * FUNDING_PER_DAY
    print(f"    one position, never rebalanced   {100*held:>+8.1f}%")
    print(f"    rebalanced daily                 {100*reb:>+8.1f}%")
    print(f"    volatility drag                  {100*(held-reb):>+8.1f}%")
    print(f"\n  The same correct short, by leverage:")
    for L in (1, 2, 3, 5, 10):
        g = (1 - L * r).clip(lower=-0.99)
        v = float(g.prod() - 1) - L * days * FUNDING_PER_DAY
        note = "   account gone" if float((1 - L * r).min()) <= 0 else ""
        print(f"    {L:>2}x  {100*v:>+9.1f}%{note}")

    print("\n" + "=" * 74)
    print("WHAT LEVERAGE THE CHAIN PICKS, given drag")
    print("=" * 74)
    print(f"{'expected move':>14} {'hold':>7} {'best lev':>9} {'net':>9} "
          f"{'drag/day':>10}")
    for mv in (0.02, 0.05, 0.10, 0.20, 0.40):
        for hd in (7, 30, 90):
            b = best_leverage(mv, hd, vol)
            print(f"{100*mv:>13.0f}% {hd:>6}d {b['leverage']:>8.2f}x "
                  f"{100*b['net']:>8.2f}% {100*b.get('drag_per_day',0):>9.3f}%"
                  + ("" if b["tradeable"] else "   not worth taking"))
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(_demo())
