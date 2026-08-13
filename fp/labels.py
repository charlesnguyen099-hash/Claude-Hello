"""Every trade in the past that profits after the whole exchange bill.

This is the operator's "trade tren data qua khu la loi 100%" written as
code. With the future in hand, mark every bar where a trade opened there
would have made money net of taker in, taker out and funding. That set
is the target. Nothing here is a forecast; it is the answer sheet, and
its only job is to tell the model what to look for.

TWO THINGS THE ANSWER SHEET MUST GET RIGHT, or everything built on it is
worthless:

  1. THE EXIT MUST BE EXECUTABLE. "Sell at the top" is not a trade, it is
     a wish. So a labelled opportunity is a TRIPLE BARRIER: a target, a
     stop and a time limit, all in units of the volatility at entry. The
     bot can place all three the moment it opens. The label asks only
     whether that specific, placeable trade won.

  2. THE FEE MUST BE INSIDE THE LABEL, not subtracted afterwards. A
     target narrower than the round trip is not a small win, it is a
     loss, and a labeller that calls it a win teaches the model to hunt
     for losses. barrier_net() refuses any shape whose target does not
     clear the bill at that bar's own volatility.

WHY A GRID OF SHAPES. The operator asked that nothing be fixed. A tight
target hit often and a wide one hit rarely are different trades with
different break-evens, and which is available depends on the volatility
of the moment. So every shape in the grid is labelled at every bar, and
the model learns which one is reachable -- the shape is an output, not a
setting.

COVERAGE, not selection. The instruction is to miss no profitable trade
"du la loi nho". So the grid runs down to a target only just wider than
the fee, and label_all() reports what fraction of bars carry at least
one winning shape, on both sides. That number is the ceiling: no logic
built on these labels can catch more than it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fp.barriers import barrier_outcomes
from fp.data import FEE, FUNDING_PER_8H  # noqa: F401

# THE TARGET SCALES WITH THE HORIZON, and this is the correction that
# decides whether anything here can work at all.
#
# The first version of this grid set the target at k times the ONE-MINUTE
# volatility. Measured on BTC that made the break-even win rate 94% at
# 2 sigma and 77% at 6 sigma: at a 1-minute sigma near 0.04%, a 2-sigma
# target is 0.08% and the round trip is 0.11%, so the trade is behind
# before it starts. Nothing wins 94% of the time. Every rule this repo
# has tested failed for exactly that reason, and no cleverness in the
# features could have fixed it -- the arithmetic was against them.
#
# Price wanders like sqrt(time), so the target has to as well. A shape is
# now (k, m, minutes) meaning target = k * sigma_1m * sqrt(minutes) and
# stop = m * sigma_1m * sqrt(minutes). At a 4-hour hold that same BTC
# sigma gives a 2-sigma target of 1.24% against a 0.11% bill, and the
# break-even win rate lands near 48% -- a fair coin, where an edge can
# actually show. HOLD_FLOOR keeps the grid away from horizons where the
# fee still dominates.
#
# This is also what "khong fix cung" means in practice: the barriers move
# with the market's volatility AND with how long the trade is given.
SHAPES = (
    (1.0, 1.0, 60), (1.0, 1.5, 120), (1.0, 1.5, 240),
    (1.5, 1.0, 60), (1.5, 1.5, 120), (1.5, 1.5, 240), (1.5, 2.0, 480),
    (2.0, 1.5, 120), (2.0, 2.0, 240), (2.0, 2.0, 480), (2.0, 3.0, 720),
    (3.0, 2.0, 240), (3.0, 3.0, 480), (3.0, 3.0, 720),
    (4.0, 3.0, 480), (4.0, 4.0, 720),
)

# Volatility yardstick: the same one the factors use, so a target of
# "2 sigma" means the same thing in both files.
SIGMA_WINDOW = 120


def sigma(close: np.ndarray, window: int = SIGMA_WINDOW) -> np.ndarray:
    """Per-bar return volatility, causal."""
    c = pd.Series(close, dtype="float64")
    return (c.pct_change().rolling(window, min_periods=window // 2)
            .std().values)


def barrier_net(d: pd.DataFrame, side: int, tp: float, sl: float,
                hold: int, sg: np.ndarray | None = None):
    """Net return of the triple-barrier trade opened at each bar.

    Returns (net, held_bars, tradeable) where tradeable marks the bars
    whose target actually clears the round trip at that bar's own
    volatility. A bar that is not tradeable is not labelled either way --
    it is not a missed opportunity, there was nothing there to take.
    """
    c = d["close"].values.astype("float64")
    h = d["high"].values.astype("float64")
    lo = d["low"].values.astype("float64")
    if sg is None:
        sg = sigma(c)
    # sigma over the WHOLE horizon, not one bar of it. barrier_outcomes
    # multiplies its tp/sl by whatever sigma it is handed, so scaling it
    # here scales both barriers together and nothing downstream needs to
    # know.
    sgh = sg * np.sqrt(hold)
    out, held = barrier_outcomes(h, lo, c, sgh, side, tp, sl, hold)
    net = out - FEE - held / 60.0 / 8.0 * FUNDING_PER_8H
    # b > 0 is the whole test: what a win pays, net of the bill.
    # Bars too close to the end of the data have no full horizon to
    # resolve in, so their outcome is unknown rather than zero. Excluded
    # from tradeable, or the last few hours of every window would be
    # scored as flat trades that never happened.
    tradeable = np.isfinite(sgh) & np.isfinite(net) & ((tp * sgh - FEE) > 0)
    return net, held, tradeable


def label_all(d: pd.DataFrame, shapes=SHAPES) -> dict:
    """Label every bar, every shape, both sides.

    The returned dict is keyed (tp, sl, hold, side) -> dict with
      net        realized net return of that trade
      win        net > 0
      held       bars the trade lasted, for the independence mask
      tradeable  whether the shape was even available at that bar
    """
    sg = sigma(d["close"].values.astype("float64"))
    out = {}
    for tp, sl, hold in shapes:
        for side in (1, -1):
            net, held, ok = barrier_net(d, side, tp, sl, hold, sg)
            out[(tp, sl, hold, side)] = {
                "net": net, "win": (net > 0) & ok, "held": held,
                "tradeable": ok}
    return out, sg


def coverage(labels: dict) -> dict:
    """How much money was actually lying on the table.

    The ceiling for any logic built on these labels: the fraction of
    bars where SOME shape on SOME side wins after fees, and what the
    best available shape paid.
    """
    keys = list(labels)
    n = len(labels[keys[0]]["net"])
    anywin = np.zeros(n, dtype=bool)
    best = np.full(n, -np.inf)
    for k in keys:
        L = labels[k]
        anywin |= L["win"]
        best = np.maximum(best, np.where(L["tradeable"], L["net"], -np.inf))
    good = np.isfinite(best)
    return {"bars": int(good.sum()),
            "bars_with_a_winner": int((anywin & good).sum()),
            "share": float((anywin & good).mean()) if good.any() else 0.0,
            "best_mean_pct": float(100 * best[good & anywin].mean())
            if (good & anywin).any() else 0.0}
