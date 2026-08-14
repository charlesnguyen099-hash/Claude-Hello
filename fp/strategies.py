"""Combine the methods into strategies, and price every one of them.

The operator's instruction: take the ways people trade futures, cross them
with the factors and with each other, and let the combination be the
logic. That is what happens here.

  SOLO        each of the 47 methods on its own
  CONFIRM     a trend method AND a flow/breakout method must agree
  VOTE        the majority of a whole family
  REGIME      a method allowed to act only in one volatility regime
  CROSSED     a single-coin method AND a cross-sectional one must agree

Several hundred strategies come out of that, and every one of them is a
STATE: +1, -1 or 0 at each bar.

NOTHING IS FIXED ABOUT THE TRADE ITSELF.

  entry price   whatever the market is when the state turns on
  exit price    whatever it is when the state turns off or flips
  duration      however long that takes -- four minutes or four days

There is no target multiple, no stop multiple and no time limit anywhere
in this file. The previous design chose an exit from a grid of sixteen
(target, stop, hold) shapes; that grid was the last hard-coded thing in
the system and it is gone. A trade lasts exactly as long as its reason
lasts.

WHAT IS MEASURED. A strategy's trades are its state runs. Entry at the
close where the run starts, exit at the close where it ends, minus taker
in, taker out and funding for the hours actually held. Runs do not
overlap by construction, so every trade is already independent -- the
correction that has bitten this repo three times cannot apply here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fp import methods as M
from fp.data import FEE, FUNDING_PER_8H

# A run shorter than this is noise in the state, not a trade -- and at
# one-minute bars a two-bar flicker pays the round trip twice for nothing.
MIN_RUN = 5


def _runs(state: np.ndarray):
    """Start index, end index and side of every non-zero run.

    A run is a trade: it begins when the state turns on and ends when it
    turns off or flips. Nothing about its length is decided here.
    """
    s = state
    n = len(s)
    if n == 0:
        return np.empty(0, "int64"), np.empty(0, "int64"), np.empty(0, "int8")
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = s[1:] != s[:-1]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], n] - 1
    side = s[starts]
    keep = side != 0
    return starts[keep], ends[keep], side[keep].astype("int8")


def trades(close: np.ndarray, state: np.ndarray, min_run: int = MIN_RUN):
    """Every trade a state produces, priced at the full exchange bill.

    Returns (entry_idx, exit_idx, side, net) with net as a fraction of
    notional after taker in, taker out and funding for the hours held.
    """
    st, en, sd = _runs(state)
    if len(st) == 0:
        return (np.empty(0, "int64"),) * 2 + (np.empty(0, "int8"),
                                              np.empty(0, "float64"))
    # THE PERSISTENCE RULE, and getting it wrong is a second free peek at
    # the future -- worse than the first.
    #
    # Discarding runs that turned out to be shorter than min_run SELECTS
    # ON THE OUTCOME: a run is only known to be short once it has ended,
    # and short runs are precisely the breakouts that failed. Filtering
    # them out keeps the winners and throws away the losers, using
    # information no live bot has. With that filter in place
    # solo:boll_break measured +0.47%/trade on training and every fold of
    # the search "beat its null" at p = 0.000.
    #
    # The honest version of the same idea is to WAIT: enter only after the
    # state has already persisted min_run bars. By then the persistence is
    # observed rather than assumed, and the entry price is the one you
    # actually get for waiting.
    delay = max(int(min_run) - 1, 0)
    entry_i = st + delay
    alive = entry_i <= en
    st, en, sd, entry_i = st[alive], en[alive], sd[alive], entry_i[alive]
    if len(st) == 0:
        return (np.empty(0, "int64"),) * 2 + (np.empty(0, "int8"),
                                              np.empty(0, "float64"))
    # THE ONE-BAR RULE, and it is worth more than every strategy in this
    # file put together.
    #
    # A run's last bar is only KNOWN to be the last once the next bar's
    # state comes out different -- and that is known at the next bar's
    # close, not this one. Exiting at close[en] therefore sells one bar
    # before the flip, using information the bot cannot have. Measured
    # with that bug in place, solo:rsi_trend read +0.2491%/trade over
    # 19,897 trades at t = +40.85. It is not an edge; it is a peek. On a
    # sixteen-minute average hold, one free bar of hindsight is about six
    # percent of the trade.
    #
    # So the exit is close[en + 1]: the first price available AFTER the
    # state is observed to have changed.
    ex = np.minimum(en + 1, len(close) - 1)
    entry = close[entry_i]
    exit_ = close[ex]
    gross = sd * (exit_ / entry - 1.0)
    hours = (ex - entry_i) / 60.0
    net = gross - FEE - hours / 8.0 * FUNDING_PER_8H
    return entry_i, ex, sd, net


# ------------------------------------------------------------ combining

def _vote(states: pd.DataFrame, names, need=0.5):
    v = states[list(names)].astype("float32").mean(axis=1)
    return np.sign(np.where(v.abs() >= need, v, 0.0)).astype("int8")


def build(states: pd.DataFrame) -> dict[str, np.ndarray]:
    """Every strategy, as a signed state array. The whole search space."""
    out: dict[str, np.ndarray] = {}
    S = {c: states[c].values.astype("int8") for c in states.columns}

    # 1. SOLO -- each method alone.
    for name, v in S.items():
        out[f"solo:{name}"] = v

    # 2. CONFIRM -- a trend must be confirmed by flow or by a breakout.
    #    Both have to agree on the SAME side or the state is 0. This is
    #    the classic "signal plus filter" and it is where most published
    #    futures systems live.
    confirmers = M.FAMILY["flow"] + M.FAMILY["breakout"]
    for a in M.FAMILY["trend"]:
        for b in confirmers:
            va, vb = S[a], S[b]
            out[f"and:{a}+{b}"] = np.where(va == vb, va, 0).astype("int8")

    # 3. REVERT + CONFIRM -- fade a stretch, but only with flow agreeing.
    for a in M.FAMILY["revert"]:
        for b in M.FAMILY["flow"]:
            va, vb = S[a], S[b]
            out[f"and:{a}+{b}"] = np.where(va == vb, va, 0).astype("int8")

    # 4. VOTE -- the majority of a family. One method is an opinion; a
    #    family agreeing is a regime.
    for fam, names in M.FAMILY.items():
        have = [n for n in names if n in S]
        if len(have) >= 3:
            for need in (0.3, 0.5, 0.7):
                out[f"vote:{fam}@{need}"] = _vote(states, have, need)

    # 5. REGIME -- a method allowed to act in one volatility state only.
    #    Nothing about the regime threshold is per-method: it is the
    #    median of the coin's own recent volatility, so it moves with the
    #    market rather than being a number chosen here.
    return out


def with_regime(states: pd.DataFrame, base: dict[str, np.ndarray],
                d: pd.DataFrame, top=60) -> dict[str, np.ndarray]:
    """The best `top` strategies again, each split by volatility regime.

    Split, not filtered: a method that only works when the market is busy
    and one that only works when it is quiet are two different logics, and
    lumping them together averages one against the other.
    """
    r = d["close"].pct_change()
    fast = r.rolling(30, min_periods=15).std()
    slow = r.rolling(480, min_periods=120).std()
    busy = (fast > slow).values
    out = {}
    for name in list(base)[:top]:
        v = base[name]
        out[f"{name}|busy"] = np.where(busy, v, 0).astype("int8")
        out[f"{name}|quiet"] = np.where(~busy, v, 0).astype("int8")
    return out


def cross_confirm(states: pd.DataFrame,
                  base: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Single-coin conviction AND the board agreeing."""
    out = {}
    S = {c: states[c].values.astype("int8") for c in states.columns}
    for a in M.FAMILY["trend"] + M.FAMILY["revert"]:
        for b in M.FAMILY["cross"]:
            va, vb = S[a], S[b]
            out[f"xs:{a}+{b}"] = np.where(va == vb, va, 0).astype("int8")
    return out


def all_strategies(d: pd.DataFrame, panel=None, sym=None):
    """States for every strategy on one symbol."""
    st = M.states(d, panel, sym)
    base = build(st)
    base.update(cross_confirm(st, base))
    base.update(with_regime(st, base, d))
    return base
