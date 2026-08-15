"""One position per coin, always the best one available, target +1%.

The operator's rules, exactly:

  1. A coin holds ONE position at a time.
  2. A signal is skipped only when the coin already holds a position whose
     expectation is HIGHER. A better signal replaces a worse one.
  3. A trade counts as profitable only above +1% net of every fee.
  4. Catch as many of those as possible.

WHY RULE 1 CHANGES EVERYTHING. Across 623 strategies and ten coins there
are 2,527,601 trades that finish above zero and 280,005 that finish above
+1%. Both numbers are inflated beyond recognition: hundreds of strategies
ride the same move, so the same opportunity is counted hundreds of times.
Force one position per coin and the ceiling collapses to 7,864 DISTINCT
trades -- 2.81% of the 280,005 -- worth +15,686% if every one of them
were caught perfectly.

That 7,864 is the real target. It is computed here exactly, by weighted
interval scheduling: sort every above-threshold trade by exit, and take
the highest-value set that never overlaps. It is a hindsight number --
it needs to know each trade's outcome before choosing -- so no live logic
can reach it. But it is the honest denominator, and every result below is
reported against it rather than against the 2.5 million.

THE LIVE RULE. At each bar a coin may have many strategies signalling.
Each carries an expectation measured on the TRAINING side only. The coin
holds the highest-expectation signal that is currently on; a new signal
displaces the held one only if its expectation is higher, and the switch
pays the round trip both ways, which is charged here.
"""
from __future__ import annotations

import bisect

import numpy as np

from fp.data import FEE, FUNDING_PER_8H

# A trade is a win only above this, net of everything.
WIN = 0.01


def ceiling(entries, exits, nets, threshold: float = WIN):
    """The most a single position per coin could ever have captured.

    entries/exits are bar indices, nets are per-trade returns. Returns
    (count, total) for the best non-overlapping subset above threshold.
    """
    keep = nets > threshold
    if not keep.any():
        return 0, 0.0
    rows = sorted(zip(exits[keep].tolist(), entries[keep].tolist(),
                      nets[keep].tolist()))
    ends = [r[0] for r in rows]
    best = [0.0] * (len(rows) + 1)
    cnt = [0] * (len(rows) + 1)
    for i, (ex, en, v) in enumerate(rows, 1):
        j = bisect.bisect_right(ends, en - 1, 0, i - 1)
        if best[j] + v > best[i - 1]:
            best[i], cnt[i] = best[j] + v, cnt[j] + 1
        else:
            best[i], cnt[i] = best[i - 1], cnt[i - 1]
    return cnt[-1], best[-1]


def simulate(close, states: dict, expect: dict, threshold: float = WIN,
             min_hold: int = 0, take_profit: bool = False,
             giveback: float = 0.0, only_win_exit: bool = False):
    """Trade one coin under the operator's rules.

    states  {name: signed int8 array}   what each strategy says, per bar
    expect  {name: float}               its measured expectation, from
                                        TRAINING data only

    THE RULES, and why each one is here:

      ONE POSITION. A coin holds at most one. Without it, 623 strategies
      ride the same move and the account is asked for exposure it does
      not have.

      A WINNER IS LEFT ALONE. While the open position is in profit, every
      other signal is skipped -- that is the operator's rule verbatim, and
      it is also what stops the churn. The first version switched to any
      higher-ranked signal the moment it appeared: 4,911 trades on one
      coin in thirty-three days, mean hold eight minutes, gross +192.8%
      turned into net -347.4% by a -540.2% fee bill. The signal was never
      the problem; the turnover was.

      +1% BANKS IT. A trade is only a win above +1% net, so once it is
      there the position is closed rather than handed back to the market.

      A LOSER MAY BE REPLACED, but only by a strictly better expectation
      and only after min_hold bars, so a flickering state cannot bounce
      the account in and out on consecutive minutes.

    Returns a list of (entry_bar, exit_bar, side, net, strategy).
    """
    names = [n for n in states if expect.get(n, 0.0) > 0]
    if not names:
        return []
    names.sort(key=lambda n: -expect[n])
    A = np.vstack([states[n] for n in names])
    n_bars = A.shape[1]

    out = []
    held, side, entry_i, peak = -1, 0, 0, 0.0
    spent: dict = {}
    for i in range(n_bars):
        col = A[:, i]
        if held >= 0:
            live = _net(close, entry_i, i, side)
            age = i - entry_i
            peak = max(peak, live)
            # 1. Banked, if the caller wants a hard target. OFF by
            #    default, and that default matters: capping winners at
            #    +1% while a loser runs until its signal dies is the
            #    wrong way round, and it is what turned a 72%-hit-rate
            #    run into -305%. +1% is the operator's DEFINITION of a
            #    win, not an instruction to sell there.
            if take_profit and live >= threshold:
                out.append((entry_i, i, side, live, names[held]))
                # LOCK IT OUT until its state actually cycles. Banking a
                # win and re-entering the same signal on the very next
                # bar is not two trades, it is one trade paying the round
                # trip twice -- and it was most of the 4,077 trades that
                # buried a 72% hit rate under a 448% fee bill.
                spent[held] = side
                held = -1
                continue
            # 2. GIVEBACK. The only thing that cuts a loser before its
            #    signal dies, and it is not a fixed price: it is a share
            #    of what this trade had already earned, so it moves with
            #    the trade. At giveback=0 it is off entirely.
            if giveback > 0 and peak > 0 and live <= peak * (1.0 - giveback):
                out.append((entry_i, i, side, live, names[held]))
                held = -1
                continue
            if only_win_exit:
                # THE OPERATOR'S RULE, taken literally: never close at a
                # loss. A position is released only once it is worth more
                # than +1% net, so by construction EVERY CLOSED TRADE IS A
                # WINNER. What that cannot do is make the money appear --
                # it moves the loss from the realized column to the open
                # one, and the open one still spends the account. See
                # run_hold.py, which reports both.
                continue
            still = col[held] == side
            if still and live > 0:
                # 2. In profit and its own reason still stands: leave it
                #    alone. Every other signal is skipped here.
                continue
            on = np.flatnonzero(col)
            best = int(on[0]) if len(on) else -1
            better = best >= 0 and best < held and age >= min_hold
            if not still or better:
                out.append((entry_i, i, side, live, names[held]))
                held = -1
            else:
                continue
        # A locked-out strategy is released once its own state stops
        # saying what it said when the win was banked.
        for k in [k for k, v in spent.items() if col[k] != v]:
            del spent[k]
        if held < 0:
            on = [j for j in np.flatnonzero(col) if j not in spent]
            if on:
                held = int(on[0])
                side = int(col[held])
                entry_i = i
                peak = 0.0
    if held >= 0 and entry_i < n_bars - 1:
        out.append((entry_i, n_bars - 1, side,
                    _net(close, entry_i, n_bars - 1, side), names[held]))
    return out


def _net(close, i, j, side):
    gross = side * (close[j] / close[i] - 1.0)
    hours = (j - i) / 60.0
    return gross - FEE - hours / 8.0 * FUNDING_PER_8H


def score(trades, threshold: float = WIN):
    if not trades:
        return dict(n=0, total=0.0, wins=0, mean=np.nan, hits=0)
    v = np.array([t[3] for t in trades])
    return dict(n=len(v), total=float(v.sum()), wins=int((v > 0).sum()),
                mean=float(v.mean()), hits=int((v > threshold).sum()))
