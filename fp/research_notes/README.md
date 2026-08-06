# What was searched for a difference between winning and losing trades

Each script is runnable and re-checks itself against whatever is in
`bybit_bot/data/`. Full cost throughout: taker in, taker out, funding.

    python fp/research_notes/separate.py    # 106 entry features
    python fp/research_notes/context.py     # prior-signal context
    python fp/research_notes/streak.py      # previous outcome, per period
    python fp/research_notes/permethod.py   # the same, inside each method

## 1. Entry features — nothing separates them

All 106 features ranked by how well each splits winners from losers at
entry, on 2025+2026, corrected for testing 106 of them.

    best AUC 0.5122    threshold 0.5291    0 of 106 clear it

AUC 0.50 is no separation at all. At the moment of entry a winner and a
loser look the same.

## 2. Prior-signal context — nine new features, none significant

Time since the last signal, signal density over the last 20 and 50 bars,
direction persistence, how the last 3 and last 10 resolved. Strictly
past-only: a prior trade counts only once it has CLOSED at or before the
entry bar.

    best AUC 0.5080    threshold 0.5179    0 of 9 clear it

## 3. The previous outcome — the one real effect found

After a LOSS the next trade does better than after a WIN, and the sign
holds in every period:

    period   after a loss   after a win   p
    2025        33.5%          30.3%      0.022
    2026        35.5%          33.5%      0.270
    AUG         54.0%          15.4%      (n=150)

Statistically detectable in 2025 and directionally consistent everywhere.
It is also the only thing in this whole search that survived a holdout.

And it is still not enough. The gap is 2-3 points of win rate against a
cost that needs about 12. Best cell anywhere: 2026, trading only after
exactly one loss, at **-0.012% per trade**. That is the closest anything
has come to breaking even, and it is still the wrong side of zero.

Deeper loss streaks do not help monotonically -- run=1 beats run=2 and
run=3 in both fit years -- which is what noise looks like.

## 4. Inside each method — 0 of 12 profitable in every period

Per method, with each method's own trade history supplying the "previous
outcome". The effect does vary by method: ADX_TrendStrength gains +0.070
points from the filter in 2025, Bollinger_MeanRev loses 0.033 in 2026. No
method is profitable in every period either way.

    best worst-period result: ADX_TrendStrength at -0.0777%

ADX_TrendStrength shows +0.337% in 2026 and -0.148% in 2025 -- a sign
flip of that size across adjacent years is regime, not edge.

## 5. Learn per cell: forward, reversed, or dropped — fails worst of all

    python fp/research_notes/adapt.py

The full proposal, implemented as described. Split trades into cells by
what is visible at entry (ATR band, previous outcome, direction, vote
count, RSI side). In each cell measure what the forward trade returned
and what the reverse would have returned. Trade forward where forward
pays, reverse where reverse pays, drop the cell where neither does.

On the data it was fitted to, it is spectacular:

    min n   cells  fwd  rev  drop   fit net/trade   fit total
       20     118   23   18    77       +0.1753%     +439.2%

Walk-forward over the same data -- learn on everything before a block,
trade that block, never look ahead:

    min n    out-of-sample trades   net/trade      total       t
       20                   3,319    -0.1439%    -477.7%   -7.24
       50                   1,791    -0.1727%    -309.3%   -6.39
      100                     636    -0.2680%    -170.4%   -5.89
      200                      41    -0.6568%     -26.9%   -3.94

Fit +439%, out of sample -478%. The sign does not survive at all, and
t = -7.24 on 3,319 trades means that is not bad luck.

Worse, it is beaten by doing nothing clever: trading every signal forward
returns -0.1094%, and the adaptive rule returns -0.1439%. Choosing which
cells to reverse and which to drop DESTROYS value out of sample.

And the more selective the rule, the worse it gets -- -0.144% at 20
trades per cell down to -0.657% at 200. If the cells held signal, more
evidence per cell would make them more reliable. Instead the cells that
clear a higher bar are the ones whose in-sample deviation was most
extreme, which is selection on noise by construction.

## Summary

Five searches, five different framings, one consistent answer: the
difference between a winning trade and a losing one is not visible before
the trade. The one effect that is real -- outcome mean-reversion -- is
about a fifth of the size needed to pay the fee.

The fifth search is the important one, because it is the natural thing to
try and it fails in the most informative way. Reading the winners'
features and applying them next time is exactly what it does, and out of
sample it lands at -0.1439% against -0.1094% for making no decisions at
all. The features of a winning trade are the features of a trade that
happened to win. Applying them forward costs money.
