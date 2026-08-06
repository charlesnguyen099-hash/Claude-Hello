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

## Summary

Four searches, four different framings, one consistent answer: the
difference between a winning trade and a losing one is not visible before
the trade. The one effect that is real -- outcome mean-reversion -- is
about a fifth of the size needed to pay the fee.
