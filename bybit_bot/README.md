# Bybit Futures Bot — research record

> **All trading logic has been removed from this branch on request.** Two
> generations of it: the EMA-cross bot with its backtester and research
> scripts, and the later work built from the uploaded PureLogic /
> Logic_Final trade tables. Only the datasets in `data/` and this write-up
> remain. Nothing is lost — both are in git history:
>
> ```bash
> git log --oneline                        # see every version
> git checkout e6ae121 -- bybit_bot/       # the EMA-cross bot + research
> git checkout 19bb2ec -- pl run_bot.py    # the Logic_Final bot
> ```
>
> `e6ae121` is the last commit holding the first generation, `19bb2ec` the
> last holding the second. What follows is the record of what was tried and
> what the data actually said, kept because the measurements outlast the
> code that produced them.
>
> The logic in the branch today is `fp/slow.py`, built on the volatility
> drag measurement below. Everything after that section is the older record.

## The logic that is here now: `fp/slow.py` + `--signals slow`

```bash
pip install -r requirements.txt
python -m fp.slow                    # the measurement it is built on
python run_bot.py --signals slow     # $10 virtual, every USDT perpetual
python -m fp.test_bot                # 198 checks, no network
python -m fp.horizon                 # which horizon can pay for its fees
```

### Correction: the one positive result was a one-bar look-ahead

`fp/regime.py` reported the only out-of-sample-positive result in this
repo — state-conditioned logic selection, `trend` top5, **+89.5%** at
Sharpe 2.72, picked twice independently by nested validation. It is
wrong, and the cause is one line.

`states()` labels bar `i` from bar `i`'s own close. The return the
backtest credits to bar `i` is `close[i]/close[i-1] - 1`, driven by that
same close. Conditioning the choice of logic on the unlagged label let
the choice see part of the outcome it was about to collect. With 2,602
logics competing to exploit it, that sliver was the whole result:

| | before | after lagging the label |
|---|---|---|
| `trend` top5, daily | **+89.5%** | **-21.5%** |
| 36-cell state sweep | 9 positive | **0 positive** |
| 1h, same procedure | **+600,007%** | **-26.8%** |
| 4h, same procedure | **+10,632%** | **-35.4%** |

The live bot was never affected — it reads the last *closed* bar and
holds through the next one, which is the lag. Only the measurements were
misaligned. Every state label now goes through
`fp.regime.lagged_states()`, and `fp/test_bot.py` fails if the lag is
removed.

### Every logic tested on its own (`python -m fp.survivors`)

No hold floor. A logic qualifies on its own economics — the lower 95%
bound of its in-sample net return per trade must exceed zero, already
net of 0.055% each way and of funding over the hold it actually ran — so
a ten-minute logic and a ten-day logic face the identical bar. Then it
must clear a Bonferroni threshold out of sample, and its timeframe must
beat a rotation null that rolls each logic's timing at random while
keeping the market, the drift, and that logic's own long/short tilt.

| tf | built | no edge | tested | t bar | survivors | best t | best hold |
|---|---|---|---|---|---|---|---|
| 1m | 1,140 | 1,140 | — | | | | |
| 5m | 1,154 | 1,142 | — | | | | |
| 15m | 1,146 | 1,132 | — | | | | |
| 30m | 1,156 | 1,136 | — | | | | |
| 1h | 3,098 | 3,076 | — | | | | |
| 4h | 3,066 | 2,951 | 15 | 2.94 | **0** | 1.11 | 8.0h |
| 1d | 2,602 | 342 | — | | | | |

The "no edge" column is the honest version of the floor that used to be
here. Nothing was banned for being fast — every fast logic was measured,
and every one failed the same economic test the slow ones took. At 1m
that is all 1,140 of them.

Only 4h produced anything: 15 logics whose in-sample edge is genuinely
positive at 95% confidence after real fees. Out of sample the best
reaches t = 1.11 against a bar of 2.94. **The in-sample edge is real and
does not carry forward.**

And the picking procedure at 4h: -0.2639% per trade against a rotation
null of +0.6232% — selecting on past performance lands **below** random
timing, p = 1.000.

`fp/survivors.json` is therefore empty and `--signals survivors` opens
nothing. That is the output, not a missing setting.

### Bars with no clock in them (`python -m fp.eventbars`)

The deepest hard-coded constant left was the time grid itself: a 4h bar
is 4h whether the market traded a billion dollars in it or nothing.
Four replacements, each self-adjusting, at three rates each:

| bars | built | median | p10 | p90 | longest |
|---|---|---|---|---|---|
| volume/4 | 2,328 | 5.1h | 94m | 11.3h | 38.3h |
| volume/24 | 13,968 | 44m | 10m | 2.1h | 9.2h |
| volume/96 | 55,874 | 10m | 2m | 35m | 3.1h |
| cusum/4 | 1,766 | 6.2h | 75m | 17.2h | 32.4h |

A volume/4 bar takes 94 minutes when the market is busy and 38 hours
when it is not. Nobody chose either number.

| bars | logics | no edge | tested | t bar | surv | best t | picked | null | p |
|---|---|---|---|---|---|---|---|---|---|
| volume/4 | 3,014 | 2,763 | 47 | 3.27 | **0** | 1.13 | -0.4408% | +0.7571% | 1.000 |
| volume/24 | 3,036 | 3,007 | 3 | 2.39 | **0** | 0.33 | +0.0490% | +0.2227% | 0.775 |
| dollar/4 | 3,128 | 2,825 | 57 | 3.33 | **0** | 1.23 | -0.3176% | +0.9390% | 1.000 |
| range/4 | 2,980 | 2,707 | 57 | 3.33 | **0** | 1.42 | -0.3254% | +0.7430% | 1.000 |
| range/24 | 3,012 | 2,989 | 1 | 1.96 | **0** | 0.50 | +0.2606% | +0.2008% | 0.350 |
| cusum/4 | 2,952 | 2,387 | 23 | 3.07 | **0** | 1.50 | -0.2032% | +0.9935% | 1.000 |
| cusum/24 | 2,956 | 2,934 | 4 | 2.50 | **0** | 0.74 | +0.0141% | +0.1729% | 0.875 |

28,590 logic instances, twelve series, zero survivors. Best t anywhere:
1.50 against a bar of 3.07. At the fastest sampling (median bar ten
minutes) **no logic on any of the four bar types has a positive
in-sample edge after fees** — the same answer the ten-minute clock bars
gave, by a completely different route.

**The clock was not the problem.** Sampling now adapts to volume, to
turnover, to distance travelled and to volatility itself, and the result
did not move.

### Correction: the hold floor was wrong

An earlier version refused any logic held under an hour, arguing from
the average one-minute move (0.040%) being smaller than the round trip
(0.110%). That is true of a *random* one-minute position and says
nothing about a *selected* one — it compares the fee to the mean and
discards the distribution. **39.1% of ten-minute moves already exceed
the round trip**, and with leverage a 0.3% move is 30% on margin. The
floor threw every such logic away untested. Replaced by the economic
gate above; no time constant remains in the search or the bot.

### Correction: a second look-ahead, in the exit

A position runs from bar `i` to bar `j` and the signal flips at `j+1`.
The code booked the exit at `close[j]` — but at `close[j]` the signal
still says hold, and the bar being skipped is precisely the one that
caused the flip, usually the bar that ran against the position. One bar:

| | with the look-ahead | corrected |
|---|---|---|
| 4h survivors | **176** | **0** |
| 4h best t | 7.61 | 2.07 |
| 4h best OOS | +1471.9% | +38.3% |
| 1d picking, mean/trade | +2.1433% | +0.0534% |
| 1d picking, p vs null | 0.000 | 0.665 |

Scope: `trade_returns` in `fp/ensemble.py` (where it originated, unused
elsewhere) and `fp/survivors.py`. The horizon, regime and symmetry
results use per-bar accounting with `pos.shift(1)` and are unaffected.

### Which horizon can pay for its own fees (`python -m fp.horizon`)

Bybit charges 0.055% in and 0.055% out. A move's size grows with the
square root of time; the fee does not grow at all. So each hold length
has a minimum hit rate below which nothing can work, `p* = 0.5·(1 +
cost/E|move|)`, computed on all 838,112 one-minute bars:

| hold | E&#124;move&#124; | cost | hit rate needed |
|---|---|---|---|
| 1 min | 0.040% | 0.110% | **185.9% — impossible** |
| 5 min | 0.090% | 0.110% | **111.2% — impossible** |
| 15 min | 0.155% | 0.110% | 85.6% |
| 1 hour | 0.307% | 0.111% | 68.1% |
| 4 hours | 0.623% | 0.115% | 59.2% |
| 1 day | 1.617% | 0.140% | 54.3% |
| **2-3 days** | 2.78% | 0.200% | **53.6% — the minimum** |
| 10 days | 5.020% | 0.410% | 54.1% |

Below ten minutes, a forecast that is right *every single time* still
loses money: the move it captures is smaller than the fee it pays. The
requirement bottoms out at **2-3 days**, then rises again as funding
outgrows the move.

Running the full search at every timeframe, all of it lags correctly:

| tf | bars | days | trades | median hold | total | Sharpe |
|---|---|---|---|---|---|---|
| 1d | 583 | 582 | 1 | 2.0d | -3.4% | -9.16 |
| 4h | 3,493 | 582 | 290 | 8.0h | -35.4% | -2.20 |
| 1h | 13,969 | 582 | 1,127 | 3.0h | -26.8% | -0.61 |
| 30m | 27,938 | 582 | 2,168 | 90m | -64.6% | -2.32 |
| 15m | 55,875 | 582 | 3,526 | 45m | -71.2% | -2.32 |
| 5m | 60,000 | 208 | 2,524 | 15m | -41.2% | -1.91 |
| 1m | 60,000 | 41 | 7,039 | 3m | -16.4% | -5.00 |

Seven timeframes, seven losses, on both sides in every one.

### The finding that produced it

BTCUSDT fell **31.6%** over 2025 and the first eight months of 2026, with
a 53% maximum drawdown. So perfect directional knowledge was available in
hindsight: be short the whole way. Two ways of holding that same correct
short, over that same fall:

| how the position was held | gross | net |
|---|---|---|
| one position, never rebalanced | +31.6% | **+14.1%** |
| rebalanced to constant notional daily | +8.1% | **-9.2%** |

The 23.5-point gap is **volatility drag**. Daily returns have a 2.28%
standard deviation, and a position reset through that path compounds
sigma^2/2 against itself every period. Nothing about the forecast changed
between those rows -- only how often the position was reset.

**And leverage multiplies drag by the square.** Same perfectly-correct
short, net of funding:

| leverage | net |
|---|---|
| 1x | -9.3% |
| 3x | -102.1% |
| 10x | **-274.6%, account gone** |

The previous bot opened and closed every ~7 hours at 17-100x. That is the
worst available combination of both effects, and it loses on this data
**while holding a position that was correct throughout**. This is
arithmetic, not statistics: no amount of signal quality repairs it, which
is why six searches for a better signal all failed against it.

### What the design follows from

| rule | why |
|---|---|
| daily bars | a signal that can change every 30m forces 48 turnovers a day |
| held until the trend flips | no TP, no SL, no re-entry -- drag is paid per turnover |
| leverage 1-3x, solved per trade | return scales with L, drag with L^2, so there is a maximum |
| a thin move over a long hold is skipped | its solved leverage is 0 |

Leverage is still flexible by the trade's potential, as before -- the
change is that potential now means `move x L - costs x L - drag x L^2`,
which has a maximum instead of rewarding more:

```
 expected move    hold  best lev       net
            2%     90d     0.00x    -0.35%   not worth taking
            5%     30d     2.55x    +5.12%
           10%      7d     3.00x   +27.41%
```

### What is measured and what is not

Held without rebalancing, MA50 on daily bars returns **+26.1% at 1x**
over the period, positive in both years (+20.5% in 2025, +4.6% in 2026).

It does **not** survive walk-forward parameter selection: choosing the
best trend length on prior data every 60 days returns -8.5% over 35
out-of-sample trades, t = -0.12. That is statistically zero, not a loss,
but it is not an edge either. Across 45 momentum lengths only 38% are
profitable, median -10.8%, so 50 is a lucky value rather than a robust one.

**The structure is proven; the direction rule is not.** The drag
arithmetic holds regardless of what signal fills it. If you have a better
direction rule, this is the frame to put it in.

---

The strategy that was here: EMA9/EMA21 cross in the direction of a
confirmed higher-timeframe trend, with tiered position sizing,
volatility-capped leverage, ATR stop-loss, partial take-profit plus
breakeven, ATR chandelier trailing stop, and a daily loss circuit
breaker. Final measured results: **-1.38% on 2026, -4.69% on 2025.**

## Important: read this before running anything live

**No trading strategy is 100% accurate, and none ever will be — including
this one.** Any claim otherwise is false. This bot does not promise
guaranteed profit, does not use fixed maximum leverage, and does not
all-in on any single trade, on purpose:

- It only opens a position when a symbol's own price action actually
  matches the strategy in `bot/strategy.py`. If nothing matches, it does
  nothing — it never forces a trade to stay "active."
- Position size and leverage scale with a 0-100 signal-confidence score
  (`bot/risk.py`), capped well below "all-in" even for the strongest
  signal (see `ABSOLUTE_MAX_EQUITY_RISK_PCT` / `ABSOLUTE_MAX_LEVERAGE`).
  Leverage is further cut down when a symbol's own volatility (ATR%) is
  high, specifically to reduce liquidation risk.
- A daily loss circuit breaker stops new entries for the rest of the day
  if the account is down more than `risk.DAILY_LOSS_LIMIT_PCT`.

Defaults are `BYBIT_TESTNET=true` and `DRY_RUN=true` — the bot will not
place a single real order until you deliberately change both in `.env`.

## Which factors actually matter, measured on 89 of them

`research/factor_study.py` ranks a library of ~89 factors — price
differential over ten spans, buying pressure, close position in range,
up-candle share, volume level, volume growth and acceleration, run
length, direction persistence, candle body and wick anatomy, volatility
regime, trend distance, price/volume correlation — by how each one
correlates with the **realised net return of a real trade** (stop,
target, time exit, fees).

The column that decides usability is not strength, it is **sign
stability across years**. A factor that predicts one way in 2025 and the
opposite way in 2026 is worse than no factor at all: it steers the bot
wrong systematically rather than randomly.

**Result: 0 of 89 factors reach |IC| ≥ 0.02 on both years.** The
strongest sign-stable ones are all range/volatility measures, at an IC
around 0.012-0.029 — a correlation of one to three percent.

And the trap is right there in the top of the LONG table:

| factor | IC 2025 | IC 2026 | |
|--------|---------|---------|---|
| `pricediff_240` | **+0.0312** | **-0.0206** | flips |
| `pricediff_120` | +0.0205 | -0.0330 | flips |
| `dist_ema_240` | +0.0205 | -0.0306 | flips |
| `vwret_60` | +0.0172 | -0.0252 | flips |
| `range_30` | +0.0204 | +0.0171 | stable |

The four strongest single-year factors for going long all reverse sign
the following year. Anyone fitting a model on 2025 alone would have
found `pricediff_240` as their best signal and traded it into a loss in
2026 — which is exactly the failure mode documented in the section
below, reproduced here at the level of individual factors.

## The exhaustion sequence: red run, volume collapse, snap-back

Tested because it is a sequence, not a snapshot, and nothing else here
could express one: seven-ish red candles with volume *building* through
them, then a candle where volume *collapses*, then a reversal bar on
heavy volume — short the first part, long the second. The
gradient-boosted model in the next section saw volume as twenty
independent numbers per window, which cannot represent "seven in a row,
then a stop, then a snap back". That needs a state machine, and
`research/factor_study.py` implements one, sweeping run length (5/7/9),
volume build (0/10/30%), collapse threshold (<50%/<80% of average) and
snap-back volume (>1.3x/>2.0x) instead of pinning the example's exact
numbers. Short and long legs are scored separately.

**No configuration was profitable after fees on either year.** Net per
trade ran from -0.084% to -0.281% against a 0.210% round trip, with
t-statistics as low as -27.

One cell is worth flagging honestly rather than burying. The deepest
version — **9 red candles, volume collapsing below 80% of average, long
on the snap-back** — came in at -0.0907% (2025) and -0.1252% (2026).
Subtract the 0.210% fee and the *gross* return is **+0.12% and +0.085%,
positive on both years**. It is the only setup tested anywhere in this
repo whose gross edge is positive in both years rather than ~zero.

It is still not tradeable, and the reason is sample size: the setup fires
37 times in 2025 and 24 times in 2026. Both t-statistics are about -0.9,
i.e. statistically indistinguishable from zero, and the win rates
disagree badly (27% vs 54%). With two years of one symbol there is no way
to get more occurrences of something this rare. The honest read is "not
disproven, not established" — the one thread here that more data could
still settle, since running the same detector across many symbols would
multiply the sample without touching the logic.

## The strongest model built here, and how it failed

The sharpest version of the "trade every case correctly" idea is this:
ten falling candles are a short, but days later the same shape with a
different volume rhythm — different volume *differences between
consecutive candles* — is a different trend, and the bot should tell
those apart. That distinction is real, and nothing in the earlier
studies here tested it: they all described volume as a *level* (this
candle against a moving average), never as a *rhythm*.

`research/deep_market_model.py` was built specifically around it — 73
features including candle-to-candle volume ratios over the last 10 bars,
the share of window volume that traded on down candles versus up ones,
volume and return acceleration between window halves, return/volume
correlation — fed to gradient-boosted regression trees (120 trees, depth
6, written directly against numpy since scikit-learn is unavailable
here). Depth 6 carves up to 64 sub-cases per tree, which is what
"handle each small case" requires without memorising prices. The label
is the realised net return of a real trade: 2 ATR stop, 4 ATR target,
120-minute time exit, fees charged, stop taken first on ambiguous bars.

**The model does learn.** In-sample fit correlation is +0.26 to +0.32,
so volume rhythm genuinely does relate to trade outcome within a year.
Win rate rises with selectivity too — in one cell, from 40.78% at the
top 5% of predictions to 55.47% at the top 0.1% — so the ranking is not
meaningless.

**It just does not transfer.** Out-of-sample correlation across all four
runs: **+0.0100, +0.0113, +0.0129, +0.0145**. Fifteen of sixteen
selections lost money on unseen data.

The sixteenth is the interesting one, and `research/verify_positive_cell.py`
exists because it deserved a real test rather than a dismissal:

| test | result | t |
|------|--------|---|
| the cell: 2026 model → 2025 data, top 0.1% | **+0.2577%** over 521 trades | +3.86 |
| its mirror: 2025 model → 2026 data | **-0.2217%** over 306 trades | -3.30 |
| fresh split: 2025 H1 → 2025 H2 | **-0.8945%** over 260 trades | -9.90 |
| fresh split: 2026 H1 → 2026 H2 | **-0.3212%** over 153 trades | -3.12 |

Taken alone the cell is significant at t=+3.86. Three independent tests
contradict it, all significant, all negative. Note what the mirror does:
it does not merely fail to repeat, it flips sign *significantly*. That is
the signature of a model fitting one year's quirks — the learned rule is
not absent on new data, it is actively wrong there. And with sixteen
cells examined, the chance one clears t=2 on luck alone is about 56%.

So the answer to "the same shape with a different volume rhythm is a
different trend" is: **true within a year, and it does not carry to the
next one.** The relationship is real and it is not stable.

## The clearest result: the patterns do predict — by less than the fee

This one is worth reading before anything else, because it is the most
informative thing found in the whole project, and it is not "there is no
pattern."

The procedure requested was: look at the candles before a profitable
move, work out why that move was coming, encode it, then trade every
similar situation from the smallest profit to the largest.
`research/pattern_matching.py` implements exactly that, generalised so
none of the example numbers are baked in — pattern lengths of 10, 20 and
40 candles, holding times of 5 to 240 minutes, targets of 10% and 20% at
25x leverage, long and short.

**Step 1: the patterns genuinely carry information.** Out-of-sample lift
(how much more often a matched setup pays, versus the base rate) ran
from 1.17x to 7.92x, and — the part that rules out luck — the two
cross-year directions agreed with each other:

| pattern | horizon | target | base rate | after matching | lift |
|---------|---------|--------|-----------|----------------|------|
| 40 candles | 15m | 10% | 3.27% | 6.72% | **2.05x** |
| 40 candles | 30m | 10% | 7.49% | 12.46% | **1.66x** |
| 40 candles | 60m | 20% | 5.72% | 9.52% | **1.67x** |
| 40 candles | 5m | 20% | 0.11% | 0.54% | **4.82x** |

A library built on 2025 lifts 2026's hit rate by about as much as a
library built on 2026 lifts 2025's. That is a real, reproducible signal,
not curve-fitting.

**Step 2: trading it still loses, by almost exactly the fee.**
`research/pattern_strategy.py` takes those same libraries and places
actual trades on the other year — real stop, real target, hard time
exit, fees and slippage, stop-first on ambiguous bars. Across 60+
configurations the average net result per trade came out between
**-0.18% and -0.25%**, against a round-trip cost of **0.21%**.

Subtract the fee and the gross edge is approximately **zero**. Win rates
landed at 19-45%, profit factors at 0.17-0.49, and nothing was
profitable in both cross-year directions.

**Why lift doesn't become profit.** An "opportunity" is labelled by the
best price reached inside the window — a perfect exit. A real trade
carries a stop, and the move against you usually arrives first:
`research/two_stage_entry.py` measures that adverse excursion at a
median of 2.0-2.1 ATR on both years, which at 25x leverage is about
**17.5% of equity** before the trade goes your way. The pattern tells
you something true about where price is heading; it does not tell you
that price will get there without taking out your stop on the way.

So the honest summary is not "no edge exists." It is: **a real edge
exists, and it is smaller than what Bybit charges to trade it.** That is
also why this is hard to fix by trying more indicators — the problem is
the size of the toll relative to the signal, not the search for signal.

## "Just find the profitable trades in past data and trade them"

This was requested directly, with the reasoning that it *cannot* lose.
`research/hindsight_proof.py` implements exactly that procedure and runs
it on both years. Run it yourself: `python3 -m research.hindsight_proof`.

**Part A — the claim is correct.** Selecting trades by looking ahead and
keeping only the moves that beat fees produces, on real BTCUSDT data:

| year | hold | trades | win rate | return |
|------|------|--------|----------|--------|
| 2026 | 15m  | 9,708  | **100.000%** | +3.5e24 % |
| 2026 | 60m  | 4,239  | **100.000%** | +7.1e19 % |
| 2025 | 15m  | 16,029 | **100.000%** | +6.4e36 % |
| 2025 | 60m  | 7,216  | **100.000%** | +2.3e30 % |

Not 99%. Exactly 100%, every time, both years. So the intuition is not
wrong — on data whose outcome is already known, a losing trade is
impossible by construction.

**Part B — but those trades cannot be identified before they happen.**
That is the whole problem, and it is testable rather than a matter of
opinion. Part A's selector reads future candles; a live bot cannot. So
Part B tries to reproduce Part A's picks from past-only information:

*B1 — exhaustive rule search.* Thousands of EMA/RSI/volume rule
combinations (both directions), scored on one year, best one carried to
the other year:

| trained | best rule found | in-sample | out-of-sample |
|---------|-----------------|-----------|---------------|
| 2025 | ema12/200, rsi 40-60, vol>1.5 | -1237.89% | **-784.69%** (2026) |
| 2026 | ema9/21, rsi 40-60, vol>1.5   | -723.87%  | **-1248.57%** (2025) |

The *best rule out of thousands* is already negative on the very data it
was picked on, once fees and non-overlapping trades are enforced.

*B2 — machine learning on Part A's own labels.* Logistic regression, 26
backward-looking features (returns over 7 horizons, 5 EMA distances,
RSI, ATR, volatility ratios, volume ratios, range position, time-of-day,
streak), trained directly on the 100%-accurate labels:

| train → test | in-sample accuracy | out-of-sample accuracy | avg net/trade |
|--------------|--------------------|------------------------|---------------|
| 2025 → 2026 | 52.10% | **51.03%** | -0.2055% |
| 2026 → 2025 | 53.62% | **51.16%** | -0.2164% |
| 2025 → 2026 (4h) | 52.52% | **49.89%** | -0.2209% |
| 2026 → 2025 (4h) | 54.24% | **50.91%** | -0.2217% |

Out-of-sample accuracy lands on ~50-51% — a coin flip. And note the last
column: even at 51% accuracy the average trade still loses ~0.21%,
because the round-trip cost is 0.21%. Being right slightly more often
than chance does not help when the wins are the same size as the losses.
**That gap — 100% on known data, ~50% on unknown data — is the entire
reason a 100%-accurate bot cannot be built, and it is measured here, not
asserted.**

## Statistical foundation: what the raw data actually shows

Before picking any indicator, the raw 1-minute return series for both
datasets (`data/BTCUSDT_2026.csv`, `data/BTCUSDT_2025.csv`, ~307k and
~526k candles) was analyzed directly, independent of any specific
strategy, to check what structure genuinely exists to trade — this is
what grounds every design choice below, not indicator-guessing.

1. **Return autocorrelation at 6 horizons (1m, 5m, 15m, 1h, 4h, 1d),
   both years**: every value came out between -0.03 and +0.01 (except
   2025's 1-day horizon at -0.11) — i.e. **the raw return series is
   close to a random walk at every horizon checked.** There is no
   simple "up begets up" (momentum) or strong "up begets down"
   (mean-reversion) pattern to mechanically exploit on its own.
2. **Forward returns conditional on the classic EMA50/EMA200 trend
   filter**: being in a "confirmed uptrend" by this definition did
   **not** predict positive forward returns in either year (2026: next
   24h averaged -0.001% in "uptrend" vs -0.265% in "downtrend" — both
   negative; 2025: next 24h averaged -0.023% in "uptrend" vs **+0.041%**
   in "downtrend" — inverted from what a trend-follower would assume).
   This lagging-MA "trend" label, on its own, doesn't predict direction
   — whatever edge the EMA-cross strategy below has comes from the
   interaction of the crossover *timing* with risk management, not from
   "uptrend implies more upside" as a standalone statistical fact.
3. **Reversion after fast moves**: after a ≥1% drop within 5 minutes,
   the average next-15-minute return was consistently positive in both
   years (+0.066%/2026, +0.143%/2025) — a real, cross-validated
   "flash-dump bounce" pattern (consistent with liquidation-cascade
   overshoot, a known feature of leveraged crypto futures). This looked
   promising enough to build and test as a real strategy — buy the
   flash dump, ATR-scaled stop/target, fee-aware.
4. It didn't survive contact with a realistic trade simulation. Across
   15+ stop/target/threshold combinations, **and** a variant requiring
   the dip to be above the 1-day EMA (a "buy the dip in an uptrend"
   filter), every single configuration was net negative on both years
   once a real stop-loss, realistic path-dependency (stop-before-target
   ambiguity resolved conservatively), and fees were included — the
   average-return statistic in point 3 is real, but it's driven by a
   skewed distribution (a minority of large bounces), not something a
   fixed stop/target rule can reliably capture. This was not adopted.

**What this means honestly**: BTCUSDT's own price/volume history, at
these horizons, does not contain a strong, mechanically-exploitable
directional edge net of Bybit's fees. This isn't a claim unique to this
bot — it's consistent with BTC perpetuals being one of the most liquid,
closely-arbitraged markets that exists, where simple technical patterns
get competed away fast. The version below is the most defensible one
found (positive in one year, mildly negative in the other, small
sample), presented with that context rather than a false promise.

## How the strategy was derived

`backtest/run_backtest.py` runs the exact same signal code
(`bot/strategy.py`) used by the live bot against real BTCUSDT 1-minute
futures data. This is not a hypothetical demo — it is the actual trade
log the strategy would have produced. The strategy went through several
iterations as more data became available (1 month → Jan-Aug 2026, ~307k
candles → full year 2025, ~526k candles, `data/BTCUSDT_2025.csv`), and
every version's result below is the honest number, not a cherry-picked
one:

1. **Pullback-to-EMA21 entry** (mean reversion): almost no signals on 1
   month of data, lost on the ones it took.
2. **Donchian breakout entry**: ~32-34% win rate *at every confidence
   tier*, profit factor < 1 on 7 months of 2026 data. Several fixes
   tried, none worked — a fresh N-bar extreme gets wicked through and
   reversed too often on this symbol/timeframe.
3. **EMA9/EMA21 crossover entry**, still gated by the HTF trend filter:
   38.1% win rate, profit factor 1.29, +9.46% on the 2026 dataset — but
   when tested **out-of-sample on all of 2025** (a much choppier year,
   whipsawing ±5-18% almost every month instead of one clean trend) it
   lost **-16.95%** with a **-19.62% drawdown**. That gap is a textbook
   overfitting signature: a strategy shaped around one dataset's specific
   character failing on a genuinely different one. This is reported
   because it's the honest result of the out-of-sample test, not because
   it's flattering.
4. Added three changes, each requested explicitly and each validated
   (not assumed) on both years:
   - **Fee-aware filter**: a signal is only taken if its smallest profit
     target (TP1) clears Bybit's round-trip taker fee + assumed slippage
     by ≥8x (`bot/risk.py: ROUND_TRIP_COST_PCT`, `strategy.py:
     MIN_TP1_TO_COST_RATIO`) — a "win" on paper must still be a win after
     real costs.
   - **Anti-chase filter**: reject longs where RSI(14) is already ≥70
     (overbought) and shorts where it's already ≤30 (oversold) — this is
     the "không đu đỉnh/đáy" rule. An earlier attempt measured distance
     from the 1h EMA50 in ATR units instead; that rejected most of the
     *good* continuation trades too (a slow average naturally trails far
     behind price in any real trend), so it was replaced with plain RSI,
     which didn't have that side effect — see the comment in
     `strategy.py` for the numbers that led to swapping it.
   - **Stale-position exit**: a position that hasn't reached TP1 within 6
     hours and isn't at least +0.3R unrealized gets closed at market
     instead of sitting on margin indefinitely (`risk.py:
     STALE_POSITION_MAX_MINUTES/MIN_R`).
5. **Current: entries evaluated on every closed 1-minute candle**
   instead of every 15-minute candle, per an explicit follow-up request
   to analyze and trade on 1m bars. This needed two real fixes, not just
   "use 1m data":
   - Naively computing EMA9/EMA21/ATR14/RSI14 directly on 1m bars was
     almost pure microstructure noise — backtested on both years with
     only the fee filter re-tuned, results ranged from **-44% to -21%**
     with no setting working on both. The periods were rescaled to the
     same *real time span* that worked at 15m (9 bars×15m = 135m, 21
     bars×15m = 315m, etc.), computed directly on 1m closes.
   - That fixed EMA/RSI, but not ATR: ATR measures the size of a bar's
     *own* range, so a longer lookback period never fixes it — a
     1-minute candle's range is always much smaller than a 15-minute
     candle's, regardless of how many of them you average. Fixed with
     `indicators.rolling_block_atr`: a sliding 15-minute window's range
     (as if it were one candle), EMA-smoothed — reproducing the original
     15m ATR's real-world units while still updating every minute.

6. **Investigated, and rejected after testing: ZigZag swing-pivot
   entries.** Checked directly against the data first (not assumed): a
   0.5% ZigZag threshold on 1-minute BTCUSDT closes finds ~14 confirmed
   swing pivots *per day* — i.e. there genuinely are many more raw
   candidate opportunities per day than versions 1-5 ever looked for.
   Built `indicators.zigzag_pivots` (causal, no lookahead) and an entry
   at every confirmed pivot whose target cleared trading costs, sized by
   swing amplitude and HTF-trend alignment (a soft confidence modifier
   instead of a hard gate, so smaller counter-trend swings could still
   be taken at reduced size). Rigorously backtested — not just once:
   - Baseline: -19.5%/2026, DD -21.4%.
   - Swept stop buffer 0.3-2.5×ATR (wider, to survive the common
     "retest the pivot before continuing" pattern): every wider setting
     was *worse* (down to -84%), because a wider stop also raises the
     TP1 target enough to newly pass the fee filter, letting in more
     low-quality signals, not fewer.
   - Swept the ZigZag threshold 0.3%-2.0%: mostly -70% to -92%; only the
     tightest, most fee-filter-restricted setting (0.3%, just 20 trades)
     was near flat, too small a sample to trust.
   - Hard-gating out counter-trend pivots instead of just down-weighting
     them: -19.4%, basically unchanged — so the losses weren't
     concentrated in the counter-trend trades specifically.
   - Exiting at the *next opposite* pivot instead of an R-multiple
     target: -16.1%, win rate 25%, still net negative.
   - A hybrid (EMA-cross entries in version 5, or a same-direction
     swing pivot, both still gated by HTF trend, R-target reduced to
     1.5R): -6.1%, still losing.
   No configuration tested showed a robust edge, on 2026 alone, let
   alone confirmed out-of-sample on 2025. This surfaces a genuine
   tension between two explicit requirements: "catch every small-to-
   large opportunity" and "guarantee profit after fees" — enforcing the
   second rigorously rejects nearly all of what the first would want to
   take, and even the survivors didn't show a proven edge. Reported
   here in full rather than quietly dropped, because a strategy this
   thoroughly tested and still losing is itself the honest answer to
   "is there money being left on the table here" — not evidence of
   insufficient effort. The zigzag detector was removed from the
   codebase after this (no dead code) — this section plus the commit
   history is the record of what was tried.
7. **Current: reverted to the version-5 (1m EMA-cross) entries**, since
   that remains the only version with genuine (if modest,
   regime-dependent) validated results. One real bug was found and
   fixed while investigating why counter-trend pivot trades were
   getting stopped out within 1-2 minutes: the trend-flip exit compared
   the *current* HTF trend to the position's direction, so a "flat" HTF
   reading (neither up nor down) counted as a flip and force-closed the
   position almost immediately. Fixed to compare against the trend *at
   entry time* instead, so a position now only exits on a genuine
   reversal, not merely a neutral reading (`entry_trend` field in
   `backtest/engine.py`, `bot/scanner.py`, `bot/paper_trading.py`).

**Results, current version, both years, out-of-sample each way**
(identical strategy structure between these two runs):

| Dataset | Trades | Win rate | Profit factor | Return | Max drawdown |
|---|---|---|---|---|---|
| Jan-Aug 2026 | 14 | 21.4% | 0.28 | -13.34% | -13.34% |
| Full year 2025 | 26 | 50.0% | 1.27 | **+4.76%** | -7.43% |

One year up, one year down, neither dramatic, with 14-26 trades a year —
this is not a strong or reliably repeatable edge, and it's reported
exactly that way. **Trading on 1m candles gives faster reaction time
(reacts to a crossover within ~1 minute instead of within 15), not a
provably better edge** — the underlying trend/momentum information is
the same either way, since the indicator periods were rescaled to cover
the same real time span. `EMA_FAST=135, EMA_MID=315,
ATR_BLOCK_MINUTES=15` in `strategy.py` is what's wired into the live
bot, scanner, and paper trading.

Re-run it yourself (the 2025 run takes noticeably longer — it's ~526k
1-minute bars processed one at a time):

```bash
python -m backtest.run_backtest data/BTCUSDT_2026.csv
python -m backtest.run_backtest data/BTCUSDT_2025.csv
python -m backtest.run_backtest data/BTCUSDT_202607.csv   # smaller, choppier reference month
```

## On the request to catch every possible trade, and the "short first, then long" idea

Two specific asks from the brief for this version deserve a direct
answer instead of a silent substitution:

**"Đảm bảo trade được các lệnh dù lời nhỏ nhất đến lớn nhất không bỏ sót
bất kỳ lệnh tiềm năng nào"** — catching literally every profitable move,
no matter how small, is not achievable by any real system. The only way
to never skip a small profitable wiggle is to trade every single price
change, which mostly means trading noise, and noise loses to fees (this
is exactly what the new fee-aware filter exists to prevent). What the
current strategy does instead: it takes every setup that clears the
trend, momentum, volume, anti-chase, and fee filters, from the smallest
qualifying tier to the largest confidence tier — nothing above the
confidence floor is skipped, but the floor itself exists on purpose.

**The "short first for a small guaranteed profit, then long" idea** (or
mirrored for downtrends): entering counter to the higher-timeframe trend
to bank a small move before reversing into the main trend. This is not
implemented as described, because "guaranteed" small profit from timing
a short-term counter-trend move is strictly *harder* than timing the
main trend itself — it requires correctly predicting a local top/bottom
within a move that hasn't happened yet, which is precisely the
"đu đỉnh/đáy" mistake the brief also explicitly warns against, just
aimed the other direction. Implementing it as "guaranteed" would be
dishonest. What was implemented instead, aimed at the same underlying
goal — not leaving capital idle — is the **stale-position exit** above:
capital that isn't working gets freed automatically rather than sitting
in a slow trade, so it's available for the next qualifying setup instead
of being tied up for the full duration of a move.

## Applying the same logic to other coins

The live bot (`bot/scanner.py`) and paper-trading mode
(`bot/paper_trading.py`) both evaluate every symbol in `SYMBOLS` against
the identical rules above, independently, every poll cycle. A coin only
gets traded when its own 1h/1m candles satisfy the regime + entry +
anti-chase + fee conditions; coins that don't match are simply skipped
that cycle. There is no "if it's Bitcoin do X, otherwise do Y" — the same
functions (`strategy.prepare_from_ltf_htf` / `strategy.signal_from_row`)
run for every symbol, in both modes.

## Setup

```bash
cd bybit_bot
pip install -r requirements.txt
cp .env.example .env
# edit .env: at minimum review SYMBOLS and POLL_INTERVAL_SECONDS
```

To eventually trade with a real account (start on testnet):

1. Create a Bybit **testnet** account and API key (Contract Trading
   permission only — do NOT enable withdrawals) at
   https://testnet.bybit.com
2. Put the key/secret in `.env` as `BYBIT_API_KEY` / `BYBIT_API_SECRET`,
   keep `BYBIT_TESTNET=true`.
3. Set `DRY_RUN=false` once you're ready for the bot to place real
   (testnet) orders. Watch the logs for at least a few days.
4. Only after that, and only if you fully accept the risk, consider
   `BYBIT_TESTNET=false` with a **mainnet** key — real funds, real risk.

## Running

Run the tests first:

```bash
python -m pytest tests/ -v
```

Run the backtest report:

```bash
python -m backtest.run_backtest data/BTCUSDT_2026.csv
```

Run the bot (safe by default: testnet + dry-run, logs what it *would*
do without placing orders):

```bash
python -m bot.main
```

Run in dry-run against testnet with a specific coin list and faster
polling:

```bash
SYMBOLS=BTCUSDT,ETHUSDT POLL_INTERVAL_SECONDS=30 python -m bot.main
```

Go live on testnet (real testnet orders, fake funds) after editing
`.env` to set `DRY_RUN=false`:

```bash
python -m bot.main
```

Go live on mainnet (real funds — only after you've reviewed everything
above) after editing `.env` to set `BYBIT_TESTNET=false` and
`DRY_RUN=false`:

```bash
python -m bot.main
```

Stop the bot any time with Ctrl+C (or SIGTERM) — it shuts down cleanly
without leaving background threads; any position already open on the
exchange stays open with its stop-loss still active on Bybit's side.

## Quick start: run it on your own machine

```bash
git clone <this repo>
cd bybit_bot
pip install -r requirements.txt
python run_paper_bot.py
```

That is the whole setup. No API key, no `.env`, no account. It reads
Bybit's public kline and ticker endpoints and never submits an order —
the $10 is a number in memory.

```bash
python run_paper_bot.py --equity 10 --symbols BTCUSDT
python run_paper_bot.py --symbols BTCUSDT,ETHUSDT,SOLUSDT --max-positions 2
python run_paper_bot.py --trades-csv session.csv
```

Press Ctrl+C to stop; it prints closed trades (winners, losers, total
profit, total loss), positions still open at that moment (how many
winning, how many losing, unrealized P&L each), and the two combined.

It defaults to **mainnet** prices deliberately: Bybit's testnet has its
own thin synthetic order flow, so paper-trading against it would say
nothing about real price action. No key is sent and no order is placed
either way.

**Expect long idle stretches.** On the 2026 data the strategy entered 6
times in eight months. Sitting still is the entry filter doing its job,
not the bot hanging.

**Before you read a green session as proof:** on the two years of history
in `data/`, this strategy loses money — **-1.38% on 2026, -4.69% on
2025**. A profitable afternoon is a small sample. The rest of this README
is the evidence for why the number is what it is.

## Paper trading against real, live Bybit data

`bot/paper_trading.py` runs the identical strategy/risk logic against
**real real-time Bybit market data** (public kline + ticker REST
endpoints — no API key needed) but against a **virtual account**, so you
can watch it trade on live price action with realistic simulated fees,
slippage, and TP/SL fills before ever using real funds:

```bash
python -m bot.paper_trading --equity 10
```

- `--equity` sets the starting virtual balance (default 10, i.e. $10 as
  requested — see the note below on what that does and doesn't prove).
- Entries are checked on every newly-closed **1-minute** candle
  (`SIGNAL_POLL_SECONDS = 60`), matching the live scanner.
- Open positions are checked every 15 seconds (`PRICE_POLL_SECONDS`), and
  the check scans the **high/low range of every 1m candle since the last
  check**, not just the price at that instant. This matters: a real
  TP/SL order rests on the exchange book and fires the moment price
  *touches* it, so sampling the last price every 15s would silently skip
  fast wicks that touched a stop and retraced — which flatters results by
  discarding losses that really happened. Same-candle ambiguity (stop and
  target both inside one bar) resolves in favour of the stop, identical
  to `backtest/engine.py`, so paper mode never looks better than the
  backtest for accounting reasons alone.
- Stop with Ctrl+C at any point. It prints a full session summary in
  three blocks: **closed trades** (count, winners, losers, total profit,
  total loss, net realized), **still open at shutdown** (count, how many
  are currently winning vs. losing, unrealized profit/loss per position
  and in total), and **combined** (closed + open together — total
  trades, total winners, total losers, total profit, total loss), plus
  final equity with and without open positions.
- `--trades-csv trades.csv` additionally writes every trade to CSV, with
  a `status` column marking each row `closed` or `open_at_shutdown`.

### Seeing the output without waiting for live trades

The strategy is deliberately selective (14 entries across all of 2026),
so a freshly started live session can easily run for hours with nothing
to show. `bot/replay.py` feeds historical candles through the *same*
`PaperBroker`, one 1m candle at a time with no lookahead, so you can see
the exact session summary immediately:

```bash
python -m bot.replay data/BTCUSDT_2026.csv --equity 10 --candles 9000 --warmup 27000
```

This is not a substitute for `backtest/run_backtest.py` (which is far
faster and covers whole years) — its purpose is to exercise the live
code path and show the live summary format.

**Why it ignores Bybit's real minimum order size**: paper trading never
submits a real order, so there's no real lot-size constraint to respect.
With a genuinely tiny account like $10, the risk-based position sizer
would compute a quantity below Bybit's real minimum for something like
BTCUSDT anyway (a real $10 account mostly couldn't trade BTC perpetuals
at real position-sizing discipline) — paper mode sizes purely off the
risk model so you can still observe the decision logic. Use a larger
`--equity` (e.g. 500-1000) if you want position sizes that would also be
realistically executable on a real account of that size.

**Network note**: klines/tickers are public Bybit endpoints, so no API
key is required to run this — but it does need real outbound access to
`api.bybit.com` (or `api-testnet.bybit.com` if `BYBIT_TESTNET=true`).
This is a genuine, confirmed environment limitation, not a code issue:
the sandbox this bot was built in runs its outbound traffic through a
policy-enforcing proxy that returns a `403` (policy denial) for
`api.bybit.com`, `api-testnet.bybit.com`, `api.bytick.com` and
`stream.bybit.com` alike — checked directly (`curl` to the kline
endpoint, plus the proxy's own status log naming the rejected host)
rather than assumed. `curl -sS "$HTTPS_PROXY/__agentproxy/status"` in
that environment shows the rejection if you want to see it yourself.

Because of that, **the live network path specifically could not be
exercised from here** — that one step needs a machine with real outbound
access to Bybit. Everything downstream of the network call *is* tested:
`tests/test_paper_trading.py` and `tests/test_scanner.py` drive the
fill/fee/signal/summary logic against a fake exchange fed real
historical candles (including a test that a stop-piercing wick which
closes back at entry still stops the position out), and `bot/replay.py`
runs the entire live loop end-to-end on historical data. So run
`python -m bot.paper_trading --equity 10` on your own machine and it
should work — but treat the first live session as the real
confirmation, since the HTTP layer itself is the one untested link.

**Why the kline fetch paginates**: the slower indicators (EMA span=315
on 1m bars) need several thousand 1-minute candles of history to
actually *converge*, not just to produce a non-NaN value — with too
short a window the EMA is still biased toward wherever the window
happened to start. `bot/exchange_bybit.py: get_klines` transparently
pages past Bybit's 1000-candles-per-call limit (walking backward with
the `end` parameter) to fetch the ~2000 bars both the live scanner and
paper trading request on every poll.

## Project layout

```
bot/
  indicators.py     EMA / RSI / ATR / ADX (pure pandas, no TA lib dependency)
  strategy.py        Signal logic shared by backtest, live bot, and paper trading
  risk.py             Position sizing tiers, dynamic leverage cap, fees, safety limits
  exchange_bybit.py   pybit v5 REST wrapper (klines, tickers, orders, stop management)
  scanner.py          Per-symbol signal evaluation + open-position management (live)
  paper_trading.py    Same logic against real-time data, virtual account, no real orders
  replay.py           Drives paper_trading's broker over historical candles (no network)
  config.py           .env-driven configuration
  main.py              Poll loop entrypoint (live bot)
backtest/
  engine.py            Bar-by-bar backtest engine (fees, slippage, funding, sizing)
  run_backtest.py      CLI report
research/
  hindsight_proof.py   Why "find profitable past trades and trade them" can't be
                       turned into a bot: 100% win rate with lookahead vs ~50%
                       accuracy without it, measured on both years
  pattern_matching.py  Do similar candle shapes repeat the same outcome? Sweeps
                       pattern length, horizon, target, direction. Answer: yes,
                       lift 1.2-7.9x out-of-sample, confirmed both directions
  pattern_strategy.py  Turns that lift into real trades with stops and fees.
                       Answer: -0.18 to -0.25% per trade vs a 0.21% round trip
  two_stage_entry.py   "Short the dip first, then long" vs waiting vs entering
                       now. Measures the adverse excursion (median ~17.5% at 25x)
  pullback_sweep.py    Entry-pullback depth swept through the real engine
  filter_sweep.py      How many trades the entry gates discard, and whether the
                       discarded ones would have made money. Also how the dead
                       RSI filter was found
  general_logic.py     Coarse regime cells learned from realised trade returns,
                       plus the fee-sensitivity sweep down to zero cost
  billion_rules.py     One rule per profitable bar, nothing combined; 100%
                       in-sample, ~fee-negative everywhere else
  deep_market_model.py 73 volume-rhythm features + gradient-boosted trees
  verify_positive_cell.py  Re-tests the single positive result on three
                       independent splits
data/
  BTCUSDT_2026.csv     Jan-Aug 2026 dataset (307k candles)
  BTCUSDT_2025.csv     Full year 2025 dataset (526k candles, out-of-sample validation)
  BTCUSDT_202607.csv   July-2026-only dataset (smaller reference month)
tests/                 Unit + integration tests (indicators, strategy, risk, scanner, paper trading)
```

## Configuration reference (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `BYBIT_API_KEY` / `BYBIT_API_SECRET` | empty | Bybit API credentials |
| `BYBIT_TESTNET` | `true` | `false` = mainnet, real funds |
| `DRY_RUN` | `true` | `false` = actually place orders |
| `SYMBOLS` | 10 major USDT perps | Comma-separated symbols to scan |
| `POLL_INTERVAL_SECONDS` | `60` | Seconds between scan cycles |
| `MAX_CONCURRENT_POSITIONS` | `4` | Hard cap on simultaneous open positions |
| `EQUITY_OVERRIDE_USDT` | `10000` | Used only when no API key is set (pure dry-run) |
| `LOG_LEVEL` | `INFO` | Python logging level |

Risk limits (`bot/risk.py`) are not environment variables on purpose —
changing them is a deliberate code change, not a one-line config flip:

- `TIERS`: confidence → (equity % risked, leverage) tiers
- `ABSOLUTE_MAX_LEVERAGE = 25`, `ABSOLUTE_MAX_EQUITY_RISK_PCT = 0.25`
- `DAILY_LOSS_LIMIT_PCT = 0.15`
- `ROUND_TRIP_COST_PCT`: assumed Bybit taker fee + slippage round trip,
  used by the fee-aware signal filter in `strategy.py`
- `STALE_POSITION_MAX_MINUTES = 360`, `STALE_POSITION_MIN_R = 0.3`:
  capital-efficiency exit for positions that aren't progressing

If you deliberately want higher risk limits, edit those constants with
full understanding that it increases both potential return and the
chance of large/liquidating losses.
