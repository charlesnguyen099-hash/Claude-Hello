# Bybit Futures Trend-Following Bot

A trend-following bot for Bybit USDT perpetual futures (EMA9/EMA21 cross
in the direction of a confirmed higher-timeframe trend), with strict risk
management (tiered position sizing, volatility-capped leverage, ATR
stop-loss, partial take-profit + breakeven, ATR chandelier trailing
stop, daily loss circuit breaker).

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
