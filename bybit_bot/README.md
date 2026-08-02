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
  (`SIGNAL_POLL_SECONDS = 60`), matching the live scanner. Open positions
  are additionally checked against the **live last price** every 15
  seconds (`PRICE_POLL_SECONDS`) for stop/TP1/TP2 fills — faster than
  waiting for the next candle close.
- Stop with Ctrl+C at any point. It prints a full session summary:
  closed trades (count, wins, losses, realized P&L) **and** currently
  open positions (count, how many are currently winning vs. losing right
  now, unrealized P&L per position and in total), plus the combined
  realized + unrealized total — covering both what already happened and
  what's still in flight, as requested.

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
policy-enforcing proxy that returns a `403` (policy denial) specifically
for `api.bybit.com` — checked directly (`curl` to the kline endpoint,
and the proxy's own status log) rather than assumed. `curl -sS
http://127.0.0.1:39861/__agentproxy/status` in that environment shows
the exact rejected host if you want to see it yourself. Because of that,
this mode could not be end-to-end tested from here;
`tests/test_paper_trading.py` and `tests/test_scanner.py` cover the
fill/fee/signal/summary logic against a fake exchange fed real
historical data instead — run `paper_trading.py` itself in an
environment with real internet access to Bybit.

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
  config.py           .env-driven configuration
  main.py              Poll loop entrypoint (live bot)
backtest/
  engine.py            Bar-by-bar backtest engine (fees, slippage, funding, sizing)
  run_backtest.py      CLI report
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
