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
log the strategy would have produced. The strategy went through three
iterations as more data became available (1 month, then Jan-Aug 2026 /
~307k 1-minute candles, `data/BTCUSDT_2026.csv`), and every version's
result below is the honest number, not a cherry-picked one:

1. **Pullback-to-EMA21 entry** (mean reversion): almost no signals on 1
   month of data, lost on the ones it took — real trends often never
   pull back to a fast EMA.
2. **Donchian breakout entry** (close beyond prior 20-bar high/low): far
   more signals, but on the full 7-month dataset it had a ~32-34% win
   rate *at every confidence tier* and profit factor < 1 (net losing).
   Several principled variants (2-bar breakout confirmation, wider ATR
   trail, larger partial-TP) were tested and none fixed it — a fresh
   N-bar extreme on this symbol/timeframe gets wicked through and
   reversed often enough to erase the edge.
3. **Current: EMA9/EMA21 crossover entry**, still gated by the same
   higher-timeframe trend filter. Instead of waiting for a new extreme,
   it enters as soon as short-term momentum turns to agree with the
   already-confirmed HTF trend, which caught the real multi-week 2026
   BTC trends (e.g. the ~87.6k → ~62.8k Jan-Jun downtrend) that the
   breakout version mostly missed or chopped through.

The full pipeline (current version):

1. **Regime filter (1h):** only trade in the direction of the 1h trend
   — EMA50 vs EMA200 + ADX(14) ≥ 25 (Wilder's standard "trending market"
   threshold), and the regime must have held for ≥3 consecutive 1h bars
   before being trusted (filters fresh/fake regime flips).
2. **Entry (15m):** EMA9 crosses EMA21 in the direction of the HTF
   trend, with above-average volume.
3. **Exit:** initial stop at 2×ATR(14, 15m). Half the position closes at
   +2R and the stop moves to breakeven. The remainder trails with a
   3×ATR chandelier stop, or exits immediately if the 1h trend flips
   against the position.
4. **Confidence (0-100):** built from HTF ADX strength and volume ratio.
   Maps to a position-size tier in `bot/risk.py` — never "all-in," even
   at the top tier.

**Results on the full Jan-Aug 2026 BTCUSDT dataset** (the statistically
meaningful one — 7 months, 42 trades): 38.1% win rate, profit factor
1.29, **+9.46% return, -12.18% max drawdown**. On the July-2026-only
subset alone (one choppy, mostly sideways month) it's 4 trades, all
losses, -2.31% — that month simply didn't contain a clean trend for a
trend-following system to catch, and the daily-loss circuit breaker and
per-trade risk caps kept that loss small and bounded rather than
compounding it. Both numbers are real and both are reported here on
purpose: **this strategy needs an actual trend to make money, and does
small, controlled damage when the market doesn't provide one.** 7 months
of one symbol is better evidence than 1 month, but it is still not proof
of a permanent edge — forward-test on testnet before trusting this (or
any strategy) with real funds.

Re-run it yourself:

```bash
python -m backtest.run_backtest data/BTCUSDT_2026.csv
python -m backtest.run_backtest data/BTCUSDT_2026.csv --equity 5000
python -m backtest.run_backtest data/BTCUSDT_202607.csv   # smaller, choppier reference month
```

## Applying the same logic to other coins

The live bot (`bot/scanner.py`) evaluates every symbol in `SYMBOLS`
against the identical rules above, independently, every poll cycle. A
coin only gets traded when its own 1h/15m candles satisfy the regime +
breakout + confirmation conditions; coins that don't match are simply
skipped that cycle. There is no "if it's Bitcoin do X, otherwise do Y" —
the same function (`strategy.prepare_from_ltf_htf` /
`strategy.signal_from_row`) runs for every symbol.

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

## Project layout

```
bot/
  indicators.py     EMA / RSI / ATR / ADX (pure pandas, no TA lib dependency)
  strategy.py        Signal logic shared by backtest and live bot
  risk.py             Position sizing tiers, dynamic leverage cap, safety limits
  exchange_bybit.py   pybit v5 REST wrapper (klines, orders, stop management)
  scanner.py          Per-symbol signal evaluation + open-position management
  config.py           .env-driven configuration
  main.py              Poll loop entrypoint
backtest/
  engine.py            Bar-by-bar backtest engine (fees, slippage, funding, sizing)
  run_backtest.py      CLI report
data/
  BTCUSDT_2026.csv     Jan-Aug 2026 dataset (main backtest, 7 months)
  BTCUSDT_202607.csv   July-2026-only dataset (smaller reference month)
tests/                 Unit + integration tests (indicators, strategy, risk, scanner)
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

If you deliberately want higher risk limits, edit those constants with
full understanding that it increases both potential return and the
chance of large/liquidating losses.
