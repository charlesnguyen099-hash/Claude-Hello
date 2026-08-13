# Bybit USDT-perpetual paper trader

Virtual $10, real Bybit prices, no API key and no order ever placed.

```bat
pip install -r requirements.txt
python -m fp.test_engine        # the logic
python -m fp.test_bot           # the broker
python run_bot.py --max-margin-pct 100
```

---

## The one thing that decides whether any of this can work

A trade has to clear the exchange bill: 0.055% taker in, 0.055% taker
out, plus funding. Whether that is easy or impossible depends entirely on
how big the target is next to the bill.

A target set at `k x` the **one-minute** volatility is impossible. BTC's
one-minute sigma is about 0.04%, so a 2-sigma target is 0.08% against a
0.110% round trip: the trade is behind before it starts. Measured on real
BTC bars, the break-even win rates that implies are:

| target | break-even win rate needed |
|---|---|
| 2 sigma (1m) | **94.0%** |
| 3 sigma (1m) | 86.7% |
| 6 sigma (1m) | 76.7% |

Nothing wins 94% of the time. Every earlier version of this repo failed
for that reason and no amount of feature engineering could have fixed it
— the arithmetic was against them.

Price wanders like `sqrt(time)`, so the target has to as well. A shape is
`(k, m, minutes)` meaning `target = k x sigma_1m x sqrt(minutes)`. At a
four-hour hold the same BTC sigma gives a 2-sigma target of 1.24%, and
break-even lands near a coin flip:

| shape | break-even win rate needed |
|---|---|
| 2.0 / 1.5 / 120m | 56.8% |
| 3.0 / 2.0 / 240m | 46.9% |
| 4.0 / 3.0 / 480m | 46.3% |

That is a gap a signal can actually close.

---

## How the logic is built

**`fp/labels.py` — the answer sheet.** With the future in hand, mark every
triple-barrier trade in the past that profits after the whole exchange
bill. The exit is a target, a stop and a time limit, all placeable the
moment the trade opens — "sell at the top" is a wish, not a trade. The fee
lives *inside* the label, so a target the bill eats is a loss, not a small
win.

**`fp/factors.py` — 153 numbers per bar.** Three families:

- *price and flow* — returns, volatility, range position, volume,
  order-flow proxy, candle shape, at nine lookbacks from 1 to 120 minutes,
  so the model finds the horizon instead of being told it
- *state* — distance to rolling extremes in volatility units, VWAP
  stretch, run length, volatility regime, session
- *cross-section* — what the **other nine coins** are doing at that
  instant: this coin's rank, the board's median move, the dispersion, the
  residual, the breadth. A coin rising alone and a coin carried by the
  tide used to be the same row.

Every factor is a ratio, a z-score or a rank, so BTC at $100,000 and a
$0.02 coin present the same numbers. That is the only reason **one** model
can serve all ten.

**`fp/engine.py` — potential.** One classifier per `(target, stop, hold,
side)`, fitted across all ten coins at once, isotonically calibrated on
rows the trees never saw, and lower-bounded by that calibration's own
sample size. Its win probability meets the shape's break-even to give one
number:

```
POTENTIAL = 100 x (p - p_be) / (1 - p_be)
```

0 = break-even, 100 = cannot lose by its own barriers. Same scale for a
one-hour scalp and a twelve-hour swing, so they can be ranked against each
other and share one account. **The stake reads straight off it**: 60/100
commits 60% of equity, 100/100 may take all of it — with half-Kelly and a
ruin cap on top so a big score on a wide stop still cannot bet the
account.

---

## How it is judged

- **Walk-forward.** Train on the past, trade the future, never the
  reverse.
- **Independent trades only.** Overlapping barrier windows watch the same
  price path; counting both inflates `t`. This correction alone took an
  earlier result from `t = 7.81` to `t = 2.26`.
- **A rotation null.** Roll the prediction series — same drift, same
  volatility, same long/short balance, alignment destroyed. A logic that
  cannot beat a rolled copy of itself has found nothing.
- **Bonferroni over everything attempted**, not over the survivors. 32
  shape/side combinations means `|t| >= 3.2`.
- **A shape ships only if it was profitable in every fold**, with at least
  20 independent trades in each.

`fp/run_engine.py` writes the verdict to `fp/engine_shapes.json`.
`fp/train_engine.py` fits **only** what survived. If nothing survived it
writes an empty model and the bot opens nothing — which is a result, not a
fault, and the banner says so.

```bat
python -m fp.run_engine       # walk-forward: what survives
python -m fp.train_engine     # fit the survivors
python run_bot.py --max-margin-pct 100
```

---

## The bot

- One position per **coin x shape x side**, so a twelve-hour swing does
  not lock a coin out of every one-hour shape.
- Leverage is **solved** per trade against its own stop, its hold and the
  volatility drag — never chosen. `leverage.solvent_leverage` also caps it
  so the stop always fires **before** liquidation, and refuses the setup
  when it cannot. (Without that guard a wide-stop shape returned 3x with
  the stop four times past the liquidation price: a sized loss becoming a
  total one.)
- Costs come from the exchange, not a flag: taker both sides plus each
  symbol's live funding rate, signed — a short *collects* positive
  funding.
- Three threads: one keeps a standing signal per slot, one re-prices open
  positions every second so targets and stops fire promptly, one prints
  the dashboard. Ctrl+C prints the session summary.

### Files

| file | what it is |
|---|---|
| `fp/data.py` | the bar cache and the exchange bill |
| `fp/barriers.py` | the barrier arithmetic, no dependencies |
| `fp/factors.py` | 153 factors, price / state / cross-section |
| `fp/labels.py` | every past trade that profits after fees |
| `fp/engine.py` | classifier, calibration, potential score |
| `fp/run_engine.py` | walk-forward and shape selection |
| `fp/train_engine.py` | fit the survivors, write the artefacts |
| `fp/live_engine.py` | live factor build and scoring |
| `fp/leverage.py` | the leverage solver and the solvency cap |
| `fp/bot.py` | the paper broker |
| `run_bot.py` | the command line |

### A warning worth keeping

Perfect selection on these bars earns **+0.70%/trade, +2339% over two
weeks**. That number is real and it is why hindsight always looks easy. It
is also unreachable: it is computed from the answer. The gap between it
and what a past-only logic earns is the whole problem, and every number in
this repo is reported next to its own null so that gap stays visible.
