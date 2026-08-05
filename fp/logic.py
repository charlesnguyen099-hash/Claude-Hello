"""FINAL_Logic_PotentialScaledLeverage — leverage scaled by trade potential.

22,533 rows, every one profitable (final_net +0.000% to +8.17%, no
negative row). What this file adds over the previous one is the leverage
chain, and all of it was recovered from the data rather than guessed:

    potential_score        0.035 .. 0.990, median 0.498
    potential_multiplier   = potential_score + 0.500      (exact, resid 5e-4)
    lev_base               17 .. 100, median 85.5
    leverage_potential     = min(lev_base x multiplier, lev_base)   96.4%
    net_leveraged_potential = final_net x leverage_potential        100.0%

WHAT potential_score ACTUALLY IS

It correlates +0.951 with the percentile rank of atr14_pct. It is a
volatility ranking: the more the instrument is moving, the higher the
"potential" of the setup.

That matters for how much weight to put on it. potential_score correlates
+0.739 with final_net, which looks like it predicts the payoff — but
atr14_pct alone correlates +0.891 with the same column, higher. The
relationship is mechanical, not predictive: the exits are ATR multiples,
so a high-ATR setup that reaches TP3.0 necessarily books a larger percent
than a low-ATR one. The score tells you how big a winner would be, not
how likely the trade is to win. The file cannot show the difference
because it contains only winners.

THE TWO ATR EFFECTS PULL OPPOSITE WAYS

lev_base falls as volatility rises (correlation -0.953 with atr14_pct):
a wider stop needs less leverage to keep it inside the margin. The
potential multiplier rises with volatility. And because the product is
capped at lev_base, the multiplier can only ever cut leverage, never add
it. So in practice: volatile setups get the full (already low) base, calm
setups get a fraction of their (high) base.

WHAT A BOT CAN AND CANNOT TAKE FROM THE FILE

final_direction is not the methods' verdict. On the 19,717 rows where the
twelve methods reached a consensus, it follows that consensus 50.4% of
the time and reverses it 49.6% — a coin flip, so the column carries no
information the methods produced. It is whichever way the trade turned
out to work. The bot trades consensus_dir instead.

final_exit_strategy is likewise the exit that won for that trade. TP3.0
is chosen in 67.7% of rows, so the bot fixes TP3.0 at entry.

The whole leverage chain IS computable at entry — it depends only on ATR
— so it is implemented exactly as the file has it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BAR_MINUTES = 30

# Exits. TP multiples are of ATR, against a 1.5 ATR stop.
EXIT_STRATEGIES = ["net_TP1.5_SL1.5", "net_TP2.0_SL1.5",
                   "net_TP3.0_SL1.5", "net_TRAILING"]
TP_MULTIPLES = {"net_TP1.5_SL1.5": 1.5, "net_TP2.0_SL1.5": 2.0,
                "net_TP3.0_SL1.5": 3.0}
SL_MULTIPLE = 1.5
TRAIL_MULTIPLE = 1.5
DEFAULT_EXIT = "net_TP3.0_SL1.5"        # the file's choice in 67.7% of rows
MAX_HOLD_BARS = 1000

# Leverage chain, recovered from the file.
LEVERAGE_MIN, LEVERAGE_MAX = 17.0, 100.0
LEV_BASE_K = 28.0                       # lev_base = clip(K / atr14_pct, 17, 100)
POTENTIAL_OFFSET = 0.50                 # multiplier = score + 0.50

# Solvency. The bot's liquidation model puts the liquidation price where
# 90% of the margin is gone; the stop must sit inside that with room to
# spare, or the position dies at 100% of margin instead of the 42% the
# stop was sized for. See solvency_cap().
LIQ_MARGIN_FRACTION = 0.90
SOLVENCY_BUFFER = 1.30

# Per-trade potential. Vote margin at which conviction counts as full,
# and the smallest fraction of the file's leverage a minimum-conviction
# setup is allowed to take.
CONVICTION_FULL_MARGIN = 3.0
CONVICTION_FLOOR = 0.60

FEE_ROUND_TRIP = 0.0025                 # taker, at the file's leveraged rate
MAKER_ROUND_TRIP = 0.0004               # resting the order instead

# potential_score is a percentile, so it needs a distribution to rank
# against. These are the ATR percentiles of BTCUSDT 30m bars over
# 2025-2026, so a live bar can be scored without waiting for history.
ATR_PCT_REFERENCE = np.array([
    0.1333, 0.1588, 0.1780, 0.1940, 0.2098, 0.2243, 0.2399, 0.2555, 0.2722,
    0.2901, 0.3099, 0.3310, 0.3542, 0.3807, 0.4121, 0.4488, 0.4978, 0.5748,
    0.6966,
])


def potential_score(atr14_pct: float,
                    reference: np.ndarray = ATR_PCT_REFERENCE) -> float:
    """How much this instrument is moving, as a 0..1 percentile.

    Correlates +0.951 with the file's own column.
    """
    if not np.isfinite(atr14_pct) or atr14_pct <= 0:
        return 0.5
    # searchsorted over the 5%..95% quantiles gives the percentile directly.
    return float(np.clip(np.searchsorted(reference, atr14_pct)
                         / (len(reference) + 1), 0.0, 1.0))


def potential_multiplier(score: float) -> float:
    """The file's exact relation: multiplier = score + 0.5."""
    return score + POTENTIAL_OFFSET


def conviction_score(votes: int, vote_margin: int) -> float:
    """How strongly the twelve methods agreed on THIS setup, 0..1.

    This is the only quantity in the whole chain that belongs to the
    individual trade rather than to the instrument. Two trades on the
    same coin at the same minute, one long and one short, get different
    conviction; they get identical atr14_pct.
    """
    if votes <= 0 or vote_margin <= 0:
        return 0.0
    return float(np.clip(vote_margin / CONVICTION_FULL_MARGIN, 0.0, 1.0))


def conviction_haircut(votes: int, vote_margin: int,
                       floor: float = CONVICTION_FLOOR) -> float:
    """Per-trade sizing factor in [CONVICTION_FLOOR, 1.0].

    Applied as a MULTIPLIER on the leverage the file's chain produced, so
    it can only ever cut. That shape is deliberate, and it is the second
    attempt: blending conviction into the score as an average was tried
    first and measured, and it RAISED mean leverage from 64.3x to 65.3x
    (a low ATR rank paired with high conviction scored above the ATR rank
    alone). Raising leverage on the strength of a factor with no measured
    edge is exactly the wrong trade, and P&L per $1 of margin got worse in
    five of six year/fee cells. As a haircut the direction is guaranteed.

    MEASURED, NOT ASSUMED. On 15,617 consensus trades across 2025-2026,
    conviction does not predict whether a trade wins:

        methods fired   1: 33.6% win   2: 35.1%   3: 33.8%   4: 32.8%   5: 33.3%
        correlation of votes with gross return   -0.0011
        correlation of vote margin with it       +0.0009

    and the ranking flips between years -- in 2025 three-plus votes beat
    one vote, in 2026 one vote beat three-plus. A factor that changes sign
    across years is worse than no factor.

    So this is a risk preference, not an edge: it stands smaller on the
    setups the methods agreed on least. If the conviction reading is
    meaningless -- and the measurement says it is -- the cost is sizing
    some good trades smaller, never sizing a bad one larger.

    AND IT WAS CHECKED AGAINST THE OBVIOUS NULL. At the default floor the
    haircut cuts mean leverage 23.3%, which alone improves P&L per $1 of
    margin from -15.48% to -11.85% at taker fees. That improvement is
    NOT selection. Cutting every trade by a flat 0.7673 -- same mean
    leverage, no conviction anywhere -- gives -11.88%. The difference
    between choosing which trades to cut and cutting all of them equally
    is +0.027%, and it flips sign by year (+0.086% in 2025, -0.076% in
    2026). The haircut's correlation with the trade's own net return is
    +0.0012.

    Read plainly: this term is a deleveraging knob wearing a per-trade
    shape. It is kept because the shape is the right one and it cannot do
    harm, not because the data says agreement is worth anything. Set the
    floor to 1.0 to switch it off entirely.
    """
    floor = float(np.clip(floor, 0.0, 1.0))
    return float(floor + (1.0 - floor) * conviction_score(votes, vote_margin))


def lev_base(atr14_pct: float) -> float:
    """Base leverage, falling as volatility rises so the stop stays inside
    the margin. Fitted against the file's column (correlation -0.953 with
    atr14_pct)."""
    if not np.isfinite(atr14_pct) or atr14_pct <= 0:
        return LEVERAGE_MAX
    return float(np.clip(LEV_BASE_K / atr14_pct, LEVERAGE_MIN, LEVERAGE_MAX))


# Win rate each exit actually achieved on BTCUSDT 30m, 2025 + 2026, on the
# same twelve-method consensus the bot trades. Every one of them sits a
# hair above the breakeven rate its own payoff ratio implies -- 35.1% for
# TP3.0 against a breakeven of 33.3%, 43.9% for TP2.0 against 42.9%, 51.3%
# for TP1.5 against 50.0%. That is the martingale signature: the gross
# edge is nearly zero and the fee decides the outcome.
MEASURED_WIN_RATE = {
    "net_TP1.5_SL1.5": 0.513,
    "net_TP2.0_SL1.5": 0.439,
    "net_TP3.0_SL1.5": 0.351,
    "net_TRAILING": 0.349,
}


def expectancy(atr14_pct: float, exit_name: str = DEFAULT_EXIT,
               fee: float = FEE_ROUND_TRIP,
               win_rate: float | None = None) -> float:
    """Expected return of this trade per unit of notional, after fees.

    win * P(win) - loss * P(loss) - fee, with the targets in ATR terms.
    Leverage is deliberately absent: it multiplies wins, losses and fees
    alike, so it cannot turn a negative expectancy positive. It only
    decides how fast the account gets there.

    This is the "only take trades that clear the fee" test, done properly.
    Filtering on ATR alone does not work and was measured: at taker fees
    net stays flat at -0.26% (2025) and -0.20% (2026) per trade at every
    ATR threshold, because ATR scales the win and the loss together.
    What has to clear the fee is the EDGE, not the move.
    """
    p = MEASURED_WIN_RATE.get(exit_name, 0.35) if win_rate is None else win_rate
    tp = TP_MULTIPLES.get(exit_name, TRAIL_MULTIPLE)
    a = atr14_pct / 100.0
    return p * tp * a - (1.0 - p) * SL_MULTIPLE * a - fee


def min_atr_for_edge(exit_name: str = DEFAULT_EXIT, fee: float = FEE_ROUND_TRIP,
                     win_rate: float | None = None) -> float:
    """Smallest atr14_pct at which expectancy() turns positive.

    At taker fees and the measured 35.1% win rate this is 3.33%, and
    BTCUSDT 30m never reached it in two years (max 2.42%). Read plainly:
    at taker fees this logic has no positive-expectancy trade, and no
    filter over ATR, votes or anything else changes that. At maker fees
    the same figure is 0.53%, which about a quarter of bars clear.
    """
    p = MEASURED_WIN_RATE.get(exit_name, 0.35) if win_rate is None else win_rate
    tp = TP_MULTIPLES.get(exit_name, TRAIL_MULTIPLE)
    edge = p * tp - (1.0 - p) * SL_MULTIPLE
    if edge <= 0:
        return float("inf")
    return 100.0 * fee / edge


def solvency_cap(atr14_pct: float) -> float:
    """Highest leverage at which the 1.5-ATR stop still sits INSIDE the
    liquidation price.

    Why this is needed. The stop is a fixed multiple of ATR, so the move
    it sits at is 1.5 x atr_pct. Liquidation is a fixed fraction of
    margin, so the move IT sits at is 0.90 / leverage. Setting the first
    below the second gives

        1.5 x atr_pct/100 <= 0.90 / lev     ->    lev <= 60 / atr_pct

    and lev_base is 28/atr_pct, comfortably below it — so through the
    normal range the file's chain is already solvent and this cap never
    binds. What breaks it is the 17x FLOOR: once atr_pct passes 3.53%,
    28/atr_pct falls under 17, the clip lifts leverage back to 17, and
    the stop lands past the liquidation price. The position then dies at
    100% of margin instead of the 42% the stop was sized for.

    Measured across the ATR range, that is the only place the chain is
    unsafe, so this is the only place the cap acts. It is a function of
    ATR like every other term, so leverage stays continuous and
    volatility-driven — nothing here is a fixed number of x.
    """
    if not np.isfinite(atr14_pct) or atr14_pct <= 0:
        return LEVERAGE_MAX
    sl_move = SL_MULTIPLE * atr14_pct / 100.0
    return float(LIQ_MARGIN_FRACTION / (SOLVENCY_BUFFER * sl_move))


def leverage_potential(atr14_pct: float, cap: float | None = None,
                       votes: int | None = None,
                       vote_margin: int | None = None,
                       conviction_floor: float = CONVICTION_FLOOR) -> dict:
    """The full chain: base, score, multiplier, and the final leverage.

    The file's own chain runs first and unchanged: potential_score is its
    volatility rank, multiplier = score + 0.5, product capped at the base
    so a high multiplier never raises leverage above what the stop allows.

    Pass `votes` and `vote_margin` and a per-trade term is then applied on
    top, as a haircut. That is the only quantity here belonging to the
    individual trade rather than to the instrument: the file's score is
    identical for a long and a short taken on the same bar, conviction is
    not. Being a haircut, it can only reduce — see conviction_haircut().

    Last comes the solvency bound, which only ever binds where the 17x
    floor would have put the stop past the liquidation price.

    `tradeable` is False when even 1x cannot keep the stop inside the
    liquidation price — an instrument moving more than ~46% per 30m bar.
    The bot skips those rather than clamping to 1x and taking a position
    it knows is unsound.
    """
    base = lev_base(atr14_pct)
    score = potential_score(atr14_pct)
    mult = potential_multiplier(score)
    lev = min(base * mult, base)                    # the file's chain, as written
    haircut = 1.0
    if votes is not None and vote_margin is not None:
        haircut = conviction_haircut(votes, vote_margin, conviction_floor)
        lev *= haircut                              # per-trade, cuts only
    solvent = solvency_cap(atr14_pct)
    lev = min(lev, solvent)
    if cap is not None:
        lev = min(lev, cap)
    return {"lev_base": base, "potential_score": score,
            "potential_multiplier": mult, "solvency_cap": solvent,
            "conviction": (conviction_score(votes, vote_margin)
                           if votes is not None and vote_margin is not None
                           else float("nan")),
            "conviction_haircut": haircut,
            "leverage": max(1.0, lev), "tradeable": lev >= 1.0}


def simulate_exit(high, low, close, i: int, direction: int, atr: float,
                  exit_name: str = DEFAULT_EXIT,
                  fee: float = FEE_ROUND_TRIP) -> tuple[float, int, str]:
    """Run one trade to its exit. Returns (net return, bars held, reason).

    On a bar whose range covers both the stop and the target, the stop is
    taken — an ambiguous bar never resolves in the trade's favour.
    """
    entry = close[i]
    n = len(close)
    sl = entry - direction * SL_MULTIPLE * atr

    if exit_name == "net_TRAILING":
        best = entry
        for j in range(i + 1, min(i + 1 + MAX_HOLD_BARS, n)):
            if (low[j] <= sl) if direction > 0 else (high[j] >= sl):
                return -SL_MULTIPLE * atr / entry - fee, j - i, "stop"
            if direction > 0:
                best = max(best, high[j])
                trail = best - TRAIL_MULTIPLE * atr
                if low[j] <= trail:
                    return (trail - entry) / entry - fee, j - i, "trail"
            else:
                best = min(best, low[j])
                trail = best + TRAIL_MULTIPLE * atr
                if high[j] >= trail:
                    return (entry - trail) / entry - fee, j - i, "trail"
    else:
        tp = entry + direction * TP_MULTIPLES[exit_name] * atr
        for j in range(i + 1, min(i + 1 + MAX_HOLD_BARS, n)):
            if (low[j] <= sl) if direction > 0 else (high[j] >= sl):
                return -SL_MULTIPLE * atr / entry - fee, j - i, "stop"
            if (high[j] >= tp) if direction > 0 else (low[j] <= tp):
                return TP_MULTIPLES[exit_name] * atr / entry - fee, j - i, "target"

    j = min(i + MAX_HOLD_BARS, n - 1)
    return (close[j] - entry) / entry * direction - fee, j - i, "timeout"


def to_bars(df_1m: pd.DataFrame, minutes: int = BAR_MINUTES) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    return (df_1m.set_index("datetime").resample(f"{minutes}min")
            .agg(agg).dropna().reset_index())
