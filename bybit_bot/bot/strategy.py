"""Trend-following EMA-cross strategy.

Design history — this module went through three versions while being
validated against real BTCUSDT data (first 1 month, then 7 months
Jan-Aug 2026, ~307k 1-minute candles; see backtest/run_backtest.py):

1. A mean-reversion "pullback to EMA21" entry produced almost no signals
   on 1 month of data and lost on the few it took — in a real trend,
   price often never comes back to touch a fast EMA.
2. A Donchian-style breakout entry (close beyond the prior 20-bar
   high/low) produced far more signals, but backtested over the full
   7-month 2026 dataset it had a ~32-34% win rate *at every confidence
   tier* and a profit factor < 1 (net losing), including several
   principled variants (2-bar breakout confirmation, wider ATR trail,
   larger partial-TP target) — none fixed it. Conclusion: on this
   symbol/timeframe, buying a fresh N-bar extreme walks into short-term
   mean-reversion/stop-hunt wicks often enough to erase the edge.
3. **Current: EMA9/EMA21 crossover** in the direction of the HTF trend.
   Instead of waiting for price to make a new extreme (breakout) or
   come back to a level that may never return (pullback), this enters
   right as short-term momentum turns to agree with the HTF trend —
   which on the 7-month dataset caught the real multi-week trends (e.g.
   the Jan-Jun 2026 BTC downtrend from ~87.6k to ~62.8k) that the
   breakout version mostly missed or chopped through. Result on that
   dataset: 42 trades, 38.1% win rate, profit factor 1.29, +9.46%
   return, -12.18% max drawdown. See "Applying this to other coins" in
   README.md for why this is not a claim that generalizes automatically.

The rest of the pipeline is unchanged across all three versions:

- HIGHER TIMEFRAME (1h) defines the *regime*: only trade in the direction
  of the 1h trend. Trend = EMA50 vs EMA200 slope/position + ADX(14) >
  ADX_MIN (a flat/choppy market is excluded, not force-traded), and the
  regime must have held for TREND_MIN_BARS consecutive 1h bars.
- Every entry carries a `confidence` score (0-100) built from trend
  strength (ADX) and volume. Confidence maps to a position-size tier in
  bot/risk.py — there is no scenario that produces 100% confidence,
  because no such setup exists; the tiers are capped well below
  "all-in" on purpose (see risk.py).
- Exits use an initial ATR stop, a partial take-profit at 2R that moves
  the remaining stop to breakeven, and after that an ATR chandelier
  trailing stop on the remainder — so winners are allowed to run with
  the trend instead of being capped at a fixed target.

This module is shared, unmodified, between the backtester and the live
bot so backtest results and live behaviour cannot drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bot import indicators as ind
from bot import risk as _risk

ADX_MIN = 25.0  # Wilder's textbook "trending market" threshold
TREND_MIN_BARS = 3  # HTF regime must have held for this many 1h bars before trading it

# LTF indicator periods, in 1-minute bars. The entry signal is evaluated
# on every closed 1m candle (fastest possible reaction), but EMA9/EMA21/
# ATR14/RSI14 computed directly, natively on 1m bars turned out to be
# almost pure microstructure noise: backtested on both 2025 and 2026 (1m
# EMA-cross, no other change), results ranged from -44% to -21% depending
# only on how the fee filter was tuned, with no setting working on both
# years — the crossings just don't carry real trend information at that
# granularity. Scaling the periods up to the real time span that worked
# at 15m (9 bars * 15m = 135m, 21 bars * 15m = 315m, etc.) keeps the
# proven smoothing while still checking for a new signal every minute.
#
# ATR needed a different fix than "just use a longer period": True Range
# is a per-bar-size measure (a 1-minute candle's range is always much
# smaller than a 15-minute candle's, no matter the lookback length), so
# ATR_LEN alone can't fix it — see indicators.rolling_block_atr, which
# measures the range of a rolling ATR_BLOCK_MINUTES-wide window instead
# of the native 1m bar, then EMA-smooths ATR_LEN of those block ranges,
# reproducing the original 15m ATR14's real-world units at 1m cadence.
RSI_LEN = 210
ATR_BLOCK_MINUTES = 15
ATR_LEN = 14
EMA_FAST = 135
EMA_MID = 315
EMA_SLOW_HTF = 50
EMA_TREND_HTF = 200
VOL_MA_LEN = 300
ATR_INIT_MULT = 2.0
ATR_TRAIL_MULT = 3.0
TP1_R_MULT = 2.0

# Anti-chase filter: don't buy something that's already overbought / sell
# something that's already oversold — this is what "đu đỉnh" (buying a
# top) / "đu đáy" (selling a bottom) looks like mechanically: piling onto
# a move that has already run hard instead of catching one that's
# starting. An earlier version of this filter instead measured distance
# from the 1h EMA50 in ATR units, but on the full 2025+2026 backtest that
# rejected most of the good continuation trades too (a slow trend average
# naturally trails far behind price during any real multi-week trend, so
# "far from EMA50" doesn't distinguish a good continuation entry from a
# genuine blow-off top). Plain RSI(14) overbought/oversold on the entry
# timeframe does distinguish them and didn't have that side effect.
RSI_OVERBOUGHT = 70.0
RSI_OVERSOLD = 30.0

# Fee-awareness: Bybit USDT-perpetual taker fee is ~0.055% per fill. A
# round trip (entry + exit) plus assumed slippage costs roughly this
# much; TP1 — the smallest profit target any trade can bank — must clear
# it by a wide safety margin, or a "win" on paper can still be a loss
# after real costs. See bot/risk.py ROUND_TRIP_COST_PCT.
MIN_TP1_TO_COST_RATIO = 8.0

# Don't chase the market. Rather than entering at the signal bar's close,
# rest the entry PULLBACK_ATR_MULT ATR below price (above, for a short)
# and take the trade only if price actually comes back to it within
# PENDING_MAX_BARS. If it never does, the trade is simply skipped and the
# capital stays free.
#
# This is the mechanical form of "don't buy the top / don't sell the
# bottom": an EMA cross fires *after* a move has already started, so
# entering at that close is by construction entering into extension.
#
# Be careful reading the evidence for this one. A simplified equal-sized
# fixed-target model (research/two_stage_entry.py) suggested waiting for
# a pullback more than halves the loss on BOTH years. Re-run inside the
# real engine — confidence-scaled sizing, dynamic leverage, partial TP,
# breakeven, chandelier trail, stale exit — that did not hold up
# (research/pullback_sweep.py, full-year returns):
#
#   pullback   2026     2025    combined   worst DD
#      0.00   -12.78%  +6.01%    -6.77%    -12.78%
#      0.50    -8.85%  -7.14%   -15.99%     -8.85%
#      0.75    -6.31%  -5.14%   -11.45%     -7.50%
#      1.00    -2.35%  -5.68%    -8.03%     -6.45%
#      2.00    -3.45%  -6.08%    -9.53%     -6.13%
#
# No setting is profitable on both years, and with only 3-26 trades per
# year the differences in return are inside the noise — the headline
# "2026 improved from -12.78% to -2.35%" is a small-sample result, not a
# demonstrated edge, and it came at the cost of 2025's positive year.
#
# The pullback is kept anyway, for two reasons that do survive: it is
# the explicit "don't chase extension" behaviour that was asked for, and
# it roughly halves the worst drawdown (-12.78% -> -6.45%), which is a
# risk property rather than a return claim. It does NOT make the
# strategy profitable. See README.
PULLBACK_ATR_MULT = 1.0
PENDING_MAX_BARS = 120  # cancel an unfilled entry after this many 1m bars

# Rejected alternative, kept documented so it isn't re-tried blind:
# "capture the dip first, then enter" (open a counter-direction trade
# with a tight target, flip into the signal after it fills). The adverse
# excursion it aims at is real — median 2.0-2.1 ATR, ~17.5% of equity at
# 25x leverage, on both years — but trading it is a coin flip: the
# counter legs won 42.5% (2026) / 40.8% (2025) and cost an extra round
# trip each, taking the totals from -17.53%/-16.12% to -24.29%/-35.46%.
# The dip is real; its *timing* is not predictable at signal time.


@dataclass(frozen=True)
class Signal:
    side: str  # "long" or "short"
    entry: float          # limit price the order rests at
    signal_price: float   # close of the bar that produced the signal
    stop: float
    take_profit_1: float
    take_profit_2: float
    confidence: float
    reason: str


def compute_htf(df_1h: pd.DataFrame) -> pd.DataFrame:
    out = df_1h.copy()
    out["ema50"] = ind.ema(out["close"], EMA_SLOW_HTF)
    out["ema200"] = ind.ema(out["close"], EMA_TREND_HTF)
    out["adx14"] = ind.adx(out, 14)

    up = (out["ema50"] > out["ema200"]) & (out["close"] > out["ema50"]) & (out["adx14"] >= ADX_MIN)
    down = (out["ema50"] < out["ema200"]) & (out["close"] < out["ema50"]) & (out["adx14"] >= ADX_MIN)
    trend_raw = np.select([up, down], ["up", "down"], default="flat")
    trend_raw = pd.Series(trend_raw, index=out.index)

    # Only trust a regime once it has held for TREND_MIN_BARS consecutive
    # bars — filters out trading directly into a fresh, possibly-fake
    # regime flip. Uses only past bars (shift >= 1), no lookahead.
    stable = pd.Series(True, index=out.index)
    for k in range(1, TREND_MIN_BARS):
        stable &= trend_raw == trend_raw.shift(k)
    out["trend"] = np.where(stable, trend_raw, "flat")
    return out


def compute_ltf(df_ltf: pd.DataFrame) -> pd.DataFrame:
    out = df_ltf.copy()
    out["ema9"] = ind.ema(out["close"], EMA_FAST)
    out["ema21"] = ind.ema(out["close"], EMA_MID)
    out["rsi14"] = ind.rsi(out["close"], RSI_LEN)
    out["atr14"] = ind.rolling_block_atr(out, ATR_BLOCK_MINUTES, ATR_LEN)
    out["vol_ma20"] = out["volume"].rolling(VOL_MA_LEN, min_periods=VOL_MA_LEN).mean()
    return out


def merge_htf_trend(df_ltf: pd.DataFrame, df_htf: pd.DataFrame) -> pd.DataFrame:
    """Attach the HTF trend to each LTF bar using only HTF bars that have
    already *closed* by that LTF bar's timestamp (merge_asof backward) —
    no lookahead into a still-forming HTF candle.
    """
    left = df_ltf.sort_values("datetime").reset_index(drop=True)
    right = df_htf[["datetime", "trend", "adx14", "ema50", "ema200"]].sort_values("datetime")
    right = right.rename(columns={"adx14": "htf_adx14"})
    merged = pd.merge_asof(left, right, on="datetime", direction="backward")
    merged["trend"] = merged["trend"].fillna("flat")
    return merged


def build_signal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized entry-condition columns on an LTF dataframe that already
    has compute_ltf() + merge_htf_trend() applied.

    Entry = EMA9/EMA21 crossover in the direction of the HTF trend, with
    above-average volume. See the module docstring for why this replaced
    an earlier breakout-based entry.
    """
    out = df.copy()

    cross_up = (out["ema9"].shift(1) <= out["ema21"].shift(1)) & (out["ema9"] > out["ema21"])
    cross_down = (out["ema9"].shift(1) >= out["ema21"].shift(1)) & (out["ema9"] < out["ema21"])
    vol_confirm = out["volume"] > out["vol_ma20"]
    not_overbought = out["rsi14"] <= RSI_OVERBOUGHT
    not_oversold = out["rsi14"] >= RSI_OVERSOLD

    out["long_setup"] = (out["trend"] == "up") & cross_up & vol_confirm & not_overbought
    out["short_setup"] = (out["trend"] == "down") & cross_down & vol_confirm & not_oversold

    vol_ratio = (out["volume"] / out["vol_ma20"]).clip(upper=3.0)
    adx_score = (out["htf_adx14"].clip(upper=60.0) / 60.0) * 60.0
    vol_score = (vol_ratio.clip(lower=0) / 3.0) * 35.0

    confidence = (adx_score + vol_score).clip(0, 95)
    out["confidence_long"] = confidence
    out["confidence_short"] = confidence

    return out


def signal_from_row(row: pd.Series) -> Signal | None:
    """Given one fully-indicator-populated LTF row (as produced by
    build_signal_columns), return an entry Signal if this bar's close is
    an entry trigger, else None. Stop/TP use ATR so they scale with each
    symbol's own volatility instead of a fixed pip amount.
    """
    atr = row["atr14"]
    if not np.isfinite(atr) or atr <= 0:
        return None

    if bool(row.get("long_setup", False)):
        signal_price = float(row["close"])
        entry = signal_price - PULLBACK_ATR_MULT * atr
        stop = entry - ATR_INIT_MULT * atr
        risk = entry - stop
        tp1 = entry + TP1_R_MULT * risk
        if not _clears_fees(entry, tp1):
            return None
        return Signal(
            side="long",
            entry=entry,
            signal_price=signal_price,
            stop=stop,
            take_profit_1=tp1,
            take_profit_2=entry + 8.0 * risk,
            confidence=float(row["confidence_long"]),
            reason="uptrend, EMA9 crossed above EMA21, volume confirm, limit entry on pullback",
        )

    if bool(row.get("short_setup", False)):
        signal_price = float(row["close"])
        entry = signal_price + PULLBACK_ATR_MULT * atr
        stop = entry + ATR_INIT_MULT * atr
        risk = stop - entry
        tp1 = entry - TP1_R_MULT * risk
        if not _clears_fees(entry, tp1):
            return None
        return Signal(
            side="short",
            entry=entry,
            signal_price=signal_price,
            stop=stop,
            take_profit_1=tp1,
            take_profit_2=entry - 8.0 * risk,
            confidence=float(row["confidence_short"]),
            reason="downtrend, EMA9 crossed below EMA21, volume confirm, limit entry on pullback",
        )

    return None


def limit_order_filled(side: str, limit_price: float, bar_high: float, bar_low: float) -> bool:
    """Would a resting entry order at `limit_price` have filled on a bar
    with this high/low? A long rests below the market, a short above.
    """
    return (bar_low <= limit_price) if side == "long" else (bar_high >= limit_price)


def _clears_fees(entry: float, tp1: float) -> bool:
    """Reject a signal whose smallest profit target wouldn't clear real
    Bybit round-trip trading costs by a comfortable margin — a "win" on
    paper must still be a win after fees + slippage.
    """
    if entry <= 0:
        return False
    gain_pct = abs(tp1 - entry) / entry
    return gain_pct >= _risk.ROUND_TRIP_COST_PCT * MIN_TP1_TO_COST_RATIO


def prepare(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: 1m OHLCV -> entry frame (native 1m — entries are
    analyzed and taken on every closed 1-minute candle, per the user's
    explicit request) with HTF(1h) trend and signal columns attached.
    The LTF frame needs no resampling since the input already is 1m; only
    the HTF (1h) trend filter is built by resampling up.
    """
    df = df_1m.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")

    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    df_1h = df.resample("1h").agg(agg).dropna().reset_index()
    df_1m_flat = df.reset_index()
    return prepare_from_ltf_htf(df_1m_flat, df_1h)


def prepare_from_ltf_htf(df_ltf_raw: pd.DataFrame, df_1h_raw: pd.DataFrame) -> pd.DataFrame:
    """Same pipeline as prepare(), but starting from native LTF (1m)/1h
    OHLCV already fetched from the exchange's kline endpoints. This is the
    single function both the backtester (via prepare()) and the live bot
    call, so behaviour cannot drift between them.
    """
    df_htf = compute_htf(df_1h_raw)
    df_ltf = compute_ltf(df_ltf_raw)
    merged = merge_htf_trend(df_ltf, df_htf)
    merged = build_signal_columns(merged)
    return merged
