"""Trend-following breakout strategy.

Design (derived from backtesting BTCUSDT 1m futures data, July 2026 —
see backtest/run_backtest.py for the report this is based on; a first
version of this module used a mean-reversion "pullback to EMA21" entry,
which on this dataset produced almost no signals and lost on the few it
took, because in a real trend price often never comes back to touch a
fast EMA. It was replaced with a breakout-continuation entry, which is
the classical trend-following approach and matched this data far better):

1. HIGHER TIMEFRAME (1h) defines the *regime*: only trade in the direction
   of the 1h trend. Trend = EMA50 vs EMA200 slope/position + ADX(14) >
   ADX_MIN (a flat/choppy market is excluded, not force-traded).
2. LOWER TIMEFRAME (15m) defines the *entry*: a Donchian-style breakout —
   price closes beyond its own N-bar high/low channel in the direction of
   the HTF trend, with volume confirmation and short/mid EMA alignment.
   This buys/sells strength in the direction of the trend instead of
   trying to catch a pullback that may never come.
3. Every entry carries a `confidence` score (0-100) built from trend
   strength (ADX), EMA alignment, breakout strength (in ATR units) and
   volume confirmation. Confidence maps to a position-size tier in
   bot/risk.py — there is no scenario that produces 100% confidence,
   because no such setup exists; the tiers are capped well below
   "all-in" on purpose (see risk.py).
4. Exits use an initial ATR stop, a partial take-profit at 2R that moves
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

ADX_MIN = 25.0  # Wilder's textbook "trending market" threshold
TREND_MIN_BARS = 3  # HTF regime must have held for this many 1h bars before trading it
RSI_LEN = 14
ATR_LEN = 14
EMA_FAST = 9
EMA_MID = 21
EMA_SLOW_HTF = 50
EMA_TREND_HTF = 200
VOL_MA_LEN = 20
BREAKOUT_LOOKBACK = 20
ATR_INIT_MULT = 2.0
ATR_TRAIL_MULT = 3.0
TP1_R_MULT = 2.0


@dataclass(frozen=True)
class Signal:
    side: str  # "long" or "short"
    entry: float
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
    out["atr14"] = ind.atr(out, ATR_LEN)
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

    Entry = Donchian breakout: close beyond the prior N-bar high/low
    channel (shift(1) so the channel never includes the breakout bar
    itself), in the direction of the HTF trend, with EMA alignment and
    volume confirmation.
    """
    out = df.copy()

    prior_high = out["high"].shift(1).rolling(BREAKOUT_LOOKBACK, min_periods=BREAKOUT_LOOKBACK).max()
    prior_low = out["low"].shift(1).rolling(BREAKOUT_LOOKBACK, min_periods=BREAKOUT_LOOKBACK).min()

    vol_confirm = out["volume"] > (out["vol_ma20"] * 1.2)
    ema_bull = out["ema9"] > out["ema21"]
    ema_bear = out["ema9"] < out["ema21"]

    bar_range = (out["high"] - out["low"]).replace(0.0, np.nan)
    close_position = (out["close"] - out["low"]) / bar_range  # 0=at low, 1=at high
    strong_bull_close = close_position >= 0.65
    strong_bear_close = close_position <= 0.35

    out["long_setup"] = (
        (out["trend"] == "up") & (out["close"] > prior_high) & vol_confirm & ema_bull & strong_bull_close
    )
    out["short_setup"] = (
        (out["trend"] == "down") & (out["close"] < prior_low) & vol_confirm & ema_bear & strong_bear_close
    )

    breakout_dist_long = ((out["close"] - prior_high) / out["atr14"]).clip(lower=0, upper=3.0)
    breakout_dist_short = ((prior_low - out["close"]) / out["atr14"]).clip(lower=0, upper=3.0)
    vol_ratio = (out["volume"] / out["vol_ma20"]).clip(upper=3.0)
    ema_slope = (out["ema9"] - out["ema21"]) / out["ema21"]

    adx_score = (out["htf_adx14"].clip(upper=60.0) / 60.0) * 40.0
    vol_score = (vol_ratio.clip(lower=0) / 3.0) * 20.0
    long_ema_score = (ema_slope.clip(lower=0, upper=0.008) / 0.008) * 15.0
    short_ema_score = ((-ema_slope).clip(lower=0, upper=0.008) / 0.008) * 15.0

    out["confidence_long"] = (
        adx_score + vol_score + long_ema_score + (breakout_dist_long / 3.0) * 25.0
    ).clip(0, 95)
    out["confidence_short"] = (
        adx_score + vol_score + short_ema_score + (breakout_dist_short / 3.0) * 25.0
    ).clip(0, 95)

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
        entry = float(row["close"])
        stop = entry - ATR_INIT_MULT * atr
        risk = entry - stop
        return Signal(
            side="long",
            entry=entry,
            stop=stop,
            take_profit_1=entry + TP1_R_MULT * risk,
            take_profit_2=entry + 8.0 * risk,
            confidence=float(row["confidence_long"]),
            reason="uptrend breakout above prior 20-bar high, EMA9>EMA21, volume confirm",
        )

    if bool(row.get("short_setup", False)):
        entry = float(row["close"])
        stop = entry + ATR_INIT_MULT * atr
        risk = stop - entry
        return Signal(
            side="short",
            entry=entry,
            stop=stop,
            take_profit_1=entry - TP1_R_MULT * risk,
            take_profit_2=entry - 8.0 * risk,
            confidence=float(row["confidence_short"]),
            reason="downtrend breakdown below prior 20-bar low, EMA9<EMA21, volume confirm",
        )

    return None


def prepare(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: 1m OHLCV -> 15m entry frame with HTF(1h) trend and
    signal columns attached. Used by the backtester, which only has 1m
    history to work from.
    """
    df = df_1m.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")

    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    df_15m = df.resample("15min").agg(agg).dropna().reset_index()
    df_1h = df.resample("1h").agg(agg).dropna().reset_index()
    return prepare_from_ltf_htf(df_15m, df_1h)


def prepare_from_ltf_htf(df_15m_raw: pd.DataFrame, df_1h_raw: pd.DataFrame) -> pd.DataFrame:
    """Same pipeline as prepare(), but starting from native 15m/1h OHLCV
    (e.g. fetched directly from the exchange's kline endpoints, which is
    far cheaper than pulling enough 1m history to resample). This is the
    single function both the backtester (via prepare()) and the live bot
    call, so behaviour cannot drift between them.
    """
    df_htf = compute_htf(df_1h_raw)
    df_ltf = compute_ltf(df_15m_raw)
    merged = merge_htf_trend(df_ltf, df_htf)
    merged = build_signal_columns(merged)
    return merged
