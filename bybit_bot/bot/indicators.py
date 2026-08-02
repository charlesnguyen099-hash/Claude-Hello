"""Pure pandas/numpy technical indicators. No external TA dependency,
so behavior is identical between backtest and live bot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(50.0)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    a = df["high"] - df["low"]
    b = (df["high"] - prev_close).abs()
    c = (df["low"] - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def rolling_block_atr(df: pd.DataFrame, block_minutes: int = 15, length: int = 14) -> pd.Series:
    """ATR computed over a rolling N-minute block instead of the native
    bar. Plain ATR is an inherently per-bar-size measure — a 1-minute
    candle's true range is always much smaller than a 15-minute candle's,
    no matter how many 1-minute bars you average over, because widening
    the lookback window changes how many ranges get averaged, not how big
    each individual range is. This computes the range of a sliding
    `block_minutes`-wide window (as if it were one candle) at every row,
    then EMA-smooths those block ranges — giving a volatility measure in
    the same real-world units as a coarser timeframe's ATR, while still
    updating every time a new 1-minute bar closes.
    """
    block_high = df["high"].rolling(block_minutes, min_periods=block_minutes).max()
    block_low = df["low"].rolling(block_minutes, min_periods=block_minutes).min()
    prev_close = df["close"].shift(block_minutes)
    a = block_high - block_low
    b = (block_high - prev_close).abs()
    c = (block_low - prev_close).abs()
    tr = pd.concat([a, b, c], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(df)
    atr_ = tr.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()

    plus_dm_s = pd.Series(plus_dm, index=df.index).ewm(
        alpha=1.0 / length, adjust=False, min_periods=length
    ).mean()
    minus_dm_s = pd.Series(minus_dm, index=df.index).ewm(
        alpha=1.0 / length, adjust=False, min_periods=length
    ).mean()

    plus_di = 100.0 * (plus_dm_s / atr_.replace(0.0, np.nan))
    minus_di = 100.0 * (minus_dm_s / atr_.replace(0.0, np.nan))

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean().fillna(0.0)

