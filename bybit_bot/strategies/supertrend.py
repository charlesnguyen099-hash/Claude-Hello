"""
Strategy 4: Supertrend
- Supertrend flip từ bear → bull → Long
- Supertrend flip từ bull → bear → Short
- Kết hợp ADX để chỉ trade khi trend đủ mạnh (ADX > 25)
"""

import numpy as np
import pandas as pd
from .base import BaseStrategy, Signal, compute_atr
import config


def compute_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    atr      = compute_atr(df, period)
    hl2      = (df["high"] + df["low"]) / 2
    upper    = hl2 + multiplier * atr
    lower    = hl2 - multiplier * atr

    supertrend = pd.Series(index=df.index, dtype=float)
    direction  = pd.Series(index=df.index, dtype=int)

    supertrend.iloc[0] = upper.iloc[0]
    direction.iloc[0]  = 1

    for i in range(1, len(df)):
        prev_st  = supertrend.iloc[i - 1]
        prev_dir = direction.iloc[i - 1]
        close    = df["close"].iloc[i]

        curr_up  = upper.iloc[i]
        curr_lo  = lower.iloc[i]

        if prev_dir == 1:  # was bullish
            if close < prev_st:
                supertrend.iloc[i] = curr_up
                direction.iloc[i]  = -1
            else:
                supertrend.iloc[i] = min(curr_lo, prev_st) if curr_lo > prev_st else curr_lo
                direction.iloc[i]  = 1
        else:  # was bearish
            if close > prev_st:
                supertrend.iloc[i] = curr_lo
                direction.iloc[i]  = 1
            else:
                supertrend.iloc[i] = max(curr_up, prev_st) if curr_up < prev_st else curr_up
                direction.iloc[i]  = -1

    return supertrend, direction


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)

    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    dm_plus  = ((high - prev_high).clip(lower=0)).where((high - prev_high) > (prev_low - low), 0)
    dm_minus = ((prev_low - low).clip(lower=0)).where((prev_low - low) > (high - prev_high), 0)

    atr14     = tr.ewm(span=period, adjust=False).mean()
    di_plus   = 100 * dm_plus.ewm(span=period, adjust=False).mean()  / atr14
    di_minus  = 100 * dm_minus.ewm(span=period, adjust=False).mean() / atr14
    dx        = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx       = dx.ewm(span=period, adjust=False).mean()
    return adx


class SupertrendStrategy(BaseStrategy):
    name = "supertrend"

    def __init__(self, period=10, multiplier=3.0, adx_threshold=25):
        self.period       = period
        self.multiplier   = multiplier
        self.adx_threshold = adx_threshold

    def generate_signal(self, df: pd.DataFrame, df_trend: pd.DataFrame, df_macro: pd.DataFrame) -> Signal:
        null = Signal(0, 0.0, self.name, 0, 0)
        if len(df) < self.period + 10:
            return null

        _, direction = compute_supertrend(df, self.period, self.multiplier)
        adx   = compute_adx(df)
        atr   = compute_atr(df, config.ATR_PERIOD).iloc[-1]
        price = df["close"].iloc[-1]

        flipped_bull = direction.iloc[-2] == -1 and direction.iloc[-1] == 1
        flipped_bear = direction.iloc[-2] ==  1 and direction.iloc[-1] == -1
        trend_strong = adx.iloc[-1] > self.adx_threshold

        if flipped_bull and trend_strong:
            return Signal(1, 0.85, self.name, price, atr, f"Supertrend flip BULL, ADX={adx.iloc[-1]:.1f}")

        if flipped_bear and trend_strong:
            return Signal(-1, 0.85, self.name, price, atr, f"Supertrend flip BEAR, ADX={adx.iloc[-1]:.1f}")

        return null
