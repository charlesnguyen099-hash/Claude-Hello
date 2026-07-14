"""
Strategy 3: Bollinger Bands + RSI Divergence
- Giá chạm lower band + RSI < 35 + nến đảo chiều → Long
- Giá chạm upper band + RSI > 65 + nến đảo chiều → Short
- Dùng khi thị trường sideways (macro trend = 0)
"""

import numpy as np
import pandas as pd
from .base import BaseStrategy, Signal, compute_rsi, compute_atr
import config


class BollingerStrategy(BaseStrategy):
    name = "bollinger"

    def __init__(self, period=20, std_dev=2.0):
        self.period  = period
        self.std_dev = std_dev

    def generate_signal(self, df: pd.DataFrame, df_trend: pd.DataFrame, df_macro: pd.DataFrame) -> Signal:
        null = Signal(0, 0.0, self.name, 0, 0)
        if len(df) < self.period + 5:
            return null

        close  = df["close"]
        high   = df["high"]
        low    = df["low"]

        ma     = close.rolling(self.period).mean()
        std    = close.rolling(self.period).std()
        upper  = ma + self.std_dev * std
        lower  = ma - self.std_dev * std
        bw     = (upper - lower) / ma  # Bandwidth — volatility gauge

        rsi    = compute_rsi(close, 14)
        atr    = compute_atr(df, config.ATR_PERIOD).iloc[-1]
        price  = close.iloc[-1]

        prev_low  = low.iloc[-2]
        prev_high = high.iloc[-2]
        prev_close = close.iloc[-2]
        curr_close = close.iloc[-1]

        # Không trade trong sideways quá hẹp (bandwidth < 1%)
        if bw.iloc[-1] < 0.01:
            return null

        # Nến đảo chiều bullish (hammer / engulfing)
        bullish_reversal = curr_close > prev_close and prev_low <= lower.iloc[-2]
        bearish_reversal = curr_close < prev_close and prev_high >= upper.iloc[-2]

        macro_d = self._trend_direction(df_macro)

        if bullish_reversal and rsi.iloc[-1] < 35 and macro_d >= 0:
            pct_below = (lower.iloc[-1] - price) / lower.iloc[-1]
            strength  = min(0.85, 0.5 + abs(pct_below) * 10)
            return Signal(1, strength, self.name, price, atr, "BB lower touch + RSI oversold reversal")

        if bearish_reversal and rsi.iloc[-1] > 65 and macro_d <= 0:
            pct_above = (price - upper.iloc[-1]) / upper.iloc[-1]
            strength  = min(0.85, 0.5 + abs(pct_above) * 10)
            return Signal(-1, strength, self.name, price, atr, "BB upper touch + RSI overbought reversal")

        return null
