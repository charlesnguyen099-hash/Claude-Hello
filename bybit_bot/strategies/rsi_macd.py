"""
Strategy 2: RSI + MACD Momentum
- RSI thoát khỏi vùng oversold (< 35 → > 35) + MACD histogram dương → Long
- RSI thoát khỏi vùng overbought (> 65 → < 65) + MACD histogram âm → Short
- Trend filter từ 1h
"""

import pandas as pd
from .base import BaseStrategy, Signal, compute_rsi, compute_macd, compute_atr
import config


class RSIMACDStrategy(BaseStrategy):
    name = "rsi_macd"

    def __init__(self, rsi_period=14, oversold=35, overbought=65):
        self.rsi_period  = rsi_period
        self.oversold    = oversold
        self.overbought  = overbought

    def generate_signal(self, df: pd.DataFrame, df_trend: pd.DataFrame, df_macro: pd.DataFrame) -> Signal:
        null = Signal(0, 0.0, self.name, 0, 0)
        if len(df) < 40:
            return null

        close = df["close"]
        rsi   = compute_rsi(close, self.rsi_period)
        _, _, hist = compute_macd(close)
        atr   = compute_atr(df, config.ATR_PERIOD).iloc[-1]
        price = close.iloc[-1]

        trend = self._trend_direction(df_trend)

        # RSI exit oversold in last 4 bars + MACD histogram rising
        for i in range(1, 5):
            r_prev = rsi.iloc[-(i+1)]
            r_curr = rsi.iloc[-i]
            h_prev = hist.iloc[-(i+1)]
            h_curr = hist.iloc[-i]
            if r_prev < self.oversold and r_curr >= self.oversold and h_curr > h_prev and trend >= 0:
                strength = min(0.9, 0.5 + (self.oversold - r_prev) / 100)
                return Signal(1, strength, self.name, price, atr,
                              f"RSI {r_prev:.0f}->{r_curr:.0f} exit oversold, MACD^")

        # RSI exit overbought in last 4 bars + MACD histogram falling
        for i in range(1, 5):
            r_prev = rsi.iloc[-(i+1)]
            r_curr = rsi.iloc[-i]
            h_prev = hist.iloc[-(i+1)]
            h_curr = hist.iloc[-i]
            if r_prev > self.overbought and r_curr <= self.overbought and h_curr < h_prev and trend <= 0:
                strength = min(0.9, 0.5 + (r_prev - self.overbought) / 100)
                return Signal(-1, strength, self.name, price, atr,
                              f"RSI {r_prev:.0f}->{r_curr:.0f} exit overbought, MACD v")

        return null
