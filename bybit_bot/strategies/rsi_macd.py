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

        rsi_prev, rsi_curr = rsi.iloc[-2], rsi.iloc[-1]
        hist_prev, hist_curr = hist.iloc[-2], hist.iloc[-1]

        trend = self._trend_direction(df_trend)

        # RSI exit oversold + MACD turning positive
        if (rsi_prev < self.oversold and rsi_curr >= self.oversold
                and hist_curr > hist_prev and trend >= 0):
            strength = min(0.9, 0.5 + (self.oversold - rsi_prev) / 100)
            return Signal(1, strength, self.name, price, atr,
                          f"RSI {rsi_prev:.0f}→{rsi_curr:.0f} exit oversold, MACD↑")

        # RSI exit overbought + MACD turning negative
        if (rsi_prev > self.overbought and rsi_curr <= self.overbought
                and hist_curr < hist_prev and trend <= 0):
            strength = min(0.9, 0.5 + (rsi_prev - self.overbought) / 100)
            return Signal(-1, strength, self.name, price, atr,
                          f"RSI {rsi_prev:.0f}→{rsi_curr:.0f} exit overbought, MACD↓")

        return null
