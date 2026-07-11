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
        rsi_now  = rsi.iloc[-1]
        hist_now = hist.iloc[-1]
        hist_prev = hist.iloc[-2]

        # Long: RSI trong vùng momentum dương (35-60) + MACD histogram đang tăng + trend >= 0
        if 35 <= rsi_now <= 60 and hist_now > hist_prev and hist_now > 0 and trend >= 0:
            strength = min(0.85, 0.5 + (rsi_now - 35) / 100 + (hist_now - hist_prev) / (abs(hist_now) + 1e-9) * 0.1)
            return Signal(1, strength, self.name, price, atr,
                          f"RSI={rsi_now:.0f} MACD hist rising {hist_prev:.4f}->{hist_now:.4f}")

        # Short: RSI trong vùng momentum âm (40-65) + MACD histogram đang giảm + trend <= 0
        if 40 <= rsi_now <= 65 and hist_now < hist_prev and hist_now < 0 and trend <= 0:
            strength = min(0.85, 0.5 + (65 - rsi_now) / 100 + (hist_prev - hist_now) / (abs(hist_now) + 1e-9) * 0.1)
            return Signal(-1, strength, self.name, price, atr,
                          f"RSI={rsi_now:.0f} MACD hist falling {hist_prev:.4f}->{hist_now:.4f}")

        return null
