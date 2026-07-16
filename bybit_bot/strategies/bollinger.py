"""
Strategy 3: Bollinger Bands + RSI Divergence (reversal)
- Giá chạm lower band + RSI < 35 + nến đảo chiều → Long (reversal from oversold)
- Giá chạm upper band + RSI > 65 + nến đảo chiều → Short (reversal from overbought)
- 1h+4h không được oppose: LONG chỉ khi sum(1h,4h) >= 0; SHORT chỉ khi sum <= 0
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

        # Nen dao chieu bullish: nen hien tai HOAC nen truoc cham lower band
        curr_low  = df["low"].iloc[-1]
        bullish_reversal = curr_close > prev_close and (
            prev_low <= lower.iloc[-2] or curr_low <= lower.iloc[-1]
        )
        bearish_reversal = curr_close < prev_close and (
            prev_high >= upper.iloc[-2] or df["high"].iloc[-1] >= upper.iloc[-1]
        )

        # RSI: accept neu nen hien tai HOAC nen truoc o vung extreme
        rsi_oversold   = rsi.iloc[-1] < 35 or rsi.iloc[-2] < 35
        rsi_overbought = rsi.iloc[-1] > 65 or rsi.iloc[-2] > 65

        macro_d  = self._trend_direction(df_macro)   # 4h
        trend_1h = self._trend_direction(df_trend)   # 1h

        if bullish_reversal and rsi_oversold and trend_1h >= 0 and macro_d >= 0:
            # Do khoang cach gia vs lower band (khoang phuc hoi tu band)
            band_dist = abs(price - lower.iloc[-1]) / (lower.iloc[-1] + 1e-9)
            strength  = min(0.85, 0.5 + band_dist * 10)
            return Signal(1, strength, self.name, price, atr, "BB lower touch + RSI oversold reversal")

        if bearish_reversal and rsi_overbought and trend_1h <= 0 and macro_d <= 0:
            band_dist = abs(price - upper.iloc[-1]) / (upper.iloc[-1] + 1e-9)
            strength  = min(0.85, 0.5 + band_dist * 10)
            return Signal(-1, strength, self.name, price, atr, "BB upper touch + RSI overbought reversal")

        return null
