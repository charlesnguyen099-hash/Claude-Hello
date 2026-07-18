"""
Strategy 2: RSI + MACD Momentum
- RSI thoát khỏi vùng oversold (< 35 -> > 35) + MACD histogram dương -> Long
- RSI thoát khỏi vùng overbought (> 65 -> < 65) + MACD histogram âm -> Short
- Trend filter từ 1h
"""

import pandas as pd
from .base import BaseStrategy, Signal, compute_rsi, compute_macd, compute_atr
import config


class RSIMACDStrategy(BaseStrategy):
    name = "rsi_macd"

    def __init__(self, rsi_period=14, oversold=30, overbought=70):
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

        trend_1h = self._trend_direction(df_trend)   # EMA100/250 ~ medium trend
        macro_d  = self._macro_direction(df_macro)   # EMA300/600 ~ macro trend
        rsi_now  = rsi.iloc[-1]
        hist_now = hist.iloc[-1]
        hist_prev = hist.iloc[-2]

        # Momentum zone dong bo voi reversal routing moi (30/70):
        # Long: RSI 30-65 — tranh entry khi RSI > 65 (sap overbought reversal zone)
        # Short: RSI 35-70 — tranh entry khi RSI < 35 (sap oversold reversal zone)
        # Ca 2 loai tru vung reversal (< 30 / > 70) — khu vuc do main.py xu ly rieng
        hist_prev2 = hist.iloc[-3]
        macd_accel_up   = hist_now > hist_prev > hist_prev2  # tang 2 nen lien tiep
        macd_accel_down = hist_now < hist_prev < hist_prev2  # giam 2 nen lien tiep

        if 30 <= rsi_now <= 65 and macd_accel_up and hist_now > 0 and (trend_1h >= 1 or macro_d >= 1):
            strength = min(0.85, 0.5 + (rsi_now - 30) / 100 + (hist_now - hist_prev) / (abs(hist_now) + 1e-9) * 0.1)
            return Signal(1, strength, self.name, price, atr,
                          f"RSI={rsi_now:.0f} MACD accel up {hist_prev2:.4f}->{hist_prev:.4f}->{hist_now:.4f}")

        if 35 <= rsi_now <= 70 and macd_accel_down and hist_now < 0 and (trend_1h <= -1 or macro_d <= -1):
            strength = min(0.85, 0.5 + (70 - rsi_now) / 100 + (hist_prev - hist_now) / (abs(hist_now) + 1e-9) * 0.1)
            return Signal(-1, strength, self.name, price, atr,
                          f"RSI={rsi_now:.0f} MACD accel down {hist_prev2:.4f}->{hist_prev:.4f}->{hist_now:.4f}")

        return null
