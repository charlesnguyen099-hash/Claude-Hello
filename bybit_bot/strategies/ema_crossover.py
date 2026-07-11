"""
Strategy 1: EMA Crossover + Trend Filter
- EMA 9 cắt lên EMA 21 + giá trên EMA 50 → Long
- EMA 9 cắt xuống EMA 21 + giá dưới EMA 50 → Short
- Xác nhận bằng volume spike
"""

import pandas as pd
from .base import BaseStrategy, Signal, compute_ema, compute_atr
import config


class EMACrossoverStrategy(BaseStrategy):
    name = "ema_crossover"

    def __init__(self, fast=9, slow=21, trend=50):
        self.fast  = fast
        self.slow  = slow
        self.trend = trend

    def generate_signal(self, df: pd.DataFrame, df_trend: pd.DataFrame, df_macro: pd.DataFrame) -> Signal:
        null = Signal(0, 0.0, self.name, 0, 0)
        if len(df) < self.trend + 5:
            return null

        close = df["close"]
        ema_f = compute_ema(close, self.fast)
        ema_s = compute_ema(close, self.slow)
        ema_t = compute_ema(close, self.trend)
        atr   = compute_atr(df, config.ATR_PERIOD).iloc[-1]

        # Crossover detection
        cross_up   = (ema_f.iloc[-2] <= ema_s.iloc[-2]) and (ema_f.iloc[-1] > ema_s.iloc[-1])
        cross_down = (ema_f.iloc[-2] >= ema_s.iloc[-2]) and (ema_f.iloc[-1] < ema_s.iloc[-1])

        price    = close.iloc[-1]
        trend_ok = price > ema_t.iloc[-1]
        macro_d  = self._trend_direction(df_macro)

        # Volume confirmation
        vol_avg  = df["volume"].rolling(20).mean().iloc[-1]
        vol_curr = df["volume"].iloc[-1]
        vol_spike = vol_curr > vol_avg * 1.2

        if cross_up and trend_ok and macro_d >= 0:
            strength = 0.8 if vol_spike else 0.6
            return Signal(1, strength, self.name, price, atr, "EMA cross up + trend up")

        if cross_down and not trend_ok and macro_d <= 0:
            strength = 0.8 if vol_spike else 0.6
            return Signal(-1, strength, self.name, price, atr, "EMA cross down + trend down")

        return null
