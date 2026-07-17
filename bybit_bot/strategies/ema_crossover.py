"""
Strategy 1: EMA Crossover + Trend Filter
- EMA 9 cắt lên EMA 21 + giá trên EMA 50 -> Long
- EMA 9 cắt xuống EMA 21 + giá dưới EMA 50 -> Short
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

        price    = close.iloc[-1]
        macro_d  = self._trend_direction(df_macro)   # 4h
        trend_1h = self._trend_direction(df_trend)   # 1h

        # Trạng thái EMA hiện tại: fast > slow = bullish alignment
        ema_f_now = ema_f.iloc[-1]
        ema_s_now = ema_s.iloc[-1]
        ema_t_now = ema_t.iloc[-1]
        gap_pct   = (ema_f_now - ema_s_now) / ema_s_now  # % khoảng cách EMA9 - EMA21

        # Volume confirmation
        vol_avg   = df["volume"].rolling(20).mean().iloc[-1]
        vol_curr  = df["volume"].iloc[-3:].mean()  # trung bình 3 nến gần nhất
        vol_ok    = vol_curr > vol_avg * 1.1

        # Kiem tra EMA cross moi hinh thanh (trong 3 nen gan nhat)
        ema_f_prev3 = ema_f.iloc[-4:-1]
        ema_s_prev3 = ema_s.iloc[-4:-1]
        recently_crossed_up   = any(ema_f_prev3.values[i] <= ema_s_prev3.values[i] for i in range(3))
        recently_crossed_down = any(ema_f_prev3.values[i] >= ema_s_prev3.values[i] for i in range(3))

        # Long: EMA cross up + it nhat 1 TF xac nhan uptrend (tranh double-sideways)
        if ema_f_now > ema_s_now > ema_t_now and gap_pct > 0.001 and recently_crossed_up and vol_ok and (trend_1h + macro_d) >= 1:
            strength = min(0.85, 0.55 + gap_pct * 10)
            return Signal(1, strength, self.name, price, atr, f"EMA bullish cross gap={gap_pct*100:.2f}%")

        # Short: EMA cross down + it nhat 1 TF xac nhan downtrend
        if ema_f_now < ema_s_now < ema_t_now and abs(gap_pct) > 0.001 and recently_crossed_down and vol_ok and (trend_1h + macro_d) <= -1:
            strength = min(0.85, 0.55 + abs(gap_pct) * 10)
            return Signal(-1, strength, self.name, price, atr, f"EMA bearish cross gap={abs(gap_pct)*100:.2f}%")

        return null
