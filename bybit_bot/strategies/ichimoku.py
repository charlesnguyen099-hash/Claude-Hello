"""
Strategy 6: Ichimoku Cloud
- Giá trên cloud + Tenkan cắt lên Kijun + Chikou xác nhận → Long
- Giá dưới cloud + Tenkan cắt xuống Kijun + Chikou xác nhận → Short
Ichimoku là hệ thống hoàn chỉnh nhất cho trend trading.
"""

import pandas as pd
from .base import BaseStrategy, Signal, compute_atr
import config


def ichimoku(df: pd.DataFrame, t=9, k=26, s=52, d=26):
    high, low = df["high"], df["low"]

    tenkan  = (high.rolling(t).max() + low.rolling(t).min()) / 2
    kijun   = (high.rolling(k).max() + low.rolling(k).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(d)
    senkou_b = ((high.rolling(s).max() + low.rolling(s).min()) / 2).shift(d)
    chikou  = df["close"].shift(-d)

    return tenkan, kijun, senkou_a, senkou_b, chikou


class IchimokuStrategy(BaseStrategy):
    name = "ichimoku"

    def generate_signal(self, df: pd.DataFrame, df_trend: pd.DataFrame, df_macro: pd.DataFrame) -> Signal:
        null = Signal(0, 0.0, self.name, 0, 0)
        if len(df) < 80:
            return null

        tenkan, kijun, senkou_a, senkou_b, chikou = ichimoku(df)
        atr   = compute_atr(df, config.ATR_PERIOD).iloc[-1]
        price = df["close"].iloc[-1]

        cloud_top    = max(senkou_a.iloc[-1], senkou_b.iloc[-1])
        cloud_bottom = min(senkou_a.iloc[-1], senkou_b.iloc[-1])

        # TK cross
        tk_cross_up   = tenkan.iloc[-2] <= kijun.iloc[-2] and tenkan.iloc[-1] > kijun.iloc[-1]
        tk_cross_down = tenkan.iloc[-2] >= kijun.iloc[-2] and tenkan.iloc[-1] < kijun.iloc[-1]

        above_cloud = price > cloud_top
        below_cloud = price < cloud_bottom

        # Chikou (lagging span) confirms — compare to price 26 bars ago
        if len(df) > 52:
            chikou_bullish = df["close"].iloc[-1] > df["close"].iloc[-26]
            chikou_bearish = df["close"].iloc[-1] < df["close"].iloc[-26]
        else:
            chikou_bullish = chikou_bearish = True  # skip if not enough data

        if tk_cross_up and above_cloud and chikou_bullish:
            return Signal(1, 0.9, self.name, price, atr, "Ichimoku TK cross up above cloud")

        if tk_cross_down and below_cloud and chikou_bearish:
            return Signal(-1, 0.9, self.name, price, atr, "Ichimoku TK cross down below cloud")

        return null
