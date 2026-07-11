"""
Strategy 5: VWAP + Volume Profile Breakout
- Giá breakout trên VWAP + volume tăng mạnh → Long
- Giá breakdown dưới VWAP + volume tăng mạnh → Short
- VWAP hàng ngày tính từ nến 15m (không cần tick data)
"""

import pandas as pd
import numpy as np
from .base import BaseStrategy, Signal, compute_atr, compute_ema
import config


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP tính rolling 96 nến (= 1 ngày với 15m candles)."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    tpv     = typical * df["volume"]
    window  = 96  # 96 × 15m = 24h
    vwap    = tpv.rolling(window, min_periods=1).sum() / df["volume"].rolling(window, min_periods=1).sum()
    return vwap


class VWAPVolumeStrategy(BaseStrategy):
    name = "vwap_volume"

    def generate_signal(self, df: pd.DataFrame, df_trend: pd.DataFrame, df_macro: pd.DataFrame) -> Signal:
        null = Signal(0, 0.0, self.name, 0, 0)
        if len(df) < 50:
            return null

        close  = df["close"]
        vwap   = compute_vwap(df)
        atr    = compute_atr(df, config.ATR_PERIOD).iloc[-1]
        price  = close.iloc[-1]

        vol_ma  = df["volume"].rolling(20).mean()
        vol_ratio = df["volume"] / vol_ma.replace(0, np.nan)

        # Breakout: cross VWAP trong 3 nến gần nhất + volume xác nhận
        # Look back 3 bars so signal stays valid for a few candles after the cross
        cross_above = any(
            close.iloc[-(i+2)] <= vwap.iloc[-(i+2)] and close.iloc[-(i+1)] > vwap.iloc[-(i+1)]
            for i in range(3)
        )
        cross_below = any(
            close.iloc[-(i+2)] >= vwap.iloc[-(i+2)] and close.iloc[-(i+1)] < vwap.iloc[-(i+1)]
            for i in range(3)
        )
        # Volume confirm: bất kỳ nến nào trong 3 nến vừa rồi có vol cao
        vol_confirm = vol_ratio.iloc[-3:].max() > 1.5
        vol_peak    = vol_ratio.iloc[-3:].max()

        trend = self._trend_direction(df_trend)

        if cross_above and vol_confirm and trend >= 0 and price > vwap.iloc[-1]:
            strength = min(0.9, 0.5 + (vol_peak - 1.5) * 0.1)
            return Signal(1, strength, self.name, price, atr,
                          f"VWAP breakout up, vol×{vol_peak:.1f}")

        if cross_below and vol_confirm and trend <= 0 and price < vwap.iloc[-1]:
            strength = min(0.9, 0.5 + (vol_peak - 1.5) * 0.1)
            return Signal(-1, strength, self.name, price, atr,
                          f"VWAP breakdown, vol×{vol_peak:.1f}")

        return null
