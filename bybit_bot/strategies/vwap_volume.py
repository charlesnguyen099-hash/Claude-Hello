"""
Strategy 5: VWAP + Volume Profile Breakout
- Giá breakout trên VWAP + volume tăng mạnh -> Long
- Giá breakdown dưới VWAP + volume tăng mạnh -> Short
- VWAP hàng ngày tính từ nến 15m (không cần tick data)
"""

import logging
import pandas as pd
import numpy as np
from .base import BaseStrategy, Signal, compute_atr, compute_ema
import config

logger = logging.getLogger(__name__)


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP rolling 24h.
    1m candles: window=1440 (1440 x 1m = 24h).
    15m candles: window=96 (96 x 15m = 24h).
    Tu dong chon window dua tren so nen va khoang cach timestamp.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3
    tpv     = typical * df["volume"]
    # Detect interval: neu >= 500 nen trong df, gia su la 1m (signal=2000c), nguoc lai 15m/5m
    # Chinh xac hon: tinh tu 2 timestamp lien tiep
    window = 96  # default 15m
    if len(df) >= 2:
        try:
            dt_diff = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[-2]).total_seconds()
            if dt_diff <= 60:      # 1m
                window = 1440
            elif dt_diff <= 300:   # 5m
                window = 288
            else:                  # 15m+
                window = 96
        except Exception:
            window = 96
    vwap = tpv.rolling(window, min_periods=1).sum() / df["volume"].rolling(window, min_periods=1).sum()
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

        vol_ma    = df["volume"].rolling(20).mean()
        vol_ratio = df["volume"] / vol_ma.replace(0, np.nan)
        vol_peak  = vol_ratio.iloc[-3:].max()
        vol_ok    = vol_peak > 1.2

        vwap_now  = vwap.iloc[-1]
        # Khoảng cách giá so với VWAP (%)
        dist_pct  = (price - vwap_now) / vwap_now

        trend_1h = self._trend_direction(df_trend)   # 1h
        macro_d  = self._trend_direction(df_macro)   # 4h

        # Long: gia breakout VWAP (0.1% - 3%) + 1h HOAC 4h xac nhan uptrend
        if 0.001 < dist_pct < 0.03 and vol_ok and (trend_1h >= 1 or macro_d >= 1):
            strength = min(0.9, 0.5 + min(dist_pct, 0.03) * 5 + (vol_peak - 1.2) * 0.05)
            return Signal(1, strength, self.name, price, atr,
                          f"Price {dist_pct*100:.2f}% above VWAP, vol*{vol_peak:.1f}")

        # Short: gia breakdown VWAP (-0.1% den -3%) + 1h HOAC 4h xac nhan downtrend
        if -0.03 < dist_pct < -0.001 and vol_ok and (trend_1h <= -1 or macro_d <= -1):
            strength = min(0.9, 0.5 + min(abs(dist_pct), 0.03) * 5 + (vol_peak - 1.2) * 0.05)
            return Signal(-1, strength, self.name, price, atr,
                          f"Price {abs(dist_pct)*100:.2f}% below VWAP, vol*{vol_peak:.1f}")

        return null
