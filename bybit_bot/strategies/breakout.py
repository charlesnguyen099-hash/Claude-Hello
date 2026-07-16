"""
Breakout Strategy — bat volume spike + pha vo vung tich luy
- Volume nen hien tai > 3.0x MA10 volume (vol_mult=3.0, khoi tao trong __init__.py)
- Gia pha vo high/low cua 20 nen truoc (lookback=20)
- Yeu cau sum(1h+4h) >= 1 cho LONG / <= -1 cho SHORT (check trong main.py bo_trend_ok)
"""

import pandas as pd
from .base import BaseStrategy, Signal, compute_atr
import config


class BreakoutStrategy(BaseStrategy):
    name = "breakout"

    def __init__(self, vol_mult=2.5, lookback=20):
        self.vol_mult = vol_mult   # volume spike nguong
        self.lookback = lookback   # so nen xet vung tich luy

    def generate_signal(self, df: pd.DataFrame, df_trend: pd.DataFrame, df_macro: pd.DataFrame) -> Signal:
        null = Signal(0, 0.0, self.name, 0, 0)
        n = len(df)
        if n < self.lookback + 10:
            return null

        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]

        atr   = compute_atr(df, config.ATR_PERIOD).iloc[-1]
        price = close.iloc[-1]

        # Volume MA10 va volume nen hien tai
        vol_ma10  = volume.iloc[-11:-1].mean()   # 10 nen truoc, khong tinh nen hien tai
        vol_now   = volume.iloc[-1]
        if vol_ma10 <= 0:
            return null

        vol_ratio = vol_now / vol_ma10
        if vol_ratio < self.vol_mult:
            return null   # Chua du volume spike

        # Vung tich luy: high/low cua lookback nen truoc (khong tinh nen hien tai)
        prev_high = high.iloc[-self.lookback-1:-1].max()
        prev_low  = low.iloc[-self.lookback-1:-1].min()

        # Pha vo len tren: nen hien tai dong cao hon high 20 nen truoc
        if close.iloc[-1] > prev_high and close.iloc[-1] > close.iloc[-2]:
            strength = min(0.95, 0.70 + min(vol_ratio - self.vol_mult, 5) * 0.05)
            return Signal(
                direction=1,
                strength=strength,
                strategy_name=self.name,
                entry_price=price,
                atr=atr,
                reason=f"Breakout UP: vol={vol_ratio:.1f}x MA10, broke {prev_high:.4f}"
            )

        # Pha vo xuong duoi: nen hien tai dong thap hon low 20 nen truoc
        if close.iloc[-1] < prev_low and close.iloc[-1] < close.iloc[-2]:
            strength = min(0.95, 0.70 + min(vol_ratio - self.vol_mult, 5) * 0.05)
            return Signal(
                direction=-1,
                strength=strength,
                strategy_name=self.name,
                entry_price=price,
                atr=atr,
                reason=f"Breakout DOWN: vol={vol_ratio:.1f}x MA10, broke {prev_low:.4f}"
            )

        return null
