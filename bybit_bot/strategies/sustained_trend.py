"""
Sustained Trend + Reversal Strategy
- Bat sustained downtrend/uptrend: EMA doc deu >= 20 nen lien tiep
- Bat reversal tai day/dinh: RSI < 35 hoac > 65 + nen dao chieu + volume tang
"""

import pandas as pd
from .base import BaseStrategy, Signal, compute_ema, compute_rsi, compute_atr
import config


class SustainedTrendStrategy(BaseStrategy):
    name = "sustained_trend"

    def __init__(self, slope_bars=20, rsi_oversold=35, rsi_overbought=65):
        self.slope_bars     = slope_bars
        self.rsi_oversold   = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def generate_signal(self, df: pd.DataFrame, df_trend: pd.DataFrame, df_macro: pd.DataFrame) -> Signal:
        null = Signal(0, 0.0, self.name, 0, 0)
        n = len(df)
        if n < self.slope_bars + 30:
            return null

        close  = df["close"]
        open_  = df["open"]
        volume = df["volume"]
        price  = close.iloc[-1]
        atr    = compute_atr(df, config.ATR_PERIOD).iloc[-1]
        rsi    = compute_rsi(close).iloc[-1]

        ema9  = compute_ema(close, 9)
        ema21 = compute_ema(close, 21)
        ema28 = compute_ema(close, 28)

        e9_now  = ema9.iloc[-1]
        e21_now = ema21.iloc[-1]
        e28_now = ema28.iloc[-1]

        # EMA slope: do doc trong slope_bars nen vua qua
        e9_slope  = (ema9.iloc[-1]  - ema9.iloc[-self.slope_bars])  / (abs(ema9.iloc[-self.slope_bars])  + 1e-9)
        e21_slope = (ema21.iloc[-1] - ema21.iloc[-self.slope_bars]) / (abs(ema21.iloc[-self.slope_bars]) + 1e-9)

        trend_1h = self._trend_direction(df_trend)   # 1h
        macro_d  = self._trend_direction(df_macro)   # 4h

        # ── SUSTAINED DOWNTREND -> SHORT ──────────────────────────────────────
        # EMA xep theo thu tu giam + ca 2 EMA dang doc xuong + RSI chua oversold
        ema_bear   = price < e9_now < e21_now < e28_now
        slope_down = e9_slope < -0.001 and e21_slope < -0.0005
        rsi_mid    = 35 < rsi < 65   # Chua oversold, con du cho xuong tiep

        # Dung OR: EMA alignment da xac nhan xu huong, chi can 1h HOAC 4h dong thuan
        if ema_bear and slope_down and rsi_mid and (trend_1h <= -1 or macro_d <= -1):
            strength = min(0.85, 0.60 + abs(e9_slope) * 20)
            return Signal(
                direction=-1,
                strength=strength,
                strategy_name=self.name,
                entry_price=price,
                atr=atr,
                reason=f"Sustained DOWN: EMA9slope={e9_slope*100:.3f}% RSI={rsi:.0f}"
            )

        # ── SUSTAINED UPTREND -> LONG ─────────────────────────────────────────
        ema_bull   = price > e9_now > e21_now > e28_now
        slope_up   = e9_slope > 0.001 and e21_slope > 0.0005
        rsi_mid_up = 35 < rsi < 65

        if ema_bull and slope_up and rsi_mid_up and (trend_1h >= 1 or macro_d >= 1):
            strength = min(0.85, 0.60 + abs(e9_slope) * 20)
            return Signal(
                direction=1,
                strength=strength,
                strategy_name=self.name,
                entry_price=price,
                atr=atr,
                reason=f"Sustained UP: EMA9slope={e9_slope*100:.3f}% RSI={rsi:.0f}"
            )

        # ── REVERSAL TẠI ĐÁY -> LONG ─────────────────────────────────────────
        # RSI oversold (< 35) + nen dao chieu (close > open, than lon) + volume tang
        vol_now  = volume.iloc[-1]
        vol_prev = volume.iloc[-2]
        last_body    = close.iloc[-1] - open_.iloc[-1]   # duong = xanh
        prev_body    = close.iloc[-2] - open_.iloc[-2]   # am = do

        reversal_long = (
            rsi < self.rsi_oversold and
            last_body > 0 and           # nen xanh
            prev_body < 0 and           # nen truoc do
            last_body > abs(prev_body) * 0.5 and  # than xanh >= 50% than do
            vol_now >= vol_prev * 0.8   # volume khong giam qua manh
        )
        # Reversal: chi can 1h HOAC 4h khong phai STRONG BEAR (>= -1 cho phep sideways)
        if reversal_long and (trend_1h >= 0 or macro_d >= 0):
            strength = min(0.90, 0.70 + (self.rsi_oversold - rsi) / 50)
            return Signal(
                direction=1,
                strength=strength,
                strategy_name=self.name,
                entry_price=price,
                atr=atr,
                reason=f"Reversal LONG at bottom: RSI={rsi:.0f} engulf up"
            )

        # ── REVERSAL TẠI ĐỈNH -> SHORT ────────────────────────────────────────
        reversal_short = (
            rsi > self.rsi_overbought and
            last_body < 0 and           # nen do
            prev_body > 0 and           # nen truoc xanh
            abs(last_body) > prev_body * 0.5 and
            vol_now >= vol_prev * 0.8
        )
        if reversal_short and (trend_1h <= 0 or macro_d <= 0):
            strength = min(0.90, 0.70 + (rsi - self.rsi_overbought) / 50)
            return Signal(
                direction=-1,
                strength=strength,
                strategy_name=self.name,
                entry_price=price,
                atr=atr,
                reason=f"Reversal SHORT at top: RSI={rsi:.0f} engulf down"
            )

        return null
