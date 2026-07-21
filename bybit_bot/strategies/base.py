"""
Abstract base class cho tất cả strategies.
Signal: 1 = Long, -1 = Short, 0 = Không vào lệnh
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np


@dataclass
class Signal:
    direction: int          # 1=Long, -1=Short, 0=No trade
    strength: float         # 0.0 - 1.0
    strategy_name: str
    entry_price: float
    atr: float              # Dùng để tính SL/TP dong
    reason: str = ""        # Mo ta ly do vao lenh (positional arg thu 6)
    symbol: str = ""
    consensus: int = 1      # So strategies dong thuan cung chieu
    swing_sl: float = 0.0   # Swing high/low 15m lam SL reference (0 = dung ATR thuan tuy)
    tp_roi_override: float = 0.0  # != 0 → dung truc tiep lam TP ROI (bo qua potential scaling)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast   = compute_ema(series, fast)
    ema_slow   = compute_ema(series, slow)
    macd_line  = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high  = df["high"]
    low   = df["low"]
    close = df["close"]
    plus_dm  = (high.diff()).clip(lower=0).where(high.diff() > -low.diff(), 0)
    minus_dm = (-low.diff()).clip(lower=0).where(-low.diff() > high.diff(), 0)
    atr      = compute_atr(df, period)
    plus_di  = 100 * plus_dm.ewm(span=period, adjust=False).mean()  / atr
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr
    dx       = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.ewm(span=period, adjust=False).mean()


class BaseStrategy(ABC):
    name: str = "base"

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, df_trend: pd.DataFrame, df_macro: pd.DataFrame) -> Signal:
        """
        df       : nến 15m (signal timeframe)
        df_trend : nến 1h  (trend confirmation)
        df_macro : nến 4h  (macro direction)
        """
        ...

    def _trend_direction(self, df: pd.DataFrame, fast: int = 100, slow: int = 250) -> int:
        """+1 up, -1 down, 0 sideways.
        Chay tren 1m data 2000 nen: EMA(100/250) tuong duong EMA(20/50) tren 5m.
        EMA(20/50) cu qua ngan (20-50 phut) -> lag, fire khi move da xong -> Long o dinh / Short o day.
        EMA(100/250) on dinh hon, xac nhan trend that su (1.7h / ~4h trend)."""
        if len(df) < slow + 5:
            return 0
        ema_f = compute_ema(df["close"], fast).iloc[-1]
        ema_s = compute_ema(df["close"], slow).iloc[-1]
        price = df["close"].iloc[-1]
        if price > ema_f > ema_s:
            return 1
        if price < ema_f < ema_s:
            return -1
        return 0

    def _macro_direction(self, df: pd.DataFrame) -> int:
        """Macro trend tren 1m data: EMA(300/600) ~ EMA(20/40) tren 15m (~5h/~10h trend).
        Dung cho df_macro de phan biet voi _trend_direction (medium trend)."""
        return self._trend_direction(df, fast=300, slow=600)
