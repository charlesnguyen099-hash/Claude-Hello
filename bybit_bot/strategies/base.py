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
    atr: float              # Dùng để tính SL/TP động
    reason: str = ""


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

    def _trend_direction(self, df: pd.DataFrame) -> int:
        """1h trend: +1 up, -1 down, 0 sideways."""
        if len(df) < 50:
            return 0
        ema20 = compute_ema(df["close"], 20).iloc[-1]
        ema50 = compute_ema(df["close"], 50).iloc[-1]
        price = df["close"].iloc[-1]
        if price > ema20 > ema50:
            return 1
        if price < ema20 < ema50:
            return -1
        return 0
