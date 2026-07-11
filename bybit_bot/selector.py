"""
Strategy Selector — Auto chọn strategy tốt nhất cho mỗi symbol
bằng cách backtest nhanh trên dữ liệu nến từ Bybit API (không lưu local).

Metric chọn: Expectancy = WinRate × AvgWin - LossRate × AvgLoss
(Tốt hơn Sharpe ratio thuần vì phản ánh được kỳ vọng lợi nhuận thực tế)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np

from strategies.base import BaseStrategy, Signal, compute_atr
from strategies import ALL_STRATEGIES
import config

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    strategy_name: str
    trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    expectancy: float       # kỳ vọng lợi nhuận mỗi lệnh (%)
    profit_factor: float


def quick_backtest(
    strategy: BaseStrategy,
    df: pd.DataFrame,
    df_trend: pd.DataFrame,
    df_macro: pd.DataFrame,
    eval_bars: int = 100,
    sl_atr: float = 1.5,
    tp_atr: float = 2.0,
) -> BacktestResult:
    """
    Backtest đơn giản: với mỗi nến trong eval_bars nến cuối,
    tạo signal và giả lập kết quả SL/TP cố định.
    """
    null = BacktestResult(strategy.name, 0, 0, 0, 0, -999, 0)

    n = len(df)
    start = max(80, n - eval_bars)

    wins, losses = [], []

    for i in range(start, n - 3):  # -3 để có nến sau để tính kết quả
        sub_df       = df.iloc[:i].copy()
        sub_trend    = df_trend.iloc[:max(1, len(df_trend) * i // n)].copy()
        sub_macro    = df_macro.iloc[:max(1, len(df_macro) * i // n)].copy()

        if len(sub_df) < 50:
            continue

        try:
            signal = strategy.generate_signal(sub_df, sub_trend, sub_macro)
        except Exception:
            continue

        if signal.direction == 0 or signal.atr == 0:
            continue

        entry = signal.entry_price
        atr   = signal.atr
        sl    = entry - signal.direction * sl_atr * atr
        tp    = entry + signal.direction * tp_atr * atr

        # Kiểm tra 3 nến tiếp theo xem đạt SL hay TP trước
        future = df.iloc[i:i+4]
        hit_tp = hit_sl = False

        for _, row in future.iterrows():
            if signal.direction == 1:
                if row["high"] >= tp:
                    hit_tp = True; break
                if row["low"] <= sl:
                    hit_sl = True; break
            else:
                if row["low"] <= tp:
                    hit_tp = True; break
                if row["high"] >= sl:
                    hit_sl = True; break

        if hit_tp:
            wins.append(abs(tp - entry) / entry * 100)
        elif hit_sl:
            losses.append(abs(entry - sl) / entry * 100)

    total = len(wins) + len(losses)
    if total < 5:
        return null

    win_rate = len(wins) / total
    avg_win  = np.mean(wins)  if wins   else 0
    avg_loss = np.mean(losses) if losses else 0

    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss
    profit_factor = (win_rate * avg_win) / ((1 - win_rate) * avg_loss + 1e-9)

    return BacktestResult(
        strategy_name=strategy.name,
        trades=total,
        win_rate=win_rate,
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss,
        expectancy=expectancy,
        profit_factor=profit_factor,
    )


class StrategySelector:
    def __init__(self):
        self.strategies = ALL_STRATEGIES
        # Cache: symbol → (best_strategy, counter)
        self._cache: dict[str, tuple[BaseStrategy, int]] = {}

    def select(
        self,
        symbol: str,
        df: pd.DataFrame,
        df_trend: pd.DataFrame,
        df_macro: pd.DataFrame,
    ) -> tuple[Optional[BaseStrategy], Optional[BacktestResult]]:
        """
        Trả về strategy tốt nhất cho symbol tại thời điểm này.
        Dùng cache để không backtest lại mỗi vòng lặp.
        """
        cache_entry = self._cache.get(symbol)
        if cache_entry:
            strategy, counter = cache_entry
            counter -= 1
            if counter > 0:
                self._cache[symbol] = (strategy, counter)
                return strategy, None
            # Hết cache, chạy lại
            del self._cache[symbol]

        results: list[BacktestResult] = []
        for strat in self.strategies:
            try:
                r = quick_backtest(strat, df, df_trend, df_macro,
                                   config.STRATEGY_EVAL_CANDLES)
                results.append(r)
            except Exception as e:
                logger.debug(f"Backtest error {strat.name} on {symbol}: {e}")

        valid = [r for r in results
                 if r.trades >= 5
                 and r.win_rate >= config.MIN_WIN_RATE
                 and r.expectancy > 0]

        if not valid:
            logger.debug(f"{symbol}: No valid strategy (all below threshold)")
            return None, None

        best_result = max(valid, key=lambda r: r.expectancy)
        best_strategy = next(s for s in self.strategies if s.name == best_result.strategy_name)

        self._cache[symbol] = (best_strategy, config.STRATEGY_RESCAN_BARS)

        logger.info(
            f"{symbol} → best strategy: {best_result.strategy_name} | "
            f"WR={best_result.win_rate:.0%} | "
            f"E={best_result.expectancy:.3f}% | "
            f"PF={best_result.profit_factor:.2f} | "
            f"trades={best_result.trades}"
        )
        return best_strategy, best_result
