"""Bar-by-bar backtest engine for the strategy in bot/strategy.py.

Simulates: taker fees on entry+exit, slippage, funding-fee approximation,
risk-based position sizing (bot/risk.py), and a two-stage exit model
(partial close at TP1 + move stop to breakeven, remainder targets TP2 or
gets stopped at breakeven). This is intentionally the same code path
structure the live bot uses to evaluate signals, so the report below
reflects what the bot would have done, not a hypothetical.

Same-bar ambiguity: OHLC bars don't tell us whether the stop or a target
was touched first intra-bar. We resolve this conservatively — stop-loss
is always checked before targets — so results are not inflated by
assuming favorable intra-bar order.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from bot import risk, strategy

TAKER_FEE = risk.TAKER_FEE_PCT           # Bybit USDT perpetual taker fee (approx)
SLIPPAGE_PCT = risk.ASSUMED_SLIPPAGE_PCT  # assumed adverse slippage per fill
FUNDING_PER_8H = 0.0001    # rough average funding assumption while a trade is open
TP1_CLOSE_FRACTION = 0.5   # fraction of position closed at TP1


@dataclass
class Trade:
    side: str
    entry_time: pd.Timestamp
    entry: float
    stop: float
    tp1: float
    tp2: float
    confidence: float
    leverage: int
    equity_risk_pct: float
    margin_used: float
    qty_initial: float
    entry_high: float
    entry_low: float
    qty_remaining: float = 0.0
    tp1_hit: bool = False
    realized_pnl_usd: float = 0.0
    high_water: float = 0.0
    low_water: float = 0.0
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    pnl_usd: float = 0.0
    r_multiple: float = 0.0

    def __post_init__(self) -> None:
        self.qty_remaining = self.qty_initial
        self.high_water = self.entry_high
        self.low_water = self.entry_low


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[tuple[pd.Timestamp, float]] = field(default_factory=list)
    starting_equity: float = 10_000.0
    ending_equity: float = 10_000.0

    def summary(self) -> dict:
        closed = [t for t in self.trades if t.exit_price is not None]
        if not closed:
            return {"trades": 0}
        wins = [t for t in closed if t.pnl_usd > 0]
        losses = [t for t in closed if t.pnl_usd <= 0]
        gross_win = sum(t.pnl_usd for t in wins)
        gross_loss = -sum(t.pnl_usd for t in losses)
        equity_series = pd.Series([e for _, e in self.equity_curve])
        running_max = equity_series.cummax()
        drawdown = (equity_series - running_max) / running_max
        return {
            "trades": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": round(100.0 * len(wins) / len(closed), 2),
            "avg_r": round(float(np.mean([t.r_multiple for t in closed])), 3),
            "profit_factor": round(gross_win / gross_loss, 3) if gross_loss > 0 else float("inf"),
            "total_pnl_usd": round(sum(t.pnl_usd for t in closed), 2),
            "starting_equity": self.starting_equity,
            "ending_equity": round(self.ending_equity, 2),
            "return_pct": round(100.0 * (self.ending_equity / self.starting_equity - 1.0), 2),
            "max_drawdown_pct": round(float(drawdown.min()) * 100.0, 2),
        }


def _fill_price(price: float, side: str, is_entry: bool) -> float:
    """Apply adverse slippage. Entries slip against us in the trade
    direction; exits slip against us in the opposite direction.
    """
    adverse_up = (side == "long") == is_entry
    slip = price * SLIPPAGE_PCT
    return price + slip if adverse_up else price - slip


def _fees(qty: float, price_a: float, price_b: float) -> float:
    return TAKER_FEE * qty * (price_a + price_b)


def run_backtest(df_1m: pd.DataFrame, starting_equity: float = 10_000.0) -> BacktestResult:
    df = strategy.prepare(df_1m)
    result = BacktestResult(starting_equity=starting_equity)
    equity = starting_equity
    open_trade: Trade | None = None
    daily_start_equity = starting_equity
    current_day = None
    halted_for_day = False
    just_closed_ts = None

    for _, row in df.iterrows():
        ts = row["datetime"]
        day = ts.date()
        if day != current_day:
            current_day = day
            daily_start_equity = equity
            halted_for_day = False

        if open_trade is not None:
            long = open_trade.side == "long"
            hit_stop = (row["low"] <= open_trade.stop) if long else (row["high"] >= open_trade.stop)
            hit_tp2 = (row["high"] >= open_trade.tp2) if long else (row["low"] <= open_trade.tp2)
            hit_tp1 = (
                (not open_trade.tp1_hit)
                and ((row["high"] >= open_trade.tp1) if long else (row["low"] <= open_trade.tp1))
            )
            trend_flip = row["trend"] != ("up" if long else "down")

            if hit_stop:
                raw = open_trade.stop
                fill = _fill_price(raw, open_trade.side, is_entry=False)
                gross = (fill - open_trade.entry) * open_trade.qty_remaining if long else (
                    open_trade.entry - fill
                ) * open_trade.qty_remaining
                fees = _fees(open_trade.qty_remaining, open_trade.entry, fill)
                funding = FUNDING_PER_8H * (open_trade.entry * open_trade.qty_remaining)
                pnl = gross - fees - funding
                total_pnl = open_trade.realized_pnl_usd + pnl
                risk_amount = abs(open_trade.entry - open_trade.stop) * open_trade.qty_initial

                open_trade.exit_time = ts
                open_trade.exit_price = fill
                open_trade.exit_reason = "stop_loss" if not open_trade.tp1_hit else "breakeven_stop"
                open_trade.pnl_usd = total_pnl
                open_trade.r_multiple = total_pnl / risk_amount if risk_amount > 0 else 0.0
                equity += pnl
                result.trades.append(open_trade)
                open_trade = None
                just_closed_ts = ts

            elif hit_tp2:
                raw = open_trade.tp2
                fill = _fill_price(raw, open_trade.side, is_entry=False)
                gross = (fill - open_trade.entry) * open_trade.qty_remaining if long else (
                    open_trade.entry - fill
                ) * open_trade.qty_remaining
                fees = _fees(open_trade.qty_remaining, open_trade.entry, fill)
                funding = FUNDING_PER_8H * (open_trade.entry * open_trade.qty_remaining)
                pnl = gross - fees - funding
                total_pnl = open_trade.realized_pnl_usd + pnl
                risk_amount = abs(open_trade.entry - open_trade.stop) * open_trade.qty_initial

                open_trade.exit_time = ts
                open_trade.exit_price = fill
                open_trade.exit_reason = "take_profit_2"
                open_trade.pnl_usd = total_pnl
                open_trade.r_multiple = total_pnl / risk_amount if risk_amount > 0 else 0.0
                equity += pnl
                result.trades.append(open_trade)
                open_trade = None
                just_closed_ts = ts

            elif hit_tp1:
                close_qty = open_trade.qty_initial * TP1_CLOSE_FRACTION
                raw = open_trade.tp1
                fill = _fill_price(raw, open_trade.side, is_entry=False)
                gross = (fill - open_trade.entry) * close_qty if long else (open_trade.entry - fill) * close_qty
                fees = _fees(close_qty, open_trade.entry, fill)
                pnl = gross - fees
                open_trade.realized_pnl_usd += pnl
                open_trade.qty_remaining -= close_qty
                open_trade.tp1_hit = True
                open_trade.stop = open_trade.entry  # move stop to breakeven
                equity += pnl

            elif trend_flip:
                raw = row["close"]
                fill = _fill_price(raw, open_trade.side, is_entry=False)
                gross = (fill - open_trade.entry) * open_trade.qty_remaining if long else (
                    open_trade.entry - fill
                ) * open_trade.qty_remaining
                fees = _fees(open_trade.qty_remaining, open_trade.entry, fill)
                funding = FUNDING_PER_8H * (open_trade.entry * open_trade.qty_remaining)
                pnl = gross - fees - funding
                total_pnl = open_trade.realized_pnl_usd + pnl
                risk_amount = abs(open_trade.entry - open_trade.stop) * open_trade.qty_initial

                open_trade.exit_time = ts
                open_trade.exit_price = fill
                open_trade.exit_reason = "trend_flip"
                open_trade.pnl_usd = total_pnl
                open_trade.r_multiple = total_pnl / risk_amount if risk_amount > 0 else 0.0
                equity += pnl
                result.trades.append(open_trade)
                open_trade = None
                just_closed_ts = ts

            elif not open_trade.tp1_hit:
                # Capital-efficiency exit: a position that hasn't hit TP1
                # (so its stop is still the original one, not breakeven)
                # and hasn't meaningfully progressed after many bars is
                # closed at market instead of tying up margin indefinitely.
                risk_distance = abs(open_trade.entry - open_trade.stop)
                unrealized_r = (
                    (row["close"] - open_trade.entry) / risk_distance
                    if long
                    else (open_trade.entry - row["close"]) / risk_distance
                ) if risk_distance > 0 else 0.0
                elapsed_bars = (ts - open_trade.entry_time) / pd.Timedelta(minutes=15)

                if elapsed_bars >= risk.STALE_POSITION_MAX_BARS and unrealized_r < risk.STALE_POSITION_MIN_R:
                    raw = row["close"]
                    fill = _fill_price(raw, open_trade.side, is_entry=False)
                    gross = (fill - open_trade.entry) * open_trade.qty_remaining if long else (
                        open_trade.entry - fill
                    ) * open_trade.qty_remaining
                    fees = _fees(open_trade.qty_remaining, open_trade.entry, fill)
                    funding = FUNDING_PER_8H * (open_trade.entry * open_trade.qty_remaining)
                    pnl = gross - fees - funding
                    total_pnl = open_trade.realized_pnl_usd + pnl
                    risk_amount = risk_distance * open_trade.qty_initial

                    open_trade.exit_time = ts
                    open_trade.exit_price = fill
                    open_trade.exit_reason = "stale_position"
                    open_trade.pnl_usd = total_pnl
                    open_trade.r_multiple = total_pnl / risk_amount if risk_amount > 0 else 0.0
                    equity += pnl
                    result.trades.append(open_trade)
                    open_trade = None
                    just_closed_ts = ts

            # Chandelier trailing stop on the runner, only after TP1 has
            # locked in breakeven. Uses THIS bar's high/low to update the
            # water-mark, so the resulting stop only takes effect from the
            # NEXT bar onward — never tested against the same bar's H/L.
            if open_trade is not None and open_trade.tp1_hit:
                atr_now = row["atr14"]
                if long:
                    open_trade.high_water = max(open_trade.high_water, row["high"])
                    candidate = open_trade.high_water - strategy.ATR_TRAIL_MULT * atr_now
                    open_trade.stop = max(open_trade.stop, candidate)
                else:
                    open_trade.low_water = min(open_trade.low_water, row["low"])
                    candidate = open_trade.low_water + strategy.ATR_TRAIL_MULT * atr_now
                    open_trade.stop = min(open_trade.stop, candidate)

        result.equity_curve.append((ts, equity))

        if halted_for_day or equity <= 0:
            continue
        if (equity - daily_start_equity) / daily_start_equity <= -risk.DAILY_LOSS_LIMIT_PCT:
            halted_for_day = True
            continue
        if open_trade is not None:
            continue

        sig = strategy.signal_from_row(row)
        if sig is None:
            continue
        if just_closed_ts == ts:
            continue  # don't re-enter on the exact bar we just closed on

        atr_pct = row["atr14"] / row["close"] if row["close"] else 0.0
        plan = risk.plan_position(equity, sig.entry, sig.stop, sig.confidence, atr_pct)
        if plan is None or plan.qty <= 0:
            continue

        fill_entry = _fill_price(sig.entry, sig.side, is_entry=True)

        open_trade = Trade(
            side=sig.side,
            entry_time=ts,
            entry=fill_entry,
            stop=sig.stop,
            tp1=sig.take_profit_1,
            tp2=sig.take_profit_2,
            confidence=sig.confidence,
            leverage=plan.leverage,
            equity_risk_pct=plan.equity_risk_pct,
            margin_used=plan.margin_used,
            qty_initial=plan.qty,
            entry_high=row["high"],
            entry_low=row["low"],
        )

    result.ending_equity = equity
    return result
