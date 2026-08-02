"""Multi-symbol live scanner: applies the exact same strategy.py signal
logic used in the backtest to every configured symbol, manages any open
position's trailing stop / partial take-profit, and opens new positions
only when a signal + risk plan both clear (bot/strategy.py, bot/risk.py).

A symbol with no matching setup is simply left alone — there is no
"force a trade" path anywhere in this module, matching the requirement
that the bot only acts when a symbol's price action actually matches the
trained logic.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from bot import risk, strategy
from bot.config import Config
from bot.exchange_bybit import BybitExchange

logger = logging.getLogger("bybit_bot.scanner")

KLINES_1M_LIMIT = 2000  # EMA span=315 needs ~5x that many bars to actually converge
KLINES_1H_LIMIT = 300


@dataclass
class PositionState:
    symbol: str
    side: str
    entry: float
    stop: float
    tp1: float
    tp2: float
    qty_initial: float
    qty_remaining: float
    leverage: int
    entry_time: pd.Timestamp
    entry_trend: str
    tp1_hit: bool = False
    high_water: float = 0.0
    low_water: float = 0.0


@dataclass
class Scanner:
    exchange: BybitExchange
    config: Config
    open_positions: dict[str, PositionState] = field(default_factory=dict)
    daily_start_equity: float | None = None
    current_day: object = None
    halted_for_day: bool = False

    def _latest_closed_frame(self, symbol: str) -> pd.DataFrame | None:
        try:
            df_1m = self.exchange.get_klines(symbol, "1m", KLINES_1M_LIMIT)
            df_1h = self.exchange.get_klines(symbol, "1h", KLINES_1H_LIMIT)
        except Exception:
            logger.exception("Failed to fetch klines for %s", symbol)
            return None
        if len(df_1m) < 2 or len(df_1h) < 2:
            return None
        # Drop the last row of each: it's the still-forming candle.
        df_1m = df_1m.iloc[:-1].reset_index(drop=True)
        df_1h = df_1h.iloc[:-1].reset_index(drop=True)
        merged = strategy.prepare_from_ltf_htf(df_1m, df_1h)
        if merged.empty:
            return None
        return merged

    def _check_daily_halt(self, equity: float) -> bool:
        today = pd.Timestamp.now("UTC").date()
        if today != self.current_day:
            self.current_day = today
            self.daily_start_equity = equity
            self.halted_for_day = False
        if self.daily_start_equity and self.daily_start_equity > 0:
            dd = (equity - self.daily_start_equity) / self.daily_start_equity
            if dd <= -risk.DAILY_LOSS_LIMIT_PCT:
                if not self.halted_for_day:
                    logger.warning(
                        "Daily loss limit hit (%.2f%%) — halting new entries until next day",
                        dd * 100,
                    )
                self.halted_for_day = True
        return self.halted_for_day

    def _manage_position(self, symbol: str, row: pd.Series) -> None:
        state = self.open_positions[symbol]
        long = state.side == "long"
        # Entries no longer require a matching HTF trend (trend is a
        # confidence modifier now, not a gate — see strategy.py), so a
        # "flat" HTF reading isn't a flip, and neither is "still opposite"
        # for a trade that was deliberately opened counter-trend (already
        # priced into its lower confidence/size at entry) — only exit here
        # if the trend has actively reversed relative to entry time.
        opposite_trend = "down" if long else "up"

        if row["trend"] == opposite_trend and state.entry_trend != opposite_trend:
            logger.info("%s: trend flipped against %s, closing remaining position", symbol, state.side)
            self.exchange.close_position_market(symbol, state.side, state.qty_remaining)
            del self.open_positions[symbol]
            return

        if not state.tp1_hit:
            hit_tp1 = (row["high"] >= state.tp1) if long else (row["low"] <= state.tp1)
            if hit_tp1:
                close_qty = self.exchange.round_qty(symbol, state.qty_initial * 0.5)
                if close_qty > 0:
                    self.exchange.close_position_market(symbol, state.side, close_qty)
                    state.qty_remaining -= close_qty
                state.tp1_hit = True
                state.stop = state.entry
                self.exchange.update_stop_loss(symbol, state.stop)
                logger.info("%s: TP1 hit, closed %.6f, stop moved to breakeven %.6f", symbol, close_qty, state.stop)
                return  # re-evaluate trailing next poll

            # Capital-efficiency exit: hasn't reached TP1 (so risk distance
            # is still the original stop) and hasn't meaningfully
            # progressed after many bars — free the margin instead of
            # tying it up indefinitely waiting for a move that may not come.
            risk_distance = abs(state.entry - state.stop)
            if risk_distance > 0:
                unrealized_r = (
                    (row["close"] - state.entry) / risk_distance
                    if long
                    else (state.entry - row["close"]) / risk_distance
                )
                elapsed_minutes = (row["datetime"] - state.entry_time) / pd.Timedelta(minutes=1)
                if elapsed_minutes >= risk.STALE_POSITION_MAX_MINUTES and unrealized_r < risk.STALE_POSITION_MIN_R:
                    logger.info(
                        "%s: stale position (%.0f min, %.2fR unrealized), closing to free capital",
                        symbol, elapsed_minutes, unrealized_r,
                    )
                    self.exchange.close_position_market(symbol, state.side, state.qty_remaining)
                    del self.open_positions[symbol]
                    return

        if state.tp1_hit:
            atr_now = row["atr14"]
            if long:
                state.high_water = max(state.high_water, row["high"])
                candidate = state.high_water - strategy.ATR_TRAIL_MULT * atr_now
                new_stop = max(state.stop, candidate)
            else:
                state.low_water = min(state.low_water, row["low"])
                candidate = state.low_water + strategy.ATR_TRAIL_MULT * atr_now
                new_stop = min(state.stop, candidate)
            if new_stop != state.stop:
                state.stop = new_stop
                self.exchange.update_stop_loss(symbol, state.stop)
                logger.info("%s: trailing stop -> %.6f", symbol, state.stop)

    def _try_open(self, symbol: str, row: pd.Series, equity: float) -> None:
        sig = strategy.signal_from_row(row)
        if sig is None:
            return

        atr_pct = row["atr14"] / row["close"] if row["close"] else 0.0
        plan = risk.plan_position(equity, sig.entry, sig.stop, sig.confidence, atr_pct)
        if plan is None:
            logger.debug("%s: signal below min-confidence tier, skipping", symbol)
            return

        info = self.exchange.get_instrument_info(symbol)
        qty = self.exchange.round_qty(symbol, plan.qty)
        if qty < info.min_order_qty:
            logger.info("%s: sized qty %.8f below exchange minimum %.8f, skipping", symbol, qty, info.min_order_qty)
            return

        leverage = min(plan.leverage, int(info.max_leverage))
        self.exchange.set_leverage(symbol, leverage)
        self.exchange.place_market_entry_with_stop(symbol, sig.side, qty, sig.stop)

        self.open_positions[symbol] = PositionState(
            symbol=symbol,
            side=sig.side,
            entry=sig.entry,
            stop=sig.stop,
            tp1=sig.take_profit_1,
            tp2=sig.take_profit_2,
            qty_initial=qty,
            qty_remaining=qty,
            leverage=leverage,
            entry_time=row["datetime"],
            entry_trend=row["trend"],
            high_water=row["high"],
            low_water=row["low"],
        )
        logger.info(
            "%s: OPENED %s entry=%.6f stop=%.6f conf=%.0f risk%%=%.1f lev=%sx (%s)",
            symbol, sig.side.upper(), sig.entry, sig.stop, sig.confidence,
            plan.equity_risk_pct * 100, leverage, sig.reason,
        )

    def run_once(self) -> None:
        equity = self.exchange.get_wallet_equity_usdt()
        halted = self._check_daily_halt(equity)

        for symbol in self.config.symbols:
            merged = self._latest_closed_frame(symbol)
            if merged is None:
                continue
            row = merged.iloc[-1]

            if symbol in self.open_positions:
                try:
                    self._manage_position(symbol, row)
                except Exception:
                    logger.exception("Error managing open position for %s", symbol)
                continue

            if halted:
                continue
            if len(self.open_positions) >= self.config.max_concurrent_positions:
                continue
            try:
                self._try_open(symbol, row, equity)
            except Exception:
                logger.exception("Error evaluating entry for %s", symbol)
