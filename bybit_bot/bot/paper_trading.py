"""Paper-trading mode: runs the exact same signal/risk logic as the live
bot (bot/strategy.py, bot/risk.py) against REAL real-time Bybit market
data (public REST endpoints — klines + tickers, no API key required),
but against a virtual account instead of placing real orders.

This exists to let you watch the strategy trade in real time on real
price action, with realistic simulated fees/slippage and real TP/SL
levels, before ever risking real money. It is NOT a demo with fabricated
prices — every candle and every tick it trades on came from Bybit.

Because this never places a real order, it does not need to respect
Bybit's real minOrderQty/lot-size — those only matter when an order is
actually submitted. A $10 virtual account can therefore still open
(fractional, unrealistic-to-execute-for-real) positions sized purely by
the risk model, which is the point: you're testing the DECISION logic,
not exchange order mechanics. Use a larger --equity if you want position
sizes that would also be realistically tradable on a real account.
"""
from __future__ import annotations

import argparse
import logging
import signal
import time
from dataclasses import dataclass

import pandas as pd

from bot import risk, strategy
from bot.config import CONFIG
from bot.exchange_bybit import BybitExchange
from bot.main import setup_logging

logger = logging.getLogger("bybit_bot.paper")

PRICE_POLL_SECONDS = 15    # how often to check open positions against live price
SIGNAL_POLL_SECONDS = 60   # how often to refresh trend / look for new entries
KLINES_15M_LIMIT = 200
KLINES_1H_LIMIT = 300


@dataclass
class PaperPosition:
    symbol: str
    side: str
    entry: float
    stop: float
    tp1: float
    tp2: float
    qty_initial: float
    leverage: int
    confidence: float
    entry_time: pd.Timestamp
    qty_remaining: float = 0.0
    tp1_hit: bool = False
    realized_pnl_usd: float = 0.0
    high_water: float = 0.0
    low_water: float = 0.0

    def __post_init__(self) -> None:
        self.qty_remaining = self.qty_initial
        self.high_water = self.entry
        self.low_water = self.entry


@dataclass
class ClosedPaperTrade:
    symbol: str
    side: str
    entry: float
    exit_price: float
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    exit_reason: str
    pnl_usd: float
    r_multiple: float


def _fees(qty: float, price_a: float, price_b: float) -> float:
    return risk.TAKER_FEE_PCT * qty * (price_a + price_b)


def _fill_price(price: float, side: str, is_entry: bool) -> float:
    adverse_up = (side == "long") == is_entry
    slip = price * risk.ASSUMED_SLIPPAGE_PCT
    return price + slip if adverse_up else price - slip


class PaperBroker:
    def __init__(self, exchange: BybitExchange, config, starting_equity: float):
        self.exchange = exchange
        self.config = config
        self.equity = starting_equity
        self.starting_equity = starting_equity
        self.open_positions: dict[str, PaperPosition] = {}
        self.closed_trades: list[ClosedPaperTrade] = []

    def _merged_frame(self, symbol: str) -> pd.DataFrame | None:
        try:
            df_15m = self.exchange.get_klines(symbol, "15m", KLINES_15M_LIMIT)
            df_1h = self.exchange.get_klines(symbol, "1h", KLINES_1H_LIMIT)
        except Exception:
            logger.exception("Failed to fetch klines for %s", symbol)
            return None
        if len(df_15m) < 2 or len(df_1h) < 2:
            return None
        df_15m = df_15m.iloc[:-1].reset_index(drop=True)
        df_1h = df_1h.iloc[:-1].reset_index(drop=True)
        merged = strategy.prepare_from_ltf_htf(df_15m, df_1h)
        return merged if not merged.empty else None

    def try_open(self, symbol: str) -> None:
        merged = self._merged_frame(symbol)
        if merged is None:
            return
        row = merged.iloc[-1]
        sig = strategy.signal_from_row(row)
        if sig is None:
            return

        atr_pct = row["atr14"] / row["close"] if row["close"] else 0.0
        plan = risk.plan_position(self.equity, sig.entry, sig.stop, sig.confidence, atr_pct)
        if plan is None or plan.qty <= 0:
            return

        try:
            last_price = self.exchange.get_last_price(symbol)
        except Exception:
            logger.exception("Failed to fetch last price for %s", symbol)
            return

        fill = _fill_price(last_price, sig.side, is_entry=True)
        entry_fee = risk.TAKER_FEE_PCT * plan.qty * fill
        self.equity -= entry_fee

        self.open_positions[symbol] = PaperPosition(
            symbol=symbol,
            side=sig.side,
            entry=fill,
            stop=sig.stop,
            tp1=sig.take_profit_1,
            tp2=sig.take_profit_2,
            qty_initial=plan.qty,
            leverage=plan.leverage,
            confidence=sig.confidence,
            entry_time=pd.Timestamp.now(tz="UTC"),
        )
        logger.info(
            "%s: OPENED %s @%.6f stop=%.6f tp1=%.6f conf=%.0f risk%%=%.1f lev=%sx (%s)",
            symbol, sig.side.upper(), fill, sig.stop, sig.take_profit_1,
            sig.confidence, plan.equity_risk_pct * 100, plan.leverage, sig.reason,
        )

    def _close(self, symbol: str, price: float, reason: str) -> None:
        pos = self.open_positions.pop(symbol)
        long = pos.side == "long"
        fill = _fill_price(price, pos.side, is_entry=False)
        gross = (fill - pos.entry) * pos.qty_remaining if long else (pos.entry - fill) * pos.qty_remaining
        fees = _fees(pos.qty_remaining, pos.entry, fill)
        pnl = gross - fees
        total_pnl = pos.realized_pnl_usd + pnl
        risk_amount = abs(pos.entry - pos.stop) * pos.qty_initial
        r_multiple = total_pnl / risk_amount if risk_amount > 0 else 0.0

        self.equity += pnl
        self.closed_trades.append(
            ClosedPaperTrade(
                symbol=symbol, side=pos.side, entry=pos.entry, exit_price=fill,
                entry_time=pos.entry_time, exit_time=pd.Timestamp.now(tz="UTC"),
                exit_reason=reason, pnl_usd=total_pnl, r_multiple=r_multiple,
            )
        )
        logger.info(
            "%s: CLOSED %s @%.6f (%s) pnl=$%.4f R=%.2f equity=$%.4f",
            symbol, pos.side.upper(), fill, reason, total_pnl, r_multiple, self.equity,
        )

    def manage_with_price(self, symbol: str, price: float) -> None:
        pos = self.open_positions.get(symbol)
        if pos is None:
            return
        long = pos.side == "long"

        hit_stop = (price <= pos.stop) if long else (price >= pos.stop)
        if hit_stop:
            self._close(symbol, pos.stop, "stop_loss" if not pos.tp1_hit else "breakeven_stop")
            return

        hit_tp2 = (price >= pos.tp2) if long else (price <= pos.tp2)
        if hit_tp2:
            self._close(symbol, pos.tp2, "take_profit_2")
            return

        if not pos.tp1_hit:
            hit_tp1 = (price >= pos.tp1) if long else (price <= pos.tp1)
            if hit_tp1:
                close_qty = pos.qty_initial * 0.5
                fill = _fill_price(pos.tp1, pos.side, is_entry=False)
                gross = (fill - pos.entry) * close_qty if long else (pos.entry - fill) * close_qty
                fees = _fees(close_qty, pos.entry, fill)
                pnl = gross - fees
                pos.realized_pnl_usd += pnl
                pos.qty_remaining -= close_qty
                pos.tp1_hit = True
                pos.stop = pos.entry
                self.equity += pnl
                logger.info("%s: TP1 hit @%.6f, closed half, stop -> breakeven %.6f", symbol, fill, pos.stop)
                return

        if pos.tp1_hit:
            if long:
                pos.high_water = max(pos.high_water, price)
            else:
                pos.low_water = min(pos.low_water, price)

    def refresh_trend_and_trail(self, symbol: str) -> None:
        """Slower-cadence check: trend flip, ATR-based trailing stop
        update, and the stale-position capital-efficiency exit — these
        all need the 15m/1h indicator frame, not just a live price tick.
        """
        pos = self.open_positions.get(symbol)
        if pos is None:
            return
        merged = self._merged_frame(symbol)
        if merged is None:
            return
        row = merged.iloc[-1]
        long = pos.side == "long"
        expected_trend = "up" if long else "down"

        if row["trend"] != expected_trend:
            try:
                price = self.exchange.get_last_price(symbol)
            except Exception:
                logger.exception("Failed to fetch last price for %s", symbol)
                return
            self._close(symbol, price, "trend_flip")
            return

        if pos.tp1_hit:
            atr_now = row["atr14"]
            if long:
                candidate = pos.high_water - strategy.ATR_TRAIL_MULT * atr_now
                pos.stop = max(pos.stop, candidate)
            else:
                candidate = pos.low_water + strategy.ATR_TRAIL_MULT * atr_now
                pos.stop = min(pos.stop, candidate)
        else:
            risk_distance = abs(pos.entry - pos.stop)
            if risk_distance > 0:
                try:
                    price = self.exchange.get_last_price(symbol)
                except Exception:
                    return
                unrealized_r = (
                    (price - pos.entry) / risk_distance if long else (pos.entry - price) / risk_distance
                )
                elapsed_bars = (pd.Timestamp.now(tz="UTC") - pos.entry_time) / pd.Timedelta(minutes=15)
                if elapsed_bars >= risk.STALE_POSITION_MAX_BARS and unrealized_r < risk.STALE_POSITION_MIN_R:
                    self._close(symbol, price, "stale_position")

    def summary(self) -> dict:
        closed = self.closed_trades
        wins = [t for t in closed if t.pnl_usd > 0]
        losses = [t for t in closed if t.pnl_usd <= 0]
        realized_pnl = sum(t.pnl_usd for t in closed)

        open_rows = []
        unrealized_total = 0.0
        open_wins = 0
        open_losses = 0
        for symbol, pos in self.open_positions.items():
            try:
                price = self.exchange.get_last_price(symbol)
            except Exception:
                price = pos.entry
            long = pos.side == "long"
            gross = (price - pos.entry) * pos.qty_remaining if long else (pos.entry - price) * pos.qty_remaining
            unrealized = pos.realized_pnl_usd + gross
            unrealized_total += unrealized
            if unrealized > 0:
                open_wins += 1
            else:
                open_losses += 1
            open_rows.append((symbol, pos.side, pos.entry, price, unrealized))

        return {
            "closed_trades": len(closed),
            "closed_wins": len(wins),
            "closed_losses": len(losses),
            "realized_pnl_usd": realized_pnl,
            "open_positions": len(self.open_positions),
            "open_currently_winning": open_wins,
            "open_currently_losing": open_losses,
            "unrealized_pnl_usd": unrealized_total,
            "open_rows": open_rows,
            "starting_equity": self.starting_equity,
            "realized_equity": self.equity,
            "equity_incl_unrealized": self.equity + unrealized_total,
        }


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 60)
    print("PAPER TRADING SESSION SUMMARY")
    print("=" * 60)
    print(f"Starting virtual equity : ${summary['starting_equity']:.4f}")
    print(f"Closed trades           : {summary['closed_trades']}"
          f" (win {summary['closed_wins']} / loss {summary['closed_losses']})")
    print(f"Realized P&L            : ${summary['realized_pnl_usd']:+.4f}")
    print(f"Open positions now      : {summary['open_positions']}"
          f" (currently winning {summary['open_currently_winning']}"
          f" / currently losing {summary['open_currently_losing']})")
    for symbol, side, entry, price, unrealized in summary["open_rows"]:
        print(f"    {symbol:10s} {side.upper():5s} entry={entry:.4f} last={price:.4f}"
              f" unrealized=${unrealized:+.4f}")
    print(f"Unrealized P&L          : ${summary['unrealized_pnl_usd']:+.4f}")
    print(f"Equity (realized only)  : ${summary['realized_equity']:.4f}")
    print(f"Equity incl. unrealized : ${summary['equity_incl_unrealized']:.4f}")
    print("=" * 60)


def run(starting_equity: float, poll_seconds: int = PRICE_POLL_SECONDS) -> None:
    setup_logging(CONFIG.log_level)
    exchange = BybitExchange(CONFIG)
    broker = PaperBroker(exchange, CONFIG, starting_equity)

    logger.warning(
        "Paper trading started: virtual equity=$%.4f symbols=%s "
        "(real Bybit %s market data, NO real orders placed)",
        starting_equity, CONFIG.symbols, "TESTNET" if CONFIG.testnet else "MAINNET",
    )

    running = True

    def _stop(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    last_signal_check = 0.0
    try:
        while running:
            now = time.time()
            for symbol in CONFIG.symbols:
                if symbol in broker.open_positions:
                    try:
                        price = exchange.get_last_price(symbol)
                        broker.manage_with_price(symbol, price)
                    except Exception:
                        logger.exception("Error managing paper position for %s", symbol)

            if now - last_signal_check >= SIGNAL_POLL_SECONDS:
                last_signal_check = now
                for symbol in CONFIG.symbols:
                    try:
                        if symbol in broker.open_positions:
                            broker.refresh_trend_and_trail(symbol)
                        elif len(broker.open_positions) < CONFIG.max_concurrent_positions:
                            broker.try_open(symbol)
                    except Exception:
                        logger.exception("Error evaluating %s", symbol)

            for _ in range(poll_seconds):
                if not running:
                    break
                time.sleep(1)
    finally:
        print_summary(broker.summary())


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-trade against real Bybit market data")
    parser.add_argument("--equity", type=float, default=10.0, help="Starting virtual equity in USDT")
    parser.add_argument("--poll-seconds", type=int, default=PRICE_POLL_SECONDS)
    args = parser.parse_args()
    run(starting_equity=args.equity, poll_seconds=args.poll_seconds)


if __name__ == "__main__":
    main()
