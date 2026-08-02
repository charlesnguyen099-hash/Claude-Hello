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
SIGNAL_POLL_SECONDS = 60   # how often to refresh trend / look for new entries (matches a new 1m candle)
KLINES_1M_LIMIT = 2000  # EMA span=315 needs ~5x that many bars to actually converge
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
    entry_trend: str
    qty_remaining: float = 0.0
    tp1_hit: bool = False
    realized_pnl_usd: float = 0.0
    high_water: float = 0.0
    low_water: float = 0.0
    # Timestamp of the last CLOSED 1m candle already scanned for TP/SL
    # touches. The still-forming candle is deliberately re-scanned every
    # tick (its high/low keeps extending), which is safe because every
    # exit path below is guarded by a flag or removes the position.
    last_check_ts: pd.Timestamp | None = None

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
            df_1m = self.exchange.get_klines(symbol, "1m", KLINES_1M_LIMIT)
            df_1h = self.exchange.get_klines(symbol, "1h", KLINES_1H_LIMIT)
        except Exception:
            logger.exception("Failed to fetch klines for %s", symbol)
            return None
        if len(df_1m) < 2 or len(df_1h) < 2:
            return None
        df_1m = df_1m.iloc[:-1].reset_index(drop=True)
        df_1h = df_1h.iloc[:-1].reset_index(drop=True)
        merged = strategy.prepare_from_ltf_htf(df_1m, df_1h)
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
            entry_trend=row["trend"],
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

    def manage_with_candles(self, symbol: str) -> None:
        """Check TP/SL against the full high/low range of every 1m candle
        since the last check, not just the price at this instant.

        A real TP/SL order sits on the exchange book and fires the moment
        price *touches* it. Sampling `get_last_price` every N seconds
        misses fast wicks that touched a level and retraced, which
        silently skips stop-outs and makes paper results look better than
        the same logic would do live. Scanning candle high/low closes that
        gap, and matches how backtest/engine.py measures the same trade.
        """
        pos = self.open_positions.get(symbol)
        if pos is None:
            return
        try:
            candles = self.exchange.get_klines(symbol, "1m", 10)
        except Exception:
            logger.exception("Failed to fetch 1m candles for %s", symbol)
            return
        if candles.empty:
            return

        pending = candles if pos.last_check_ts is None else candles[
            candles["datetime"] > pos.last_check_ts
        ]
        for row in pending.itertuples():
            if symbol not in self.open_positions:
                break
            self._apply_range(symbol, float(row.high), float(row.low))

        still_open = self.open_positions.get(symbol)
        if still_open is not None and len(candles) >= 2:
            # Second-to-last is the newest *closed* candle; the last one
            # is still forming, so leave it eligible for re-scanning.
            still_open.last_check_ts = candles.iloc[-2]["datetime"]

    def manage_with_price(self, symbol: str, price: float) -> None:
        """Check the position against a single instantaneous price -- a
        zero-width range. Kept for callers that already hold a tick and
        don't need the candle fetch in manage_with_candles().
        """
        self._apply_range(symbol, price, price)

    def _apply_range(self, symbol: str, high: float, low: float) -> None:
        """Resolve one candle's range against the position's levels.

        Same-candle ambiguity (both the stop and a target inside one
        bar's range) is resolved conservatively in favour of the stop,
        identical to the backtest engine, so paper results never flatter
        the strategy relative to the backtest.
        """
        pos = self.open_positions.get(symbol)
        if pos is None:
            return
        long = pos.side == "long"

        hit_stop = (low <= pos.stop) if long else (high >= pos.stop)
        if hit_stop:
            self._close(symbol, pos.stop, "stop_loss" if not pos.tp1_hit else "breakeven_stop")
            return

        hit_tp2 = (high >= pos.tp2) if long else (low <= pos.tp2)
        if hit_tp2:
            self._close(symbol, pos.tp2, "take_profit_2")
            return

        if not pos.tp1_hit:
            hit_tp1 = (high >= pos.tp1) if long else (low <= pos.tp1)
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
                pos.high_water = max(pos.high_water, high)
            else:
                pos.low_water = min(pos.low_water, low)

    def refresh_trend_and_trail(self, symbol: str) -> None:
        """Slower-cadence check: trend flip, ATR-based trailing stop
        update, and the stale-position capital-efficiency exit — these
        all need the 1m/1h indicator frame, not just a live price tick.
        """
        pos = self.open_positions.get(symbol)
        if pos is None:
            return
        merged = self._merged_frame(symbol)
        if merged is None:
            return
        row = merged.iloc[-1]
        long = pos.side == "long"
        # Entries no longer require a matching HTF trend (trend is a
        # confidence modifier now, not a gate — see strategy.py), so a
        # "flat" HTF reading isn't a flip, and neither is "still opposite"
        # for a trade that was deliberately opened counter-trend (already
        # priced into its lower confidence/size at entry) — only exit here
        # if the trend has actively reversed relative to entry time.
        opposite_trend = "down" if long else "up"

        if row["trend"] == opposite_trend and pos.entry_trend != opposite_trend:
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
                elapsed_minutes = (pd.Timestamp.now(tz="UTC") - pos.entry_time) / pd.Timedelta(minutes=1)
                if elapsed_minutes >= risk.STALE_POSITION_MAX_MINUTES and unrealized_r < risk.STALE_POSITION_MIN_R:
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

        gross_profit = sum(t.pnl_usd for t in wins)
        gross_loss = sum(t.pnl_usd for t in losses)
        open_profit = sum(u for *_, u in open_rows if u > 0)
        open_loss = sum(u for *_, u in open_rows if u <= 0)

        return {
            "closed_trades": len(closed),
            "closed_wins": len(wins),
            "closed_losses": len(losses),
            "gross_profit_usd": gross_profit,
            "gross_loss_usd": gross_loss,
            "realized_pnl_usd": realized_pnl,
            "open_positions": len(self.open_positions),
            "open_currently_winning": open_wins,
            "open_currently_losing": open_losses,
            "open_profit_usd": open_profit,
            "open_loss_usd": open_loss,
            "unrealized_pnl_usd": unrealized_total,
            "open_rows": open_rows,
            # Combined view: closed trades + still-open positions marked
            # to market, which is what "how many winners / losers do I
            # have in total right now" actually means at shutdown.
            "total_trades": len(closed) + len(self.open_positions),
            "total_winning": len(wins) + open_wins,
            "total_losing": len(losses) + open_losses,
            "total_profit_usd": gross_profit + open_profit,
            "total_loss_usd": gross_loss + open_loss,
            "starting_equity": self.starting_equity,
            "realized_equity": self.equity,
            "equity_incl_unrealized": self.equity + unrealized_total,
        }

    def export_trades_csv(self, path: str) -> None:
        rows = [
            {
                "symbol": t.symbol, "side": t.side, "status": "closed",
                "entry": t.entry, "exit_or_last": t.exit_price,
                "entry_time": t.entry_time, "exit_time": t.exit_time,
                "reason": t.exit_reason, "pnl_usd": t.pnl_usd, "r_multiple": t.r_multiple,
            }
            for t in self.closed_trades
        ]
        for symbol, side, entry, price, unrealized in self.summary()["open_rows"]:
            pos = self.open_positions[symbol]
            rows.append({
                "symbol": symbol, "side": side, "status": "open_at_shutdown",
                "entry": entry, "exit_or_last": price,
                "entry_time": pos.entry_time, "exit_time": pd.Timestamp.now(tz="UTC"),
                "reason": "still_open", "pnl_usd": unrealized, "r_multiple": float("nan"),
            })
        if rows:
            pd.DataFrame(rows).to_csv(path, index=False)
            logger.info("Wrote %d trade rows to %s", len(rows), path)


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 60)
    print("PAPER TRADING SESSION SUMMARY")
    print("=" * 60)
    print(f"Starting virtual equity : ${summary['starting_equity']:.4f}")

    print("\n-- CLOSED TRADES (finished: hit TP or SL) " + "-" * 18)
    print(f"  Trades closed         : {summary['closed_trades']}")
    print(f"    winning             : {summary['closed_wins']}")
    print(f"    losing              : {summary['closed_losses']}")
    print(f"  Total profit          : ${summary['gross_profit_usd']:+.4f}")
    print(f"  Total loss            : ${summary['gross_loss_usd']:+.4f}")
    print(f"  Net realized P&L      : ${summary['realized_pnl_usd']:+.4f}")

    print("\n-- STILL OPEN AT SHUTDOWN (marked to last price) " + "-" * 11)
    print(f"  Positions open        : {summary['open_positions']}")
    print(f"    currently winning   : {summary['open_currently_winning']}")
    print(f"    currently losing    : {summary['open_currently_losing']}")
    for symbol, side, entry, price, unrealized in summary["open_rows"]:
        print(f"      {symbol:10s} {side.upper():5s} entry={entry:.4f} last={price:.4f}"
              f" unrealized=${unrealized:+.4f}")
    print(f"  Unrealized profit     : ${summary['open_profit_usd']:+.4f}")
    print(f"  Unrealized loss       : ${summary['open_loss_usd']:+.4f}")
    print(f"  Net unrealized P&L    : ${summary['unrealized_pnl_usd']:+.4f}")

    print("\n-- COMBINED (closed + still open) " + "-" * 26)
    print(f"  Total trades          : {summary['total_trades']}")
    print(f"    winning             : {summary['total_winning']}")
    print(f"    losing              : {summary['total_losing']}")
    print(f"  Total profit          : ${summary['total_profit_usd']:+.4f}")
    print(f"  Total loss            : ${summary['total_loss_usd']:+.4f}")

    print("\n-- EQUITY " + "-" * 50)
    print(f"  Realized only         : ${summary['realized_equity']:.4f}")
    print(f"  Incl. open positions  : ${summary['equity_incl_unrealized']:.4f}")
    net = summary["equity_incl_unrealized"] - summary["starting_equity"]
    pct = 100 * net / summary["starting_equity"] if summary["starting_equity"] else 0.0
    print(f"  Net result            : ${net:+.4f}  ({pct:+.2f}%)")
    print("=" * 60)


def run(starting_equity: float, poll_seconds: int = PRICE_POLL_SECONDS,
        trades_csv: str | None = None, config=None) -> None:
    config = config or CONFIG
    setup_logging(config.log_level)
    exchange = BybitExchange(config)
    broker = PaperBroker(exchange, config, starting_equity)

    logger.warning(
        "Paper trading started: virtual equity=$%.4f symbols=%s "
        "(real Bybit %s market data, NO real orders placed)",
        starting_equity, config.symbols, "TESTNET" if config.testnet else "MAINNET",
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
            for symbol in config.symbols:
                if symbol in broker.open_positions:
                    try:
                        broker.manage_with_candles(symbol)
                    except Exception:
                        logger.exception("Error managing paper position for %s", symbol)

            if now - last_signal_check >= SIGNAL_POLL_SECONDS:
                last_signal_check = now
                for symbol in config.symbols:
                    try:
                        if symbol in broker.open_positions:
                            broker.refresh_trend_and_trail(symbol)
                        elif len(broker.open_positions) < config.max_concurrent_positions:
                            broker.try_open(symbol)
                    except Exception:
                        logger.exception("Error evaluating %s", symbol)

            for _ in range(poll_seconds):
                if not running:
                    break
                time.sleep(1)
    finally:
        print_summary(broker.summary())
        if trades_csv:
            try:
                broker.export_trades_csv(trades_csv)
            except Exception:
                logger.exception("Failed to write trade log to %s", trades_csv)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-trade against real Bybit market data")
    parser.add_argument("--equity", type=float, default=10.0, help="Starting virtual equity in USDT")
    parser.add_argument("--poll-seconds", type=int, default=PRICE_POLL_SECONDS)
    parser.add_argument("--trades-csv", default=None,
                        help="Write every trade (closed + open at shutdown) to this CSV")
    args = parser.parse_args()
    run(starting_equity=args.equity, poll_seconds=args.poll_seconds, trades_csv=args.trades_csv)


if __name__ == "__main__":
    main()
