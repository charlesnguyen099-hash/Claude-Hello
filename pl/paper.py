"""Paper trading on live Bybit data with the PureLogic rule.

Reads Bybit's public kline endpoint — no API key, no account, no order is
ever submitted. The balance is a number in memory. Scans every configured
symbol, opens a position only where the market matches a cell the table
had a verdict on, holds for that cell's median duration, and prints a full
session summary on Ctrl+C.
"""
from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass, field

import pandas as pd

from pl import features as F
from pl import strategy as S

logger = logging.getLogger("purelogic.paper")

KLINES_1M = 1000          # enough 1m bars to build 30m features with history
POSITION_POLL_SECONDS = 15
SIGNAL_POLL_SECONDS = 60


@dataclass
class Position:
    symbol: str
    direction: int             # +1 long, -1 short
    entry: float
    qty: float
    leverage: int
    opened_at: pd.Timestamp
    hold_minutes: int
    conviction: float
    expected_move_pct: float
    liq_price: float
    peak_adverse_pct: float = 0.0


@dataclass
class Closed:
    symbol: str
    direction: int
    entry: float
    exit_price: float
    opened_at: pd.Timestamp
    closed_at: pd.Timestamp
    reason: str
    pnl_usd: float
    return_pct: float


class Broker:
    """Virtual account. Every fill is simulated; nothing reaches an exchange."""

    def __init__(self, client, logic: S.PureLogic, symbols: list[str],
                 equity: float, leverage: int, max_positions: int):
        self.client = client
        self.logic = logic
        self.symbols = symbols
        self.equity = equity
        self.starting_equity = equity
        self.leverage = leverage
        self.max_positions = max_positions
        self.open: dict[str, Position] = {}
        self.closed: list[Closed] = []
        self.skipped_no_match = 0

    # -- market data --------------------------------------------------
    def klines(self, symbol: str) -> pd.DataFrame | None:
        try:
            resp = self.client.get_kline(category="linear", symbol=symbol,
                                         interval="1", limit=KLINES_1M)
            rows = resp["result"]["list"]
        except Exception:
            logger.exception("kline fetch failed for %s", symbol)
            return None
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low",
                                         "close", "volume", "turnover"])
        df["datetime"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms")
        for c in ("open", "high", "low", "close", "volume"):
            df[c] = df[c].astype(float)
        return df.sort_values("datetime").reset_index(drop=True)

    def last_price(self, symbol: str) -> float | None:
        try:
            r = self.client.get_tickers(category="linear", symbol=symbol)
            return float(r["result"]["list"][0]["lastPrice"])
        except Exception:
            logger.exception("ticker fetch failed for %s", symbol)
            return None

    # -- entries ------------------------------------------------------
    def try_open(self, symbol: str) -> None:
        df = self.klines(symbol)
        if df is None or len(df) < 200:
            return
        df = df.iloc[:-1].reset_index(drop=True)      # drop the forming bar
        feats = S.features_30m_on_1m(df, F.build)
        if feats.empty:
            return
        cell = self.logic.cells_for(feats.tail(1))
        sig = self.logic.signal(int(cell[-1]))
        if sig is None:
            self.skipped_no_match += 1
            return

        price = self.last_price(symbol)
        if price is None or price <= 0:
            return

        margin = self.equity / max(1, self.max_positions)
        notional = margin * self.leverage
        qty = notional / price
        fee = notional * S.TAKER_FEE_PCT
        self.equity -= fee

        d = sig["direction"]
        liq = price * (1 - d * 0.9 / self.leverage)
        self.open[symbol] = Position(
            symbol=symbol, direction=d, entry=price, qty=qty,
            leverage=self.leverage, opened_at=pd.Timestamp.now(tz="UTC"),
            hold_minutes=sig["hold_minutes"], conviction=sig["conviction"],
            expected_move_pct=sig["expected_move_pct"], liq_price=liq,
        )
        logger.info(
            "%s OPEN %s @%.6f qty=%.6f lev=%dx hold=%dmin conv=%.2f "
            "expected=%.3f%% liq=%.6f",
            symbol, "LONG" if d > 0 else "SHORT", price, qty, self.leverage,
            sig["hold_minutes"], sig["conviction"], sig["expected_move_pct"], liq)

    # -- exits --------------------------------------------------------
    def manage(self, symbol: str) -> None:
        pos = self.open.get(symbol)
        if pos is None:
            return
        price = self.last_price(symbol)
        if price is None:
            return

        move = (price - pos.entry) / pos.entry * pos.direction
        pos.peak_adverse_pct = max(pos.peak_adverse_pct, -move * 100)

        liquidated = (price <= pos.liq_price) if pos.direction > 0 else (price >= pos.liq_price)
        elapsed = (pd.Timestamp.now(tz="UTC") - pos.opened_at) / pd.Timedelta(minutes=1)

        if liquidated:
            self._close(symbol, pos.liq_price, "liquidated")
        elif elapsed >= pos.hold_minutes:
            self._close(symbol, price, "hold_elapsed")

    def _close(self, symbol: str, price: float, reason: str) -> None:
        pos = self.open.pop(symbol)
        move = (price - pos.entry) / pos.entry * pos.direction
        gross = move * pos.qty * pos.entry
        fee = pos.qty * price * S.TAKER_FEE_PCT
        pnl = gross - fee
        self.equity += pnl
        self.closed.append(Closed(
            symbol=symbol, direction=pos.direction, entry=pos.entry,
            exit_price=price, opened_at=pos.opened_at,
            closed_at=pd.Timestamp.now(tz="UTC"), reason=reason,
            pnl_usd=pnl, return_pct=100 * move * pos.leverage,
        ))
        logger.info("%s CLOSE @%.6f (%s) pnl=$%.4f equity=$%.4f",
                    symbol, price, reason, pnl, self.equity)

    # -- reporting ----------------------------------------------------
    def summary(self) -> dict:
        wins = [t for t in self.closed if t.pnl_usd > 0]
        losses = [t for t in self.closed if t.pnl_usd <= 0]
        gross_profit = sum(t.pnl_usd for t in wins)
        gross_loss = sum(t.pnl_usd for t in losses)

        rows, open_win, open_lose, unreal = [], 0, 0, 0.0
        for sym, pos in self.open.items():
            price = self.last_price(sym) or pos.entry
            move = (price - pos.entry) / pos.entry * pos.direction
            pnl = move * pos.qty * pos.entry
            unreal += pnl
            if pnl > 0:
                open_win += 1
            else:
                open_lose += 1
            rows.append((sym, pos.direction, pos.entry, price, pnl))

        open_profit = sum(p for *_, p in rows if p > 0)
        open_loss = sum(p for *_, p in rows if p <= 0)
        return {
            "closed": len(self.closed), "closed_wins": len(wins),
            "closed_losses": len(losses),
            "gross_profit": gross_profit, "gross_loss": gross_loss,
            "realized": gross_profit + gross_loss,
            "liquidations": sum(1 for t in self.closed if t.reason == "liquidated"),
            "open": len(self.open), "open_wins": open_win, "open_losses": open_lose,
            "open_profit": open_profit, "open_loss": open_loss,
            "unrealized": unreal, "open_rows": rows,
            "total_trades": len(self.closed) + len(self.open),
            "total_wins": len(wins) + open_win,
            "total_losses": len(losses) + open_lose,
            "total_profit": gross_profit + open_profit,
            "total_loss": gross_loss + open_loss,
            "starting_equity": self.starting_equity,
            "equity": self.equity,
            "equity_incl_open": self.equity + unreal,
            "skipped_no_match": self.skipped_no_match,
        }

    def export_csv(self, path: str) -> None:
        rows = [{"symbol": t.symbol, "side": "LONG" if t.direction > 0 else "SHORT",
                 "status": "closed", "entry": t.entry, "exit_or_last": t.exit_price,
                 "opened_at": t.opened_at, "closed_at": t.closed_at,
                 "reason": t.reason, "pnl_usd": t.pnl_usd,
                 "return_pct_leveraged": t.return_pct} for t in self.closed]
        for sym, d, entry, price, pnl in self.summary()["open_rows"]:
            pos = self.open[sym]
            rows.append({"symbol": sym, "side": "LONG" if d > 0 else "SHORT",
                         "status": "open_at_shutdown", "entry": entry,
                         "exit_or_last": price, "opened_at": pos.opened_at,
                         "closed_at": pd.Timestamp.now(tz="UTC"),
                         "reason": "still_open", "pnl_usd": pnl,
                         "return_pct_leveraged": float("nan")})
        if rows:
            pd.DataFrame(rows).to_csv(path, index=False)
            logger.info("wrote %d trade rows to %s", len(rows), path)


def print_summary(s: dict) -> None:
    print("\n" + "=" * 64)
    print("PAPER TRADING SESSION SUMMARY")
    print("=" * 64)
    print(f"Starting virtual equity : ${s['starting_equity']:.4f}")

    print("\n-- CLOSED TRADES " + "-" * 44)
    print(f"  Trades closed         : {s['closed']}")
    print(f"    winning             : {s['closed_wins']}")
    print(f"    losing              : {s['closed_losses']}")
    print(f"    liquidated          : {s['liquidations']}")
    print(f"  Total profit          : ${s['gross_profit']:+.4f}")
    print(f"  Total loss            : ${s['gross_loss']:+.4f}")
    print(f"  Net realized P&L      : ${s['realized']:+.4f}")

    print("\n-- STILL OPEN AT SHUTDOWN " + "-" * 35)
    print(f"  Positions open        : {s['open']}")
    print(f"    currently winning   : {s['open_wins']}")
    print(f"    currently losing    : {s['open_losses']}")
    for sym, d, entry, price, pnl in s["open_rows"]:
        print(f"      {sym:12s} {'LONG' if d > 0 else 'SHORT':5s} "
              f"entry={entry:.6f} last={price:.6f} unrealized=${pnl:+.4f}")
    print(f"  Unrealized profit     : ${s['open_profit']:+.4f}")
    print(f"  Unrealized loss       : ${s['open_loss']:+.4f}")
    print(f"  Net unrealized P&L    : ${s['unrealized']:+.4f}")

    print("\n-- COMBINED (closed + still open) " + "-" * 27)
    print(f"  Total trades          : {s['total_trades']}")
    print(f"    winning             : {s['total_wins']}")
    print(f"    losing              : {s['total_losses']}")
    print(f"  Total profit          : ${s['total_profit']:+.4f}")
    print(f"  Total loss            : ${s['total_loss']:+.4f}")

    print("\n-- EQUITY " + "-" * 51)
    print(f"  Realized only         : ${s['equity']:.4f}")
    print(f"  Incl. open positions  : ${s['equity_incl_open']:.4f}")
    net = s["equity_incl_open"] - s["starting_equity"]
    pct = 100 * net / s["starting_equity"] if s["starting_equity"] else 0.0
    print(f"  Net result            : ${net:+.4f}  ({pct:+.2f}%)")
    print(f"\n  Scans with no matching setup: {s['skipped_no_match']:,}")
    print("=" * 64)


def run(client, logic: S.PureLogic, symbols: list[str], equity: float,
        leverage: int, max_positions: int, poll_seconds: int,
        trades_csv: str | None = None) -> None:
    broker = Broker(client, logic, symbols, equity, leverage, max_positions)
    running = True

    def stop(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    last_scan = 0.0
    try:
        while running:
            for sym in list(broker.open):
                try:
                    broker.manage(sym)
                except Exception:
                    logger.exception("manage failed for %s", sym)

            now = time.time()
            if now - last_scan >= SIGNAL_POLL_SECONDS:
                last_scan = now
                for sym in symbols:
                    if not running:
                        break
                    if sym in broker.open:
                        continue
                    if len(broker.open) >= max_positions:
                        break
                    try:
                        broker.try_open(sym)
                    except Exception:
                        logger.exception("entry check failed for %s", sym)

            for _ in range(max(1, poll_seconds)):
                if not running:
                    break
                time.sleep(1)
    finally:
        print_summary(broker.summary())
        if trades_csv:
            try:
                broker.export_csv(trades_csv)
            except Exception:
                logger.exception("failed writing %s", trades_csv)
