"""Paper-trading broker for the Sheet16_Logic_Final logic.

Virtual money, real Bybit prices from the public kline and ticker
endpoints. No API key, no account, no order ever submitted.

Executes exactly what the table specifies and nothing more: the direction
its rows traded, for the duration they held, at the leverage they used,
paying the fee they paid. Liquidation is modelled because the table itself
records it (37 of 23,213 rows would liquidate at 100x); no stop-loss is
added, because the table does not have one.
"""
from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass

import pandas as pd

from pl import features as F
from pl import logic as L

logger = logging.getLogger("logic_final.bot")

KLINES_1M = 1000
SIGNAL_POLL_SECONDS = 60


@dataclass
class Position:
    symbol: str
    direction: int
    entry: float
    qty: float
    leverage: int
    margin: float
    opened_at: pd.Timestamp
    hold_minutes: int
    conviction: float
    expected_move_pct: float
    liq_price: float


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
    return_pct_leveraged: float


class Broker:
    def __init__(self, client, logic: L.Logic, symbols: list[str],
                 equity: float, max_positions: int):
        self.client = client
        self.logic = logic
        self.symbols = symbols
        self.equity = equity
        self.starting_equity = equity
        self.max_positions = max_positions
        self.open: dict[str, Position] = {}
        self.closed: list[Closed] = []
        self.scans = 0
        self.no_match = 0

    def klines(self, symbol: str) -> pd.DataFrame | None:
        try:
            rows = self.client.get_kline(category="linear", symbol=symbol,
                                         interval="1", limit=KLINES_1M)["result"]["list"]
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

    def try_open(self, symbol: str) -> None:
        self.scans += 1
        df = self.klines(symbol)
        if df is None or len(df) < 200:
            return
        df = df.iloc[:-1].reset_index(drop=True)     # drop the forming bar
        feats = L.features_30m_on_1m(df, F.build)
        if feats.empty:
            return
        sig = self.logic.signal(int(self.logic.cells_for(feats.tail(1))[-1]))
        if sig is None:
            self.no_match += 1
            return

        price = self.last_price(symbol)
        if price is None or price <= 0:
            return

        lev = sig["leverage"]
        margin = self.equity / max(1, self.max_positions)
        if margin <= 0:
            return
        notional = margin * lev
        qty = notional / price
        # Half the round trip on the way in, half on the way out.
        self.equity -= notional * L.FEE_ROUND_TRIP_PCT / 2

        d = sig["direction"]
        self.open[symbol] = Position(
            symbol=symbol, direction=d, entry=price, qty=qty, leverage=lev,
            margin=margin, opened_at=pd.Timestamp.now(tz="UTC"),
            hold_minutes=sig["hold_minutes"], conviction=sig["conviction"],
            expected_move_pct=sig["expected_move_pct"],
            liq_price=price * (1 - d * L.LIQUIDATION_MOVE),
        )
        logger.info("%s OPEN %s @%.6f qty=%.8f lev=%dx hold=%dmin conv=%.2f "
                    "expected=%.3f%% liq=%.6f",
                    symbol, "LONG" if d > 0 else "SHORT", price, qty, lev,
                    sig["hold_minutes"], sig["conviction"],
                    sig["expected_move_pct"], self.open[symbol].liq_price)

    def manage(self, symbol: str) -> None:
        pos = self.open.get(symbol)
        if pos is None:
            return
        price = self.last_price(symbol)
        if price is None:
            return
        hit_liq = (price <= pos.liq_price) if pos.direction > 0 else (price >= pos.liq_price)
        elapsed = (pd.Timestamp.now(tz="UTC") - pos.opened_at) / pd.Timedelta(minutes=1)
        if hit_liq:
            self._close(symbol, pos.liq_price, "liquidated")
        elif elapsed >= pos.hold_minutes:
            self._close(symbol, price, "hold_elapsed")

    def _close(self, symbol: str, price: float, reason: str) -> None:
        pos = self.open.pop(symbol)
        move = (price - pos.entry) / pos.entry * pos.direction
        if reason == "liquidated":
            pnl = -pos.margin                    # margin gone, nothing returned
        else:
            pnl = move * pos.qty * pos.entry - pos.qty * price * L.FEE_ROUND_TRIP_PCT / 2
        self.equity += pnl
        self.closed.append(Closed(
            symbol=symbol, direction=pos.direction, entry=pos.entry,
            exit_price=price, opened_at=pos.opened_at,
            closed_at=pd.Timestamp.now(tz="UTC"), reason=reason, pnl_usd=pnl,
            return_pct_leveraged=100 * move * pos.leverage))
        logger.info("%s CLOSE @%.6f (%s) pnl=$%.4f equity=$%.4f",
                    symbol, price, reason, pnl, self.equity)

    def summary(self) -> dict:
        wins = [t for t in self.closed if t.pnl_usd > 0]
        losses = [t for t in self.closed if t.pnl_usd <= 0]
        rows, ow, ol, unreal = [], 0, 0, 0.0
        for sym, pos in self.open.items():
            price = self.last_price(sym) or pos.entry
            pnl = (price - pos.entry) / pos.entry * pos.direction * pos.qty * pos.entry
            unreal += pnl
            ow += pnl > 0
            ol += pnl <= 0
            rows.append((sym, pos.direction, pos.entry, price, pnl, pos.leverage))
        gp = sum(t.pnl_usd for t in wins)
        gl = sum(t.pnl_usd for t in losses)
        op = sum(r[4] for r in rows if r[4] > 0)
        olo = sum(r[4] for r in rows if r[4] <= 0)
        return {
            "closed": len(self.closed), "closed_wins": len(wins),
            "closed_losses": len(losses),
            "liquidations": sum(1 for t in self.closed if t.reason == "liquidated"),
            "gross_profit": gp, "gross_loss": gl, "realized": gp + gl,
            "open": len(self.open), "open_wins": ow, "open_losses": ol,
            "open_profit": op, "open_loss": olo, "unrealized": unreal,
            "open_rows": rows,
            "total_trades": len(self.closed) + len(self.open),
            "total_wins": len(wins) + ow, "total_losses": len(losses) + ol,
            "total_profit": gp + op, "total_loss": gl + olo,
            "starting_equity": self.starting_equity, "equity": self.equity,
            "equity_incl_open": self.equity + unreal,
            "scans": self.scans, "no_match": self.no_match,
        }

    def export_csv(self, path: str) -> None:
        rows = [{"symbol": t.symbol, "side": "LONG" if t.direction > 0 else "SHORT",
                 "status": "closed", "entry": t.entry, "exit_or_last": t.exit_price,
                 "opened_at": t.opened_at, "closed_at": t.closed_at,
                 "reason": t.reason, "pnl_usd": t.pnl_usd,
                 "return_pct_leveraged": t.return_pct_leveraged}
                for t in self.closed]
        for sym, d, entry, price, pnl, lev in self.summary()["open_rows"]:
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
    print("\n" + "=" * 66)
    print("PAPER TRADING SESSION SUMMARY")
    print("=" * 66)
    print(f"Starting virtual equity : ${s['starting_equity']:.4f}")

    print("\n-- CLOSED TRADES " + "-" * 46)
    print(f"  Trades closed         : {s['closed']}")
    print(f"    winning             : {s['closed_wins']}")
    print(f"    losing              : {s['closed_losses']}")
    print(f"    liquidated          : {s['liquidations']}")
    print(f"  Total profit          : ${s['gross_profit']:+.4f}")
    print(f"  Total loss            : ${s['gross_loss']:+.4f}")
    print(f"  Net realized P&L      : ${s['realized']:+.4f}")

    print("\n-- STILL OPEN AT SHUTDOWN " + "-" * 37)
    print(f"  Positions open        : {s['open']}")
    print(f"    currently winning   : {s['open_wins']}")
    print(f"    currently losing    : {s['open_losses']}")
    for sym, d, entry, price, pnl, lev in s["open_rows"]:
        print(f"      {sym:12s} {'LONG' if d > 0 else 'SHORT':5s} {lev:3d}x "
              f"entry={entry:.6f} last={price:.6f} unrealized=${pnl:+.4f}")
    print(f"  Unrealized profit     : ${s['open_profit']:+.4f}")
    print(f"  Unrealized loss       : ${s['open_loss']:+.4f}")
    print(f"  Net unrealized P&L    : ${s['unrealized']:+.4f}")

    print("\n-- COMBINED (closed + still open) " + "-" * 29)
    print(f"  Total trades          : {s['total_trades']}")
    print(f"    winning             : {s['total_wins']}")
    print(f"    losing              : {s['total_losses']}")
    print(f"  Total profit          : ${s['total_profit']:+.4f}")
    print(f"  Total loss            : ${s['total_loss']:+.4f}")

    print("\n-- EQUITY " + "-" * 53)
    print(f"  Realized only         : ${s['equity']:.4f}")
    print(f"  Incl. open positions  : ${s['equity_incl_open']:.4f}")
    net = s["equity_incl_open"] - s["starting_equity"]
    pct = 100 * net / s["starting_equity"] if s["starting_equity"] else 0.0
    print(f"  Net result            : ${net:+.4f}  ({pct:+.2f}%)")
    print(f"\n  Scans: {s['scans']:,}   with no matching setup: {s['no_match']:,}")
    print("=" * 66)


def run(client, logic: L.Logic, symbols: list[str], equity: float,
        max_positions: int, poll_seconds: int,
        trades_csv: str | None = None) -> None:
    broker = Broker(client, logic, symbols, equity, max_positions)
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
                    if not running or len(broker.open) >= max_positions:
                        break
                    if sym in broker.open:
                        continue
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
