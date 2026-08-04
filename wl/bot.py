"""Paper-trading broker for the twelve-method voting logic.

Virtual money, real Bybit prices from the public kline and ticker
endpoints. No API key, no account, no order ever submitted.

Runs the full pipeline live: 30-minute bars, 106 features,
twelve methods voting, consensus direction, flexible leverage, and one of
the file's five exits (chosen up front and held, since which one wins is
only knowable afterwards).
"""
from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from wl import exits as X
from wl import features as F
from wl import methods as M

logger = logging.getLogger("wide_logic.bot")

# 30m bars are fetched directly rather than resampled from 1m: Bybit caps a
# kline call at 1000 candles, and 1000 1m bars is only 33 30m bars -- far
# short of the ~200 the slowest feature (EMA200) needs to converge.
KLINES_30M = 1000
SIGNAL_POLL_SECONDS = 60


@dataclass
class Position:
    symbol: str
    direction: int
    entry: float
    qty: float
    leverage: float
    margin: float
    opened_at: pd.Timestamp
    tp_price: float
    sl_price: float
    liq_price: float
    exit_name: str
    methods: str
    votes: int
    best_price: float
    trail_atr: float


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
    methods: str


class Broker:
    def __init__(self, client, symbols: list[str], equity: float,
                 max_positions: int, exit_name: str, min_votes: int):
        self.client = client
        self.symbols = symbols
        self.equity = equity
        self.starting_equity = equity
        self.max_positions = max_positions
        self.exit_name = exit_name
        self.min_votes = min_votes
        self.open: dict[str, Position] = {}
        self.closed: list[Closed] = []
        self.scans = 0
        self.no_signal = 0

    def klines(self, symbol: str) -> pd.DataFrame | None:
        try:
            rows = self.client.get_kline(category="linear", symbol=symbol,
                                         interval="30",
                                         limit=KLINES_30M)["result"]["list"]
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
        bars = self.klines(symbol)
        if bars is None or len(bars) < 250:
            return
        bars = bars.iloc[:-1].reset_index(drop=True)   # drop the forming bar

        feats = F.build(bars)
        votes = M.evaluate_all(feats)
        v = votes.iloc[-1]

        if v["n_methods_fired"] < self.min_votes or v["consensus_dir"] == "TIE":
            self.no_signal += 1
            return

        atr_pct = float(feats["atr14_pct"].iloc[-1])
        if not np.isfinite(atr_pct) or atr_pct <= 0:
            return

        price = self.last_price(symbol)
        if price is None or price <= 0:
            return

        d = 1 if v["consensus_dir"] == M.LONG else -1
        atr = atr_pct / 100.0 * price
        lev = X.leverage_flexible(atr_pct * X.SL_MULTIPLE)
        margin = self.equity / max(1, self.max_positions)
        if margin <= 0:
            return
        notional = margin * lev
        qty = notional / price
        self.equity -= notional * X.FEE_ROUND_TRIP / 2

        tp_mult = X.TP_MULTIPLES.get(self.exit_name)
        fired = ", ".join(m.split("_", 1)[1] for m in M.METHOD_NAMES if v[m] != "-")

        self.open[symbol] = Position(
            symbol=symbol, direction=d, entry=price, qty=qty, leverage=lev,
            margin=margin, opened_at=pd.Timestamp.now(tz="UTC"),
            tp_price=(price + d * tp_mult * atr) if tp_mult else float("nan"),
            sl_price=price - d * X.SL_MULTIPLE * atr,
            liq_price=price * (1 - d * 0.9 / lev),
            exit_name=self.exit_name, methods=fired,
            votes=int(v["n_methods_fired"]), best_price=price,
            trail_atr=X.TRAIL_MULTIPLE * atr,
        )
        logger.info("%s OPEN %s @%.6f lev=%.0fx votes=%d [%s] exit=%s",
                    symbol, "LONG" if d > 0 else "SHORT", price, lev,
                    v["n_methods_fired"], fired, self.exit_name)

    def manage(self, symbol: str) -> None:
        pos = self.open.get(symbol)
        if pos is None:
            return
        price = self.last_price(symbol)
        if price is None:
            return
        d = pos.direction

        if (price <= pos.liq_price) if d > 0 else (price >= pos.liq_price):
            self._close(symbol, pos.liq_price, "liquidated")
            return
        if (price <= pos.sl_price) if d > 0 else (price >= pos.sl_price):
            self._close(symbol, pos.sl_price, "stop_loss")
            return

        if pos.exit_name == "net_TRAILING":
            pos.best_price = max(pos.best_price, price) if d > 0 else min(pos.best_price, price)
            trail = pos.best_price - d * pos.trail_atr
            if (price <= trail) if d > 0 else (price >= trail):
                self._close(symbol, trail, "trailing")
        elif np.isfinite(pos.tp_price):
            if (price >= pos.tp_price) if d > 0 else (price <= pos.tp_price):
                self._close(symbol, pos.tp_price, "take_profit")

    def _close(self, symbol: str, price: float, reason: str) -> None:
        pos = self.open.pop(symbol)
        move = (price - pos.entry) / pos.entry * pos.direction
        if reason == "liquidated":
            pnl = -pos.margin
        else:
            pnl = move * pos.qty * pos.entry - pos.qty * price * X.FEE_ROUND_TRIP / 2
        self.equity += pnl
        self.closed.append(Closed(
            symbol=symbol, direction=pos.direction, entry=pos.entry,
            exit_price=price, opened_at=pos.opened_at,
            closed_at=pd.Timestamp.now(tz="UTC"), reason=reason, pnl_usd=pnl,
            return_pct_leveraged=100 * move * pos.leverage, methods=pos.methods))
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
            rows.append((sym, pos.direction, pos.entry, price, pnl,
                         pos.leverage, pos.methods))
        gp, gl = sum(t.pnl_usd for t in wins), sum(t.pnl_usd for t in losses)
        op = sum(r[4] for r in rows if r[4] > 0)
        olo = sum(r[4] for r in rows if r[4] <= 0)
        return {
            "closed": len(self.closed), "closed_wins": len(wins),
            "closed_losses": len(losses),
            "liquidations": sum(1 for t in self.closed if t.reason == "liquidated"),
            "stops": sum(1 for t in self.closed if t.reason == "stop_loss"),
            "gross_profit": gp, "gross_loss": gl, "realized": gp + gl,
            "open": len(self.open), "open_wins": ow, "open_losses": ol,
            "open_profit": op, "open_loss": olo, "unrealized": unreal,
            "open_rows": rows,
            "total_trades": len(self.closed) + len(self.open),
            "total_wins": len(wins) + ow, "total_losses": len(losses) + ol,
            "total_profit": gp + op, "total_loss": gl + olo,
            "starting_equity": self.starting_equity, "equity": self.equity,
            "equity_incl_open": self.equity + unreal,
            "scans": self.scans, "no_signal": self.no_signal,
        }

    def export_csv(self, path: str) -> None:
        rows = [{"symbol": t.symbol, "side": "LONG" if t.direction > 0 else "SHORT",
                 "status": "closed", "entry": t.entry, "exit_or_last": t.exit_price,
                 "opened_at": t.opened_at, "closed_at": t.closed_at,
                 "reason": t.reason, "pnl_usd": t.pnl_usd,
                 "return_pct_leveraged": t.return_pct_leveraged,
                 "methods": t.methods} for t in self.closed]
        for sym, d, entry, price, pnl, lev, meth in self.summary()["open_rows"]:
            pos = self.open[sym]
            rows.append({"symbol": sym, "side": "LONG" if d > 0 else "SHORT",
                         "status": "open_at_shutdown", "entry": entry,
                         "exit_or_last": price, "opened_at": pos.opened_at,
                         "closed_at": pd.Timestamp.now(tz="UTC"),
                         "reason": "still_open", "pnl_usd": pnl,
                         "return_pct_leveraged": float("nan"), "methods": meth})
        if rows:
            pd.DataFrame(rows).to_csv(path, index=False)
            logger.info("wrote %d trade rows to %s", len(rows), path)


def print_summary(s: dict) -> None:
    print("\n" + "=" * 70)
    print("PAPER TRADING SESSION SUMMARY")
    print("=" * 70)
    print(f"Starting virtual equity : ${s['starting_equity']:.4f}")

    print("\n-- CLOSED TRADES " + "-" * 50)
    print(f"  Trades closed         : {s['closed']}")
    print(f"    winning             : {s['closed_wins']}")
    print(f"    losing              : {s['closed_losses']}")
    print(f"    stopped out         : {s['stops']}")
    print(f"    liquidated          : {s['liquidations']}")
    print(f"  Total profit          : ${s['gross_profit']:+.4f}")
    print(f"  Total loss            : ${s['gross_loss']:+.4f}")
    print(f"  Net realized P&L      : ${s['realized']:+.4f}")

    print("\n-- STILL OPEN AT SHUTDOWN " + "-" * 41)
    print(f"  Positions open        : {s['open']}")
    print(f"    currently winning   : {s['open_wins']}")
    print(f"    currently losing    : {s['open_losses']}")
    for sym, d, entry, price, pnl, lev, meth in s["open_rows"]:
        print(f"      {sym:12s} {'LONG' if d > 0 else 'SHORT':5s} {lev:5.0f}x "
              f"entry={entry:.6f} last={price:.6f} unreal=${pnl:+.4f}")
        print(f"        methods: {meth}")
    print(f"  Unrealized profit     : ${s['open_profit']:+.4f}")
    print(f"  Unrealized loss       : ${s['open_loss']:+.4f}")
    print(f"  Net unrealized P&L    : ${s['unrealized']:+.4f}")

    print("\n-- COMBINED (closed + still open) " + "-" * 33)
    print(f"  Total trades          : {s['total_trades']}")
    print(f"    winning             : {s['total_wins']}")
    print(f"    losing              : {s['total_losses']}")
    print(f"  Total profit          : ${s['total_profit']:+.4f}")
    print(f"  Total loss            : ${s['total_loss']:+.4f}")

    print("\n-- EQUITY " + "-" * 57)
    print(f"  Realized only         : ${s['equity']:.4f}")
    print(f"  Incl. open positions  : ${s['equity_incl_open']:.4f}")
    net = s["equity_incl_open"] - s["starting_equity"]
    pct = 100 * net / s["starting_equity"] if s["starting_equity"] else 0.0
    print(f"  Net result            : ${net:+.4f}  ({pct:+.2f}%)")
    print(f"\n  Scans: {s['scans']:,}   no qualifying vote: {s['no_signal']:,}")
    print("=" * 70)


def run(client, symbols: list[str], equity: float, max_positions: int,
        exit_name: str, min_votes: int, poll_seconds: int,
        trades_csv: str | None = None) -> None:
    broker = Broker(client, symbols, equity, max_positions, exit_name, min_votes)
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
