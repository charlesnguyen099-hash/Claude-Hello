"""Paper-trading broker for FINAL_Logic_PotentialScaledLeverage.

Virtual money, real Bybit prices from the public kline and ticker
endpoints. No API key, no account, no order ever submitted.

Twelve methods vote on each 30-minute bar; a position opens where they
fire and agree on direction. Leverage runs the file's full potential
chain -- base from ATR, a potential score from the volatility percentile,
multiplier = score + 0.5, product capped at the base. The exit is fixed
at entry.
"""
from __future__ import annotations

import concurrent.futures
import logging
import signal
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fp import features as F
from fp import logic as L
from fp import methods as M

logger = logging.getLogger("potential_leverage.bot")

# 30m bars are fetched directly: Bybit caps a kline call at 1000 candles,
# and 1000 1m bars is only 33 30m bars, far short of the ~200 the slowest
# feature (EMA200) needs.
KLINES_30M = 1000
SIGNAL_POLL_SECONDS = 60
# Scanning the whole board means one kline call per symbol. Done serially
# that is ~200ms x 500 symbols = well over a minute per cycle, longer than
# the cycle itself. Bybit's public endpoints allow far more than this in
# parallel, so the fetches are threaded and only the maths stays serial.
FETCH_WORKERS = 12


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
    trail_dist: float
    potential_score: float
    lev_base: float


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
                 max_positions: int, exit_name: str, min_votes: int,
                 fee: float = L.FEE_ROUND_TRIP, max_leverage: float | None = None,
                 margin_pct: float = 0.10):
        self.client = client
        self.symbols = symbols
        self.equity = equity
        self.starting_equity = equity
        # 0 means "no cap on the number of positions" -- what actually
        # limits it then is free margin, which is the real constraint.
        self.max_positions = max_positions
        self.margin_pct = margin_pct
        self.exit_name = exit_name
        self.min_votes = min_votes
        self.fee = fee
        self.max_leverage = max_leverage
        self.open: dict[str, Position] = {}
        self.closed: list[Closed] = []
        self.scans = 0
        self.no_signal = 0
        self.skipped_no_margin = 0

    @property
    def committed_margin(self) -> float:
        return sum(p.margin for p in self.open.values())

    @property
    def free_margin(self) -> float:
        return self.equity - self.committed_margin

    def has_room(self) -> bool:
        if self.max_positions and len(self.open) >= self.max_positions:
            return False
        return self.free_margin >= self.equity * self.margin_pct

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

    def prefetch(self, symbols: list[str]) -> dict:
        """Fetch klines for many symbols at once so a full-board scan is
        bounded by the slowest request rather than the sum of all of them."""
        out: dict = {}
        with concurrent.futures.ThreadPoolExecutor(FETCH_WORKERS) as pool:
            futures = {pool.submit(self.klines, s): s for s in symbols}
            for fut in concurrent.futures.as_completed(futures):
                sym = futures[fut]
                try:
                    out[sym] = fut.result()
                except Exception:
                    logger.exception("prefetch failed for %s", sym)
                    out[sym] = None
        return out

    def try_open(self, symbol: str, bars=None) -> None:
        self.scans += 1
        if bars is None:
            bars = self.klines(symbol)
        if bars is None or len(bars) < 250:
            return
        bars = bars.iloc[:-1].reset_index(drop=True)   # drop the forming bar

        feats = F.build(bars)
        v = M.evaluate_all(feats).iloc[-1]
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
        chain = L.leverage_potential(atr_pct, self.max_leverage)
        lev = chain["leverage"]

        # A fixed slice of current equity per trade, and never more than is
        # actually free -- with no position cap, free margin is what stops
        # the bot opening more than the account can carry.
        margin = min(self.equity * self.margin_pct, self.free_margin)
        if margin <= 0:
            self.skipped_no_margin += 1
            return
        notional = margin * lev
        qty = notional / price
        self.equity -= notional * self.fee / 2

        tp_mult = L.TP_MULTIPLES.get(self.exit_name)
        fired = ", ".join(m.split("_", 1)[1] for m in M.METHOD_NAMES if v[m] != "-")

        self.open[symbol] = Position(
            symbol=symbol, direction=d, entry=price, qty=qty, leverage=lev,
            margin=margin, opened_at=pd.Timestamp.now(tz="UTC"),
            tp_price=(price + d * tp_mult * atr) if tp_mult else float("nan"),
            sl_price=price - d * L.SL_MULTIPLE * atr,
            liq_price=price * (1 - d * 0.9 / lev),
            exit_name=self.exit_name, methods=fired,
            votes=int(v["n_methods_fired"]), best_price=price,
            trail_dist=L.TRAIL_MULTIPLE * atr,
            potential_score=chain["potential_score"],
            lev_base=chain["lev_base"],
        )
        logger.info("%s OPEN %s @%.6f potential=%.2f base=%.0fx -> lev=%.0fx "
                    "votes=%d [%s] exit=%s tp=%.6f sl=%.6f",
                    symbol, "LONG" if d > 0 else "SHORT", price,
                    chain["potential_score"], chain["lev_base"], lev,
                    v["n_methods_fired"], fired, self.exit_name,
                    self.open[symbol].tp_price, self.open[symbol].sl_price)

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
            trail = pos.best_price - d * pos.trail_dist
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
            pnl = move * pos.qty * pos.entry - pos.qty * price * self.fee / 2
        self.equity += pnl
        self.closed.append(Closed(
            symbol=symbol, direction=pos.direction, entry=pos.entry,
            exit_price=price, opened_at=pos.opened_at,
            closed_at=pd.Timestamp.now(tz="UTC"), reason=reason, pnl_usd=pnl,
            return_pct_leveraged=100 * move * pos.leverage, methods=pos.methods))
        logger.info("%s CLOSE %s @%.6f (%s) pnl=$%.4f equity=$%.4f",
                    symbol, "LONG" if pos.direction > 0 else "SHORT",
                    price, reason, pnl, self.equity)

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
                         pos.leverage, pos.methods, pos.potential_score))
        gp, gl = sum(t.pnl_usd for t in wins), sum(t.pnl_usd for t in losses)
        op = sum(r[4] for r in rows if r[4] > 0)
        olo = sum(r[4] for r in rows if r[4] <= 0)
        return {
            "closed": len(self.closed), "closed_wins": len(wins),
            "closed_losses": len(losses),
            "liquidations": sum(1 for t in self.closed if t.reason == "liquidated"),
            "stops": sum(1 for t in self.closed if t.reason == "stop_loss"),
            "targets": sum(1 for t in self.closed if t.reason == "take_profit"),
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
            "skipped_no_margin": self.skipped_no_margin,
            "committed_margin": self.committed_margin,
            "free_margin": self.free_margin,
        }

    def export_csv(self, path: str) -> None:
        rows = [{"symbol": t.symbol, "side": "LONG" if t.direction > 0 else "SHORT",
                 "status": "closed", "entry": t.entry, "exit_or_last": t.exit_price,
                 "opened_at": t.opened_at, "closed_at": t.closed_at,
                 "reason": t.reason, "pnl_usd": t.pnl_usd,
                 "return_pct_leveraged": t.return_pct_leveraged,
                 "methods": t.methods} for t in self.closed]
        for sym, d, entry, price, pnl, lev, meth, pot in self.summary()["open_rows"]:
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
    print(f"    hit take-profit     : {s['targets']}")
    print(f"    stopped out         : {s['stops']}")
    print(f"    liquidated          : {s['liquidations']}")
    print(f"  Total profit          : ${s['gross_profit']:+.4f}")
    print(f"  Total loss            : ${s['gross_loss']:+.4f}")
    print(f"  Net realized P&L      : ${s['realized']:+.4f}")

    print("\n-- STILL OPEN AT SHUTDOWN " + "-" * 41)
    print(f"  Positions open        : {s['open']}")
    print(f"    currently winning   : {s['open_wins']}")
    print(f"    currently losing    : {s['open_losses']}")
    for sym, d, entry, price, pnl, lev, meth, pot in s["open_rows"]:
        print(f"      {sym:12s} {'LONG' if d > 0 else 'SHORT':5s} {lev:5.0f}x "
              f"potential={pot:.2f} entry={entry:.6f} last={price:.6f} "
              f"unreal=${pnl:+.4f}")
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
    print(f"\n  Scans: {s['scans']:,}   no qualifying vote: {s['no_signal']:,}"
          f"   skipped for margin: {s['skipped_no_margin']:,}")
    print("=" * 70)


def run(client, symbols: list[str], equity: float, max_positions: int,
        exit_name: str, min_votes: int, poll_seconds: int,
        trades_csv: str | None = None, fee: float = L.FEE_ROUND_TRIP,
        max_leverage: float | None = None, margin_pct: float = 0.10) -> None:
    broker = Broker(client, symbols, equity, max_positions, exit_name,
                    min_votes, fee, max_leverage, margin_pct)
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
                # Every symbol without a position is a candidate. The scan
                # runs the whole board rather than stopping at the first
                # few -- a coin further down the list is as tradeable as
                # one near the top.
                candidates = [s for s in symbols if s not in broker.open]
                if candidates and broker.has_room():
                    t0 = time.time()
                    fetched = broker.prefetch(candidates)
                    opened_before = len(broker.open)
                    for sym in candidates:
                        if not running or not broker.has_room():
                            break
                        try:
                            broker.try_open(sym, fetched.get(sym))
                        except Exception:
                            logger.exception("entry check failed for %s", sym)
                    logger.info("scanned %d symbols in %.1fs, opened %d, "
                                "positions %d, free margin $%.2f",
                                len(candidates), time.time() - t0,
                                len(broker.open) - opened_before,
                                len(broker.open), broker.free_margin)

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
