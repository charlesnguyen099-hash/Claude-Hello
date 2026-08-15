#!/usr/bin/env python3
"""Paper-trade the fp/full.py logic on live Bybit prices. Virtual $10.

    pip install pandas numpy scikit-learn pybit
    python -m fp.full          # build the logic (once, slow)
    python run_full_bot.py     # trade it (no key, no orders, ever)

No API key, no account, no order is ever placed. It reads Bybit's public
kline and ticker endpoints and keeps the balance in memory.

WHAT IT TRADES. One logic, built by fp/full.py and nothing else:

  find    every bar in each coin's history where a trade clears +1%
          after that coin's real round trip -- taker in, taker out,
          funding, and a spread estimated from the coin's own bars.
  exit    where the data showed: the best price reached before the move
          that would have stopped the trade out. Unbounded above -- +1%
          is the floor that makes a bar count, not the target.
  learn   three models per coin over the same 200 columns: which side,
          how far it runs, how much it hurts on the way.
  size    one number, POTENTIAL, scales the stake from 5% to all-in AND
          the leverage from 1x to that coin's Bybit ceiling.

NOTHING IS PINNED TO A NUMBER. The exit is a fraction of whatever price
the entry happened at; the hold runs as long as the move takes; the stop
comes from the predicted adverse excursion; the leverage ceiling is read
from Bybit per coin. A coin trading at $0.002 and one at $100,000 run
the same code with no constant changed.

    python run_full_bot.py
    python run_full_bot.py --symbols BTCUSDT,ETHUSDT
    python run_full_bot.py --equity 100
    python run_full_bot.py --once        # one scan, print, exit

Ctrl+C prints the session summary, open positions included.
"""
from __future__ import annotations

# OpenBLAS reserves a per-thread scratch buffer for as many threads as it
# believes the machine has, at import time, before the bot knows whether
# it needs any. It does not: the models score one row at a time. Two live
# runs died on "OpenBLAS error: Memory allocation still failed after 10
# retries" before scanning a single bar. This must run before numpy is
# imported by anything, so it sits directly under the __future__ import.
import os as _os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse
import signal
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fp import costs as C
from fp.data import SYMBOLS
from fp.live_full import Decision, FullLogic

BARS = 1000          # Bybit's cap for one kline call
SCAN_SECONDS = 20    # how often to re-price open positions
TAKER = C.TAKER_PER_SIDE


# --------------------------------------------------------------- market data
def make_client():
    from pybit.unified_trading import HTTP
    return HTTP(testnet=False)


def fetch_klines(client, symbol: str, limit: int = BARS):
    """The last `limit` closed 1-minute bars, oldest first."""
    r = client.get_kline(category="linear", symbol=symbol,
                         interval="1", limit=limit)
    rows = r.get("result", {}).get("list", [])
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close",
                                     "volume", "turnover"])
    df = df.astype(float)
    df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms")
    df = df.sort_values("ts").set_index("ts")
    # The most recent candle is still forming; a decision made on a
    # half-built bar is a decision made on a different bar than the one
    # the logic was fitted on.
    return df.iloc[:-1]


def fetch_tickers(client, symbols):
    """Last price for every symbol in one call."""
    out = {}
    try:
        r = client.get_tickers(category="linear")
        for row in r.get("result", {}).get("list", []):
            s = row.get("symbol")
            if s in symbols:
                try:
                    out[s] = float(row["lastPrice"])
                except (KeyError, TypeError, ValueError):
                    pass
    except Exception:
        pass
    return out


# ------------------------------------------------------------------ account
@dataclass
class Position:
    symbol: str
    dec: Decision
    entry: float
    margin: float
    notional: float
    opened_at: float
    bars_held: int = 0
    peak: float = 0.0
    high: float = 0.0
    low: float = 0.0

    def move(self, price: float) -> float:
        return (price / self.entry - 1.0) * self.dec.side


@dataclass
class Closed:
    symbol: str
    side: int
    reason: str
    move: float
    pnl: float
    fees: float
    held_s: float
    leverage: float
    stake: float
    potential: float


@dataclass
class Account:
    equity: float
    start: float
    open: dict = field(default_factory=dict)
    closed: list = field(default_factory=list)
    fees_paid: float = 0.0
    rejected: int = 0

    def can_open(self, symbol: str) -> bool:
        # ONE POSITION PER COIN AT A TIME. Coverage of every signal is a
        # build-time property; at trade time a coin holds one position
        # and a second signal on it simply waits.
        return symbol not in self.open

    def open_position(self, dec: Decision, price: float) -> Position | None:
        margin = dec.stake * self.equity
        if margin <= 0 or margin > self.equity:
            return None
        p = Position(symbol=dec.symbol, dec=dec, entry=price, margin=margin,
                     notional=margin * dec.leverage, opened_at=time.time(),
                     high=price, low=price)
        self.open[dec.symbol] = p
        return p

    def close_position(self, p: Position, reason: str, move: float) -> Closed:
        # ONE cost, charged once, at close. `dec.cost` is the coin's full
        # round trip -- taker in, taker out, the spread estimated from
        # its own bars, and funding. Charging a taker fee at open and
        # another at close ON TOP of it bills the same two fills three
        # times and would make the bot look worse than the logic it is
        # running. This is exactly the arithmetic fp/full.py replayed.
        fees = p.dec.cost * p.notional
        pnl = (move - p.dec.cost) * p.notional
        # A position cannot lose more than its margin: past that the
        # exchange liquidates and the loss stops at what was posted.
        pnl = max(pnl, -p.margin)
        self.equity += pnl
        self.fees_paid += fees
        c = Closed(symbol=p.symbol, side=p.dec.side, reason=reason,
                   move=move, pnl=pnl, fees=fees,
                   held_s=time.time() - p.opened_at,
                   leverage=p.dec.leverage, stake=p.dec.stake,
                   potential=p.dec.potential)
        self.closed.append(c)
        self.open.pop(p.symbol, None)
        return c


# --------------------------------------------------------------------- loop
def scan(client, logic: FullLogic, acct: Account, symbols, panel):
    """Refresh bars, close what is due, open what is offered."""
    for s in symbols:
        df = fetch_klines(client, s)
        if df is not None and len(df) > 60:
            panel[s] = df
    if len(panel) < 2:
        return 0, 0

    # --- exits first, so capital frees up before entries are considered
    closed_now = 0
    for s in list(acct.open):
        p = acct.open[s]
        df = panel.get(s)
        if df is None or df.empty:
            continue
        bar = df.iloc[-1]
        p.bars_held += 1
        p.high = max(p.high, float(bar["high"]))
        p.low = min(p.low, float(bar["low"]))
        reason, val = FullLogic.exit_now(
            p.dec, p.entry, float(bar["high"]), float(bar["low"]),
            float(bar["close"]), p.peak, p.bars_held)
        if reason is None:
            p.peak = val
            continue
        c = acct.close_position(p, reason, val)
        closed_now += 1
        print(f"  CLOSE {c.symbol:<12} {'LONG' if c.side > 0 else 'SHORT':<5} "
              f"{c.reason:<7} move {100*c.move:+6.2f}%  "
              f"pnl ${c.pnl:+.4f}  equity ${acct.equity:.4f}", flush=True)

    # --- entries
    opened_now = 0
    for s in symbols:
        if not acct.can_open(s):
            continue
        df = panel.get(s)
        if df is None or len(df) < 60:
            continue
        dec = logic.decide(df, panel, s)
        if dec is None:
            continue
        price = float(df["close"].iloc[-1])
        p = acct.open_position(dec, price)
        if p is None:
            acct.rejected += 1
            continue
        opened_now += 1
        print(f"  OPEN  {s:<12} {'LONG' if dec.side > 0 else 'SHORT':<5} "
              f"@{price:<12.6f} pot {dec.potential:5.1f}  "
              f"stake {100*dec.stake:4.0f}%  lev {dec.leverage:4.1f}x  "
              f"target {100*dec.target:+5.2f}%  stop {100*dec.stop:.2f}%",
              flush=True)
    return opened_now, closed_now


def dashboard(acct: Account, panel, started: float):
    wins = [c for c in acct.closed if c.pnl > 0]
    unreal = 0.0
    for s, p in acct.open.items():
        df = panel.get(s)
        if df is not None and not df.empty:
            unreal += (p.move(float(df["close"].iloc[-1]))
                       - p.dec.cost) * p.notional
    eq = acct.equity + unreal
    ret = 100.0 * (eq / acct.start - 1.0)
    up = time.time() - started
    print(f"\n  equity ${eq:.4f}  ({ret:+.2f}%)   realised "
          f"${acct.equity - acct.start:+.4f}   unrealised ${unreal:+.4f}")
    print(f"  open {len(acct.open)}   closed {len(acct.closed)}   "
          f"wins {len(wins)}/{len(acct.closed)}"
          + (f" ({100.0*len(wins)/len(acct.closed):.1f}%)"
             if acct.closed else "")
          + f"   fees ${acct.fees_paid:.4f}   up {up/60:.1f}m", flush=True)


def summary(acct: Account, panel):
    print("\n" + "=" * 78)
    print("  SESSION SUMMARY")
    print("=" * 78)
    wins = [c for c in acct.closed if c.pnl > 0]
    unreal = 0.0
    for s, p in acct.open.items():
        df = panel.get(s)
        px = float(df["close"].iloc[-1]) if df is not None and len(df) else p.entry
        m = p.move(px) - p.dec.cost
        unreal += m * p.notional
        print(f"  OPEN  {s:<12} {'LONG' if p.dec.side > 0 else 'SHORT':<5} "
              f"move {100*m:+6.2f}%  lev {p.dec.leverage:.1f}x  "
              f"mark ${m * p.notional:+.4f}")
    if acct.closed:
        by = {}
        for c in acct.closed:
            by.setdefault(c.reason, []).append(c)
        print("\n  exits")
        for r, cs in sorted(by.items()):
            w = sum(1 for c in cs if c.pnl > 0)
            print(f"    {r:<8} {len(cs):>4}   won {w}/{len(cs)}   "
                  f"pnl ${sum(c.pnl for c in cs):+.4f}")
        best = max(acct.closed, key=lambda c: c.pnl)
        worst = min(acct.closed, key=lambda c: c.pnl)
        print(f"\n  best  {best.symbol} {100*best.move:+.2f}% "
              f"${best.pnl:+.4f} at {best.leverage:.1f}x")
        print(f"  worst {worst.symbol} {100*worst.move:+.2f}% "
              f"${worst.pnl:+.4f} at {worst.leverage:.1f}x")
    eq = acct.equity + unreal
    print(f"\n  start    ${acct.start:.4f}")
    print(f"  realised ${acct.equity:.4f}")
    print(f"  mark     ${eq:.4f}   ({100.0*(eq/acct.start-1.0):+.2f}%)")
    print(f"  trades   {len(acct.closed)}   wins {len(wins)}"
          + (f"   ({100.0*len(wins)/len(acct.closed):.1f}%)"
             if acct.closed else ""))
    print(f"  fees     ${acct.fees_paid:.4f}")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(SYMBOLS))
    ap.add_argument("--equity", type=float, default=10.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=SCAN_SECONDS)
    a = ap.parse_args()

    symbols = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    logic = FullLogic(symbols)
    if not logic.ready:
        print("No fitted logic found. Build it first:\n\n"
              "    python -m fp.full\n\n"
              "That writes fp/models/<SYMBOL>.pkl per coin.", file=sys.stderr)
        return 2
    if logic.missing:
        print(f"  no model for {', '.join(logic.missing)} -- not traded")
    symbols = logic.ready

    print("=" * 78)
    print("  PAPER TRADING fp/full.py  --  virtual money, real prices")
    print(f"  equity ${a.equity:.2f}   coins {len(symbols)}   "
          f"one position per coin")
    for s in symbols:
        m = logic.meta[s]
        print(f"    {s:<12} lev<= {m['lev_cap']:.0f}x   "
              f"cost {100*m['cost']:.4f}%   gate {m['gate']:.3f}   "
              f"fitted win {100*m.get('win_rate', 0):.2f}% "
              f"on {m.get('trades', 0):,} past trades")
    print("=" * 78, flush=True)

    client = make_client()
    acct = Account(equity=a.equity, start=a.equity)
    panel: dict = {}
    started = time.time()
    stop = {"flag": False}

    def onint(*_):
        stop["flag"] = True
    signal.signal(signal.SIGINT, onint)

    try:
        while not stop["flag"]:
            t0 = time.time()
            try:
                scan(client, logic, acct, symbols, panel)
            except Exception as exc:                      # keep trading
                print(f"  scan error: {exc}", flush=True)
            dashboard(acct, panel, started)
            if a.once:
                break
            while time.time() - t0 < a.interval and not stop["flag"]:
                time.sleep(0.25)
    finally:
        summary(acct, panel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
