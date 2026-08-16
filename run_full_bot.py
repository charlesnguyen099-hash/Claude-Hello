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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fp import costs as C
from fp import full as FU
from fp import universe as U
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


class LiveFeed:
    """Closed 1-minute bars from Bybit's public endpoint.

    Fetched in parallel. Fifty symbols served one at a time do not fit
    inside a scan interval, and a scan that runs late is scoring bars
    that have already moved.
    """

    def __init__(self, client, workers: int = 8):
        self.client = client
        self.workers = workers
        self.live = True

    def bars(self, symbol: str):
        return fetch_klines(self.client, symbol)

    def bars_many(self, symbols):
        out = {}
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for sym, df in zip(symbols, pool.map(self.bars, symbols)):
                if df is not None:
                    out[sym] = df
        return out

    def advance(self) -> bool:
        return True


class ReplayFeed:
    """The same bars, from the cache, one minute at a time.

    Bybit is not reachable from every environment -- this sandbox's
    proxy denies api.bybit.com outright -- and "it imported cleanly" is
    not evidence that a bot trades. Replay walks cached history through
    the SAME scan, exits and accounting the live path uses, so the whole
    route is exercised without a network.

    The cursor is a TIMESTAMP, not a row number. Slicing by row number
    lines up BTCUSDT's 1,500th bar (January 2025) with BLESSUSDT's
    (June 2026) and hands the cross-section features -- a third of the
    200 columns -- readings from coins eighteen months apart. Live, all
    ten coins are always at the same minute; replay has to be too.
    """

    def __init__(self, panel: dict, start: int, steps: int):
        self.full = panel
        lo = max(d.index[0] for d in panel.values())
        hi = min(d.index[-1] for d in panel.values())
        axis = pd.date_range(lo, hi, freq="min")
        first = min(max(start, 300), max(len(axis) - 2, 0))
        self.axis = axis[first:first + max(steps, 1)]
        self.i = 0
        self.live = False

    @property
    def now(self):
        return self.axis[min(self.i, len(self.axis) - 1)]

    def bars(self, symbol: str):
        d = self.full.get(symbol)
        if d is None:
            return None
        return d.loc[:self.now].tail(BARS)

    def bars_many(self, symbols):
        return {s: d for s in symbols
                if (d := self.bars(s)) is not None}

    def advance(self) -> bool:
        self.i += 1
        return self.i < len(self.axis)


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
    """One finished trade, kept in full.

    Every field the operator needs to audit a trade after the fact:
    what was entered, at what price, on whose signal, how it was sized,
    where it came out and why. Aggregates hide the trade that went
    wrong; this does not.
    """
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
    entry: float = 0.0
    exit: float = 0.0
    margin: float = 0.0
    notional: float = 0.0
    target: float = 0.0
    stop: float = 0.0
    opened_at: float = 0.0
    closed_at: float = 0.0
    equity_after: float = 0.0
    n: int = 0
    source: str = "own"


@dataclass
class Account:
    equity: float
    start: float
    open: dict = field(default_factory=dict)
    closed: list = field(default_factory=list)
    fees_paid: float = 0.0
    rejected: int = 0
    # Below this a trade is dust: the fee is a bigger share of it than
    # any move it could make.
    min_margin: float = 0.01

    @property
    def committed(self) -> float:
        """Margin currently posted against open positions."""
        return sum(p.margin for p in self.open.values())

    @property
    def free(self) -> float:
        """What is left to open anything new with."""
        return max(self.equity - self.committed, 0.0)

    def can_open(self, symbol: str) -> bool:
        # ONE POSITION PER COIN AT A TIME. Coverage of every signal is a
        # build-time property; at trade time a coin holds one position
        # and a second signal on it simply waits.
        return symbol not in self.open

    def open_position(self, dec: Decision, price: float) -> Position | None:
        """Post margin out of FREE equity, never out of the total.

        This is the bug that put a paper account at -$27.21 on a $10
        start. Stake was a fraction of TOTAL equity and margin was never
        set aside, so a potential-100 signal took "100% of the account"
        -- and then did it again on the next coin, and the next. The
        scanner found 88 signals across 713 perpetuals and opened all
        88, each sized at the whole account and levered up to 54x. The
        account was committed 88 times over before a single trade
        closed, so the first losses had nothing left to come out of.
        fp/full.py never showed this: there, a coin trades one position
        at a time and the equity path is sequential, so total exposure
        can never exceed the account by construction. Live, it is 713
        coins at once and the constraint has to be enforced.

        Stake is now a fraction of what is FREE. When the account is
        fully committed there is nothing to open with, which is the
        correct answer rather than a hidden loan.
        """
        margin = dec.stake * self.free
        if margin < self.min_margin or margin > self.free:
            self.rejected += 1
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
        exit_px = p.entry * (1.0 + p.dec.side * move)
        c = Closed(symbol=p.symbol, side=p.dec.side, reason=reason,
                   move=move, pnl=pnl, fees=fees,
                   held_s=time.time() - p.opened_at,
                   leverage=p.dec.leverage, stake=p.dec.stake,
                   potential=p.dec.potential,
                   entry=p.entry, exit=exit_px, margin=p.margin,
                   notional=p.notional, target=p.dec.target,
                   stop=p.dec.stop, opened_at=p.opened_at,
                   closed_at=time.time(), equity_after=self.equity,
                   n=len(self.closed) + 1, source=p.dec.source)
        self.closed.append(c)
        self.open.pop(p.symbol, None)
        return c


# --------------------------------------------------------------------- loop
def scan(feed, logic: FullLogic, acct: Account, symbols, panel,
         reference=()):
    """Refresh bars, close what is due, open what is offered.

    `reference` is the FIXED set of coins the cross-section columns are
    computed against, and it is fetched every cycle whatever else is.
    A third of the 200 features ask "what are the other coins doing
    right now" -- rank, breadth, market move, dispersion, residual --
    so their meaning depends on which coins are in the panel. The
    models were fitted with the ten in the cache; scoring them against
    a panel of three, or against a rotating slice of six hundred, feeds
    the same column a different question every cycle. Holding the
    reference fixed is what makes a fitted model portable at all.
    """
    # Open positions are always refreshed, whatever slice of the board
    # this cycle is scanning -- a position the scanner rotated past is a
    # position with no stop.
    need = list(dict.fromkeys(list(reference) + list(symbols)
                              + list(acct.open)))
    for s, df in feed.bars_many(need).items():
        if len(df) > 60:
            panel[s] = df
    if not panel:
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
              f"[{dec.source}]  "
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
    # Committed and free are printed because their absence is what let
    # the account go to -$27.21 unnoticed: 88 positions each sized at
    # "100% of the account".
    print(f"  committed ${acct.committed:.4f}   free ${acct.free:.4f}   "
          f"exposure ${sum(p.notional for p in acct.open.values()):.2f}")
    print(f"  open {len(acct.open)}   closed {len(acct.closed)}   "
          f"wins {len(wins)}/{len(acct.closed)}"
          + (f" ({100.0*len(wins)/len(acct.closed):.1f}%)"
             if acct.closed else "")
          + f"   fees ${acct.fees_paid:.4f}   rejected {acct.rejected}"
          + f"   up {up/60:.1f}m", flush=True)


def _px(x: float) -> str:
    """Prices span $0.000002 to $100,000 on this board -- one width fails."""
    if x >= 1000:
        return f"{x:,.2f}"
    if x >= 1:
        return f"{x:.4f}"
    if x >= 0.01:
        return f"{x:.6f}"
    return f"{x:.8f}"


def _dur(sec: float) -> str:
    sec = int(max(sec, 0))
    if sec < 3600:
        return f"{sec // 60}m{sec % 60:02d}s"
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m"


def write_log(acct: Account, path: str = "trades.csv") -> str | None:
    """Every closed trade to CSV, because scrollback is not a record."""
    if not acct.closed:
        return None
    import csv
    cols = ["n", "symbol", "side", "opened", "closed", "held_s", "reason",
            "entry", "exit", "target", "stop", "potential", "stake",
            "leverage", "margin", "notional", "move", "pnl", "fees",
            "equity_after", "source"]
    try:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            for c in acct.closed:
                w.writerow([
                    c.n, c.symbol, "LONG" if c.side > 0 else "SHORT",
                    time.strftime("%Y-%m-%d %H:%M:%S",
                                  time.localtime(c.opened_at)),
                    time.strftime("%Y-%m-%d %H:%M:%S",
                                  time.localtime(c.closed_at)),
                    round(c.held_s, 1), c.reason,
                    c.entry, c.exit, round(c.target, 6), round(c.stop, 6),
                    round(c.potential, 2), round(c.stake, 4),
                    round(c.leverage, 2), round(c.margin, 6),
                    round(c.notional, 6), round(c.move, 6),
                    round(c.pnl, 6), round(c.fees, 6),
                    round(c.equity_after, 6), c.source])
        return path
    except OSError:
        return None


def summary(acct: Account, panel):
    print("\n" + "=" * 118)
    print("  SESSION SUMMARY")
    print("=" * 118)

    # EVERY trade, won or lost, in the order they happened. Aggregates
    # hide the one that went wrong; a ledger does not.
    if acct.closed:
        print("\n  ALL TRADES")
        print(f"  {'#':>4} {'coin':<14}{'side':<6}{'opened':<9}{'held':>8}"
              f"{'entry':>14}{'exit':>14}{'lev':>6}{'stake':>7}{'pot':>6}"
              f"{'move':>9}{'pnl $':>11}  {'why':<7}{'equity $':>11}")
        print("  " + "-" * 114)
        for c in acct.closed:
            print(f"  {c.n:>4} {c.symbol:<14}"
                  f"{'LONG' if c.side > 0 else 'SHORT':<6}"
                  f"{time.strftime('%H:%M:%S', time.localtime(c.opened_at)):<9}"
                  f"{_dur(c.held_s):>8}"
                  f"{_px(c.entry):>14}{_px(c.exit):>14}"
                  f"{c.leverage:>5.1f}x{100*c.stake:>6.0f}%{c.potential:>6.1f}"
                  f"{100*c.move:>+8.2f}%{c.pnl:>+11.4f}  {c.reason:<7}"
                  f"{c.equity_after:>11.4f}")

    unreal = 0.0
    if acct.open:
        print("\n  STILL OPEN")
        print(f"  {'coin':<14}{'side':<6}{'held':>8}{'entry':>14}{'mark':>14}"
              f"{'lev':>6}{'stake':>7}{'move':>9}{'unreal $':>11}")
        print("  " + "-" * 89)
        for sym, p in sorted(acct.open.items()):
            df = panel.get(sym)
            px = (float(df["close"].iloc[-1])
                  if df is not None and len(df) else p.entry)
            m = p.move(px) - p.dec.cost
            unreal += m * p.notional
            print(f"  {sym:<14}{'LONG' if p.dec.side > 0 else 'SHORT':<6}"
                  f"{_dur(time.time() - p.opened_at):>8}"
                  f"{_px(p.entry):>14}{_px(px):>14}"
                  f"{p.dec.leverage:>5.1f}x{100*p.dec.stake:>6.0f}%"
                  f"{100*m:>+8.2f}%{m * p.notional:>+11.4f}")

    wins = [c for c in acct.closed if c.pnl > 0]
    losses = [c for c in acct.closed if c.pnl <= 0]
    if acct.closed:
        by = {}
        for c in acct.closed:
            by.setdefault(c.reason, []).append(c)
        print("\n  BY EXIT")
        for r, cs in sorted(by.items()):
            w = sum(1 for c in cs if c.pnl > 0)
            print(f"    {r:<8} {len(cs):>5}   won {w}/{len(cs)}   "
                  f"pnl ${sum(c.pnl for c in cs):+.4f}")
        print("\n  BY LOGIC")
        # The split that matters: a coin's own logic against logic
        # borrowed from another coin. On the operator's first live run
        # every one of the twelve losses came from a borrowed logic and
        # not one fitted coin closed red.
        src = {}
        for c in acct.closed:
            src.setdefault(c.source, []).append(c)
        for k, cs in sorted(src.items(),
                            key=lambda kv: -sum(c.pnl for c in kv[1])):
            w = sum(1 for c in cs if c.pnl > 0)
            label = "own coin's logic" if k == "own" else f"logic from {k}"
            print(f"    {label:<26} {len(cs):>4} trades   won {w}/{len(cs)}"
                  f"   pnl ${sum(c.pnl for c in cs):+.4f}")

        print("\n  BY COIN")
        per = {}
        for c in acct.closed:
            per.setdefault(c.symbol, []).append(c)
        for sym, cs in sorted(per.items(),
                              key=lambda kv: -sum(c.pnl for c in kv[1])):
            w = sum(1 for c in cs if c.pnl > 0)
            print(f"    {sym:<14} {len(cs):>4} trades   won {w}/{len(cs)}   "
                  f"pnl ${sum(c.pnl for c in cs):+.4f}")
        if losses:
            print(f"\n  LOSING TRADES: {len(losses)}")
            for c in losses:
                print(f"    #{c.n} {c.symbol} "
                      f"{'LONG' if c.side > 0 else 'SHORT'} "
                      f"{100*c.move:+.2f}% at {c.leverage:.1f}x "
                      f"= ${c.pnl:+.4f}   exit: {c.reason}")
        else:
            print("\n  LOSING TRADES: none")

    eq = acct.equity + unreal
    print(f"\n  start    ${acct.start:.4f}")
    print(f"  realised ${acct.equity:.4f}   ({acct.equity - acct.start:+.4f})")
    print(f"  mark     ${eq:.4f}   ({100.0*(eq/acct.start-1.0):+.2f}%)")
    print(f"  trades   {len(acct.closed)} closed, {len(acct.open)} open   "
          f"wins {len(wins)}"
          + (f" ({100.0*len(wins)/len(acct.closed):.1f}%)"
             if acct.closed else ""))
    print(f"  fees     ${acct.fees_paid:.4f}")
    print(f"  capital  committed ${acct.committed:.4f}, "
          f"free ${acct.free:.4f}, {acct.rejected} signals skipped for "
          f"want of margin")
    if acct.equity < 0:
        print("  WARNING: negative equity means the exposure accounting "
              "let positions be opened on capital that was not there.")
    path = write_log(acct)
    if path:
        print(f"\n  full ledger written to {path}")
    print("=" * 118)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=None,
                    help="explicit list; default is the whole Bybit board")
    ap.add_argument("--top", type=int, default=50,
                    help="scanned every cycle, ranked by 24h turnover")
    ap.add_argument("--sweep", type=int, default=50,
                    help="tail coins added per cycle, rotating")
    ap.add_argument("--fitted-only", action="store_true",
                    help="trade only coins fp/full.py fitted directly")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--equity", type=float, default=10.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=SCAN_SECONDS)
    ap.add_argument("--replay", action="store_true",
                    help="walk cached history instead of live Bybit")
    ap.add_argument("--replay-start", type=int, default=1500)
    ap.add_argument("--replay-steps", type=int, default=4000)
    a = ap.parse_args()

    fitted = sorted({f.name.split(".")[0]
                     for f in FU.MODELS.glob("*.pkl*")})
    if not fitted:
        print("No fitted logic in fp/models/. Build it first:\n\n"
              "    python -m fp.full\n", file=sys.stderr)
        return 2
    logic = FullLogic(fitted)

    print("=" * 78)
    print("  PAPER TRADING fp/full.py  --  virtual money, real prices")
    print(f"  equity ${a.equity:.2f}   one position per coin, no cap on "
          f"how many coins")
    for q in fitted:
        m = logic.meta[q]
        print(f"    fitted {q:<12} lev<= {m['lev_cap']:.0f}x  "
              f"cost {100*m['cost']:.4f}%  gate {m['gate']:.3f}  "
              f"win {100*m.get('win_rate', 0):.2f}% on "
              f"{m.get('trades', 0):,} past trades")

    rotation = None
    if a.replay:
        from fp import data as D
        cached = D.load()
        if a.symbols:
            symbols = [q.strip().upper() for q in a.symbols.split(",")
                       if q.strip()]
        else:
            symbols = fitted
        symbols = [q for q in symbols if q in cached]
        # The feed carries EVERY cached coin even when only a few are
        # traded, because the cross-section columns are computed against
        # the reference panel and a reference the feed cannot serve is
        # not a reference. --symbols narrows what is traded, not what is
        # looked at.
        feed = ReplayFeed(dict(cached), a.replay_start, a.replay_steps)
        print(f"  REPLAY: {len(symbols)} coin(s), "
              f"{feed.axis[0]} .. {feed.axis[-1]} from cache, no network")
    else:
        client = make_client()
        feed = LiveFeed(client, workers=a.workers)
        if a.symbols:
            symbols = [q.strip().upper() for q in a.symbols.split(",")
                       if q.strip()]
            U.all_perpetuals(client)          # cache real leverage caps
        elif a.fitted_only:
            symbols = fitted
            U.all_perpetuals(client)
        else:
            # THE WHOLE BOARD. Every USDT perpetual Bybit lists, ranked
            # by 24h turnover, with the real per-coin leverage ceiling
            # cached for the sizing code to read.
            symbols, caps, tvr = U.ranked(client)
            print(f"  universe: {len(symbols)} USDT perpetuals from Bybit")
        rotation = U.Rotation(symbols, top=a.top, slice_size=a.sweep)
        print(f"  scanning: top {len(rotation.top)} every cycle, "
              f"{len(rotation.tail)} more in slices of {a.sweep} "
              f"(full sweep every {rotation.cycles_for_full_sweep} cycles)")
        print(f"  logic:    {len(fitted)} fitted models; coins without one "
              f"are scored by unanimous vote of all {len(fitted)}")
    print("=" * 78, flush=True)
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
                batch = rotation.next_batch() if rotation else symbols
                scan(feed, logic, acct, batch, panel, reference=fitted)
            except Exception as exc:                      # keep trading
                print(f"  scan error: {exc}", flush=True)
            if a.once:
                dashboard(acct, panel, started)
                break
            if not feed.advance():
                break
            if feed.live:
                dashboard(acct, panel, started)
                while time.time() - t0 < a.interval and not stop["flag"]:
                    time.sleep(0.25)
            elif len(acct.closed) and len(acct.closed) % 25 == 0:
                dashboard(acct, panel, started)
    finally:
        summary(acct, panel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
