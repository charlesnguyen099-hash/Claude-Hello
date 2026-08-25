#!/usr/bin/env python3
"""Trade the fp/full.py logic on live Bybit prices. Paper by default.

    pip install pandas numpy scikit-learn pybit
    python -m fp.full          # build the logic (once, slow)
    python run_full_bot.py     # trade it (no key, no orders, ever)

By default: no API key, no account, no order is ever placed. It reads
Bybit's public kline and ticker endpoints and keeps the balance in
memory. --real-trade (see fp/broker.py) turns this into a real Bybit
account with real orders -- it needs its own flag pair to run at all
(see --real-trade below), on purpose.

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

    python run_full_bot.py               # the coins the logic was built on
    python run_full_bot.py --all-coins   # the whole Bybit board, borrowed logic
    python run_full_bot.py --equity 100
    python run_full_bot.py --once        # one scan, print, exit
    python run_full_bot.py --retrain-hours 0   # disable the retrain thread
    python run_full_bot.py --retrain-continuous  # back-to-back, no wait

    # REAL orders, real money -- both flags required together, on purpose:
    python run_full_bot.py --real-trade --i-understand-real-money
    # Prove the order path works with fake money first:
    python run_full_bot.py --real-trade --i-understand-real-money --testnet

BY DEFAULT IT TRADES ONLY THE FITTED COINS. Every measurement that says
this logic works -- 100% of reachable signals recovered, 100% win rate,
never liquidated -- was made on those coins and only those. Applying a
coin's logic to a coin it was never built on is a separate claim, and
the evidence is against it: fp/transfer.py finds 0/10 coins clearing
their own break-even at p<0.05, and the first live run of the whole
board lost twelve trades out of fifteen, every one of them on borrowed
logic. --all-coins is there because it was asked for; it is not the
default because it has not been shown to work.

A FITTED MODEL GOES STALE. It reproduces 100% of the trades in the
window it was shown, but that window stops moving the moment the fit
finishes, and the market does not. Measured honestly -- fp/full.py's
oof_gate(), several expanding folds walked forward in time, isotonic
calibration, no lookahead -- confidence on genuinely unseen bars falls
the CLOSER the held-out fold sits to the present, the signature of a
regime the training window never saw. So this drives its own retraining
by default: every --retrain-hours (24 by default, live mode only), it
pulls whatever bars Bybit has produced since the last cycle, folds them
into the cache, and reruns the exact fp/full.py pipeline this session
built -- the 100%-recovery, 100%-win requirement on all data then in the
cache is enforced identically every time, it is just that "all data"
keeps growing. On success the running bot swaps in the freshly fitted
models without a restart. See fp/retrain.py for the mechanism and why a
single fixed fit cannot be the end state of this design.

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
import json
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from fp import costs as C
from fp import full as FU
from fp import retrain as RT
from fp import universe as U
from fp.live_full import Decision, FullLogic

BARS = 1000          # Bybit's cap for one kline call
SCAN_SECONDS = 20    # how often to re-price open positions
TAKER = C.TAKER_PER_SIDE


# --------------------------------------------------------------- market data
def make_client(testnet: bool = False):
    from pybit.unified_trading import HTTP
    return HTTP(testnet=testnet)


def real_credentials() -> tuple[str, str]:
    """The API key/secret for --real-trade, never written anywhere.

    Env vars first (BYBIT_API_KEY/BYBIT_API_SECRET) -- so this can run
    unattended once configured -- then an interactive, non-echoing
    prompt. Never logged, never part of an argv (which shells keep in
    history and other local users can read via `ps`), never cached to
    disk: the only place either value exists is this process's memory
    and whatever pybit's HTTP client does with them over TLS.
    """
    import getpass
    key = _os.environ.get("BYBIT_API_KEY")
    secret = _os.environ.get("BYBIT_API_SECRET")
    if key and secret:
        return key, secret
    print("\n  Real trading needs a Bybit API key with trading "
          "permission (not withdrawal).")
    print("  Create one at https://www.bybit.com/app/user/api-management "
          "if you have not.")
    key = key or input("  API key: ").strip()
    secret = secret or getpass.getpass("  API secret (hidden): ").strip()
    if not key or not secret:
        print("  Both an API key and secret are required.", file=sys.stderr)
        sys.exit(2)
    return key, secret


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
    # The timestamp of the last CLOSED bar this position was updated on.
    # Scans run every 20s but a 1-minute bar changes once a minute, so
    # counting scans counted the same bar three times.
    last_bar: object = None
    # The EXACT quantity a real order opened this position with (0 in
    # paper mode). The close order must send back this same number, not
    # a freshly recomputed one -- margin/notional can drift a hair from
    # rounding, and closing anything other than what was actually opened
    # leaves a dangling real position on the exchange.
    qty: float = 0.0

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
    # Bar a coin was last acted on, so the same candle is never traded
    # twice. Without it the bot re-entered SNDKUSDT three times off one
    # bar -- same entry 1,773.26, same exit, three sets of fees -- and
    # BLESSUSDT twice off another.
    last_action: dict = field(default_factory=dict)
    # A LiveBroker, or None for paper trading. Every open/close still
    # sizes and prices itself exactly as before -- the broker only
    # decides whether a REAL market order also gets sent alongside the
    # internal bookkeeping, never changes what that bookkeeping is.
    broker: object = None
    # REAL MODE: symbols with a real position this process has no
    # persisted history for. Never opened, never closed, never counted
    # -- see can_open()'s docstring.
    unmanaged: set = field(default_factory=set)

    @property
    def committed(self) -> float:
        """Margin currently posted against open positions."""
        return sum(p.margin for p in self.open.values())

    @property
    def free(self) -> float:
        """What is left to open anything new with -- 'vốn lúc trade'."""
        return max(self.equity - self.committed, 0.0)

    def can_open(self, symbol: str) -> bool:
        # ONE POSITION PER COIN AT A TIME. Coverage of every signal is a
        # build-time property; at trade time a coin holds one position
        # and a second signal on it simply waits.
        #
        # `unmanaged` (REAL MODE only): a real position exists on the
        # exchange for this symbol but no persisted Decision does --
        # opened outside this bot, or the state file from before a
        # restart was lost. Opening a SECOND position on top of one
        # this process cannot exit correctly would be worse than
        # refusing; the operator handles that symbol by hand.
        return symbol not in self.open and symbol not in self.unmanaged

    def open_position(self, dec: Decision, price: float) -> Position | None:
        """5% of capital available RIGHT NOW, never less than the
        exchange will actually accept, never rejected for being small.

        THE FLOOR IS RELATIVE, NOT ABSOLUTE. "Vốn lúc trade" -- the
        capital a trade is sized against -- is whatever is free at that
        moment, which already reflects every earlier trade: $10 start,
        first trade at least 5% of $10 = $0.50; if that leaves $9.50
        free, the next trade is at least 5% of $9.50 = $0.475; and so
        on, compounding down as capital gets committed and up again as
        positions close and profit returns to the pool. dec.stake is
        already 5%..100% BY CONSTRUCTION (fp.full's MIN_STAKE), a
        fraction of THE ACCOUNT -- so `dec.stake * self.free` already
        IS "at least 5% of capital right now" with no extra floor
        logic needed. An earlier version multiplied against a fixed
        equity snapshot instead and rejected trades free capital
        couldn't cover; that answered a question nobody asked.

        THE OTHER FLOOR IS THE EXCHANGE'S, NOT OURS. Bybit will not
        open a position below that symbol's own minimum order value,
        and that minimum turns into a MARGIN requirement once leverage
        is applied: min_notional / leverage. If the 5%-of-free amount
        would fall under it, the trade is bumped up to the exchange
        minimum instead of being silently sent to fail at the order --
        a higher-leverage setup needs less margin to clear the same
        notional floor, which is the "kết hợp với leverage" part.

        The ONLY reason to skip a trade is that even the exchange's own
        minimum does not fit in what is free -- not that our number was
        smaller than 5% of the original stake, which is expected and
        fine.
        """
        margin = dec.stake * self.free
        need = C.min_notional(dec.symbol, price) / max(dec.leverage, 1e-9)
        if margin < need:
            margin = need
        # margin <= 0 covers free == 0 exactly: 5% of nothing is
        # nothing, and a zero-margin "position" is not a trade.
        if margin <= 0 or margin > self.free:
            self.rejected += 1
            return None
        notional = margin * dec.leverage
        qty = 0.0
        entry = price
        # REAL MODE: a real market order must actually fill before any
        # internal bookkeeping is created. A rejected or errored order
        # must produce NO Position -- an internal record of a trade the
        # exchange never made would silently diverge from the real
        # account with every scan after it.
        if self.broker is not None:
            qty = C.round_qty(dec.symbol, notional / price, price)
            if qty <= 0:
                self.rejected += 1
                return None
            try:
                self.broker.market_order(dec.symbol, dec.side, qty,
                                        reduce_only=False)
                real = self.broker.open_positions().get(dec.symbol)
            except Exception as exc:
                print(f"  REAL ORDER FAILED opening {dec.symbol}: {exc}",
                      flush=True)
                self.rejected += 1
                return None
            if real is None or real["qty"] <= 0:
                print(f"  REAL ORDER for {dec.symbol} reported no error "
                      f"but no position exists after it -- treating as "
                      f"failed, not tracking internally", flush=True)
                self.rejected += 1
                return None
            entry = real["entry"] or price
            qty = real["qty"]
            notional = qty * entry
            margin = notional / dec.leverage
        p = Position(symbol=dec.symbol, dec=dec, entry=entry, margin=margin,
                     notional=notional, opened_at=time.time(),
                     high=entry, low=entry, qty=qty)
        self.open[dec.symbol] = p
        return p

    def close_position(self, p: Position, reason: str,
                       move: float) -> Closed | None:
        # REAL MODE: the exit order goes out FIRST, and only on success
        # does bookkeeping treat the position as gone. A failed close
        # order must leave the position in self.open exactly as it was
        # -- still genuinely open on the exchange -- so the next scan
        # retries it, rather than the bot silently believing a still-
        # live, unprotected position is flat. Returning None here (not
        # popping, not appending to self.closed) is what makes that
        # retry automatic: exit_now() runs again next cycle against
        # whatever the market did in between.
        if self.broker is not None:
            try:
                self.broker.market_order(p.symbol, -p.dec.side, p.qty,
                                        reduce_only=True)
            except Exception as exc:
                print(f"  REAL ORDER FAILED closing {p.symbol}: {exc} "
                      f"-- position remains open, retrying next scan",
                      flush=True)
                return None
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


# --------------------------------------------------------- state persistence
# REAL MODE ONLY. A restart must never touch a position genuinely open on
# the exchange -- not close it, not lose the target/stop/trail it was
# opened with. Those numbers exist only in the Decision this process
# computed at entry time; nothing about them can be recovered from the
# exchange itself, which knows a quantity and an average price and
# nothing about why. So every open/close writes the full Position (and
# its Decision) here, and a restart reads it back and MATCHES it against
# what the exchange says is actually open -- adopting a position only
# when both agree, refusing to touch one only either side knows about.
STATE_FILE = Path(__file__).resolve().parent / "data" / "real_positions.json"


def save_state(acct: "Account") -> None:
    if acct.broker is None:
        return
    try:
        data = {}
        for sym, p in acct.open.items():
            d = asdict(p)
            d["last_bar"] = str(p.last_bar) if p.last_bar is not None else None
            data[sym] = d
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        tmp.replace(STATE_FILE)          # atomic: never a half-written file
    except Exception as exc:
        print(f"  could not persist open-position state: {exc}", flush=True)


def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}


def position_from_state(sym: str, saved: dict) -> Position:
    dec = Decision(**saved["dec"])
    kwargs = {k: v for k, v in saved.items() if k not in ("dec", "last_bar")}
    lb = saved.get("last_bar")
    return Position(dec=dec, last_bar=pd.Timestamp(lb) if lb else None,
                    **kwargs)


class LogicHolder:
    """The live FullLogic, swappable in place without a restart.

    CPython attribute assignment is atomic under the GIL, so the scan
    loop reading `holder.logic` and the retrain thread writing it never
    tears -- the loop always sees either the old, fully-formed object or
    the new one, never a half-built one.
    """

    def __init__(self, logic: FullLogic):
        self.logic = logic
        # Retrain status, read by dashboard() every cycle so "is it
        # retraining right now" doesn't require scrolling back through
        # scan output to find the last RETRAIN CYCLE banner. Written
        # only by retrain_worker's own thread; CPython attribute
        # assignment is atomic under the GIL, same guarantee as .logic
        # above.
        self.retrain_status = "idle"     # "idle" | "running"
        self.retrain_started = None      # time.time() of the current/last start
        self.retrain_finished = None     # time.time() of the last finish
        self.retrain_result = None       # "OK" | "FAILED" | None (never run)


def retrain_worker(holder: LogicHolder, symbols, hours: float,
                   stop_event: threading.Event, continuous: bool = False
                   ) -> None:
    """Refresh the cache and refit on a schedule; swap the logic in.

    Runs fp.retrain.cycle(), which is itself a subprocess call to
    `python -m fp.full --fresh` -- fitting happens in a CHILD process,
    so this thread only blocks waiting for it, and the fitting pipeline's
    own memory history (the OOM history fp/full.py's comments record)
    never touches a bot that may already have been running for days.

    CONTINUOUS MEANS BACK-TO-BACK, NOT INSTANT. A full board refit is
    itself hours long -- BTCUSDT's fit and held-out gate alone run
    past an hour -- so the next cycle can never start sooner than that
    regardless of how this is configured. `continuous=True` removes
    only the ARTIFICIAL wait between cycles (no --retrain-hours pause,
    and the first cycle starts immediately instead of waiting a full
    interval before ever running), so a live_gate check runs against
    every stretch of new market data as soon as it exists, rather than
    on a fixed clock that may sit idle for the rest of an interval
    after a cycle already finished early.
    """
    while True:
        if continuous:
            if stop_event.is_set():
                break
        elif stop_event.wait(hours * 3600):
            break
        holder.retrain_status = "running"
        holder.retrain_started = time.time()
        try:
            ok = RT.cycle(symbols, log=lambda s: print(s, flush=True))
        except Exception as exc:
            print(f"  retrain cycle raised: {exc}", flush=True)
            holder.retrain_status = "idle"
            holder.retrain_finished = time.time()
            holder.retrain_result = "FAILED"
            continue
        if not ok:
            print("  retrain cycle failed; keeping the current models",
                  flush=True)
            holder.retrain_status = "idle"
            holder.retrain_finished = time.time()
            holder.retrain_result = "FAILED"
            continue
        fitted_now = sorted({f.name.split(".")[0]
                             for f in FU.MODELS.glob("*.pkl*")})
        if symbols:
            fitted_now = [s for s in fitted_now if s in symbols]
        if not fitted_now:
            print("  retrain produced no models; keeping the current ones",
                  flush=True)
            holder.retrain_status = "idle"
            holder.retrain_finished = time.time()
            holder.retrain_result = "FAILED"
            continue
        holder.logic = FullLogic(fitted_now)
        holder.retrain_status = "idle"
        holder.retrain_finished = time.time()
        holder.retrain_result = "OK"
        print(f"  retrain cycle complete: {len(fitted_now)} coin(s) "
              f"reloaded, live from the next scan", flush=True)


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
    # REAL MODE: re-anchor to the account's actual current balance
    # every cycle, whichever direction it moved and for whatever
    # reason -- a winning trade, a losing one, or money the operator
    # added or withdrew by hand. Internal bookkeeping (`+= pnl` on
    # every close, so entries later in THIS same cycle size off an
    # up-to-date number) still runs between resyncs; this is what keeps
    # it from drifting from the truth for longer than one scan interval.
    if acct.broker is not None:
        try:
            acct.equity = acct.broker.wallet_equity()
        except Exception as exc:
            print(f"  could not resync real equity this cycle, keeping "
                  f"the last known value: {exc}", flush=True)
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
        stamp = df.index[-1]
        if p.last_bar is not None and stamp == p.last_bar:
            continue                 # same candle, nothing new to judge
        p.last_bar = stamp
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
        if c is None:
            # The real close order failed -- p is still in acct.open,
            # untouched, so exit_now() runs again next scan against
            # fresh bars. last_action is NOT stamped either: this was
            # not a decision that was acted on.
            continue
        closed_now += 1
        # Mark this bar as already acted on for s, or the entries loop
        # below -- running later in this SAME scan() call, against this
        # SAME just-fetched panel -- sees the coin free again (can_open
        # is true the instant close_position pops it) and a stale
        # last_action (frozen from before this position was opened,
        # since the entries loop skips a symbol outright while it holds
        # a position and never touches last_action for it) reads as "a
        # new bar", so it reopens on the exact price that just closed.
        # That is how one HYPEUSDT setup traded itself from $10 to
        # $2000+ and back in twenty-second steps: not a losing signal,
        # a signal replayed dozens of times against its own single move.
        acct.last_action[s] = stamp
        print(f"  CLOSE {c.symbol:<12} {'LONG' if c.side > 0 else 'SHORT':<5} "
              f"{c.reason:<7} move {100*c.move:+6.2f}%  "
              f"pnl ${c.pnl:+.4f}  equity ${acct.equity:.4f}", flush=True)

    # --- entries
    # SCORE EVERY SIGNAL FIRST, ALLOCATE THE STRONGEST FIRST. The shared
    # pool is finite; granting it in scan order (roughly 24h turnover
    # rank) let whichever coin happened to be evaluated first exhaust
    # free capital before a much higher-potential signal later in the
    # same cycle was even scored -- capital sized by potential, but
    # HANDED OUT by an order that had nothing to do with it. This does
    # not shrink any trade's stake or leverage; every number a Decision
    # carries is exactly what fp.live_full computed. It only decides
    # WHICH signal gets first claim when several compete for the same
    # dollars in the same cycle.
    candidates = []
    for s in symbols:
        if not acct.can_open(s):
            continue
        df = panel.get(s)
        if df is None or len(df) < 60:
            continue
        stamp = df.index[-1]
        # ONE DECISION PER CANDLE. A signal is a property of a closed
        # bar, not of the clock: acting on it again twenty seconds later
        # is the same trade booked twice, at the same entry, paying the
        # round trip each time.
        if acct.last_action.get(s) == stamp:
            continue
        dec = logic.decide(df, panel, s)
        acct.last_action[s] = stamp
        if dec is None:
            continue
        price = float(df["close"].iloc[-1])
        candidates.append((dec.potential, s, dec, price))
    candidates.sort(key=lambda c: c[0], reverse=True)

    opened_now = 0
    for _, s, dec, price in candidates:
        p = acct.open_position(dec, price)
        if p is None:
            acct.rejected += 1
            continue
        p.last_bar = acct.last_action[s]
        opened_now += 1
        print(f"  OPEN  {s:<12} {'LONG' if dec.side > 0 else 'SHORT':<5} "
              f"@{price:<12.6f} pot {dec.potential:5.1f}  "
              f"stake {100*dec.stake:4.0f}%  lev {dec.leverage:4.1f}x  "
              f"[{dec.source}]  "
              f"target {100*dec.target:+5.2f}%  stop {100*dec.stop:.2f}%",
              flush=True)
    if opened_now or closed_now:
        save_state(acct)
    return opened_now, closed_now


def _retrain_line(holder: "LogicHolder | None") -> str:
    """One line answering 'is it retraining right now', for dashboard().

    Without this, the only evidence is the RETRAIN CYCLE banner
    scrolling by in a stream of P&L lines -- easy to lose track of
    over a session running for hours or days.
    """
    if holder is None:
        return ""
    if holder.retrain_status == "running":
        mins = (time.time() - holder.retrain_started) / 60.0
        return f"  retrain: RUNNING (started {mins:.0f}m ago)"
    if holder.retrain_finished is not None:
        mins = (time.time() - holder.retrain_finished) / 60.0
        return (f"  retrain: idle, last cycle {holder.retrain_result} "
                f"{mins:.0f}m ago")
    return "  retrain: not started yet"


def dashboard(acct: Account, panel, started: float,
             holder: "LogicHolder | None" = None):
    wins = [c for c in acct.closed if c.pnl > 1e-9]
    flat = [c for c in acct.closed if abs(c.pnl) <= 1e-9]
    losses = [c for c in acct.closed if c.pnl < -1e-9]
    unreal = 0.0
    for s, p in acct.open.items():
        df = panel.get(s)
        if df is not None and not df.empty:
            unreal += (p.move(float(df["close"].iloc[-1]))
                       - p.dec.cost) * p.notional
    eq = acct.equity + unreal
    ret = 100.0 * (eq / acct.start - 1.0)
    up = time.time() - started
    # ONE LINE, ALWAYS VISIBLE, THAT ANSWERS "IS THIS BOT WINNING OR
    # LOSING RIGHT NOW": the operator should never have to wait for
    # Ctrl+C or scroll back through OPEN/CLOSE lines to find out.
    print(f"\n  P&L: {'+' if eq >= acct.start else ''}${eq - acct.start:.4f} "
          f"({ret:+.2f}%) on ${acct.start:.2f} start   "
          f"won {len(wins)} / flat {len(flat)} / lost {len(losses)}"
          + (f"  ({100.0*len(wins)/len(acct.closed):.1f}% won)"
             if acct.closed else ""))
    print(f"  equity ${eq:.4f}  ({ret:+.2f}%)   realised "
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
    line = _retrain_line(holder)
    if line:
        print(line, flush=True)
    # REAL MODE: the internal `eq` above is arithmetic on top of every
    # fill this process believes happened. It should track the real
    # wallet closely -- this is the check that says so, or says it does
    # not, rather than the operator having to trust the arithmetic on
    # faith. Never fed back into acct.equity automatically: that would
    # blend real unrealised pnl (which includes positions this process
    # may not know about) into bookkeeping this process's own sizing
    # logic depends on being self-consistent.
    if acct.broker is not None:
        try:
            real_eq = acct.broker.wallet_equity()
            drift = real_eq - eq
            flag = "" if abs(drift) < 0.02 * max(eq, 1.0) else "  <-- CHECK"
            print(f"  real wallet equity ${real_eq:.4f}   "
                  f"(internal tracking is {'+' if drift >= 0 else ''}"
                  f"{drift:.4f} off){flag}", flush=True)
        except Exception as exc:
            print(f"  could not read real wallet equity: {exc}", flush=True)


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

    unreal = 0.0
    for sym, p in acct.open.items():
        df = panel.get(sym)
        px = (float(df["close"].iloc[-1])
              if df is not None and len(df) else p.entry)
        unreal += (p.move(px) - p.dec.cost) * p.notional
    eq = acct.equity + unreal
    wins = [c for c in acct.closed if c.pnl > 1e-9]
    flat = [c for c in acct.closed if abs(c.pnl) <= 1e-9]
    losses = [c for c in acct.closed if c.pnl < -1e-9]
    # THE ANSWER, FIRST LINE, BEFORE ANY TABLE: won or lost, how much,
    # how many of each. Everything below this is how it got there.
    print(f"\n  P&L: {'+' if eq >= acct.start else ''}${eq - acct.start:.4f} "
          f"({100.0*(eq/acct.start-1.0):+.2f}%) on ${acct.start:.2f} start"
          f"   won {len(wins)} / flat {len(flat)} / lost {len(losses)}"
          + (f"  ({100.0*len(wins)/len(acct.closed):.1f}% won)"
             if acct.closed else "  (no trades closed)"))

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
            print(f"  {sym:<14}{'LONG' if p.dec.side > 0 else 'SHORT':<6}"
                  f"{_dur(time.time() - p.opened_at):>8}"
                  f"{_px(p.entry):>14}{_px(px):>14}"
                  f"{p.dec.leverage:>5.1f}x{100*p.dec.stake:>6.0f}%"
                  f"{100*m:>+8.2f}%{m * p.notional:>+11.4f}")

    # Three outcomes, not two. A trade the break-even ratchet closes at
    # zero neither won nor lost, and lumping it in with the losses hides
    # whether anything actually lost money. (wins/flat/losses/unreal/eq
    # were computed above, for the P&L headline.)
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
        if flat:
            print(f"\n  FLAT (closed at break-even by the ratchet): "
                  f"{len(flat)}")
        if losses:
            print(f"\n  LOSING TRADES: {len(losses)}")
            for c in losses:
                print(f"    #{c.n} {c.symbol} "
                      f"{'LONG' if c.side > 0 else 'SHORT'} "
                      f"{100*c.move:+.2f}% at {c.leverage:.1f}x "
                      f"= ${c.pnl:+.4f}   exit: {c.reason}")
        else:
            print("\n  LOSING TRADES: none")

    print(f"\n  start    ${acct.start:.4f}")
    print(f"  realised ${acct.equity:.4f}   ({acct.equity - acct.start:+.4f})")
    print(f"  mark     ${eq:.4f}   ({100.0*(eq/acct.start-1.0):+.2f}%)")
    print(f"  trades   {len(acct.closed)} closed, {len(acct.open)} open   "
          f"won {len(wins)}, flat {len(flat)}, lost {len(losses)}"
          + (f"   ({100.0*len(wins)/len(acct.closed):.1f}% won)"
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
    ap.add_argument("--all-coins", action="store_true",
                    help="scan the whole Bybit board, applying fitted "
                         "logic to coins it was not built on")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--equity", type=float, default=10.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=SCAN_SECONDS)
    ap.add_argument("--replay", action="store_true",
                    help="walk cached history instead of live Bybit")
    ap.add_argument("--replay-start", type=int, default=1500)
    ap.add_argument("--replay-steps", type=int, default=4000)
    ap.add_argument("--retrain-hours", type=float, default=24.0,
                    help="refresh the cache and refit on this cadence "
                         "(live mode only); 0 disables it")
    ap.add_argument("--retrain-continuous", action="store_true",
                    help="retrain back-to-back with no wait between "
                         "cycles (overrides --retrain-hours); a cycle "
                         "is still hours long on its own, so this is "
                         "'as soon as the last one finished', not "
                         "'instant'")
    ap.add_argument("--real-trade", action="store_true",
                    help="place REAL orders on Bybit with a REAL API "
                         "key against REAL money. Requires "
                         "--i-understand-real-money too, on purpose --"
                         " a single flag is too easy to pass by habit.")
    ap.add_argument("--i-understand-real-money", action="store_true",
                    help="required alongside --real-trade; exists so "
                         "real trading never starts from one flag typed "
                         "on reflex")
    ap.add_argument("--testnet", action="store_true",
                    help="with --real-trade, use Bybit's TESTNET "
                         "instead of mainnet -- fake money, same API, "
                         "the way to prove the order path works before "
                         "risking anything real")
    a = ap.parse_args()

    if a.real_trade and not a.i_understand_real_money:
        print("--real-trade also needs --i-understand-real-money. "
              "This places real orders with real money -- both flags "
              "exist so that never happens by accident.", file=sys.stderr)
        return 2
    if a.real_trade and a.replay:
        print("--real-trade and --replay cannot be combined: replay "
              "walks CACHED history, and a real order against a bar "
              "from the past makes no sense.", file=sys.stderr)
        return 2

    fitted = sorted({f.name.split(".")[0]
                     for f in FU.MODELS.glob("*.pkl*")})
    if not fitted:
        print("No fitted logic in fp/models/. Build it first:\n\n"
              "    python -m fp.full\n", file=sys.stderr)
        return 2
    holder = LogicHolder(FullLogic(fitted))
    logic = holder.logic       # for the startup banner below only

    print("=" * 78)
    # This banner used to hardcode "PAPER TRADING" no matter what --
    # a real-money run under --real-trade still opened with that line,
    # and the only correction came 90-odd lines later. Say up front
    # which one this actually is.
    if a.real_trade:
        print("  fp/full.py  --  REAL MONEY, real prices (--real-trade)")
    else:
        print("  PAPER TRADING fp/full.py  --  virtual money, real prices")
    print(f"  equity ${a.equity:.2f}   one position per coin, no cap on "
          f"how many coins")
    for q in fitted:
        m = logic.meta[q]
        # live_gate, not gate: gate is measured on the rows the model
        # was fit to reproduce and is ~0 by construction (see
        # fp/full.py's oof_gate docstring); live_gate is the number
        # this process actually compares confidence against. A model
        # file with no live_gate at all is shown -- and traded -- as
        # gate 1.0 (unproven), matching fp/live_full.py's fallback.
        lg = m.get("live_gate", 1.0)
        print(f"    fitted {q:<12} lev<= {m['lev_cap']:.0f}x  "
              f"cost {100*m['cost']:.4f}%  live_gate {lg:.4f}  "
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
        client = make_client(testnet=a.testnet)
        feed = LiveFeed(client, workers=a.workers)
        if a.symbols:
            symbols = [q.strip().upper() for q in a.symbols.split(",")
                       if q.strip()]
            U.all_perpetuals(client)          # cache real leverage caps
        elif a.all_coins:
            # THE WHOLE BOARD. Every USDT perpetual Bybit lists, ranked
            # by 24h turnover, with the real per-coin leverage ceiling
            # cached for the sizing code to read. Off by default -- see
            # below.
            symbols, caps, tvr = U.ranked(client)
            print(f"  universe: {len(symbols)} USDT perpetuals from Bybit")
            print("  WARNING: borrowed logic. On the first live run of "
                  "the whole board")
            print("  all twelve losses came from a logic built on a "
                  "different coin,")
            print("  and no fitted coin closed red. fp/transfer.py "
                  "measures the same")
            print("  thing: 0/10 coins clear their own break-even at "
                  "p<0.05.")
        else:
            # THE DEFAULT IS THE COINS THE LOGIC WAS BUILT ON. Every
            # measurement that says this works -- 100% recovery, 100%
            # win, never liquidated -- was made on these nine and only
            # these nine. Trading elsewhere is a separate, unproven
            # claim, so it is opt-in rather than the default.
            symbols = fitted
            U.all_perpetuals(client)      # real leverage caps per coin
        rotation = U.Rotation(symbols, top=a.top, slice_size=a.sweep)
        print(f"  scanning: top {len(rotation.top)} every cycle, "
              f"{len(rotation.tail)} more in slices of {a.sweep} "
              f"(full sweep every {rotation.cycles_for_full_sweep} cycles)")
        borrowed = [q for q in symbols if q not in set(fitted)]
        print(f"  logic:    {len(fitted)} fitted models"
              + (f"; {len(borrowed)} coins scored by borrowed logic"
                 if borrowed else " -- every coin traded by its own"))
        if a.retrain_continuous:
            print(f"  retrain:  continuous -- next cycle starts the "
                  f"instant the last one finishes, models swapped in "
                  f"live -- see fp/retrain.py")
        elif a.retrain_hours > 0:
            print(f"  retrain:  every {a.retrain_hours:.1f}h, models "
                  f"swapped in live -- see fp/retrain.py")
    broker = None
    if a.real_trade:
        from fp.broker import LiveBroker, OrderError
        api_key, api_secret = real_credentials()
        broker = LiveBroker(api_key, api_secret, testnet=a.testnet)
        try:
            real_equity = broker.wallet_equity()
        except OrderError as exc:
            print(f"  Could not read wallet balance -- check the API key "
                  f"and its permissions: {exc}", file=sys.stderr)
            return 2
        mode_label = "(testnet)" if a.testnet else "(MAINNET -- real money)"
        print(f"  REAL TRADING {mode_label}   "
              f"wallet equity ${real_equity:.2f}")
        # A REFRESHED instrument cache, not the stale one a prior paper
        # run may have left behind -- real order quantities are rounded
        # against qty_step/min_qty read from here (fp.costs.round_qty),
        # and a leverage cap read wrong is a leverage cap ignored.
        instruments = C.refresh(client, symbols)
        # Say so BEFORE the first order, not after N identical failures.
        # A symbol Bybit's linear list doesn't return usable
        # lotSizeFilter for (seen on SNDKUSDT: an exotic instrument,
        # not a typical crypto perp) makes round_qty() refuse every
        # real order on it -- which is correct, but silent about why
        # unless said here.
        no_step = [s for s in symbols
                  if not instruments.get(s, {}).get("qty_step")]
        if no_step:
            print(f"  no usable qty_step from Bybit for: "
                  f"{', '.join(sorted(no_step))} -- real orders on "
                  f"{'these' if len(no_step) > 1 else 'this'} will be "
                  f"skipped, not attempted", flush=True)
        # RECONCILE, DO NOT REFUSE. A prior run of THIS bot may have
        # been Ctrl+C'd or crashed with positions still open -- those
        # have a persisted Decision (save_state() below) and are
        # ADOPTED, resuming with their original target/stop/trail
        # exactly as if the process never stopped. A position with no
        # persisted record is one this process cannot safely manage
        # (see can_open()) and is left alone, not traded around.
        already_open = broker.open_positions()
        saved = load_state()
        adopted, unmanaged = {}, set()
        for sym in already_open:
            if sym in saved:
                try:
                    adopted[sym] = position_from_state(sym, saved[sym])
                except Exception as exc:
                    print(f"  could not restore saved state for {sym}, "
                          f"treating as unmanaged: {exc}", flush=True)
                    unmanaged.add(sym)
            else:
                unmanaged.add(sym)
        if adopted:
            print(f"  RESUMING {len(adopted)} position(s) from a prior "
                  f"run: {', '.join(sorted(adopted))}")
        if unmanaged:
            print(f"  UNMANAGED (real position exists, no saved history "
                  f"-- left alone, not traded around): "
                  f"{', '.join(sorted(unmanaged))}")
        for s in symbols:
            try:
                broker.set_leverage(s, C.max_leverage(s))
            except OrderError as exc:
                print(f"  could not set leverage for {s}: {exc}",
                      flush=True)
        print(f"  equity used for sizing: ${real_equity:.2f} (read from "
              f"the wallet just now, not --equity)")
        a.equity = real_equity
    print("=" * 78, flush=True)
    acct = Account(equity=a.equity, start=a.equity, broker=broker)
    if broker is not None:
        acct.open.update(adopted)
        acct.unmanaged = unmanaged
    panel: dict = {}
    started = time.time()
    stop = {"flag": False}
    retrain_stop = threading.Event()
    retrain_thread = None
    # Only against live Bybit, never in replay: a scheduled cycle would
    # try to fetch fresh candles into a run that is deliberately walking
    # OLD cached bars, and a model swap mid-replay would silently change
    # which logic scored which minute.
    if not a.replay and (a.retrain_hours > 0 or a.retrain_continuous):
        # Retrain always targets the coins already SEEDED with history
        # in the cache (None -> fp.full processes whatever is there),
        # never the live scan's symbol list -- in --all-coins mode that
        # list is ~700 tickers, most with no cached bars at all, and
        # refresh_cache would try to backfill each one from scratch.
        # Retraining keeps the fitted logic current; it does not onboard
        # a new coin, which needs its own seeded history first.
        retrain_thread = threading.Thread(
            target=retrain_worker, args=(holder, None, a.retrain_hours,
                                         retrain_stop, a.retrain_continuous),
            daemon=True)
        retrain_thread.start()

    def onint(*_):
        stop["flag"] = True
    signal.signal(signal.SIGINT, onint)

    try:
        while not stop["flag"]:
            t0 = time.time()
            try:
                batch = rotation.next_batch() if rotation else symbols
                scan(feed, holder.logic, acct, batch, panel,
                    reference=fitted)
            except Exception as exc:                      # keep trading
                print(f"  scan error: {exc}", flush=True)
            if a.once:
                dashboard(acct, panel, started, holder)
                break
            if not feed.advance():
                break
            if feed.live:
                dashboard(acct, panel, started, holder)
                while time.time() - t0 < a.interval and not stop["flag"]:
                    time.sleep(0.25)
            elif len(acct.closed) and len(acct.closed) % 25 == 0:
                dashboard(acct, panel, started, holder)
    finally:
        retrain_stop.set()
        summary(acct, panel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
