"""Paper-trading broker for FINAL_Logic_PotentialScaledLeverage.

Virtual money, real Bybit prices from the public kline and ticker
endpoints. No API key, no account, no order ever submitted.

Twelve methods vote on each 30-minute bar; a position opens where they
fire and agree on direction. Leverage runs the file's full potential
chain -- base from ATR, a potential score from the volatility percentile,
multiplier = score + 0.5, product capped at the base -- and is then cut
by how strongly the methods agreed on THAT setup, which is the only term
in the chain belonging to the trade rather than to the instrument. The
exit is fixed at entry.

Three things run at once, on their own threads, so none waits on another:

    scanner    keeps a standing signal for every symbol on the board and
               opens any of them the moment margin allows
    manager    re-prices every open position every second off one
               whole-board ticker call, so TP/SL fire promptly
    dashboard  prints capital, P&L, margin and scan statistics on a
               fixed beat while the other two work

HOW "SCAN CONTINUOUSLY, MISS NOTHING" IS ACTUALLY ACHIEVED

Naively that means refetching every symbol's klines in a tight loop. It
does not work: the methods read a 30-minute bar, so the verdict cannot
change until that bar closes, and a tight loop just asks Bybit the same
question hundreds of times for the same answer -- measured at 125 kline
calls a second on a 40-symbol board, which on the real ~690-symbol board
is a rate-limit ban, and a banned bot misses everything.

So klines are fetched once per symbol per 30-minute bar, and the verdict
is cached as a standing signal. The fast loop then runs continuously over
those standing signals and opens each one as soon as there is margin for
it. A signal raised while the book was full is not lost -- it stays
standing until its bar rolls over, and the next freed slot fills it. That
is what makes nothing get missed; polling harder would not.
"""
from __future__ import annotations

import concurrent.futures
import logging
import signal
import threading
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
BAR_MS = L.BAR_MINUTES * 60 * 1000
# The slowest feature needs about 200 bars; below this a symbol is skipped
# rather than scored off half-formed indicators.
MIN_BARS = 250
# Scanning the whole board means one kline call per symbol. Done serially
# that is ~200ms x 690 symbols, longer than the pass itself. Bybit's public
# endpoints allow far more in parallel, so fetches are threaded.
FETCH_WORKERS = 12
# Signals are refreshed in chunks rather than one board-wide burst, so
# entries keep flowing while the refresh runs instead of stalling for the
# ~20s a full re-evaluation takes.
REFRESH_CHUNK = 48
# How long the fast loop pauses between attempts to fill standing signals.
# It costs no API calls, so this only needs to be short enough that a freed
# margin slot is reused promptly.
FILL_INTERVAL_SECONDS = 0.5
# How often open positions are re-priced. One ticker call covers the whole
# board, so this costs one request per second no matter how many are open.
MANAGE_INTERVAL_SECONDS = 1.0
# How often the financial dashboard reprints.
DASHBOARD_INTERVAL_SECONDS = 5.0
# A whole-board ticker snapshot older than this is not trusted for pricing.
PRICE_STALE_SECONDS = 10.0


def closed_bar_ts(now_ms: int | None = None) -> int:
    """Start of the most recently CLOSED 30m bar, in epoch ms.

    Signals are computed on closed bars only, so this is what a cached
    verdict is keyed on and what makes it stale when the bar rolls.
    """
    now = int(time.time() * 1000) if now_ms is None else now_ms
    return (now // BAR_MS) * BAR_MS - BAR_MS


@dataclass
class Signal:
    """A standing verdict for one symbol, valid until its bar rolls over."""
    bar_ts: int
    direction: int
    atr_pct: float
    votes: int
    vote_margin: int
    methods: str


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
    vote_margin: int
    best_price: float
    trail_dist: float
    potential_score: float
    conviction: float
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
                 margin_pct: float = 0.05, max_notional_x: float = 10.0,
                 conviction_floor: float = L.CONVICTION_FLOOR,
                 expectancy_gate: bool = True,
                 assumed_win_rate: float | None = None):
        self.client = client
        self.symbols = symbols
        self.equity = equity
        self.starting_equity = equity
        self.peak_equity = equity
        self.max_drawdown = 0.0
        # 0 means "no cap on the number of positions" -- what actually
        # limits it then is free margin, which is the real constraint.
        self.max_positions = max_positions
        self.margin_pct = margin_pct
        # Ceiling on total notional as a multiple of equity. Per-trade
        # limits do not bound this: nine positions each risking a modest
        # slice still put ~25x the account into the market, and crypto
        # moves together, so they are one bet, not nine. 0 disables it.
        self.max_notional_x = max_notional_x
        # Smallest fraction of the file's leverage a minimum-conviction
        # setup may take. 1.0 disables the per-trade haircut entirely.
        self.conviction_floor = conviction_floor
        # Refuse trades whose expected value after fees is negative. What
        # has to clear the fee is the edge, not the move -- see
        # logic.expectancy(). At taker fees this refuses nearly everything,
        # which is the correct answer, not a malfunction.
        self.expectancy_gate = expectancy_gate
        self.assumed_win_rate = assumed_win_rate
        self.exit_name = exit_name
        self.min_votes = min_votes
        self.fee = fee
        self.max_leverage = max_leverage

        self.open: dict[str, Position] = {}
        self.closed: list[Closed] = []
        # Standing signals, one per symbol, replaced when its bar rolls.
        self.signals: dict[str, Signal] = {}
        # Bar each symbol was last scored on, whether or not it produced a
        # signal. Staleness keys off this rather than off self.signals --
        # otherwise every symbol the methods pass over looks unevaluated and
        # gets refetched on every single pass, forever.
        self.evaluated_bar: dict[str, int] = {}
        # Bar a symbol was last traded on, so a stopped-out position is not
        # instantly reopened by the same standing signal.
        self.traded_bar: dict[str, int] = {}

        self.evaluations = 0
        self.no_signal = 0
        self.skipped_no_margin = 0
        self.skipped_max_notional = 0
        self.skipped_unsolvent = 0
        self.skipped_negative_ev = 0
        self.blocked_signals = 0
        self.blocked_by: dict[str, int] = {"margin": 0, "exposure": 0,
                                           "unsolvent": 0}
        self.fees_paid = 0.0
        self.passes = 0
        self.refreshes = 0
        self.last_refresh_seconds = 0.0
        self.kline_calls = 0
        self.started_at = time.time()

        # Whole-board ticker snapshot, refreshed by the manager thread and
        # read by everything else, so no thread makes its own price call.
        self.prices: dict[str, float] = {}
        self.prices_at = 0.0
        # Scanning, management and reporting run on separate threads and
        # all touch equity and the open book, so both are lock-guarded.
        self.lock = threading.RLock()
        # kline_calls is bumped from the fetch pool, so it gets its own
        # lock -- taking the main one there would serialise the fetches.
        self.counter_lock = threading.Lock()

    # ---------------------------------------------------------------- state

    @property
    def committed_margin(self) -> float:
        with self.lock:
            return sum(p.margin for p in self.open.values())

    @property
    def notional(self) -> float:
        with self.lock:
            return sum(p.margin * p.leverage for p in self.open.values())

    @property
    def equity_total(self) -> float:
        """Cash plus open P&L -- what the account is really worth now.

        Sizing and free margin both key off this rather than off cash. A
        real cross-margin account works this way, and the difference is
        not cosmetic: with cash-only accounting the bot kept opening at
        full size while its open book was 25% underwater, because losses
        it had not yet realised were invisible to it.
        """
        with self.lock:
            total = self.equity
            for p in self.open.values():
                price = self.prices.get(p.symbol) or p.entry
                total += (price - p.entry) / p.entry * p.direction * p.qty * p.entry
            return total

    @property
    def free_margin(self) -> float:
        with self.lock:
            return self.equity_total - sum(p.margin for p in self.open.values())

    def slice_size(self) -> float:
        """Margin one trade commits: a fixed slice of current equity."""
        return max(0.0, self.equity_total) * self.margin_pct

    def notional_headroom(self) -> float:
        """How much more notional the exposure ceiling still allows."""
        if not self.max_notional_x:
            return float("inf")
        return max(0.0, self.equity_total * self.max_notional_x - self.notional)

    def mark(self) -> float:
        """Mark to market: update the peak and the drawdown off total
        equity, not cash. A drawdown that happens while positions are
        still open is a real drawdown, and cash-only accounting missed
        every one of them."""
        with self.lock:
            total = self.equity_total
            self.peak_equity = max(self.peak_equity, total)
            self.max_drawdown = max(self.max_drawdown, self.peak_equity - total)
            return total

    def has_room(self) -> bool:
        with self.lock:
            if self.max_positions and len(self.open) >= self.max_positions:
                return False
            if self.notional_headroom() <= 0:
                return False
            return self.free_margin >= self.slice_size() > 0

    # ---------------------------------------------------------------- market

    def klines(self, symbol: str) -> pd.DataFrame | None:
        try:
            rows = self.client.get_kline(category="linear", symbol=symbol,
                                         interval="30",
                                         limit=KLINES_30M)["result"]["list"]
        except Exception:
            logger.debug("kline fetch failed for %s", symbol, exc_info=True)
            return None
        with self.counter_lock:
            self.kline_calls += 1
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low",
                                         "close", "volume", "turnover"])
        df["ts"] = df["ts"].astype("int64")
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms")
        for c in ("open", "high", "low", "close", "volume"):
            df[c] = df[c].astype(float)
        return df.sort_values("datetime").reset_index(drop=True)

    def refresh_prices(self) -> int:
        """One call gives the last price of every linear perpetual. Managing
        200 positions then costs the same one request as managing one."""
        try:
            rows = self.client.get_tickers(category="linear")["result"]["list"]
        except Exception:
            logger.debug("whole-board ticker fetch failed", exc_info=True)
            return 0
        snap = {}
        for r in rows:
            try:
                p = float(r["lastPrice"])
            except (TypeError, ValueError, KeyError):
                continue
            if p > 0:
                snap[r["symbol"]] = p
        if snap:
            self.prices = snap
            self.prices_at = time.time()
        return len(snap)

    def last_price(self, symbol: str) -> float | None:
        """The board snapshot if it is fresh, otherwise a direct call."""
        if time.time() - self.prices_at <= PRICE_STALE_SECONDS:
            p = self.prices.get(symbol)
            if p:
                return p
        try:
            r = self.client.get_tickers(category="linear", symbol=symbol)
            return float(r["result"]["list"][0]["lastPrice"])
        except Exception:
            logger.debug("ticker fetch failed for %s", symbol, exc_info=True)
            return None

    def prefetch(self, symbols: list[str]) -> dict:
        """Fetch klines for many symbols at once so a refresh is bounded by
        the slowest request rather than the sum of all of them."""
        out: dict = {}
        if not symbols:
            return out
        with concurrent.futures.ThreadPoolExecutor(FETCH_WORKERS) as pool:
            futures = {pool.submit(self.klines, s): s for s in symbols}
            for fut in concurrent.futures.as_completed(futures):
                sym = futures[fut]
                try:
                    out[sym] = fut.result()
                except Exception:
                    logger.debug("prefetch failed for %s", sym, exc_info=True)
                    out[sym] = None
        return out

    # ------------------------------------------------------------ evaluation

    def evaluate(self, symbol: str, bars: pd.DataFrame | None) -> Signal | None:
        """Score one symbol on its last closed bar and store the verdict.

        Returns the standing signal, or None if the methods did not fire,
        did not agree, or the data was unusable.
        """
        self.evaluations += 1
        now_bar = closed_bar_ts()
        if bars is None or len(bars) < MIN_BARS:
            # Mark it evaluated anyway: a symbol Bybit cannot serve, or one
            # too young to have 250 bars, must not be retried every pass.
            self.evaluated_bar[symbol] = now_bar
            self.signals.pop(symbol, None)
            return None

        # The newest row is the bar still forming; the verdict is taken on
        # the last bar that actually closed.
        bars = bars.iloc[:-1].reset_index(drop=True)
        bar_ts = int(bars["ts"].iloc[-1])
        self.evaluated_bar[symbol] = max(bar_ts, now_bar)

        try:
            feats = F.build(bars)
            v = M.evaluate_all(feats).iloc[-1]
        except Exception:
            logger.debug("evaluation failed for %s", symbol, exc_info=True)
            self.signals.pop(symbol, None)
            return None

        atr_pct = float(feats["atr14_pct"].iloc[-1])
        if (v["n_methods_fired"] < self.min_votes
                or v["consensus_dir"] == M.TIE
                or not np.isfinite(atr_pct) or atr_pct <= 0):
            self.no_signal += 1
            self.signals.pop(symbol, None)
            return None

        sig = Signal(
            bar_ts=bar_ts,
            direction=1 if v["consensus_dir"] == M.LONG else -1,
            atr_pct=atr_pct,
            votes=int(v["n_methods_fired"]),
            vote_margin=int(v["vote_margin"]),
            methods=", ".join(m.split("_", 1)[1]
                              for m in M.METHOD_NAMES if v[m] != "-"),
        )
        self.signals[symbol] = sig
        return sig

    def refresh_signals(self, symbols: list[str]) -> int:
        """Refetch and re-score a chunk of symbols. Returns signals standing."""
        fetched = self.prefetch(symbols)
        n = 0
        for sym in symbols:
            if self.evaluate(sym, fetched.get(sym)) is not None:
                n += 1
        return n

    def stale_symbols(self) -> list[str]:
        """Symbols not yet scored on the most recently closed bar.

        Symbols already holding a position are left out: their entry is
        decided, and re-scoring them would only spend kline calls.
        """
        want = closed_bar_ts()
        return [s for s in self.symbols
                if s not in self.open and self.evaluated_bar.get(s, -1) < want]

    # ----------------------------------------------------------------- entry

    def try_open(self, symbol: str) -> bool:
        """Open the symbol's standing signal if there is margin for it."""
        sig = self.signals.get(symbol)
        if sig is None or symbol in self.open:
            return False
        # One entry per symbol per bar: without this a stop-out would be
        # reopened immediately by the same standing verdict.
        if self.traded_bar.get(symbol) == sig.bar_ts:
            return False

        price = self.last_price(symbol)
        if price is None or price <= 0:
            return False

        # Expectancy first: it costs nothing and it is the only test that
        # can tell this trade is not worth taking at all.
        ev = L.expectancy(sig.atr_pct, self.exit_name, self.fee,
                          self.assumed_win_rate)
        if self.expectancy_gate and ev <= 0:
            self.skipped_negative_ev += 1
            return False

        d = sig.direction
        atr = sig.atr_pct / 100.0 * price
        chain = L.leverage_potential(sig.atr_pct, self.max_leverage,
                                     votes=sig.votes,
                                     vote_margin=sig.vote_margin,
                                     conviction_floor=self.conviction_floor)
        if not chain["tradeable"]:
            # Even 1x cannot keep the stop inside the liquidation price.
            self.skipped_unsolvent += 1
            return False
        lev = chain["leverage"]
        tp_mult = L.TP_MULTIPLES.get(self.exit_name)

        # A fixed slice of current equity per trade, and only if that whole
        # slice is free -- with no position cap, free margin is what stops
        # the bot opening more than the account can carry. Sizing, the fee
        # debit and the book entry happen under one lock so two passes
        # cannot spend the same margin twice.
        with self.lock:
            if symbol in self.open:
                return False
            if self.max_positions and len(self.open) >= self.max_positions:
                self.skipped_no_margin += 1
                return False
            margin = self.slice_size()
            if margin <= 0 or self.free_margin < margin:
                self.skipped_no_margin += 1
                return False
            notional = margin * lev
            # The exposure ceiling is the only thing bounding correlated
            # risk: nine uncorrelated-looking positions in crypto are one
            # bet, and per-trade sizing cannot see that.
            if notional > self.notional_headroom():
                self.skipped_max_notional += 1
                return False
            entry_fee = notional * self.fee / 2
            self.equity -= entry_fee
            self.fees_paid += entry_fee
            pos = Position(
                symbol=symbol, direction=d, entry=price, qty=notional / price,
                leverage=lev, margin=margin, opened_at=pd.Timestamp.now(tz="UTC"),
                tp_price=(price + d * tp_mult * atr) if tp_mult else float("nan"),
                sl_price=price - d * L.SL_MULTIPLE * atr,
                liq_price=price * (1 - d * 0.9 / lev),
                exit_name=self.exit_name, methods=sig.methods,
                votes=sig.votes, vote_margin=sig.vote_margin,
                best_price=price, trail_dist=L.TRAIL_MULTIPLE * atr,
                potential_score=chain["potential_score"],
                conviction=chain["conviction"],
                lev_base=chain["lev_base"],
            )
            self.open[symbol] = pos
            self.traded_bar[symbol] = sig.bar_ts

        logger.info("%s OPEN %s @%.6f atr_rank=%.2f conviction=%.2f "
                    "base=%.0fx x%.2f -> lev=%.0fx votes=%d(margin %d) [%s] "
                    "exit=%s tp=%.6f sl=%.6f margin=$%.3f",
                    symbol, "LONG" if d > 0 else "SHORT", price,
                    chain["potential_score"], chain["conviction"],
                    chain["lev_base"], chain["conviction_haircut"], lev,
                    sig.votes, sig.vote_margin, sig.methods, self.exit_name,
                    pos.tp_price, pos.sl_price, margin)
        return True

    def fill_standing(self) -> tuple[int, int]:
        """Try to open every standing signal. Returns (opened, still waiting).

        This is the pass that runs continuously. It makes no kline calls, so
        it can run as often as we like, and it is what guarantees a signal
        raised while the book was full is taken the moment a slot frees.
        """
        opened, waiting = 0, 0
        # Per-pass reason gauges. The cumulative counters keep rising every
        # pass a signal stays blocked, which says nothing useful on a live
        # screen -- what matters is why the CURRENT waiting set is waiting.
        before = (self.skipped_no_margin, self.skipped_max_notional,
                  self.skipped_unsolvent)
        for sym in list(self.signals):
            if sym in self.open:
                continue
            sig = self.signals.get(sym)
            if sig is None or self.traded_bar.get(sym) == sig.bar_ts:
                continue
            if not self.has_room():
                waiting += 1
                continue
            try:
                if self.try_open(sym):
                    opened += 1
                else:
                    waiting += 1
            except Exception:
                logger.debug("entry failed for %s", sym, exc_info=True)
        self.blocked_signals = waiting
        self.blocked_by = {
            "margin": self.skipped_no_margin - before[0],
            "exposure": self.skipped_max_notional - before[1],
            "unsolvent": self.skipped_unsolvent - before[2],
        }
        return opened, waiting

    # ------------------------------------------------------------------ exit

    def manage(self, symbol: str, price: float | None = None) -> None:
        pos = self.open.get(symbol)
        if pos is None:
            return
        if price is None:
            price = self.last_price(symbol)
        if price is None or price <= 0:
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
                return
        elif np.isfinite(pos.tp_price):
            if (price >= pos.tp_price) if d > 0 else (price <= pos.tp_price):
                self._close(symbol, pos.tp_price, "take_profit")
                return

        # The same timeout simulate_exit applies, so a position cannot tie
        # capital up indefinitely if neither side is ever touched.
        held = (pd.Timestamp.now(tz="UTC") - pos.opened_at).total_seconds()
        if held >= L.MAX_HOLD_BARS * L.BAR_MINUTES * 60:
            self._close(symbol, price, "timeout")

    def manage_all(self) -> None:
        """Re-price the whole open book off one ticker snapshot."""
        self.refresh_prices()
        with self.lock:
            book = list(self.open)
        for sym in book:
            try:
                self.manage(sym, self.prices.get(sym))
            except Exception:
                logger.debug("manage failed for %s", sym, exc_info=True)
        self.mark()

    def _close(self, symbol: str, price: float, reason: str) -> None:
        with self.lock:
            pos = self.open.pop(symbol, None)
            if pos is None:
                return
            move = (price - pos.entry) / pos.entry * pos.direction
            if reason == "liquidated":
                # The margin is gone; the exit fee comes out of it, not on top.
                pnl = -pos.margin
            else:
                exit_fee = pos.qty * price * self.fee / 2
                pnl = move * pos.qty * pos.entry - exit_fee
                self.fees_paid += exit_fee
            self.equity += pnl
            self.closed.append(Closed(
                symbol=symbol, direction=pos.direction, entry=pos.entry,
                exit_price=price, opened_at=pos.opened_at,
                closed_at=pd.Timestamp.now(tz="UTC"), reason=reason, pnl_usd=pnl,
                return_pct_leveraged=100 * move * pos.leverage,
                methods=pos.methods))
        self.mark()
        logger.info("%s CLOSE %s @%.6f (%s) pnl=$%.4f equity=$%.4f",
                    symbol, "LONG" if pos.direction > 0 else "SHORT",
                    price, reason, pnl, self.equity)

    # ------------------------------------------------------------- reporting

    def open_rows(self) -> list[tuple]:
        with self.lock:
            book = list(self.open.items())
        rows = []
        for sym, pos in book:
            price = self.prices.get(sym) or pos.entry
            pnl = (price - pos.entry) / pos.entry * pos.direction * pos.qty * pos.entry
            rows.append((sym, pos.direction, pos.entry, price, pnl,
                         pos.leverage, pos.methods, pos.potential_score))
        return rows

    def snapshot(self) -> dict:
        """Everything the live dashboard shows, taken at one instant."""
        with self.lock:
            closed = list(self.closed)
            n_open = len(self.open)
            equity = self.equity
            committed = sum(p.margin for p in self.open.values())
            notional = sum(p.margin * p.leverage for p in self.open.values())
            avg_lev = (sum(p.leverage for p in self.open.values()) / n_open
                       if n_open else 0.0)
        rows = self.open_rows()
        unreal = sum(r[4] for r in rows)
        ow = sum(1 for r in rows if r[4] > 0)
        ol = len(rows) - ow
        wins = [t for t in closed if t.pnl_usd > 0]
        losses = [t for t in closed if t.pnl_usd <= 0]
        gp = sum(t.pnl_usd for t in wins)
        gl = sum(t.pnl_usd for t in losses)
        eq_open = equity + unreal
        standing = len(self.signals)
        return {
            "elapsed": time.time() - self.started_at,
            "starting_equity": self.starting_equity,
            "equity": equity, "equity_incl_open": eq_open,
            "net": eq_open - self.starting_equity,
            "roi_pct": (100 * (eq_open - self.starting_equity) / self.starting_equity
                        if self.starting_equity else 0.0),
            "realized": gp + gl, "gross_profit": gp, "gross_loss": gl,
            "unrealized": unreal, "fees_paid": self.fees_paid,
            "committed_margin": committed, "free_margin": equity - committed,
            "margin_used_pct": (100 * committed / eq_open) if eq_open > 0 else 0.0,
            "slice_size": self.slice_size(),
            "notional": notional, "avg_leverage": avg_lev,
            "exposure_x": (notional / eq_open) if eq_open > 0 else 0.0,
            "max_notional_x": self.max_notional_x,
            "notional_headroom": self.notional_headroom(),
            "max_drawdown": self.max_drawdown, "peak_equity": self.peak_equity,
            "open": n_open, "open_wins": ow, "open_losses": ol,
            "open_profit": sum(r[4] for r in rows if r[4] > 0),
            "open_loss": sum(r[4] for r in rows if r[4] <= 0),
            "closed": len(closed), "closed_wins": len(wins),
            "closed_losses": len(losses),
            "targets": sum(1 for t in closed if t.reason == "take_profit"),
            "stops": sum(1 for t in closed if t.reason == "stop_loss"),
            "liquidations": sum(1 for t in closed if t.reason == "liquidated"),
            "win_rate": (100 * len(wins) / len(closed)) if closed else 0.0,
            "avg_win": (gp / len(wins)) if wins else 0.0,
            "avg_loss": (gl / len(losses)) if losses else 0.0,
            "profit_factor": (gp / abs(gl)) if gl else (float("inf") if gp else 0.0),
            "passes": self.passes, "refreshes": self.refreshes,
            "last_refresh_seconds": self.last_refresh_seconds,
            "evaluations": self.evaluations, "no_signal": self.no_signal,
            "skipped_no_margin": self.skipped_no_margin,
            "skipped_max_notional": self.skipped_max_notional,
            "skipped_unsolvent": self.skipped_unsolvent,
            "skipped_negative_ev": self.skipped_negative_ev,
            "min_atr_for_edge": L.min_atr_for_edge(self.exit_name, self.fee,
                                                   self.assumed_win_rate),
            "blocked_by": dict(self.blocked_by),
            "standing_signals": standing, "blocked_signals": self.blocked_signals,
            "stale": len(self.stale_symbols()), "kline_calls": self.kline_calls,
            "universe": len(self.symbols),
        }

    def summary(self) -> dict:
        self.refresh_prices()
        rows = self.open_rows()
        wins = [t for t in self.closed if t.pnl_usd > 0]
        losses = [t for t in self.closed if t.pnl_usd <= 0]
        ow = sum(1 for r in rows if r[4] > 0)
        ol = len(rows) - ow
        unreal = sum(r[4] for r in rows)
        gp, gl = sum(t.pnl_usd for t in wins), sum(t.pnl_usd for t in losses)
        op = sum(r[4] for r in rows if r[4] > 0)
        olo = sum(r[4] for r in rows if r[4] <= 0)
        eq_total = self.equity + unreal
        exposure = (self.notional / eq_total) if eq_total > 0 else 0.0
        return {
            "closed": len(self.closed), "closed_wins": len(wins),
            "closed_losses": len(losses),
            "liquidations": sum(1 for t in self.closed if t.reason == "liquidated"),
            "stops": sum(1 for t in self.closed if t.reason == "stop_loss"),
            "targets": sum(1 for t in self.closed if t.reason == "take_profit"),
            "timeouts": sum(1 for t in self.closed if t.reason == "timeout"),
            "gross_profit": gp, "gross_loss": gl, "realized": gp + gl,
            "open": len(rows), "open_wins": ow, "open_losses": ol,
            "open_profit": op, "open_loss": olo, "unrealized": unreal,
            "open_rows": rows,
            "total_trades": len(self.closed) + len(rows),
            "total_wins": len(wins) + ow, "total_losses": len(losses) + ol,
            "total_profit": gp + op, "total_loss": gl + olo,
            "win_rate": (100 * len(wins) / len(self.closed)) if self.closed else 0.0,
            "starting_equity": self.starting_equity, "equity": self.equity,
            "equity_incl_open": self.equity + unreal,
            "fees_paid": self.fees_paid, "max_drawdown": self.max_drawdown,
            "peak_equity": self.peak_equity,
            "evaluations": self.evaluations, "no_signal": self.no_signal,
            "skipped_no_margin": self.skipped_no_margin,
            "skipped_max_notional": self.skipped_max_notional,
            "skipped_unsolvent": self.skipped_unsolvent,
            "skipped_negative_ev": self.skipped_negative_ev,
            "notional": self.notional, "exposure_x": exposure,
            "standing_signals": len(self.signals),
            "blocked_signals": self.blocked_signals,
            "passes": self.passes, "refreshes": self.refreshes,
            "kline_calls": self.kline_calls,
            "elapsed": time.time() - self.started_at,
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
        for sym, d, entry, price, pnl, lev, meth, pot in self.open_rows():
            pos = self.open.get(sym)
            if pos is None:
                continue
            rows.append({"symbol": sym, "side": "LONG" if d > 0 else "SHORT",
                         "status": "open_at_shutdown", "entry": entry,
                         "exit_or_last": price, "opened_at": pos.opened_at,
                         "closed_at": pd.Timestamp.now(tz="UTC"),
                         "reason": "still_open", "pnl_usd": pnl,
                         "return_pct_leveraged": float("nan"), "methods": meth})
        if rows:
            pd.DataFrame(rows).to_csv(path, index=False)
            logger.info("wrote %d trade rows to %s", len(rows), path)


def _hms(seconds: float) -> str:
    s = int(max(0, seconds))
    return f"{s // 3600:d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def print_dashboard(s: dict) -> None:
    """The live financial readout, reprinted on every beat while running."""
    pf = "inf" if s["profit_factor"] == float("inf") else f"{s['profit_factor']:.2f}"
    print()
    print("=" * 78)
    print(f"  LIVE  up {_hms(s['elapsed'])}   "
          f"equity ${s['equity_incl_open']:.4f}   "
          f"net ${s['net']:+.4f}  ({s['roi_pct']:+.2f}%)")
    print("-" * 78)
    print(f"  CAPITAL   start ${s['starting_equity']:.4f}"
          f"   cash ${s['equity']:.4f}"
          f"   equity+open ${s['equity_incl_open']:.4f}"
          f"   peak ${s['peak_equity']:.4f}")
    print(f"  MARGIN    committed ${s['committed_margin']:.4f}"
          f"   free ${s['free_margin']:.4f}"
          f"   used {s['margin_used_pct']:.1f}%"
          f"   per trade ${s['slice_size']:.4f}")
    ceiling = (f"   ceiling {s['max_notional_x']:.0f}x"
               f" (${s['notional_headroom']:.2f} left)"
               if s["max_notional_x"] else "   ceiling off")
    print(f"  EXPOSURE  notional ${s['notional']:.2f}"
          f"   {s['exposure_x']:.1f}x equity"
          f"   avg leverage {s['avg_leverage']:.0f}x{ceiling}")
    print(f"  P&L       realized ${s['realized']:+.4f}"
          f"   unrealized ${s['unrealized']:+.4f}"
          f"   fees ${s['fees_paid']:.4f}"
          f"   max DD ${s['max_drawdown']:.4f}")
    print(f"  CLOSED    {s['closed']}"
          f"   win {s['closed_wins']} / loss {s['closed_losses']}"
          f"   rate {s['win_rate']:.1f}%"
          f"   TP {s['targets']} / SL {s['stops']} / liq {s['liquidations']}"
          f"   PF {pf}")
    print(f"            avg win ${s['avg_win']:+.4f}"
          f"   avg loss ${s['avg_loss']:+.4f}")
    print(f"  OPEN      {s['open']} positions"
          f"   in profit {s['open_wins']} (${s['open_profit']:+.4f})"
          f" / in loss {s['open_losses']} (${s['open_loss']:+.4f})")
    print(f"  SIGNALS   standing {s['standing_signals']}"
          f"   waiting {s['blocked_signals']}"
          f"   due a refresh {s['stale']}"
          f"   no signal {s['no_signal']:,}")
    bb = s["blocked_by"]
    print(f"  BLOCKED   this pass: margin {bb['margin']}"
          f"   exposure ceiling {bb['exposure']}"
          f"   stop past liquidation {bb['unsolvent']}")
    print(f"  EDGE      negative-expectancy skips {s['skipped_negative_ev']:,}"
          f"   (needs atr14 >= {s['min_atr_for_edge']:.3f}% at this fee)")
    print(f"  SCAN      fill pass #{s['passes']:,}"
          f"   bar refreshes {s['refreshes']:,} (last {s['last_refresh_seconds']:.1f}s)"
          f"   universe {s['universe']}"
          f"   klines {s['kline_calls']:,}")
    print("=" * 78)


def print_summary(s: dict) -> None:
    print("\n" + "=" * 70)
    print("PAPER TRADING SESSION SUMMARY")
    print("=" * 70)
    print(f"Starting virtual equity : ${s['starting_equity']:.4f}")
    print(f"Ran for                 : {_hms(s['elapsed'])}")

    print("\n-- CLOSED TRADES " + "-" * 50)
    print(f"  Trades closed         : {s['closed']}")
    print(f"    winning             : {s['closed_wins']}")
    print(f"    losing              : {s['closed_losses']}")
    print(f"    win rate            : {s['win_rate']:.1f}%")
    print(f"    hit take-profit     : {s['targets']}")
    print(f"    stopped out         : {s['stops']}")
    print(f"    liquidated          : {s['liquidations']}")
    print(f"    timed out           : {s['timeouts']}")
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

    print("\n-- CAPITAL " + "-" * 56)
    print(f"  Realized only         : ${s['equity']:.4f}")
    print(f"  Incl. open positions  : ${s['equity_incl_open']:.4f}")
    print(f"  Peak equity           : ${s['peak_equity']:.4f}")
    print(f"  Max drawdown          : ${s['max_drawdown']:.4f}")
    print(f"  Fees paid             : ${s['fees_paid']:.4f}")
    print(f"  Margin committed      : ${s['committed_margin']:.4f}")
    print(f"  Margin free           : ${s['free_margin']:.4f}")
    print(f"  Notional at shutdown  : ${s['notional']:.2f} "
          f"({s['exposure_x']:.1f}x equity)")
    net = s["equity_incl_open"] - s["starting_equity"]
    pct = 100 * net / s["starting_equity"] if s["starting_equity"] else 0.0
    print(f"  Net result            : ${net:+.4f}  ({pct:+.2f}%)")

    print("\n-- SCANNING " + "-" * 55)
    print(f"  Bar refreshes         : {s['refreshes']:,}")
    print(f"  Fill passes           : {s['passes']:,}")
    print(f"  Kline calls           : {s['kline_calls']:,}")
    print(f"  Evaluations           : {s['evaluations']:,}"
          f"  ({s['no_signal']:,} produced no qualifying vote)")
    print(f"  Signals still standing: {s['standing_signals']}"
          f"  ({s['blocked_signals']} of them still waiting)")
    print(f"  Entries blocked by    : margin {s['skipped_no_margin']:,}, "
          f"exposure ceiling {s['skipped_max_notional']:,}, "
          f"stop past liquidation {s['skipped_unsolvent']:,}, "
          f"negative expectancy {s['skipped_negative_ev']:,}")
    print("=" * 70)


def _scanner(broker: Broker, stop_event: threading.Event) -> None:
    """Keep signals current and fill them continuously.

    Two jobs interleaved. Refreshing re-scores the symbols whose 30m bar
    has rolled, a chunk at a time so entries never stall behind a whole
    board pass. Filling opens any standing signal that has margin, and
    costs no API calls, so it runs on every iteration -- a signal raised
    while the book was full is taken the moment a position closes.
    """
    while not stop_event.is_set():
        stale = broker.stale_symbols()
        if stale:
            t0 = time.time()
            chunk = stale[:REFRESH_CHUNK]
            broker.refresh_signals(chunk)
            broker.refreshes += 1
            broker.last_refresh_seconds = time.time() - t0
            if stop_event.is_set():
                return
            broker.fill_standing()
            broker.passes += 1
            logger.info("refreshed %d/%d stale symbols in %.1fs, "
                        "standing %d, open %d, free $%.3f",
                        len(chunk), len(stale), broker.last_refresh_seconds,
                        len(broker.signals), len(broker.open),
                        broker.free_margin)
            continue

        opened, waiting = broker.fill_standing()
        broker.passes += 1
        if opened:
            logger.info("filled %d standing signal(s), %d still waiting, "
                        "open %d, free $%.3f, exposure %.1fx",
                        opened, waiting, len(broker.open), broker.free_margin,
                        broker.notional / max(broker.equity_total, 1e-9))
        stop_event.wait(FILL_INTERVAL_SECONDS)


def _manager(broker: Broker, stop_event: threading.Event) -> None:
    """Re-price every open position once a second, off one ticker call."""
    while not stop_event.is_set():
        try:
            broker.manage_all()
        except Exception:
            logger.exception("position management pass failed")
        stop_event.wait(MANAGE_INTERVAL_SECONDS)


def run(client, symbols: list[str], equity: float, max_positions: int,
        exit_name: str, min_votes: int, poll_seconds: int,
        trades_csv: str | None = None, fee: float = L.FEE_ROUND_TRIP,
        max_leverage: float | None = None, margin_pct: float = 0.05,
        max_notional_x: float = 10.0,
        conviction_floor: float = L.CONVICTION_FLOOR,
        expectancy_gate: bool = True,
        assumed_win_rate: float | None = None) -> None:
    broker = Broker(client, symbols, equity, max_positions, exit_name,
                    min_votes, fee, max_leverage, margin_pct, max_notional_x,
                    conviction_floor, expectancy_gate, assumed_win_rate)
    stop_event = threading.Event()

    def stop(signum, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    # Prices first, so the manager and the dashboard have a board to read
    # before the first refresh finishes.
    broker.refresh_prices()

    threads = [
        threading.Thread(target=_scanner, args=(broker, stop_event),
                         name="scanner", daemon=True),
        threading.Thread(target=_manager, args=(broker, stop_event),
                         name="manager", daemon=True),
    ]
    for t in threads:
        t.start()

    beat = poll_seconds if poll_seconds > 0 else DASHBOARD_INTERVAL_SECONDS
    try:
        while not stop_event.is_set():
            try:
                print_dashboard(broker.snapshot())
            except Exception:
                logger.exception("dashboard failed")
            stop_event.wait(beat)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=10)
        print_summary(broker.summary())
        if trades_csv:
            try:
                broker.export_csv(trades_csv)
            except Exception:
                logger.exception("failed writing %s", trades_csv)
