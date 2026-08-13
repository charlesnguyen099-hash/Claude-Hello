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
import json
import logging
import math
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fp import logic as L


logger = logging.getLogger("potential_leverage.bot")

# 30m bars are fetched directly: Bybit caps a kline call at 1000 candles,
# and 1000 1m bars is only 33 30m bars, far short of the ~200 the slowest
# feature (EMA200) needs.
KLINES_30M = 1000
BAR_MS = L.BAR_MINUTES * 60 * 1000   # methods; patterns override per-broker
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


def closed_bar_ts(now_ms: int | None = None,
                  bar_minutes: int = L.BAR_MINUTES) -> int:
    """Start of the most recently CLOSED bar, in epoch ms.

    Signals are computed on closed bars only, so this is what a cached
    verdict is keyed on and what makes it stale when the bar rolls. The
    bar size is a parameter because the pattern library runs on 15m while
    the voting methods run on 30m -- using the wrong one here would make
    every symbol look permanently stale, or permanently fresh.
    """
    now = int(time.time() * 1000) if now_ms is None else now_ms
    ms = bar_minutes * 60 * 1000
    return (now // ms) * ms - ms


@dataclass
class Signal:
    """A standing verdict for one symbol, valid until its bar rolls over."""
    bar_ts: int
    direction: int
    atr_pct: float
    votes: int
    vote_margin: int
    methods: str
    # Only the pattern library sets this; the voting methods leave it None
    # and their leverage haircut comes from the vote margin instead.
    confidence: float | None = None
    # Win rate this signal is entitled to claim, as a 95% LOWER bound on
    # its own live record. None means it has no record and falls back to
    # the rate the exit achieved across 2025-2026.
    p_win: float | None = None
    live_n: int = 0
    live_wins: int = 0
    # Set only in slow mode: leverage already solved for, with drag.
    slow_leverage: float | None = None
    # Set only in book mode. The rule that fired carries its OWN target and
    # stop, as FRACTIONAL distances already scaled by the volatility at the
    # signal bar. They are distances rather than prices on purpose: the
    # signal is computed on a bar close and the position fills at the live
    # ticker, and pinning absolute prices to the bar close puts the
    # barriers in the wrong place by exactly that gap -- which in a fast
    # market can be larger than the target itself.
    tp_dist: float | None = None
    sl_dist: float | None = None
    max_hold_min: float | None = None
    rule: str = ""
    # What this rule actually earned per trade where it was measured. The
    # dashboard's expected-value column comes from here for a book trade,
    # because the alternative -- L.ev_per_margin -- rebuilds the old ATR
    # leverage ladder internally and would report one rule's prospects
    # using another rule's model.
    rule_mean: float = 0.0
    # Book mode runs one position PER RULE, not one per symbol, so a
    # symbol carrying a six-day daily rule can still take a 15m one. The
    # slot is what the open book, the standing signals and the per-bar
    # entry lock are all keyed on.
    symbol: str = ""
    # Fraction of equity this signal's own edge justifies as margin,
    # solved at signal time from the rule's measured edge and its own
    # barrier geometry. None means "not a book signal, use the old path".
    margin_frac: float | None = None
    kelly_full: float = 0.0        # before any cap, for reporting
    edge_used: float = 0.0         # the shrunk per-trade edge behind it
    score: float = 0.0             # potential out of 100 -- see Broker.potential
    # Opened purely to gather evidence, on no claim at all. Carried to the
    # position so the close can bill it to the experiment's budget.
    probing: bool = False

    @property
    def slot(self) -> str:
        """Where this signal lives in the open book."""
        return f"{self.symbol}|{self.rule}" if self.rule else self.symbol


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
    ev_per_margin: float = 0.0
    margin_weight: float = 1.0
    # Book mode: the rule's own time limit, in minutes. A barrier trade
    # that touches neither side is closed at the limit it was measured
    # with, not at the generic timeout.
    max_hold_min: float | None = None
    rule: str = ""
    # What was already paid to open. Carried so a closed trade can report
    # its own full round trip: fees are charged on NOTIONAL, so at 3x
    # leverage they cost 3x what a glance at the margin suggests, and
    # whether they ate the edge is the first question to ask of a losing
    # session.
    entry_fee: float = 0.0
    slot: str = ""
    score: float = 0.0
    kelly_full: float = 0.0
    edge_used: float = 0.0
    probing: bool = False


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
    fees_usd: float = 0.0
    rule: str = ""
    held_minutes: float = 0.0


class Broker:
    def __init__(self, client, symbols: list[str], equity: float,
                 max_positions: int, exit_name: str, min_votes: int,
                 fee: float = L.FEE_ROUND_TRIP, max_leverage: float | None = None,
                 margin_pct: float = 0.05, max_notional_x: float = 10.0,
                 conviction_floor: float = L.CONVICTION_FLOOR,
                 expectancy_gate: bool = True,
                 assumed_win_rate: float | None = None,
                 signal_source: str = "methods",
                 book_file: str = "book.json",
                 book_anywhere: bool = False,
                 trust_book: bool = False,
                 earn_stake: bool = False,
                 stake_curve: str = "linear",
                 share_stakes: bool = False,
                 potential_sizing: bool = True,
                 sizing: str = "kelly",
                 max_margin_pct: float = L.MAX_MARGIN_FRACTION,
                 limit_entry: bool = False, slippage: float = 0.0,
                 fee_override: float | None = None,
                 mtf_gate: str = "band", probe_pct: float = 2.0,
                 probe_n: int = 30, probe_budget: float = 0.05):
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
        # "methods" = the twelve voting rules on 30m bars.
        # "patterns" = the hard-coded lookup table on 15m bars, which
        # carries its own per-shape confidence into the leverage.
        self.signal_source = signal_source
        # Scale each trade's margin by its expected return per dollar of
        # margin. False restores a flat slice for every trade.
        self.potential_sizing = potential_sizing
        # "kelly"     - fraction of equity from the Kelly criterion on the
        #               signal's own lower-bounded win rate. The surer the
        #               trade, the more of the account it takes.
        # "potential" - base slice scaled by return per dollar of margin.
        # "flat"      - the same slice for every trade.
        self.sizing = sizing
        self.max_margin_pct = max_margin_pct
        # Real costs, no choosing: market in, market out, and each
        # symbol's live funding rate straight off the ticker feed.
        self.limit_entry = limit_entry
        self.slippage = slippage
        self.fee_override = fee_override
        self.library = None
        self.survivors: list[dict] = []
        self.book: list[dict] = []
        self.book_anywhere = book_anywhere
        # Let the potential score run the whole account from the first
        # trade, instead of earning its way up from the base slice. The
        # book is fitted; this is the operator's call, not the default.
        self.trust_book = trust_book
        # Off by default: the stake follows the setup in front of it, not
        # the record of the ones behind it. On, the ceiling starts at
        # --margin-pct and opens toward --max-margin-pct as each rule
        # builds a live record.
        self.earn_stake = earn_stake
        # "linear" -> stake% = score. "square" -> stake% = score^2/100.
        self.stake_curve = stake_curve
        # Sizing reads only this trade's own potential when True.
        self.pure_potential = not earn_stake
        # False: fill by score rank until margin runs out, each setup
        # getting the full stake its own score asked for. True: scale the
        # whole standing set down so all of them fit.
        self.share_stakes = share_stakes
        self._book_cache: dict[tuple, tuple] = {}
        self._fired_cache: dict[tuple, tuple] = {}
        self.bar_minutes = L.BAR_MINUTES
        # FORWARD EVIDENCE. The in-sample band table said one band of six
        # paid +0.406 of a win at t = 2.76. Refit on a clean split and
        # scored on 24 days it had never seen, the same band paid +0.102
        # at t = 0.42, the model's rank correlation with the outcome was
        # -0.0096, and the shipped gate lost 0.678%/trade -- worse than a
        # rotation of its own predictions. See fp/verdict.py.
        #
        # So no in-sample number sizes anything any more. In probe mode
        # every shape and side trades a fixed small stake, and the ONLY
        # thing that lets a rule grow past it is its own live record.
        # probe_n is where the record's standard error is small enough to
        # separate the fee from an edge; before that, nothing is claimed.
        self.mtf_gate = mtf_gate
        self.probe_pct = probe_pct
        self.probe_n = probe_n
        self.probe_budget = probe_budget
        self.probe_trades = 0
        # What probing has actually cost, as a fraction of STARTING equity.
        # The banner promises the experiment is bounded; this is what makes
        # that true rather than a figure of speech. Losses only -- a probe
        # that pays is not spending the budget.
        self.probe_spent = 0.0
        self.promoted: set[str] = set()
        self.retired: set[str] = set()
        # ONE signal source. The twelve-method vote, the fitted rule
        # books, the survivor list, the trend follower and the
        # multi-timeframe regressor are all gone: every one of them was
        # measured and every one of them failed out of sample. What is
        # left is the logic in fp/engine.py, and it only ships the shapes
        # that survived fp/run_engine.py's walk-forward.
        from fp.live_engine import Engine
        self.engine = Engine()
        self._eng_cache: dict = {}
        self._eng_lock = threading.Lock()
        # Scored on one-minute bars: the finest resolution the factors
        # were built at, and re-asking faster only re-reads a closed bar.
        self.bar_minutes = 1
        self.exit_name = exit_name
        self.min_votes = min_votes
        self.fee = fee
        self.max_leverage = max_leverage
        # No hold floor. A ten-minute trade and a ten-day trade face the
        # same question -- does the edge clear the cost of taking it --
        # and a whitelist entry has to answer it with its own measured
        # numbers. Time is reported, never required.
        self.survivors = [w for w in self.survivors
                          if float(w.get("oos_mean", 0.0)) > 0.0]

        # Keyed by SLOT, not by symbol. In book mode a slot is
        # "SYMBOL|tf:name:side", so one symbol can carry several rules at
        # once: a daily rule holding SOL for six days used to lock that
        # symbol out of every 15m rule for the same six days, which is
        # most of why twelve hours produced fifteen trades.
        self.open: dict[str, Position] = {}
        self.closed: list[Closed] = []
        # Standing signals, one per slot, replaced when its bar rolls.
        self.signals: dict[str, Signal] = {}
        # Live per-rule record: slot-rule -> [n, sum of net returns as a
        # fraction of notional]. This is what lets a rule's stake follow
        # what it is ACTUALLY earning rather than what it was fitted to.
        self.rule_record: dict[str, list] = {}
        # Global calibration of the book against reality: how much of the
        # edge the book claimed has actually turned up. Two running sums,
        # claimed and realized, over every closed book trade.
        self.book_claimed = 0.0
        self.book_realized = 0.0
        # Sum of each closed book trade's own barrier-implied variance.
        # See book_calibration() for why the realized variance will not do.
        self.book_var = 0.0
        self.book_trades = 0
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
        # Setups whose claimed mean implied a win rate above 100% at the
        # live sigma. Counted separately: that is a refuted claim, not a
        # thin edge, and the two mean different things.
        self.refuted = 0
        self.no_survivors = 0
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
        # The same call already carries each symbol's REAL funding rate, so
        # the cost of a trade is the exchange's own number, not a guess.
        self.funding: dict[str, float] = {}
        self.next_funding: dict[str, int] = {}
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

    # ------------------------------------------------------- potential
    def book_calibration(self) -> float:
        """How much of the book's claimed edge has actually shown up.

        The book is fitted to its own data, so its per-trade means are
        the best case, not the expected case. Rather than pick a haircut
        by hand, this measures one: the ratio of realized to claimed edge
        over every book trade closed so far, clipped to [0, 1].

        With no trades it is 1.0 -- the book is taken at its word until
        it has had a chance to be wrong. As trades accumulate it moves on
        its own, and a book that is not paying shrinks its own stake to
        nothing without anybody editing a constant.
        """
        if self.book_trades <= 0 or self.book_claimed <= 0:
            return 1.0
        ratio = max(0.0, min(1.0, self.book_realized / self.book_claimed))
        # Shrink toward 1, not straight to the raw ratio. Six unlucky
        # stop-outs drive the raw ratio to zero, and a zero calibration
        # takes no further trades -- so it can never gather the evidence
        # that would lift it again. That is an absorbing state, not a
        # measurement. The weight on the observed ratio is n/(n+n0),
        # where n0 is the number of trades at which the realized mean's
        # standard error equals the edge being tested: below that the
        # sample cannot tell a dead book from a quiet one, above it the
        # ratio takes over completely and a book that truly does not pay
        # does shut itself down.
        n = self.book_trades
        mean_claim = self.book_claimed / n
        if n < 1 or mean_claim <= 0 or self.book_var <= 0:
            return 1.0
        # n0 comes from the BARRIER-IMPLIED dispersion of the trades taken,
        # not from their realized sample variance. Two losing trades at the
        # same designed stop have a sample variance of ZERO, which sent n0
        # to its floor, the weight to 2/3, and the calibration to 0.33 --
        # a bad afternoon condemning the book, which is the exact failure
        # this shrinkage exists to prevent. The barrier dispersion is known
        # before any trade happens and cannot collapse.
        var = self.book_var / n
        n0 = var / (mean_claim * mean_claim)
        w = n / (n + max(n0, 1.0))
        return (1 - w) * 1.0 + w * ratio

    def rule_edge(self, rule: str, claimed: float,
                  a: float, b: float) -> tuple[float, float]:
        """The per-trade edge a rule is really earning, and how well known.

        Two sources, blended by how much each is worth. The claim comes
        from the book, discounted by what the book as a whole has
        delivered. The record comes from this rule's own closed trades.
        The weight on the record is n/(n+n*), where n* is the number of
        trades at which the record's standard error equals the claimed
        edge -- derived from the rule's own barrier geometry, so nothing
        here is a chosen constant.

        The claim is NOT clipped here. A claim above what the barriers
        can pay implies a win rate over 100%, and clipping it to the
        maximum would hand the least credible setups the largest stake --
        exactly backwards. potential() sees the raw number, calls it
        refuted and scores it zero.

        Returns (edge, evidence weight in [0, 1]).
        """
        if self.pure_potential:
            # The stake is a function of THIS trade's potential and
            # nothing else. No calibration from other rules, no blend with
            # this rule's own past: a setup scoring 60 gets the same stake
            # on its first trade as on its hundredth. History still stops
            # a rule that is losing -- see the cut in potential() -- but it
            # no longer decides how much a live setup is worth.
            return claimed, 0.0
        prior = claimed * self.book_calibration()
        n, tot = self.rule_record.get(rule, (0, 0.0))
        if n <= 0:
            return prior, 0.0
        # n* is how many live trades it takes for the record to be worth
        # as much as the claim. It is the sample size at which a 2-sigma
        # distinction between the claim and BREAK-EVEN becomes possible,
        # so the dispersion is measured under break-even -- the null --
        # not under the claim.
        #
        # Under the claim it collapses exactly when the claim is least
        # believable: a rule asserting it wins 98% of the time has almost
        # no variance under its own hypothesis, so n* fell to its floor of
        # 1, one trade bought a weight of 0.5, and the ceiling opened to
        # 52% of the account on a single result. Break-even dispersion
        # does not collapse, because break-even is a coin flip between the
        # target and the stop whatever the claim says.
        p_be = a / (a + b)
        var_null = p_be * b * b + (1 - p_be) * a * a
        sd = math.sqrt(max(var_null, 1e-12))
        n_star = (2.0 * sd / claimed) ** 2 if claimed > 0 else float("inf")
        w = n / (n + max(n_star, 1.0))
        return (1 - w) * prior + w * (tot / n), w

    def potential(self, tp_dist: float, sl_dist: float, rule: str,
                  claimed: float, fee: float) -> dict:
        """Score this setup out of 100, on one scale for every rule.

        Everything a trade can be judged on reduces to one question: how
        far above break-even is its win rate? The barriers fix what
        break-even IS -- a bet paying +b against -a needs

            p_be = a / (a + b)

        just to stand still, so a wide target is not free, it is a higher
        bar. The credible edge implies

            p_est = (edge + a) / (a + b)

        and the score is where that sits between break-even and certainty:

            POTENTIAL = 100 x (p_est - p_be) / (1 - p_be)

        0 means break-even, 100 means it never loses. The scale is
        bounded, dimensionless and directly comparable across
        timeframes, so a 15m setup and a daily one can be ranked against
        each other and sized off the same number.

        Both a and b are NET of the round trip, so a target the fee eats
        scores low without any separate cost rule -- which is the thing
        that actually sank the live session, where fees were 59% of the
        loss.

        A claim implying p_est > 1 is not a strong claim, it is a refuted
        one: no rule wins more often than always. That happens when a
        fitted mean meets a real sigma -- at 0.2% the book's median rule
        implies 109% -- and the honest reading is that the claim carries
        no information, so the setup scores 0 and is not traded. It is
        the low-volatility, tight-barrier setups this removes, which are
        exactly the ones whose fees do not clear.
        """
        b = tp_dist - fee              # what a win actually pays
        a = sl_dist + fee              # what a loss actually costs
        out = {"score": 0.0, "a": a, "b": b, "edge": 0.0, "w": 0.0,
               "p_be": 1.0, "p_est": 0.0, "refuted": False, "cut": False}
        if b <= 0 or a <= 0:
            # The target does not clear the round trip. No bet here.
            return out
        edge, w = self.rule_edge(rule, claimed, a, b)
        out["w"] = w
        # Sizing UP and cutting a loser are different questions and do not
        # deserve the same standard of proof. The blend above is symmetric
        # -- it asks how many trades it takes to confirm the CLAIM, which
        # is deliberately slow. Losses are direct evidence of loss and need
        # no such patience: if this rule's own record is more than two
        # standard errors below zero, it stops, whatever the claim says.
        n, tot = self.rule_record.get(rule, (0, 0.0))
        if n >= 2:
            p_be0 = a / (a + b)
            sd0 = math.sqrt(max(p_be0 * b * b + (1 - p_be0) * a * a, 1e-12))
            if tot / n < -2.0 * sd0 / math.sqrt(n):
                out["cut"] = True
                return out
        p_be = a / (a + b)
        p_est = (edge + a) / (a + b)
        out.update(p_be=p_be, p_est=p_est, edge=edge)
        if p_est > 1.0:
            out["refuted"] = True
            return out
        if p_est <= p_be:
            return out
        out["score"] = 100.0 * (p_est - p_be) / (1.0 - p_be)
        return out

    def book_margin_fraction(self, tp_dist: float, sl_dist: float,
                             lev: float, rule: str, claimed: float,
                             fee: float) -> tuple[float, float, float]:
        """What fraction of equity this trade's own potential justifies.

        A book rule is a two-outcome bet: it pays +tp_dist or -sl_dist on
        notional, both net of the round trip. Kelly for such a bet stakes
        f* = edge / (loss x gain) of NOTIONAL, so the margin behind it is
        f*/leverage. A rule with a large measured edge against tight
        barriers asks for the whole account, and is allowed to have it --
        subject to two hard limits that are arithmetic, not taste:

          * a stop must not be able to cost more than the account:
            margin x leverage x loss <= equity
          * --max-margin-pct, the operator's own ceiling

        Half-Kelly is used because the edge is an ESTIMATE. Full Kelly is
        optimal only when the edge is known exactly; at half stake you
        keep three quarters of the growth for a quarter of the variance,
        and you survive being wrong about the mean -- which, on a fitted
        book, is the way to bet.

        Returns (margin fraction, uncapped Kelly, edge used).
        """
        P = self.potential(tp_dist, sl_dist, rule, claimed, fee)
        a, b, edge, w = P["a"], P["b"], P["edge"], P["w"]
        if P["score"] <= 0:
            return 0.0, 0.0, edge
        kelly = edge / (a * b)                  # fraction of equity, notional
        # THE STAKE IS THE SCORE, read as a percentage. A setup scoring
        # 60 out of 100 commits 60% of the account; one scoring 100 --
        # a setup that by its own barriers cannot lose -- commits all of
        # it. Linear, so the number on the dashboard IS the number spent.
        #
        # --stake-curve square instead spends score^2, which keeps most of
        # the growth for a quarter of the variance and is what a Kelly
        # bettor does with an ESTIMATED edge. Linear is the operator's
        # choice and is the default because it does what it says.
        frac_of_max = (P["score"] / 100.0)
        if self.stake_curve == "square":
            frac_of_max *= frac_of_max
        by_score = self.max_margin_pct * frac_of_max
        frac = 0.5 * kelly / max(lev, 1e-9)     # margin behind it, half Kelly
        ruin_cap = 1.0 / (lev * a)              # one stop may not clear the account
        # A stake above the base slice has to be EARNED. The book is
        # fitted, so on its own numbers Kelly asks for 45x the account --
        # sizing off that on day one is betting the account on a
        # backtest. The ceiling therefore starts at the base slice and
        # opens toward --max-margin-pct only as the rule's OWN live
        # record accumulates, on the same weight w that decides how much
        # of the edge estimate comes from that record. A rule that proves
        # itself can end up taking the whole account; one that has never
        # traded cannot. --trust-book lifts this for an operator who
        # wants the score to run the account from the first trade.
        # The ramp opens on evidence the rule is PAYING, not on evidence
        # of any kind. w measures how much the live record is worth
        # knowing, and losing trades are worth knowing too -- so a rule
        # that lost three in a row used to see its ceiling RISE from the
        # base slice, which is the opposite of the ramp's purpose. The
        # opening is now w scaled by how much of the claim the record has
        # actually delivered, floored at nothing.
        if self.earn_stake:
            n_, tot_ = self.rule_record.get(rule, (0, 0.0))
            paid = (max(0.0, min(1.0, (tot_ / n_) / claimed))
                    if n_ > 0 and claimed > 0 else 0.0)
            earned = (self.margin_pct
                      + (self.max_margin_pct - self.margin_pct) * w * paid)
        else:
            earned = self.max_margin_pct
        return (min(by_score, frac, ruin_cap, earned, self.max_margin_pct),
                kelly, edge)

    def stake_scale(self) -> float:
        """How far every standing stake must be scaled to fit the account.

        Kelly solves ONE bet at a time. Twenty-four rules firing on the
        same bar are twenty-four simultaneous bets, and each one asking
        for its solitary optimum means the first alphabetically takes the
        whole account and the other twenty-three never trade -- which is
        both worse diversified and fewer trades.

        So the standing set shares. If the asks total less than the
        account, everyone gets what they asked for, and a lone
        high-potential signal still takes all of it. If they total more,
        every stake is scaled by the same factor, which leaves the SPLIT
        proportional to potential while the SUM fits.
        """
        if not self.share_stakes:
            # Fill in score order and let free margin do the truncating.
            # Scaling everyone down proportionally means a 97/100 setup
            # and a 12/100 setup both get a quarter of what they asked
            # for, which is not what a score is FOR. Ranked filling gives
            # the best setup its full stake and the next one whatever is
            # left -- so a setup that cannot lose takes the account, and
            # a marginal one waits for a pass where there is room.
            return 1.0
        want = sum(g.margin_frac for g in self.signals.values()
                   if g.margin_frac is not None and g.slot not in self.open)
        return 1.0 / want if want > 1.0 else 1.0

    def slice_size(self, atr_pct: float | None = None,
                   p_win: float | None = None,
                   fee: float | None = None,
                   sig: "Signal | None" = None,
                   scale: float = 1.0) -> float:
        """Margin this trade commits.

        Under Kelly the answer does not start from a fixed slice at all:
        it is the fraction of equity the trade's own edge justifies, from
        zero up to max_margin_pct. A trade with a real 70% record takes
        90% of the account; one at the system's measured 35.1% takes
        nothing, because at that rate there is no edge to bet.
        """
        eq = max(0.0, self.equity_total)
        fee = self.fee if fee is None else fee
        if sig is not None and sig.margin_frac is not None:
            # A book signal solved its own stake against its own barriers
            # and its own live record. See book_margin_fraction(). `scale`
            # is how the standing set shares one account -- see
            # stake_scale().
            return eq * sig.margin_frac * scale
        # No per-signal solution available (a bare slice_size() call --
        # the has_room() probe, or the dashboard). The floor slice is what
        # those fall back to. There is no ATR ladder behind this any more:
        # every real stake comes from the signal's own potential.
        return eq * self.margin_pct

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
            if True:
                # When each signal solves its own stake the slice is
                # per-trade, so "is there room" can only mean "is there
                # anything left at all".
                return self.free_margin > 0
            return self.free_margin >= self.slice_size() > 0

    # ---------------------------------------------------------------- market

    def klines(self, symbol: str) -> pd.DataFrame | None:
        # The logic fetches its own history in _engine_panel, because a
        # third of its factors need the whole board at once. This stays
        # only so the price-refresh path has a fallback.
        try:
            rows = self.client.get_kline(category="linear", symbol=symbol,
                                         interval=str(self.bar_minutes),
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
        snap, fund, nxt = {}, {}, {}
        for r in rows:
            try:
                p = float(r["lastPrice"])
            except (TypeError, ValueError, KeyError):
                continue
            if p <= 0:
                continue
            snap[r["symbol"]] = p
            try:
                fund[r["symbol"]] = float(r.get("fundingRate") or 0.0)
            except (TypeError, ValueError):
                fund[r["symbol"]] = 0.0
            try:
                nxt[r["symbol"]] = int(r.get("nextFundingTime") or 0)
            except (TypeError, ValueError):
                nxt[r["symbol"]] = 0
        if snap:
            self.prices = snap
            self.funding = fund
            self.next_funding = nxt
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

    def evaluate(self, symbol: str, bars: pd.DataFrame | None):
        """Score one symbol on its last closed bar and store the verdict.

        Returns a LIST of standing signals -- one per surviving shape and
        side that qualifies. A coin can carry several at once; being long
        a one-hour setup is not a reason to skip a twelve-hour one.

        The bars argument is ignored. The logic reads the whole board at
        once because a third of its factors are cross-sectional, so it
        fetches its own history in _engine_panel rather than being handed
        one symbol's.
        """
        self.evaluations += 1
        now_bar = closed_bar_ts(bar_minutes=self.bar_minutes)
        self.evaluated_bar[symbol] = now_bar
        return self._evaluate_engine(symbol, now_bar)

    def _engine_panel(self, bar_ts: int):
        """Factors for the WHOLE board, built once per bar.

        A third of the factors are cross-sectional -- this coin's rank
        among the ten, the board's median move, the dispersion, the
        residual. None of them can be computed one symbol at a time, so
        the panel is built for every scanned symbol together and cached
        until the bar rolls. Building it per symbol would silently change
        every rank: no exception, no warning, just a different number
        than the model was trained on.
        """
        with self._eng_lock:
            if self._eng_cache.get("bar") == bar_ts:
                return self._eng_cache.get("X") or {}
            from fp.live_engine import fetch_1m

            def bump():
                with self.counter_lock:
                    self.kline_calls += 1

            bars = {}
            for s in self.symbols:
                d = fetch_1m(self.client, s, self.engine.need_bars,
                             counter=bump)
                if d is not None:
                    bars[s] = d
            try:
                X = self.engine.panel(bars)
            except Exception:
                logger.exception("engine panel failed")
                X = {}
            self._eng_cache = {"bar": bar_ts, "X": X}
            return X

    def _evaluate_engine(self, symbol: str, bar_ts: int) -> list:
        """Score one symbol with the shipped logic.

        Every surviving (shape, side) gets its own slot, so a coin can
        carry several at once instead of being locked by whichever fired
        first. The potential score decides both WHETHER to trade and HOW
        MUCH: it is the same 0-100 number for a one-hour scalp and a
        twelve-hour swing, which is the only way those two can share one
        account.
        """
        if not getattr(self, "engine", None) or not self.engine.ok:
            self.no_survivors += 1
            self._drop_signals(symbol)
            return []
        X = self._engine_panel(bar_ts)
        x = X.get(symbol)
        if x is None:
            self._drop_signals(symbol)
            return []
        try:
            cands = self.engine.candidates(x)
        except Exception:
            logger.debug("engine scoring failed %s", symbol, exc_info=True)
            self._drop_signals(symbol)
            return []
        if not cands:
            self.no_signal += 1
            self._drop_signals(symbol)
            return []

        from fp import leverage as S
        out, keep = [], set()
        for c in cands:
            fee = self.trade_cost(symbol, c["side"])
            # The stop must fire BEFORE liquidation -- see
            # leverage.solvent_leverage, which is the one place that
            # arithmetic lives so the bot and its tests cannot disagree.
            chain = S.solvent_leverage(c["tp_dist"], c["sl_dist"],
                                       c["hold_min"], c["sigma"],
                                       self.max_leverage or S.LEVERAGE_MAX)
            if not chain["tradeable"]:
                self.skipped_unsolvent += 1
                continue
            # The live fee can differ from the one the study assumed --
            # funding is per symbol and signed -- so break-even is
            # recomputed here rather than trusted from the scorer, and the
            # score is converted back into an edge at THIS trade's cost.
            b = c["tp_dist"] - fee
            a = c["sl_dist"] + fee
            if b <= 0:
                self.skipped_negative_ev += 1
                continue
            p_be = a / (a + b)
            p = p_be + (c["score"] / 100.0) * (1.0 - p_be)
            claimed = p * b - (1 - p) * a
            frac, kelly, edge = self.book_margin_fraction(
                c["tp_dist"], c["sl_dist"], chain["leverage"], c["rule"],
                claimed, fee)
            if frac <= 0:
                self.skipped_negative_ev += 1
                continue
            sig = Signal(
                bar_ts=bar_ts, direction=c["side"],
                atr_pct=100.0 * c["sigma"], votes=1, vote_margin=1,
                methods=(f"engine p={100*c['p']:.1f}% vs break-even "
                         f"{100*p_be:.1f}% | tp{c['tp']}/sl{c['sl']} "
                         f"hold<={c['hold_min']}m "
                         f"lev={chain['leverage']:.2f}x "
                         f"potential={c['score']:.0f}/100 "
                         f"stake={100*frac:.1f}%"),
                confidence=None, slow_leverage=chain["leverage"],
                tp_dist=c["tp_dist"], sl_dist=c["sl_dist"],
                max_hold_min=float(c["hold_min"]),
                rule=c["rule"], rule_mean=claimed, symbol=symbol,
                margin_frac=frac, kelly_full=kelly, edge_used=edge,
                score=c["score"])
            self.signals[sig.slot] = sig
            keep.add(sig.slot)
            out.append(sig)
        self._drop_signals(symbol, keep)
        if not out:
            self.no_signal += 1
        return out

    def _drop_signals(self, symbol: str, keep: set[str] | None = None) -> None:
        """Forget this symbol's standing signals, except the ones named."""
        pre = f"{symbol}|"
        for k in [k for k in self.signals
                  if (k == symbol or k.startswith(pre))
                  and (keep is None or k not in keep)]:
            self.signals.pop(k, None)

    def _load_survivors() -> list[dict]:
        """The whitelist, or an empty list if none was ever earned."""
        p = Path(__file__).resolve().parent / "survivors.json"
        if not p.exists():
            logger.warning("no survivors.json -- run python -m fp.survivors; "
                           "until then this mode opens nothing")
            return []
        try:
            data = json.loads(p.read_text())
        except Exception:
            logger.warning("survivors.json unreadable", exc_info=True)
            return []
        got = data.get("logics", [])
        logger.info("survivors.json: %d logics cleared the bar", len(got))
        return got

    def refresh_signals(self, symbols: list[str]) -> int:
        """Refetch and re-score a chunk of symbols. Returns signals standing."""
        fetched = self.prefetch(symbols)
        n = 0
        for sym in symbols:
            got = self.evaluate(sym, fetched.get(sym))
            n += len(got) if isinstance(got, list) else (1 if got else 0)
        return n

    def stale_symbols(self) -> list[str]:
        """Symbols not yet scored on the most recently closed bar.

        Symbols already holding a position are INCLUDED: a coin carries
        one position per shape and side, so an open twelve-hour swing
        must not stop its one-hour shapes being scored.
        """
        want = closed_bar_ts(bar_minutes=self.bar_minutes)
        # A coin is never "done". It holds one position PER SHAPE AND
        # SIDE, so one already carrying a twelve-hour swing must keep
        # being scored or its one-hour shapes never get a turn. Skipping
        # open symbols is what made a slow trade lock a coin for a day.
        return [s for s in self.symbols
                if self.evaluated_bar.get(s, -1) < want]

    def _closed_fees(self) -> float:
        """Fees belonging to CLOSED trades only. self.fees_paid also holds
        the entry fee of every position still open, so comparing it to the
        realized P&L overstates what the closed book was charged."""
        return sum(t.fees_usd for t in self.closed)

    def _tradeable_bands(self) -> tuple[int, int]:
        """How many shipped shapes can currently clear their own cost.

        This used to count ATR bands in a table measured for the
        twelve-method exit -- another logic's ruler applied to this one.
        It now counts what the bot actually holds: shapes that survived
        the walk-forward.
        """
        n = len(getattr(self.engine, "models", {}) or {})
        return n, n

    def trade_cost(self, symbol: str, direction: int) -> float:
        """What THIS trade will really cost, as a fraction of notional.

        Taker in and taker out, because the bot sends market orders and a
        TP/SL is a market order either way, plus the symbol's own funding
        rate as Bybit is publishing it right now.

        Funding is signed. A positive rate means longs pay shorts, so a
        short collects it -- for that side it is a rebate, not a cost, and
        pretending otherwise would reject trades that are actually cheaper
        than average.
        """
        if self.fee_override is not None:
            return self.fee_override
        base = L.ENTRY_FEE_TAKER + L.EXIT_FEE_TAKER + 2 * self.slippage
        if self.limit_entry:
            base = L.ENTRY_FEE_MAKER + L.EXIT_FEE_TAKER + 2 * self.slippage
        rate = self.funding.get(symbol, L.FUNDING_RATE_TYPICAL)
        events = L.EXPECTED_HOLD_HOURS.get(self.exit_name,
                                           5.0) / L.FUNDING_INTERVAL_HOURS
        return base + direction * rate * events

    # ----------------------------------------------------------------- entry

    def try_open(self, slot: str, scale: float = 1.0) -> bool:
        """Open the slot's standing signal if there is margin for it.

        A slot is a symbol outside book mode and a (symbol, rule) pair
        inside it, so two rules on the same symbol are two independent
        trades rather than one blocking the other.
        """
        sig = self.signals.get(slot)
        if sig is None or slot in self.open:
            return False
        symbol = sig.symbol or slot
        # One entry per SLOT per bar: without this a stop-out would be
        # reopened immediately by the same standing verdict. Keyed on the
        # slot, not the symbol, or the first rule to trade a symbol would
        # silence every other rule on it for that bar.
        if self.traded_bar.get(slot) == sig.bar_ts:
            return False

        price = self.last_price(symbol)
        if price is None or price <= 0:
            return False

        # Expectancy first: it costs nothing and it is the only test that
        # can tell this trade is not worth taking at all.
        fee = self.trade_cost(symbol, sig.direction)
        # The old expectancy gate read fp/edge_table.json, a table
        # measured for the twelve-method logic on its own TP/SL ladder.
        # Judging a shape by another logic's record is exactly the mistake
        # this rebuild removes: every signal now arrives with its own
        # break-even, computed from its own barriers at its own
        # volatility, and potential() has already refused it if the target
        # does not clear the bill.

        d = sig.direction
        atr = sig.atr_pct / 100.0 * price
        # A stop past the liquidation price turns a sized loss into a
        # total one. Refused here as well as in the scorer, so no future
        # signal path can slip around it.
        if sig.sl_dist is not None and sig.slow_leverage:
            if (sig.sl_dist * sig.slow_leverage * L.SOLVENCY_BUFFER
                    >= L.LIQ_MARGIN_FRACTION):
                self.skipped_unsolvent += 1
                return False
        if sig.slow_leverage is not None:
            chain = {"leverage": sig.slow_leverage, "lev_base": sig.slow_leverage,
                     "potential_score": float("nan"), "conviction": float("nan"),
                     "conviction_haircut": 1.0, "solvency_cap": float("inf"),
                     "tradeable": True}
        else:
            # Every signal now arrives with its leverage already solved
            # from its own stop distance and hold -- see fp/leverage.py.
            # A signal without one is a bug, not a fallback.
            raise ValueError(f"signal for {sig.symbol} carries no leverage")
        if not chain["tradeable"]:
            # Even 1x cannot keep the stop inside the liquidation price.
            self.skipped_unsolvent += 1
            return False
        lev = chain["leverage"]
        tp_mult = L.TP_MULTIPLES.get(self.exit_name)
        if sig.sl_dist is not None:
            # Liquidation sits at 0.9/lev away, so a stop further out than
            # that is never reached -- the position is liquidated first and
            # the rule's risk model is a fiction. Refuse rather than trade
            # a stop the account cannot survive.
            if not (sig.sl_dist * lev < 0.9):
                self.skipped_unsolvent += 1
                return False

        # A fixed slice of current equity per trade, and only if that whole
        # slice is free -- with no position cap, free margin is what stops
        # the bot opening more than the account can carry. Sizing, the fee
        # debit and the book entry happen under one lock so two passes
        # cannot spend the same margin twice.
        with self.lock:
            if slot in self.open:
                return False
            if self.max_positions and len(self.open) >= self.max_positions:
                self.skipped_no_margin += 1
                return False
            # Kelly may ask for the whole account, so the entry fee has
            # to be reserved out of the same free margin: committing
            # everything and paying the fee afterwards leaves the book
            # oversubscribed. margin*(1 + lev*fee_rate) <= free.
            want = self.slice_size(sig.atr_pct, sig.p_win, fee, sig, scale)
            entry_rate = ((L.ENTRY_FEE_MAKER if self.limit_entry
                           else L.ENTRY_FEE_TAKER) + self.slippage) * lev
            affordable = self.free_margin / (1.0 + entry_rate)
            margin = min(want, affordable)
            # Below a cent of margin the fee dominates whatever is left.
            if want <= 0 or margin < 0.01:
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
                # A book rule's barriers ARE the rule, so they are used
                # verbatim rather than replaced by the generic ATR ladder --
                # and they are placed around the price this position
                # ACTUALLY fills at, not around the bar close that produced
                # the signal.
                tp_price=(price * (1 + d * sig.tp_dist)
                          if sig.tp_dist is not None
                          else ((price + d * tp_mult * atr) if tp_mult
                                else float("nan"))),
                sl_price=(price * (1 - d * sig.sl_dist)
                          if sig.sl_dist is not None
                          else price - d * L.SL_MULTIPLE * atr),
                liq_price=price * (1 - d * L.LIQ_MARGIN_FRACTION / lev),
                exit_name=self.exit_name, methods=sig.methods,
                votes=sig.votes, vote_margin=sig.vote_margin,
                best_price=price, trail_dist=L.TRAIL_MULTIPLE * atr,
                # A book rule's potential is its own MEASURED return per
                # trade, scaled by the leverage it is taken at. The old
                # potential_score is the ATR ladder's and is nan here --
                # printing that told the operator nothing.
                potential_score=(sig.rule_mean * lev if sig.rule
                                 else chain["potential_score"]),
                conviction=chain["conviction"],
                lev_base=chain["lev_base"],
                ev_per_margin=sig.rule_mean * lev,
                margin_weight=(margin / (self.equity_total * self.margin_pct)
                               if self.equity_total > 0 else 1.0),
                max_hold_min=sig.max_hold_min, rule=sig.rule,
                entry_fee=entry_fee, slot=slot, score=sig.score,
                kelly_full=sig.kelly_full, edge_used=sig.edge_used,
            )
            self.open[slot] = pos
            self.traded_bar[slot] = sig.bar_ts

        logger.info("%s OPEN %s @%.6f atr_rank=%.2f conviction=%.2f "
                    "base=%.0fx x%.2f -> lev=%.0fx votes=%d(margin %d) [%s] "
                    "exit=%s tp=%.6f sl=%.6f margin=$%.3f",
                    symbol, "LONG" if d > 0 else "SHORT", price,
                    chain["potential_score"], chain["conviction"],
                    chain["lev_base"], chain["conviction_haircut"], lev,
                    sig.votes, sig.vote_margin, sig.methods, self.exit_name,
                    pos.tp_price, pos.sl_price, margin)
        if sig.margin_frac is not None:
            logger.info("    %s stake %.1f%% of equity -- Kelly %.1f%% on a "
                        "%+.3f%%/trade edge (book claims %+.3f%%, "
                        "calibration %.2f), capped by %s",
                        slot, 100 * margin / max(self.equity_total, 1e-9),
                        100 * sig.kelly_full, 100 * sig.edge_used,
                        100 * sig.rule_mean, self.book_calibration(),
                        "free margin" if margin < want else "its own Kelly")
        elif self.sizing == "kelly":
            p = (sig.p_win if sig.p_win is not None
                 else L.MEASURED_WIN_RATE.get(self.exit_name, 0.35))
            logger.info("    %s Kelly %.1f%% of equity on p=%.1f%% "
                        "(%d/%d live), fee eats %.1f%% of margin",
                        symbol, 100 * margin / max(self.equity_total, 1e-9),
                        100 * p, sig.live_wins, sig.live_n, 100 * lev * self.fee)
        else:
            logger.info("    %s sized at %.2fx the base slice "
                        "(EV %+.2f%% per $ of margin, fee eats %.1f%% of it)",
                        symbol, pos.margin_weight, 100 * pos.ev_per_margin,
                        100 * lev * self.fee)
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
        # Best first. The scarce resource is the exposure ceiling, not
        # margin, so whichever signals are tried first spend it -- and in
        # dict order that is arrival order, which has nothing to do with
        # quality. Ranked by expected return per dollar of margin, the
        # budget goes to the setups that keep the most of it.
        def rank(k):
            sg = self.signals[k]
            # One scale for every setup: the potential score. Ranking by
            # anything else would order the board by a model none of these
            # signals came from.
            return -sg.score
        order = sorted(self.signals, key=rank)
        scale = self.stake_scale()
        for slot in order:
            if slot in self.open:
                continue
            sig = self.signals.get(slot)
            if sig is None or self.traded_bar.get(slot) == sig.bar_ts:
                continue
            if not self.has_room():
                waiting += 1
                continue
            try:
                if self.try_open(slot, scale):
                    opened += 1
                else:
                    waiting += 1
            except Exception:
                logger.debug("entry failed for %s", slot, exc_info=True)
        self.blocked_signals = waiting
        self.blocked_by = {
            "margin": self.skipped_no_margin - before[0],
            "exposure": self.skipped_max_notional - before[1],
            "unsolvent": self.skipped_unsolvent - before[2],
        }
        return opened, waiting

    # ------------------------------------------------------------------ exit

    def manage(self, slot: str, price: float | None = None) -> None:
        pos = self.open.get(slot)
        if pos is None:
            return
        symbol = pos.symbol
        if price is None:
            price = self.last_price(symbol)
        if price is None or price <= 0:
            return
        d = pos.direction

        if (price <= pos.liq_price) if d > 0 else (price >= pos.liq_price):
            self._close(slot, pos.liq_price, "liquidated")
            return
        if (price <= pos.sl_price) if d > 0 else (price >= pos.sl_price):
            self._close(slot, pos.sl_price, "stop_loss")
            return

        if pos.rule:
            # A book position is managed ONLY by the barriers its rule was
            # measured with: the stop above, the target below, and the time
            # limit after that. --exit is not consulted, because a trailing
            # stop laid over a barrier rule is a third rule that nobody
            # tested.
            if np.isfinite(pos.tp_price):
                if (price >= pos.tp_price) if d > 0 else (price <= pos.tp_price):
                    self._close(slot, pos.tp_price, "take_profit")
                    return
        elif pos.exit_name == "net_TRAILING":
            pos.best_price = max(pos.best_price, price) if d > 0 else min(pos.best_price, price)
            trail = pos.best_price - d * pos.trail_dist
            if (price <= trail) if d > 0 else (price >= trail):
                self._close(slot, trail, "trailing")
                return
        elif np.isfinite(pos.tp_price):
            if (price >= pos.tp_price) if d > 0 else (price <= pos.tp_price):
                self._close(slot, pos.tp_price, "take_profit")
                return

        # A barrier trade that touches neither side is closed at the time
        # limit it was MEASURED with. Using the generic timeout instead
        # would trade a different rule from the one that was tested.
        held = (pd.Timestamp.now(tz="UTC") - pos.opened_at).total_seconds()
        limit = ((pos.max_hold_min * 60) if pos.max_hold_min
                 else L.MAX_HOLD_BARS * L.BAR_MINUTES * 60)
        if held >= limit:
            self._close(slot, price, "time_limit" if pos.max_hold_min
                        else "timeout")

    def manage_all(self) -> None:
        """Re-price the whole open book off one ticker snapshot."""
        self.refresh_prices()
        with self.lock:
            book = [(k, p.symbol) for k, p in self.open.items()]
        for slot, sym in book:
            try:
                self.manage(slot, self.prices.get(sym))
            except Exception:
                logger.debug("manage failed for %s", slot, exc_info=True)
        self.mark()

    def _close(self, slot: str, price: float, reason: str) -> None:
        with self.lock:
            pos = self.open.pop(slot, None)
            if pos is None:
                return
            symbol = pos.symbol or slot
            move = (price - pos.entry) / pos.entry * pos.direction
            exit_fee = 0.0
            if reason == "liquidated":
                # The margin is gone; the exit fee comes out of it, not on top.
                pnl = -pos.margin
            else:
                exit_fee = pos.qty * price * (L.EXIT_FEE_TAKER + self.slippage)
                pnl = move * pos.qty * pos.entry - exit_fee
                self.fees_paid += exit_fee
            self.equity += pnl
            now = pd.Timestamp.now(tz="UTC")
            self.closed.append(Closed(
                symbol=symbol, direction=pos.direction, entry=pos.entry,
                exit_price=price, opened_at=pos.opened_at,
                closed_at=now, reason=reason, pnl_usd=pnl,
                return_pct_leveraged=100 * move * pos.leverage,
                methods=pos.methods,
                fees_usd=pos.entry_fee + exit_fee, rule=pos.rule,
                held_minutes=(now - pos.opened_at).total_seconds() / 60.0))
            if pos.rule:
                # Feed the result back into the sizing. `net` is the
                # per-trade return as a FRACTION OF NOTIONAL -- the same
                # unit the book's `mean` is in, which is what makes the
                # two comparable. The rule's next stake is solved partly
                # from this, so a rule that stops paying stops being
                # backed, without anybody editing a file.
                notional = pos.margin * pos.leverage
                # pnl carries only the EXIT fee -- the entry fee was
                # debited from equity when the position opened. The book's
                # mean is net of the WHOLE round trip, so the entry fee has
                # to come off here or every live result is flattered by
                # half the cost and the sizing keeps backing a rule that is
                # really breaking even.
                net = ((pnl - pos.entry_fee) / notional) if notional > 0 else 0.0
                rec = self.rule_record.setdefault(pos.rule, [0, 0.0])
                rec[0] += 1
                rec[1] += net
                if pos.probing:
                    loss = (pos.entry_fee - pnl) / max(self.starting_equity,
                                                       1e-9)
                    if loss > 0:
                        self.probe_spent += loss
                self.book_trades += 1
                claim = pos.ev_per_margin / max(pos.leverage, 1e-9)
                self.book_claimed += claim
                self.book_realized += net
                # The dispersion this trade was DESIGNED to have, recovered
                # from its own barriers: it paid +b or -a, and the mix that
                # produces the claim sets the odds.
                bb = abs(pos.tp_price - pos.entry) / pos.entry - self.fee
                aa = abs(pos.sl_price - pos.entry) / pos.entry + self.fee
                if np.isfinite(bb) and np.isfinite(aa) and (aa + bb) > 0:
                    pw = min(1.0, max(0.0, (claim + aa) / (aa + bb)))
                    self.book_var += max(
                        pw * bb * bb + (1 - pw) * aa * aa - claim * claim, 0.0)
        self.mark()
        logger.info("%s CLOSE %s @%.6f (%s) pnl=$%.4f equity=$%.4f",
                    symbol, "LONG" if pos.direction > 0 else "SHORT",
                    price, reason, pnl, self.equity)

    # ------------------------------------------------------------- reporting

    def open_rows(self) -> list[tuple]:
        with self.lock:
            book = list(self.open.items())
        rows = []
        for slot, pos in book:
            # Keyed by slot now, so the SYMBOL to price against comes off
            # the position -- self.prices.get(slot) would miss every time
            # and silently mark the whole book at its entry price.
            price = self.prices.get(pos.symbol) or pos.entry
            pnl = (price - pos.entry) / pos.entry * pos.direction * pos.qty * pos.entry
            rows.append((pos.symbol, pos.direction, pos.entry, price, pnl,
                         pos.leverage, pos.methods, pos.score, slot))
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
        # A book position runs to ITS rule's time limit, which varies from
        # one timeframe to the next; the old constant described a hold
        # length none of them use -- and the funding half of the cost line
        # is that hold length times the rate.
        holds = [p.max_hold_min / 60.0 for p in self.open.values()
                 if p.max_hold_min]
        hold_h = (float(np.mean(holds)) if holds
                  else L.EXPECTED_HOLD_HOURS.get(self.exit_name, 5.0))
        return {
            "elapsed": time.time() - self.started_at,
            "starting_equity": self.starting_equity,
            "equity": equity, "equity_incl_open": eq_open,
            "net": eq_open - self.starting_equity,
            "roi_pct": (100 * (eq_open - self.starting_equity) / self.starting_equity
                        if self.starting_equity else 0.0),
            "realized": gp + gl, "gross_profit": gp, "gross_loss": gl,
            "unrealized": unreal, "fees_paid": self.fees_paid,
            # Free margin is total equity minus committed, the same as the
            # entry gate uses. It read cash-minus-committed here, which
            # printed a negative number while the gate saw a positive one.
            "committed_margin": committed, "free_margin": eq_open - committed,
            "margin_used_pct": (100 * committed / eq_open) if eq_open > 0 else 0.0,
            "slice_size": self.slice_size(),
            "sizing_mode": self.sizing, "exit_name": self.exit_name,
            "signal_source": self.signal_source,
            "mtf_gate": self.mtf_gate,
            "probe_pct": self.probe_pct, "probe_n": self.probe_n,
            "probe_trades": self.probe_trades,
            "probe_budget": self.probe_budget,
            "probe_spent": self.probe_spent,
            "promoted": set(self.promoted), "retired": set(self.retired),
            "rule_records": dict(self.rule_record),
            "book_rules": len(self.book),
            "max_margin_pct": self.max_margin_pct,
            "book_calibration": self.book_calibration(),
            "refuted": self.refuted,
            "scores": [g.score for g in self.signals.values()
                       if g.margin_frac is not None],
            "book_trades": self.book_trades,
            "rules_live": len(self.rule_record),
            "hold_hours": hold_h,
            # What the open book is worth if every position runs to its
            # exit at the measured win rate, rather than at today's mark.
            # A book position settles at ITS OWN rule's measured mean --
            # ev_per_margin already carries it -- because the 0.84/0.42
            # payoff below belongs to the old TP3.0/SL1.5 exit and would
            # describe a rule this position is not trading.
            "expected_settle": sum(
                (p.ev_per_margin * p.margin if p.rule else
                 (L.MEASURED_WIN_RATE.get(self.exit_name, 0.35) * 0.84
                  - (1 - L.MEASURED_WIN_RATE.get(self.exit_name, 0.35)) * 0.42)
                 * p.margin - p.margin * p.leverage * self.fee)
                for p in self.open.values()),
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
            "live_funding": (float(np.median(list(self.funding.values())))
                             if self.funding else 0.0),
            "cost_long": (L.ENTRY_FEE_TAKER + L.EXIT_FEE_TAKER
                          + float(np.median(list(self.funding.values())) or 0.0)
                          * hold_h
                          / L.FUNDING_INTERVAL_HOURS) if self.funding else 0.0,
            "min_atr_for_edge": float("nan"),  # no ATR ladder any more
            "tradeable_bands": self._tradeable_bands()[0],
            "total_bands": self._tradeable_bands()[1],
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
            # A book position closes as "time_limit", not "timeout" -- it
            # runs to ITS rule's measured limit. Counting only "timeout"
            # left those trades out of the breakdown entirely: a session
            # with 15 closed showed 4 TP + 6 SL + 0 liq + 0 timed out, and
            # the missing 5 were the ones that ran out of time.
            "timeouts": sum(1 for t in self.closed
                            if t.reason in ("timeout", "time_limit")),
            # Where the money actually went, split by how each trade ended.
            # Win rate alone cannot tell you whether the target is too far
            # or the clock too short; this can.
            "by_reason": _by_reason(self.closed),
            # Fees are charged on notional, so at 3x leverage a round trip
            # costs 0.33% of margin before the price moves at all. This is
            # what share of the gross move they consumed.
            "gross_before_fees": gp + gl + self._closed_fees(),
            "closed_fees": self._closed_fees(),
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
        for sym, d, entry, price, pnl, lev, meth, pot, slot in self.open_rows():
            pos = self.open.get(slot)
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


def _by_reason(closed: list) -> list[dict]:
    """Split the realized P&L by how each trade ended.

    A win rate on its own cannot say WHY a book is losing. A target that
    is too far shows up as few TPs and many time-limit exits; a stop that
    is too tight shows up as SLs that outnumber TPs at a ratio the rule
    was never measured at; fees eating the edge show up as time-limit
    exits whose gross is positive and whose net is not. Those three call
    for different fixes, and only this table tells them apart.
    """
    out = {}
    for t in closed:
        r = out.setdefault(t.reason, {"reason": t.reason, "n": 0, "net": 0.0,
                                      "fees": 0.0, "wins": 0, "minutes": 0.0})
        r["n"] += 1
        r["net"] += t.pnl_usd
        r["fees"] += t.fees_usd
        r["wins"] += 1 if t.pnl_usd > 0 else 0
        r["minutes"] += t.held_minutes
    for r in out.values():
        r["avg"] = r["net"] / r["n"]
        r["gross"] = r["net"] + r["fees"]
        r["avg_minutes"] = r["minutes"] / r["n"]
    return sorted(out.values(), key=lambda r: r["net"])


def _hms(seconds: float) -> str:
    s = int(max(0, seconds))
    return f"{s // 3600:d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def print_dashboard(s: dict) -> None:
    """The live financial readout, reprinted on every beat while running."""
    pf = "inf" if s["profit_factor"] == float("inf") else f"{s['profit_factor']:.2f}"
    print()
    print("=" * 78)
    settled = s["realized"]
    print(f"  LIVE  up {_hms(s['elapsed'])}    "
          f"SETTLED ${settled:+.4f} from {s['closed']} closed trades"
          f"    open mark ${s['unrealized']:+.4f} on {s['open']}")
    print("-" * 78)
    # The old single "net" line added these two together, and an open book
    # marked to market dwarfed the settled figure -- which is how a session
    # with three closed trades, all losses, read as +17.74%.
    if (s["open"] and abs(s["unrealized"]) > abs(settled)
            and s.get("signal_source") not in ("book", "mtf")):
        print(f"  Most of what you see is UNSETTLED. Positions last "
              f"{s.get('hold_hours', 5.0):.1f}h on")
        print(f"  average and stops finish sooner than targets, so an open "
              f"book reads")
        print(f"  better than it will settle. At the measured win rate this "
              f"one settles")
        print(f"  near ${s['expected_settle']:+.4f}, not ${s['unrealized']:+.4f}.")
        print("-" * 78)
    print(f"  CAPITAL   start ${s['starting_equity']:.4f}"
          f"   cash ${s['equity']:.4f}"
          f"   equity+open ${s['equity_incl_open']:.4f}"
          f"   peak ${s['peak_equity']:.4f}")
    print(f"  MARGIN    committed ${s['committed_margin']:.4f}"
          f"   free ${s['free_margin']:.4f}"
          f"   {s['margin_used_pct']:.1f}% of equity deployed")
    # BUG: this line used to read potential_sizing, a different flag, and
    # so described a mode the bot was not running.
    if s.get("signal_source") == "mtf" and s.get("mtf_gate") == "probe":
        how = (f"{s['probe_pct']:.1f}% of equity while a rule is still "
               f"gathering its first\n            {s['probe_n']} trades, then "
               f"that rule's OWN live record and nothing else,\n            "
               f"0% to {100*s['max_margin_pct']:.0f}%. The in-sample table is "
               f"not consulted.")
    elif s.get("signal_source") == "mtf":
        how = (f"the model's predicted net, mapped through the band table to "
               f"a\n            measured %/trade, then that as a percent of "
               f"equity. 0% to {100*s['max_margin_pct']:.0f}%.")
    elif s.get("signal_source") == "book":
        how = (f"each rule's own half-Kelly on its own edge and barriers, "
               f"0% to {100*s['max_margin_pct']:.0f}% of equity")
        cal = s.get("book_calibration", 1.0)
        how += (f"\n            book calibration {cal:.2f} "
                + ("(no closed trades yet -- the book is taken at its word)"
                   if s.get("book_trades", 0) <= 0 else
                   f"({s['book_trades']} closed: it has delivered "
                   f"{100*cal:.0f}% of the edge it claimed, and every stake "
                   f"is scaled by that)"))
    elif s.get("signal_source") in ("slow", "regime", "survivors"):
        # slice_size() short-circuits to the flat slice for these modes, so
        # printing the --sizing flag here described a branch never taken.
        how = (f"flat, "
               f"{100*s['slice_size']/max(s['equity_incl_open'], 1e-9):.1f}% of equity "
               f"every trade (${s['slice_size']:.4f}); the rule's own "
               f"leverage carries its potential")
    else:
        how = {"kelly": "Kelly on each signal's own lower-bounded win rate, "
                        "0% to 100% of equity",
               "potential": f"base slice x {L.MARGIN_WEIGHT_MIN:.2f}-"
                            f"{L.MARGIN_WEIGHT_MAX:.2f} by return per $ of margin",
               "flat": "the same slice for every trade"}[s["sizing_mode"]]
    print(f"            sizing: {how}")
    ceiling = (f"   ceiling {s['max_notional_x']:.0f}x"
               f" (${s['notional_headroom']:.2f} left)"
               if s["max_notional_x"] else "   no ceiling -- potential decides")
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
    print(f"  COST      taker in+out {100*(L.ENTRY_FEE_TAKER+L.EXIT_FEE_TAKER):.3f}%"
          f"   live funding median {100*s['live_funding']:+.4f}%/8h"
          f"   -> a long costs {100*s['cost_long']:.3f}% round trip")
    if s.get("signal_source") == "mtf" and s.get("mtf_gate") == "probe":
        print(f"  EDGE      FORWARD EVIDENCE. The in-sample band table was "
              f"refuted out of")
        print(f"            sample (rank correlation -0.0096 over 24 unseen "
              f"days), so no")
        print(f"            claim from it is used. Each rule probes at "
              f"{s['probe_pct']:.1f}% and grows")
        print(f"            only on its own results.")
        print(f"            probe entries {s.get('probe_trades', 0):,}"
              f"   promoted {len(s.get('promoted') or [])}"
              f"   retired {len(s.get('retired') or [])}"
              f"   refused {s['skipped_negative_ev']:,}")
        rr = s.get("rule_records") or {}
        if rr:
            print(f"  RULES     each rule's own live record "
                  f"(what decides its stake):")
            for r, (n_, tot_) in sorted(rr.items(),
                                        key=lambda kv: -kv[1][0])[:8]:
                state = ("RETIRED " if r in (s.get("retired") or ()) else
                         "promoted" if r in (s.get("promoted") or ()) else
                         f"probe {n_}/{s['probe_n']}")
                print(f"            {r:<34} {n_:>3} trades "
                      f"{100*tot_/max(n_,1):+.3f}%/trade  {state}")
    elif s.get("signal_source") == "mtf":
        print(f"  EDGE      multi-timeframe model, BAND gate. Out of sample no "
              f"band was")
        print(f"            measured profitable, so nothing qualifies. "
              f"Skipped this")
        print(f"            session: {s['skipped_negative_ev']:,}.")
        sc = s.get("scores") or []
        if sc:
            print(f"  POTENTIAL standing scores /100: best {max(sc):.0f}   "
                  f"median {sorted(sc)[len(sc)//2]:.0f}   worst {min(sc):.0f}")
    elif s.get("signal_source") == "book":
        # The ATR-band table belongs to the twelve-method exit. Book mode
        # never consults it -- each rule was measured against the full cost
        # on its own -- so reporting it here graded this book with another
        # book's ruler.
        print(f"  EDGE      {s['book_rules']} book rules, each already net of "
              f"fees + funding where it was measured"
              f"   {s['rules_live']} with a live record")
        sc = s.get("scores") or []
        if sc:
            print(f"  POTENTIAL standing scores /100: "
                  f"best {max(sc):.0f}   median {sorted(sc)[len(sc)//2]:.0f}"
                  f"   worst {min(sc):.0f}"
                  f"   (stake = ceiling x (score/100)^2)")
        print(f"            refused this session: {s['refuted']:,} refuted "
              f"(claimed win rate > 100%)   "
              f"{s['skipped_negative_ev'] - s['refuted']:,} no edge after cost")
    else:
        print(f"  EDGE      negative-expectancy skips {s['skipped_negative_ev']:,}"
              f"   ({s['tradeable_bands']} of {s['total_bands']} measured ATR bands"
              f" clear the cost)")
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
    if s.get("by_reason"):
        # Where the money went, and how much of it the exchange took.
        # Price move and fee are separated because they call for
        # different fixes: a losing gross is the rule, a losing net on a
        # winning gross is the cost.
        print(f"    {'exit':<12}{'n':>4}{'win':>5}{'gross':>10}"
              f"{'fees':>9}{'net':>10}{'avg':>10}{'held':>9}")
        for r in s["by_reason"]:
            print(f"    {r['reason']:<12}{r['n']:>4}"
                  f"{100*r['wins']/r['n']:>4.0f}%"
                  f"{r['gross']:>+10.4f}{-r['fees']:>9.4f}"
                  f"{r['net']:>+10.4f}{r['avg']:>+10.4f}"
                  f"{_hms(60*r['avg_minutes']):>9}")
        g, f = s.get("gross_before_fees", 0.0), s.get("closed_fees", 0.0)
        share = (100 * f / abs(g)) if g else float("inf")
        print(f"  Price move alone      : ${g:+.4f}"
              f"   fees ${f:.4f}"
              + (f"   ({share:.0f}% of the gross move)"
                 if g else "   (the whole loss is fees)"))

    print("\n-- STILL OPEN AT SHUTDOWN " + "-" * 41)
    print(f"  Positions open        : {s['open']}")
    print(f"    currently winning   : {s['open_wins']}")
    print(f"    currently losing    : {s['open_losses']}")
    for sym, d, entry, price, pnl, lev, meth, pot, _slot in s["open_rows"]:
        # For a book position `pot` is the rule's own measured return per
        # trade at this leverage, so it is shown as the percentage it is.
        p_txt = ("  n/a" if pot != pot else f"{pot:.0f}/100")
        print(f"      {sym:12s} {'LONG' if d > 0 else 'SHORT':5s} {lev:5.0f}x "
              f"potential={p_txt} entry={entry:.6f} last={price:.6f} "
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
        assumed_win_rate: float | None = None,
        signal_source: str = "methods",
        potential_sizing: bool = True, sizing: str = "kelly",
        max_margin_pct: float = L.MAX_MARGIN_FRACTION,
        limit_entry: bool = False, slippage: float = 0.0,
        fee_override: float | None = None,
        book_file: str = "book.json",
        book_anywhere: bool = False, trust_book: bool = False,
        earn_stake: bool = False, stake_curve: str = "linear",
        share_stakes: bool = False, mtf_gate: str = "band",
        probe_pct: float = 2.0, probe_n: int = 30,
        probe_budget: float = 0.05) -> None:
    broker = Broker(client, symbols, equity, max_positions, exit_name,
                    min_votes, fee, max_leverage, margin_pct, max_notional_x,
                    conviction_floor, expectancy_gate, assumed_win_rate,
                    signal_source, book_file, book_anywhere, trust_book,
                    earn_stake, stake_curve, share_stakes,
                    potential_sizing, sizing, max_margin_pct, limit_entry,
                    slippage, fee_override, mtf_gate, probe_pct, probe_n,
                    probe_budget)
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
