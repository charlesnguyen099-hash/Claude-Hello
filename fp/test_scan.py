"""Regression test for the same-bar re-entry bug.

A live session pasted by the operator showed a coin opening, closing on
a trailing stop, and opening again at the identical entry price 18-20
seconds later -- over and over, on some bars a dozen times in a row --
compounding equity from $10 into the thousands and then back down in a
single stop-out. The cause: `scan()`'s exits loop closes a position and
frees the symbol, and the entries loop later in that SAME call (or on
the next call, still looking at the same last-closed bar) re-evaluates
it, because `acct.last_action[s]` is only ever touched by the entries
loop -- which is skipped outright while a position is open -- so it is
still frozen on whichever older bar the position was originally opened
on, not the bar that just closed it. The fix marks the bar in
`acct.last_action` the moment a position closes, so a symbol cannot be
re-entered until a genuinely new bar arrives.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_full_bot as B
from fp.live_full import Decision, FullLogic

PASS = 0


def chk(cond, msg):
    global PASS
    if not cond:
        raise AssertionError(msg)
    PASS += 1


class AlwaysDecides:
    """A stub logic that fires the same decision on every bar -- the
    worst case for the re-entry bug, so the test fails loudly if the
    fix regresses."""

    def __init__(self, dec: Decision):
        self._dec = dec

    def decide(self, d, panel, sym):
        return self._dec


def make_df(n: int, close: float, high: float | None = None,
           low: float | None = None):
    idx = pd.date_range("2026-01-01", periods=n, freq="min")
    high = close if high is None else high
    low = close if low is None else low
    return pd.DataFrame({
        "open": close, "high": high, "low": low, "close": close,
        "volume": 1.0,
    }, index=idx)


def test_no_reentry_within_same_bar():
    """Closing a position must not free it up for a fresh entry on the
    very bar that closed it."""
    sym = "AUSDT"
    dec = Decision(symbol=sym, side=1, potential=100.0, stake=1.0,
                   leverage=1.0, target=0.10, stop=0.02, trail=0.001,
                   cost=0.0)
    acct = B.Account(equity=10.0, start=10.0)

    # An existing position, opened earlier on an OLDER bar (last_bar
    # deliberately older than what this scan will fetch), about to be
    # closed by a trailing-stop move on the bar this scan sees.
    p = B.Position(symbol=sym, dec=dec, entry=100.0, margin=10.0,
                   notional=10.0, opened_at=time.time(),
                   high=100.0, low=100.0,
                   last_bar=pd.Timestamp("2020-01-01"))
    acct.open[sym] = p

    # One bar: a favourable spike to 101.5 (peak +1.5%) that closes
    # back at 101.3 -- comfortably past dec.trail (0.1%) and dec.cost
    # (0), so exit_now must return "trail" on this single bar.
    df = make_df(65, close=101.3, high=101.5, low=100.0)
    reason, val = FullLogic.exit_now(dec, p.entry, 101.5, 100.0, 101.3,
                                     p.peak, p.bars_held + 1)
    chk(reason == "trail", f"test setup must trigger a trail exit, got {reason!r}")

    class OneShotFeed:
        def bars_many(self, symbols):
            return {sym: df}

    logic = AlwaysDecides(dec)
    panel = {}
    opened, closed = B.scan(OneShotFeed(), logic, acct, [sym], panel,
                            reference=[sym])

    chk(closed == 1, f"the stale position must close this scan, got {closed}")
    chk(sym not in acct.open,
        "the symbol must NOT be reopened on the same bar that closed it "
        f"-- got a fresh position: {acct.open.get(sym)}")
    chk(len(acct.closed) == 1, f"exactly one closed trade, got {len(acct.closed)}")
    chk(acct.last_action.get(sym) == df.index[-1],
        "closing a position must stamp last_action so the entries loop "
        "-- this call or the next -- skips the coin until a new bar")


def test_reentry_allowed_on_a_genuinely_new_bar():
    """The fix must not become a permanent lockout: once a NEW bar
    arrives, the coin is tradeable again."""
    sym = "AUSDT"
    dec = Decision(symbol=sym, side=1, potential=100.0, stake=1.0,
                   leverage=1.0, target=0.10, stop=0.02, trail=0.001,
                   cost=0.0)
    acct = B.Account(equity=10.0, start=10.0)
    p = B.Position(symbol=sym, dec=dec, entry=100.0, margin=10.0,
                   notional=10.0, opened_at=time.time(),
                   high=100.0, low=100.0,
                   last_bar=pd.Timestamp("2020-01-01"))
    acct.open[sym] = p

    df1 = make_df(65, close=101.3, high=101.5, low=100.0)
    logic = AlwaysDecides(dec)
    panel = {}

    class Feed:
        def __init__(self, df):
            self.df = df

        def bars_many(self, symbols):
            return {sym: self.df}

    B.scan(Feed(df1), logic, acct, [sym], panel, reference=[sym])
    chk(sym not in acct.open, "closed and must not reopen on the same bar")

    # A genuinely new bar: one more row appended, a new timestamp.
    df2 = pd.concat([df1, make_df(1, close=101.4, high=101.4, low=101.4)])
    df2.index = pd.date_range("2026-01-01", periods=len(df2), freq="min")
    B.scan(Feed(df2), logic, acct, [sym], panel, reference=[sym])
    chk(sym in acct.open,
        "a genuinely new bar must let the coin trade again")


def test_missing_live_gate_refuses_to_trade():
    """A model file saved before live_gate existed must be treated as
    unproven (gate 1.0), never silently as meta['gate'] (~0.0000 by
    construction) -- that fallback is exactly the "every one called at
    gate 0.0000" failure this session traced 17 losing live trades to.
    """
    import numpy as np
    from fp.live_full import FullLogic as FL

    logic = FL.__new__(FL)
    logic.models = {"AUSDT": {}}
    logic.meta = {"AUSDT": {"gate": 0.0, "floor": 0.01,
                            "min_stake": 0.05, "max_stake": 1.0,
                            "lev_floor": 1.0, "liq_safety": 0.6,
                            "cost": 0.001}}
    # No "live_gate" key at all -- the old-format case.
    row = np.zeros((1, 4), dtype="float32")

    called = {}

    def fake_call(models, X):
        called["hit"] = True
        return (np.array([1], dtype="int8"), np.array([0.9]),
                np.array([0.05]), np.array([0.01]))

    module_globals = FL.decide.__globals__
    real_call = module_globals["FU"].call
    real_dir = module_globals["DIR"]

    class FakeDIR:
        @staticmethod
        def features(d, panel, sym):
            return pd.DataFrame(np.zeros((len(d), 4)))

    module_globals["FU"].call = fake_call
    module_globals["DIR"] = FakeDIR
    try:
        import pandas as pd
        d = pd.DataFrame({"high": [1.0] * 5, "low": [1.0] * 5,
                          "close": [1.0] * 5, "open": [1.0] * 5})
        dec = logic.decide(d, {}, "AUSDT")
    finally:
        module_globals["FU"].call = real_call
        module_globals["DIR"] = real_dir

    chk(called.get("hit") is True, "the stub model must have been scored")
    chk(dec is None,
        "confidence 0.90 against a MISSING live_gate must NOT trade -- "
        f"got {dec}")


def test_entries_allocate_highest_potential_first():
    """When several coins signal in the same cycle and the shared pool
    cannot cover them all, the strongest signal must claim capital
    first -- not whichever coin happened to be scanned first. A live
    session showed committed sitting at $9.995/$10 for hours: the
    scan-order coins (by turnover rank) exhausted the pool before
    later, possibly stronger, signals were even granted a chance."""
    sym_weak, sym_strong, sym_mid = "AUSDT", "BUSDT", "CUSDT"

    def dec_for(sym, potential, stake):
        return Decision(symbol=sym, side=1, potential=potential,
                        stake=stake, leverage=1.0, target=0.10,
                        stop=0.02, trail=0.01, cost=0.0)

    # Weak signal scanned FIRST (as if ranked ahead by 24h turnover),
    # asking for nearly the whole pool. Strong signal scanned LAST,
    # asking for a modest slice. Priced so the weak one alone would
    # exhaust free capital if entries were granted in scan order.
    decisions = {
        sym_weak: dec_for(sym_weak, potential=10.0, stake=0.95),
        sym_mid: dec_for(sym_mid, potential=50.0, stake=0.50),
        sym_strong: dec_for(sym_strong, potential=90.0, stake=0.50),
    }

    class ByCoin:
        def decide(self, d, panel, sym):
            return decisions.get(sym)

    df = make_df(65, close=100.0)
    panel = {}

    class Feed:
        def bars_many(self, symbols):
            return {s: df for s in symbols}

    acct = B.Account(equity=10.0, start=10.0)
    # Scan order deliberately puts the WEAK signal first.
    order = [sym_weak, sym_mid, sym_strong]
    B.scan(Feed(), ByCoin(), acct, order, panel, reference=order)

    chk(sym_strong in acct.open,
        "the highest-potential signal must be granted capital even "
        "though it was scanned last")
    chk(sym_weak not in acct.open or acct.open[sym_weak].margin < 9.5,
        "the weakest signal must not be free to exhaust the whole "
        "pool ahead of stronger ones")
    strong_margin = acct.open[sym_strong].margin
    chk(strong_margin >= 0.5 * 9.0 - 1e-6,
        f"the strong signal should have received close to its full "
        f"50% stake off a near-full pool, got margin {strong_margin}")


def test_retrain_continuous_runs_back_to_back():
    """--retrain-continuous must not wait between cycles -- several
    cycles should run within a fraction of a second, proving the
    thread never blocks on the --retrain-hours clock. Non-continuous
    mode, given an enormous hours value, must not run a cycle at all
    in that same window -- the contrast proves continuous genuinely
    bypasses the wait rather than the stub just being fast."""
    import threading

    calls = {"n": 0}

    def fake_cycle(symbols, log=print):
        calls["n"] += 1
        return False   # "failed" -- keeps the worker from touching
                        # FU.MODELS / FullLogic, irrelevant to this test

    real_cycle = B.RT.cycle
    B.RT.cycle = fake_cycle
    try:
        holder = B.LogicHolder(None)
        stop = threading.Event()
        t = threading.Thread(target=B.retrain_worker,
                             args=(holder, None, 24.0, stop),
                             kwargs={"continuous": True}, daemon=True)
        t.start()
        time.sleep(0.2)
        stop.set()
        t.join(timeout=2.0)
        chk(not t.is_alive(), "continuous worker must stop promptly once "
            "stop_event is set")
        chk(calls["n"] >= 3,
            f"continuous mode must run several cycles with no wait "
            f"between them in 0.2s, got {calls['n']}")
    finally:
        B.RT.cycle = real_cycle

    calls["n"] = 0
    B.RT.cycle = fake_cycle
    try:
        stop = threading.Event()
        t = threading.Thread(target=B.retrain_worker,
                             args=(holder, None, 999999.0, stop),
                             kwargs={"continuous": False}, daemon=True)
        t.start()
        time.sleep(0.2)
        stop.set()
        t.join(timeout=2.0)
        chk(calls["n"] == 0,
            f"non-continuous mode with a huge --retrain-hours must not "
            f"run a cycle while still waiting, got {calls['n']}")
    finally:
        B.RT.cycle = real_cycle


class FakeBroker:
    """A stand-in LiveBroker: records every order, never touches a
    network. `fail_open`/`fail_close` make market_order raise, the
    same shape a real OrderError would take."""

    def __init__(self, fail_open=False, fail_close=False):
        self.fail_open = fail_open
        self.fail_close = fail_close
        self.orders = []
        self._position = None

    def market_order(self, symbol, side, qty, reduce_only=False):
        if reduce_only and self.fail_close:
            raise RuntimeError("simulated close failure")
        if not reduce_only and self.fail_open:
            raise RuntimeError("simulated open failure")
        self.orders.append((symbol, side, qty, reduce_only))
        if reduce_only:
            self._position = None
        else:
            self._position = {"side": side, "qty": qty, "entry": 100.0}
        return {}

    def open_positions(self):
        return {"AUSDT": self._position} if self._position else {}


def test_real_trade_open_places_order_and_uses_real_fill():
    """A successful real order must set Position.qty/entry from what
    the broker reports back, not from the simulated price alone."""
    sym = "AUSDT"
    dec = Decision(symbol=sym, side=1, potential=50.0, stake=0.5,
                   leverage=2.0, target=0.10, stop=0.02, trail=0.001,
                   cost=0.0)
    broker = FakeBroker()
    acct = B.Account(equity=10.0, start=10.0, broker=broker)
    p = acct.open_position(dec, price=99.0)
    chk(p is not None, "a successful real order must open a position")
    chk(len(broker.orders) == 1, f"exactly one order, got {broker.orders}")
    sym_o, side_o, qty_o, reduce_o = broker.orders[0]
    chk(sym_o == sym and side_o == 1 and not reduce_o,
        f"open order must be a non-reduce-only buy for {sym}, "
        f"got {broker.orders[0]}")
    chk(p.entry == 100.0,
        f"entry must come from the broker's reported fill (100.0), "
        f"not the simulated price (99.0), got {p.entry}")
    chk(p.qty == qty_o > 0, "position qty must match the real order qty")


def test_real_trade_open_order_failure_creates_no_position():
    """A rejected/errored real order must leave no internal position
    behind -- an internal record of a trade the exchange never made."""
    sym = "AUSDT"
    dec = Decision(symbol=sym, side=1, potential=50.0, stake=0.5,
                   leverage=2.0, target=0.10, stop=0.02, trail=0.001,
                   cost=0.0)
    broker = FakeBroker(fail_open=True)
    acct = B.Account(equity=10.0, start=10.0, broker=broker)
    p = acct.open_position(dec, price=99.0)
    chk(p is None, "a failed real order must not open a position")
    chk(sym not in acct.open, "no position may be tracked internally")
    chk(acct.rejected == 1, f"the attempt must count as rejected, "
        f"got {acct.rejected}")


def test_real_trade_close_order_failure_keeps_position_open():
    """A failed close order must leave the position exactly as it was
    -- still open internally, so the next scan retries it -- rather
    than the bot believing a still-live position is flat."""
    sym = "AUSDT"
    dec = Decision(symbol=sym, side=1, potential=50.0, stake=0.5,
                   leverage=2.0, target=0.10, stop=0.02, trail=0.001,
                   cost=0.0)
    broker = FakeBroker(fail_close=True)
    acct = B.Account(equity=10.0, start=10.0, broker=broker)
    p = acct.open_position(dec, price=99.0)
    chk(p is not None, "the open must succeed for this test to be valid")
    result = acct.close_position(p, "target", 0.05)
    chk(result is None, "a failed close order must return None, not "
        "a Closed record")
    chk(sym in acct.open, "the position must remain in acct.open after "
        "a failed close order")
    chk(len(acct.closed) == 0, "nothing may be appended to acct.closed "
        "when the real close order failed")


def test_real_trade_close_places_reduce_only_order():
    """A successful close must place a reduce-only order on the
    OPPOSITE side, for the exact quantity that was opened."""
    sym = "AUSDT"
    dec = Decision(symbol=sym, side=1, potential=50.0, stake=0.5,
                   leverage=2.0, target=0.10, stop=0.02, trail=0.001,
                   cost=0.0)
    broker = FakeBroker()
    acct = B.Account(equity=10.0, start=10.0, broker=broker)
    p = acct.open_position(dec, price=99.0)
    opened_qty = p.qty
    result = acct.close_position(p, "target", 0.05)
    chk(result is not None, "a successful close must return a Closed record")
    chk(len(broker.orders) == 2, f"open + close, got {broker.orders}")
    sym_o, side_o, qty_o, reduce_o = broker.orders[1]
    chk(reduce_o, "the close order must be reduce-only")
    chk(side_o == -dec.side, f"the close order must be the opposite "
        f"side of the position, got {side_o}")
    chk(qty_o == opened_qty, f"the close order must send back the EXACT "
        f"quantity that was opened ({opened_qty}), got {qty_o}")
    chk(sym not in acct.open, "the position must be gone after a "
        "successful close")


def main():
    for fn in (test_no_reentry_within_same_bar,
               test_reentry_allowed_on_a_genuinely_new_bar,
               test_missing_live_gate_refuses_to_trade,
               test_entries_allocate_highest_potential_first,
               test_retrain_continuous_runs_back_to_back,
               test_real_trade_open_places_order_and_uses_real_fill,
               test_real_trade_open_order_failure_creates_no_position,
               test_real_trade_close_order_failure_keeps_position_open,
               test_real_trade_close_places_reduce_only_order):
        fn()
    print(f"all {PASS} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
