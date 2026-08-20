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


def main():
    for fn in (test_no_reentry_within_same_bar,
               test_reentry_allowed_on_a_genuinely_new_bar,
               test_missing_live_gate_refuses_to_trade):
        fn()
    print(f"all {PASS} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
