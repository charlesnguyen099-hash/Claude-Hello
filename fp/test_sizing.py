"""Checks for the position-sizing floor: 5% of capital available RIGHT
NOW, bumped up to the exchange's own minimum order size when that is
larger, and never rejected just for being small.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_full_bot as B
from fp import costs as C
from fp.live_full import Decision

PASS = 0


def chk(cond, msg):
    global PASS
    if not cond:
        raise AssertionError(msg)
    PASS += 1


def dec(sym="AUSDT", stake=0.05, lev=2.0):
    return Decision(symbol=sym, side=1, potential=1.0, stake=stake,
                    leverage=lev, target=0.02, stop=0.011, trail=0.005,
                    cost=0.0011)


def test_floor_is_relative_and_compounds(monkeypatch=None):
    """The operator's own numbers: $10 -> $0.50 -> (free $9.50) -> $0.475."""
    real = C.min_notional
    C.min_notional = lambda sym, px: 0.0     # isolate the 5% behaviour
    try:
        acct = B.Account(equity=10.0, start=10.0)
        p1 = acct.open_position(dec(), 100.0)
        chk(p1 is not None, "a valid signal must open")
        chk(abs(p1.margin - 0.50) < 1e-9,
            f"5% of $10 free is $0.50, got {p1.margin}")
        chk(abs(acct.free - 9.50) < 1e-9,
            f"free after trade 1 must be $9.50, got {acct.free}")

        p2 = acct.open_position(dec("BUSDT"), 100.0)
        chk(p2 is not None, "the second signal must also open")
        chk(abs(p2.margin - 0.475) < 1e-9,
            f"5% of the NEW $9.50 free is $0.475, got {p2.margin}")
    finally:
        C.min_notional = real


def test_bumped_to_exchange_minimum():
    """A coin whose minimum order value exceeds the 5% figure gets sized
    up to that minimum, not left below what the exchange will accept."""
    real = C.min_notional
    C.min_notional = lambda sym, px: 5.0     # a $5 exchange floor
    try:
        acct = B.Account(equity=10.0, start=10.0)
        d = dec(stake=0.05, lev=10.0)         # naive 5% margin = $0.50
        p = acct.open_position(d, 100.0)
        chk(p is not None, "must still open once bumped")
        # need = min_notional / leverage = 5.0 / 10.0 = $0.50 -- exactly
        # the naive figure here, so try a case where it clearly binds.
        d2 = dec("BUSDT", stake=0.05, lev=2.0)   # naive margin $0.50 (free now $9.5)
        need = 5.0 / 2.0                          # $2.50
        p2 = acct.open_position(d2, 100.0)
        chk(p2 is not None, "must open when free can cover the bump")
        chk(abs(p2.margin - need) < 1e-9,
            f"must be bumped to min_notional/leverage = ${need}, got {p2.margin}")
        chk(abs(p2.notional - 5.0) < 1e-6,
            f"notional at that margin and leverage must clear $5, got {p2.notional}")
    finally:
        C.min_notional = real


def test_rejected_only_when_exchange_minimum_does_not_fit():
    """Small is fine. Only reject when even the venue's own floor
    cannot be covered by what is free."""
    real = C.min_notional
    C.min_notional = lambda sym, px: 5.0
    try:
        acct = B.Account(equity=10.0, start=10.0)
        # Commit almost everything elsewhere first.
        big = dec("BUSDT", stake=0.99, lev=10.0)
        C.min_notional = lambda sym, px: 0.0    # let the big one through cheaply
        acct.open_position(big, 100.0)
        C.min_notional = lambda sym, px: 5.0
        chk(acct.free < 5.0 / 10.0,
            f"expected little free capital left, got {acct.free}")
        before = len(acct.open)
        p = acct.open_position(dec("CUSDT", stake=0.05, lev=10.0), 100.0)
        chk(p is None, "the venue's own minimum did not fit in free capital")
        chk(acct.rejected == 1, "must be counted as rejected")
        chk(len(acct.open) == before, "no phantom position")
    finally:
        C.min_notional = real


def test_never_exceeds_total_equity():
    real = C.min_notional
    C.min_notional = lambda sym, px: 0.0
    try:
        acct = B.Account(equity=10.0, start=10.0)
        opened = 0
        for i in range(30):
            d = dec(f"D{i}USDT", stake=1.0, lev=10.0)
            if acct.open_position(d, 100.0):
                opened += 1
        chk(acct.committed <= acct.equity + 1e-9,
            f"committed {acct.committed} must never exceed equity "
            f"{acct.equity}")
        chk(opened == 1,
            f"a 100%-stake signal leaves no free room for a second, "
            f"got {opened}")
    finally:
        C.min_notional = real


def main():
    for fn in (test_floor_is_relative_and_compounds,
               test_bumped_to_exchange_minimum,
               test_rejected_only_when_exchange_minimum_does_not_fit,
               test_never_exceeds_total_equity):
        fn()
    print(f"all {PASS} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
