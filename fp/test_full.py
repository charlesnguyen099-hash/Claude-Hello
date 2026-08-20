"""Checks for the full logic: ranges, opportunities, sizing, the gate.

The interesting ones are the range structures. `opportunities` rests on
a sparse table answering "the best price between here and a bar that is
different for every entry", and a wrong answer there would not crash --
it would quietly invent profit. So they are checked against the slow,
obviously-correct version on random data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fp import full as FU

PASS = 0


def chk(cond, msg):
    global PASS
    if not cond:
        raise AssertionError(msg)
    PASS += 1


def ramp(up=True, n=400):
    px = np.concatenate([np.full(100, 100.0), np.linspace(100, 105, 100),
                         np.full(n - 200, 105.0)])
    if not up:
        px = px[::-1].copy()
    return pd.DataFrame(
        {"open": px, "high": px, "low": px, "close": px,
         "volume": np.ones(n), "turnover": np.ones(n)},
        index=pd.date_range("2026-01-01", periods=n, freq="min"))


def test_ranges():
    rng = np.random.default_rng(0)
    a = rng.normal(size=500).cumsum() + 100
    for want_max in (True, False):
        tab = FU.build_table(a, 64, want_max)
        lo = rng.integers(0, 450, 200)
        hi = lo + rng.integers(0, 60, 200)
        got = FU.query(tab, lo, hi, want_max)
        exp = np.array([(a[l:h + 1].max() if want_max else a[l:h + 1].min())
                        for l, h in zip(lo, hi)])
        chk(np.allclose(got, exp), f"range max={want_max} disagrees with numpy")


def test_first_cross():
    b = np.array([10., 10, 10, 9, 10, 8, 10, 10.])
    tab = FU.build_table(b, 8, want_max=False)
    thr = np.full(len(b), 9.5)
    fc = FU.first_cross(tab, thr, 4, below=True)
    chk(fc[0] == 3, f"first crossing below 9.5 is bar 3, got {fc[0]}")
    chk(fc[4] == 5, f"from bar 4 the crossing is bar 5, got {fc[4]}")
    chk(fc[6] > len(b), "no crossing must return a sentinel past the end")


def test_opportunities():
    n = 400
    d = ramp(up=True, n=n)
    cost = np.full(n, 0.0011)
    side, profit, mae, bars = FU.opportunities(d, cost, horizon=300)
    chk(side[100] == 1, f"a rising ramp is a long, got {side[100]}")
    chk(profit[100] > 0.04, f"a 5% ramp nets over 4%, got {profit[100]:.4f}")
    chk(mae[100] == 0.0, "a monotone ramp has no drawdown")
    chk(side[350] == 0, "a flat tail offers nothing")
    chk(bars[100] > 0, "an opportunity takes time to play out")

    s2, _, _, _ = FU.opportunities(ramp(up=False, n=n), cost, horizon=300)
    chk(s2[100] == -1, f"a falling ramp is a short, got {s2[100]}")


def test_cost_bites():
    n = 400
    d = ramp(up=True, n=n)
    _, base, _, _ = FU.opportunities(d, np.full(n, 0.0011), horizon=300)
    _, free, _, _ = FU.opportunities(d, np.full(n, 0.0), horizon=300)
    _, dear, _, _ = FU.opportunities(d, np.full(n, 0.06), horizon=300)
    chk(free[100] > base[100] > 0, "a higher fee must lower the net")
    chk(abs((free[100] - base[100]) - 0.0011) < 1e-9,
        "the fee must come off the net exactly once")
    chk(dear[100] == 0.0, "a fee above the move kills the opportunity")


def test_potential():
    sc = FU.potential(np.array([0.5, 1.0]), np.array([0.01, 0.20]),
                      np.array([0.0, 0.0]), np.array([0.0011, 0.0011]))
    chk(sc[1] > sc[0], "a bigger, surer move must score higher")
    chk(sc.min() >= 0 and sc.max() <= 100, "potential lives in 0..100")


def test_gate():
    g, ne = FU.calibrate_gate(np.array([1, 1, -1]), np.array([.9, .7, .8]),
                              np.array([1, 1, 1]))
    chk(abs(g - 0.8) < 1e-9, f"the gate is the worst mistake, got {g}")
    chk(ne == 1, "one false positive")
    g0, n0 = FU.calibrate_gate(np.array([1, 1]), np.array([.9, .7]),
                               np.array([1, 1]))
    chk(g0 == 0.0 and n0 == 0, "no mistakes means no gate")


def test_rotation():
    """Top stays every cycle; the tail is covered, never dropped."""
    from fp.universe import Rotation
    syms = [f"C{i}" for i in range(130)]
    r = Rotation(syms, top=50, slice_size=20)
    chk(len(r) == 130, "rotation must account for every symbol")
    chk(r.cycles_for_full_sweep == 4, f"80 tail / 20 = 4 cycles, "
        f"got {r.cycles_for_full_sweep}")
    seen = set()
    for _ in range(r.cycles_for_full_sweep):
        b = r.next_batch()
        chk(all(q in b for q in syms[:50]), "the top must be in every batch")
        chk(len(b) == len(set(b)), "a batch must not repeat a symbol")
        seen |= set(b)
    chk(seen == set(syms), f"a full sweep must reach every coin, "
        f"missed {len(set(syms) - seen)}")


def test_ensemble_recognition():
    """Whichever logic recognises the bar trades it; a dead heat does not."""
    from fp.live_full import FullLogic

    class Fake:
        def __init__(self, side, conf, profit, mae):
            self.v = (side, conf, profit, mae)

    def fake_call(m, row):
        s, c, p, a = m.v
        return (np.array([s]), np.array([c]), np.array([p]), np.array([a]))

    import fp.full as _FU
    real, _FU.call = _FU.call, fake_call
    lg = FullLogic.__new__(FullLogic)
    # live_gate is what ensemble_call actually compares confidence
    # against (see fp/live_full.py); every real model file has one, so
    # the test's synthetic meta carries one too rather than exercising
    # the "no live_gate at all" fallback, which fp/test_scan.py covers
    # on its own.
    meta = {"gate": 0.5, "live_gate": 0.5, "floor": 0.01}
    try:
        # One model recognises its setup, the others see nothing. The
        # old unanimity rule silenced this; recognition trades it.
        lg.models = {"A": Fake(1, 0.95, 0.05, 0.01),
                     "B": Fake(0, 0.10, 0.00, 0.00),
                     "C": Fake(0, 0.20, 0.00, 0.00)}
        lg.meta = {k: dict(meta) for k in lg.models}
        side, conf, profit, mae, gate, who = lg.ensemble_call(None, "X")
        chk(side == 1, "a lone recognised signal must trade")
        chk(abs(profit - 0.05) < 1e-9, "the recogniser sets the target")

        # Below its own gate is not recognition.
        lg.models = {"A": Fake(1, 0.40, 0.05, 0.01)}
        lg.meta = {"A": dict(meta)}
        side, *_ = lg.ensemble_call(None, "X")
        chk(side == 0, "confidence under the gate is not a signal")

        # Below the floor is not a trade either.
        lg.models = {"A": Fake(1, 0.95, 0.002, 0.01)}
        lg.meta = {"A": dict(meta)}
        side, *_ = lg.ensemble_call(None, "X")
        chk(side == 0, "a move under the floor is not a trade")

        # The most confident recogniser wins a disagreement.
        lg.models = {"A": Fake(1, 0.95, 0.05, 0.01),
                     "B": Fake(-1, 0.70, 0.09, 0.03)}
        lg.meta = {k: dict(meta) for k in lg.models}
        side, conf, profit, mae, gate, who = lg.ensemble_call(None, "X")
        chk(side == 1, "the stronger signal names the side")
        chk(abs(conf - 0.95) < 1e-9, "and its confidence is the one used")
        chk(who == "A", f"and the trade is attributed to it, got {who}")

        # An exact dead heat pointing both ways is a coin flip, not a signal.
        lg.models = {"A": Fake(1, 0.90, 0.05, 0.01),
                     "B": Fake(-1, 0.90, 0.05, 0.01)}
        lg.meta = {k: dict(meta) for k in lg.models}
        side, _, _, _, gate, _ = lg.ensemble_call(None, "X")
        chk(side == 0, "a dead heat both ways must not trade")
        chk(gate == 1.0, "and its gate must admit nothing")
    finally:
        _FU.call = real


def main():
    for fn in (test_ranges, test_first_cross, test_opportunities,
               test_cost_bites, test_potential, test_gate,
               test_rotation, test_ensemble_recognition):
        fn()
    print("=" * 70)
    print(f"all {PASS} checks passed")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
