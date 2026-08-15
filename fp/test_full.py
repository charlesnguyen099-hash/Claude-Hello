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


def main():
    for fn in (test_ranges, test_first_cross, test_opportunities,
               test_cost_bites, test_potential, test_gate):
        fn()
    print("=" * 70)
    print(f"all {PASS} checks passed")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
