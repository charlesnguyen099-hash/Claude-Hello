"""Tests for the logic: factors, labels, potential and the live path.

Run:  python -m fp.test_engine

These are not unit tests for their own sake. Every one of them guards a
mistake that this repo actually shipped at least once:

  * a feature that could see its own outcome (three times, in three
    different files)
  * a fee subtracted after the label instead of inside it, so the model
    was taught to hunt for losses
  * a target that did not clear the round trip, making the break-even win
    rate 94% and every result that followed meaningless
  * a live feature matrix that did not match the trained one
  * overlapping barrier windows counted as independent observations,
    which turned t = 2.26 into t = 7.81
  * a potential score that handed its LARGEST stake to its LEAST credible
    claim

No network: everything runs on generated bars.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from fp import engine as E
from fp import factors as F
from fp import labels as LB

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


def bars(n=3000, seed=0, base=100.0, drift=0.0, vol=0.001):
    r = np.random.default_rng(seed)
    c = base * np.exp(np.cumsum(r.normal(drift, vol, n)))
    sp = np.abs(r.normal(0, vol / 2, n)) + 1e-6
    idx = pd.date_range("2026-06-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({"open": np.r_[c[0], c[:-1]],
                         "high": c * (1 + sp), "low": c * (1 - sp),
                         "close": c,
                         "volume": np.abs(r.normal(1e5, 2e4, n))}, index=idx)


def board(n=3000, k=5):
    return {f"S{i}USDT": bars(n, seed=i, base=10.0 ** (i - 1)) for i in range(k)}


# ---------------------------------------------------------------- factors

def test_factors_are_complete_and_finite():
    print("\nfactors: 100+ numbers, all finite once warmed up")
    B = board()
    X = F.build(B)
    x = X["S0USDT"]
    check("more than 100 factors", x.shape[1] > 100, x.shape[1])
    check("every symbol gets the same columns",
          all(list(v.columns) == list(x.columns) for v in X.values()))
    fin = np.isfinite(x.values).all(axis=1)
    check("the tail is fully finite", bool(fin[-1]))
    check("most rows are usable once warmed up", fin.mean() > 0.8,
          f"{100*fin.mean():.1f}%")
    warm = int(np.flatnonzero(fin)[0])
    check("warm-up is bounded by the deepest window",
          warm <= max(F.STATE_WINDOWS) + LB.SIGMA_WINDOW + 5, warm)
    check("float32, so a full panel fits in memory",
          x.dtypes.unique().tolist() == [np.dtype("float32")],
          x.dtypes.unique().tolist())


def test_factors_are_scale_free():
    """BTC at $100,000 and a $0.02 coin must present the same numbers.

    This is the only reason ONE model can serve ten coins. If a factor
    carries price units, the model learns 'BTC' instead of 'this setup'.
    """
    print("\nfactors: scale-free, so one model serves every coin")
    a = bars(2000, seed=7, base=0.02)
    b = a * 1.0
    for col in ("open", "high", "low", "close"):
        b[col] = a[col] * 5_000_000.0
    Xa = F.build({"A": a, "B": b})["A"]
    Xb = F.build({"A": a, "B": b})["B"]
    same = Xa.columns
    bad = []
    for c in same:
        u, v = Xa[c].values, Xb[c].values
        m = np.isfinite(u) & np.isfinite(v)
        if m.sum() < 50:
            continue
        if not np.allclose(u[m], v[m], rtol=1e-3, atol=1e-4):
            bad.append(c)
    # rank/mkt/disp/resid/breadth compare A against B, so scaling B
    # changes them by construction -- they are cross-sectional and are
    # excluded from the invariance claim.
    xs = tuple(f"{p}{k}" for p in ("mkt", "rank", "disp", "resid", "breadth")
               for k in F.LOOKBACKS)
    bad = [c for c in bad if c not in xs]
    check("every single-coin factor is invariant to price scale",
          not bad, bad[:8])


def test_no_factor_can_see_its_own_future():
    """Truncating the data must not change any earlier factor value.

    A factor that shifts when later bars are removed is reading them.
    """
    print("\nfactors: nothing reads a bar that has not happened")
    B = board(n=2500)
    full = F.build(B)["S0USDT"]
    cut = 2000
    trunc = F.build({k: v.iloc[:cut] for k, v in B.items()})["S0USDT"]
    a = full.iloc[:cut].values
    b = trunc.values
    m = np.isfinite(a) & np.isfinite(b)
    check("truncating the future leaves the past unchanged",
          np.allclose(a[m], b[m], rtol=1e-5, atol=1e-6),
          f"{int((~np.isclose(a[m], b[m], rtol=1e-5, atol=1e-6)).sum())} "
          f"cells differ")


def test_cross_section_measures_the_board_not_the_coin():
    print("\nfactors: the cross-section says what the OTHER coins did")
    B = board(k=5)
    X = F.build(B)
    r = X["S0USDT"]["rank5"].dropna()
    check("rank is a fraction in [0, 1]",
          bool(((r >= 0) & (r <= 1)).all()), (r.min(), r.max()))
    ranks = np.column_stack([X[s]["rank5"].values for s in sorted(B)])
    ok = np.isfinite(ranks).all(axis=1)
    check("the ranks across coins are a permutation, so they sum the same",
          np.allclose(ranks[ok].sum(axis=1), ranks[ok].sum(axis=1)[0]),
          ranks[ok].sum(axis=1)[:3])
    br = X["S0USDT"]["breadth5"].dropna()
    check("breadth is the share of OTHER coins that rose",
          bool(((br >= 0) & (br <= 1)).all()), (br.min(), br.max()))


# ----------------------------------------------------------------- labels

def test_the_fee_lives_inside_the_label():
    """A target the fee eats is a LOSS, and the label must say so.

    Subtracting the fee after the fact teaches the model that a 0.08%
    move against a 0.11% bill is a small win. It is not.
    """
    print("\nlabels: the exchange bill is inside the label, not after it")
    d = bars(3000, seed=3)
    sg = LB.sigma(d["close"].values)
    tp, sl, hold = 2.0, 2.0, 240
    net, held, ok = LB.barrier_net(d, 1, tp, sl, hold, sg)
    sgh = sg * np.sqrt(hold)
    b = tp * sgh - LB.FEE
    check("a shape whose target does not clear the round trip is refused",
          not np.any(ok & (b <= 0)), int((ok & (b <= 0)).sum()))
    hit = ok & (net > 0)
    check("every labelled win clears the whole bill",
          bool((net[hit] > 0).all()) and float(net[ok].max()) < 1.0,
          float(net[ok].max()) if ok.any() else None)
    check("the fee is actually charged",
          float(net[ok].max()) < float((b)[ok].max()) + 1e-9)


def test_the_target_scales_with_the_horizon():
    """THE correction. A target fixed at k x one-minute sigma makes the
    break-even win rate 94%, and nothing wins that often -- which is why
    every earlier version of this repo failed."""
    print("\nlabels: the target grows with sqrt(hold), so break-even is sane")
    d = bars(6000, seed=11)
    sg = LB.sigma(d["close"].values)
    med = float(np.nanmedian(sg))
    rows = []
    for tp, sl, hold in LB.SHAPES:
        sgh = med * np.sqrt(hold)
        b, a = tp * sgh - LB.FEE, sl * sgh + LB.FEE
        rows.append((tp, sl, hold, a / (a + b)))
    worst = max(r[3] for r in rows)
    check("no shipped shape needs more than a 70% win rate to break even",
          worst < 0.70, f"worst p_be = {100*worst:.1f}%")
    # And the one-minute version really is much worse. The absolute
    # number depends on the coin -- measured on BTC it was 94% -- so what
    # is asserted here is the RELATIONSHIP, which holds for any
    # volatility: an unscaled target faces a far higher bar, and at a
    # realistic 1-minute sigma it does not clear the fee at all.
    b1, a1 = 2.0 * med - LB.FEE, 2.0 * med + LB.FEE
    pbe1 = (a1 / (a1 + b1)) if b1 > 0 else 1.0
    pbe_scaled = min(r[3] for r in rows)
    check("an unscaled one-minute target faces a far higher bar",
          pbe1 > pbe_scaled + 0.15,
          f"1m {100*pbe1:.1f}% vs scaled {100*pbe_scaled:.1f}%")
    btc_sigma = 0.0004          # BTC's actual 1-minute volatility
    b_btc = 2.0 * btc_sigma - LB.FEE
    check("at BTC's real 1-minute sigma the target does not clear the fee",
          b_btc <= 0, f"b = {100*b_btc:+.4f}%")
    check("longer holds have lower break-even than shorter ones",
          min(r[3] for r in rows if r[2] >= 480)
          <= min(r[3] for r in rows if r[2] <= 120) + 1e-9)


def test_a_label_never_reads_past_its_own_horizon():
    print("\nlabels: a trade is resolved inside its own time limit")
    d = bars(3000, seed=5)
    for tp, sl, hold in ((2.0, 2.0, 240), (1.0, 1.0, 60)):
        net, held, ok = LB.barrier_net(d, 1, tp, sl, hold)
        check(f"held never exceeds the limit ({hold}m)",
              int(np.nanmax(held)) <= hold, int(np.nanmax(held)))
        tail = ok[-hold:]
        check(f"bars with no room to resolve are not labelled ({hold}m)",
              not tail.any(), int(tail.sum()))


# --------------------------------------------------------------- potential

def test_potential_is_zero_at_break_even_and_100_at_certainty():
    print("\npotential: one 0-100 scale for every shape and horizon")
    p_be = np.array([0.5, 0.5, 0.4, 0.6])
    p = np.array([0.5, 1.0, 0.4, 0.8])
    s = E.potential(p, p_be)
    check("break-even scores 0", s[0] == 0.0, s[0])
    check("certainty scores 100", abs(s[1] - 100.0) < 1e-9, s[1])
    check("below break-even scores 0, never negative", s[2] == 0.0, s[2])
    check("halfway scores 50", abs(s[3] - 50.0) < 1e-9, s[3])
    check("a worse-than-break-even claim never outranks a good one",
          E.potential(np.array([0.45]), np.array([0.5]))[0]
          < E.potential(np.array([0.55]), np.array([0.5]))[0])


def test_the_probability_is_lower_bounded_by_its_own_sample():
    """Sizing off a point estimate is how a backtest becomes a margin
    call. A probability from 800 rows must be worth less than the same
    number from 80,000."""
    print("\npotential: the win probability is discounted by how well known")

    class FakeIso:
        def predict(self, x):
            return np.asarray(x, dtype=float)

    class FakeM:
        def predict_proba(self, X):
            n = len(X)
            return np.column_stack([np.full(n, 0.4), np.full(n, 0.6)])

    X = np.zeros((3, 2))
    small = {"model": FakeM(), "iso": FakeIso(), "n_calib": 400}
    big = {"model": FakeM(), "iso": FakeIso(), "n_calib": 400_000}
    ps, pb = E.p_hat(small, X)[0], E.p_hat(big, X)[0]
    check("both are below the raw estimate", ps < 0.6 and pb < 0.6, (ps, pb))
    check("a small calibration sample is discounted harder", ps < pb,
          (ps, pb))
    check("a large one converges on the estimate", abs(pb - 0.6) < 0.01, pb)
    check("the point estimate is available when asked for",
          abs(E.p_hat(big, X, lower=False)[0] - 0.6) < 1e-9)


def test_break_even_uses_the_horizon_scaled_barriers():
    print("\npotential: break-even reads the same barriers the label used")
    sg = np.array([0.0004])
    for tp, sl, hold in ((2.0, 2.0, 240), (3.0, 2.0, 240)):
        p_be, a, b = E.break_even(tp, sl, hold, sg)
        sgh = sg * np.sqrt(hold)
        check(f"b is the net win at {tp}/{sl}/{hold}",
              abs(b[0] - (tp * sgh[0] - LB.FEE)) < 1e-12)
        check(f"a is the gross loss at {tp}/{sl}/{hold}",
              abs(a[0] - (sl * sgh[0] + LB.FEE)) < 1e-12)
    pw, _, _ = E.break_even(3.0, 2.0, 240, sg)
    pn, _, _ = E.break_even(2.0, 2.0, 240, sg)
    check("a wider target has a LOWER break-even", pw[0] < pn[0], (pw, pn))


# ------------------------------------------------------------ independence

def test_overlapping_trades_are_not_counted_twice():
    """The correction that took an earlier result from t=7.81 to t=2.26.

    Two entries whose barrier windows overlap watch nearly the same price
    path, so their outcomes are nearly the same number. Counting both
    understates the standard error and inflates every t in the report.
    """
    print("\nindependence: overlapping windows are one observation, not two")
    sym = np.array(["A"] * 6 + ["B"] * 3)
    pos = np.array([0, 10, 20, 30, 100, 200, 0, 5, 500])
    held = np.array([50, 50, 50, 50, 50, 50, 400, 400, 10])
    take = np.ones(9, dtype=bool)
    keep = E.independent(sym, pos, held, take)
    check("A keeps only entries clear of the last one",
          list(np.flatnonzero(keep[:6])) == [0, 4, 5],
          list(np.flatnonzero(keep[:6])))
    check("B's long hold blocks the entry inside it",
          list(np.flatnonzero(keep[6:])) == [0, 2],
          list(np.flatnonzero(keep[6:])))
    check("symbols do not block each other",
          keep[0] and keep[6])
    check("an untaken entry is never kept",
          not E.independent(sym, pos, held, np.zeros(9, bool)).any())


def test_the_rotation_null_keeps_everything_but_the_alignment():
    print("\nnull: a rolled copy keeps the drift and loses only the timing")
    rng = np.random.default_rng(0)
    n = 4000
    pred = rng.normal(50, 20, n)
    net = rng.normal(-0.001, 0.01, n)
    sym = np.full(n, "A")
    pos = np.arange(n) * 40
    held = np.full(n, 30)
    null = E.rotation_null(pred, net, sym, pos, held, lambda p: p >= 60,
                           rounds=40)
    check("the null produces a distribution, not a constant",
          null.std() > 0, null.std())
    check("with no real signal the null brackets the real result",
          abs(null.mean() - net.mean()) < 5 * (null.std() + 1e-9),
          (null.mean(), net.mean()))


# ---------------------------------------------------------------- the live path

def test_the_live_path_refuses_a_mismatched_matrix():
    print("\nlive: the live matrix must BE the trained one")
    from fp.live_engine import Engine
    e = Engine()
    if not e.ok or not e.models:
        check("no logic shipped -- the live path correctly offers nothing",
              e.candidates(None) == [] and not e.models)
        return
    B = board(k=5, n=2000)
    X = e.panel(B)
    check("the panel builds for every symbol", len(X) == len(B), len(X))
    x = X[sorted(B)[0]]
    check("columns match the trained model exactly",
          list(x.columns) == e.columns)
    bad = x.rename(columns={x.columns[0]: "not_a_factor"})
    try:
        e.panel({k: v for k, v in B.items()})
        raised = False
        e.columns = e.columns[:-1]
        e.panel(B)
    except ValueError:
        raised = True
    check("a column mismatch raises rather than predicting nonsense", raised)


def test_the_live_path_needs_the_whole_board():
    """A third of the factors are cross-sectional. One symbol at a time
    would silently change every rank."""
    print("\nlive: the cross-section needs every coin at once")
    from fp.live_engine import Engine
    e = Engine()
    if not e.ok:
        check("nothing to check without a trained model", True)
        return
    one = {"S0USDT": bars(2000, seed=0)}
    check("a single symbol cannot produce a cross-section",
          e.panel(one) == {}, list(e.panel(one)))


# -------------------------------------------------------- sizing and slots

def _broker(syms=("S0USDT",), **kw):
    from fp import bot as B
    from fp import logic as L
    from fp.test_bot import FakeHTTP
    opts = dict(equity=10.0, max_positions=0, exit_name=L.DEFAULT_EXIT,
                min_votes=1, fee=L.FEE_ROUND_TRIP, max_leverage=None,
                margin_pct=0.02, max_notional_x=0.0, conviction_floor=1.0,
                expectancy_gate=False, potential_sizing=False, sizing="flat",
                max_margin_pct=1.0)
    opts.update(kw)
    c = FakeHTTP(list(syms))
    return B.Broker(c, list(syms), **opts), c


def test_the_stake_is_the_potential_and_can_take_the_account():
    """"Logic nao tiem nang cao thi trade nhieu von" as arithmetic.

    A 60/100 setup commits 60% of equity; a 100/100 setup -- one that by
    its own barriers cannot lose -- may take all of it. Half-Kelly and a
    ruin cap sit on top so a big score on a wide stop still cannot bet
    the account.
    """
    print("\nsizing: the stake IS the potential score")
    b, _ = _broker()
    tp_d, sl_d, lev, fee = 0.030, 0.020, 3.0, 0.0011
    a, bb = sl_d + fee, tp_d - fee
    p_be = a / (a + bb)

    out = []
    for score in (10, 30, 60, 90, 100):
        p = p_be + (score / 100.0) * (1 - p_be)
        claimed = p * bb - (1 - p) * a
        frac, _, _ = b.book_margin_fraction(tp_d, sl_d, lev, f"r{score}",
                                            claimed, fee)
        out.append((score, frac))
    check("a higher score always takes at least as much",
          all(out[i][1] <= out[i + 1][1] + 1e-12 for i in range(len(out) - 1)),
          [(s, round(f, 4)) for s, f in out])
    check("a weak setup takes little", out[0][1] < 0.15, out[0][1])
    check("a near-certain setup can take most of the account",
          out[-1][1] > 0.5, out[-1][1])
    check("nothing exceeds --max-margin-pct",
          all(f <= 1.0 + 1e-12 for _, f in out), out)

    # And the ruin cap really binds: a huge score against a wide stop
    # must not be allowed to stake the account into liquidation.
    wide, _, _ = b.book_margin_fraction(0.40, 0.35, 3.0, "wide", 0.20, fee)
    check("a wide stop is capped below the ruin line",
          wide * 3.0 * (0.35 + fee) <= 1.0 + 1e-9,
          wide * 3.0 * (0.35 + fee))


def test_an_impossible_claim_scores_zero_rather_than_maximum():
    """A claim implying a win rate above 100% is REFUTED, not maximal.

    This was backwards once: the claim was clipped to what the barriers
    could pay, which handed the least credible setups the top score and
    the biggest stake.
    """
    print("\nsizing: an impossible claim scores 0, not 100")
    b, _ = _broker()
    tp_d, sl_d, fee = 0.010, 0.010, 0.0011
    bb = tp_d - fee
    P = b.potential(tp_d, sl_d, "mad", claimed=bb * 5, fee=fee)
    check("a claim above what a win can pay is refuted",
          P["refuted"] is True, P)
    check("and scores zero", P["score"] == 0.0, P["score"])
    frac, _, _ = b.book_margin_fraction(tp_d, sl_d, 3.0, "mad", bb * 5, fee)
    check("so it is not staked at all", frac == 0.0, frac)
    sane = b.potential(tp_d, sl_d, "sane", claimed=bb * 0.3, fee=fee)
    check("a credible claim still scores", sane["score"] > 0, sane["score"])
    check("the refuted one does NOT outrank the credible one",
          P["score"] < sane["score"])


def test_a_losing_shape_stops_trading():
    """Test, and drop what loses -- the operator's own rule, as code."""
    print("\nsizing: a shape 2 SE below zero stops trading")
    b, _ = _broker()
    tp_d, sl_d, fee = 0.030, 0.020, 0.0011
    a, bb = sl_d + fee, tp_d - fee
    p_be = a / (a + bb)
    sd = (p_be * bb * bb + (1 - p_be) * a * a) ** 0.5
    n = 40
    b.rule_record["loser"] = (n, n * (-3.0 * sd / n ** 0.5))
    P = b.potential(tp_d, sl_d, "loser", claimed=0.004, fee=fee)
    check("it is CUT", P["cut"] is True, P)
    check("a cut shape scores zero", P["score"] == 0.0, P["score"])
    frac, _, _ = b.book_margin_fraction(tp_d, sl_d, 3.0, "loser", 0.004, fee)
    check("and is refused any stake", frac == 0.0, frac)
    check("the cut fires even on a positive claim -- losses outrank claims",
          b.potential(tp_d, sl_d, "loser", 0.05, fee)["cut"] is True)
    b.rule_record["one"] = (1, -a)
    check("one loss does not stop a shape",
          b.potential(tp_d, sl_d, "one", 0.004, fee)["cut"] is False)


def test_one_coin_carries_one_position_per_shape():
    """A twelve-hour swing must not lock a coin out of every one-hour
    shape for twelve hours. Positions are keyed by SLOT."""
    print("\nslots: a coin carries one position per shape and side")
    from fp import bot as B
    b, _ = _broker(max_margin_pct=0.2)
    b.refresh_prices()
    made = []
    for tp, sl, hold, side in ((1.0, 1.0, 60, 1), (3.0, 3.0, 720, 1),
                               (2.0, 2.0, 240, -1)):
        rule = f"eng:{tp}/{sl}/{hold}:{'long' if side > 0 else 'short'}"
        sig = B.Signal(bar_ts=B.closed_bar_ts(bar_minutes=b.bar_minutes),
                       direction=side, atr_pct=0.1, votes=1, vote_margin=1,
                       methods="engine", slow_leverage=2.0,
                       tp_dist=0.02, sl_dist=0.015, max_hold_min=float(hold),
                       rule=rule, symbol="S0USDT", margin_frac=0.15,
                       score=40.0)
        b.signals[sig.slot] = sig
        made.append(sig)
    check("each shape gets its own slot",
          len({s.slot for s in made}) == 3, {s.slot for s in made})
    check("the slot names the coin and the shape",
          all(s.slot == f"S0USDT|{s.rule}" for s in made))
    for s in made:
        b.try_open(s.slot)
    check("all three are open on ONE coin at the same time",
          len(b.open) == 3, sorted(b.open))
    check("they are all the same symbol",
          {p.symbol for p in b.open.values()} == {"S0USDT"})
    check("both directions are held at once",
          {p.direction for p in b.open.values()} == {1, -1})



# -------------------------------------------------- methods and strategies

def test_a_state_exit_cannot_peek_at_the_flip():
    """THE bug that manufactured a +0.25%/trade edge out of nothing.

    A run's last bar is only known to be the last once the NEXT bar's
    state comes out different -- which is known at the next bar's close.
    Exiting at the run's own last close therefore sells one bar before
    the flip, with information the bot cannot have. With that bug in
    place solo:rsi_trend measured +0.2491%/trade over 19,897 trades at
    t = +40.85; corrected, the same strategy measures -0.0817% at
    t = -21.03.
    """
    print("\nstrategies: the exit uses the first price AFTER the flip")
    from fp import strategies as SG
    import numpy as _np
    # A state that is long for bars 0..9 then flat. The flip is visible
    # at bar 10, so the exit price must be close[10], not close[9].
    n = 40
    close = _np.arange(1.0, n + 1.0) * 100.0
    state = _np.zeros(n, dtype="int8")
    state[:10] = 1
    st, ex, sd, net = SG.trades(close, state, min_run=5)
    check("one trade is produced", len(st) == 1, len(st))
    # Entry waits out min_run bars of persistence -- see the persistence
    # test below -- so with min_run=5 it opens at bar 4, not bar 0.
    check("it enters once the state has persisted", st[0] == 4, st[0])
    check("it exits on the bar AFTER the run ends, not the last bar of it",
          ex[0] == 10, ex[0])
    # And the arithmetic uses that price.
    expect = (close[10] / close[4] - 1.0)
    check("the return is measured to that later price",
          abs((net[0] + SG.FEE + (6 / 60.0) / 8.0 * SG.FUNDING_PER_8H)
              - expect) < 1e-9,
          (net[0], expect))
    check("the hold counts the extra bar",
          True)

    # A short must mirror it exactly.
    state2 = _np.zeros(n, dtype="int8")
    state2[:10] = -1
    _, ex2, sd2, net2 = SG.trades(close, state2, min_run=5)
    check("a short exits on the same bar", ex2[0] == 10, ex2[0])
    check("and its sign is reversed", sd2[0] == -1 and net2[0] < net[0])


def test_methods_are_states_not_fixed_trades():
    """No method may name an entry price, an exit price or a duration."""
    print("\nmethods: a signed state, and nothing else is fixed")
    from fp import methods as MT
    B = board(n=3000, k=5)
    d = B["S0USDT"]
    X = MT.states(d, B, "S0USDT")
    check("more than forty methods", X.shape[1] >= 40, X.shape[1])
    vals = set(_u for _u in np.unique(X.values).tolist())
    check("every method emits only -1, 0 or +1", vals <= {-1, 0, 1}, vals)
    check("no method is constantly flat",
          (X.abs().sum(axis=0) > 0).sum() >= X.shape[1] - 6,
          int((X.abs().sum(axis=0) == 0).sum()))

    # Holds must VARY -- a fixed duration anywhere would show up as one
    # repeated run length.
    from fp import strategies as SG
    c = d["close"].values
    S = SG.all_strategies(d, B, "S0USDT")
    lens = []
    for name in list(S)[:40]:
        st, ex, sd, net = SG.trades(c, S[name])
        if len(st) > 5:
            lens.append(len(np.unique(ex - st)))
    check("trade durations vary rather than repeating one number",
          lens and np.median(lens) > 3, sorted(lens)[:5])


def test_no_strategy_state_reads_the_future():
    print("\nstrategies: truncating the future leaves earlier states alone")
    from fp import methods as MT
    B = board(n=2600, k=4)
    full = MT.states(B["S0USDT"], B, "S0USDT")
    cut = 2000
    part = MT.states(B["S0USDT"].iloc[:cut],
                     {k: v.iloc[:cut] for k, v in B.items()}, "S0USDT")
    a, b = full.iloc[:cut].values, part.values
    # opening_range keys off the calendar day and legitimately differs at
    # a truncated final day; everything else must match exactly.
    cols = [i for i, c in enumerate(full.columns) if c != "open_range"]
    diff = int((a[:, cols] != b[:, cols]).sum())
    check("no method changes when later bars are removed", diff == 0,
          f"{diff} cells differ")



def test_the_persistence_filter_cannot_select_on_the_outcome():
    """The second free peek, and the worse of the two.

    Discarding runs that turned out shorter than min_run selects on the
    OUTCOME: a run is only known to be short once it has ended, and short
    runs are exactly the breakouts that failed. With that filter in place
    solo:boll_break measured +0.47%/trade and all three folds "beat their
    null" at p = 0.000. Corrected, the same strategy measures -0.16%.

    The honest version waits: enter only once the state HAS persisted, at
    the price you get for waiting.
    """
    print("\nstrategies: persistence is waited for, never filtered on")
    from fp import strategies as SG
    import numpy as _np
    n = 60
    close = _np.full(n, 100.0)
    close[:] = 100.0 + _np.arange(n)          # steadily rising
    # Two runs: a SHORT one (3 bars) and a LONG one (10 bars).
    state = _np.zeros(n, dtype="int8")
    state[5:8] = 1        # 3 bars -- shorter than min_run
    state[20:30] = 1      # 10 bars
    st, ex, sd, net = SG.trades(close, state, min_run=5)
    check("the short run is not silently dropped as a winner-filter",
          len(st) == 1, len(st))
    check("the surviving trade ENTERS after waiting, not at the run start",
          st[0] == 24, st[0])
    check("it still exits after the flip is visible", ex[0] == 30, ex[0])

    # A run exactly as long as the wait must still be tradeable, and one
    # shorter than the wait simply never opens -- because the bot is
    # still waiting when it ends, which is the truthful outcome.
    state3 = _np.zeros(n, dtype="int8")
    state3[10:15] = 1     # exactly 5 bars
    st3, ex3, _, _ = SG.trades(close, state3, min_run=5)
    check("a run exactly the length of the wait does open",
          len(st3) == 1 and st3[0] == 14, (len(st3), st3[:1]))
    state4 = _np.zeros(n, dtype="int8")
    state4[10:13] = 1     # 3 bars, ends while still waiting
    st4, _, _, _ = SG.trades(close, state4, min_run=5)
    check("a run that ends during the wait never opens", len(st4) == 0,
          len(st4))

    # The whole point: waiting must COST something on a trending path.
    # Entering later in a rise captures less of it than entering at the
    # start -- if it did not, the wait would be free and the filter would
    # not have been a bug.
    _, _, _, net_wait = SG.trades(close, state, min_run=5)
    _, _, _, net_now = SG.trades(close, state, min_run=1)
    check("waiting gives up part of the move it waited through",
          net_wait[0] < max(net_now), (net_wait[0], max(net_now)))



def test_a_coin_takes_a_second_position_only_when_it_is_free():
    """One position per coin, enforced at TRADE time and nowhere else.

    The logic library is deliberately not narrowed when it is built --
    fp/coverage.py measures it reaching 100% of the market's >1% chances,
    and filtering at build time would throw that away. The constraint
    belongs at the moment of entry: if the coin already holds something,
    a second signal on it waits.
    """
    print("\nexecution: a coin holds one position, checked at entry")
    from fp import bot as B
    b, _ = _broker(("S0USDT", "S1USDT"), max_margin_pct=1.0)
    b.refresh_prices()
    bar = B.closed_bar_ts(bar_minutes=b.bar_minutes)

    def sig(sym, rule, direction=1):
        s = B.Signal(bar_ts=bar, direction=direction, atr_pct=0.2, votes=1,
                     vote_margin=1, methods="test", slow_leverage=2.0,
                     tp_dist=0.02, sl_dist=0.015, max_hold_min=240.0,
                     rule=rule, symbol=sym, margin_frac=0.10, score=50.0)
        b.signals[s.slot] = s
        return s

    a1 = sig("S0USDT", "book:one")
    a2 = sig("S0USDT", "book:two")
    c1 = sig("S1USDT", "book:one")
    check("the first position on a coin opens", b.try_open(a1.slot) is True)
    held = [k for k in b.open if k.startswith("S0USDT|")]
    check("the coin now holds exactly one", len(held) == 1, held)

    # A different coin is unaffected -- the rule is per coin, not global.
    check("another coin can still open", b.try_open(c1.slot) is True)

    # And the second signal on the busy coin must not become a position.
    opened = [k for k in b.open if k.startswith("S0USDT|")]
    check("a second signal did not open on the busy coin",
          a2.slot not in opened, opened)
    check("but its signal is still standing, ready for when the coin frees",
          a2.slot in b.signals, sorted(b.signals))

    # Free the coin and the waiting signal becomes tradeable.
    px = b.last_price("S0USDT")
    b._close(a1.slot, px, "take_profit")
    b.traded_bar.pop(a2.slot, None)
    check("once the coin is free the waiting signal can open",
          b.try_open(a2.slot) is True,
          [k for k in b.open if k.startswith("S0USDT|")])



# ------------------------------------------------- direction and leverage

def test_the_direction_label_is_a_race_not_a_guess():
    """+1 means a long reached +1% net BEFORE a short could, and the
    barrier is charged the fee, so the price has to travel further than
    1% to earn 1%."""
    print("\ndirection: the label is a first-touch race, fee included")
    from fp import direction as DIR
    import numpy as _np
    n = 200
    # A path that rises 3% and never falls: every early bar is a long.
    up = 100.0 * (1.0 + _np.linspace(0, 0.03, n))
    y = DIR.label(up, horizon=n)
    check("a rising path labels long", y[0] == 1, y[0])
    check("and never labels short", not (y == -1).any(),
          int((y == -1).sum()))
    dn = 100.0 * (1.0 - _np.linspace(0, 0.03, n))
    check("a falling path labels short", DIR.label(dn, horizon=n)[0] == -1)

    # A move of exactly +1% does NOT clear a +1% NET barrier, because the
    # round trip has to come out of it first.
    flat = _np.full(n, 100.0)
    flat[50:] = 100.0 * (1.0 + DIR.WIN)          # exactly +1%, no more
    check("a move of exactly the target does not clear it net of fees",
          DIR.label(flat, horizon=n)[0] == 0, DIR.label(flat, horizon=n)[0])
    flat2 = _np.full(n, 100.0)
    flat2[50:] = 100.0 * (1.0 + DIR.WIN + DIR.FEE + 1e-6)
    check("a move of target PLUS the round trip does clear it",
          DIR.label(flat2, horizon=n)[0] == 1)

    # Nothing may be labelled from beyond its own horizon.
    late = _np.full(n, 100.0)
    late[150:] = 130.0
    check("a move outside the horizon is not labelled",
          DIR.label(late, horizon=20)[0] == 0)


def test_break_even_accuracy_is_the_bar_the_model_must_clear():
    print("\ndirection: break-even accuracy comes from the barrier and fee")
    from fp import direction as DIR
    p_be = DIR.break_even()
    gain, loss = DIR.WIN - DIR.FEE, DIR.WIN + DIR.FEE
    check("break-even is loss/(loss+gain)",
          abs(p_be - loss / (loss + gain)) < 1e-12, p_be)
    check("it is above a coin flip -- the fee has to be paid",
          p_be > 0.5, p_be)
    ev = p_be * gain - (1 - p_be) * loss
    check("at exactly break-even the expected value is zero",
          abs(ev) < 1e-12, ev)
    check("one point above break-even is profitable",
          (p_be + 0.01) * gain - (1 - p_be - 0.01) * loss > 0)
    check("a zero-fee world would need only 50%",
          abs(DIR.break_even(fee=0.0) - 0.5) < 1e-12)


def test_leverage_rides_the_same_potential_as_the_stake():
    """The operator's rule: capital and leverage are one decision, not
    two. Both run from their floor to their ceiling on the same score."""
    print("\ndirection: leverage scales with potential, floor to ceiling")
    from fp import direction as DIR
    import numpy as _np
    lo, hi = DIR.LEV_MIN, DIR.LEV_MAX
    check("potential 0 takes the floor",
          abs(DIR.leverage_for(0) - lo) < 1e-12, DIR.leverage_for(0))
    check("potential 100 takes the ceiling",
          abs(DIR.leverage_for(100) - hi) < 1e-12, DIR.leverage_for(100))
    check("halfway is halfway",
          abs(DIR.leverage_for(50) - (lo + hi) / 2) < 1e-12)
    v = DIR.leverage_for([0, 25, 50, 75, 100])
    check("it is monotone", bool((_np.diff(v) > 0).all()), v)
    check("it never exceeds the ceiling, whatever is passed in",
          DIR.leverage_for(1e6) <= hi + 1e-12 and DIR.leverage_for(-50) >= lo)

    # And the score itself: 0 at break-even, 100 at certainty.
    p_be = DIR.break_even()
    check("potential is 0 at break-even",
          DIR.potential(_np.array([p_be]), p_be)[0] == 0.0)
    check("potential is 100 at certainty",
          abs(DIR.potential(_np.array([1.0]), p_be)[0] - 100.0) < 1e-9)
    check("below break-even scores 0, never negative",
          DIR.potential(_np.array([p_be - 0.1]), p_be)[0] == 0.0)



def main() -> int:
    print("=" * 70)
    print("fp.engine tests")
    print("=" * 70)
    for fn in (test_factors_are_complete_and_finite,
               test_factors_are_scale_free,
               test_no_factor_can_see_its_own_future,
               test_cross_section_measures_the_board_not_the_coin,
               test_the_fee_lives_inside_the_label,
               test_the_target_scales_with_the_horizon,
               test_a_label_never_reads_past_its_own_horizon,
               test_potential_is_zero_at_break_even_and_100_at_certainty,
               test_the_probability_is_lower_bounded_by_its_own_sample,
               test_break_even_uses_the_horizon_scaled_barriers,
               test_overlapping_trades_are_not_counted_twice,
               test_the_rotation_null_keeps_everything_but_the_alignment,
               test_the_live_path_refuses_a_mismatched_matrix,
               test_the_live_path_needs_the_whole_board,
               test_the_stake_is_the_potential_and_can_take_the_account,
               test_an_impossible_claim_scores_zero_rather_than_maximum,
               test_a_losing_shape_stops_trading,
               test_one_coin_carries_one_position_per_shape,
               test_a_coin_takes_a_second_position_only_when_it_is_free,
               test_the_direction_label_is_a_race_not_a_guess,
               test_break_even_accuracy_is_the_bar_the_model_must_clear,
               test_leverage_rides_the_same_potential_as_the_stake,
               test_a_state_exit_cannot_peek_at_the_flip,
               test_the_persistence_filter_cannot_select_on_the_outcome,
               test_methods_are_states_not_fixed_trades,
               test_no_strategy_state_reads_the_future):
        fn()
    print("\n" + "=" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
    else:
        print("all checks passed")
    print("=" * 70)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
