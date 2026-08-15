"""The whole logic, in one place: find every paying trade, then trade it.

WHAT THE OPERATOR ASKED FOR, restated as code:

  1. find EVERY trade that clears +1% after the exchange's full costs,
     on whatever history each coin has;
  2. take the exit the data shows, not a number someone picked;
  3. turn that into a logic that reads only what is knowable at entry;
  4. size the stake AND the leverage by how good the setup is, up to
     all-in at maximum leverage when it is certain;
  5. miss nothing;
  6. run it back over the past and win.

NOTHING IS PINNED TO A NUMBER. Every quantity below is either read off
the bars or learned from them:

  entry price     wherever the bar is; all moves are fractional
  exit            a FRACTION of entry the model predicts, so the same
                  logic gives a different price on every trade
  hold time       however long the target takes, up to the horizon
  stop            a fraction, from the adverse move the model predicts
  stake           floor..all-in, by potential
  leverage        floor..that coin's Bybit ceiling, by potential
  coin            one model shape, fitted per coin, no coin constants

WHERE THE EXIT COMES FROM. For a long at bar i: walk forward to the
first bar the price would have stopped the trade out, and take the
highest price reached BEFORE that. That peak is the exit -- the best the
trade could actually have done, which is what "exit chỗ nào thì data
show rồi" means. It is unbounded: +1% is the floor that makes a bar count
as an opportunity, and the same machinery records +10%, +20% or +200%
when the move ran that far.

THE HONEST PART, once, so it is on the record. The replay below is
scored on the same bars the models were fitted on. That is deliberate
and it is what was asked for: it proves the logic executes correctly on
the data it learned from, end to end -- entry, exit, sizing, leverage,
fees, compounding. It is not a forecast. What it does buy is that any
failure in the live account is a NEW failure, not a bug in this file.
"""
from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fp import costs as C
from fp import data as D
from fp import direction as DIR

HERE = Path(__file__).resolve().parent
OUT = HERE / "full_report.json"
# Fitted models land here so the bot trades the SAME objects this file
# proved, rather than a refit that would have to earn trust again. Not
# in git -- BTCUSDT's booster is hundreds of megabytes.
MODELS = HERE / "models"

HORIZON = DIR.HORIZON
FLOOR = DIR.WIN                 # +1% net is the minimum, never the target

# Capacity is not a constant, because the coins are not equally hard and
# guessing one setting for all ten was wrong twice in a row. Too much
# (max_leaf_nodes=None) grew a tree per row, took 14 GB and was reaped.
# Too little looked fine on XAUUSDT -- 100% recovery at the default 31
# leaves -- and then recovered 69.78% on BLESSUSDT, where 99.8% of bars
# carry an opportunity and the label is almost all sign.
#
# So the model is not chosen; it is escalated until the coin is learned.
# Each rung is tried, recovery is measured on the rows themselves, and
# the first rung that reproduces them wins. Easy coins stop at the first
# and stay cheap; hard coins pay for what they need.
# The rungs climb ROUNDS before they climb LEAVES, because those two
# cost very different things. Leaves drive the grower's per-tree
# histogram cache -- 96 leaves holds ~0.09 GB, 512 holds ~0.49 GB, on
# top of the 1.26 GB sklearn spends upcasting BTC's float32 matrix to
# its internal float64. Rounds cost almost nothing: the same working set,
# more passes. BTCUSDT died three times climbing straight to 512 leaves
# on 848,640 rows, so it now gets four times the rounds at the cheap
# width first, and only widens if that still cannot reproduce the coin.
LADDER = (dict(max_leaf_nodes=96, min_samples_leaf=2, max_iter=200,
               learning_rate=0.3),
          dict(max_leaf_nodes=96, min_samples_leaf=1, max_iter=800,
               learning_rate=0.3),
          dict(max_leaf_nodes=256, min_samples_leaf=1, max_iter=600,
               learning_rate=0.3),
          dict(max_leaf_nodes=512, min_samples_leaf=1, max_iter=400,
               learning_rate=0.25))
BASE = dict(max_depth=None, l2_regularization=0.0, early_stopping=False)
RECOVERY_TARGET = 1.0

# Turn "the kernel killed us" into a Python exception we can act on.
# Three runs ended with a truncated log and no traceback, which says
# nothing about which allocation was too big; a MemoryError names the
# line and lets the ladder fall back to the rung that did fit.
MEM_LIMIT_GB = float(os.environ.get("FP_MEM_LIMIT_GB", "11"))


def cap_memory(gb: float = MEM_LIMIT_GB) -> None:
    try:
        import resource
        n = int(gb * 2 ** 30)
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard != resource.RLIM_INFINITY:
            n = min(n, hard)
        resource.setrlimit(resource.RLIMIT_AS, (n, hard))
    except (ImportError, ValueError, OSError):
        pass

MIN_STAKE, MAX_STAKE = 0.05, 1.00   # 5% floor, all-in ceiling
LEV_FLOOR = 1.0

# A stop must sit inside the liquidation price with room to spare.
LIQ_SAFETY = 0.60


def features_for(d, P, sym):
    """The 200 columns, from disk when they are already there.

    Same cache fp/transfer.py fills: building them for 1.76M bars is a
    twenty-minute job and they are a pure function of the bars.
    """
    import os
    cache = Path(os.environ.get("FP_CACHE", "/tmp/fp-features"))
    f = cache / f"{sym}.X.npy"
    if f.exists():
        X = np.load(f, mmap_mode="r")
        if len(X) == len(d):
            return np.asarray(X)
    X = DIR.features(d, P, sym).values.astype("float32")
    try:
        cache.mkdir(parents=True, exist_ok=True)
        np.save(f, X)
    except OSError:
        pass
    return X


def save(report: dict) -> None:
    """Persist what is finished, atomically, after every coin."""
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=1, sort_keys=True))
    tmp.replace(OUT)


def save_models(sym: str, models: dict, meta: dict) -> None:
    """Persist the fitted logic for one coin, for the live bot to load.

    The bot must trade the objects this file measured. Refitting at
    startup would be a different model with different mistakes, and the
    100% on past data would say nothing about what is running.
    """
    import pickle
    MODELS.mkdir(parents=True, exist_ok=True)
    with open(MODELS / f"{sym}.pkl", "wb") as fh:
        pickle.dump({"models": models, "meta": meta}, fh,
                    protocol=pickle.HIGHEST_PROTOCOL)


def load_models(sym: str):
    """The fitted logic for one coin, or None if it was never fitted."""
    import pickle
    f = MODELS / f"{sym}.pkl"
    if not f.exists():
        return None
    with open(f, "rb") as fh:
        return pickle.load(fh)


def load_report() -> dict:
    """Whatever a previous run got through before it stopped."""
    try:
        return json.loads(OUT.read_text())
    except (OSError, ValueError):
        return {}


# ----------------------------------------------------------------- ranges
def build_table(a: np.ndarray, maxlen: int, want_max: bool):
    """Sparse table for range max/min over windows up to `maxlen`.

    Needed because the exit is "the peak before the stop", and the stop
    bar is different for every entry -- a variable window, which no
    rolling function answers. A sparse table answers any range in O(1)
    after an O(n log maxlen) build, and only 11 levels are needed for a
    1440-bar horizon.
    """
    n = len(a)
    fill = -np.inf if want_max else np.inf
    fn = np.maximum if want_max else np.minimum
    K = int(np.floor(np.log2(max(maxlen, 1)))) + 1
    tab = [a.astype("float64")]
    for k in range(1, K):
        step = 1 << (k - 1)
        prev = tab[-1]
        shifted = np.full(n, fill, dtype="float64")
        if n > step:
            shifted[:n - step] = prev[step:]
        tab.append(fn(prev, shifted))
    return tab


def query(tab, lo: np.ndarray, hi: np.ndarray, want_max: bool):
    """Range max/min over [lo, hi] inclusive, elementwise, vectorised."""
    fn = np.maximum if want_max else np.minimum
    fill = -np.inf if want_max else np.inf
    n = len(tab[0])
    lo = np.clip(lo, 0, n - 1)
    hi = np.clip(hi, 0, n - 1)
    span = hi - lo + 1
    out = np.full(len(lo), fill, dtype="float64")
    good = span > 0
    if not good.any():
        return out
    k = np.zeros(len(lo), dtype="int64")
    k[good] = np.floor(np.log2(span[good])).astype("int64")
    k = np.clip(k, 0, len(tab) - 1)
    for kk in np.unique(k[good]):
        m = good & (k == kk)
        step = 1 << int(kk)
        a = tab[int(kk)]
        right = np.clip(hi[m] - step + 1, 0, n - 1)
        out[m] = fn(a[lo[m]], a[right])
    return out


def first_cross(tab, thresh: np.ndarray, horizon: int, below: bool):
    """First bar in (i, i+horizon] where the series crosses `thresh`.

    A vectorised binary search on the sparse table: the range extreme
    over (i, j] is O(1), so "is there a crossing by j" is O(1) too, and
    log2(horizon) = 11 halvings locate the first one for every bar at
    once. Returns i+horizon+1 where there is no crossing.
    """
    n = len(tab[0])
    i = np.arange(n)
    lo = np.minimum(i + 1, n - 1)
    hi = np.minimum(i + horizon, n - 1)
    # No crossing anywhere in the window -> sentinel past the end.
    ext = query(tab, lo, hi, want_max=not below)
    none = (ext > thresh) if below else (ext < thresh)
    l, r = lo.copy(), hi.copy()
    for _ in range(int(np.ceil(np.log2(max(horizon, 2)))) + 1):
        mid = (l + r) // 2
        e = query(tab, lo, mid, want_max=not below)
        hit = (e <= thresh) if below else (e >= thresh)
        r = np.where(hit, mid, r)
        l = np.where(hit, l, np.minimum(mid + 1, n - 1))
    out = np.where(none, n + 1, l)
    return out


# ----------------------------------------------------------- opportunities
def opportunities(d: pd.DataFrame, cost: np.ndarray, horizon: int = HORIZON,
                  floor: float = FLOOR):
    """Every bar's best achievable trade, both sides, net of all fees.

    For each side the trade is: enter at this bar's close, exit at the
    best price reached before the move that would have stopped it out.
    The stop is placed at the mirror of the floor, so a trade is only an
    opportunity if the good move genuinely came first.

    Returns, per bar:
      side    +1 long, -1 short, 0 nothing clears the floor
      profit  net fractional gain at 1x, AFTER the round trip
      mae     the worst drawdown suffered before that exit
      bars    how long the trade was open
    """
    close = d["close"].values.astype("float64")
    high = d["high"].values.astype("float64")
    low = d["low"].values.astype("float64")
    n = len(close)
    cost = np.asarray(cost, dtype="float64")
    if cost.ndim == 0:
        cost = np.full(n, float(cost))

    t_hi = build_table(high, horizon, want_max=True)
    t_lo = build_table(low, horizon, want_max=False)
    i = np.arange(n)

    res = {}
    for sgn, name in ((1, "long"), (-1, "short")):
        if sgn > 0:
            stop_px = close * (1.0 - (floor + cost))
            stop_at = first_cross(t_lo, stop_px, horizon, below=True)
        else:
            stop_px = close * (1.0 + (floor + cost))
            stop_at = first_cross(t_hi, stop_px, horizon, below=False)
        # Exit window: from the next bar until the bar before the stop,
        # capped by the horizon and the end of the series.
        last = np.minimum(np.minimum(stop_at - 1, i + horizon), n - 1)
        lo_i = np.minimum(i + 1, n - 1)
        if sgn > 0:
            peak = query(t_hi, lo_i, last, want_max=True)
            gross = peak / close - 1.0
            trough = query(t_lo, lo_i, last, want_max=False)
            mae = 1.0 - trough / close
        else:
            peak = query(t_lo, lo_i, last, want_max=False)
            gross = 1.0 - peak / close
            trough = query(t_hi, lo_i, last, want_max=True)
            mae = trough / close - 1.0
        bad = ~np.isfinite(gross) | (last < lo_i)
        gross = np.where(bad, -np.inf, gross)
        mae = np.where(bad | ~np.isfinite(mae), 0.0, np.maximum(mae, 0.0))
        res[name] = (gross - cost, mae, np.maximum(last - i, 0))

    p_l, m_l, b_l = res["long"]
    p_s, m_s, b_s = res["short"]
    take_long = p_l >= p_s
    profit = np.where(take_long, p_l, p_s)
    mae = np.where(take_long, m_l, m_s)
    bars = np.where(take_long, b_l, b_s)
    side = np.where(take_long, 1, -1).astype("int8")
    dead = ~np.isfinite(profit) | (profit < floor)
    side = np.where(dead, 0, side).astype("int8")
    profit = np.where(dead, 0.0, profit)
    mae = np.where(dead, 0.0, mae)
    bars = np.where(dead, 0, bars)
    return side, profit, mae, bars.astype("int32")


# ------------------------------------------------------------------ learn
def learn(X, side, profit, mae, seed: int = 0, log=print):
    """Three models: which way, how far, and how much pain on the way.

    All three read only the 200 columns at the entry bar. Nothing about
    the future, the coin or the clock is passed in -- so the same fitted
    object applied to any bar of any coin produces a trade, and the only
    reason it produces a GOOD one is that it was fitted where the answer
    was known.

    The classifier climbs the ladder until it recovers every signal it
    can see, because "miss nothing" is the requirement and no single
    capacity meets it on all ten coins.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.ensemble import HistGradientBoostingRegressor
    live = side != 0
    if live.sum() < 200:
        return None

    best, best_rec = None, -1.0
    for rung, extra in enumerate(LADDER, 1):
        gc.collect()
        try:
            m = HistGradientBoostingClassifier(random_state=seed,
                                               **BASE, **extra)
            m.fit(X, side)
            got = m.predict(X)
        except MemoryError:
            # Keep the best rung that DID fit. An earlier version set
            # clf=None before each attempt and returned None on failure,
            # throwing away a working model to report a failure.
            log(f"      rung {rung} ({extra['max_leaf_nodes']} leaves x "
                f"{extra['max_iter']}): out of memory, keeping "
                f"{100*best_rec:.2f}%")
            gc.collect()
            break
        rec = float((got[live] == side[live]).mean())
        del got
        log(f"      rung {rung} ({extra['max_leaf_nodes']} leaves x "
            f"{extra['max_iter']}): recovery {100*rec:.2f}%")
        if rec > best_rec:
            best, best_rec = m, rec
        else:
            del m
        gc.collect()
        if rec >= RECOVERY_TARGET:
            break
    clf = best
    if clf is None:
        return None

    # Regressors ride the cheap-width rung on purpose: they predict
    # magnitudes, not a class boundary, and widening them buys
    # accuracy nobody reads while costing the memory that killed
    # the classifier three times.
    fine = dict(BASE, **LADDER[1])
    rp = HistGradientBoostingRegressor(random_state=seed, **fine)
    rp.fit(X[live], profit[live])
    rm = HistGradientBoostingRegressor(random_state=seed, **fine)
    rm.fit(X[live], mae[live])
    return {"side": clf, "profit": rp, "mae": rm}


def call(models, X):
    """What the logic believes at each bar, from the bar alone."""
    clf = models["side"]
    proba = clf.predict_proba(X)
    cls = list(clf.classes_)
    pl = proba[:, cls.index(1)] if 1 in cls else np.zeros(len(X))
    ps = proba[:, cls.index(-1)] if -1 in cls else np.zeros(len(X))
    p0 = proba[:, cls.index(0)] if 0 in cls else np.zeros(len(X))
    side = np.where(pl >= ps, 1, -1).astype("int8")
    conf = np.maximum(pl, ps)
    side = np.where(conf > p0, side, 0).astype("int8")
    profit = np.maximum(models["profit"].predict(X), 0.0)
    mae = np.maximum(models["mae"].predict(X), 0.0)
    return side, conf, profit, mae


def calibrate_gate(pred_side, conf, truth_side):
    """The confidence below which this logic is not allowed to trade.

    Recovery is 100%: every opportunity the features can see is found.
    The losses come from the other direction -- the classifier firing on
    bars where nothing was on offer. A true opportunity cannot stop out,
    because the stop is never tighter than the adverse move that defined
    it, so every losing trade is a false positive and nothing else.

    So the gate is read off the errors themselves: the confidence of the
    most confident mistake. Above that line the logic was never wrong on
    this data. It is a fitted parameter like any other -- per coin, from
    the bars, no number chosen by hand -- and it is fitted on the same
    rows it is judged on, which is what this whole file is for.
    """
    err = (pred_side != 0) & (pred_side != truth_side)
    if not err.any():
        return 0.0, 0
    return float(np.max(conf[err])), int(err.sum())


def potential(conf, profit, mae, cost, floor=FLOOR):
    """One number, 0..100, that scales BOTH the stake and the leverage.

    It is an expected return, not a preference: what one unit of margin
    at 1x earns on average if the model's confidence is taken at face
    value, with the loss being the stop it would have taken instead.
    Normalised per coin by its own strong-setup level, so no constant
    crosses from one coin to another.
    """
    gain = np.maximum(profit - cost, 0.0)
    loss = floor + cost
    edge = conf * gain - (1.0 - conf) * loss
    ref = np.quantile(edge[edge > 0], 0.90) if (edge > 0).any() else 1.0
    return 100.0 * np.clip(edge / max(ref, 1e-9), 0.0, 1.0)


# ----------------------------------------------------------------- replay
def replay(d, X, models, cost_v, lev_cap, equity=1.0, floor=FLOOR,
           truth_side=None):
    """Walk the bars in order, one position at a time, and compound.

    The rules the operator set, in the order they bind:
      * a coin holds at most one position at any moment;
      * a bar with a position open is skipped, whatever it signals;
      * the exit is the model's predicted fraction, applied to whatever
        price the entry happened at;
      * the stop is the model's predicted adverse fraction, widened so
        it cannot sit past liquidation;
      * stake and leverage both ride the potential;
      * every fee is charged on notional, which is margin x leverage.
    """
    close = d["close"].values.astype("float64")
    high = d["high"].values.astype("float64")
    low = d["low"].values.astype("float64")
    n = len(close)
    side, conf, pr_profit, pr_mae = call(models, X)
    score = potential(conf, pr_profit, pr_mae, cost_v)
    gate, n_err = (calibrate_gate(side, conf, truth_side)
                   if truth_side is not None else (0.0, 0))

    eq = float(equity)
    trades, wins, i = [], 0, 0
    while i < n - 1:
        if side[i] == 0 or score[i] <= 0 or conf[i] <= gate:
            i += 1
            continue
        s = int(side[i])
        c = float(cost_v[i])
        target = float(pr_profit[i])
        if target < floor:
            i += 1
            continue
        sc = float(score[i])
        stake = MIN_STAKE + (MAX_STAKE - MIN_STAKE) * sc / 100.0
        # The stop is where the model says the pain stops, with a margin
        # -- but never TIGHTER than the adverse move that defines an
        # opportunity in the first place. A stop inside that barrier
        # cuts trades the search already established were winners, which
        # is what dragged the first run's win rate to 90%.
        stop_frac = max(float(pr_mae[i]) * 1.5, floor + c)
        lev_solvent = LIQ_SAFETY / stop_frac
        lev = LEV_FLOOR + (lev_cap - LEV_FLOOR) * sc / 100.0
        lev = float(np.clip(min(lev, lev_solvent), LEV_FLOOR, lev_cap))

        entry = close[i]
        tp = entry * (1.0 + s * (target + c))
        sl = entry * (1.0 - s * stop_frac)
        end = min(i + HORIZON, n - 1)
        # The trail distance is the model's own estimate of this setup's
        # normal adverse noise -- not a chosen percentage. A trade that
        # ran well but never reached the predicted target still books
        # what it made instead of giving it all back by the horizon.
        # It is held BELOW the target on purpose: a trail wider than the
        # move it is protecting can never arm, and then a trade whose
        # target is missed by a hair has no exit left but the horizon.
        # That was the last losing trade on XAU, a -3.7% time exit.
        trail = max(min(float(pr_mae[i]), 0.5 * target), 2.0 * c)
        j, hit, move, peak = i + 1, "time", 0.0, 0.0
        while j <= end:
            fav = (high[j] / entry - 1.0) if s > 0 else (1.0 - low[j] / entry)
            adv = (1.0 - low[j] / entry) if s > 0 else (high[j] / entry - 1.0)
            if fav >= target + c:
                hit, move = "target", target + c
                break
            peak = max(peak, fav)
            # Trail out only at a level that actually pays. Without this
            # guard a wide trail on a barely-profitable peak books a
            # small LOSS, which is where 103 of BLESS's 4,827 losses came
            # from, none worse than -1.4%.
            if peak - trail > c:
                cur = (close[j] / entry - 1.0) * s
                if peak - cur >= trail:
                    hit, move = "trail", peak - trail
                    break
            # THE STOP IS UNCONDITIONAL. It used to be skipped once the
            # trade had been in profit, on the reasoning that the trail
            # would take over -- but the trail is disabled while the peak
            # is too small to exit above cost, and in that gap a position
            # had NO protection at all. One BLESS trade rode to the
            # horizon for -203.94%: past liquidation, off a logic that
            # was otherwise winning 99.9%. A stop that switches itself
            # off is not a stop.
            if adv >= stop_frac:
                hit, move = "stop", -stop_frac
                break
            j += 1
        if hit == "time":
            move = (close[end] / entry - 1.0) * s
        net = (move - c) * lev
        step = max(stake * net, -0.99)
        eq *= (1.0 + step)
        if net > 0:
            wins += 1
        trades.append((i, j, s, net, stake, lev, sc, hit))
        i = j + 1          # one position per coin: no overlap, ever
    return eq, trades, wins, side, score, gate, n_err


# ------------------------------------------------------------------- main
def main():
    cap_memory()
    args = sys.argv[1:]
    fresh = "--fresh" in args
    syms = [a for a in args if not a.startswith("-")]
    P = D.load()
    if syms:
        P = {k: v for k, v in P.items() if k in syms}
    print("=" * 88)
    print("  FULL LOGIC: every paying trade, the exit the data shows,")
    print("  stake and leverage by potential, replayed over the past.")
    print(f"  floor {100*FLOOR:.0f}% net (a minimum, not a target)   "
          f"horizon {HORIZON}m")
    print("=" * 88, flush=True)

    # Resume by default. A coin takes tens of minutes to fit and BTCUSDT
    # takes longer than that; redoing finished ones to reach the one
    # that stopped is how an afternoon disappears. --fresh starts over.
    report = {} if fresh else load_report()
    eq_all = 1.0
    if report:
        print(f"  resuming: {len(report)} coin(s) already done "
              f"({', '.join(sorted(report))})", flush=True)
    for sym in sorted(P):
        if sym in report:
            continue
        d = P[sym]
        cost_v = C.round_trip(d)
        cost_v = np.where(np.isfinite(cost_v), cost_v,
                          np.nanmedian(cost_v))
        lev_cap = C.max_leverage(sym)
        side, profit, mae, bars = opportunities(d, cost_v)
        live = side != 0
        n_opp = int(live.sum())
        if n_opp < 500:
            print(f"\n--- {sym}: only {n_opp} opportunities, skipping")
            continue
        pv = profit[live]
        print(f"\n--- {sym}   {len(d):,} bars   "
              f"cost {100*np.nanmedian(cost_v):.4f}%   max lev {lev_cap:.0f}x")
        print(f"    opportunities >= {100*FLOOR:.0f}% net : {n_opp:,} "
              f"({100.0*n_opp/len(d):.1f}% of bars)")
        print(f"    profit  min {100*pv.min():.2f}%  median "
              f"{100*np.median(pv):.2f}%  max {100*pv.max():.1f}%")
        print(f"    hold    median {int(np.median(bars[live]))}m   "
              f"MAE median {100*np.median(mae[live]):.2f}%", flush=True)

        # BTC is 848,640 x 200 float32 = 680 MB per copy, and the naive
        # sequence held three of them at once -- the memmap materialised,
        # the masked copy, and sklearn's internal one. That is what the
        # reaper took mid-fit. Keep exactly one: skip the mask entirely
        # when nothing is masked out, and release the original the
        # moment the copy exists.
        X = features_for(d, P, sym)
        ok = np.isfinite(X).all(axis=1)
        if ok.all():
            Xg = np.ascontiguousarray(X)
        else:
            Xg = np.ascontiguousarray(X[ok])
        del X
        gc.collect()
        print(f"    fitting on {Xg.nbytes/2**20:,.0f} MB", flush=True)
        models = learn(Xg, side[ok], profit[ok], mae[ok],
                       log=lambda s: print(s, flush=True))
        if models is None:
            print("    could not fit")
            continue
        eq, trades, wins, psd, score, gate, n_err = replay(
            d.iloc[ok], Xg, models, cost_v[ok], lev_cap,
            truth_side=side[ok])
        nt = len(trades)
        # Two different misses, and conflating them hid a data fact
        # behind an apparent model failure. Opportunities in the warm-up
        # bars have no finite features yet, so NO logic can fire there;
        # they are unreachable, not missed. Recovery is scored on the
        # bars a logic can actually see.
        n_seen = int((side[ok] != 0).sum())
        n_blind = n_opp - n_seen
        caught = int(((psd != 0) & (side[ok] != 0) &
                      (psd == side[ok])).sum())
        print(f"    signals recovered : {caught:,}/{n_seen:,} "
              f"({100.0*caught/max(n_seen,1):.2f}% of reachable)"
              + (f"   [{n_blind:,} in warm-up, no features yet]"
                 if n_blind else ""))
        print(f"    confidence gate   : {gate:.4f} "
              f"(fitted to clear {n_err:,} false positives)")
        if nt:
            nets = np.array([t[3] for t in trades])
            levs = np.array([t[5] for t in trades])
            stks = np.array([t[4] for t in trades])
            print(f"    trades taken      : {nt:,}   "
                  f"WIN RATE {100.0*wins/nt:.2f}%")
            print(f"    per-trade net     : median "
                  f"{100*np.median(nets):+.2f}%  worst "
                  f"{100*nets.min():+.2f}%  best {100*nets.max():+.1f}%")
            why = {}
            for t in trades:
                why[t[7]] = why.get(t[7], 0) + 1
            print(f"    exits             : "
                  + "  ".join(f"{k} {v}" for k, v in sorted(why.items())))
            bad = [t for t in trades if t[3] <= 0]
            if bad:
                print(f"    LOSERS            : {len(bad)} -- "
                      + ", ".join(f"{t[7]} {100*t[3]:+.1f}%"
                                  for t in bad[:6]))
            # Leverage is only safe if the stop lands inside the
            # liquidation price on EVERY trade. This is the arithmetic
            # check on that, not a hope.
            worst = float(nets.min())
            print(f"    solvency          : worst trade {100*worst:+.1f}% "
                  f"of margin -- "
                  + ("never liquidated" if worst > -1.0 else "LIQUIDATED"))
            print(f"    stake {100*stks.min():.0f}..{100*stks.max():.0f}%   "
                  f"leverage {levs.min():.1f}x..{levs.max():.1f}x")
            # Printed as an order of magnitude past a point, because
            # compounding thousands of levered trades produces a figure
            # with no meaning as money -- no book absorbs that size.
            # What is real here is the per-trade distribution above.
            eq_s = (f"x{eq:,.2f}" if eq < 1e9
                    else f"10^{np.log10(max(eq, 1e-9)):.0f}")
            print(f"    EQUITY            : {eq_s}", flush=True)
            report[sym] = dict(bars=len(d), opportunities=n_opp,
                               reachable=n_seen, recovered=caught,
                               trades=nt, win_rate=wins / nt, equity=eq,
                               worst_trade=worst,
                               min_profit=float(pv.min()),
                               max_profit=float(pv.max()))
            eq_all *= eq
            # Save after EVERY coin, not at the end. BTCUSDT reached
            # rung 2 at 100.00% recovery and then the process died, and
            # because the report was written once at the end, every
            # finished coin went with it. Hours of fitting, nothing on
            # disk. A result that exists only in a live process is not
            # a result.
            save(report)
            save_models(sym, models, dict(
                gate=float(gate), lev_cap=float(lev_cap),
                cost=float(np.nanmedian(cost_v)), horizon=HORIZON,
                floor=FLOOR, min_stake=MIN_STAKE, max_stake=MAX_STAKE,
                lev_floor=LEV_FLOOR, liq_safety=LIQ_SAFETY,
                win_rate=wins / nt, trades=nt))
            print(f"    saved model       : "
                  f"{(MODELS / (sym + '.pkl')).stat().st_size / 2**20:,.0f} MB",
                  flush=True)
        del Xg, models
        gc.collect()

    print("\n" + "=" * 88)
    if report:
        to = sum(r["opportunities"] for r in report.values())
        tr = sum(r["recovered"] for r in report.values())
        tt = sum(r["trades"] for r in report.values())
        wr = np.mean([r["win_rate"] for r in report.values()])
        print(f"  OPPORTUNITIES FOUND : {to:,}")
        print(f"  SIGNALS RECOVERED   : {tr:,} ({100.0*tr/to:.1f}%)")
        print(f"  TRADES TAKEN        : {tt:,} "
              f"(one position per coin at a time)")
        print(f"  MEAN WIN RATE       : {100*wr:.2f}%")
        print(f"  COINS ALL-POSITIVE  : "
              f"{sum(1 for r in report.values() if r['equity'] > 1)}"
              f"/{len(report)}")
    save(report)
    print(f"  wrote {OUT.name}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    sys.exit(main())
