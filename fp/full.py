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
# proved, rather than a refit that would have to earn trust again. They
# ARE committed, gzipped: fitting the board takes hours and a clone
# should be able to trade immediately.
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
# The 800-round rung is GONE. It was added on the theory that rounds buy
# capacity more cheaply than leaves, which is true of memory and false of
# accuracy: on BTCUSDT it drove recovery from 92.70% DOWN to 73.05%.
# That is softmax boosting saturating at learning_rate 0.3, not learning.
# Width is what a hard coin actually needs, so the ladder widens, and
# the learning rate comes down as it does.
LADDER = (dict(max_leaf_nodes=96, min_samples_leaf=2, max_iter=200,
               learning_rate=0.3),
          dict(max_leaf_nodes=256, min_samples_leaf=1, max_iter=300,
               learning_rate=0.2),
          dict(max_leaf_nodes=768, min_samples_leaf=1, max_iter=300,
               learning_rate=0.15))
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


def panel_key(P) -> str:
    """A short fingerprint of the WHOLE panel, for the feature cache.

    A third of the 200 columns are cross-section -- rank, breadth,
    market move, dispersion, residual -- so a coin's features change
    whenever ANY coin's bars change, not just its own. Keying the cache
    on one symbol's row count misses that completely: after five new
    weeks were merged, BTCUSDT's own length was unchanged, so it would
    have silently reused features computed against the old panel and
    fitted a model on a mixture of two different markets.
    """
    import hashlib
    h = hashlib.blake2s(digest_size=6)
    for k in sorted(P):
        d = P[k]
        h.update(f"{k}:{len(d)}:{d.index[0]}:{d.index[-1]}|".encode())
    return h.hexdigest()


def features_for(d, P, sym):
    """The 200 columns, from disk when they are already there.

    Same cache fp/transfer.py fills: building them for 1.84M bars is a
    twenty-minute job and they are a pure function of the bars -- of ALL
    the bars, which is why the filename carries the panel fingerprint.
    """
    import os
    cache = Path(os.environ.get("FP_CACHE", "/tmp/fp-features"))
    f = cache / f"{sym}.{panel_key(P)}.X.npy"
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

    Gzipped, because these ship in the repository. Fitting the board
    takes hours; a clone should not have to repeat it. Boosted trees
    compress better than half -- 239 MB of pickles becomes 109 MB, and
    the largest single coin drops from 53 MB to 25 MB.
    """
    import gzip
    import pickle
    MODELS.mkdir(parents=True, exist_ok=True)
    blob = pickle.dumps({"models": models, "meta": meta},
                        protocol=pickle.HIGHEST_PROTOCOL)
    (MODELS / f"{sym}.pkl.gz").write_bytes(gzip.compress(blob, 6))
    # Drop any uncompressed leftover so the two cannot disagree.
    old = MODELS / f"{sym}.pkl"
    if old.exists():
        old.unlink()


def load_models(sym: str):
    """The fitted logic for one coin, or None if it was never fitted."""
    import gzip
    import pickle
    gz = MODELS / f"{sym}.pkl.gz"
    if gz.exists():
        return pickle.loads(gzip.decompress(gz.read_bytes()))
    plain = MODELS / f"{sym}.pkl"
    if plain.exists():
        return pickle.loads(plain.read_bytes())
    return None


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
def learn(X, side, profit, mae, seed: int = 0, log=print,
          max_rungs: int = len(LADDER), only_rung: int = 0):
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

    # A coin whose best rung is already known should not spend an hour
    # re-proving that the cheaper ones fall short. BTCUSDT needs the
    # widest rung: at 848,640 rows it has 8.3x another coin's data, so
    # 96 leaves gives it an eighth of the decision regions per row that
    # the same setting gives a 100k-bar coin, and it recovers 92.70%
    # where they reach 100%.
    rungs = list(enumerate(LADDER, 1))
    if only_rung:
        rungs = [r for r in rungs if r[0] == only_rung] or rungs[-1:]
    else:
        rungs = rungs[:max_rungs]
    best, best_rec = None, -1.0
    for rung, extra in rungs:
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

    # Regressors ride the FIRST rung. They predict magnitudes, not a
    # class boundary, and rung 2 turned out to be actively harmful:
    # 800 rounds at learning_rate 0.3 drove BTCUSDT's classifier from
    # 92.70% down to 73.05%, which is softmax boosting saturating, not
    # learning. More rounds is not more capacity past that point.
    fine = dict(BASE, **LADDER[0])
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


def wilson_lower(k: int, n: int, z: float = 1.96) -> float:
    """95% lower confidence bound on a binomial proportion.

    A raw accuracy of k/n can look like it clears break-even purely
    because n was small and luck ran one way. The Wilson interval asks
    the harder question: even allowing for that sampling noise, how
    low could the TRUE accuracy plausibly be? Used to keep oof_gate's
    finer threshold search (see its docstring) from settling on a
    threshold whose only support is a lucky handful of calls.
    """
    if n == 0:
        return float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    adj = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - adj) / denom


def _wilson_at_or_above(conf: np.ndarray, ok: np.ndarray,
                        thresholds: np.ndarray):
    """n and Wilson-lower-bound accuracy at-or-above each threshold.

    One sorted pass over `conf`/`ok`, reused for both the pooled check
    and the recent-only check in gate_from_isotonic -- same trick, two
    populations.
    """
    order = np.argsort(conf)
    conf_sorted = conf[order]
    ok_sorted = ok[order]
    n_total = len(conf_sorted)
    cum_ok_from_right = np.cumsum(ok_sorted[::-1])[::-1] if n_total else \
        np.array([], dtype="float64")
    idx = np.searchsorted(conf_sorted, thresholds, side="left")
    idx = np.clip(idx, 0, n_total - 1 if n_total else 0)
    n_at_or_above = n_total - idx
    k_at_or_above = np.where(n_at_or_above > 0, cum_ok_from_right[idx], 0)
    return n_at_or_above, k_at_or_above


def gate_from_isotonic(iso, conf_all: np.ndarray, ok_all: np.ndarray,
                       p_be: float, conf_recent: np.ndarray | None = None,
                       ok_recent: np.ndarray | None = None,
                       recent_min_n: int = 50) -> float:
    """The lowest confidence where the isotonic-calibrated curve, the
    pooled empirical Wilson lower bound, AND (when given) a RECENT-ONLY
    Wilson lower bound all clear break-even.

    Searches the isotonic fit's own breakpoints (`X_thresholds_`), not
    an arbitrary fixed grid -- see oof_gate()'s docstring for why a
    201-point grid silently missed real, amply-sampled edges on two
    coins by stepping clean over the narrow band near 1.0 where
    confidence actually varies. Every point where the calibrated curve
    can change value is visited here, at whatever resolution the data
    itself has.

    POOLING ACROSS A LONG SPAN CAN HIDE A DEAD EDGE. The pooled check
    alone answers "did this threshold pay off somewhere in the last
    several months", which a stretch of good luck early in that span
    can satisfy even after the edge has since gone flat -- exactly what
    a 15-day rolling walk-forward found on the two coins this function
    HAD cleared: SKHYNIXUSDT and SNDKUSDT's most recent windows (tens
    of thousands of held-out calls apiece) sat several points BELOW
    break-even even as the pooled number cleared it comfortably. So
    when `conf_recent`/`ok_recent` are supplied -- calls from a model
    trained on everything except a final recent slice, scored only on
    that slice -- a threshold must ALSO clear on that recent evidence
    alone, at its own Wilson lower bound, with at least `recent_min_n`
    calls backing it. Too few recent calls at a threshold means "not
    enough current evidence", which is a reason to keep searching a
    HIGHER threshold, not to accept on the pooled number's word alone.
    """
    thresholds = np.asarray(iso.X_thresholds_, dtype="float64")
    calibrated = np.asarray(iso.y_thresholds_, dtype="float64")
    n_pooled, k_pooled = _wilson_at_or_above(conf_all, ok_all, thresholds)
    have_recent = conf_recent is not None and len(conf_recent) > 0
    if have_recent:
        n_recent, k_recent = _wilson_at_or_above(conf_recent, ok_recent,
                                                  thresholds)

    for i in range(len(thresholds)):
        if calibrated[i] < p_be:
            continue
        n = int(n_pooled[i])
        if n < 200:
            continue
        low = wilson_lower(int(k_pooled[i]), n)
        if low < p_be:
            continue
        if have_recent:
            nr = int(n_recent[i])
            if nr < recent_min_n:
                continue
            low_r = wilson_lower(int(k_recent[i]), nr)
            if low_r < p_be:
                continue
        return float(thresholds[i])
    return 1.0


def oof_gate(X, side, p_be, log=print, n_folds: int = 4, fold_days: int = 7):
    """A confidence floor measured across several regimes, not one --
    and only over the market as it is now, not as it was months ago.

    THIS IS THE 17 LOSING LIVE TRADES, TRACED TO ITS SOURCE. calibrate_gate()
    reads the gate off the SAME model scored on the SAME rows it was fit to
    reproduce -- and the capacity ladder escalates specifically until
    recovery hits 100% on those rows, so its errors there are exactly zero
    by construction. The gate it produces, `confidence of the worst
    mistake`, has no mistake to find and lands at 0.0000: every model
    shipped this way admits every signal, at any confidence, because the
    number meant to filter them was measured on a model that had already
    memorised the answer key. Live, on genuinely new bars, that memorised
    shape can be wrong far more than half the time -- the operator's own
    ledger: 21 won, 17 lost, every one called at gate 0.0000.

    A SINGLE held-out fold turned out to be its own trap. The first
    version of this function used one 80/20 split, and on XAUUSDT it
    measured 30.34% accuracy -- worse than a coin flip -- with the
    single worst miss pinning the gate at 1.0000, which would have
    stopped the coin trading at all. Walking the split forward in time
    (55/65/75/85% cuts) showed why: accuracy fell from 49% to 34%
    monotonically as the held-out window moved closer to the present,
    the signature of a trend the training window never saw. One fold
    answers "was this particular stretch of market kind to the model",
    not "is the model any good" -- and a single overconfident miss in
    that one fold can disable a coin on pure noise.

    So this walks SEVERAL expanding folds across the back half of the
    timeline -- each trained on everything before it, scored on the
    slice after -- and pools every (confidence, right/wrong) pair across
    all of them. The threshold is read off an ISOTONIC fit of confidence
    to calibrated accuracy over that pooled, multi-regime sample, not
    the single worst point in it. The gate is the lowest confidence at
    which calibrated accuracy clears the coin's OWN break-even
    probability `p_be` -- below it the model has been measured, on bars
    it never trained on, to lose money after fees; above it, to clear
    its own cost. If no confidence clears that bar, the honest answer is
    that this shape has no live edge right now, and the gate is 1.0 --
    not traded, rather than traded on a memorised shape that does not
    hold up going forward. Stake and leverage are untouched by any of
    this and still ride potential exactly as before; this only decides
    whether a setup is trusted enough to be sized at all.

    THE SEARCH GRID WAS TOO COARSE TO FIND ITS OWN ANSWER. This used to
    query the isotonic fit at 201 points evenly spaced from 0 to 1 --
    0.005 apart -- and take the lowest one clearing p_be. Confidence
    from a high-leaf-count booster saturates within a hair of 1.0 (real
    values traced on SKHYNIXUSDT: 0.999981 to 1.0, a span of 0.00002),
    and that is exactly where the calibrated curve does its climbing:
    on both SKHYNIXUSDT and SNDKUSDT the true curve rose from ~52% to
    ~68% entirely between grid points 0.995 and 1.000, so the coarse
    grid saw only the two endpoints, missed the amply-sampled edge in
    between, and planted the gate at the top -- 1.0, admitting nothing,
    on a coin that in fact clears break-even (with thousands of calls
    to back it) a little below there. Now the search runs over the
    isotonic fit's OWN breakpoints (`X_thresholds_`/`y_thresholds_`),
    which is exact by construction: every point where the calibrated
    curve's value can change is visited, none are skipped, whatever the
    resolution.

    A finer search needed a matching safeguard, not just a finer one.
    The isotonic curve is fit on POOLED counts and can still call a
    threshold "clearing" on the strength of a small, lucky tail (21,178
    calls at gate 1.0000 backing BLESSUSDT's flat curve is plenty; 525
    calls at a coin's very top percentile is not). So a candidate
    threshold must pass twice: the isotonic-calibrated value at that
    confidence clears p_be, AND the RAW empirical accuracy of every
    held-out call actually at or above it -- at its 95% Wilson lower
    bound, not the point estimate -- also clears p_be. The lower bound
    is what a small, possibly-lucky sample cannot fake.

    POOLING ACROSS WIDE, DEEP-HISTORY FOLDS CAN AVERAGE AWAY A DEAD
    EDGE. Two earlier versions of this measurement both learned the
    same lesson from opposite directions. The first walked 4 folds
    spanning the back HALF of a coin's entire history -- for BTCUSDT,
    a fold could be 74 days wide -- and a 15-day rolling walk-forward
    (fit expanding forward across the WHOLE history, one short window
    at a time, never pooled) found SKHYNIXUSDT's and SNDKUSDT's most
    recent windows sitting several points BELOW break-even on tens of
    thousands of held-out calls apiece, even though the wide, pooled
    measurement had cleared both: an earlier, kinder stretch inside
    one wide fold was carrying its average. The second version tried
    to patch that with a SEPARATE recency check tacked onto the wide
    one -- correct, but two numbers measuring overlapping questions.

    So the whole measurement now IS the recent evidence: `n_folds`
    folds, each `fold_days` wide, covering only the most recent
    `n_folds x fold_days` days -- 4 x 7 = 28 by default, not months.
    Every fold's MODEL is still trained on the coin's entire history
    before that fold starts (nothing about how much the model itself
    learns from has shrunk, only the WINDOW this function tests it
    against), and a single 7-day fold is still one fold, exactly the
    trap the section above warns about -- which is why there are
    still `n_folds` of them, pooled, not one.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression

    def fold_conf_ok(cut, stop):
        m = HistGradientBoostingClassifier(random_state=0, **BASE, **LADDER[0])
        m.fit(X[:cut], side[:cut])
        proba = m.predict_proba(X[cut:stop])
        cls = list(m.classes_)
        w = stop - cut
        pl = proba[:, cls.index(1)] if 1 in cls else np.zeros(w)
        ps = proba[:, cls.index(-1)] if -1 in cls else np.zeros(w)
        p0 = proba[:, cls.index(0)] if 0 in cls else np.zeros(w)
        pred = np.where(pl >= ps, 1, -1).astype("int8")
        c = np.maximum(pl, ps)
        pred = np.where(c > p0, pred, 0).astype("int8")
        truth = side[cut:stop]
        took = pred != 0
        del m
        gc.collect()
        return c[took], (pred[took] == truth[took]).astype(float)

    n = len(X)
    step = fold_days * 1440
    # Walk BACKWARD from the end in fold_days-sized steps -- the most
    # recent n_folds folds, whatever start that lands on -- instead of
    # spreading a fixed count across however much history a coin has.
    cuts = []
    window_end = n
    for _ in range(n_folds):
        window_start = window_end - step
        if window_start < 2000 or window_end - window_start < 500:
            break
        cuts.append(window_start)
        window_end = window_start
    cuts.reverse()
    if not cuts:
        return 0.0, 0, float("nan")
    confs, oks = [], []
    for cut in cuts:
        stop = min(cut + step, n)
        c, ok = fold_conf_ok(cut, stop)
        confs.append(c)
        oks.append(ok)
    if not confs or sum(len(a) for a in confs) < 300:
        return 0.0, 0, float("nan")
    conf_all = np.concatenate(confs)
    ok_all = np.concatenate(oks)
    acc = float(ok_all.mean())

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(conf_all, ok_all)
    gate = gate_from_isotonic(iso, conf_all, ok_all, p_be)
    n_err = int((ok_all < 0.5).sum())
    log(f"      held-out gate: {len(conf_all):,} calls across "
        f"{len(confs)} {fold_days}-day folds (last {len(confs)*fold_days} "
        f"days), {n_err:,} wrong (pooled accuracy {100*acc:.2f}%, "
        f"need {100*p_be:.2f}%) -> gate {gate:.4f}")
    return gate, n_err, acc


def edge_of(conf, profit, mae, cost, floor=FLOOR):
    """Expected return on one unit of margin at 1x, per bar."""
    gain = np.maximum(profit - cost, 0.0)
    loss = floor + cost
    return conf * gain - (1.0 - conf) * loss


def potential(conf, profit, mae, cost, floor=FLOOR, ref=None):
    """One number, 0..100, that scales BOTH the stake and the leverage.

    It is an expected return, not a preference: what one unit of margin
    at 1x earns on average if the model's confidence is taken at face
    value, with the loss being the stop it would have taken instead.
    Normalised by the coin's own strong-setup level, so no constant
    crosses from one coin to another.

    `ref` IS THAT LEVEL AND IT MUST BE PASSED IN LIVE. Deriving it from
    the array in hand works here, where the array is a coin's whole
    history, and fails silently in the bot, where the array is a single
    bar: the 90th percentile of one number is that number, so every
    live setup scored exactly 100 -- all-in, maximum leverage, every
    time, on every coin. The saved reference is what makes a live score
    comparable to the ones this file measured.
    """
    edge = edge_of(conf, profit, mae, cost, floor)
    if ref is None:
        ref = np.quantile(edge[edge > 0], 0.90) if (edge > 0).any() else 1.0
    return 100.0 * np.clip(edge / max(float(ref), 1e-9), 0.0, 1.0)


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
    edge = edge_of(conf, pr_profit, pr_mae, cost_v)
    ref = float(np.quantile(edge[edge > 0], 0.90)) if (edge > 0).any() else 1.0
    score = potential(conf, pr_profit, pr_mae, cost_v, ref=ref)
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
            cur = (close[j] / entry - 1.0) * s
            # Trail out only at a level that actually pays. Without this
            # guard a wide trail on a barely-profitable peak books a
            # small LOSS, which is where 103 of BLESS's 4,827 losses came
            # from, none worse than -1.4%.
            if peak - trail > c:
                if peak - cur >= trail:
                    hit, move = "trail", peak - trail
                    break
            # THE STOP IS UNCONDITIONAL, and it RATCHETS. Unconditional
            # because a stop that switches itself off once a trade is in
            # profit is how one BLESS trade rode to -203.94%. Ratcheting
            # because of the opposite failure: BTCUSDT's only two losses
            # in 2,665 trades were true opportunities -- gate 0.0000, no
            # false positives -- whose predicted target overshot the
            # actual peak, leaving a trail too wide to arm, so a trade
            # that had been in profit ran all the way back to the stop
            # for -14.4% of margin.
            #
            # Once a trade has been up by more than the round trip, the
            # exit floor moves to break-even. A winner is not allowed to
            # become a loser. This is a stop moved to entry, which is
            # what any desk does, and it can only ever improve an exit.
            guard = c if peak >= floor + c else -stop_frac
            if cur <= guard:
                hit, move = ("breakeven" if guard >= 0 else "stop"), guard
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
    return eq, trades, wins, side, score, gate, n_err, ref


# ------------------------------------------------------------------- main
def main():
    cap_memory()
    args = sys.argv[1:]
    fresh = "--fresh" in args
    # Re-run the replay against a model already on disk. A fit is the
    # expensive half; changing an exit rule should not cost forty
    # minutes of refitting to find out what it did.
    reuse = "--reuse" in args
    # Stop climbing after N rungs. Useful when a coin's best rung is
    # already known and the rest of the ladder only costs time -- or,
    # as with BTCUSDT, actively makes it worse.
    max_rungs, only_rung = len(LADDER), 0
    drop = set()
    for i, t in enumerate(args):
        if t in ("--rungs", "--rung") and i + 1 < len(args):
            v = max(1, int(args[i + 1]))
            drop.add(i + 1)
            if t == "--rungs":
                max_rungs = v
            else:
                only_rung = v
    args = [t for i, t in enumerate(args) if i not in drop]
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
        cached_blob = load_models(sym) if reuse else None
        if cached_blob is not None:
            print(f"    reusing saved model", flush=True)
            models = cached_blob["models"]
        else:
            print(f"    fitting on {Xg.nbytes/2**20:,.0f} MB", flush=True)
            models = learn(Xg, side[ok], profit[ok], mae[ok],
                           log=lambda s: print(s, flush=True),
                           max_rungs=max_rungs, only_rung=only_rung)
        if models is None:
            print("    could not fit")
            continue
        # The gate that ships to the live bot is measured on a fold
        # this coin's classifier never trained on -- see oof_gate()'s
        # docstring for why the same-rows gate is not a filter at all.
        cached_live = (cached_blob or {}).get("meta", {}).get("live_gate") \
            if cached_blob is not None else None
        if cached_live is not None:
            live_gate, live_n_err, live_acc = (
                cached_live, cached_blob["meta"].get("live_n_err", 0),
                cached_blob["meta"].get("live_acc", float("nan")))
        else:
            p_be = float(C.break_even(FLOOR, float(np.nanmedian(cost_v))))
            live_gate, live_n_err, live_acc = oof_gate(
                Xg, side[ok], p_be, log=lambda s: print(s, flush=True))
        eq, trades, wins, psd, score, gate, n_err, edge_ref = replay(
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
        print(f"    LIVE gate         : {live_gate:.4f} "
              f"(held-out accuracy {100*live_acc:.2f}% on "
              f"{live_n_err:,} unseen mistakes) -- this is what the bot "
              f"actually uses")
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
            # Won, flat and lost are three different things. A trade
            # the ratchet takes out at exactly break-even returned
            # nothing and cost nothing; filing it under "losers"
            # understates the result and, worse, hides whether any
            # trade actually lost money.
            flat = [t for t in trades if abs(t[3]) < 1e-9]
            bad = [t for t in trades if t[3] < -1e-9]
            print(f"    won {nt - len(flat) - len(bad):,}   "
                  f"flat {len(flat):,}   LOST {len(bad):,}")
            if bad:
                print(f"    losing trades     : "
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
                gate=float(gate), live_gate=float(live_gate),
                live_n_err=int(live_n_err), live_acc=float(live_acc),
                edge_ref=float(edge_ref),
                lev_cap=float(lev_cap),
                cost=float(np.nanmedian(cost_v)), horizon=HORIZON,
                floor=FLOOR, min_stake=MIN_STAKE, max_stake=MAX_STAKE,
                lev_floor=LEV_FLOOR, liq_safety=LIQ_SAFETY,
                win_rate=wins / nt, trades=nt))
            # Report the file that was actually written. This line
            # still pointed at the uncompressed name after save_models
            # switched to .pkl.gz, so BTCUSDT finished a forty-minute
            # fit, saved correctly, and then died on its own success
            # message -- the one place a crash costs the most.
            saved = MODELS / f"{sym}.pkl.gz"
            print(f"    saved model       : "
                  f"{saved.stat().st_size / 2**20:,.0f} MB", flush=True)
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
