"""A high-capacity model conditioned on candle-by-candle volume rhythm.

The case put, and it is a sharper one than anything tested earlier here:

  "Ten candles falling over ten minutes, with varying volume -- that is a
   short. Days later the same shape appears, but the volume is different,
   the rhythm is different, the volume DIFFERENCE BETWEEN CANDLES is
   different, and so the trend is different. I want a general logic that
   gets each of those small cases right."

The distinction being drawn is real, and it is one the earlier scripts in
this folder never actually tested. research/pattern_matching.py described
volume only as a level (this candle's volume against a moving average).
It never described the RHYTHM: how volume changed from each candle to the
next, whether the selling was accelerating or drying up, whether volume
concentrated on the down candles or the up ones. Two sequences can have
the same shape and opposite volume rhythm, and nothing tested so far
could tell them apart.

So this builds the largest model that can be built here, on features
designed around that objection:

  - per-candle returns and ranges over the last 20 candles
  - per-candle volume, relative to its own recent average
  - candle-to-candle volume RATIOS -- the rhythm itself
  - volume split by direction: how much traded on down candles versus up
  - volume-weighted momentum, and volume/return correlation
  - acceleration: is the move speeding up or stalling
  - run structure: consecutive same-direction candles and their volume
  - multi-scale trend context

~60 features, then gradient-boosted regression trees (implemented here
directly; scikit-learn is not available in this environment) with enough
depth and enough trees to carve out fine-grained cases rather than
averaging them together.

The label is the honest one: the net return of a REAL trade opened at
that bar -- fixed ATR stop, fixed target, hard time exit, fees and
slippage charged, stop taken first whenever a bar's range covers both.
Not the best price in the window.

Two conditions from the request are applied as stated:
  - setups whose target does not clear the round-trip cost are discarded
    outright rather than traded for a small gross gain that fees erase
  - the model is trained on one year and tested on the other, in both
    directions, because a rule that only works one way round is a
    coincidence

Run:  python3 -m research.deep_market_model
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bot import risk

MAX_LEVERAGE = risk.ABSOLUTE_MAX_LEVERAGE
FEE_ROUNDTRIP = 2 * (risk.TAKER_FEE_PCT + risk.ASSUMED_SLIPPAGE_PCT)
DATASETS = {"2025": "data/BTCUSDT_2025.csv", "2026": "data/BTCUSDT_2026.csv"}

LOOKBACK = 20          # candles of rhythm the model sees
HORIZON = 120          # max holding time, minutes
STOP_ATR = 2.0
TARGET_ATR = 4.0

# Gradient boosting settings. Deep enough to separate fine cases rather
# than averaging them, with shrinkage and subsampling so it generalises.
N_TREES = 120
MAX_DEPTH = 6
LEARNING_RATE = 0.05
MIN_LEAF = 400
N_BINS = 32
SUBSAMPLE_ROWS = 220_000
RNG = np.random.default_rng(7)


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def build_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    o, h, l, c, v = (df[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    n = len(c)
    eps = 1e-12

    ret = np.zeros(n)
    ret[1:] = c[1:] / c[:-1] - 1.0
    vol_ma = pd.Series(v).rolling(240, min_periods=60).mean().to_numpy()
    vol_rel = v / np.maximum(vol_ma, eps)
    # The rhythm: how volume changed from the previous candle to this one.
    vol_ratio = np.ones(n)
    vol_ratio[1:] = v[1:] / np.maximum(v[:-1], eps)
    rng_pct = (h - l) / np.maximum(c, eps)
    body = (c - o) / np.maximum(h - l, eps)

    scale = pd.Series(ret).rolling(LOOKBACK, min_periods=LOOKBACK).std().to_numpy()

    cols, names = [], []

    def add(arr, name):
        cols.append(arr.astype(np.float32))
        names.append(name)

    # Per-candle detail over the lookback window.
    for k in range(1, LOOKBACK + 1):
        add(np.roll(ret, k - 1) / np.maximum(scale, eps), f"ret_lag{k}")
        add(np.roll(vol_rel, k - 1), f"volrel_lag{k}")
    for k in range(1, 11):
        add(np.log(np.maximum(np.roll(vol_ratio, k - 1), eps)), f"volrhythm_lag{k}")
        add(np.roll(body, k - 1), f"body_lag{k}")

    s = pd.Series(ret)
    sv = pd.Series(v)
    up = pd.Series((ret > 0).astype(float))
    down = pd.Series((ret < 0).astype(float))

    # Where the volume actually went: onto the down candles or the up ones?
    vol_down = pd.Series(v * down).rolling(LOOKBACK, min_periods=LOOKBACK).sum().to_numpy()
    vol_up = pd.Series(v * up).rolling(LOOKBACK, min_periods=LOOKBACK).sum().to_numpy()
    add(vol_down / np.maximum(vol_down + vol_up, eps), "vol_share_on_down")

    add(s.rolling(LOOKBACK).sum().to_numpy() / np.maximum(scale, eps), "cum_ret")
    add(pd.Series(ret * vol_rel).rolling(LOOKBACK).sum().to_numpy()
        / np.maximum(scale, eps), "vol_weighted_ret")
    add(s.rolling(LOOKBACK).corr(sv).to_numpy(), "corr_ret_volume")
    add(s.rolling(LOOKBACK).std().to_numpy() / np.maximum(
        s.rolling(LOOKBACK * 5, min_periods=LOOKBACK).std().to_numpy(), eps), "vol_of_vol")
    add(down.rolling(LOOKBACK).sum().to_numpy(), "n_down_candles")

    # Acceleration: is the second half of the window moving faster?
    half = LOOKBACK // 2
    add((s.rolling(half).sum().to_numpy()
         - s.shift(half).rolling(half).sum().to_numpy()) / np.maximum(scale, eps),
        "accel_return")
    add(sv.rolling(half).mean().to_numpy()
        / np.maximum(sv.shift(half).rolling(half).mean().to_numpy(), eps), "accel_volume")

    # Longer context so the same rhythm is read differently in different regimes.
    for span in (60, 240, 1440):
        ema = pd.Series(c).ewm(span=span, adjust=False).mean().to_numpy()
        add(c / np.maximum(ema, eps) - 1.0, f"dist_ema{span}")
    add(pd.Series(c).pct_change(240).to_numpy(), "ret_4h")
    add(pd.Series(rng_pct).rolling(LOOKBACK).mean().to_numpy(), "avg_range")

    X = np.column_stack(cols)
    return X, names


def block_atr(df: pd.DataFrame, block: int = 15, span: int = 14) -> np.ndarray:
    high = df["high"].rolling(block).max()
    low = df["low"].rolling(block).min()
    prev = df["close"].shift(block)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(span=span, adjust=False).mean().to_numpy()


def realised_net(df: pd.DataFrame, side: str, atr: np.ndarray) -> np.ndarray:
    """What a real trade opened at each bar would have returned, net."""
    c = df["close"].to_numpy(float)
    long = side == "long"
    stop_d, targ_d = STOP_ATR * atr, TARGET_ATR * atr

    fut_low = df["low"].rolling(HORIZON).min().shift(-HORIZON).to_numpy()
    fut_high = df["high"].rolling(HORIZON).max().shift(-HORIZON).to_numpy()
    fut_close = df["close"].shift(-HORIZON).to_numpy()

    if long:
        hit_sl = fut_low <= c - stop_d
        hit_tp = fut_high >= c + targ_d
        timeout = (fut_close - c) / c
    else:
        hit_sl = fut_high >= c + stop_d
        hit_tp = fut_low <= c - targ_d
        timeout = (c - fut_close) / c

    gross = np.where(hit_sl, -stop_d / c, np.where(hit_tp, targ_d / c, timeout))
    net = gross - FEE_ROUNDTRIP
    net[~np.isfinite(fut_close) | ~np.isfinite(atr) | (atr <= 0)] = np.nan
    return net


def worth_trading(df: pd.DataFrame, atr: np.ndarray) -> np.ndarray:
    """Discard setups whose target cannot clear the round trip -- the
    request's "drop the small trades that are negative before fees"."""
    return (TARGET_ATR * atr / df["close"].to_numpy(float)) > FEE_ROUNDTRIP


# ---------------------------------------------------------------------------
# Gradient-boosted regression trees (histogram split finding, numpy only)
# ---------------------------------------------------------------------------
def bin_features(X: np.ndarray, edges: list[np.ndarray] | None = None):
    if edges is None:
        edges = [np.nanquantile(X[:, j], np.linspace(0, 1, N_BINS + 1)[1:-1])
                 for j in range(X.shape[1])]
        edges = [np.unique(e) for e in edges]
    B = np.empty(X.shape, dtype=np.uint8)
    for j in range(X.shape[1]):
        B[:, j] = np.digitize(X[:, j], edges[j]).astype(np.uint8)
    return B, edges


def _best_split(B, resid, idx, n_features):
    best = None
    total_sum, total_cnt = resid[idx].sum(), len(idx)
    for j in range(n_features):
        bins = B[idx, j]
        cnt = np.bincount(bins, minlength=N_BINS + 1).astype(np.float64)
        ssum = np.bincount(bins, weights=resid[idx], minlength=N_BINS + 1)
        left_cnt, left_sum = np.cumsum(cnt), np.cumsum(ssum)
        right_cnt, right_sum = total_cnt - left_cnt, total_sum - left_sum
        valid = (left_cnt >= MIN_LEAF) & (right_cnt >= MIN_LEAF)
        if not valid.any():
            continue
        gain = np.where(valid,
                        left_sum ** 2 / np.maximum(left_cnt, 1)
                        + right_sum ** 2 / np.maximum(right_cnt, 1)
                        - total_sum ** 2 / max(total_cnt, 1),
                        -np.inf)
        b = int(np.argmax(gain))
        if best is None or gain[b] > best[0]:
            best = (gain[b], j, b)
    return best


def _build_tree(B, resid, idx, depth, n_features):
    if depth >= MAX_DEPTH or len(idx) < 2 * MIN_LEAF:
        return {"leaf": float(resid[idx].mean())}
    split = _best_split(B, resid, idx, n_features)
    if split is None or not np.isfinite(split[0]) or split[0] <= 0:
        return {"leaf": float(resid[idx].mean())}
    _, j, b = split
    mask = B[idx, j] <= b
    li, ri = idx[mask], idx[~mask]
    if len(li) < MIN_LEAF or len(ri) < MIN_LEAF:
        return {"leaf": float(resid[idx].mean())}
    return {"feat": j, "bin": b,
            "left": _build_tree(B, resid, li, depth + 1, n_features),
            "right": _build_tree(B, resid, ri, depth + 1, n_features)}


def _predict_tree(node, B):
    out = np.empty(len(B), dtype=np.float64)
    stack = [(node, np.arange(len(B)))]
    while stack:
        nd, idx = stack.pop()
        if "leaf" in nd:
            out[idx] = nd["leaf"]
            continue
        mask = B[idx, nd["feat"]] <= nd["bin"]
        stack.append((nd["left"], idx[mask]))
        stack.append((nd["right"], idx[~mask]))
    return out


def fit_gbdt(B, y):
    base = float(y.mean())
    pred = np.full(len(y), base)
    trees = []
    n_features = B.shape[1]
    for _ in range(N_TREES):
        resid = y - pred
        sub = RNG.choice(len(y), min(len(y), SUBSAMPLE_ROWS), replace=False)
        tree = _build_tree(B, resid, sub, 0, n_features)
        trees.append(tree)
        pred += LEARNING_RATE * _predict_tree(tree, B)
    return base, trees


def predict_gbdt(model, B):
    base, trees = model
    out = np.full(len(B), base)
    for t in trees:
        out += LEARNING_RATE * _predict_tree(t, B)
    return out


def run(train_df, test_df, side, label):
    Xtr, names = build_features(train_df)
    Xte, _ = build_features(test_df)
    atr_tr, atr_te = block_atr(train_df), block_atr(test_df)
    ytr = realised_net(train_df, side, atr_tr)
    yte = realised_net(test_df, side, atr_te)

    ok_tr = np.isfinite(Xtr).all(1) & np.isfinite(ytr) & worth_trading(train_df, atr_tr)
    ok_te = np.isfinite(Xte).all(1) & np.isfinite(yte) & worth_trading(test_df, atr_te)
    if ok_tr.sum() < 5000 or ok_te.sum() < 2000:
        print(f"  {label} {side}: not enough usable bars "
              f"(train {ok_tr.sum():,}, test {ok_te.sum():,})")
        return None

    Btr, edges = bin_features(Xtr[ok_tr])
    Bte, _ = bin_features(Xte[ok_te], edges)
    model = fit_gbdt(Btr, ytr[ok_tr])

    p_tr = predict_gbdt(model, Btr)
    p_te = predict_gbdt(model, Bte)
    y_te_ok = yte[ok_te]

    rows = []
    for pct in (0.1, 0.5, 1.0, 5.0):
        thr = np.percentile(p_te, 100 - pct)
        take = p_te >= thr
        if take.sum() < 20:
            continue
        rows.append({
            "top_pct": pct, "trades": int(take.sum()),
            "avg_net_pct": round(100 * float(y_te_ok[take].mean()), 4),
            "win_rate_pct": round(100 * float((y_te_ok[take] > 0).mean()), 2),
        })
    return {
        "features": Xtr.shape[1],
        "train_bars": int(ok_tr.sum()),
        "test_bars": int(ok_te.sum()),
        "train_fit_corr": round(float(np.corrcoef(p_tr, ytr[ok_tr])[0, 1]), 4),
        "test_corr": round(float(np.corrcoef(p_te, y_te_ok)[0, 1]), 4),
        "baseline_pct": round(100 * float(y_te_ok.mean()), 4),
        "rows": rows,
    }


def main() -> None:
    print(f"Gradient-boosted trees: {N_TREES} trees, depth {MAX_DEPTH}, "
          f"lr {LEARNING_RATE}, min leaf {MIN_LEAF}")
    print(f"Label: realised net return of a real trade "
          f"(stop {STOP_ATR} ATR, target {TARGET_ATR} ATR, {HORIZON}m max hold, "
          f"{FEE_ROUNDTRIP*100:.3f}% cost)")
    print("Setups whose target cannot clear the round trip are discarded.\n")

    data = {y: load(p) for y, p in DATASETS.items()}
    summary = {}
    for side in ("short", "long"):
        for lib in DATASETS:
            test = next(y for y in DATASETS if y != lib)
            label = f"{lib}->{test}"
            print(f"--- {side.upper()} {label} ---")
            r = run(data[lib], data[test], side, label)
            if r is None:
                continue
            print(f"  {r['features']} features | train {r['train_bars']:,} bars, "
                  f"test {r['test_bars']:,} bars")
            print(f"  in-sample fit corr {r['train_fit_corr']:+.4f} | "
                  f"OUT-OF-SAMPLE corr {r['test_corr']:+.4f}")
            print(f"  baseline (all bars): {r['baseline_pct']:+.4f}% per trade")
            for row in r["rows"]:
                print(f"     top {row['top_pct']:>4.1f}% of predictions: "
                      f"{row['trades']:>6,d} trades  "
                      f"win {row['win_rate_pct']:5.2f}%  "
                      f"net {row['avg_net_pct']:+.4f}% per trade")
            summary[(side, label)] = r
            print()

    print("=" * 78)
    good = []
    for (side, label), r in summary.items():
        for row in r["rows"]:
            if row["avg_net_pct"] > 0:
                good.append((side, label, row))
    if good:
        print("Positive out-of-sample selections found:")
        for side, label, row in sorted(good, key=lambda g: -g[2]["avg_net_pct"]):
            print(f"  {side} {label} top {row['top_pct']}%: "
                  f"{row['avg_net_pct']:+.4f}% over {row['trades']:,} trades")
        print("\nCheck whether the same side is positive in BOTH cross-year")
        print("directions before treating any of this as real.")
    else:
        print("No selection at any confidence level was profitable out-of-sample.")
        best = max((row for r in summary.values() for row in r["rows"]),
                   key=lambda x: x["avg_net_pct"], default=None)
        if best:
            print(f"Best: {best['avg_net_pct']:+.4f}% per trade "
                  f"over {best['trades']:,} trades.")


if __name__ == "__main__":
    main()
