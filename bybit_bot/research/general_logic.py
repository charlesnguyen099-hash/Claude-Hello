"""Learn a general market logic from the data, not a table of past trades.

The point made in the request, and it is the right one: a rule that
demands this exact price, this exact volume, this exact candle count is
useless, because that never recurs. What the bot needs is a general
understanding -- what the trend is doing, what volume is doing, where to
get in -- learned from history and applied to comparable situations.

So this describes the market with a handful of broad, regime-level
properties, each cut into a few coarse levels:

  trend position   where price sits relative to its long trend
  trend strength   how firmly the fast/slow averages are separated
  volume           current volume against its own recent average
  volatility       current volatility against its own recent average
  momentum         where price has come from over the last hour

That gives a few hundred market "situations", not billions -- each one
broad enough that thousands of past bars fall into it, so the average
behaviour inside it is a real statistic rather than a memorised moment.

It also fixes a flaw in the earlier studies in this folder. Those labelled
an opportunity by the BEST price reached inside the window, which assumes
a perfect exit. Here every bar is labelled with what a real trade would
actually have returned: fixed stop, fixed target, hard time exit, fees and
slippage charged, and stop-first whenever a bar's range covers both. That
is the number the bot has to make positive, so it is the number the model
is fitted on.

The rule learned is simply: for each situation, what did trades opened in
it actually return on average, in the training year? Trade the situations
whose average was positive. Then check that on the other year.

A situation only counts as usable if it was profitable in BOTH cross-year
directions. Picking the ones that worked on a single year is how you build
something that looks excellent and then loses money live.

Run:  python3 -m research.general_logic
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bot import risk

MAX_LEVERAGE = risk.ABSOLUTE_MAX_LEVERAGE
FEE_ROUNDTRIP = 2 * (risk.TAKER_FEE_PCT + risk.ASSUMED_SLIPPAGE_PCT)
DATASETS = {"2025": "data/BTCUSDT_2025.csv", "2026": "data/BTCUSDT_2026.csv"}

HORIZON = 120          # max holding time in minutes
STOP_ATR = 2.0         # stop distance, in ATR
TARGET_ATR = 4.0       # target distance, in ATR (2R)
MIN_CELL_SAMPLES = 300  # a situation must be common enough to mean anything
BIN_COUNTS = [3, 4, 5]


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def market_state(df: pd.DataFrame) -> pd.DataFrame:
    """Broad regime descriptors. Deliberately coarse and all backward."""
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
    f = pd.DataFrame(index=df.index)

    ema_fast = close.ewm(span=135, adjust=False).mean()
    ema_slow = close.ewm(span=315, adjust=False).mean()
    ema_trend = close.ewm(span=1440, adjust=False).mean()   # ~1 day

    f["trend_position"] = close / ema_trend - 1.0
    f["trend_strength"] = (ema_fast - ema_slow) / close
    f["volume_rel"] = vol / vol.rolling(1440).mean()
    ret = close.pct_change()
    f["volatility_rel"] = ret.rolling(60).std() / ret.rolling(1440).std()
    f["momentum"] = close.pct_change(60)
    return f


def true_atr(df: pd.DataFrame, block: int = 15, span: int = 14) -> pd.Series:
    """ATR measured over `block`-minute windows so its units match a real
    swing rather than a single 1-minute candle."""
    high = df["high"].rolling(block).max()
    low = df["low"].rolling(block).min()
    prev = df["close"].shift(block)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(span=span, adjust=False).mean()


def realised_net(df: pd.DataFrame, side: str, atr: pd.Series) -> np.ndarray:
    """What a real trade opened at each bar would actually have returned.

    Conservative on same-bar ambiguity: if the window's range covers both
    the stop and the target, the stop is taken. Fees charged on every
    trade, win or lose.
    """
    close = df["close"].to_numpy(float)
    a = atr.to_numpy(float)
    n = len(close)
    long = side == "long"

    stop_d, targ_d = STOP_ATR * a, TARGET_ATR * a
    fut_low = df["low"].rolling(HORIZON).min().shift(-HORIZON).to_numpy()
    fut_high = df["high"].rolling(HORIZON).max().shift(-HORIZON).to_numpy()
    fut_close = df["close"].shift(-HORIZON).to_numpy()

    if long:
        sl, tp = close - stop_d, close + targ_d
        hit_sl, hit_tp = fut_low <= sl, fut_high >= tp
        time_exit = (fut_close - close) / close
        gross = np.where(hit_sl, -stop_d / close,
                         np.where(hit_tp, targ_d / close, time_exit))
    else:
        sl, tp = close + stop_d, close - targ_d
        hit_sl, hit_tp = fut_high >= sl, fut_low <= tp
        time_exit = (close - fut_close) / close
        gross = np.where(hit_sl, -stop_d / close,
                         np.where(hit_tp, targ_d / close, time_exit))

    net = gross - FEE_ROUNDTRIP
    net[~np.isfinite(fut_close) | ~np.isfinite(a) | (a <= 0)] = np.nan
    return net


def cells(train: pd.DataFrame, frames: list[pd.DataFrame], bins: int, cols: list[str]):
    """Bin each descriptor at the TRAIN year's quantiles."""
    edges = {c: np.unique(np.nanquantile(train[c].to_numpy(),
                                         np.linspace(0, 1, bins + 1)[1:-1]))
             for c in cols}
    out = []
    for fr in frames:
        code = np.zeros(len(fr), dtype=np.int64)
        ok = np.ones(len(fr), dtype=bool)
        mult = 1
        for c in cols:
            v = fr[c].to_numpy()
            ok &= np.isfinite(v)
            code += np.digitize(v, edges[c]) * mult
            mult *= len(edges[c]) + 1
        out.append((code, ok))
    return out


def learn_and_test(train_df, test_df, bins, cols, side):
    ftr, fte = market_state(train_df), market_state(test_df)
    atr_tr, atr_te = true_atr(train_df), true_atr(test_df)
    y_tr = realised_net(train_df, side, atr_tr)
    y_te = realised_net(test_df, side, atr_te)

    (c_tr, ok_tr), (c_te, ok_te) = cells(ftr, [ftr, fte], bins, cols)
    ok_tr &= np.isfinite(y_tr)
    ok_te &= np.isfinite(y_te)

    # Average realised net return of each market situation, in the train year.
    s = pd.DataFrame({"cell": c_tr[ok_tr], "y": y_tr[ok_tr]})
    stats = s.groupby("cell")["y"].agg(["mean", "count"])
    good = stats[(stats["count"] >= MIN_CELL_SAMPLES) & (stats["mean"] > 0)]

    take = np.isin(c_te, good.index.to_numpy()) & ok_te
    if take.sum() == 0:
        return None
    sel = y_te[take]
    baseline = y_te[ok_te]
    return {
        "situations_total": int(len(stats)),
        "situations_kept": int(len(good)),
        "train_expected_pct": round(100 * float(good["mean"].mean()), 4),
        "bars_traded": int(take.sum()),
        "test_avg_net_pct": round(100 * float(sel.mean()), 4),
        "test_baseline_pct": round(100 * float(baseline.mean()), 4),
        "test_win_rate_pct": round(100 * float((sel > 0).mean()), 2),
    }


def fee_sensitivity(data: dict[str, pd.DataFrame]) -> None:
    """At what trading cost does this become profitable?

    Everything measured so far says the same thing: the gross edge before
    costs is about zero, and the loss per trade is about the round trip.
    If that is right, cost is the only lever left -- so sweep it.

    The levels are real Bybit numbers, not hypotheticals:
      0.210%  taker both ways + 0.05% assumed slippage each way (current)
      0.110%  taker both ways, no slippage
      0.075%  taker in, maker out
      0.040%  maker both ways (limit entry AND limit exit)
      0.000%  a hypothetical zero-cost exchange, as an upper bound
    """
    print("\n" + "=" * 78)
    print("FEE SENSITIVITY: how cheap would trading have to be?")
    print("=" * 78)
    print("The bot already rests a LIMIT entry, so the maker rate on the way in")
    print("is reachable today; a limit take-profit would earn it on the way out.\n")

    cols = ["trend_position", "trend_strength", "volume_rel", "volatility_rel", "momentum"]
    levels = [0.00210, 0.00110, 0.00075, 0.00040, 0.0]
    global FEE_ROUNDTRIP
    original = FEE_ROUNDTRIP

    print(f"{'cost':>7s} | {'2025->2026 long':>16s} {'2026->2025 long':>16s} "
          f"{'2025->2026 short':>17s} {'2026->2025 short':>17s}")
    print("-" * 78)
    try:
        for cost in levels:
            FEE_ROUNDTRIP = cost
            cells_out = []
            for side in ("long", "short"):
                for lib in ("2025", "2026"):
                    test = "2026" if lib == "2025" else "2025"
                    r = learn_and_test(data[lib], data[test], 4, cols, side)
                    cells_out.append("none" if r is None
                                     else f"{r['test_avg_net_pct']:+.4f}%")
            print(f"{cost*100:6.3f}% | {cells_out[0]:>16s} {cells_out[1]:>16s} "
                  f"{cells_out[2]:>17s} {cells_out[3]:>17s}")
    finally:
        FEE_ROUNDTRIP = original

    print("\nIf the numbers only turn positive at 0.000%, the gross edge really is")
    print("zero and no fee tier saves it. If they turn positive at 0.040%, then")
    print("maker-only execution is the whole difference and it is worth building.")


def main() -> None:
    print(f"Stop {STOP_ATR} ATR, target {TARGET_ATR} ATR, max hold {HORIZON}m, "
          f"round trip {FEE_ROUNDTRIP*100:.3f}%")
    print("Labels are the realised net return of a real trade (stop-first on")
    print("ambiguous bars), NOT the best price in the window.\n")

    data = {y: load(p) for y, p in DATASETS.items()}
    col_sets = {
        "trend+vol": ["trend_position", "trend_strength", "volume_rel"],
        "trend+vola": ["trend_position", "trend_strength", "volatility_rel"],
        "all5": ["trend_position", "trend_strength", "volume_rel",
                 "volatility_rel", "momentum"],
    }

    hdr = (f"{'features':11s} {'bins':>4s} {'dir':>5s} {'lib->test':>12s} | "
           f"{'kept':>9s} {'trainExp%':>9s} | {'bars':>8s} {'testNet%':>9s} "
           f"{'base%':>8s} {'win%':>6s}")
    print(hdr)
    print("-" * len(hdr))

    results = {}
    for name, cols in col_sets.items():
        for bins in BIN_COUNTS:
            for side in ("long", "short"):
                for lib in DATASETS:
                    test = next(y for y in DATASETS if y != lib)
                    r = learn_and_test(data[lib], data[test], bins, cols, side)
                    if r is None:
                        continue
                    results.setdefault((name, bins, side), []).append(r)
                    print(f"{name:11s} {bins:4d} {side:>5s} {lib+'->'+test:>12s} | "
                          f"{r['situations_kept']:4d}/{r['situations_total']:<4d} "
                          f"{r['train_expected_pct']:9.4f} | {r['bars_traded']:8,d} "
                          f"{r['test_avg_net_pct']:9.4f} {r['test_baseline_pct']:8.4f} "
                          f"{r['test_win_rate_pct']:6.2f}")

    print("\n" + "=" * 78)
    robust = {k: v for k, v in results.items()
              if len(v) == 2 and all(x["test_avg_net_pct"] > 0 for x in v)}
    if robust:
        print("Situations profitable out-of-sample in BOTH directions:")
        for (name, bins, side), v in sorted(
                robust.items(), key=lambda kv: -sum(x["test_avg_net_pct"] for x in kv[1])):
            print(f"  {name} bins={bins} {side}: "
                  f"{v[0]['test_avg_net_pct']:+.4f}% and {v[1]['test_avg_net_pct']:+.4f}% "
                  f"per trade")
        print("\nThis is a real edge -- next step is wiring it into bot/strategy.py")
        print("and re-running the full engine with sizing and leverage.")
    else:
        print("Nothing was profitable out-of-sample in both directions.")
        best = max((x for v in results.values() for x in v),
                   key=lambda r: r["test_avg_net_pct"])
        print(f"Best single out-of-sample average: {best['test_avg_net_pct']:+.4f}% "
              f"per trade over {best['bars_traded']:,} bars "
              f"(baseline {best['test_baseline_pct']:+.4f}%).")

    fee_sensitivity(data)


if __name__ == "__main__":
    main()
