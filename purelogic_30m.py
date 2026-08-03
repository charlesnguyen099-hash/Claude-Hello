"""The PureLogic rule, rebuilt on the timeframe the table was actually computed on.

The first pass at this used 1-minute features and they did not line up with
the table. They were not supposed to: fitting the table's own numbers against
resampled BTCUSDT 2025-2026 identifies the source as **30-minute bars**, with
a log-RMSE of 0.065 across six independent scale probes:

    feature            table    30m bars
    ret_lag1 (std)     0.3555     0.3406
    atr14_pct          0.39       0.4104
    range_pct          0.38       0.3543
    realized_vol_20    0.24       0.2462
    dist_ema21_pct     0.36       0.3184
    bb_width_20_2      1.50       1.4713

Every other timeframe is far worse (1m: 1.843, 15m: 0.408, 60m: 0.358), so
this is not a close call. The trade timing stays at 1-minute resolution —
the durations sum to 832,928 minutes against 832,979 available, which only
works if entries and exits are placed on 1m bars.

So the correct construction, and what this script does: compute the 106
features on 30-minute bars, carry each completed 30m reading forward onto
the 1m bars that follow it (never onto the bar it was still forming on),
then trade at 1m resolution. Direction comes from the table; profit comes
from what BTCUSDT actually did, at the table's own 0.11% round trip.

Run:  python3 purelogic_30m.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "bybit_bot")
from purelogic import features as F  # noqa: E402

TABLE = ("/root/.claude/uploads/2499e73f-5145-5c6f-b255-816732633901/"
         "d6bcbae7-Sheet16_PureLogic_49783_1.txt")
DATASETS = {"2025": "bybit_bot/data/BTCUSDT_2025.csv",
            "2026": "bybit_bot/data/BTCUSDT_2026.csv"}
FEE_PCT = 0.11 / 100.0
BAR_MINUTES = 30

SUBSETS = {
    "4-feature": ["rsi14", "stoch_k14", "range_pos_20", "consec_streak"],
    "6-feature": ["rsi14", "stoch_k14", "range_pos_20", "consec_streak",
                  "vol_ratio_20", "adx14"],
    "8-feature": ["rsi14", "stoch_k14", "range_pos_20", "consec_streak",
                  "vol_ratio_20", "adx14", "bb_pctb_20_2", "willr14"],
    "10-feature": ["rsi14", "stoch_k14", "range_pos_20", "consec_streak",
                   "vol_ratio_20", "adx14", "bb_pctb_20_2", "willr14",
                   "macd_hist", "atr14_pct"],
}
BIN_COUNTS = [3, 4, 5]


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def features_30m_on_1m(df_1m: pd.DataFrame) -> pd.DataFrame:
    """30-minute features, aligned onto 1-minute bars without lookahead.

    A 30m bar stamped 10:00 only *closes* at 10:30, so its reading is
    stamped forward to 10:30 and merged backward from there. A 1m bar at
    10:05 therefore sees the 09:30 bar's values, which is what a live bot
    would have had.
    """
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    bars = (df_1m.set_index("datetime").resample(f"{BAR_MINUTES}min")
            .agg(agg).dropna().reset_index())
    feats = F.build(bars)
    feats["datetime"] = feats["datetime"] + pd.Timedelta(minutes=BAR_MINUTES)
    return pd.merge_asof(
        df_1m[["datetime"]].sort_values("datetime"),
        feats.sort_values("datetime"),
        on="datetime", direction="backward",
    )


def to_rank(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref = np.sort(reference[np.isfinite(reference)])
    if ref.size == 0:
        return np.full(values.shape, np.nan)
    out = np.searchsorted(ref, values, side="left") / ref.size
    return np.where(np.isfinite(values), out, np.nan)


def encode(ranks: np.ndarray, bins: int) -> np.ndarray:
    codes = np.zeros(len(ranks), dtype=np.int64)
    ok = np.ones(len(ranks), dtype=bool)
    for j in range(ranks.shape[1]):
        col = ranks[:, j]
        ok &= np.isfinite(col)
        codes = codes * bins + np.clip((np.nan_to_num(col) * bins).astype(np.int64),
                                       0, bins - 1)
    return np.where(ok, codes, -1)


def build_rules(table: pd.DataFrame, cols: list[str], bins: int) -> pd.DataFrame:
    ranks = np.column_stack([to_rank(table[c].to_numpy(float),
                                     table[c].to_numpy(float)) for c in cols])
    t = pd.DataFrame({
        "cell": encode(ranks, bins),
        "is_long": (table["direction"] == "LONG").to_numpy(),
        "dur": table["duration_min"].to_numpy(float),
    })
    t = t[t["cell"] >= 0]
    g = t.groupby("cell").agg(n=("is_long", "size"),
                              long_share=("is_long", "mean"),
                              median_dur=("dur", "median"))
    g["direction"] = np.where(g["long_share"] >= 0.5, 1, -1)
    g["conviction"] = (g["long_share"] - 0.5).abs() * 2.0
    return g


def simulate(close: np.ndarray, cell: np.ndarray, rules: pd.DataFrame) -> dict:
    direction = dict(zip(rules.index, rules["direction"]))
    duration = dict(zip(rules.index, rules["median_dur"]))
    n = len(close)
    pnl, busy = [], -1
    for i in range(n - 1):
        c = cell[i]
        if c < 0 or c not in direction or i <= busy:
            continue
        hold = int(max(1, min(duration[c], 500)))
        j = min(i + hold, n - 1)
        pnl.append((close[j] - close[i]) / close[i] * direction[c] - FEE_PCT)
        busy = j
    if not pnl:
        return {"trades": 0}
    a = np.array(pnl)
    return {
        "trades": len(a),
        "win_rate_pct": round(100 * float((a > 0).mean()), 2),
        "avg_gross_pct": round(100 * float(a.mean() + FEE_PCT), 4),
        "avg_net_pct": round(100 * float(a.mean()), 4),
        "total_pct": round(100 * float(a.sum()), 1),
    }


def main() -> None:
    table = pd.read_csv(TABLE, sep="\t")
    candles = {y: load(p) for y, p in DATASETS.items()}

    print("=" * 100)
    print(f"PureLogic rule on {BAR_MINUTES}-minute features (the timeframe the table was built on)")
    print("=" * 100)
    print("Direction from the table; profit from real BTCUSDT movement; "
          f"{FEE_PCT*100:.2f}% round trip.")
    print("GROSS is the edge before costs — the number that has to clear the fee.\n")

    feats = {y: features_30m_on_1m(df) for y, df in candles.items()}

    hdr = (f"{'rule keyed on':13s} {'bins':>4s} {'cells':>6s} {'conv':>5s} | " + " | ".join(
        f"{y} {'trades':>7s} {'win%':>6s} {'gross%':>8s} {'net%':>8s} {'total%':>9s}"
        for y in DATASETS))
    print(hdr)
    print("-" * len(hdr))

    for name, cols in SUBSETS.items():
        for bins in BIN_COUNTS:
            rules = build_rules(table, cols, bins)
            cells = []
            for y in DATASETS:
                fr = feats[y]
                ranks = np.column_stack([
                    to_rank(fr[c].to_numpy(float), table[c].to_numpy(float))
                    for c in cols])
                res = simulate(candles[y]["close"].to_numpy(float),
                               encode(ranks, bins), rules)
                if res["trades"] == 0:
                    cells.append(f"{y} {0:7d} {'-':>6s} {'-':>8s} {'-':>8s} {'-':>9s}")
                else:
                    cells.append(f"{y} {res['trades']:7d} {res['win_rate_pct']:6.2f} "
                                 f"{res['avg_gross_pct']:8.4f} {res['avg_net_pct']:8.4f} "
                                 f"{res['total_pct']:9.1f}")
            print(f"{name:13s} {bins:4d} {len(rules):6d} "
                  f"{rules['conviction'].mean():5.2f} | " + " | ".join(cells))

    print()
    print("=" * 100)
    print("Compare the gross column against the 0.110% fee. Direction accuracy shows")
    print("up as gross; everything else is execution.")


if __name__ == "__main__":
    main()
