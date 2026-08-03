"""Turn the PureLogic trade table into a trading rule, then run it on
2025-2026 BTCUSDT data and report the profit.

The uploaded file (Sheet16_PureLogic_49783.txt) is 49,783 trades that
were profitable after fees, each with a 106-column snapshot of the market
at entry. This script does exactly what was asked: read that as the
trading logic, apply it to the 2025-2026 data, and measure the result.

HOW THE TABLE IS TURNED INTO A RULE

Each row says "with the market looking like THIS, direction D paid".
So the rule is a lookup: describe the current bar with the same 106
features, find the rows that look like it, and trade the direction they
used. Features are compared by PERCENTILE RANK within their own dataset
rather than by raw value, because the file's raw scale does not match
BTCUSDT 1-minute candles (its per-bar return standard deviation is
0.3555% against BTC 1m's 0.0648%, a factor of 5.5 — see the report at
the bottom). Ranking makes the rule transferable across instruments:
"RSI in its top decile" means the same thing everywhere, while
"atr14_pct > 0.39" is unreachable on a market that never gets there.

WHAT IS MEASURED

The direction comes from the table. The profit does NOT — it comes from
what BTCUSDT actually did over the holding period, charged the file's own
0.11% round-trip fee. Reusing the file's own net_ret_pct would just
re-report its contents; the point is to find out what the rule earns on
price data.

Run:  python3 purelogic_backtest.py
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "bybit_bot")
from purelogic import features as F  # noqa: E402

TABLE = ("/root/.claude/uploads/2499e73f-5145-5c6f-b255-816732633901/"
         "8adb2686-Sheet16_PureLogic_49783.txt")
DATASETS = {"2025": "bybit_bot/data/BTCUSDT_2025.csv",
            "2026": "bybit_bot/data/BTCUSDT_2026.csv"}

FEE_PCT = 0.11 / 100.0   # the file's own assumption, used verbatim

META = ["direction", "net_ret_pct", "duration_min",
        "gross_ret_pct", "fee_pct", "funding_cost_pct"]

# Feature subsets, smallest first. A rule keyed on more features is more
# specific: it describes the past situation more exactly, and therefore
# matches the future less often. Sweeping this is the point.
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


def load_candles(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def to_rank(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Percentile rank of each value within `reference`, in [0, 1]."""
    ref = np.sort(reference[np.isfinite(reference)])
    if ref.size == 0:
        return np.full(values.shape, np.nan)
    idx = np.searchsorted(ref, values, side="left")
    out = idx / max(ref.size, 1)
    return np.where(np.isfinite(values), out, np.nan)


def encode(ranks: np.ndarray, bins: int) -> np.ndarray:
    """Pack per-feature percentile ranks into one integer cell id."""
    codes = np.zeros(len(ranks), dtype=np.int64)
    ok = np.ones(len(ranks), dtype=bool)
    for j in range(ranks.shape[1]):
        col = ranks[:, j]
        ok &= np.isfinite(col)
        b = np.clip((np.nan_to_num(col) * bins).astype(np.int64), 0, bins - 1)
        codes = codes * bins + b
    return np.where(ok, codes, -1)


def build_rules(table: pd.DataFrame, cols: list[str], bins: int) -> pd.DataFrame:
    """One rule per distinct market cell, with how the table voted."""
    ranks = np.column_stack([
        to_rank(table[c].to_numpy(float), table[c].to_numpy(float)) for c in cols
    ])
    cell = encode(ranks, bins)
    t = pd.DataFrame({
        "cell": cell,
        "is_long": (table["direction"] == "LONG").to_numpy(),
        "net": table["net_ret_pct"].to_numpy(float),
        "dur": table["duration_min"].to_numpy(float),
    })
    t = t[t["cell"] >= 0]
    g = t.groupby("cell").agg(
        n=("is_long", "size"),
        long_share=("is_long", "mean"),
        median_dur=("dur", "median"),
        mean_net=("net", "mean"),
    )
    g["direction"] = np.where(g["long_share"] >= 0.5, 1, -1)
    # How lopsided the vote was. 0.5 means the table said long and short
    # equally often for this exact market description.
    g["conviction"] = np.abs(g["long_share"] - 0.5) * 2.0
    return g


def simulate(candles: pd.DataFrame, cell: np.ndarray, rules: pd.DataFrame,
             min_conviction: float) -> dict:
    """Trade the rule on real candles: enter at close, exit `median_dur`
    bars later, charge the file's fee. Non-overlapping positions.
    """
    close = candles["close"].to_numpy(float)
    n = len(close)

    usable = rules[rules["conviction"] >= min_conviction]
    if usable.empty:
        return {"trades": 0}
    direction = dict(zip(usable.index, usable["direction"]))
    duration = dict(zip(usable.index, usable["median_dur"]))

    pnl, busy_until, matched = [], -1, 0
    for i in range(n - 1):
        c = cell[i]
        if c < 0 or c not in direction:
            continue
        matched += 1
        if i <= busy_until:
            continue
        hold = int(max(1, min(duration[c], 500)))
        j = min(i + hold, n - 1)
        gross = (close[j] - close[i]) / close[i] * direction[c]
        pnl.append(gross - FEE_PCT)
        busy_until = j

    if not pnl:
        return {"trades": 0, "matched_bars": matched}
    a = np.array(pnl)
    wins, losses = a[a > 0], a[a <= 0]
    return {
        "trades": len(a),
        "matched_bars": matched,
        "win_rate_pct": round(100 * float((a > 0).mean()), 2),
        "avg_net_pct": round(100 * float(a.mean()), 4),
        "total_pct": round(100 * float(a.sum()), 1),
        "compounded_pct": round(100 * float(np.expm1(np.log1p(a).sum())), 1),
        "profit_factor": (round(float(wins.sum() / -losses.sum()), 3)
                          if len(losses) and losses.sum() < 0 else float("inf")),
    }


def contradiction_report(table: pd.DataFrame) -> None:
    print("=" * 78)
    print("WHAT THE TABLE CONTAINS")
    print("=" * 78)
    n = len(table)
    print(f"  trades                     : {n:,}")
    print(f"  LONG / SHORT               : {(table['direction']=='LONG').sum():,}"
          f" / {(table['direction']=='SHORT').sum():,}")
    print(f"  losing trades              : {(table['net_ret_pct'] < 0).sum():,}")
    print(f"  net return  min / median   : {table['net_ret_pct'].min():.3f}%"
          f" / {table['net_ret_pct'].median():.3f}%")
    held = table["duration_min"].sum()
    available = 525_600 + 307_379
    print(f"  total minutes held         : {held:,.0f}")
    print(f"  minutes in 2025+2026       : {available:,}")
    print(f"  coverage                   : {held/available:.2f}x  "
          f"(the trades tile the whole period end to end)")

    feat_cols = [c for c in table.columns if c not in META]
    M = table[feat_cols].to_numpy(float)
    M = np.where(np.isnan(M), -9.99e9, np.round(M, 6))
    keys = [m.tobytes() for m in M]
    tmp = pd.DataFrame({"k": keys, "d": table["direction"].to_numpy()})
    g = tmp.groupby("k")["d"].nunique()
    both = int((g > 1).sum())
    rows_both = int(tmp["k"].isin(g[g > 1].index).sum())
    print()
    print(f"  distinct market snapshots  : {g.size:,}")
    print(f"  snapshots labelled BOTH    : {both:,} ({100*both/g.size:.1f}%)")
    print(f"  trades on those snapshots  : {rows_both:,} ({100*rows_both/n:.1f}%)")
    print()
    print("  Those rows are the same 106 numbers appearing as a winning LONG and")
    print("  a winning SHORT. For 62% of the table, the features do not decide")
    print("  the direction — the table answers both ways for the same input.")


def scale_report(table: pd.DataFrame, candles: pd.DataFrame) -> None:
    print()
    print("=" * 78)
    print("DOES THE TABLE MATCH THIS BTCUSDT DATA?")
    print("=" * 78)
    r_file = table["ret_lag1"].dropna()
    r_btc = (candles["close"].pct_change() * 100).dropna()
    print(f"  per-bar return std, table  : {r_file.std():.4f}%")
    print(f"  per-bar return std, BTC 1m : {r_btc.std():.4f}%")
    print(f"  ratio                      : {r_file.std()/r_btc.std():.2f}x")
    for tf, lab in ((5, "5m"), (15, "15m"), (60, "1h")):
        s = (candles["close"].iloc[::tf].pct_change() * 100).dropna().std()
        print(f"    vs BTC {lab:<3s}                : {r_file.std()/s:.2f}x")
    print()
    print("  The table was not computed from this BTCUSDT 1-minute data. Raw")
    print("  thresholds from it would never fire here, so the rule below matches")
    print("  on percentile rank instead, which transfers across instruments.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-conviction", type=float, default=0.0,
                    help="skip cells where the table's vote was closer than this")
    args = ap.parse_args()

    table = pd.read_csv(TABLE, sep="\t")
    candles = {y: load_candles(p) for y, p in DATASETS.items()}
    combined = pd.concat(candles.values(), ignore_index=True)

    contradiction_report(table)
    scale_report(table, combined)

    print()
    print("=" * 78)
    print("THE TABLE'S OWN NUMBERS, REPLAYED AS-IS")
    print("=" * 78)
    r = table["net_ret_pct"].to_numpy(float) / 100.0
    print(f"  sum of the 49,783 net returns : {100*r.sum():,.1f}%")
    print(f"  compounded                    : {100*np.expm1(np.log1p(r).sum()):.3e}%")
    print("  Every trade in the table is a winner, so replaying the table on the")
    print("  data it was extracted from cannot lose. That number is what the file")
    print("  already says, not a test of anything.")

    print()
    print("=" * 78)
    print("THE RULE, APPLIED TO BTCUSDT PRICE DATA")
    print("=" * 78)
    print("Direction from the table; profit from what BTC actually did, at the")
    print(f"file's own {FEE_PCT*100:.2f}% round-trip fee.\n")

    feats = {y: F.build(df) for y, df in candles.items()}

    hdr = (f"{'rule keyed on':12s} {'bins':>5s} {'cells':>7s} {'conv':>6s} | "
           + " | ".join(f"{y:>4s} {'trades':>7s} {'win%':>6s} {'net%/t':>8s} {'total%':>9s}"
                        for y in DATASETS))
    print(hdr)
    print("-" * len(hdr))

    for name, cols in SUBSETS.items():
        for bins in BIN_COUNTS:
            rules = build_rules(table, cols, bins)
            usable = rules[rules["conviction"] >= args.min_conviction]
            mean_conv = float(rules["conviction"].mean())

            cells_out = []
            for y in DATASETS:
                fr = feats[y]
                ranks = np.column_stack([
                    to_rank(fr[c].to_numpy(float), table[c].to_numpy(float))
                    for c in cols
                ])
                cell = encode(ranks, bins)
                res = simulate(candles[y], cell, rules, args.min_conviction)
                if res["trades"] == 0:
                    cells_out.append(f"{y:>4s} {'0':>7s} {'-':>6s} {'-':>8s} {'-':>9s}")
                else:
                    cells_out.append(
                        f"{y:>4s} {res['trades']:7d} {res['win_rate_pct']:6.2f} "
                        f"{res['avg_net_pct']:8.4f} {res['total_pct']:9.1f}")
            print(f"{name:12s} {bins:5d} {len(usable):7d} {mean_conv:6.2f} | "
                  + " | ".join(cells_out))

    print()
    print("=" * 78)
    print("The 'conv' column is how one-sided the table's vote was for an average")
    print("market cell: 1.00 would mean it always said the same direction for the")
    print("same conditions, 0.00 that it said long and short equally often.")


if __name__ == "__main__":
    main()
