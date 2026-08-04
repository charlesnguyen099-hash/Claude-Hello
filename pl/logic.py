"""The trading logic of Sheet16_Logic_Final_23214, and nothing else.

WHAT THE FILE IS

23,213 trades over BTCUSDT 2025-2026. Read from the file itself:

  direction        SHORT 11,607 / LONG 11,606   -> strictly alternating
  duration_min     sums to 832,890 against 832,979 minutes available
                   -> the trades tile the whole period end to end, no overlap
  net_ret_pct      minimum +0.010, no negative rows -> every trade is a winner
  leverage_used    100x on 23,143 of 23,213 rows
  fee_pct_leveraged  median 25.000  -> 0.250% round trip at 100x
  mae_pct          median 0.030    -> the position almost never moves against
  would_liquidate_at_100x   37 of 23,213

Alternating direction plus a 1.00x tiling is a zigzag: buy every low, sell
every high, one continuous path. MAE of 0.030% is the signature of entering
at the exact pivot — that is why 100x survives here.

WHAT THIS MODULE IMPLEMENTS

Each row also carries 106 indicator columns describing the market at entry.
Those are the only part a bot can act on, so the logic is:

  1. describe the current bar with the same 106 features
  2. find the rows whose market looked like it
  3. trade the direction they traded, for the duration they held,
     at the leverage they used, paying the fee they paid

Comparison is by percentile rank rather than raw value, so the rule carries
to any symbol instead of only the one it was fitted on.

Nothing is added beyond that. No stop, no trailing exit, no extra filter,
no re-sizing — the file specifies direction, duration, leverage and fee, and
those four are what gets executed.

The features were computed on 30-MINUTE bars: fitting the file's own scale
against resampled 2025-2026 puts the log-RMSE at 0.065 for 30m against
1.843 for 1m and 0.358 for 60m. Trade timing stays at 1 minute, which is
what the duration column requires.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BAR_MINUTES = 30

# The file's own cost and sizing, used verbatim.
LEVERAGE = 100
FEE_ROUND_TRIP_PCT = 0.0025      # fee_pct_leveraged 25% at 100x
LIQUIDATION_MOVE = 0.9 / LEVERAGE

# The lookup key. Eight of the 106 columns, chosen to span what the table
# varies over -- momentum, position in range, trend strength, volume --
# without slicing 23,213 rows so finely that each cell holds nothing.
FEATURE_KEYS = ["rsi14", "stoch_k14", "range_pos_20", "consec_streak",
                "vol_ratio_20", "adx14", "bb_pctb_20_2", "willr14"]
N_BINS = 4
MIN_CELL_TRADES = 20


def to_rank(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Percentile position of each value within `reference`, in [0, 1]."""
    ref = np.sort(reference[np.isfinite(reference)])
    if ref.size == 0:
        return np.full(np.shape(values), np.nan)
    out = np.searchsorted(ref, values, side="left") / ref.size
    return np.where(np.isfinite(values), out, np.nan)


def encode(ranks: np.ndarray, bins: int = N_BINS) -> np.ndarray:
    """Pack per-feature ranks into one integer cell id (-1 = unusable)."""
    codes = np.zeros(len(ranks), dtype=np.int64)
    ok = np.ones(len(ranks), dtype=bool)
    for j in range(ranks.shape[1]):
        col = ranks[:, j]
        ok &= np.isfinite(col)
        codes = codes * bins + np.clip(
            (np.nan_to_num(col) * bins).astype(np.int64), 0, bins - 1)
    return np.where(ok, codes, -1)


class Logic:
    """The table compiled into a lookup the bot queries bar by bar."""

    def __init__(self, table_path: str, keys: list[str] | None = None,
                 bins: int = N_BINS, min_trades: int = MIN_CELL_TRADES):
        self.keys = keys or FEATURE_KEYS
        self.bins = bins
        self.table = pd.read_csv(table_path, sep="\t")
        self.reference = {k: self.table[k].to_numpy(float) for k in self.keys}

        ranks = np.column_stack([to_rank(self.reference[k], self.reference[k])
                                 for k in self.keys])
        frame = pd.DataFrame({
            "cell": encode(ranks, bins),
            "is_long": (self.table["direction"] == "LONG").to_numpy(),
            "dur": pd.to_numeric(self.table["duration_min"], errors="coerce"),
            "lev": pd.to_numeric(self.table["leverage_used"], errors="coerce"),
            "mae": pd.to_numeric(self.table["mae_pct"], errors="coerce"),
            "net": pd.to_numeric(self.table["net_ret_pct"], errors="coerce"),
        })
        frame = frame[frame["cell"] >= 0]

        g = frame.groupby("cell").agg(
            n=("is_long", "size"),
            long_share=("is_long", "mean"),
            hold=("dur", "median"),
            leverage=("lev", "median"),
            mae=("mae", "median"),
            edge=("net", "median"),
        )
        g["direction"] = np.where(g["long_share"] >= 0.5, 1, -1)
        g["conviction"] = (g["long_share"] - 0.5).abs() * 2.0
        self.all_cells = g
        self.rules = g[g["n"] >= min_trades]

    def describe(self) -> str:
        conv = self.all_cells["conviction"]
        return (f"{len(self.table):,} trades -> {len(self.all_cells):,} cells, "
                f"{len(self.rules):,} with >={MIN_CELL_TRADES} trades; "
                f"mean conviction {conv.mean():.3f}; "
                f"median hold {self.rules['hold'].median():.0f} min; "
                f"leverage {LEVERAGE}x; fee {FEE_ROUND_TRIP_PCT*100:.3f}% round trip")

    def cells_for(self, feats: pd.DataFrame) -> np.ndarray:
        ranks = np.column_stack([
            to_rank(feats[k].to_numpy(float), self.reference[k]) for k in self.keys
        ])
        return encode(ranks, self.bins)

    def signal(self, cell: int) -> dict | None:
        """What the table did in this market, or None if it never saw it."""
        if cell < 0 or cell not in self.rules.index:
            return None
        row = self.rules.loc[cell]
        return {
            "direction": int(row["direction"]),
            "hold_minutes": int(max(1, min(row["hold"], 1002))),
            "leverage": int(round(row["leverage"])),
            "conviction": float(row["conviction"]),
            "table_mae_pct": float(row["mae"]),
            "expected_move_pct": float(row["edge"]),
            "support": int(row["n"]),
        }


def features_30m_on_1m(df_1m: pd.DataFrame, build_fn) -> pd.DataFrame:
    """30-minute features aligned onto 1-minute bars, without lookahead.

    A 30m bar stamped 10:00 has not closed until 10:30, so its reading is
    stamped to 10:30 and merged backward from there. A 1m bar at 10:05 sees
    the 09:30 bar, which is what a live bot would have had.
    """
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    left = df_1m[["datetime"]].copy()
    left["datetime"] = pd.to_datetime(left["datetime"]).astype("datetime64[ns]")

    bars = (df_1m.set_index("datetime").resample(f"{BAR_MINUTES}min")
            .agg(agg).dropna().reset_index())
    feats = build_fn(bars)
    feats["datetime"] = (pd.to_datetime(feats["datetime"]).astype("datetime64[ns]")
                         + pd.Timedelta(minutes=BAR_MINUTES))
    return pd.merge_asof(left.sort_values("datetime"),
                         feats.sort_values("datetime"),
                         on="datetime", direction="backward")
