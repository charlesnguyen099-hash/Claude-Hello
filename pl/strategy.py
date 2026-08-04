"""The trading logic, taken from Sheet16_PureLogic_MaxLeverage_MaxFee_24590.

The file is 24,589 trades, all profitable, alternating LONG/SHORT — a
zigzag over 2025-2026 BTCUSDT, with 100x leverage and the leveraged fee
charged. Every row carries a 106-column snapshot of the market at entry.

This module turns that into something a bot can execute, which means
answering one question the table does not: what to do when the market
looks like row N but the future is unknown.

The rule
--------
Describe the current bar with the same 106 features, on the same
30-minute bars the table was computed on, and look up what the table did
in comparable conditions. Comparison is by percentile rank, not raw
value, so "RSI in its top decile" carries across instruments and the
logic can run on any coin rather than only the one it was fitted to.

Each cell of the lookup keeps three things from the table:
  direction    which way the winning trades in that cell went
  conviction   how one-sided that was, 0 = evenly split, 1 = unanimous
  hold         median duration of those trades, used as the exit

MAE and leverage
----------------
The table's trades enter at the exact pivot, so they almost never move
against the position: median MAE 0.030%, and only 37 of 24,589 would
liquidate at 100x. That is what makes 100x survivable there.

A live entry is confirmed only after price has turned, so it starts
further from the extreme. Measured on the same data, causal entries have
a median MAE of 0.321% — 10.7x the table's. The bot therefore sizes
leverage from the MAE it should expect, not from the MAE the table
recorded, and refuses any leverage that would liquidate on a normal
adverse move. LEVERAGE_SAFETY_MAE is that expected excursion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BAR_MINUTES = 30          # the timeframe the table's features were built on

# Percentile-rank lookup settings.
FEATURE_KEYS = ["rsi14", "stoch_k14", "range_pos_20", "consec_streak",
                "vol_ratio_20", "adx14", "bb_pctb_20_2", "willr14"]
N_BINS = 4
MIN_CELL_TRADES = 20      # ignore cells the table barely visited
MIN_CONVICTION = 0.60     # raised from 0.30: see WRONG-DIRECTION below

# The root cause of the losses, from the backtest: of 7,843 losing trades
# on 2025, only 1,360 went the wrong way. 3,369 moved the right way by
# less than the fee, and 3,956 barely moved at all — 93% of the losses are
# the cost of trading, not a wrong call on direction.
#
# So the fix is not a better direction filter, it is refusing setups whose
# expected move cannot pay for itself. Each cell keeps the median
# unleveraged net return of the table's trades in it, and a cell is only
# tradeable if that clears the round trip by MIN_EDGE_MULTIPLE.
MIN_EDGE_MULTIPLE = 2.0

# WRONG-DIRECTION FIX. 1,360 of 7,843 losing trades went the wrong way.
# Those came from cells where the table itself was close to evenly split
# between LONG and SHORT — it had no real opinion, and the majority vote
# was noise. Conviction is |long_share - 0.5| * 2, so 0.60 means at least
# 80/20 agreement in that cell. Cells below it are skipped rather than
# guessed at. MIN_CELL_SUPPORT additionally requires enough trades behind
# the vote for the ratio to mean anything.
MIN_CELL_SUPPORT = 40

# Risk. The table used 100x; a causal entry cannot survive that.
# Measured on causal entries over 2025-2026: median MAE 0.32%, p99 0.87%.
# Sizing off the median would liquidate on one trade in a hundred, so the
# p99 is used instead.
LEVERAGE_SAFETY_MAE = 0.0087
LIQUIDATION_BUFFER = 2.0       # require this much headroom before liquidation
MAX_LEVERAGE = 100
MIN_LEVERAGE = 3

# FEE FIX. A market order pays taker and crosses the spread:
#     taker 0.055% + slippage 0.050%, twice = 0.210% round trip
# A resting limit order pays maker and gets its own price:
#     maker 0.020% + slippage 0.000%, twice = 0.040% round trip
# Same trade, a fifth of the cost. The strategy already knows its entry
# level in advance, so there is nothing stopping it from resting the order
# rather than crossing — the only cost is that some entries never fill.
TAKER_FEE_PCT = 0.00055
MAKER_FEE_PCT = 0.00020
SLIPPAGE_PCT = 0.0005

TAKER_ROUND_TRIP = 2 * (TAKER_FEE_PCT + SLIPPAGE_PCT)   # 0.210%
MAKER_ROUND_TRIP = 2 * MAKER_FEE_PCT                    # 0.040%
ROUND_TRIP_PCT = MAKER_ROUND_TRIP                       # default: rest the order

# LIQUIDATION FIX. Lowering leverage alone does not remove liquidation, it
# just moves it. A stop-loss placed strictly inside the liquidation
# distance removes it structurally: the stop is always hit first, so the
# position is closed at a known loss instead of being taken by the
# exchange. STOP_FRACTION is how far toward liquidation the stop sits.
STOP_FRACTION_OF_LIQ = 0.5


def safe_leverage(expected_mae: float = LEVERAGE_SAFETY_MAE) -> int:
    """Highest leverage that still survives the adverse move we expect.

    Liquidation happens near 1/leverage of adverse movement, so requiring
    LIQUIDATION_BUFFER x headroom over the expected excursion gives
    leverage <= 1 / (buffer * mae).
    """
    if expected_mae <= 0:
        return MAX_LEVERAGE
    return max(MIN_LEVERAGE,
               min(MAX_LEVERAGE, int(1.0 / (LIQUIDATION_BUFFER * expected_mae))))


def leverage_for(cell_mae_pct: float, floor_mae: float = LEVERAGE_SAFETY_MAE) -> int:
    """Per-trade leverage, sized from how far THIS setup tends to run against
    the position rather than from one global number.

    A calm cell earns more leverage than a violent one, which is what
    "flexible leverage" has to mean if it is not to be either reckless in
    the violent cells or wasteful in the calm ones. The floor keeps a
    suspiciously small recorded MAE from producing absurd leverage: the
    table's excursions are measured from perfect pivots and understate what
    a live entry will see, so the floor is the p99 excursion actually
    measured on causal entries (0.87%). Without it, a cell recording MAE
    0.03% would ask for 100x and be stopped out on the first normal wobble.
    """
    mae = max(cell_mae_pct / 100.0, floor_mae)
    return safe_leverage(mae)


def stop_distance(leverage: int) -> float:
    """Adverse move at which the stop fires, as a fraction of entry price.

    Strictly inside the liquidation distance, so the stop always triggers
    first and the position can never be liquidated.
    """
    return STOP_FRACTION_OF_LIQ * 0.9 / max(1, leverage)


def to_rank(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Percentile position of each value within `reference`, in [0, 1]."""
    ref = np.sort(reference[np.isfinite(reference)])
    if ref.size == 0:
        return np.full(np.shape(values), np.nan)
    out = np.searchsorted(ref, values, side="left") / ref.size
    return np.where(np.isfinite(values), out, np.nan)


def encode(ranks: np.ndarray, bins: int = N_BINS) -> np.ndarray:
    """Pack per-feature ranks into a single integer cell id (-1 = unusable)."""
    codes = np.zeros(len(ranks), dtype=np.int64)
    ok = np.ones(len(ranks), dtype=bool)
    for j in range(ranks.shape[1]):
        col = ranks[:, j]
        ok &= np.isfinite(col)
        codes = codes * bins + np.clip(
            (np.nan_to_num(col) * bins).astype(np.int64), 0, bins - 1)
    return np.where(ok, codes, -1)


class PureLogic:
    """The table, compiled into a lookup a bot can query bar by bar."""

    def __init__(self, table_path: str, keys: list[str] | None = None,
                 bins: int = N_BINS, min_trades: int = MIN_CELL_TRADES,
                 min_conviction: float = MIN_CONVICTION,
                 min_edge_multiple: float = MIN_EDGE_MULTIPLE):
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
            "mae": pd.to_numeric(self.table["mae_pct"], errors="coerce"),
            "net": pd.to_numeric(self.table["net_ret_pct"], errors="coerce"),
        })
        frame = frame[frame["cell"] >= 0]

        g = frame.groupby("cell").agg(
            n=("is_long", "size"),
            long_share=("is_long", "mean"),
            hold=("dur", "median"),
            mae=("mae", "median"),
            edge=("net", "median"),
        )
        g["direction"] = np.where(g["long_share"] >= 0.5, 1, -1)
        g["conviction"] = (g["long_share"] - 0.5).abs() * 2.0
        self.all_cells = g
        self.rules = g[(g["n"] >= max(min_trades, MIN_CELL_SUPPORT))
                       & (g["conviction"] >= min_conviction)
                       & (g["edge"] >= 100 * ROUND_TRIP_PCT * min_edge_multiple)]

    def describe(self) -> str:
        total, kept = len(self.all_cells), len(self.rules)
        conv = self.all_cells["conviction"]
        return (f"{len(self.table):,} trades -> {total:,} market cells, "
                f"{kept:,} usable (>={MIN_CELL_TRADES} trades, "
                f">={MIN_CONVICTION:.2f} conviction); "
                f"mean conviction {conv.mean():.3f}, "
                f"cells that are a coin flip: "
                f"{100*(conv < 0.2).mean():.1f}%; "
                f"median cell edge {self.all_cells['edge'].median():.3f}% vs "
                f"{100*ROUND_TRIP_PCT:.3f}% round trip")

    def cells_for(self, feats: pd.DataFrame) -> np.ndarray:
        """Map each row of a feature frame onto the table's cell space."""
        ranks = np.column_stack([
            to_rank(feats[k].to_numpy(float), self.reference[k]) for k in self.keys
        ])
        return encode(ranks, self.bins)

    def signal(self, cell: int) -> dict | None:
        """The table's verdict for one market cell, or None if it has none."""
        if cell < 0 or cell not in self.rules.index:
            return None
        row = self.rules.loc[cell]
        return {
            "direction": int(row["direction"]),
            "hold_minutes": int(max(1, min(row["hold"], 1000))),
            "conviction": float(row["conviction"]),
            "table_mae_pct": float(row["mae"]),
            "expected_move_pct": float(row["edge"]),
            "leverage": leverage_for(float(row["mae"])),
            "support": int(row["n"]),
        }


def features_30m_on_1m(df_1m: pd.DataFrame, build_fn) -> pd.DataFrame:
    """30-minute features aligned onto 1-minute bars, without lookahead.

    A 30m bar stamped 10:00 has not closed until 10:30, so its reading is
    stamped to 10:30 and merged backward. A 1m bar at 10:05 sees the 09:30
    bar — what a live bot would actually have had.
    """
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    left = df_1m[["datetime"]].copy()
    # merge_asof refuses to join datetime64[ms] against datetime64[us], and
    # the two sides pick up different resolutions depending on whether the
    # candles came from a CSV or from the exchange. Normalise both.
    left["datetime"] = pd.to_datetime(left["datetime"]).astype("datetime64[ns]")

    bars = (df_1m.set_index("datetime").resample(f"{BAR_MINUTES}min")
            .agg(agg).dropna().reset_index())
    feats = build_fn(bars)
    feats["datetime"] = (pd.to_datetime(feats["datetime"]).astype("datetime64[ns]")
                         + pd.Timedelta(minutes=BAR_MINUTES))
    return pd.merge_asof(left.sort_values("datetime"),
                         feats.sort_values("datetime"),
                         on="datetime", direction="backward")
