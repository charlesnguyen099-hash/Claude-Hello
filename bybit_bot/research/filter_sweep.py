"""How many trades are the entry filters throwing away, and is throwing
them away actually helping?

Request behind this: "make sure it trades every potential trade, from
the smallest profit to the largest -- don't miss any." The current
settings fire only 14 times across all of 2026, because several gates
compound:

  MIN_TP1_TO_COST_RATIO = 8  -> TP1 must be >= 8x the 0.21% round trip,
                                i.e. >= 1.68%. Every smaller-profit
                                setup is discarded outright.
  ADX_MIN = 25               -> only "textbook trending" 1h regimes.
  TREND_MIN_BARS = 3         -> regime must have held 3 hours.
  volume > vol_ma300         -> above-average volume required.
  RSI 30/70 band             -> no buying overbought / selling oversold.

This sweeps them on BOTH years and reports, for each setting, how many
trades it takes and whether the extra trades are worth taking. The
question is not "can we trade more" (obviously yes, drop the filters)
but "do the trades we're currently skipping make or lose money."

The expensive indicator pass is computed once per year and reused, so
the whole grid runs in minutes rather than hours.

Run:  python3 -m research.filter_sweep
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from backtest.engine import run_backtest_prepared
from bot import indicators as ind
from bot import strategy

DATASETS = {"2026": "data/BTCUSDT_2026.csv", "2025": "data/BTCUSDT_2025.csv"}


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


class Cache:
    """One expensive indicator pass per dataset, reused for every combo."""

    def __init__(self, df_1m: pd.DataFrame):
        d = df_1m.set_index("datetime")
        agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        df_1h = d.resample("1h").agg(agg).dropna().reset_index()

        self.ltf = strategy.compute_ltf(df_1m)
        htf = df_1h.copy()
        htf["ema50"] = ind.ema(htf["close"], strategy.EMA_SLOW_HTF)
        htf["ema200"] = ind.ema(htf["close"], strategy.EMA_TREND_HTF)
        htf["adx14"] = ind.adx(htf, 14)
        self.htf_base = htf

    def prepared(self, adx_min: float, trend_min_bars: int) -> pd.DataFrame:
        htf = self.htf_base.copy()
        up = ((htf["ema50"] > htf["ema200"]) & (htf["close"] > htf["ema50"])
              & (htf["adx14"] >= adx_min))
        down = ((htf["ema50"] < htf["ema200"]) & (htf["close"] < htf["ema50"])
                & (htf["adx14"] >= adx_min))
        raw = pd.Series(np.select([up, down], ["up", "down"], default="flat"), index=htf.index)
        stable = pd.Series(True, index=htf.index)
        for k in range(1, trend_min_bars):
            stable &= raw == raw.shift(k)
        htf["trend"] = np.where(stable, raw, "flat")

        merged = strategy.merge_htf_trend(self.ltf, htf)
        return strategy.build_signal_columns(merged)


def apply_gates(df: pd.DataFrame, use_volume: bool, rsi_band: tuple[float, float]) -> pd.DataFrame:
    """Re-derive the setup columns with volume / RSI gates toggled."""
    out = df.copy()
    cross_up = (out["ema9"].shift(1) <= out["ema21"].shift(1)) & (out["ema9"] > out["ema21"])
    cross_down = (out["ema9"].shift(1) >= out["ema21"].shift(1)) & (out["ema9"] < out["ema21"])
    vol_ok = (out["volume"] > out["vol_ma20"]) if use_volume else True
    lo, hi = rsi_band
    out["long_setup"] = (out["trend"] == "up") & cross_up & vol_ok & (out["rsi14"] <= hi)
    out["short_setup"] = (out["trend"] == "down") & cross_down & vol_ok & (out["rsi14"] >= lo)
    return out


def evaluate(df_prepared: pd.DataFrame, tp1_ratio: float) -> dict:
    original = strategy.MIN_TP1_TO_COST_RATIO
    strategy.MIN_TP1_TO_COST_RATIO = tp1_ratio
    try:
        return run_backtest_prepared(df_prepared).summary()
    finally:
        strategy.MIN_TP1_TO_COST_RATIO = original


def main() -> None:
    caches = {y: Cache(load(p)) for y, p in DATASETS.items()}
    print(f"round-trip cost = {strategy._risk.ROUND_TRIP_COST_PCT*100:.3f}%  "
          f"(TP1 must clear ratio x this)\n")

    tp1_ratios = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
    adx_mins = [15.0, 20.0, 25.0]
    gate_sets = [
        ("vol+rsi", True, (30.0, 70.0)),
        ("rsi only", False, (30.0, 70.0)),
        ("none", False, (0.0, 100.0)),
    ]

    header = (f"{'gates':9s} {'adx':>4s} {'bars':>4s} {'tp1x':>5s} | "
              f"{'2026 trades':>11s} {'win%':>6s} {'PF':>6s} {'ret%':>8s} | "
              f"{'2025 trades':>11s} {'win%':>6s} {'PF':>6s} {'ret%':>8s} | both+")
    print(header)
    print("-" * len(header))

    rows = []
    for (gname, use_vol, band), adx_min, bars, tp1 in itertools.product(
            gate_sets, adx_mins, [1, 3], tp1_ratios):
        res = {}
        for year, cache in caches.items():
            prep = apply_gates(cache.prepared(adx_min, bars), use_vol, band)
            res[year] = evaluate(prep, tp1)
        a, b = res["2026"], res["2025"]
        both = a["return_pct"] > 0 and b["return_pct"] > 0
        rows.append((gname, adx_min, bars, tp1, a, b, both))
        print(f"{gname:9s} {adx_min:4.0f} {bars:4d} {tp1:5.1f} | "
              f"{a['trades']:11d} {a['win_rate_pct']:6.1f} {a['profit_factor']:6.2f} "
              f"{a['return_pct']:8.2f} | "
              f"{b['trades']:11d} {b['win_rate_pct']:6.1f} {b['profit_factor']:6.2f} "
              f"{b['return_pct']:8.2f} | {'YES' if both else ''}")

    print("\n" + "=" * 74)
    winners = [r for r in rows if r[6]]
    if not winners:
        print("No setting in this grid was profitable on BOTH years.")
    else:
        print(f"{len(winners)} setting(s) profitable on BOTH years:")
        for gname, adx_min, bars, tp1, a, b, _ in sorted(
                winners, key=lambda r: -(r[4]["return_pct"] + r[5]["return_pct"])):
            print(f"  gates={gname:9s} adx>={adx_min:.0f} bars={bars} tp1x={tp1:.1f}  "
                  f"2026 {a['trades']:>4d} trades {a['return_pct']:+7.2f}%  |  "
                  f"2025 {b['trades']:>4d} trades {b['return_pct']:+7.2f}%")


if __name__ == "__main__":
    main()
