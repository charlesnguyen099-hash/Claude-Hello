"""Backtest the twelve-method voting logic over 2025-2026.

Runs the whole pipeline: 30-minute bars, 106 features, twelve methods
voting, consensus direction, all five exits scored, flexible leverage.

Reports each fixed exit separately and the pick-the-best-one column
separately, because the file's own numbers show the gap between them is
the entire result: every individual strategy averages between -0.2191%
and -0.2563%, while best_exit_net averages +0.1343%. The difference is
knowing which exit will win before the trade, which a live bot does not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wl import exits as X
from wl import features as F
from wl import methods as M

BAR_MINUTES = 30


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def to_bars(df_1m: pd.DataFrame, minutes: int = BAR_MINUTES) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    return (df_1m.set_index("datetime").resample(f"{minutes}min")
            .agg(agg).dropna().reset_index())


def run(df_1m: pd.DataFrame, min_votes: int = 1,
        require_consensus: bool = True) -> pd.DataFrame:
    """Every bar where the vote clears `min_votes`, scored under all exits."""
    bars = to_bars(df_1m)
    feats = F.build(bars)
    votes = M.evaluate_all(feats)

    high = bars["high"].to_numpy(float)
    low = bars["low"].to_numpy(float)
    close = bars["close"].to_numpy(float)
    atr_pct = feats["atr14_pct"].to_numpy(float)

    rows = []
    for i in range(len(bars) - 2):
        v = votes.iloc[i]
        if v["n_methods_fired"] < min_votes:
            continue
        d = v["consensus_dir"]
        if d == "TIE":
            if require_consensus:
                continue
            d = M.LONG
        direction = 1 if d == M.LONG else -1

        atr = atr_pct[i] / 100.0 * close[i]
        if not np.isfinite(atr) or atr <= 0:
            continue

        res = X.simulate_all(high, low, close, i, direction, atr)
        res.update({
            "i": i, "direction": direction,
            "n_methods_fired": int(v["n_methods_fired"]),
            "vote_margin": int(v["vote_margin"]),
            "leverage_flexible": X.leverage_flexible(atr_pct[i] * X.SL_MULTIPLE),
        })
        for m in M.METHOD_NAMES:
            res[m] = v[m]
        rows.append(res)
    return pd.DataFrame(rows)


def summarise(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"trades": 0}
    out = {"trades": len(t)}
    for s in X.STRATEGIES:
        v = t[s]
        out[s] = {
            "mean_pct": round(100 * float(v.mean()), 4),
            "win_rate_pct": round(100 * float((v > 0).mean()), 2),
            "total_pct": round(100 * float(v.sum()), 1),
        }
    b = t["best_exit_net"]
    out["best_exit_net"] = {
        "mean_pct": round(100 * float(b.mean()), 4),
        "win_rate_pct": round(100 * float((b > 0).mean()), 2),
    }
    out["any_strategy_profitable_pct"] = round(
        100 * float(t["any_strategy_profitable"].mean()), 2)
    return out
