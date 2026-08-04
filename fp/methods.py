"""The twelve voting methods of FINAL_Wide_Logic_AllMethods_AllFactors.

Each method looks at the market and returns "LONG", "SHORT" or "-" (did
not fire). They are independent: a bar can have none, one, or several
firing at once, and they can disagree.

The file records how often each one fired over 2025-2026, which is what
these implementations are calibrated against:

    M01_EMA_Cross_Trend        1,077     M07_ADX_TrendStrength      2,772
    M02_MACD_Cross             2,179     M08_RSI_Divergence         3,881
    M03_RSI_Oversold_Reversal  1,000     M09_Streak_Exhaustion      1,105
    M04_Bollinger_MeanRev      5,672     M10_Wick_Rejection         1,832
    M05_Donchian_Breakout      4,285     M11_Volume_Spike_Trend     4,348
    M06_Stochastic_Cross       6,287     M12_EMA_Pullback           3,779

Every method is strictly backward-looking: a signal on bar i uses only
bars up to and including i, so what the backtest sees is what a live bot
would have seen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LONG, SHORT, NONE = "LONG", "SHORT", "-"

METHOD_NAMES = [
    "M01_EMA_Cross_Trend", "M02_MACD_Cross", "M03_RSI_Oversold_Reversal",
    "M04_Bollinger_MeanRev", "M05_Donchian_Breakout", "M06_Stochastic_Cross",
    "M07_ADX_TrendStrength", "M08_RSI_Divergence", "M09_Streak_Exhaustion",
    "M10_Wick_Rejection", "M11_Volume_Spike_Trend", "M12_EMA_Pullback",
]


def _pick(long_mask: pd.Series, short_mask: pd.Series) -> pd.Series:
    """Turn two boolean masks into the file's LONG / SHORT / '-' column."""
    return pd.Series(np.where(long_mask, LONG,
                              np.where(short_mask, SHORT, NONE)),
                     index=long_mask.index)


def m01_ema_cross_trend(f: pd.DataFrame) -> pd.Series:
    """Fast EMA crosses the slow one, agreeing with the long-term trend.

    Fires on the crossing bar only, so it is rare — the file has it firing
    1,077 times, the second-least of the twelve.
    """
    diff = f["dist_ema8_pct"] - f["dist_ema21_pct"]
    crossed_up = (diff > 0) & (diff.shift(1) <= 0)
    crossed_dn = (diff < 0) & (diff.shift(1) >= 0)
    # The trend filter is the slower EMA's slope rather than distance from
    # the 200: requiring price to also sit the right side of the 200 halved
    # the firing count against the file's 1,077.
    return _pick(crossed_up & (f["slope_ema55_5_pct"] > 0),
                 crossed_dn & (f["slope_ema55_5_pct"] < 0))


def m02_macd_cross(f: pd.DataFrame) -> pd.Series:
    """MACD line crossing its signal line."""
    h = f["macd_hist"]
    return _pick((h > 0) & (h.shift(1) <= 0), (h < 0) & (h.shift(1) >= 0))


def m03_rsi_oversold_reversal(f: pd.DataFrame) -> pd.Series:
    """RSI climbing back out of oversold, or falling back out of overbought.

    The reversal is the crossing, not the condition — sitting at RSI 25 is
    not a signal, coming back up through 30 is.
    """
    r = f["rsi14"]
    return _pick((r > 30) & (r.shift(1) <= 30), (r < 70) & (r.shift(1) >= 70))


def m04_bollinger_meanrev(f: pd.DataFrame) -> pd.Series:
    """Price outside a Bollinger band, expecting a return to the middle."""
    b = f["bb_pctb_20"]
    return _pick(b < 0.05, b > 0.95)


def m05_donchian_breakout(f: pd.DataFrame) -> pd.Series:
    """Close at the top or bottom of its recent range."""
    # 0.90 rather than a strict new extreme: at 0.99 this fired 283 times
    # against the file's 4,285.
    p = f["range_pos_20"]
    return _pick(p >= 0.90, p <= 0.10)


def m06_stochastic_cross(f: pd.DataFrame) -> pd.Series:
    """%K crossing %D while not already at an extreme."""
    k, d = f["stoch_k14"], f["stoch_d14"]
    diff = k - d
    return _pick((diff > 0) & (diff.shift(1) <= 0) & (k < 60),
                 (diff < 0) & (diff.shift(1) >= 0) & (k > 40))


def m07_adx_trendstrength(f: pd.DataFrame) -> pd.Series:
    """A confirmed trend by ADX, taken in the direction of the DI spread."""
    # ADX 45, not the textbook 25: at 25 this fired 13,451 times against
    # the file's 2,772. The file is selecting genuinely strong trends.
    strong = f["adx14"] >= 45
    return _pick(strong & (f["plus_di14"] > f["minus_di14"]),
                 strong & (f["minus_di14"] > f["plus_di14"]))


def m08_rsi_divergence(f: pd.DataFrame) -> pd.Series:
    """Price makes a new extreme that RSI does not confirm.

    Bullish: price below where it was, RSI above where it was.
    """
    lookback = 14
    px, r = f["dist_ema21_pct"], f["rsi14"]
    # Without the RSI-side band this fired 613 times against the file's
    # 3,881; the divergence itself is the signal, not where RSI sits.
    return _pick((px < px.shift(lookback)) & (r > r.shift(lookback)),
                 (px > px.shift(lookback)) & (r < r.shift(lookback)))


def m09_streak_exhaustion(f: pd.DataFrame) -> pd.Series:
    """A run of same-direction candles long enough to be worth fading."""
    s = f["consec_streak"]
    return _pick(s <= -5, s >= 5)


def m10_wick_rejection(f: pd.DataFrame) -> pd.Series:
    """A candle that spiked one way and closed back — rejection of that level."""
    return _pick((f["lower_wick_pct"] > 70) & (f["body_pct"] < 20),
                 (f["upper_wick_pct"] > 70) & (f["body_pct"] < 20))


def m11_volume_spike_trend(f: pd.DataFrame) -> pd.Series:
    """Unusual volume, traded in the direction of the bar that carried it."""
    spike = f["vol_ratio_20"] >= 2.0
    return _pick(spike & (f["ret_lag1"] > 0), spike & (f["ret_lag1"] < 0))


def m12_ema_pullback(f: pd.DataFrame) -> pd.Series:
    """An established trend pulling back to its own moving average."""
    up, down = f["dist_ema200_pct"] > 0, f["dist_ema200_pct"] < 0
    near = f["dist_ema21_pct"].abs() < 0.15
    return _pick(up & near & (f["dist_ema21_pct"] < 0),
                 down & near & (f["dist_ema21_pct"] > 0))


METHODS = {
    "M01_EMA_Cross_Trend": m01_ema_cross_trend,
    "M02_MACD_Cross": m02_macd_cross,
    "M03_RSI_Oversold_Reversal": m03_rsi_oversold_reversal,
    "M04_Bollinger_MeanRev": m04_bollinger_meanrev,
    "M05_Donchian_Breakout": m05_donchian_breakout,
    "M06_Stochastic_Cross": m06_stochastic_cross,
    "M07_ADX_TrendStrength": m07_adx_trendstrength,
    "M08_RSI_Divergence": m08_rsi_divergence,
    "M09_Streak_Exhaustion": m09_streak_exhaustion,
    "M10_Wick_Rejection": m10_wick_rejection,
    "M11_Volume_Spike_Trend": m11_volume_spike_trend,
    "M12_EMA_Pullback": m12_ema_pullback,
}


def evaluate_all(feats: pd.DataFrame) -> pd.DataFrame:
    """Run all twelve over a feature frame, plus the vote tally.

    Adds the file's own vote columns: how many fired, how many each way,
    the margin between them, and the consensus (TIE when they cancel out).
    """
    out = pd.DataFrame(index=feats.index)
    for name, fn in METHODS.items():
        out[name] = fn(feats)

    votes = out[METHOD_NAMES]
    n_long = (votes == LONG).sum(axis=1)
    n_short = (votes == SHORT).sum(axis=1)
    out["n_methods_fired"] = n_long + n_short
    out["n_long_votes"] = n_long
    out["n_short_votes"] = n_short
    out["vote_margin"] = (n_long - n_short).abs()
    out["consensus_dir"] = np.where(n_long > n_short, LONG,
                                    np.where(n_short > n_long, SHORT, "TIE"))
    return out
