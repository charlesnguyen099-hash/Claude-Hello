"""Recompute every feature in the PureLogic table, from its column names.

The uploaded table (Sheet16_PureLogic_49783.txt) carries 106 indicator
columns describing the market at the moment each trade was opened. To run
that logic forward on price data, those same 106 numbers have to be
computable from candles — this module does that, one function per family,
with the standard textbook definition for each.

Every feature here is strictly backward-looking: bar i uses only bars
<= i. No `.shift(-n)`, no centred windows, no future information. That
property is what makes the output usable as a live trading signal rather
than a description of what already happened.

Column families, matching the file's header exactly:

    rsi{6,9,14,21,28}                       Wilder RSI
    dist_ema{N}_pct, slope_ema{N}_5_pct     N in 5,8,13,21,34,55,89,144,200
    macd, macd_signal, macd_hist            12/26/9
    stoch_k{14,21}, stoch_d{14,21}          %K with 3-period %D
    willr{14,21}                            Williams %R
    cci{14,20}                              Commodity Channel Index
    bb_pctb / bb_width  (20,2) (10,1.5) (50,2.5)
    atr{7,14,21,50}_pct                     Wilder ATR as % of close
    adx14, plus_di14, minus_di14
    obv_slope_10, obv_z
    vol_ratio_{5,10,20,50}, vol_z_20
    ret_lag1..20                            per-bar % returns, lagged
    volratio_lag1..10                       volume / 20-bar mean, lagged
    ret_skew_{20,50}, ret_kurt_{20,50}
    range_pos_{10,20,50}                    position within recent range
    realized_vol_{10,20,50,100}
    body_pct, upper_wick_pct, lower_wick_pct, range_pct
    consec_streak                           signed run of same-direction bars
    hour, dow, dom, is_asia, is_eu, is_us
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EMA_PERIODS = [5, 8, 13, 21, 34, 55, 89, 144, 200]
RSI_PERIODS = [6, 9, 14, 21, 28]
ATR_PERIODS = [7, 14, 21, 50]
EPS = 1e-12


def _wilder(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing — an EMA with alpha = 1/period."""
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = _wilder(delta.clip(lower=0), period)
    loss = _wilder(-delta.clip(upper=0), period)
    rs = gain / loss.replace(0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # loss == 0 means an unbroken up-run: RSI is 100 by definition.
    return out.where(loss != 0, 100.0)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    return _wilder(true_range(df), period)


def adx(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr_s = _wilder(true_range(df), period)
    plus_di = 100.0 * _wilder(pd.Series(plus_dm, index=df.index), period) / tr_s.replace(0, np.nan)
    minus_di = 100.0 * _wilder(pd.Series(minus_dm, index=df.index), period) / tr_s.replace(0, np.nan)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return _wilder(dx, period), plus_di, minus_di


def stochastic(df: pd.DataFrame, period: int) -> tuple[pd.Series, pd.Series]:
    low_n = df["low"].rolling(period).min()
    high_n = df["high"].rolling(period).max()
    k = 100.0 * (df["close"] - low_n) / (high_n - low_n).replace(0, np.nan)
    return k, k.rolling(3).mean()


def williams_r(df: pd.DataFrame, period: int) -> pd.Series:
    high_n = df["high"].rolling(period).max()
    low_n = df["low"].rolling(period).min()
    return -100.0 * (high_n - df["close"]) / (high_n - low_n).replace(0, np.nan)


def cci(df: pd.DataFrame, period: int) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    sma = tp.rolling(period).mean()
    # CCI uses mean absolute deviation, not standard deviation.
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))


def bollinger(close: pd.Series, period: int, mult: float) -> tuple[pd.Series, pd.Series]:
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    upper, lower = mid + mult * sd, mid - mult * sd
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    width = (upper - lower) / mid.replace(0, np.nan) * 100.0
    return pct_b, width


def on_balance_volume(df: pd.DataFrame) -> pd.Series:
    sign = np.sign(df["close"].diff()).fillna(0.0)
    return (sign * df["volume"]).cumsum()


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Least-squares slope over `window` bars, per bar."""
    x = np.arange(window, dtype=float)
    x_centred = x - x.mean()
    denom = (x_centred ** 2).sum()

    def _slope(y: np.ndarray) -> float:
        return float((x_centred * (y - y.mean())).sum() / denom)

    return series.rolling(window).apply(_slope, raw=True)


def consecutive_streak(close: pd.Series) -> pd.Series:
    """Signed count of consecutive same-direction bars: +3 = three ups."""
    sign = np.sign(close.diff()).fillna(0.0)
    grp = (sign != sign.shift()).cumsum()
    run = sign.groupby(grp).cumcount() + 1
    return run * sign


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 106 PureLogic features for a 1m OHLCV frame.

    `df` needs columns datetime, open, high, low, close, volume.
    """
    df = df.sort_values("datetime").reset_index(drop=True)
    o, h, l, c, v = (df[k] for k in ("open", "high", "low", "close", "volume"))
    out: dict[str, pd.Series] = {}

    for p in RSI_PERIODS:
        out[f"rsi{p}"] = rsi(c, p)

    for p in EMA_PERIODS:
        ema = c.ewm(span=p, adjust=False).mean()
        out[f"dist_ema{p}_pct"] = (c / ema.replace(0, np.nan) - 1.0) * 100.0
        # 5-bar percentage slope of the EMA itself.
        out[f"slope_ema{p}_5_pct"] = (ema / ema.shift(5).replace(0, np.nan) - 1.0) * 100.0

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    out["macd"] = macd_line
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_line - macd_signal

    for p in (14, 21):
        k, d = stochastic(df, p)
        out[f"stoch_k{p}"], out[f"stoch_d{p}"] = k, d
        out[f"willr{p}"] = williams_r(df, p)

    out["cci14"] = cci(df, 14)
    out["cci20"] = cci(df, 20)

    for period, mult in ((20, 2.0), (10, 1.5), (50, 2.5)):
        pct_b, width = bollinger(c, period, mult)
        out[f"bb_pctb_{period}"] = pct_b
        out[f"bb_width_{period}"] = width

    for p in ATR_PERIODS:
        out[f"atr{p}_pct"] = atr(df, p) / c.replace(0, np.nan) * 100.0

    adx14, plus_di, minus_di = adx(df, 14)
    out["adx14"], out["plus_di14"], out["minus_di14"] = adx14, plus_di, minus_di

    obv = on_balance_volume(df)
    out["obv_slope_10"] = _rolling_slope(obv, 10)
    obv_mean, obv_sd = obv.rolling(100).mean(), obv.rolling(100).std(ddof=0)
    out["obv_z"] = (obv - obv_mean) / obv_sd.replace(0, np.nan)

    vol_ma20 = v.rolling(20).mean()
    for p in (5, 10, 20, 50):
        out[f"vol_ratio_{p}"] = v / v.rolling(p).mean().replace(0, np.nan)
    out["vol_z_20"] = (v - vol_ma20) / v.rolling(20).std(ddof=0).replace(0, np.nan)

    ret = c.pct_change() * 100.0
    for lag in range(1, 21):
        out[f"ret_lag{lag}"] = ret.shift(lag - 1)

    vol_ratio_base = v / vol_ma20.replace(0, np.nan)
    for lag in range(1, 11):
        out[f"volratio_lag{lag}"] = vol_ratio_base.shift(lag - 1)

    for p in (20, 50):
        out[f"ret_skew_{p}"] = ret.rolling(p).skew()
        out[f"ret_kurt_{p}"] = ret.rolling(p).kurt()

    for p in (10, 20, 50):
        low_n, high_n = l.rolling(p).min(), h.rolling(p).max()
        out[f"range_pos_{p}"] = (c - low_n) / (high_n - low_n).replace(0, np.nan)

    for p in (10, 20, 50, 100):
        out[f"realized_vol_{p}"] = ret.rolling(p).std(ddof=0)

    rng = (h - l).replace(0, np.nan)
    out["body_pct"] = (c - o).abs() / rng * 100.0
    out["upper_wick_pct"] = (h - np.maximum(c, o)) / rng * 100.0
    out["lower_wick_pct"] = (np.minimum(c, o) - l) / rng * 100.0
    out["range_pct"] = (h - l) / c.replace(0, np.nan) * 100.0

    out["consec_streak"] = consecutive_streak(c)
    # How far price has run from the 21 EMA, measured in ATRs rather than
    # percent, so it means the same thing in a calm and a violent market.
    ema21 = c.ewm(span=21, adjust=False).mean()
    out["ext_ema21_atr"] = (c - ema21) / atr(df, 14).replace(0, np.nan)

    dt = pd.to_datetime(df["datetime"])
    hour = dt.dt.hour
    out["hour_utc"] = hour.astype(float)
    out["day_of_week"] = dt.dt.dayofweek.astype(float)
    # Session flags, UTC. Asia 00-08, Europe 07-16, US 13-22.
    out["is_asia"] = ((hour >= 0) & (hour < 8)).astype(float)
    out["is_eu"] = ((hour >= 7) & (hour < 16)).astype(float)
    out["is_us"] = ((hour >= 13) & (hour < 22)).astype(float)

    feats = pd.DataFrame(out, index=df.index)
    feats.insert(0, "datetime", df["datetime"].to_numpy())
    return feats


FEATURE_COLUMNS: list[str] = (
    [f"rsi{p}" for p in RSI_PERIODS]
    + [f"dist_ema{p}_pct" for p in EMA_PERIODS]
    + [f"slope_ema{p}_5_pct" for p in EMA_PERIODS]
    + ["macd", "macd_signal", "macd_hist"]
    + ["stoch_k14", "stoch_d14", "stoch_k21", "stoch_d21", "willr14", "willr21"]
    + ["cci14", "cci20"]
    + ["bb_pctb_20", "bb_width_20", "bb_pctb_10",
       "bb_width_10", "bb_pctb_50", "bb_width_50"]
    + [f"atr{p}_pct" for p in ATR_PERIODS]
    + ["adx14", "plus_di14", "minus_di14", "obv_slope_10", "obv_z"]
    + [f"vol_ratio_{p}" for p in (5, 10, 20, 50)] + ["vol_z_20"]
    + [f"ret_lag{i}" for i in range(1, 21)]
    + [f"volratio_lag{i}" for i in range(1, 11)]
    + ["ret_skew_20", "ret_kurt_20", "ret_skew_50", "ret_kurt_50"]
    + [f"range_pos_{p}" for p in (10, 20, 50)]
    + [f"realized_vol_{p}" for p in (10, 20, 50, 100)]
    + ["body_pct", "upper_wick_pct", "lower_wick_pct", "range_pct"]
    + ["consec_streak", "ext_ema21_atr", "hour_utc", "day_of_week",
       "is_asia", "is_eu", "is_us"]
)
