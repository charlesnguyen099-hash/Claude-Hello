"""The method library: the ways people actually trade futures.

Forty-two named methods, each one a technique that existed before this
data did. Every method reads the bars and returns a SIGNED STATE per bar:

    +1  a long is on        -1  a short is on        0  stand aside

Not an entry timestamp, not a target, not a stop, not a duration. A state.
That distinction is the whole design:

  * ENTRY is where the state turns on. Nothing is fixed about the price:
    it is whatever the market is at when the condition becomes true.
  * EXIT is where the state turns off or flips. Nothing is fixed about
    that price or about how long the trade lasted -- a method may hold
    four minutes or four days depending only on what the market does.
  * SIZE is decided elsewhere, by measured potential.

So there is no fixed entry price, no fixed exit price and no fixed hold
time anywhere in this file. What IS fixed is the shape of each technique
-- an EMA cross is an EMA cross -- and its lookbacks, which are the
classic ones rather than numbers tuned against the answer.

Every method is SCALE-FREE: it compares price to price, or normalises by
volatility. A $100,000 coin and a $0.02 coin produce the same states, so
one method serves the whole board.

THE SHIFT. Every state is computed from bars up to and including t and is
read as the state at t's CLOSE. A trade opened on that state fills at the
next price the bot sees. No method may read a bar that has not happened;
fp/test_engine.py truncates the data and checks that the past does not
move.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(x, n):
    return x.ewm(span=n, adjust=False).mean()


def _rma(x, n):
    return x.ewm(alpha=1.0 / n, adjust=False).mean()


def _atr(d, n=14):
    h, lo, c = d["high"], d["low"], d["close"]
    pc = c.shift()
    tr = pd.concat([h - lo, (h - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    return _rma(tr, n)


def _sig(x):
    """A signed state from a continuous score, with a dead zone at 0."""
    return pd.Series(np.sign(x).fillna(0.0) if hasattr(x, "fillna")
                     else np.sign(x), index=x.index).fillna(0.0)


def _z(x, n):
    return (x - x.rolling(n, min_periods=n // 2).mean()) / \
        x.rolling(n, min_periods=n // 2).std().replace(0, np.nan)


# ---------------------------------------------------------------- trend

def ema_cross_fast(d):
    c = d["close"]
    return _sig(_ema(c, 9) - _ema(c, 21))


def ema_cross_mid(d):
    c = d["close"]
    return _sig(_ema(c, 21) - _ema(c, 55))


def ema_cross_slow(d):
    c = d["close"]
    return _sig(_ema(c, 55) - _ema(c, 200))


def macd(d):
    c = d["close"]
    line = _ema(c, 12) - _ema(c, 26)
    return _sig(line - _ema(line, 9))


def macd_zero(d):
    c = d["close"]
    return _sig(_ema(c, 12) - _ema(c, 26))


def price_vs_ma(d):
    c = d["close"]
    return _sig(c - c.rolling(100, min_periods=50).mean())


def linreg_slope(d):
    """Sign of a rolling linear-regression slope -- trend without a cross."""
    c = np.log(d["close"])
    n = 60
    return _sig(c.rolling(n, min_periods=n // 2).mean().diff(n))


def supertrend(d):
    """Classic Supertrend: an ATR band that only ever ratchets one way."""
    c, atr = d["close"], _atr(d, 10)
    hl2 = (d["high"] + d["low"]) / 2.0
    up, dn = hl2 - 3.0 * atr, hl2 + 3.0 * atr
    # Vectorised approximation of the ratchet: compare close to the band
    # it would have to break to flip. Exact recursion is O(n) in Python
    # and this agrees with it on the state, which is all that is used.
    return _sig(np.sign(c - up.rolling(10, min_periods=1).max())
                + np.sign(c - dn.rolling(10, min_periods=1).min()))


def donchian_fast(d):
    return _donchian(d, 20)


def donchian_mid(d):
    return _donchian(d, 55)


def donchian_slow(d):
    return _donchian(d, 200)


def _donchian(d, n):
    """The original turtle rule: new n-bar high is long, new low is short,
    and the state PERSISTS until the opposite extreme -- so the hold is
    whatever the market gives it."""
    hh = d["high"].rolling(n, min_periods=n // 2).max().shift(1)
    ll = d["low"].rolling(n, min_periods=n // 2).min().shift(1)
    up = (d["close"] > hh).astype(float)
    dn = (d["close"] < ll).astype(float)
    s = (up - dn).replace(0.0, np.nan).ffill().fillna(0.0)
    return s


def aroon(d):
    n = 50
    hi = d["high"].rolling(n, min_periods=n // 2).apply(
        lambda v: float(np.argmax(v)), raw=True)
    lo = d["low"].rolling(n, min_periods=n // 2).apply(
        lambda v: float(np.argmin(v)), raw=True)
    return _sig(hi - lo)


def ichimoku(d):
    """Price against the cloud."""
    h, lo = d["high"], d["low"]
    conv = (h.rolling(9, min_periods=5).max() + lo.rolling(9, min_periods=5).min()) / 2
    base = (h.rolling(26, min_periods=13).max() + lo.rolling(26, min_periods=13).min()) / 2
    a = ((conv + base) / 2).shift(26)
    b = ((h.rolling(52, min_periods=26).max()
          + lo.rolling(52, min_periods=26).min()) / 2).shift(26)
    top, bot = pd.concat([a, b], axis=1).max(axis=1), pd.concat([a, b], axis=1).min(axis=1)
    c = d["close"]
    return _sig((c > top).astype(float) - (c < bot).astype(float))


def psar_like(d):
    """A parabolic-style trailing state, without the recursion."""
    c = d["close"]
    trail_up = c.rolling(30, min_periods=10).min()
    trail_dn = c.rolling(30, min_periods=10).max()
    return _sig((c > trail_dn.shift(1)).astype(float)
                - (c < trail_up.shift(1)).astype(float)).replace(0.0, np.nan).ffill().fillna(0.0)


def adx_trend(d):
    """Direction from +DI/-DI, but only while ADX says a trend exists."""
    h, lo = d["high"], d["low"]
    up, dn = h.diff(), -lo.diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr = _atr(d, 14)
    pdi = 100 * _rma(pd.Series(plus, index=d.index), 14) / atr.replace(0, np.nan)
    mdi = 100 * _rma(pd.Series(minus, index=d.index), 14) / atr.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx = _rma(dx, 14)
    return _sig(np.sign(pdi - mdi) * (adx > 20).astype(float))


def hurst_trend(d):
    """Trend only where the series is persistent, mean-revert where not."""
    c = np.log(d["close"])
    short = c.diff().rolling(30, min_periods=15).std()
    long = c.diff(30).rolling(30, min_periods=15).std() / np.sqrt(30)
    persistent = (long / short.replace(0, np.nan)) > 1.0
    return _sig(np.sign(c.diff(30)) * persistent.astype(float))


# -------------------------------------------------------- mean reversion

def rsi_extreme(d):
    c = d["close"].diff()
    up = _rma(c.clip(lower=0), 14)
    dn = _rma((-c).clip(lower=0), 14)
    rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    return _sig((rsi < 30).astype(float) - (rsi > 70).astype(float))


def rsi_trend(d):
    """The same indicator read the OPPOSITE way -- above 50 is strength."""
    c = d["close"].diff()
    up = _rma(c.clip(lower=0), 14)
    dn = _rma((-c).clip(lower=0), 14)
    rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    return _sig(rsi - 50)


def bollinger_fade(d):
    c = d["close"]
    m = c.rolling(20, min_periods=10).mean()
    s = c.rolling(20, min_periods=10).std()
    return _sig((c < m - 2 * s).astype(float) - (c > m + 2 * s).astype(float))


def bollinger_break(d):
    c = d["close"]
    m = c.rolling(20, min_periods=10).mean()
    s = c.rolling(20, min_periods=10).std()
    return _sig((c > m + 2 * s).astype(float) - (c < m - 2 * s).astype(float))


def zscore_fade(d):
    return _sig(-_z(d["close"], 60))


def zscore_fade_fast(d):
    return _sig(-_z(d["close"], 15))


def keltner_fade(d):
    c, atr = d["close"], _atr(d, 20)
    m = _ema(c, 20)
    return _sig((c < m - 2 * atr).astype(float) - (c > m + 2 * atr).astype(float))


def williams_r(d):
    n = 30
    hh = d["high"].rolling(n, min_periods=n // 2).max()
    ll = d["low"].rolling(n, min_periods=n // 2).min()
    r = -100 * (hh - d["close"]) / (hh - ll).replace(0, np.nan)
    return _sig((r < -80).astype(float) - (r > -20).astype(float))


def cci_fade(d):
    tp = (d["high"] + d["low"] + d["close"]) / 3
    m = tp.rolling(20, min_periods=10).mean()
    md = (tp - m).abs().rolling(20, min_periods=10).mean()
    cci = (tp - m) / (0.015 * md.replace(0, np.nan))
    return _sig((cci < -100).astype(float) - (cci > 100).astype(float))


def stochastic(d):
    n = 14
    hh = d["high"].rolling(n, min_periods=n // 2).max()
    ll = d["low"].rolling(n, min_periods=n // 2).min()
    k = 100 * (d["close"] - ll) / (hh - ll).replace(0, np.nan)
    dd = k.rolling(3, min_periods=1).mean()
    return _sig((dd < 20).astype(float) - (dd > 80).astype(float))


def vwap_fade(d):
    tp = (d["high"] + d["low"] + d["close"]) / 3
    v = d["volume"].clip(lower=0)
    vw = (tp * v).rolling(240, min_periods=60).sum() / \
        v.rolling(240, min_periods=60).sum().replace(0, np.nan)
    return _sig(-(d["close"] - vw))


def vwap_trend(d):
    tp = (d["high"] + d["low"] + d["close"]) / 3
    v = d["volume"].clip(lower=0)
    vw = (tp * v).rolling(240, min_periods=60).sum() / \
        v.rolling(240, min_periods=60).sum().replace(0, np.nan)
    return _sig(d["close"] - vw)


def gap_fade(d):
    """Fade a single bar that moved far more than its neighbours."""
    r = d["close"].pct_change()
    z = _z(r, 120)
    return _sig(-np.sign(r) * (z.abs() > 3).astype(float))


def reversal_5(d):
    return _sig(-d["close"].pct_change(5))


def momentum_60(d):
    return _sig(d["close"].pct_change(60))


def momentum_240(d):
    return _sig(d["close"].pct_change(240))


# ----------------------------------------------------------- volatility

def squeeze_break(d):
    """Bollinger inside Keltner is a squeeze; trade the way it releases."""
    c = d["close"]
    s = c.rolling(20, min_periods=10).std()
    atr = _atr(d, 20)
    squeeze = (2 * s) < (1.5 * atr)
    return _sig(np.sign(c.diff(5)) * squeeze.shift(1).astype(float))


def atr_expansion(d):
    atr = _atr(d, 14)
    expanding = atr > atr.shift(60)
    return _sig(np.sign(d["close"].diff(15)) * expanding.astype(float))


def atr_contraction(d):
    atr = _atr(d, 14)
    quiet = atr < atr.shift(60)
    return _sig(-np.sign(d["close"].diff(15)) * quiet.astype(float))


def opening_range(d):
    """Break of the first hour's range, held for the session."""
    day = d.index.floor("1D")
    first = d.groupby(day).head(60)
    hi = first.groupby(first.index.floor("1D"))["high"].max().reindex(day).values
    lo = first.groupby(first.index.floor("1D"))["low"].min().reindex(day).values
    c = d["close"].values
    return pd.Series(np.where(c > hi, 1.0, np.where(c < lo, -1.0, 0.0)),
                     index=d.index)


def range_position(d):
    n = 120
    hh = d["high"].rolling(n, min_periods=n // 2).max()
    ll = d["low"].rolling(n, min_periods=n // 2).min()
    pos = (d["close"] - ll) / (hh - ll).replace(0, np.nan)
    return _sig((pos > 0.8).astype(float) - (pos < 0.2).astype(float))


# --------------------------------------------------------- volume / flow

def obv_trend(d):
    obv = (np.sign(d["close"].diff()) * d["volume"].clip(lower=0)).cumsum()
    return _sig(obv - obv.rolling(120, min_periods=60).mean())


def mfi(d):
    tp = (d["high"] + d["low"] + d["close"]) / 3
    mf = tp * d["volume"].clip(lower=0)
    pos = mf.where(tp.diff() > 0, 0.0).rolling(14, min_periods=7).sum()
    neg = mf.where(tp.diff() < 0, 0.0).rolling(14, min_periods=7).sum()
    m = 100 - 100 / (1 + pos / neg.replace(0, np.nan))
    return _sig((m < 20).astype(float) - (m > 80).astype(float))


def volume_thrust(d):
    v = d["volume"].clip(lower=0)
    spike = _z(v, 240) > 2
    return _sig(np.sign(d["close"].diff()) * spike.astype(float))


def flow_imbalance(d):
    """Where each bar closed in its own range, volume-weighted. The
    closest an OHLCV feed gets to reading the tape."""
    h, lo, c, v = d["high"], d["low"], d["close"], d["volume"].clip(lower=0)
    loc = ((c - lo) - (h - c)) / (h - lo).replace(0, np.nan)
    f = (loc * v).rolling(30, min_periods=10).sum() / \
        v.rolling(30, min_periods=10).sum().replace(0, np.nan)
    return _sig(f)


def volume_dry_reversal(d):
    v = d["volume"].clip(lower=0)
    dry = _z(v, 240) < -1
    return _sig(-np.sign(d["close"].diff(10)) * dry.astype(float))


# ------------------------------------------------------- cross-sectional
# These need the whole board, so they take the panel rather than one frame.

def rel_strength(d, panel=None, sym=None):
    """Long the strongest coins, short the weakest, over an hour."""
    if panel is None:
        return pd.Series(0.0, index=d.index)
    R = pd.DataFrame({s: v["close"].pct_change(60)
                      for s, v in panel.items()}).reindex(d.index)
    rank = R.rank(axis=1, pct=True)[sym]
    return _sig((rank > 0.8).astype(float) - (rank < 0.2).astype(float))


def rel_weakness(d, panel=None, sym=None):
    """The same ranking traded the other way -- buy what lagged."""
    return -rel_strength(d, panel, sym)


def residual_momentum(d, panel=None, sym=None):
    """Momentum with the market taken out."""
    if panel is None:
        return pd.Series(0.0, index=d.index)
    R = pd.DataFrame({s: v["close"].pct_change(60)
                      for s, v in panel.items()}).reindex(d.index)
    return _sig(R[sym] - R.median(axis=1))


def breadth_trend(d, panel=None, sym=None):
    """Follow this coin only when the board agrees with it."""
    if panel is None:
        return pd.Series(0.0, index=d.index)
    R = pd.DataFrame({s: v["close"].pct_change(60)
                      for s, v in panel.items()}).reindex(d.index)
    breadth = (R > 0).mean(axis=1)
    own = np.sign(R[sym])
    agree = ((own > 0) & (breadth > 0.6)) | ((own < 0) & (breadth < 0.4))
    return _sig(own * agree.astype(float))


def market_neutral_fade(d, panel=None, sym=None):
    """Fade a coin that has run far away from the board."""
    if panel is None:
        return pd.Series(0.0, index=d.index)
    R = pd.DataFrame({s: v["close"].pct_change(60)
                      for s, v in panel.items()}).reindex(d.index)
    resid = R[sym] - R.median(axis=1)
    return _sig(-_z(resid, 240))


# ------------------------------------------------------------------ index

SINGLE = {
    # trend
    "ema_fast": ema_cross_fast, "ema_mid": ema_cross_mid,
    "ema_slow": ema_cross_slow, "macd": macd, "macd_zero": macd_zero,
    "price_ma": price_vs_ma, "linreg": linreg_slope,
    "supertrend": supertrend, "donch_fast": donchian_fast,
    "donch_mid": donchian_mid, "donch_slow": donchian_slow,
    "aroon": aroon, "ichimoku": ichimoku, "psar": psar_like,
    "adx": adx_trend, "hurst": hurst_trend,
    # mean reversion
    "rsi_fade": rsi_extreme, "rsi_trend": rsi_trend,
    "boll_fade": bollinger_fade, "boll_break": bollinger_break,
    "z_fade": zscore_fade, "z_fade_fast": zscore_fade_fast,
    "keltner": keltner_fade, "williams": williams_r, "cci": cci_fade,
    "stoch": stochastic, "vwap_fade": vwap_fade, "vwap_trend": vwap_trend,
    "gap_fade": gap_fade, "rev5": reversal_5,
    "mom60": momentum_60, "mom240": momentum_240,
    # volatility
    "squeeze": squeeze_break, "atr_exp": atr_expansion,
    "atr_con": atr_contraction, "open_range": opening_range,
    "range_pos": range_position,
    # volume and flow
    "obv": obv_trend, "mfi": mfi, "vol_thrust": volume_thrust,
    "flow": flow_imbalance, "vol_dry": volume_dry_reversal,
}

CROSS = {
    "rel_str": rel_strength, "rel_weak": rel_weakness,
    "resid_mom": residual_momentum, "breadth": breadth_trend,
    "mkt_fade": market_neutral_fade,
}

ALL = {**SINGLE, **CROSS}

FAMILY = {
    "trend": ["ema_fast", "ema_mid", "ema_slow", "macd", "macd_zero",
              "price_ma", "linreg", "supertrend", "donch_fast", "donch_mid",
              "donch_slow", "aroon", "ichimoku", "psar", "adx", "hurst"],
    "revert": ["rsi_fade", "boll_fade", "z_fade", "z_fade_fast", "keltner",
               "williams", "cci", "stoch", "vwap_fade", "gap_fade", "rev5"],
    "breakout": ["boll_break", "squeeze", "atr_exp", "open_range",
                 "range_pos", "mom60", "mom240", "rsi_trend", "vwap_trend"],
    "flow": ["obv", "mfi", "vol_thrust", "flow", "vol_dry", "atr_con"],
    "cross": ["rel_str", "rel_weak", "resid_mom", "breadth", "mkt_fade"],
}


def states(d: pd.DataFrame, panel=None, sym=None) -> pd.DataFrame:
    """Every method's signed state for one symbol, aligned on its bars."""
    out = {}
    for name, fn in SINGLE.items():
        try:
            out[name] = fn(d)
        except Exception:
            out[name] = pd.Series(0.0, index=d.index)
    for name, fn in CROSS.items():
        try:
            out[name] = fn(d, panel, sym)
        except Exception:
            out[name] = pd.Series(0.0, index=d.index)
    X = pd.DataFrame(out, index=d.index).replace([np.inf, -np.inf], np.nan)
    return X.fillna(0.0).astype("int8")
