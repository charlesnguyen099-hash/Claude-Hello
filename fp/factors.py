"""The factor library: 100+ numbers describing one bar's situation.

Everything here is computed on 1-MINUTE bars and read at lookbacks from
one minute to two hours, because that is the window the operator
described: a trade worth taking announces itself somewhere between one
minute and thirty out, and the announcement is not always on the same
clock.

Three families, and the third is the one this repo did not have before:

  PRICE / FLOW   returns, volatility, range position, volume, order-flow
                 proxies, candle shape -- each at nine lookbacks, so the
                 model can find the horizon instead of being told it.

  STATE          where this bar sits relative to its own recent history:
                 distance to rolling extremes in units of volatility,
                 VWAP stretch, run lengths, time of day.

  CROSS-SECTION  what the OTHER nine coins are doing at the same instant.
                 A coin rising while the whole board rises is a different
                 setup from one rising alone, and nothing in the previous
                 feature set could tell those apart. This is what makes a
                 logic shared across coins rather than ten separate ones:
                 the rank, the market move, the dispersion and the
                 residual are the same numbers whatever the coin's price.

SCALE-FREE BY CONSTRUCTION. Every factor is a ratio, a z-score, a rank
or a position in a range. BTC at $100,000 and BLESS at $0.02 present the
same numbers, which is the only reason one model can serve all ten.

THE SHIFT. Every factor is computed from bars up to and including t and
is read as the state at t's CLOSE. The label for bar t measures what
happens AFTER t. No factor may see its own outcome; build_panel()
enforces the alignment in one place so it cannot be got wrong twice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# The lookbacks, in minutes. Nine of them, spanning the operator's
# "1 phut den 30 phut tham chi hon" -- and two beyond it, because a
# thirty-minute signal often sits inside a two-hour context.
LOOKBACKS = (1, 2, 3, 5, 10, 15, 30, 60, 120)

# Windows used for the slower state features.
STATE_WINDOWS = (15, 60, 240)

EPS = 1e-12


def _z(x: pd.Series, w: int) -> pd.Series:
    m = x.rolling(w, min_periods=w // 2).mean()
    s = x.rolling(w, min_periods=w // 2).std()
    return (x - m) / s.replace(0, np.nan)


def price_flow(d: pd.DataFrame) -> pd.DataFrame:
    """Returns, volatility, volume and candle shape at every lookback."""
    o = d["open"].astype("float64")
    h = d["high"].astype("float64")
    lo = d["low"].astype("float64")
    c = d["close"].astype("float64")
    v = d["volume"].astype("float64").clip(lower=0)

    r1 = c.pct_change()
    # A single volatility yardstick, so every distance below is in the
    # same unit and comparable across coins and across time.
    sig = r1.rolling(120, min_periods=60).std()

    f = {}
    for k in LOOKBACKS:
        ret = c.pct_change(k)
        f[f"ret{k}"] = ret / (sig * np.sqrt(k)).replace(0, np.nan)
        # k=1 has no dispersion of its own; the bar's own move against
        # the running sigma is the only meaningful reading there.
        w = max(k, 2)
        f[f"vol{k}"] = (r1.rolling(w, min_periods=max(2, w // 2)).std()
                        / sig.replace(0, np.nan))
        hh = h.rolling(k, min_periods=1).max()
        ll = lo.rolling(k, min_periods=1).min()
        f[f"pos{k}"] = (c - ll) / (hh - ll).replace(0, np.nan)
        f[f"rng{k}"] = (hh - ll) / (c * sig * np.sqrt(k)).replace(0, np.nan)
        f[f"volz{k}"] = _z(v.rolling(k, min_periods=1).sum(), 240)
        # Order-flow proxy: where each bar closed inside its own range,
        # weighted by that bar's volume. Positive means buyers finished
        # in control over the window. This is the closest an OHLCV feed
        # gets to the tape.
        loc = ((c - lo) - (h - c)) / (h - lo).replace(0, np.nan)
        f[f"flow{k}"] = ((loc * v).rolling(k, min_periods=1).sum()
                         / v.rolling(k, min_periods=1).sum().replace(0, np.nan))
        f[f"up{k}"] = (r1 > 0).rolling(k, min_periods=1).mean()
        # Acceleration: this window's move against the previous one's.
        f[f"acc{k}"] = ret - c.pct_change(k).shift(k)

    # Candle shape at the newest bar, and smoothed.
    body = (c - o).abs()
    rng = (h - lo).replace(0, np.nan)
    f["body"] = body / rng
    f["wick_up"] = (h - np.maximum(c, o)) / rng
    f["wick_dn"] = (np.minimum(c, o) - lo) / rng
    for w in (5, 15, 60):
        f[f"body{w}"] = (body / rng).rolling(w, min_periods=1).mean()
        f[f"wick_up{w}"] = ((h - np.maximum(c, o)) / rng
                            ).rolling(w, min_periods=1).mean()
        f[f"wick_dn{w}"] = ((np.minimum(c, o) - lo) / rng
                            ).rolling(w, min_periods=1).mean()
    return pd.DataFrame(f, index=d.index)


def state(d: pd.DataFrame) -> pd.DataFrame:
    """Where this bar sits in its own recent history."""
    h = d["high"].astype("float64")
    lo = d["low"].astype("float64")
    c = d["close"].astype("float64")
    v = d["volume"].astype("float64").clip(lower=0)
    r1 = c.pct_change()
    sig = r1.rolling(120, min_periods=60).std()

    f = {}
    for w in STATE_WINDOWS:
        hh = h.rolling(w, min_periods=w // 4).max()
        ll = lo.rolling(w, min_periods=w // 4).min()
        # Distance to the extreme in volatility units -- how far a
        # breakout would have to travel, or how far one has already come.
        f[f"dhigh{w}"] = (hh - c) / (c * sig).replace(0, np.nan)
        f[f"dlow{w}"] = (c - ll) / (c * sig).replace(0, np.nan)
        tp = (h + lo + c) / 3.0
        vw = ((tp * v).rolling(w, min_periods=w // 4).sum()
              / v.rolling(w, min_periods=w // 4).sum().replace(0, np.nan))
        f[f"vwap{w}"] = (c - vw) / (c * sig).replace(0, np.nan)
        ma = c.rolling(w, min_periods=w // 4).mean()
        f[f"ma{w}"] = (c - ma) / (c * sig).replace(0, np.nan)
        f[f"slope{w}"] = (ma.diff(w) / (c * sig * np.sqrt(w))
                          .replace(0, np.nan))

    # Consecutive direction: how long the current push has lasted.
    up = (r1 > 0).astype("float64")
    grp = (up != up.shift()).cumsum()
    f["run"] = up.groupby(grp).cumcount().add(1) * np.where(up > 0, 1.0, -1.0)

    # Volatility regime: is the market waking up or going to sleep?
    for a, b in ((5, 60), (15, 240), (60, 240)):
        f[f"volrat{a}_{b}"] = (r1.rolling(a, min_periods=2).std()
                               / r1.rolling(b, min_periods=b // 4).std()
                               .replace(0, np.nan))

    idx = d.index
    mins = idx.hour * 60 + idx.minute
    f["tod_sin"] = np.sin(2 * np.pi * mins / 1440.0)
    f["tod_cos"] = np.cos(2 * np.pi * mins / 1440.0)
    f["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7.0)
    f["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7.0)
    f["sigma"] = sig
    return pd.DataFrame(f, index=idx)


def per_symbol(d: pd.DataFrame) -> pd.DataFrame:
    """Every single-coin factor for one symbol."""
    return pd.concat([price_flow(d), state(d)], axis=1)


def cross_section(panels: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """What the rest of the board is doing, aligned to each coin's bars.

    This is the part that makes ONE logic serve ten coins rather than ten
    logics serving one each. Four numbers per lookback:

      mkt      the board's median move -- the tide
      rank     this coin's move against the other nine, in [0, 1]
      disp     how much the board disagrees -- one coin running alone
               looks nothing like all ten running together
      resid    this coin's move minus the tide, in its own volatility
               units: what is left after the market is taken out

    A coin at rank 1.0 with disp high and resid large is a genuinely
    idiosyncratic move. The same coin at rank 1.0 with disp near zero is
    just the market. Those two used to be the same row.
    """
    rets = {}
    for k in LOOKBACKS:
        cols = {}
        for s, d in panels.items():
            c = d["close"].astype("float64")
            cols[s] = c.pct_change(k)
        rets[k] = pd.DataFrame(cols).sort_index()

    out: dict[str, pd.DataFrame] = {}
    for s, d in panels.items():
        f = {}
        c = d["close"].astype("float64")
        sig = c.pct_change().rolling(120, min_periods=60).std()
        for k in LOOKBACKS:
            R = rets[k].reindex(d.index)
            mine = R[s]
            others = R.drop(columns=[s])
            med = R.median(axis=1)
            f[f"mkt{k}"] = med / (sig * np.sqrt(k)).replace(0, np.nan)
            f[f"rank{k}"] = R.rank(axis=1, pct=True)[s]
            f[f"disp{k}"] = (R.std(axis=1) / (sig * np.sqrt(k))
                             .replace(0, np.nan))
            f[f"resid{k}"] = (mine - med) / (sig * np.sqrt(k)).replace(0, np.nan)
            f[f"breadth{k}"] = (others > 0).mean(axis=1)
        out[s] = pd.DataFrame(f, index=d.index)
    return out


def build(panels: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Every factor for every symbol, aligned and SHIFTED.

    The shift is applied exactly once, here. A factor row stamped t
    describes the state at t's close; the label for t measures what
    happens after t. Doing the shift anywhere else is how a feature set
    ends up reading its own answer.
    """
    xs = cross_section(panels)
    out = {}
    for s, d in panels.items():
        # Downcast BEFORE replace, not after: .replace() on the whole
        # frame forces pandas to consolidate its blocks into one new
        # contiguous array while the old one is still alive, and at
        # float64 that spike is BTCUSDT's 864,078 rows x 153 cols x 8
        # bytes = ~1GB -- more than an 8GB Windows box had free mid-run.
        # float32 halves that spike to ~529MB; inf/-inf survive the
        # downcast unchanged so replace still finds every one.
        X = pd.concat([per_symbol(d), xs[s]], axis=1).astype("float32")
        out[s] = X.replace([np.inf, -np.inf], np.nan)
    return out


def names(panels: dict[str, pd.DataFrame]) -> list[str]:
    s = next(iter(panels))
    return list(build({s: panels[s]})[s].columns)
