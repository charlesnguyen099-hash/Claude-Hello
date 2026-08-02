import numpy as np
import pandas as pd

from bot import indicators as ind


def _fake_ohlcv(n=300, seed=1):
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 1.0, n).cumsum()
    close = 100 + steps
    high = close + rng.uniform(0, 1, n)
    low = close - rng.uniform(0, 1, n)
    open_ = close + rng.normal(0, 0.3, n)
    volume = rng.uniform(1, 10, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


def test_ema_tracks_price_within_reasonable_band():
    df = _fake_ohlcv()
    e = ind.ema(df["close"], 20)
    recent_mean = df["close"].iloc[-20:].mean()
    assert abs(e.iloc[-1] - recent_mean) <= abs(recent_mean) * 0.5


def test_rsi_bounded_0_100():
    df = _fake_ohlcv()
    r = ind.rsi(df["close"])
    assert r.min() >= 0.0
    assert r.max() <= 100.0


def test_atr_non_negative():
    df = _fake_ohlcv()
    a = ind.atr(df)
    assert (a.dropna() >= 0).all()


def test_adx_bounded_0_100():
    df = _fake_ohlcv()
    a = ind.adx(df)
    assert a.min() >= 0.0
    assert a.max() <= 100.0


def test_ema_all_nan_before_warmup():
    df = _fake_ohlcv(n=10)
    e = ind.ema(df["close"], 20)
    assert e.isna().all()
