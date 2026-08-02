import numpy as np
import pandas as pd
import pytest

from bot import strategy


def _fake_1m_ohlcv(days=10, seed=7, trend_per_day=0.0):
    n = days * 24 * 60
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="1min")
    noise = rng.normal(0, 5.0, n).cumsum()
    trend = np.linspace(0, trend_per_day * days, n)
    close = 50_000 + trend + noise
    high = close + rng.uniform(0, 10, n)
    low = close - rng.uniform(0, 10, n)
    open_ = close + rng.normal(0, 3, n)
    volume = rng.uniform(1, 20, n)
    return pd.DataFrame(
        {"datetime": idx, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


def test_prepare_runs_and_has_expected_columns():
    df = _fake_1m_ohlcv(days=15)
    out = strategy.prepare(df)
    for col in ("trend", "long_setup", "short_setup", "confidence_long", "confidence_short", "atr14"):
        assert col in out.columns
    assert len(out) > 0


def test_no_trade_in_flat_market():
    # Pure random walk, no sustained trend -> ADX should rarely clear
    # threshold and setups should be rare-to-none.
    df = _fake_1m_ohlcv(days=15, trend_per_day=0.0)
    out = strategy.prepare(df)
    setups = out["long_setup"].sum() + out["short_setup"].sum()
    assert setups <= len(out) * 0.05  # sanity: not constantly signaling


def test_signal_from_row_requires_positive_atr():
    row = pd.Series({"atr14": 0.0, "long_setup": True, "short_setup": False, "close": 100})
    assert strategy.signal_from_row(row) is None


def test_long_signal_has_stop_below_entry_and_positive_targets():
    row = pd.Series(
        {
            "atr14": 10.0,
            "long_setup": True,
            "short_setup": False,
            "close": 1000.0,
            "confidence_long": 70.0,
        }
    )
    sig = strategy.signal_from_row(row)
    assert sig is not None
    assert sig.side == "long"
    assert sig.stop < sig.entry
    assert sig.take_profit_1 > sig.entry
    assert sig.take_profit_2 > sig.take_profit_1


def test_short_signal_has_stop_above_entry_and_negative_targets():
    row = pd.Series(
        {
            "atr14": 10.0,
            "long_setup": False,
            "short_setup": True,
            "close": 1000.0,
            "confidence_short": 70.0,
        }
    )
    sig = strategy.signal_from_row(row)
    assert sig is not None
    assert sig.side == "short"
    assert sig.stop > sig.entry
    assert sig.take_profit_1 < sig.entry
    assert sig.take_profit_2 < sig.take_profit_1


def test_no_lookahead_truncated_prefix_matches_full_run():
    """The signal computed at bar i must be identical whether or not
    future bars exist in the input. This proves build_signal_columns /
    merge_htf_trend never peek forward.
    """
    df = _fake_1m_ohlcv(days=20, seed=3, trend_per_day=8000)
    full = strategy.prepare(df)

    # Pick a bar well before the end of `full`, then truncate the raw 1m
    # input just past that bar's own close (+20min margin) so the target
    # bar's own 15m bin and its governing 1h bin are both fully formed in
    # the truncated run too, while everything after it is invisible.
    target_ts = full["datetime"].iloc[len(full) - 50]
    cutoff_ts = target_ts + pd.Timedelta(minutes=20)
    df_truncated = df[df["datetime"] <= cutoff_ts].reset_index(drop=True)
    truncated = strategy.prepare(df_truncated)

    full_row = full[full["datetime"] == target_ts].iloc[0]
    trunc_row = truncated[truncated["datetime"] == target_ts].iloc[0]

    assert full_row["trend"] == trunc_row["trend"]
    assert bool(full_row["long_setup"]) == bool(trunc_row["long_setup"])
    assert bool(full_row["short_setup"]) == bool(trunc_row["short_setup"])
    assert full_row["atr14"] == pytest.approx(trunc_row["atr14"], rel=1e-9)
