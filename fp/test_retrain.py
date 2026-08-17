"""Checks for fp/retrain.py's paging and merge logic, no network.

The one thing that must never happen here is data loss or corruption of
the cache the whole logic is fitted from -- a bad merge silently feeds
fp.full a smaller or duplicated history and every number downstream is
wrong in a way nothing else would catch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fp import retrain as R

PASS = 0


def chk(cond, msg):
    global PASS
    if not cond:
        raise AssertionError(msg)
    PASS += 1


class FakeClient:
    """Serves fixed pages of kline rows, newest-call-agnostic order."""

    def __init__(self, pages):
        self.pages = list(pages)   # list of lists of [ts_ms, o,h,l,c,v,t]
        self.calls = []

    def get_kline(self, **kw):
        self.calls.append(kw)
        if not self.pages:
            return {"result": {"list": []}}
        return {"result": {"list": self.pages.pop(0)}}


def bar(ts_ms, px=100.0):
    return [str(ts_ms), str(px), str(px), str(px), str(px), "1", "100"]


def test_fetch_since_pages_forward():
    # Two pages: first full (forces another call), second short (stops).
    t0 = 1_700_000_000_000
    page1 = [bar(t0 + i * 60_000) for i in range(1000)]
    page2 = [bar(t0 + (1000 + i) * 60_000) for i in range(5)]
    client = FakeClient([page1, page2])
    df = R.fetch_since(client, "XUSDT", t0, log=lambda s: None)
    chk(df is not None, "must return rows")
    chk(len(client.calls) == 2, f"a full page must trigger another call, "
        f"got {len(client.calls)} calls")
    chk(df["ts"].is_monotonic_increasing, "rows must come back time-sorted")


def test_fetch_since_drops_forming_candle():
    import time
    now = int(time.time() * 1000)
    still_forming = now - 10_000          # 10s ago: this minute isn't closed
    closed = now - 130_000                # over 2 minutes ago: closed
    client = FakeClient([[bar(closed), bar(still_forming)]])
    df = R.fetch_since(client, "XUSDT", closed - 60_000, log=lambda s: None)
    chk(df is not None and len(df) == 1,
        f"the still-forming candle must be dropped, got {len(df) if df is not None else 0}")


def test_fetch_since_empty_page_stops():
    client = FakeClient([[]])
    df = R.fetch_since(client, "XUSDT", 1_700_000_000_000, log=lambda s: None)
    chk(df is None, "an empty first page means nothing new")
    chk(len(client.calls) == 1, "must not call again after an empty page")


def test_refresh_cache_merge_dedup(tmp_path=None, monkeypatch=None):
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        cache = Path(td) / "all_1m.csv.gz"
        old = pd.DataFrame({
            "symbol": ["XUSDT", "XUSDT", "YUSDT"],
            "ts": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z",
                 "2026-01-01T00:00:00Z"], utc=True),
            "open": [1.0, 1.0, 2.0], "high": [1.0, 1.0, 2.0],
            "low": [1.0, 1.0, 2.0], "close": [1.0, 1.0, 2.0],
            "volume": [1.0, 1.0, 1.0], "turnover": [1.0, 1.0, 1.0],
        })
        old.to_csv(cache, index=False, compression="gzip")

        real_cache = R.D.CACHE
        R.D.CACHE = cache
        try:
            t0 = int(pd.Timestamp("2026-01-01T00:01:00Z").timestamp() * 1000)
            # A page that RE-SENDS the already-cached 00:01 bar (a
            # boundary re-read Bybit can legitimately do) plus one truly
            # new bar. The dup must not double the row count.
            resend = bar(t0)
            new_bar = bar(t0 + 60_000)

            class Router(FakeClient):
                def __init__(self, by_symbol):
                    self.by_symbol = by_symbol
                    self.calls = []

                def get_kline(self, **kw):
                    self.calls.append(kw)
                    pages = self.by_symbol.get(kw["symbol"], [])
                    return {"result": {"list": pages.pop(0) if pages else []}}

            router = Router({"XUSDT": [[resend, new_bar]], "YUSDT": [[]]})
            gained = R.refresh_cache(router, ["XUSDT", "YUSDT"],
                                     log=lambda s: None)
            chk(gained == 1, f"exactly one genuinely new bar, got {gained}")
            got = pd.read_csv(cache)
            chk(len(got) == 4, f"3 old + 1 new = 4 rows, got {len(got)}")
            xrows = got[got.symbol == "XUSDT"]
            chk(len(xrows) == 3, f"XUSDT: 2 old + 1 new = 3, got {len(xrows)}")
        finally:
            R.D.CACHE = real_cache


def main():
    for fn in (test_fetch_since_pages_forward,
               test_fetch_since_drops_forming_candle,
               test_fetch_since_empty_page_stops,
               test_refresh_cache_merge_dedup):
        fn()
    print("=" * 70)
    print(f"all {PASS} checks passed")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
