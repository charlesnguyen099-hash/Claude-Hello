"""Fetch the data that is NOT price: funding, open interest, positioning.

    python -m fp.altdata --top 120 --days 180
    python -m fp.altdata --status

WHY THIS EXISTS

Six searches in this project have failed, and every one of them used the
same family of input: indicators computed from BTCUSDT's own OHLCV. RSI,
EMA distance, ATR, volume ratios, candle shapes, 12-candle patterns --
all of it is price, rearranged. If price is close to a martingale, every
rearrangement of price is too, and no amount of searching over that
family escapes it. That is not a claim about markets; it is a claim about
what was searched.

Market psychology is not in the price. It is in the positioning, and
Bybit publishes it:

  funding rate        what longs are paying shorts to stay long. Extreme
                      positive funding means the crowd is long and paying
                      for it -- the classic setup for a squeeze.
  open interest       whether a move is new money entering or old
                      positions closing. Price up on rising OI is very
                      different from price up on falling OI.
  long/short ratio    the account-level split, published per symbol.

And one structural family that needs no new endpoint, only more symbols:

  cross-sectional     rank 690 coins against each other rather than each
                      against its own past. Cross-sectional momentum and
                      reversal are among the most replicated effects in
                      finance, and they cannot be expressed at all in a
                      single-symbol time series. The bot already scans the
                      whole board; nothing here has ever used that.

WHAT THIS MODULE DOES

Downloads and caches all of it, so the search can be run offline
afterwards. Nothing is analysed here -- this is only the fetch, kept
separate so a rate limit or a dropped connection never corrupts a result.

    data/alt/funding_<SYMBOL>.csv    8-hourly funding, full history
    data/alt/oi_<SYMBOL>.csv        open interest, 1h
    data/alt/ls_<SYMBOL>.csv        long/short account ratio
    data/alt/kline_<SYMBOL>.csv     30m OHLCV, aligned to the above

Resumable: a symbol already on disk with enough rows is skipped, so the
command can be re-run after an interruption or to extend the history.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "bybit_bot" / "data" / "alt"
MS_HOUR = 3_600_000


def _paged(fn, key_time: str, start_ms: int, end_ms: int, **kw) -> list[dict]:
    """Walk a Bybit endpoint backwards until the window is covered."""
    rows, cursor, guard = [], end_ms, 0
    while cursor > start_ms and guard < 400:
        guard += 1
        try:
            r = fn(endTime=cursor, **kw)
        except Exception as exc:
            print(f"      stopped: {type(exc).__name__}: {exc}")
            break
        chunk = r.get("result", {}).get("list", [])
        if not chunk:
            break
        rows.extend(chunk)
        try:
            oldest = min(int(x[key_time]) for x in chunk)
        except (KeyError, ValueError):
            break
        if oldest >= cursor:
            break
        cursor = oldest - 1
        time.sleep(0.12)
    return rows


def fetch_symbol(client, sym: str, start_ms: int, end_ms: int) -> dict:
    got = {}

    rows = _paged(client.get_funding_rate_history, "fundingRateTimestamp",
                  start_ms, end_ms, category="linear", symbol=sym, limit=200)
    if rows:
        d = pd.DataFrame(rows)
        d["datetime"] = pd.to_datetime(d["fundingRateTimestamp"].astype("int64"),
                                       unit="ms")
        d["funding"] = d["fundingRate"].astype(float)
        got["funding"] = d[["datetime", "funding"]].sort_values("datetime")

    rows = _paged(client.get_open_interest, "timestamp", start_ms, end_ms,
                  category="linear", symbol=sym, intervalTime="1h", limit=200)
    if rows:
        d = pd.DataFrame(rows)
        d["datetime"] = pd.to_datetime(d["timestamp"].astype("int64"), unit="ms")
        d["oi"] = d["openInterest"].astype(float)
        got["oi"] = d[["datetime", "oi"]].sort_values("datetime")

    try:
        r = client.get_long_short_ratio(category="linear", symbol=sym,
                                        period="1h", limit=500)
        rows = r.get("result", {}).get("list", [])
        if rows:
            d = pd.DataFrame(rows)
            d["datetime"] = pd.to_datetime(d["timestamp"].astype("int64"), unit="ms")
            d["long_ratio"] = d["buyRatio"].astype(float)
            got["ls"] = d[["datetime", "long_ratio"]].sort_values("datetime")
    except Exception:
        pass

    rows, cursor, guard = [], end_ms, 0
    while cursor > start_ms and guard < 200:
        guard += 1
        try:
            r = client.get_kline(category="linear", symbol=sym, interval="30",
                                 end=cursor, limit=1000)
        except Exception:
            break
        chunk = r.get("result", {}).get("list", [])
        if not chunk:
            break
        rows.extend(chunk)
        oldest = min(int(x[0]) for x in chunk)
        if oldest >= cursor:
            break
        cursor = oldest - 1
        time.sleep(0.12)
    if rows:
        d = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close",
                                        "volume", "turnover"])
        d["datetime"] = pd.to_datetime(d["ts"].astype("int64"), unit="ms")
        for c in ("open", "high", "low", "close", "volume", "turnover"):
            d[c] = d[c].astype(float)
        got["kline"] = (d.drop(columns=["ts"]).drop_duplicates("datetime")
                        .sort_values("datetime"))
    return got


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--top", type=int, default=120,
                   help="how many symbols by 24h turnover (cross-sectional "
                        "work needs breadth, not depth -- 120 is plenty)")
    p.add_argument("--days", type=int, default=180)
    p.add_argument("--symbols", default=None)
    p.add_argument("--status", action="store_true")
    p.add_argument("--testnet", action="store_true")
    a = p.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    if a.status:
        kinds = ("funding", "oi", "ls", "kline")
        have = {k: sorted(OUT.glob(f"{k}_*.csv")) for k in kinds}
        for k in kinds:
            n = len(have[k])
            rows = sum(sum(1 for _ in f.open()) - 1 for f in have[k][:400])
            print(f"  {k:>8}: {n:>4} symbols, {rows:,} rows")
        if not any(have.values()):
            print("  nothing downloaded yet -- run without --status")
        return 0

    from pybit.unified_trading import HTTP
    client = HTTP(testnet=a.testnet)

    if a.symbols:
        syms = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    else:
        r = client.get_tickers(category="linear")
        rows = [x for x in r["result"]["list"] if x["symbol"].endswith("USDT")]
        rows.sort(key=lambda x: float(x.get("turnover24h") or 0), reverse=True)
        syms = [x["symbol"] for x in rows][:a.top]

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - a.days * 24 * MS_HOUR
    print(f"{len(syms)} symbols, {a.days} days back\n")

    for i, sym in enumerate(syms, 1):
        done = OUT / f"kline_{sym}.csv"
        if done.exists() and sum(1 for _ in done.open()) > 2000:
            print(f"[{i:>3}/{len(syms)}] {sym:<16} cached")
            continue
        print(f"[{i:>3}/{len(syms)}] {sym:<16} ", end="", flush=True)
        try:
            got = fetch_symbol(client, sym, start_ms, end_ms)
        except Exception as exc:
            print(f"failed: {type(exc).__name__}: {exc}")
            continue
        bits = []
        for kind, df in got.items():
            df.to_csv(OUT / f"{kind}_{sym}.csv", index=False)
            bits.append(f"{kind} {len(df)}")
        print(", ".join(bits) if bits else "nothing returned")

    print(f"\nsaved to {OUT}")
    print("now run:  python -m fp.crosssearch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
