"""Keep the logic current: pull fresh bars, refit, ship.

WHY THIS FILE EXISTS. fp/full.py fits once, on the history in the cache
at that moment, and ships models that reproduce every trade in it at
100%. That is exactly what was asked for -- and it is also, on its own,
a snapshot. Measured honestly on held-out time (fp/full.py's oof_gate,
walked forward across several regimes with an isotonic-calibrated
confidence floor), none of the coins tested cleared their own break-even
on bars their classifier had never trained on. Walking the split closer
to the present made it WORSE, not better -- XAUUSDT went from 49%
accuracy at a fold ending mid-history to 34% at a fold ending near the
present. That is the signature of a regime the training window never
saw, not of noise.

A model fit once on a fixed window goes stale the moment the market
moves past that window, in exactly the way that measurement shows. The
fix is not to relax the honesty of the gate -- it is to stop the window
from going stale: refetch the bars the market has produced since the
last fit, fold them in, and refit. Each cycle's classifier trains on a
little more of the CURRENT regime than the last one did, and the
held-out fold used to set its live gate moves forward with it.

THE GUARANTEE THAT DOES NOT CHANGE. Every retrain runs the exact same
fp/full.py pipeline this session built by hand: every coin escalates the
capacity ladder until it reproduces 100% of the profitable trades in
whatever history is in the cache, the same way it always has. Retraining
does not loosen that requirement -- it only means the "whatever history
is in the cache" keeps growing, so the thing being reproduced at 100% is
never more than one cycle old.

    python -m fp.retrain                  # one cycle: fetch, refit, done
    python -m fp.retrain --loop 24        # refit every 24 hours, forever
    python -m fp.retrain --loop 24 --symbols BTCUSDT,ETHUSDT

run_full_bot.py can drive this itself with --retrain-hours; see there.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fp import data as D

HERE = Path(__file__).resolve().parent
KLINE_LIMIT = 1000          # Bybit's cap for one get_kline call


def _make_client():
    from pybit.unified_trading import HTTP
    return HTTP(testnet=False)


def fetch_since(client, symbol: str, since_ms: int, log=print):
    """Every closed 1-minute bar for `symbol` after `since_ms`.

    Paged forward in 1000-bar chunks -- Bybit's per-call cap -- using
    `start` as the walking cursor. Stops when a page comes back empty or
    short, which is the exchange saying there is nothing newer.
    """
    out = []
    cursor = since_ms
    now_ms = int(time.time() * 1000)
    while cursor < now_ms:
        try:
            r = client.get_kline(category="linear", symbol=symbol,
                                 interval="1", start=cursor,
                                 limit=KLINE_LIMIT)
        except Exception as exc:
            log(f"    {symbol}: fetch failed ({exc}), stopping here")
            break
        rows = r.get("result", {}).get("list", [])
        if not rows:
            break
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low",
                                         "close", "volume", "turnover"])
        df = df.astype(float)
        df["symbol"] = symbol
        out.append(df)
        seen_max = int(df["ts"].max())
        if seen_max <= cursor:
            break                     # no forward progress; stop
        cursor = seen_max + 60_000    # one minute past the newest seen
        if len(rows) < KLINE_LIMIT:
            break                     # exchange has nothing newer
    if not out:
        return None
    df = pd.concat(out, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
    # The most recent candle is still forming; only closed bars belong
    # in the cache the fit reads.
    cutoff = pd.Timestamp.now(tz="utc").floor("min") - pd.Timedelta(minutes=1)
    df = df[df["ts"] <= cutoff]
    return df.drop_duplicates(subset="ts").sort_values("ts")


def refresh_cache(client, symbols=None, log=print) -> int:
    """Pull whatever is newer than the cache and merge it in.

    Same merge rule the manually-uploaded weeks used all session:
    concat, drop exact (symbol, ts) duplicates keeping the newer read,
    sort, write back. Returns how many new rows landed.
    """
    old = pd.read_csv(D.CACHE)
    old["ts"] = pd.to_datetime(old["ts"], utc=True)
    want = symbols or sorted(old["symbol"].unique())
    news = []
    for sym in want:
        prior = old.loc[old["symbol"] == sym, "ts"]
        since = (int(prior.max().timestamp() * 1000) + 60_000
                if len(prior) else
                int(pd.Timestamp("2025-01-01", tz="utc").timestamp() * 1000))
        add = fetch_since(client, sym, since, log=log)
        n = 0 if add is None else len(add)
        log(f"    {sym:<13} +{n:,} bars")
        if add is not None and len(add):
            news.append(add[["symbol", "ts", "open", "high", "low",
                             "close", "volume", "turnover"]])
    if not news:
        log("  nothing new")
        return 0
    both = pd.concat([old] + news, ignore_index=True)
    before = len(both)
    both = both.drop_duplicates(subset=["symbol", "ts"], keep="last")
    both = both.sort_values(["symbol", "ts"]).reset_index(drop=True)
    both.to_csv(D.CACHE, index=False, compression="gzip")
    gained = len(both) - len(old)
    log(f"  cache: {len(old):,} -> {len(both):,} rows (+{gained:,}, "
        f"{before - len(both):,} duplicates dropped)")
    D._MEM.clear()          # force the next D.load() to re-read the file
    return gained


def retrain_all(symbols=None, log=print) -> int:
    """Refit every coin from a clean slate: python -m fp.full --fresh.

    A subprocess, not an in-process call. fp/full.py's own comments
    record three OOM kills from holding a stale rung's booster and the
    NEXT rung's working set in the same process at once; a subprocess
    that exits gets all of that back regardless, and a live bot process
    that has been up for days should not carry the fitting pipeline's
    memory history into its own.

    --fresh is not optional here: the resume logic keys a finished coin
    by symbol name alone, so without it a coin already in the report
    would be skipped even though refresh_cache just gave it new bars to
    reproduce.
    """
    cmd = [sys.executable, "-m", "fp.full", "--fresh"]
    if symbols:
        cmd += list(symbols)
    log(f"  running: {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(HERE.parent))
    return r.returncode


def cycle(symbols=None, log=print) -> bool:
    """One full refresh-and-refit pass. Returns True on a clean fit."""
    log("=" * 78)
    log(f"  RETRAIN CYCLE  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 78)
    client = _make_client()
    log("\n  refreshing cache from Bybit")
    refresh_cache(client, symbols, log=log)
    log("\n  refitting")
    rc = retrain_all(symbols, log=log)
    ok = rc == 0
    log(f"\n  cycle {'OK' if ok else f'FAILED (exit {rc})'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=None)
    ap.add_argument("--loop", type=float, default=0.0,
                    help="hours between cycles; 0 runs once and exits")
    a = ap.parse_args()
    symbols = ([s.strip().upper() for s in a.symbols.split(",") if s.strip()]
              if a.symbols else None)

    cycle(symbols)
    while a.loop > 0:
        print(f"\n  sleeping {a.loop:.1f}h until the next cycle", flush=True)
        time.sleep(a.loop * 3600)
        cycle(symbols)
    return 0


if __name__ == "__main__":
    sys.exit(main())
