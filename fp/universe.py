"""Every USDT perpetual Bybit lists, ranked by what actually trades.

The logic was built on ten coins. The board is over six hundred. Nothing
in the features is coin-specific -- all 200 columns are scale-free, so a
model fitted on one coin produces a number on any other -- which is what
makes scanning the whole venue possible at all.

TWO SPEEDS, because six hundred coins cannot each be re-read every few
seconds without hitting the rate limit and without every scan arriving
late. The top of the book by 24-hour turnover is scanned EVERY cycle;
the long tail is swept in rotating slices, so every coin is still seen,
just less often. Liquidity is the right ranking for this: a coin nobody
trades cannot be entered or exited at the price the logic assumed.

Pagination matters here. get_instruments_info returns a page at a time
and the default page is smaller than the board, so a single call
silently returns a partial universe -- which looks like a working
scanner that has never heard of most of the market.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "data" / "instruments.json"


def all_perpetuals(client, quote: str = "USDT") -> dict:
    """Every tradable linear perpetual, with its real leverage ceiling.

    Returns {symbol: {"max_leverage": float, "min_leverage": float}}.
    Paginated to the end -- a partial page is a partial market.
    """
    out, cursor = {}, None
    while True:
        kw = dict(category="linear", limit=1000)
        if cursor:
            kw["cursor"] = cursor
        try:
            r = client.get_instruments_info(**kw)
        except Exception:
            break
        res = r.get("result", {}) or {}
        rows = res.get("list", []) or []
        for x in rows:
            s = x.get("symbol", "")
            if not s.endswith(quote):
                continue
            if x.get("status") != "Trading":
                continue
            if x.get("contractType") != "LinearPerpetual":
                continue
            lf = x.get("leverageFilter", {}) or {}
            out[s] = {
                "max_leverage": float(lf.get("maxLeverage", 1) or 1),
                "min_leverage": float(lf.get("minLeverage", 1) or 1),
            }
        cursor = res.get("nextPageCursor")
        if not cursor or not rows:
            break
    if out:
        try:
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            CACHE.write_text(json.dumps(out, indent=1, sort_keys=True))
        except OSError:
            pass
    return out


def turnover(client, symbols=None) -> dict:
    """24-hour turnover per symbol, in one call for the whole board."""
    out = {}
    try:
        rows = client.get_tickers(category="linear")["result"]["list"]
    except Exception:
        return out
    want = set(symbols) if symbols else None
    for x in rows:
        s = x.get("symbol")
        if want is not None and s not in want:
            continue
        try:
            out[s] = float(x.get("turnover24h") or 0.0)
        except (TypeError, ValueError):
            out[s] = 0.0
    return out


def ranked(client, quote: str = "USDT"):
    """(symbols by turnover descending, leverage caps) for the venue."""
    caps = all_perpetuals(client, quote)
    tvr = turnover(client, list(caps))
    syms = sorted(caps, key=lambda s: tvr.get(s, 0.0), reverse=True)
    return syms, caps, tvr


class Rotation:
    """Top N every cycle; everything else in rotating slices.

    The point is that no coin is ever dropped -- the tail is scanned
    less often, not never. `slice_size` sets how much of the tail each
    cycle takes, so the whole board is covered every
    ceil(len(tail) / slice_size) cycles.
    """

    def __init__(self, symbols, top: int = 50, slice_size: int = 50):
        self.top = list(symbols[:top])
        self.tail = list(symbols[top:])
        self.slice_size = max(1, slice_size)
        self.at = 0

    def __len__(self) -> int:
        return len(self.top) + len(self.tail)

    @property
    def cycles_for_full_sweep(self) -> int:
        if not self.tail:
            return 1
        return -(-len(self.tail) // self.slice_size)

    def next_batch(self) -> list:
        if not self.tail:
            return list(self.top)
        end = self.at + self.slice_size
        chunk = self.tail[self.at:end]
        if end >= len(self.tail):
            self.at = 0
            chunk += self.tail[:max(0, end - len(self.tail))]
        else:
            self.at = end
        # dict.fromkeys keeps order and drops the duplicate a wrapped
        # slice can produce when the tail is shorter than slice_size.
        return list(dict.fromkeys(self.top + chunk))
