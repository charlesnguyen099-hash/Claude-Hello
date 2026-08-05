"""Hard-coded pattern library: N candles before an entry -> which way to trade.

    python -m fp.patterns learn  --data bybit_bot/data/BTCUSDT_202608.csv
    python -m fp.patterns review --data <tomorrow's file>     # score and prune
    python -m fp.patterns show                                # what is in there

THE IDEA, AS ASKED FOR

Look at the candles, find where a trade would actually have made money
after fees, take a snapshot of the 10-20 candles leading into it, and
write that down as a rule: "when the last N candles look like THIS, trade
THAT way". No model, no fitted coefficients -- a lookup table.

Then, and this is the part that makes it honest, every new day of data
scores the table. A pattern that keeps working stays; a pattern that
stops working is dropped. The library only earns trust by surviving days
it was not built from.

WHY THE TABLE IS BUILT WITH HINDSIGHT AND WHY THAT IS NOT CHEATING HERE

Labelling uses the future: for bar i we simulate both directions to the
exit and keep whichever actually won. That is legitimate for BUILDING a
lookup key, because the key itself is computed only from bars up to and
including i. Nothing about the future enters the signature. What hindsight
buys is the answer column; what the bot sees live is the question column,
which is past-only.

The catch, stated plainly: a table built this way ALWAYS looks perfect on
the data it was built from -- every row was chosen because it won. The
in-sample numbers this prints are therefore meaningless as a forecast, and
they are labelled in-sample everywhere they appear. Only the `live` column,
filled in by `review` on data the pattern has never seen, means anything.
A pattern with n_live = 0 has proved nothing yet.

WHY 15-MINUTE BARS

Fees are a fixed fraction of notional; the target is a multiple of ATR.
On August's data the round-trip fee is 41% of a 3-ATR move at 1m, 14% at
5m, 7% at 15m and 5% at 30m. Below 15m the fee eats the move before any
pattern gets a say -- 1m needs a 60.6% win rate to break even at maker
fees and is impossible at taker. 15m needs 38.0% maker, 46.1% taker, and
still yields ~96 bars a day to learn from.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fp import logic as L

LIBRARY_PATH = Path(__file__).resolve().parent / "pattern_library.json"

BAR_MINUTES = 15
LOOKBACK = 12                   # candles in the signature (~3 hours at 15m)
MIN_SUPPORT = 5                 # occurrences before a pattern may be traded
MIN_WIN_RATE = 0.55             # of those occurrences, in-sample
PRUNE_AFTER = 8                 # live trades before a losing pattern is dropped
PRUNE_WIN_RATE = 0.40           # live win rate below which it is dropped
# Share of a shape's winning labels that must sit on one side before that
# side is traded. See Library.get() -- most shapes are labelled both ways.
MAJORITY_THRESHOLD = 0.70


# --------------------------------------------------------------- signature

def _bucket(x: float, edges: tuple[float, ...]) -> int:
    """Quantise to a small integer. Keeps the key discrete so the table is
    a real lookup rather than a nearest-neighbour model in disguise."""
    return int(np.searchsorted(edges, x))


def signature(close: np.ndarray, high: np.ndarray, low: np.ndarray,
              vol: np.ndarray, i: int, atr: float,
              lookback: int = LOOKBACK) -> str | None:
    """A discrete description of the `lookback` candles ending at bar i.

    Everything is normalised by ATR or expressed as a ratio, so the same
    key means the same shape on BTC at $60,000 and on a $0.04 altcoin.
    Uses bars i-lookback+1 .. i inclusive -- no future information.
    """
    if i < lookback or atr <= 0 or not np.isfinite(atr):
        return None
    a, b = i - lookback + 1, i + 1
    c = close[a:b]
    v = vol[a:b]
    if len(c) < lookback or c[0] <= 0:
        return None

    # 1. Net travel over the window, in ATR units. The trend.
    net = _bucket((c[-1] - c[0]) / atr, (-3.0, -1.2, -0.4, 0.4, 1.2, 3.0))
    # 2. How one-sided the candles were.
    ups = float(np.mean(np.diff(c) > 0))
    share = _bucket(ups, (0.35, 0.5, 0.65))
    # 3. Volume rhythm: second half against first half.
    half = lookback // 2
    v0, v1 = v[:half].mean(), v[half:].mean()
    vslope = _bucket(v1 / v0 if v0 > 0 else 1.0, (0.8, 1.25))
    # 4. Where the last close sits in the window's range.
    hi, lo = high[a:b].max(), low[a:b].min()
    pos = _bucket((c[-1] - lo) / (hi - lo) if hi > lo else 0.5, (0.25, 0.5, 0.75))
    # 5. Longest run of same-direction candles -- exhaustion.
    d = np.sign(np.diff(c))
    run, best = 1, 1
    for k in range(1, len(d)):
        run = run + 1 if d[k] == d[k - 1] and d[k] != 0 else 1
        best = max(best, run)
    streak = _bucket(best, (2.5, 4.5))
    # 6. Is the window volatile relative to its own recent self?
    rng = _bucket((hi - lo) / atr, (2.0, 4.0, 7.0))

    return f"{net}{share}{vslope}{pos}{streak}{rng}"


# ----------------------------------------------------------------- library

@dataclass
class Pattern:
    key: str
    direction: int
    bar_minutes: int = BAR_MINUTES
    lookback: int = LOOKBACK
    # In-sample, from the days this was learned on. Always flattering.
    n_train: int = 0
    train_wins: int = 0
    train_net: float = 0.0
    # Out-of-sample, filled in by review() on days it has never seen.
    n_live: int = 0
    live_wins: int = 0
    live_net: float = 0.0
    sources: list[str] = field(default_factory=list)
    created: str = ""
    last_seen: str = ""

    @property
    def train_win_rate(self) -> float:
        return self.train_wins / self.n_train if self.n_train else 0.0

    @property
    def live_win_rate(self) -> float:
        return self.live_wins / self.n_live if self.n_live else 0.0

    def tradeable(self) -> bool:
        """Enough in-sample support to be worth trying, and not yet
        disproved out of sample."""
        if self.n_train < MIN_SUPPORT or self.train_win_rate < MIN_WIN_RATE:
            return False
        if self.n_live >= PRUNE_AFTER and self.live_win_rate < PRUNE_WIN_RATE:
            return False
        return True

    def confidence(self) -> float:
        """0..1, and it leans on live evidence once there is any.

        With no live record this returns at most 0.5, so an unproven
        pattern can never take full leverage no matter how good it looks
        on the data it was built from.
        """
        if self.n_live == 0:
            return min(0.5, self.train_win_rate * 0.5)
        w = min(1.0, self.n_live / 20.0)
        return float((1 - w) * self.train_win_rate * 0.5 + w * self.live_win_rate)


class Library:
    def __init__(self, path: Path = LIBRARY_PATH):
        self.path = path
        self.patterns: dict[str, Pattern] = {}
        self.load()

    def _id(self, key: str, direction: int) -> str:
        return f"{key}:{'L' if direction > 0 else 'S'}"

    def load(self) -> None:
        if not self.path.exists():
            return
        raw = json.loads(self.path.read_text())
        self.patterns = {k: Pattern(**v) for k, v in raw.get("patterns", {}).items()}

    def save(self) -> None:
        self.path.write_text(json.dumps(
            {"bar_minutes": BAR_MINUTES, "lookback": LOOKBACK,
             "saved": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "patterns": {k: asdict(p) for k, p in self.patterns.items()}},
            indent=1))

    def get(self, key: str, majority: float = MAJORITY_THRESHOLD) -> Pattern | None:
        """The rule for this shape, if the table has a usable one.

        Most shapes carry BOTH labels -- the same twelve candles were
        followed by a winning long on some days and a winning short on
        others. Measured on 2025+2026: 71.9% of shapes, and 98.5% of all
        observations, are ambiguous like that, with the majority side
        holding only 55.8%.

        So a shape is only traded when one side owns at least `majority`
        of the winning labels. At 0.5 that is barely a preference; at 1.0
        it demands the shape never once paid the other way.
        """
        long_p = self.patterns.get(self._id(key, 1))
        short_p = self.patterns.get(self._id(key, -1))
        nl = long_p.n_train if long_p else 0
        ns = short_p.n_train if short_p else 0
        if nl + ns == 0:
            return None
        p = long_p if nl >= ns else short_p
        if p is None or not p.tradeable():
            return None
        if max(nl, ns) / (nl + ns) < majority:
            return None
        return p

    def observe(self, key: str, direction: int, won: bool, net: float,
                source: str, live: bool) -> Pattern:
        pid = self._id(key, direction)
        p = self.patterns.get(pid)
        if p is None:
            p = Pattern(key=key, direction=direction,
                        created=datetime.now(timezone.utc).isoformat(timespec="seconds"))
            self.patterns[pid] = p
        if live:
            p.n_live += 1
            p.live_wins += int(won)
            p.live_net += net
        else:
            p.n_train += 1
            p.train_wins += int(won)
            p.train_net += net
        if source not in p.sources:
            p.sources.append(source)
        p.last_seen = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return p

    def prune(self) -> list[str]:
        """Drop what the live record has disproved."""
        dead = [pid for pid, p in self.patterns.items()
                if p.n_live >= PRUNE_AFTER and p.live_win_rate < PRUNE_WIN_RATE]
        for pid in dead:
            del self.patterns[pid]
        return dead


# ------------------------------------------------------------------ engine

def read_bars(path: str | Path, minutes: int = BAR_MINUTES) -> pd.DataFrame:
    d = pd.read_csv(path, sep=None, engine="python")
    d.columns = [c.strip().lower() for c in d.columns]
    dt = next(c for c in d.columns if "time" in c or "date" in c)
    d["datetime"] = pd.to_datetime(d[dt])
    return L.to_bars(d[["datetime", "open", "high", "low", "close", "volume"]],
                     minutes)


def atr_series(bars: pd.DataFrame, n: int = 14) -> np.ndarray:
    h, l, c = bars["high"].values, bars["low"].values, bars["close"].values
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().values


def scan(bars: pd.DataFrame, exit_name: str, fee: float,
         lookback: int = LOOKBACK):
    """Yield (i, key, best_direction, net, won) for every usable bar.

    The direction is chosen with hindsight -- whichever side actually paid
    after fees. The key is past-only.
    """
    h, l, c = bars["high"].values, bars["low"].values, bars["close"].values
    v = bars["volume"].values
    atr = atr_series(bars)
    for i in range(max(lookback, 20), len(c) - 1):
        if atr[i] <= 0 or not np.isfinite(atr[i]):
            continue
        key = signature(c, h, l, v, i, atr[i], lookback)
        if key is None:
            continue
        rl, _, _ = L.simulate_exit(h, l, c, i, 1, atr[i], exit_name, fee)
        rs, _, _ = L.simulate_exit(h, l, c, i, -1, atr[i], exit_name, fee)
        d, net = (1, rl) if rl >= rs else (-1, rs)
        yield i, key, d, net, net > 0


def learn(lib: Library, bars: pd.DataFrame, source: str,
          exit_name: str, fee: float) -> dict:
    """Add every profitable-after-fees entry to the table."""
    seen = kept = 0
    for _, key, d, net, won in scan(bars, exit_name, fee):
        seen += 1
        if not won:
            continue                  # only winners become rules
        lib.observe(key, d, True, net, source, live=False)
        kept += 1
    return {"bars_scanned": seen, "rules_recorded": kept,
            "distinct_keys": len({p.key for p in lib.patterns.values()}),
            "entries": len(lib.patterns)}


def review(lib: Library, bars: pd.DataFrame, source: str,
           exit_name: str, fee: float,
           majority: float = MAJORITY_THRESHOLD, record: bool = True) -> dict:
    """Score the EXISTING table on data it has not seen. This is the part
    that decides what survives."""
    hits = wins = 0
    net_total = 0.0
    h, l, c = bars["high"].values, bars["low"].values, bars["close"].values
    v = bars["volume"].values
    atr = atr_series(bars)
    for i in range(max(LOOKBACK, 20), len(c) - 1):
        if atr[i] <= 0 or not np.isfinite(atr[i]):
            continue
        key = signature(c, h, l, v, i, atr[i], LOOKBACK)
        if key is None:
            continue
        p = lib.get(key, majority)
        if p is None:
            continue                  # table has no usable rule for this shape
        r, _, _ = L.simulate_exit(h, l, c, i, p.direction, atr[i], exit_name, fee)
        if record:
            lib.observe(key, p.direction, r > 0, r, source, live=True)
        hits += 1
        wins += r > 0
        net_total += r
    dead = lib.prune() if record else []
    return {"signals": hits, "wins": wins,
            "win_rate": wins / hits if hits else 0.0,
            "net_pct": 100 * net_total,
            "net_per_trade_pct": 100 * net_total / hits if hits else 0.0,
            "pruned": dead}


# --------------------------------------------------------------------- CLI

def _fee(name: str) -> float:
    return {"maker": L.MAKER_ROUND_TRIP, "taker": L.TAKER_ROUND_TRIP,
            "taker-slip": L.TAKER_WITH_SLIPPAGE}[name]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["learn", "review", "show", "clear"])
    p.add_argument("--data", action="append", default=[],
                   help="OHLCV file; repeatable")
    p.add_argument("--exit", default=L.DEFAULT_EXIT, choices=L.EXIT_STRATEGIES)
    p.add_argument("--fee", default="taker", choices=["maker", "taker", "taker-slip"])
    p.add_argument("--bar-minutes", type=int, default=BAR_MINUTES)
    p.add_argument("--library", default=str(LIBRARY_PATH))
    p.add_argument("--majority", type=float, default=MAJORITY_THRESHOLD,
                   help="share of a shape's winning labels that must sit on "
                        f"one side before it is traded (default {MAJORITY_THRESHOLD})")
    p.add_argument("--dry-run", action="store_true",
                   help="review without writing the result into the library")
    a = p.parse_args(argv)

    lib = Library(Path(a.library))
    fee = _fee(a.fee)

    if a.command == "clear":
        lib.patterns.clear()
        lib.save()
        print("library cleared")
        return 0

    if a.command == "show":
        if not lib.patterns:
            print("library is empty -- run learn first")
            return 0
        rows = sorted(lib.patterns.values(),
                      key=lambda x: (-x.n_live, -x.n_train))
        print(f"{len(rows)} entries, {sum(1 for r in rows if r.tradeable())} tradeable\n")
        print(f"{'key':>8} {'dir':>5} {'n_tr':>5} {'tr win':>7} {'n_live':>7} "
              f"{'live win':>9} {'live net%':>10} {'conf':>5} {'trade?':>7}")
        for r in rows[:60]:
            print(f"{r.key:>8} {'LONG' if r.direction > 0 else 'SHORT':>5} "
                  f"{r.n_train:>5} {100*r.train_win_rate:>6.0f}% {r.n_live:>7} "
                  f"{100*r.live_win_rate:>8.0f}% {100*r.live_net:>9.3f} "
                  f"{r.confidence():>5.2f} {'yes' if r.tradeable() else '':>7}")
        proven = [r for r in rows if r.n_live > 0]
        print(f"\n  {len(proven)} entries have any out-of-sample record at all.")
        if proven:
            tot = sum(r.n_live for r in proven)
            w = sum(r.live_wins for r in proven)
            net = sum(r.live_net for r in proven)
            print(f"  Out-of-sample overall: {tot} trades, {100*w/tot:.1f}% win, "
                  f"{100*net/tot:+.4f}%/trade, {100*net:+.2f}% total")
        else:
            print("  NOTHING here has been tested yet. In-sample numbers are")
            print("  guaranteed flattering -- every rule was recorded because")
            print("  it won. Run `review` on a day this was not built from.")
        return 0

    if not a.data:
        print("need --data", file=sys.stderr)
        return 1

    for path in a.data:
        bars = read_bars(path, a.bar_minutes)
        src = Path(path).name
        if a.command == "learn":
            r = learn(lib, bars, src, a.exit, fee)
            print(f"learn {src}: {len(bars):,} bars -> "
                  f"{r['rules_recorded']:,} winning entries recorded "
                  f"of {r['bars_scanned']:,} scanned; "
                  f"library now {r['entries']} entries "
                  f"over {r['distinct_keys']} distinct shapes")
        else:
            r = review(lib, bars, src, a.exit, fee, a.majority, not a.dry_run)
            print(f"review {src}: {len(bars):,} bars -> {r['signals']} signals, "
                  f"{100*r['win_rate']:.1f}% win, "
                  f"{r['net_per_trade_pct']:+.4f}%/trade, "
                  f"{r['net_pct']:+.2f}% total")
            if r["pruned"]:
                print(f"  pruned {len(r['pruned'])} disproved patterns: "
                      f"{', '.join(r['pruned'][:8])}")
    lib.save()
    print(f"saved {len(lib.patterns)} entries to {lib.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
