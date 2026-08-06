"""Nine coins at once: which logics are specific, which group, which universal.

    python -m fp.coins
    python -m fp.coins --tf 5m,15m --min-coins 6

WHY THIS DATA IS WORTH MORE THAN ITS LENGTH SUGGESTS

Five and a half days is far too short to judge a logic on one symbol.
But it is nine symbols, and that changes what can be asked. On BTC alone
the only test available was "does it hold up in the second half", which
a search over millions of rules defeats every time. Across nine
independent series the test becomes "does the SAME rule, with the SAME
parameters, pay on coins it was never tuned to" -- and no amount of
searching one series manufactures that.

So there is no in-sample / out-of-sample split here. The consistency
across symbols IS the test.

WHAT THE DATA IS

    BLESSUSDT     +223.91% in 5.7 days   28.95%/day vol
    SOXLUSDT        +7.84%                8.53%/day
    SNDKUSDT        -0.80%                7.35%/day
    SKHYNIXUSDT     -5.08%                5.74%/day
    HYPEUSDT        +4.71%                3.06%/day
    ETHUSDT         +2.33%                1.91%/day
    SOLUSDT         +0.34%                1.89%/day
    XRPUSDT         -1.56%                1.63%/day
    XAUUSDT         +5.83%                1.01%/day

A useful spread: a coin that tripled, tokenised equities, tokenised
gold, and three large liquid crypto pairs. Volatility runs across a
factor of thirty, which is exactly what a rule with volatility-scaled
targets should be indifferent to -- and this is the first chance to
check whether it is.

WHAT CANNOT BE TESTED HERE, SAID FIRST

fp/btc_book.py's thirty rules live on 4h and daily bars. 8,190 one-minute
bars is 34 four-hour bars and 5 daily bars, and the factors need up to
200 bars of history. The book is not refuted by this data; it is
untouched by it, and it stays untested until longer histories arrive.

What runs here is the same machinery at the timeframes this length can
carry: 1m (8,190 bars), 5m (1,638) and 15m (546).

THE THREE TIERS

Each entry x exit x side rule is run on all nine coins with identical
parameters, and then counted:

    universal   profitable on at least --universal of the nine
    group       profitable on several but not most -- and the members
                are printed, because WHICH coins share a rule is the
                thing that defines a group
    specific    profitable on one or two only

A rule is credited to a coin only if it made at least --min-trades
trades there and its mean net per trade, after 0.055% each way and
funding, is positive.

AND THE NULL, because nine coins is still only nine

Coin counts are compared against the same rules with each coin's
position series rotated by a random offset. That keeps every coin's
drift, volatility and each rule's own long/short balance, and destroys
only the alignment -- so a rule that merely sat long while eight of nine
coins rose scores the same in the null as it does for real.

WHAT IT MEASURED

At 5m, 198,000 rules across nine coins, against the rotation null:

    coins paid   rules    share   null share   ratio
        >= 7         0    0.00%        0.00%    0.00
        >= 6        14    0.01%        0.01%    1.03
        >= 5       201    0.10%        0.08%    1.22
        >= 4      1660    0.84%        0.89%    0.94
        >= 2     34731   17.54%       18.07%    0.97

Every ratio is one. The number of rules paying on K coins is exactly
what randomly-timed positions produce, and nothing at all pays on seven
or more. At five minutes there is no cross-coin structure.

At 15m, 141,540 rules, the picture is different:

    coins paid   rules    share   null share   ratio
        >= 9         1    0.00%        0.00%     inf
        >= 8         7    0.00%        0.00%    19.38
        >= 7        33    0.02%        0.01%     2.89
        >= 6       196    0.14%        0.09%     1.62
        >= 5      1072    0.76%        0.54%     1.41
        >= 4      4202    2.97%        2.73%     1.09

The excess is concentrated exactly where it should be if it is real: at
the high-agreement end, rising from 1.09x at four coins to 19x at eight.
One rule paid on all nine.

AND THE CONTROL THAT MATTERS, because the week was up-biased

Six of the nine symbols rose, so a long-biased rule pays on many coins
for free. The control is a rule with no signal at all -- enter EVERY bar,
same exit:

    ALWAYS LONG   tp4.0/sl3.0/96b   pays on 4 of 9   +0.351%/trade
    ALWAYS LONG   tp4.0/sl3.0/24b   pays on 3 of 9   +0.060%
    ALWAYS LONG   tp3.0/sl3.0/96b   pays on 3 of 9   +0.244%
    ALWAYS SHORT  tp4.0/sl3.0/96b   pays on 0 of 9   -0.552%
    ALWAYS SHORT  tp4.0/sl3.0/24b   pays on 0 of 9   -0.348%

Long exposure alone reaches four coins. The 15m universal rules reach
seven, eight and nine. So they are not long exposure with extra steps --
whatever they are picking, it is picking better than the drift.

WHAT THE UNIVERSAL RULES LOOK LIKE

22 are long and 11 are short. The twelve with the highest mean per trade
happen to all be long, which is why an earlier note here said all 33
were -- reading the head of a sorted table and describing the table. The
short side exists and pays:

    ma_dist55|zfollow45_0.5   short  7/9 coins  +0.514%/trade  tp4.0/sl2.0
    ma_dist55|zfollow45_0.5   short  7/9 coins  +0.429%/trade  tp3.0/sl3.0
    pos34|break30             short  7/9 coins  +0.343%/trade  tp3.0/sl1.5
    pos34|zfollow45_1.5       short  7/9 coins  +0.328%/trade  tp4.0/sl1.5
    mom160|break30            short  7/9 coins  +0.289%/trade  tp4.0/sl2.0

That matters, because six of the nine symbols ROSE this week. A short
rule paying on seven of nine in an up week is harder to explain as drift
than a long one is -- the always-short control returned -0.552%/trade
and paid on zero of nine.

All 33 use wide targets (3-4 sigma) and long holds (96 bars = 24 hours)
and cluster on medium-lookback factors: mom89, mom160, ma_dist55,
ma_dist89, pos34, pos55, pos89, maxdd21, maxdd120. The coin sets repeat
-- BLESSUSDT, HYPEUSDT, SKHYNIXUSDT, SNDKUSDT, SOXLUSDT, XAUUSDT and
XRPUSDT again and again.

The caveat that remains is length, not direction: 5.7 days, one
timeframe of the two tested, and a single BLESSUSDT-scale event inside
the sample.

SO: the first cross-symbol signal in this project that is not explained
by one coin, by drift, or by chance -- on 5.7 days, at one timeframe out
of two, on the long side only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fp.ensemble import build_logics
from fp.exits import HOLDS, SL_MULTS, TP_MULTS, barrier_outcomes, sigma_at
from fp.horizon import FEE_ROUND_TRIP, FUNDING_PER_8H, resample

UPLOADS = Path("/root/.claude/uploads/2499e73f-5145-5c6f-b255-816732633901")
OUT = Path(__file__).resolve().parent / "coin_tiers.json"
TF = {"1m": 1, "5m": 5, "15m": 15, "30m": 30}


def load_coins(folder: Path = UPLOADS) -> dict[str, pd.DataFrame]:
    """Every 1m bar of every symbol in the uploaded json files."""
    rows = []
    for f in sorted(folder.glob("*bybit_1m_*syms_*.json")):
        rows += json.loads(f.read_text())
    d = pd.DataFrame(rows)
    d["datetime"] = pd.to_datetime(d["datetime"])
    out = {}
    for sym, g in d.groupby("symbol"):
        g = (g.sort_values("datetime").drop_duplicates("datetime")
             .set_index("datetime"))
        out[sym] = g[["open", "high", "low", "close", "volume"]].astype(float)
    return out


def stale_share(d: pd.DataFrame) -> float:
    """Fraction of bars that did not move at all.

    Tokenised equities do not trade around the clock, so their overnight
    bars repeat the last price. A run of identical closes is not low
    volatility, it is no data, and it would make any volatility-scaled
    target meaningless if left uncounted.
    """
    return float((d["high"].values == d["low"].values).mean())


def rules_for(d: pd.DataFrame, minutes: int, min_trades: int,
              rotate: np.random.Generator | None = None) -> dict:
    """Mean net per trade for every entry x exit x side on one coin."""
    close = d["close"].values.astype(float)
    high = d["high"].values.astype(float)
    low = d["low"].values.astype(float)
    sigma = sigma_at(close)
    fund = FUNDING_PER_8H * (minutes / 480.0)
    logics = build_logics(d, fast=True)

    entries: dict[tuple[str, int], np.ndarray] = {}
    for name, pos in logics.items():
        p = pos.values.astype(float)
        if rotate is not None:
            p = np.roll(p, int(rotate.integers(1, len(p))))
        prev = np.concatenate([[0.0], p[:-1]])
        for side in (1, -1):
            e = np.flatnonzero((p == side) & (prev != side))
            if len(e) >= min_trades:
                entries[(name, side)] = e

    out: dict[tuple, tuple[float, int]] = {}
    for hmax in HOLDS:
        for tp in TP_MULTS:
            for sl in SL_MULTS:
                for side in (1, -1):
                    o, held = barrier_outcomes(high, low, close, sigma, side,
                                               tp, sl, hmax)
                    net = o - FEE_ROUND_TRIP - held * fund
                    for (name, s2), e in entries.items():
                        if s2 != side:
                            continue
                        v = net[e]
                        v = v[np.isfinite(v)]
                        if len(v) < min_trades:
                            continue
                        out[(name, side, tp, sl, hmax)] = (float(v.mean()),
                                                           len(v))
    return out


def tally(per_coin: dict[str, dict]) -> pd.DataFrame:
    """How many coins each rule paid on, and which ones."""
    keys: dict[tuple, dict] = {}
    for sym, rules in per_coin.items():
        for k, (mean, n) in rules.items():
            r = keys.setdefault(k, {"coins": [], "means": [], "tested": 0})
            r["tested"] += 1
            if mean > 0:
                r["coins"].append(sym)
                r["means"].append(mean)
    rows = []
    for k, r in keys.items():
        name, side, tp, sl, hmax = k
        rows.append({"logic": name, "side": "long" if side > 0 else "short",
                     "sidenum": side, "tp": tp, "sl": sl, "hmax": hmax,
                     "n_pos": len(r["coins"]), "tested_on": r["tested"],
                     "coins": ",".join(sorted(r["coins"])),
                     "mean": float(np.mean(r["means"])) if r["means"] else 0.0})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tf", default="1m,5m,15m")
    ap.add_argument("--min-trades", type=int, default=12)
    ap.add_argument("--universal", type=int, default=7)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--null-runs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=31)
    a = ap.parse_args(argv)

    coins = load_coins()
    rng = np.random.default_rng(a.seed)
    print(f"{len(coins)} symbols, "
          f"{len(next(iter(coins.values()))):,} one-minute bars each")
    print(f"{'symbol':>14} {'move':>9} {'vol/day':>9} {'flat bars':>10}")
    for sym, d in sorted(coins.items()):
        r = d["close"].pct_change().dropna()
        print(f"{sym:>14} {100*(d.close.iloc[-1]/d.close.iloc[0]-1):>+8.2f}% "
              f"{100*r.std()*np.sqrt(1440):>8.2f}% {100*stale_share(d):>9.1f}%")
    print("\nNo in-sample split: agreement ACROSS coins is the test.\n")

    book = []
    for label in a.tf.split(","):
        label = label.strip()
        if label not in TF:
            continue
        minutes = TF[label]
        per_coin, per_coin_null = {}, {}
        for sym, d in coins.items():
            b = resample(d, minutes)
            if len(b) < 320:
                continue
            per_coin[sym] = rules_for(b, minutes, a.min_trades)
        if not per_coin:
            print(f"{label}: too few bars at this timeframe")
            continue
        T = tally(per_coin)
        if T.empty:
            print(f"{label}: no rule made {a.min_trades} trades anywhere")
            continue

        # the same thing with every coin's timing rotated
        null_counts = []
        for _ in range(a.null_runs):
            pc = {}
            for sym, d in coins.items():
                b = resample(d, minutes)
                if len(b) < 320:
                    continue
                pc[sym] = rules_for(b, minutes, a.min_trades, rotate=rng)
            N = tally(pc)
            if not N.empty:
                null_counts.append(N["n_pos"].values)
        nullv = np.concatenate(null_counts) if null_counts else np.array([0])

        n_coins = len(per_coin)
        print("=" * 78)
        print(f"{label}: {len(T):,} rules ran on {n_coins} coins")
        print("=" * 78)
        print(f"{'coins paid':>11} {'rules':>8} {'share':>8} "
              f"{'null share':>11} {'ratio':>7}")
        for k in range(n_coins, 0, -1):
            real = float((T["n_pos"] >= k).mean())
            null = float((nullv >= k).mean())
            ratio = real / null if null > 0 else float("inf")
            flag = ""
            if k >= a.universal:
                flag = "  <- universal"
            elif k >= a.group:
                flag = "  <- group"
            print(f"{'>= ' + str(k):>11} {int((T['n_pos'] >= k).sum()):>8} "
                  f"{100*real:>7.2f}% {100*null:>10.2f}% {ratio:>7.2f}{flag}")

        uni = T[T["n_pos"] >= a.universal].sort_values("mean", ascending=False)
        print(f"\n  UNIVERSAL: {len(uni)} rules paid on {a.universal}+ coins")
        for _, r in uni.head(12).iterrows():
            print(f"    {r.logic:<26} {r.side:>5} tp{r.tp}/sl{r.sl}/{r.hmax}b "
                  f"{r.n_pos}/{r.tested_on} coins {100*r['mean']:+.3f}%/trade")
            print(f"      {r.coins}")
        for _, r in uni.iterrows():
            book.append({"tf": label, "logic": r.logic, "side": r.side,
                         "tp": float(r.tp), "sl": float(r.sl),
                         "hmax": int(r.hmax), "coins": r.coins,
                         "n_coins": int(r.n_pos), "mean": float(r["mean"])})

        grp = T[(T["n_pos"] >= a.group) & (T["n_pos"] < a.universal)]
        print(f"\n  GROUP: {len(grp)} rules paid on {a.group}-{a.universal-1} "
              f"coins")
        if len(grp):
            fam = grp["coins"].value_counts().head(6)
            print("    the coin sets that share a rule most often:")
            for cset, cnt in fam.items():
                print(f"      {cnt:>5} rules  {cset}")
        print()

    OUT.write_text(json.dumps({
        "fitted_on": "9 symbols, 2026-07-31..2026-08-06, 1m bars",
        "note": "Selected by agreement across coins, not by a time split. "
                "5.7 days is short; treat coin count as the evidence and "
                "the per-trade means as indicative only.",
        "logics": book}, indent=2))
    print(f"{len(book)} universal rules written to {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
