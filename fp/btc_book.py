"""The book of logics that DID trade BTCUSDT profitably in 2025-2026.

    python -m fp.btc_book                  # build it, write btc_book.json
    python -m fp.btc_book --size 40

WHAT THIS IS, STATED PLAINLY BEFORE ANY NUMBER

This is fitted. It is the set of entry-exit rules that were profitable
after real fees on the 838,112 one-minute bars of 2025 and 2026 that
were supplied, selected by looking at those same bars. Nothing here is
out of sample and nothing here is a forecast.

That is deliberate and it is the right next step, because the validation
set is not a slice of this data -- it is the OTHER COINS. A logic fitted
to BTC and then found to work on ETH, SOL and the rest, on data it never
saw while being built, is evidence. A logic fitted to BTC and validated
on a held-out piece of BTC is much weaker, and this repo has already
shown why: 2.1 million combinations were tested on this one series and
not one survived an honest out-of-sample test, precisely because a
search that large manufactures winners inside any single series.

So the split is being moved to where it can actually bite. Fit here,
judge on the coins that arrive next.

WHAT A LOGIC HAS TO DO TO GET INTO THE BOOK

Being profitable overall is not enough -- a single good quarter can
carry a rule that was flat or negative the rest of the time. Four
requirements, all measured on the full sample:

    both years      positive mean net per trade in 2025 AND in 2026
                    separately, so nothing rides on one regime
    enough trades   at least 15 completed trades in each year
    real expectancy the Wilson lower bound on the win rate, priced
                    against the stop's DESIGNED size, still clears the
                    round trip -- this is what rejects tp0.5/sl3.0,
                    which wins 86% of the time by geometry and loses
                    5.9 wins on every stop
    positive tail   the worst single trade must not exceed the total
                    profit the rule made, so no rule survives on a
                    record that one bad trade would erase

DIVERSIFICATION, WHICH IS THE POINT OF A BOOK RATHER THAN A RULE

The requirement stated a long time ago was many small logics covering
different signals rather than one logic covering everything. So the book
caps how many entries may come from the same factor family, and the
selected rules are run as a portfolio at equal weight, each taking 1/N
of capital. The equity curve reported is the portfolio's, not the best
member's.

WHAT CAME OUT

7,381 entry x exit x side rules were profitable after fees in BOTH 2025
and 2026 separately. Thirty were selected, at most three per factor
family, ranked by their WEAKER year.

    rules              30      (18 on 4h bars, 12 on daily; 21 long, 9 short)
    trades          1,509
    total return    +70.1%
    max drawdown     -2.6%
    2025            +28.9%
    2026            +31.9%
    win days         57.0%   over 363 days with activity
    worst trade      -7.48%  (on that rule's own 1/30 slice)
    weakest rule    +0.579%/trade in 2025, +0.593%/trade in 2026

Every rule in the book was profitable in both years on its own, so the
portfolio result is not one member carrying the rest. The holds run from
12 hours to 12 days and the targets are 1.5 to 4 sigma, so the book is
not one trade repeated under different names.

HOW MUCH OF THIS TO BELIEVE

The honest number is: unknown, and deliberately so.

7,381 rules passed "profitable in both years" out of roughly 1.7 million
tested. That ratio is not far from what chance produces on a search that
size, which is exactly why the earlier out-of-sample work on this same
series found nothing. Selecting the best thirty of them by their weaker
year makes a curve that looks good and proves nothing about tomorrow.

What it IS good for is the next step. This book is a concrete, falsifiable
hypothesis: these thirty rules, these entries, these volatility-scaled
targets. When the other coins arrive, they get run unchanged -- same
rules, same parameters, no refitting -- on data this book has never seen.

    if they pay on ETH, SOL and the rest       the rules found something
    if they pay on some coins and not others   there is a group structure
                                               worth splitting on
    if they pay nowhere else                   this was a fit, and the
                                               2.1 million failed tests
                                               already said so

That is a real experiment with a real way to be wrong, which is more
than any curve fitted to one series can offer.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fp.ensemble import build_logics
from fp.exits import (HOLDS, SL_MULTS, TP_MULTS, barrier_outcomes, sigma_at,
                      wilson_lower)
from fp.horizon import (FEE_ROUND_TRIP, FUNDING_PER_8H, TIMEFRAMES, load_1m,
                        resample)

OUT = Path(__file__).resolve().parent / "btc_book.json"


def scan(d: pd.DataFrame, label: str, minutes: int, min_year: int
         ) -> pd.DataFrame:
    """Every entry x exit x side, scored on the full sample and per year."""
    close = d["close"].values.astype(float)
    high = d["high"].values.astype(float)
    low = d["low"].values.astype(float)
    sigma = sigma_at(close)
    year = d.index.year.values
    fund = FUNDING_PER_8H * (minutes / 480.0)

    logics = build_logics(d, fast=len(d) > 30000)
    entries: dict[tuple[str, int], np.ndarray] = {}
    for name, pos in logics.items():
        p = pos.values.astype(float)
        prev = np.concatenate([[0.0], p[:-1]])
        for side in (1, -1):
            e = np.flatnonzero((p == side) & (prev != side))
            if len(e) >= 2 * min_year:
                entries[(name, side)] = e
    print(f"  {label}: {len(d):,} bars, {len(logics):,} logics, "
          f"{len(entries):,} entry/side pairs")

    rows = []
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
                        v, y = net[e], year[e]
                        ok = np.isfinite(v)
                        v, y, e2 = v[ok], y[ok], e[ok]
                        if len(v) < 2 * min_year:
                            continue
                        y25, y26 = v[y == 2025], v[y == 2026]
                        if len(y25) < min_year or len(y26) < min_year:
                            continue
                        if not (y25.mean() > 0 and y26.mean() > 0):
                            continue
                        wins = v > 0
                        if not wins.any() or wins.all():
                            aw = float(v[wins].mean()) if wins.any() else 0.0
                        else:
                            aw = float(v[wins].mean())
                        # the stop's designed size, not whatever the sample
                        # happened to show -- a run with no stop-outs in it
                        # must not price the stop at zero
                        designed_loss = -(sl / max(tp, 1e-9)) * aw
                        pl = wilson_lower(int(wins.sum()), len(v))
                        expect = pl * aw + (1 - pl) * designed_loss
                        if not (expect > 0):
                            continue
                        total = float(v.sum())
                        if total <= 0 or -float(v.min()) >= total:
                            continue
                        rows.append({
                            "tf": label, "minutes": minutes, "logic": name,
                            "side": "long" if side > 0 else "short",
                            "sidenum": side, "tp": tp, "sl": sl, "hmax": hmax,
                            "n": int(len(v)), "mean": float(v.mean()),
                            "total": total, "win": float(wins.mean()),
                            "worst": float(v.min()),
                            "expect": float(expect),
                            "y2025": float(y25.mean()), "n2025": int(len(y25)),
                            "y2026": float(y26.mean()), "n2026": int(len(y26)),
                            "family": name.split("|")[0].rstrip("0123456789_"),
                            "hold": float(np.median(held[e2])) * minutes,
                        })
    return pd.DataFrame(rows)


def select(R: pd.DataFrame, size: int, per_family: int) -> pd.DataFrame:
    """Best by the WEAKER of its two years, capped per factor family.

    Ranking on the weaker year rather than the average is what stops a
    rule with one spectacular year and one mediocre one from crowding
    out a rule that worked in both.
    """
    R = R.copy()
    R["weaker"] = R[["y2025", "y2026"]].min(axis=1)
    R = R.sort_values("weaker", ascending=False)
    picked, seen_family, seen_logic = [], {}, set()
    for _, r in R.iterrows():
        if r.logic in seen_logic:            # one exit per entry logic
            continue
        if seen_family.get(r.family, 0) >= per_family:
            continue
        picked.append(r)
        seen_logic.add(r.logic)
        seen_family[r.family] = seen_family.get(r.family, 0) + 1
        if len(picked) >= size:
            break
    return pd.DataFrame(picked)


def portfolio(book: pd.DataFrame, d1m: pd.DataFrame) -> pd.Series:
    """Equal weight across the book: each rule takes 1/N of capital.

    A trade's return is booked on the day it closes, divided by the size
    of the book, so the curve is the portfolio's and not the best
    member's.
    """
    n = len(book)
    daily: dict[pd.Timestamp, float] = {}
    for tf, grp in book.groupby("tf"):
        minutes = int(grp["minutes"].iloc[0])
        d = resample(d1m, minutes)
        close = d["close"].values.astype(float)
        high = d["high"].values.astype(float)
        low = d["low"].values.astype(float)
        sigma = sigma_at(close)
        fund = FUNDING_PER_8H * (minutes / 480.0)
        logics = build_logics(d, fast=len(d) > 30000)
        for _, r in grp.iterrows():
            p = logics[r.logic].values.astype(float)
            prev = np.concatenate([[0.0], p[:-1]])
            e = np.flatnonzero((p == r.sidenum) & (prev != r.sidenum))
            o, held = barrier_outcomes(high, low, close, sigma, int(r.sidenum),
                                       float(r.tp), float(r.sl), int(r.hmax))
            net = o - FEE_ROUND_TRIP - held * fund
            for i in e:
                if not np.isfinite(net[i]):
                    continue
                exit_i = min(i + int(held[i]), len(d) - 1)
                day = d.index[exit_i].normalize()
                daily[day] = daily.get(day, 0.0) + float(net[i]) / n
    s = pd.Series(daily).sort_index()
    full = pd.date_range(s.index.min(), s.index.max(), freq="D")
    return s.reindex(full, fill_value=0.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tf", default="1h,4h,1d")
    ap.add_argument("--size", type=int, default=30)
    ap.add_argument("--per-family", type=int, default=3)
    ap.add_argument("--min-year", type=int, default=15)
    ap.add_argument("--max-bars", type=int, default=40000)
    a = ap.parse_args(argv)

    d1m = load_1m()
    print(f"{len(d1m):,} one-minute bars of BTCUSDT, "
          f"{d1m.index[0].date()} .. {d1m.index[-1].date()}")
    print("FITTED to this data. The other coins are the validation set.\n")

    frames = []
    for label in a.tf.split(","):
        label = label.strip()
        if label not in TIMEFRAMES:
            continue
        d = resample(d1m, TIMEFRAMES[label])
        if len(d) > a.max_bars:
            d = d.iloc[-a.max_bars:]
        frames.append(scan(d, label, TIMEFRAMES[label], a.min_year))
    R = pd.concat([f for f in frames if not f.empty], ignore_index=True) \
        if any(not f.empty for f in frames) else pd.DataFrame()
    if R.empty:
        print("\nNothing was profitable in BOTH years after fees.")
        return 0

    print(f"\n{len(R):,} entry x exit x side rules were profitable after fees "
          f"in BOTH 2025 and 2026")
    book = select(R, a.size, a.per_family)
    print(f"selected {len(book)} for the book, at most {a.per_family} per "
          f"factor family\n")

    print(f"{'tf':>4} {'entry':<30} {'side':>5} {'exit':>16} {'n':>5} "
          f"{'win':>6} {'mean':>8} {'2025':>8} {'2026':>8} {'hold':>8}")
    for _, r in book.iterrows():
        h = r.hold
        hs = f"{h:.0f}m" if h < 120 else f"{h/60:.1f}h" if h < 2880 else f"{h/1440:.1f}d"
        print(f"{r.tf:>4} {r.logic:<30} {r.side:>5} "
              f"{f'tp{r.tp}/sl{r.sl}/{r.hmax}b':>16} {r.n:>5} "
              f"{100*r.win:>5.1f}% {100*r['mean']:>7.3f}% {100*r.y2025:>7.3f}% "
              f"{100*r.y2026:>7.3f}% {hs:>8}")

    curve = portfolio(book, d1m)
    eq = (1 + curve).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    trades = int(book["n"].sum())
    print("\n" + "=" * 74)
    print("THE BOOK AS A PORTFOLIO, equal weight, 1/N of capital per rule")
    print("=" * 74)
    print(f"  rules              : {len(book)}")
    print(f"  trades             : {trades:,}")
    print(f"  total return       : {100*(eq.iloc[-1]-1):+.1f}%")
    print(f"  max drawdown       : {100*dd:+.1f}%")
    print(f"  days with activity : {int((curve != 0).sum()):,}")
    act = curve[curve != 0]
    print(f"  win days           : {100*(act > 0).mean():.1f}%")
    for y, g in curve.groupby(curve.index.year):
        e = (1 + g).cumprod()
        print(f"  {y}               : {100*(e.iloc[-1]-1):+.1f}%")
    print(f"\n  worst single trade : {100*book['worst'].min():.2f}%")
    print(f"  weakest rule, 2025 : {100*book['y2025'].min():+.3f}%/trade")
    print(f"  weakest rule, 2026 : {100*book['y2026'].min():+.3f}%/trade")

    OUT.write_text(json.dumps({
        "fitted_on": f"BTCUSDT {d1m.index[0].date()}..{d1m.index[-1].date()}",
        "in_sample": True,
        "note": "Fitted to this data by construction. Validation is the "
                "other coins, on data this book never saw.",
        "logics": [{"tf": r.tf, "name": r.logic, "side": r.side,
                    "tp": float(r.tp), "sl": float(r.sl), "hmax": int(r.hmax),
                    "hold_min": float(r.hold), "n": int(r.n),
                    "win": float(r.win), "mean": float(r["mean"]),
                    "y2025": float(r.y2025), "y2026": float(r.y2026)}
                   for _, r in book.iterrows()],
    }, indent=2))
    print(f"\n  written to {OUT.name}")
    print("=" * 74)
    print("  This is what worked on 2025-2026 BTCUSDT. Whether it is a")
    print("  strategy or a very good fit is decided by the next coin, not")
    print("  by anything in this file.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
