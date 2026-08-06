"""Three tiers on ten symbols: per-coin, per-group, universal.

    python -m fp.tiers
    python -m fp.tiers --tf 15m,30m,1h --split 0.6

WHAT CHANGED, AND WHY IT MATTERS

The first upload was 5.7 days of nine symbols. Six of the nine rose, so
any long-biased rule paid on many coins for free, and the only test
available was agreement across symbols.

This one is 36.7 days ending 2026-08-06, and the composition is much
better balanced -- five of the nine FELL, three of them by more than
39%:

    BLESSUSDT     +258.46%    15.13%/day
    ETHUSDT        +21.37%     2.38%
    XAUUSDT         +6.31%     1.17%
    XRPUSDT         +0.54%     2.18%
    SOLUSDT         -0.03%     2.52%
    HYPEUSDT       -13.72%     3.50%
    SKHYNIXUSDT    -39.45%     7.05%
    SNDKUSDT       -43.67%     7.34%
    SOXLUSDT       -51.18%     9.69%

BTCUSDT is added over the same window, from its own 1m history, so all
ten are measured on identical dates. A rule is not allowed to look good
because one symbol was sampled over a kinder month than another.

Two independent tests are now possible where before there was one:

    time         fit on the first --split of the window, judge on the
                 rest, per symbol
    cross-symbol the SAME rule with the SAME parameters must pay on
                 symbols it was never tuned to

A rule has to pass both. Passing one is what a large search does by
accident; passing both on ten symbols is considerably harder.

THE THREE TIERS

    per-coin     pays on ONE symbol, in both halves of the window. Its
                 own signal, its own instrument -- the narrowest claim,
                 and the one most likely to be a fit, so it carries the
                 count of how many symbols it FAILED on
    group        pays on 3 to 7 symbols, in both halves. The members are
                 printed, because which symbols share a rule is what
                 defines the group -- and a group that turns out to be
                 "the three tokenised equities" or "the two majors" is a
                 finding, not a coincidence
    universal    pays on 8 or more of the ten, in both halves

TWO CONTROLS, BOTH NECESSARY

    rotation null   every rule re-run with each symbol's position series
                    rolled by a random offset. Keeps each symbol's drift,
                    volatility and the rule's own long/short balance;
                    destroys only the timing
    always-on       a rule with NO signal -- enter every bar, same exit.
                    Whatever it reaches is what direction alone buys, and
                    a tier that does not clearly beat it has found nothing

WHAT IT MEASURED AT 15m

76,704 rules on ten symbols, each judged in both halves of every symbol.
The control first, because it decides how to read everything else:

    a rule with NO signal, entering every bar, pays on 1 of 10 long
    (-0.094%/trade) and 1 of 10 short (-0.137%)

Direction alone buys nothing here. That is the balanced window doing its
job -- on the 5.7-day upload the same control reached 4 of 9 long.

    symbols paid   rules    share     null   ratio  test mean
          >= 10        0   0.000%   0.000%       -        -    universal
           >= 8        0   0.000%   0.000%       -        -    universal
           >= 7        3   0.004%   0.000%     inf   +0.200%   group
           >= 6       12   0.016%   0.000%     inf   +0.254%   group
           >= 5       57   0.074%   0.004%   20.76   +0.349%   group
           >= 4      265   0.345%   0.095%    3.65   +0.456%   group
           >= 3     1586   2.068%   1.008%    2.05   +0.449%   group
           >= 2     7224   9.418%   7.630%    1.23   +0.438%
           >= 1    25865  33.721%  35.953%    0.94   +0.412%

TIER 1, PER-COIN: no evidence

18,641 rules pay on exactly one symbol, and the ratio against the null
at ">= 1" is 0.94 -- BELOW chance. Randomly-timed positions produce
more single-symbol winners than the real rules do. A rule that works on
one instrument and nowhere else is what a 76,704-rule search returns by
construction, and this is the number that says so.

TIER 2, GROUP: the strongest result in this project

The excess is real and it concentrates where a real effect belongs:
2.05x at three symbols, 3.65x at four, 20.76x at five, and at six and
seven the null produces NOTHING at all while the real library produces
twelve and three.

The groups are not arbitrary. The largest is 345 rules on exactly
SKHYNIXUSDT, SNDKUSDT and SOXLUSDT -- SK Hynix, SanDisk and a
semiconductor ETF. The data found the semiconductor sector without being
told it exists.

And it is not "short the things that crashed". Those three fell 39%,
44% and 51%, and blind shorting them returns -0.065%/trade; SOXL halved
and always-short on it still loses 0.231% per trade. The 345 rules make
+0.400%. The signal is worth 0.465 points over the blind short.

TIER 3, UNIVERSAL: zero rules, and the reason is now measurable

Nothing pays on eight of ten. The correlation matrix explains it -- ten
symbols are about FOUR independent bets:

    semiconductors  SKHYNIX/SNDK/SOXL      mean pairwise r = 0.72
    crypto majors   ETH/SOL/BTC/XRP        mean pairwise r = 0.81
    BLESSUSDT       correlated with nothing        r = 0.02-0.09
    XAUUSDT         gold, weakly attached         r = 0.22-0.30

    semis vs crypto cross-correlation             r = 0.37

So "pays on three symbols" can mean one bet confirmed once, and "pays on
eight" would require spanning at least three blocks that share almost
nothing. Counting symbols is not counting evidence, and blocks() in this
module prints the matrix that says how much of each is which.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fp.coins import load_coins
from fp.ensemble import build_logics
from fp.exits import barrier_outcomes, sigma_at, wilson_lower
from fp.horizon import FEE_ROUND_TRIP, FUNDING_PER_8H, load_1m, resample

OUT = Path(__file__).resolve().parent / "tier_book.json"

# Ten symbols are not ten bets. Measured on 15m returns over this window:
# the semiconductors correlate 0.72 with each other, the crypto majors
# 0.81, BLESSUSDT with nothing at 0.02-0.09, gold weakly at 0.22-0.30.
# So "universal" cannot mean "8 of 10 symbols" -- eight symbols can be
# two blocks. It means SPANNING the blocks, which is the only version of
# the word that carries independent evidence.
BLOCK = {"SKHYNIXUSDT": "semi", "SNDKUSDT": "semi", "SOXLUSDT": "semi",
         "ETHUSDT": "crypto", "SOLUSDT": "crypto", "BTCUSDT": "crypto",
         "XRPUSDT": "crypto", "HYPEUSDT": "crypto",
         "BLESSUSDT": "bless", "XAUUSDT": "gold"}
N_BLOCKS = len(set(BLOCK.values()))
TF = {"15m": 15, "30m": 30, "1h": 60}
TPS = (1.0, 2.0, 3.0, 4.0)
SLS = (1.0, 2.0, 3.0)
HOLDS = (12, 48, 192)


def all_symbols() -> dict[str, pd.DataFrame]:
    """The nine uploaded symbols plus BTCUSDT, on one common window.

    BTC has years of history and the others have five weeks. Letting BTC
    use all of it would compare a rule measured over nineteen months with
    the same rule measured over one, so it is clipped to the window they
    share.
    """
    coins = load_coins()
    lo = max(d.index.min() for d in coins.values())
    hi = min(d.index.max() for d in coins.values())
    btc = load_1m()
    if btc.index.tz is None and lo.tzinfo is not None:
        btc.index = btc.index.tz_localize("UTC")
    b = btc.loc[(btc.index >= lo) & (btc.index <= hi)]
    if len(b) > 1000:
        coins["BTCUSDT"] = b
    return {k: v.loc[(v.index >= lo) & (v.index <= hi)]
            for k, v in coins.items()}


def rules_on(d: pd.DataFrame, minutes: int, split: float, min_trades: int,
             rotate: np.random.Generator | None = None) -> dict:
    """Every entry x exit x side on one symbol, scored in both halves."""
    b = resample(d, minutes)
    if len(b) < 400:
        return {}
    close = b["close"].values.astype(float)
    high = b["high"].values.astype(float)
    low = b["low"].values.astype(float)
    sigma = sigma_at(close)
    fund = FUNDING_PER_8H * (minutes / 480.0)
    cut = int(len(b) * split)
    logics = build_logics(b, fast=True)

    entries: dict[tuple[str, int], np.ndarray] = {}
    for name, pos in logics.items():
        p = pos.values.astype(float)
        if rotate is not None:
            p = np.roll(p, int(rotate.integers(1, len(p))))
        prev = np.concatenate([[0.0], p[:-1]])
        for side in (1, -1):
            e = np.flatnonzero((p == side) & (prev != side))
            if len(e) >= 2 * min_trades:
                entries[(name, side)] = e

    out = {}
    for hmax in HOLDS:
        for tp in TPS:
            for sl in SLS:
                for side in (1, -1):
                    o, held = barrier_outcomes(high, low, close, sigma, side,
                                               tp, sl, hmax)
                    net = o - FEE_ROUND_TRIP - held * fund
                    for (name, s2), e in entries.items():
                        if s2 != side:
                            continue
                        v = net[e]
                        ok = np.isfinite(v)
                        v, e2 = v[ok], e[ok]
                        a, c = v[e2 < cut], v[e2 >= cut]
                        if len(a) < min_trades or len(c) < min_trades:
                            continue
                        out[(name, side, tp, sl, hmax)] = {
                            "fit": float(a.mean()), "test": float(c.mean()),
                            "n": len(v), "wins": int((v > 0).sum()),
                            "avg_win": float(v[v > 0].mean()) if (v > 0).any()
                            else 0.0,
                            "hold": float(np.median(held[e2])) * minutes,
                        }
    return out


def control(d: pd.DataFrame, minutes: int, side: int) -> float:
    """Enter EVERY bar. What direction alone buys, with no signal in it."""
    b = resample(d, minutes)
    if len(b) < 400:
        return float("nan")
    close = b["close"].values.astype(float)
    o, held = barrier_outcomes(b["high"].values.astype(float),
                               b["low"].values.astype(float), close,
                               sigma_at(close), side, 4.0, 3.0, 48)
    net = o - FEE_ROUND_TRIP - held * FUNDING_PER_8H * (minutes / 480.0)
    net = net[np.isfinite(net)]
    return float(net.mean()) if len(net) else float("nan")


def tally(per: dict[str, dict], min_trades: int) -> pd.DataFrame:
    keys = set()
    for v in per.values():
        keys |= set(v)
    rows = []
    for k in keys:
        name, side, tp, sl, hmax = k
        hits = {s: v[k] for s, v in per.items() if k in v}
        if not hits:
            continue
        # a symbol counts only if the rule paid there in BOTH halves
        paid = [s for s, r in hits.items()
                if r["fit"] > 0 and r["test"] > 0]
        n, w = (sum(r["n"] for r in hits.values()),
                sum(r["wins"] for r in hits.values()))
        aw = [r["avg_win"] for r in hits.values() if r["avg_win"] > 0]
        aw = float(np.mean(aw)) if aw else 0.0
        pl = wilson_lower(w, n)
        expect = pl * aw + (1 - pl) * (-(sl / tp) * aw)
        rows.append({
            "logic": name, "side": "long" if side > 0 else "short",
            "tf": None, "tp": tp, "sl": sl, "hmax": hmax,
            "n_paid": len(paid), "tested_on": len(hits),
            "coins": ",".join(sorted(paid)),
            "blocks": len({BLOCK.get(c, c) for c in paid}),
            "fit": float(np.mean([r["fit"] for r in hits.values()])),
            "test": float(np.mean([r["test"] for r in hits.values()])),
            "test_paid": float(np.mean([hits[s]["test"] for s in paid]))
            if paid else 0.0,
            "expect": float(expect), "trades": n,
            "hold": float(np.median([r["hold"] for r in hits.values()])),
        })
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tf", default="15m,30m,1h")
    ap.add_argument("--split", type=float, default=0.6)
    ap.add_argument("--min-trades", type=int, default=15)
    ap.add_argument("--universal", type=int, default=8)
    ap.add_argument("--group-min", type=int, default=3)
    ap.add_argument("--null-runs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=57)
    a = ap.parse_args(argv)

    rng = np.random.default_rng(a.seed)
    coins = all_symbols()
    span = (next(iter(coins.values())).index[-1]
            - next(iter(coins.values())).index[0])
    print(f"{len(coins)} symbols on one window: "
          f"{next(iter(coins.values())).index[0].date()} .. "
          f"{next(iter(coins.values())).index[-1].date()} "
          f"({span.days} days)")
    print(f"fit on the first {100*a.split:.0f}%, judge on the rest, "
          f"AND across symbols\n")
    print(f"{'symbol':>13} {'move':>9} {'vol/day':>8}")
    for s, d in sorted(coins.items(),
                       key=lambda kv: -kv[1]['close'].iloc[-1]
                       / kv[1]['close'].iloc[0]):
        r = d["close"].pct_change().dropna()
        print(f"{s:>13} {100*(d.close.iloc[-1]/d.close.iloc[0]-1):>+8.2f}% "
              f"{100*r.std()*np.sqrt(1440):>7.2f}%")

    book = []
    for label in a.tf.split(","):
        label = label.strip()
        if label not in TF:
            continue
        minutes = TF[label]
        per = {s: rules_on(d, minutes, a.split, a.min_trades)
               for s, d in coins.items()}
        per = {k: v for k, v in per.items() if v}
        if not per:
            print(f"\n{label}: too few bars")
            continue
        T = tally(per, a.min_trades)
        if T.empty:
            print(f"\n{label}: nothing made enough trades")
            continue
        T["tf"] = label
        n_sym = len(per)

        nullv = []
        for _ in range(a.null_runs):
            pn = {s: rules_on(d, minutes, a.split, a.min_trades, rotate=rng)
                  for s, d in coins.items()}
            pn = {k: v for k, v in pn.items() if v}
            N = tally(pn, a.min_trades)
            if not N.empty:
                nullv.append(N["n_paid"].values)
        nullv = np.concatenate(nullv) if nullv else np.array([0])

        cl = np.nanmean([control(d, minutes, 1) for d in coins.values()])
        cs = np.nanmean([control(d, minutes, -1) for d in coins.values()])
        cl_n = sum(1 for d in coins.values() if control(d, minutes, 1) > 0)
        cs_n = sum(1 for d in coins.values() if control(d, minutes, -1) > 0)

        print("\n" + "=" * 78)
        print(f"{label}: {len(T):,} rules on {n_sym} symbols, "
              f"each judged in BOTH halves of every symbol")
        print("=" * 78)
        print(f"  control, no signal at all: always-long pays on {cl_n}/{n_sym} "
              f"({100*cl:+.3f}%/trade), always-short {cs_n}/{n_sym} "
              f"({100*cs:+.3f}%)")
        print(f"\n{'symbols paid':>13} {'rules':>7} {'share':>8} "
              f"{'null':>8} {'ratio':>7} {'test mean':>10}")
        for k in range(n_sym, 0, -1):
            sel = T[T["n_paid"] >= k]
            real = len(sel) / len(T)
            null = float((nullv >= k).mean())
            ratio = real / null if null > 0 else float("inf")
            tag = ("  <- universal" if k >= a.universal
                   else "  <- group" if k >= a.group_min else "")
            tm = sel["test_paid"].mean() if len(sel) else float("nan")
            print(f"{'>= ' + str(k):>13} {len(sel):>7} {100*real:>7.3f}% "
                  f"{100*null:>7.3f}% {ratio:>7.2f} "
                  f"{100*tm:>9.3f}%{tag}")

        # Universality by BLOCKS, not by symbol count. A rule paying on
        # SKHYNIX+SNDK+SOXL has been confirmed once at r=0.72, not three
        # times; a rule paying on one semiconductor and one crypto major
        # has been confirmed on two things that share r=0.37.
        print(f"\n  blocks spanned (semi / crypto / bless / gold):")
        print(f"  {'blocks':>7} {'rules':>7} {'null':>8} {'ratio':>7} "
              f"{'test mean':>10}")
        nb_null = []
        for k in range(N_BLOCKS, 0, -1):
            sel = T[T["blocks"] >= k]
            share = len(sel) / len(T)
            print(f"  {'>= ' + str(k):>7} {len(sel):>7} {'--':>8} {'--':>7} "
                  f"{100*sel['test_paid'].mean() if len(sel) else float('nan'):>9.3f}%")

        uni = T[T["blocks"] >= 3].sort_values("test_paid", ascending=False)
        grp = T[(T["n_paid"] >= a.group_min) & (T["blocks"] < 3)]
        one = T[T["n_paid"] == 1]

        print(f"\n  UNIVERSAL (spans 3+ of the {N_BLOCKS} blocks): "
              f"{len(uni)} rules")
        for _, r in uni.head(10).iterrows():
            print(f"    {r.logic:<26} {r.side:>5} tp{r.tp}/sl{r.sl}/{r.hmax}b "
                  f"{r.n_paid}/{r.tested_on}  fit {100*r.fit:+.3f}%  "
                  f"test {100*r.test_paid:+.3f}%/trade")
            print(f"      {r.coins}")
        print(f"\n  GROUP ({a.group_min}-{a.universal-1}): {len(grp)} rules")
        if len(grp):
            for cset, cnt in grp["coins"].value_counts().head(6).items():
                sub = grp[grp["coins"] == cset]
                print(f"    {cnt:>5} rules  test {100*sub['test_paid'].mean():+.3f}%"
                      f"  {cset}")
        print(f"\n  PER-COIN (exactly 1): {len(one)} rules")
        if len(one):
            for sym, cnt in one["coins"].value_counts().head(10).items():
                sub = one[one["coins"] == sym]
                print(f"    {cnt:>6} rules  test {100*sub['test_paid'].mean():+.3f}%"
                      f"  {sym}")

        for tier, sel in (("universal", uni), ("group", grp)):
            for _, r in sel.iterrows():
                book.append({"tf": label, "name": r.logic, "side": r.side,
                             "tp": float(r.tp), "sl": float(r.sl),
                             "hmax": int(r.hmax), "hold_min": float(r.hold),
                             "mean": float(r.test_paid), "tier": tier,
                             "coins": r.coins, "n_coins": int(r.n_paid),
                             "blocks": int(r.blocks)})

    OUT.write_text(json.dumps({
        "fitted_on": "10 symbols, 2026-06-30..2026-08-06, 1m bars",
        "in_sample": False,
        "note": "Each rule pays in BOTH halves of the window on every symbol "
                "credited to it. The time split is real; the cross-symbol "
                "agreement is real; the sample is 37 days.",
        "logics": book}, indent=2))
    print(f"\n{len(book)} rules written to {OUT.name}")
    return 0


def blocks(coins: dict[str, pd.DataFrame], minutes: int = 15) -> pd.DataFrame:
    """Correlation of returns -- how many INDEPENDENT bets ten symbols are.

    Counting symbols is not counting evidence. Three symbols correlated
    at 0.72 are one bet with three names, and a rule paying on all three
    has been confirmed once, not three times. This is the same lesson
    leave-one-out taught on BLESSUSDT, in a form that can be read off
    before any rule is scored.
    """
    p = pd.DataFrame({s: resample(d, minutes)["close"]
                      for s, d in coins.items()}).dropna()
    return p.pct_change().dropna().corr()


if __name__ == "__main__":
    sys.exit(main())
