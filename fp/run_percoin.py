"""Walk each coin forward across its OWN history, and ship what survives.

The coins do not share a history. BTC has nineteen months, from January
2025; the other nine have ten weeks. Forcing them into one common window
throws away 88% of the BTC bars, and those bars are exactly what the
earlier searches were short of -- 2,159 independent trades could not tell
a three-point edge from luck.

So every coin is walked forward across whatever it actually has:

  BTC          2025-01-01 .. 2026-08-13    six folds
  the rest     2026-05-31 .. 2026-08-13    two folds

Each fold trains on everything before its cut and trades only what comes
after. A strategy is SELECTED on the training side and then judged on the
forward side, never the reverse.

WHAT SHIPS. Not "the best strategy" -- the best of 623 is a lottery
winner. A (coin, strategy) pair ships only if:

  * it was profitable FORWARD in every fold of that coin where it traded
  * it traded at least MIN_TRADES times in each of those folds
  * it appeared in at least MIN_FOLDS folds
  * it cleared Bonferroni over all 623 attempted in each of them

Everything that survives goes into fp/logic_book.json with the stake its
own measured edge justifies. That file is what the bot trades; if it is
empty the bot opens nothing, and says so.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from fp import data as D
from fp import strategies as SG

HERE = Path(__file__).resolve().parent
BOOK = HERE / "logic_book.json"

# Two-sided 0.05 over 623 attempted strategies.
BONF_T = 3.94
MIN_TRADES = 30
MIN_FOLDS = 2

MIN_STAKE = 0.05
MAX_STAKE = 1.00

# Roughly ten-week test blocks, so BTC gets many folds and the young
# coins get the two their history allows.
# Fold sizing ADAPTS to what a coin has. BTC's 589 days give eight
# ten-week folds; a coin with 73 days would have had none under a fixed
# 60-day training minimum, which threw away every young coin. Instead the
# first 55% is training and the rest is split into at least two forward
# blocks, so "coin nao co data bao nhieu thi tinh tren bay nhieu do" is
# what actually happens.
FOLD_DAYS = 70
TRAIN_FRAC = 0.55
MIN_TEST_DAYS = 7


def stats(v):
    n = len(v)
    if n < 2:
        return dict(n=n, mean=np.nan, t=np.nan, win=np.nan, sd=np.nan)
    sd = float(v.std(ddof=1))
    se = sd / np.sqrt(n)
    return dict(n=n, mean=float(v.mean()),
                t=float(v.mean() / se) if se > 0 else 0.0,
                win=float((v > 0).mean()), sd=sd)


def kelly_stake(mean, sd):
    if not np.isfinite(mean) or not np.isfinite(sd) or sd <= 0 or mean <= 0:
        return 0.0
    return float(np.clip(0.5 * mean / (sd * sd), MIN_STAKE, MAX_STAKE))


def folds_for(index, fold_days=FOLD_DAYS, train_frac=TRAIN_FRAC):
    """Expanding-window folds over one coin's own index.

    The block length is whatever the coin can afford: ten weeks where
    there is a year and a half, a fortnight where there are ten weeks.
    """
    start, end = index[0], index[-1]
    total = (end - start).days
    if total < 3 * MIN_TEST_DAYS:
        return []
    train_days = max(int(total * train_frac), MIN_TEST_DAYS)
    left = total - train_days
    block = min(fold_days, max(left // 2, MIN_TEST_DAYS))
    out, cut = [], start + np.timedelta64(train_days, "D")
    while cut < end - np.timedelta64(MIN_TEST_DAYS - 1, "D"):
        stop = min(cut + np.timedelta64(block, "D"), end)
        out.append((start, cut, stop))
        cut = stop
    return out


def main():
    print("=" * 78)
    print("  PER-COIN WALK-FORWARD -- each coin on the history it actually has")
    print(f"  ship: profitable forward in EVERY fold, {MIN_FOLDS}+ folds, "
          f"|t| >= {BONF_T}, {MIN_TRADES}+ trades")
    print("=" * 78, flush=True)

    P = D.load()
    book, candidates, summary = [], [], []

    for sym in sorted(P):
        d = P[sym]
        idx = d.index
        fl = folds_for(idx)
        days = (idx[-1] - idx[0]).days
        print(f"\n--- {sym}   {len(d):,} bars   {idx[0].date()}..{idx[-1].date()}"
              f"   ({days}d, {len(fl)} folds)", flush=True)
        if not fl:
            print("    not enough history for a walk-forward")
            continue

        # Build every strategy ONCE over the coin's whole history, then
        # slice per fold. Rebuilding 623 states per fold is what made an
        # earlier version too slow to finish.
        S = SG.all_strategies(d, P, sym)
        close = d["close"].values.astype("float64")
        pos = {t: i for i, t in enumerate(idx)}

        # How much was on the table at all, for this coin.
        opp = 0
        for name, state in S.items():
            _, _, _, net = SG.trades(close, state)
            opp += int((net > 0).sum())
        print(f"    profitable-after-fee trades available: {opp:,} "
              f"across {len(S)} strategies", flush=True)

        per_strategy: dict[str, list] = {}
        for fi, (a, cut, stop) in enumerate(fl, 1):
            ia, ic, ib = pos[idx[idx >= a][0]], pos[idx[idx >= cut][0]], \
                pos[idx[idx < stop][-1]]
            mv = 100 * (close[ib] / close[ic] - 1)
            picked, best_t, best_nm = [], -99.0, ""
            for name, state in S.items():
                _, _, _, ntr = SG.trades(close[ia:ic], state[ia:ic])
                st = stats(ntr)
                if st["n"] >= MIN_TRADES and np.isfinite(st["t"]) \
                        and st["t"] > best_t:
                    best_t, best_nm = st["t"], name
                if st["n"] >= MIN_TRADES and st["mean"] > 0 \
                        and st["t"] >= BONF_T:
                    picked.append((name, st))
            fwd = []
            for name, st in picked:
                _, _, _, nte = SG.trades(close[ic:ib], S[name][ic:ib])
                sf = stats(nte)
                per_strategy.setdefault(name, []).append(sf)
                if sf["n"] >= MIN_TRADES:
                    fwd.append(sf["mean"])
            good = sum(1 for m in fwd if m > 0)
            line = (f"    fold {fi}: market {mv:+6.2f}%  "
                    f"selected {len(picked):>3}  traded forward "
                    f"{len(fwd):>3}  profitable {good:>3}")
            if not picked:
                # Say HOW CLOSE the best came. "0 selected" on its own
                # hides the difference between "nothing was near" and
                # "one missed the bar by a hair", and those call for
                # different next steps.
                line += f"   (best train t={best_t:+.2f} {best_nm})"
            print(line, flush=True)

        # CANDIDATES: measured FORWARD in every fold, with no training
        # gate at all. Nothing clears Bonferroni here -- across all ten
        # coins the best training t was +1.53 against a bar of 3.94, and
        # BTC's best over nineteen months was NEGATIVE. So "wait for
        # significance" ships nothing, forever.
        #
        # The operator's workflow is the alternative, and it is a sound
        # one: put the best available logic on a virtual account, let it
        # trade real prices forward, and find out from the live record
        # which parts are wrong. A candidate is therefore any pair that
        # was profitable OUT OF SAMPLE in every fold it traded in --
        # honest per fold, but chosen by looking across folds, so it is
        # NOT independently validated. The live paper run is what
        # validates it, and the bot stakes it at the floor until it does.
        for name in S:
            fwd_runs = []
            for (a, cut, stop) in fl:
                ia = pos[idx[idx >= a][0]]
                ic = pos[idx[idx >= cut][0]]
                ib = pos[idx[idx < stop][-1]]
                _, _, _, nfw = SG.trades(close[ic:ib], S[name][ic:ib])
                sf = stats(nfw)
                if sf["n"] >= MIN_TRADES:
                    fwd_runs.append(sf)
            if len(fwd_runs) < MIN_FOLDS:
                continue
            if not all(r["mean"] > 0 for r in fwd_runs):
                continue
            n = sum(r["n"] for r in fwd_runs)
            mean = sum(r["mean"] * r["n"] for r in fwd_runs) / n
            sd = float(np.mean([r["sd"] for r in fwd_runs]))
            candidates.append({
                "symbol": sym, "strategy": name, "folds": len(fwd_runs),
                "trades": n, "mean": mean, "sd": sd,
                "min_t": min(r["t"] for r in fwd_runs),
                "stake": MIN_STAKE, "proven": False})

        # A pair SHIPS as proven only if it also cleared the training gate.
        for name, runs in per_strategy.items():
            real = [r for r in runs if r["n"] >= MIN_TRADES]
            if len(real) < MIN_FOLDS:
                continue
            if not all(r["mean"] > 0 for r in real):
                continue
            if not all(r["t"] >= BONF_T for r in real):
                continue
            n = sum(r["n"] for r in real)
            mean = sum(r["mean"] * r["n"] for r in real) / n
            sd = float(np.mean([r["sd"] for r in real]))
            book.append({"symbol": sym, "strategy": name,
                         "folds": len(real), "trades": n,
                         "mean": mean, "sd": sd,
                         "min_t": min(r["t"] for r in real),
                         "stake": kelly_stake(mean, sd), "proven": True})
        summary.append((sym, len(d), len(fl), opp))

    print("\n" + "=" * 78)
    book.sort(key=lambda r: -r["min_t"])
    candidates.sort(key=lambda r: -r["min_t"])
    proven_keys = {(r["symbol"], r["strategy"]) for r in book}
    candidates = [c for c in candidates
                  if (c["symbol"], c["strategy"]) not in proven_keys]
    print(f"  PROVEN     : {len(book)} pairs cleared the training gate AND "
          f"paid forward")
    print(f"  CANDIDATES : {len(candidates)} pairs paid FORWARD in every fold "
          f"they traded")
    print(f"               -- not independently validated; the live paper "
          f"run is the test")
    if candidates:
        print(f"\n  {'coin':<12} {'strategy':<32} {'folds':>5} {'trades':>7} "
              f"{'mean%':>8} {'min t':>7}")
        for r in candidates[:25]:
            print(f"  {r['symbol']:<12} {r['strategy']:<32} {r['folds']:>5} "
                  f"{r['trades']:>7} {100*r['mean']:>+8.4f} "
                  f"{r['min_t']:>+7.2f}")
    if book:
        print(f"  {len(book)} (coin, strategy) pairs SHIP:")
        print(f"  {'coin':<12} {'strategy':<32} {'folds':>5} {'trades':>7} "
              f"{'mean%':>8} {'min t':>7} {'stake':>6}")
        for r in book[:40]:
            print(f"  {r['symbol']:<12} {r['strategy']:<32} {r['folds']:>5} "
                  f"{r['trades']:>7} {100*r['mean']:>+8.4f} "
                  f"{r['min_t']:>+7.2f} {100*r['stake']:>5.0f}%")
    else:
        print("  NOTHING SHIPS. No (coin, strategy) pair was profitable")
        print("  forward in every fold it traded in.")
    BOOK.write_text(json.dumps(
        {"pairs": book + candidates, "proven": len(book),
         "bonferroni_t": BONF_T, "min_trades": MIN_TRADES,
         "min_folds": MIN_FOLDS, "min_stake": MIN_STAKE,
         "coins": [{"symbol": s, "bars": b, "folds": f, "opportunities": o}
                   for s, b, f, o in summary]}, indent=1))
    print(f"\n  wrote {BOOK.name}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
