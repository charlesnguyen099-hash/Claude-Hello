"""Never close at a loss. Measure where the money actually goes.

The operator's argument, and it is a correct one as far as it goes:

    a trade is only "profitable" above +1% net of every fee, so if the
    bot exits only at +1%, the net cannot be negative.

Every closed trade IS a winner under that rule -- by construction, and
the measurement below confirms it. What the rule cannot do is make the
loss disappear. It moves it from the realized column into the OPEN one:
a position that never reaches +1% is simply never closed, and it sits
there, underwater, holding the coin hostage.

So this file reports three numbers, never one:

  REALIZED     the closed trades, all of them above +1%
  UNREALIZED   what the still-open positions are worth right now
  TRUE NET     the sum, which is what the account is actually worth

Reporting only the first is how a strategy with no stop-loss looks
perfect right up until the margin call.

TWO THINGS THE RULE COSTS, both measured here:

  LIQUIDATION. A position at -43% of notional is dead at any meaningful
  leverage: at 3x that is -130% of margin, and the exchange closes it
  long before. "Never close at a loss" is only executable if leverage is
  low enough that the drawdown fits inside the margin, so the leverage
  needed to survive the worst open position is reported.

  THE COIN IS LOCKED. One position per coin at trade time means a trade
  that never recovers stops that coin trading at all. The hold length of
  the open position is the opportunity cost, and it is printed.

BUILD TIME vs TRADE TIME. The one-position rule is applied ONLY here, at
execution. The logic library still covers every chance -- fp/coverage.py
measures that at 100.0% -- and nothing is filtered out when the logic is
built.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from fp import data as D
from fp import onecoin as OC
from fp import strategies as SG

HERE = Path(__file__).resolve().parent
BOOK = HERE / "logic_book.json"

MIN_TRADES = 30
TRAIN_FRAC = 0.55
FOLD_DAYS = 70
MIN_TEST_DAYS = 7
TOP_N = 40
MIN_STAKE = 0.05

# Bybit closes a position when the loss reaches roughly this share of the
# margin behind it.
LIQ_FRACTION = 0.90


def folds_for(index):
    start, end = index[0], index[-1]
    total = (end - start).days
    if total < 3 * MIN_TEST_DAYS:
        return []
    train_days = max(int(total * TRAIN_FRAC), MIN_TEST_DAYS)
    block = min(FOLD_DAYS, max((total - train_days) // 2, MIN_TEST_DAYS))
    out, cut = [], start + np.timedelta64(train_days, "D")
    while cut < end - np.timedelta64(MIN_TEST_DAYS - 1, "D"):
        stop = min(cut + np.timedelta64(block, "D"), end)
        out.append((start, cut, stop))
        cut = stop
    return out


def split(trades, n_bars):
    """(closed, open) -- the last trade is open if it runs to the end."""
    if not trades:
        return [], None
    last = trades[-1]
    if last[1] >= n_bars - 1:
        return trades[:-1], last
    return trades, None


def main():
    print("=" * 84)
    print("  NEVER CLOSE AT A LOSS -- realized, unrealized, and the truth")
    print("  logic covers every chance (see fp.coverage); ONE position per")
    print("  coin is applied here, at trade time, and nowhere else")
    print("=" * 84, flush=True)

    P = D.load()
    book, walk = {}, {}
    G = dict(real=0.0, unreal=0.0, closed=0, locked=0)

    for sym in sorted(P):
        d = P[sym]
        idx = d.index
        fl = folds_for(idx)
        if not fl:
            continue
        S = SG.all_strategies(d, P, sym)
        close = d["close"].values.astype("float64")
        pos = {t: i for i, t in enumerate(idx)}
        print(f"\n--- {sym}   {len(d):,} bars   {len(fl)} folds", flush=True)

        agg: dict[str, list] = {}
        rows = []
        for fi, (a, cut, stop) in enumerate(fl, 1):
            ia = pos[idx[idx >= a][0]]
            ic = pos[idx[idx >= cut][0]]
            ib = pos[idx[idx < stop][-1]]
            expect = {}
            for name, state in S.items():
                _, _, _, net = SG.trades(close[ia:ic], state[ia:ic])
                if len(net) >= MIN_TRADES:
                    expect[name] = float(net.mean())
            top = dict(sorted(expect.items(), key=lambda kv: -kv[1])[:TOP_N])
            for n_, v_ in top.items():
                agg.setdefault(n_, []).append(v_)
            if not top:
                print(f"    fold {fi}: no strategy with a positive "
                      f"training mean -- nothing to trade")
                rows.append(0.0)
                continue

            cc = close[ic:ib]
            sub = {n_: S[n_][ic:ib] for n_ in top}
            tr = OC.simulate(cc, sub, top, take_profit=True,
                             only_win_exit=True)
            closed, still = split(tr, len(cc))
            cv = np.array([t[3] for t in closed]) if closed else np.array([])
            un = still[3] if still else 0.0
            held = (still[1] - still[0]) if still else 0
            true_net = float(cv.sum() + un)
            rows.append(true_net)
            allwin = bool((cv > OC.WIN).all()) if len(cv) else True
            # What leverage could survive the worst point of the open
            # position? Anything above this is a liquidation, not a hold.
            max_lev = (LIQ_FRACTION / abs(un)) if un < 0 else float("inf")
            print(f"    fold {fi}: closed {len(cv):>4} (all >1%: {allwin})"
                  f"  realized {100*cv.sum():>+8.1f}%"
                  f"   open {100*un:>+7.2f}% held {held/1440:>5.1f}d"
                  f"   TRUE {100*true_net:>+8.1f}%"
                  f"   max lev {max_lev:>4.1f}x", flush=True)
            G["real"] += float(cv.sum()); G["unreal"] += un
            G["closed"] += len(cv); G["locked"] += held

        ranked = sorted(((n_, float(np.mean(v_))) for n_, v_ in agg.items()),
                        key=lambda kv: -kv[1])[:TOP_N]
        keep = [{"strategy": n_, "expect": v_, "stake": MIN_STAKE}
                for n_, v_ in ranked if v_ > 0]
        verdict = ("PROFITABLE forward, every fold"
                   if rows and all(r > 0 for r in rows) else
                   "LOSES forward" if rows and all(r <= 0 for r in rows)
                   else "MIXED")
        if keep:
            book[sym] = keep
            walk[sym] = {"folds": [float(r) for r in rows],
                         "verdict": verdict, "strategies": len(keep)}
        print(f"    {verdict}: "
              f"{', '.join(f'{100*r:+.1f}%' for r in rows) or 'no trades'}")

    print("\n" + "=" * 84)
    print(f"  REALIZED   : {100*G['real']:+.1f}% over {G['closed']:,} closed "
          f"trades -- every one of them above +1%")
    print(f"  UNREALIZED : {100*G['unreal']:+.1f}% sitting in positions that "
          f"never reached +1%")
    print(f"  TRUE NET   : {100*(G['real']+G['unreal']):+.1f}%")
    print(f"  LOCKED     : {G['locked']/1440:.0f} coin-days spent holding "
          f"trades that never came back")
    ship = [s for s, w in walk.items() if w["verdict"].startswith("PROFIT")]
    print(f"  SHIPS      : {len(ship)} coin(s) profitable forward in every "
          f"fold: {', '.join(ship) or 'none'}")
    BOOK.write_text(json.dumps(
        {"mode": "hold_until_win", "win_threshold": OC.WIN,
         "min_stake": MIN_STAKE, "top_n": TOP_N,
         "per_coin": {s: book[s] for s in ship} if ship else {},
         "walk_forward": walk,
         "realized": G["real"], "unrealized": G["unreal"],
         "closed": G["closed"]}, indent=1))
    print(f"  wrote {BOOK.name}")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    sys.exit(main())
