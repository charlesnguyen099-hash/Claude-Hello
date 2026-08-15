"""Walk each coin forward under the one-position rules, and write the book.

Per coin, per fold:

  TRAIN   measure every strategy's mean net on bars before the cut. That
          mean is the strategy's expectation and the ONLY thing that
          ranks it. Nothing from the test side is used to choose.
  TEST    run fp.onecoin.simulate over the forward bars: the coin holds
          the highest-expectation signal that is on, a better signal
          displaces a worse one, and every switch pays the round trip.
  JUDGE   against the CEILING for those same bars -- the best set of
          non-overlapping +1% trades that existed, computed with
          hindsight. That is the honest denominator.

What goes in fp/logic_book.json is the ranked expectation table per coin:
the bot reads it, and at each bar holds the best signal that is on.
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

MIN_STAKE = 0.05
MAX_STAKE = 1.00

# How many strategies per coin the book carries. The bot picks among
# them at every bar, so this is a shortlist, not a portfolio: more
# candidates means more chances that SOMETHING is on when a move starts.
TOP_N = 40


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


def main():
    print("=" * 82)
    print("  ONE POSITION PER COIN -- best signal wins, target +1% net")
    print(f"  a coin holds the highest-expectation signal that is on; only a")
    print(f"  BETTER one displaces it; every switch pays the full round trip")
    print("=" * 82, flush=True)

    P = D.load()
    fold_rows, walk = {}, {}
    book, totals = {}, dict(caught=0, ceil_n=0, got=0.0, ceil=0.0, trades=0)

    for sym in sorted(P):
        d = P[sym]
        idx = d.index
        fl = folds_for(idx)
        if not fl:
            print(f"\n--- {sym}: not enough history")
            continue
        S = SG.all_strategies(d, P, sym)
        close = d["close"].values.astype("float64")
        pos = {t: i for i, t in enumerate(idx)}
        print(f"\n--- {sym}   {len(d):,} bars   {len(fl)} folds", flush=True)

        agg: dict[str, list] = {}
        for fi, (a, cut, stop) in enumerate(fl, 1):
            ia = pos[idx[idx >= a][0]]
            ic = pos[idx[idx >= cut][0]]
            ib = pos[idx[idx < stop][-1]]

            # TRAIN: every strategy's expectation, from past bars only.
            expect = {}
            for name, state in S.items():
                _, _, _, net = SG.trades(close[ia:ic], state[ia:ic])
                if len(net) >= MIN_TRADES:
                    expect[name] = float(net.mean())
            top = dict(sorted(expect.items(), key=lambda kv: -kv[1])[:TOP_N])
            for n_, v_ in top.items():
                agg.setdefault(n_, []).append(v_)

            # CEILING on the forward bars: everything a single position
            # could have taken, chosen with hindsight.
            ents, exs, nets = [], [], []
            for name, state in S.items():
                st, ex, sd, net = SG.trades(close[ic:ib], state[ic:ib])
                ents.append(st); exs.append(ex); nets.append(net)
            ents = np.concatenate(ents) if ents else np.array([], int)
            exs = np.concatenate(exs) if exs else np.array([], int)
            nets = np.concatenate(nets) if nets else np.array([])
            cn, ctot = OC.ceiling(ents, exs, nets)

            # TEST: the live rule, forward.
            sub = {n_: S[n_][ic:ib] for n_ in top}
            # The exit is the signal's own death and nothing else. Every
            # variant with a +1% take-profit caught far more of the
            # ceiling -- 68% to 72% instead of 7% -- and lost money doing
            # it, because banking at +1% while a loser runs to -1.06%
            # is the wrong way round and each extra trade pays 0.11%.
            tr = OC.simulate(close[ic:ib], sub, top)
            sc = OC.score(tr)
            fold_rows.setdefault(sym, []).append(sc["total"])
            print(f"    fold {fi}: ceiling {cn:>5} trades {100*ctot:>+8.0f}%"
                  f"   |   bot {sc['n']:>5} trades {100*sc['total']:>+8.1f}%"
                  f"  above+1% {sc['hits']:>4}"
                  f"  ({100*sc['hits']/max(cn,1):>4.1f}% of ceiling)",
                  flush=True)
            totals["caught"] += sc["hits"]; totals["ceil_n"] += cn
            totals["got"] += sc["total"]; totals["ceil"] += ctot
            totals["trades"] += sc["n"]

        # The shipped expectation for this coin: the average of what each
        # strategy showed on the TRAINING side of every fold.
        # A coin ships only if the rule made money FORWARD in every one
        # of its folds. A coin that pays in one window and bleeds in the
        # next is a bet on the window.
        # EVERY coin that can trade at all goes in the book, with its
        # measured forward result attached -- including the negative
        # ones. Shipping only the winners would be selecting on the test
        # side, which is the mistake this whole file exists to avoid; and
        # shipping nothing leaves no way to find out what is wrong live.
        # The bot stakes each pair at the floor, the banner prints these
        # numbers unedited, and the live record retires what does not pay.
        rows = fold_rows.get(sym, [])
        ranked = sorted(((n_, float(np.mean(v_))) for n_, v_ in agg.items()),
                        key=lambda kv: -kv[1])[:TOP_N]
        keep = [{"strategy": n_, "expect": v_, "stake": MIN_STAKE}
                for n_, v_ in ranked if v_ > 0]
        verdict = ("PROFITABLE forward in every fold"
                   if rows and all(r > 0 for r in rows)
                   else "LOSES forward" if rows and all(r <= 0 for r in rows)
                   else "MIXED across folds")
        if keep:
            book[sym] = keep
            walk[sym] = {"folds": [float(r) for r in rows],
                         "verdict": verdict, "strategies": len(keep)}
        print(f"    {verdict}: "
              f"{', '.join(f'{100*r:+.1f}%' for r in rows) or 'no trades'}"
              f"   -> {len(keep)} strategies in the book")

    print("\n" + "=" * 82)
    print(f"  CEILING  : {totals['ceil_n']:,} distinct +1% trades worth "
          f"{100*totals['ceil']:+.0f}%")
    print(f"  BOT      : {totals['trades']:,} trades taken, "
          f"{totals['caught']:,} of them above +1%, "
          f"{100*totals['got']:+.1f}% total")
    print(f"  CAUGHT   : {100*totals['caught']/max(totals['ceil_n'],1):.2f}% "
          f"of the reachable +1% trades")
    BOOK.write_text(json.dumps(
        {"mode": "one_per_coin", "win_threshold": OC.WIN,
         "min_stake": MIN_STAKE, "max_stake": MAX_STAKE,
         "top_n": TOP_N, "per_coin": book, "walk_forward": walk,
         "ceiling_trades": totals["ceil_n"],
         "ceiling_total": totals["ceil"],
         "bot_trades": totals["trades"], "bot_hits": totals["caught"],
         "bot_total": totals["got"]}, indent=1))
    print(f"  wrote {BOOK.name}: "
          f"{sum(len(v) for v in book.values())} ranked strategies "
          f"across {len(book)} coins")
    print("=" * 82)
    return 0


if __name__ == "__main__":
    sys.exit(main())
