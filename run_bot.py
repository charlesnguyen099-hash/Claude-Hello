#!/usr/bin/env python3
"""Paper-trade the potential-scored logic on live Bybit data, virtual $10.

    pip install pandas numpy scikit-learn pybit
    python run_bot.py

No API key, no account, no order is ever placed. It reads Bybit's public
kline and ticker endpoints and keeps the balance in memory.

WHAT IT TRADES. One logic, built in three steps and nothing else:

  fp/labels.py   With the future in hand, mark every triple-barrier trade
                 in the past that profits after the WHOLE exchange bill --
                 taker in, taker out, funding. That is the answer sheet.
                 The target scales with sqrt(horizon), so a break-even win
                 rate lands near a coin flip instead of the 94% a
                 one-minute target implies against a 0.11% round trip.

  fp/factors.py  153 numbers describing each bar: returns, volatility,
                 range position, volume, order flow and candle shape at
                 nine lookbacks from one minute to two hours; where the
                 bar sits against its own recent extremes; and what the
                 OTHER nine coins are doing at that instant -- rank,
                 market move, dispersion, residual.

  fp/engine.py   One classifier per (target, stop, hold, side), fitted
                 across all ten coins at once, calibrated on rows the
                 trees never saw. Its win probability meets the shape's
                 break-even to produce ONE number:

                     POTENTIAL = 100 x (p - p_be) / (1 - p_be)

                 0 = break-even, 100 = cannot lose. The stake reads
                 straight off it, so a high-potential setup takes a large
                 slice and a marginal one takes little.

Only shapes that survived fp/run_engine.py's walk-forward are shipped.
A shape that lost out of sample is not traded at a smaller size -- it is
not traded.

    python run_bot.py                       # the scoped 10 coins
    python run_bot.py --symbols BTCUSDT,ETHUSDT
    python run_bot.py --max-margin-pct 100  # let potential 100 go all in
    python run_bot.py --equity 100

Three threads: one keeps a standing signal for every slot, one re-prices
every open position each second so targets and stops fire promptly, and
one prints the capital and P&L dashboard. Ctrl+C prints the session
summary.
"""
from __future__ import annotations

# OpenBLAS reserves a per-thread scratch buffer for as many threads as it
# believes the machine has, at import time, before the bot knows whether
# it needs any. It does not: the models score one row at a time. Two live
# runs died on "OpenBLAS error: Memory allocation still failed after 10
# retries" before scanning a single bar. This must run before numpy is
# imported by anything, so it sits directly under the __future__ import
# -- above that line is a SyntaxError.
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse
import json
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def scoped_symbols() -> list[str]:
    """The coins the logic was fitted and validated on.

    Every cross-sectional factor -- rank, dispersion, residual -- is
    defined against THIS set. Scanning a different board changes what
    those numbers mean, which is why the scope is read from the model's
    own metadata rather than chosen at the command line by default.
    """
    try:
        return list(json.loads(
            (HERE / "data" / "scope.json").read_text())["symbols"])
    except Exception:
        return []


def main() -> int:
    p = argparse.ArgumentParser(
        description="Paper-trade the potential-scored logic on live Bybit "
                    "data.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--equity", type=float, default=10.0,
                   help="starting virtual balance (default 10)")
    p.add_argument("--symbols", default=None,
                   help="comma-separated. Omit to scan the coins the logic "
                        "was validated on -- the cross-sectional factors are "
                        "defined against that set")
    p.add_argument("--max-positions", type=int, default=0,
                   help="cap concurrent positions (default 0 = no cap; free "
                        "margin is the limit)")
    p.add_argument("--min-stake", type=float, default=5.0,
                   help="FLOOR on one trade's margin, percent of live equity "
                        "(default 5). If a setup is worth taking it is taken "
                        "properly; the floor never overrides the ruin cap")
    p.add_argument("--margin-pct", type=float, default=5.0,
                   help="base slice, percent of equity. Only used with "
                        "--earn-stake")
    p.add_argument("--max-margin-pct", type=float, default=100.0,
                   help="ceiling on ONE trade's margin, percent of equity "
                        "(default 100: a setup scoring 100/100 may take the "
                        "whole account)")
    p.add_argument("--stake-curve", choices=("linear", "square"),
                   default="linear",
                   help="linear (default): stake%% = potential, so a 60/100 "
                        "setup commits 60%% of equity. square: "
                        "stake%% = potential^2/100, the conservative Kelly "
                        "answer for an ESTIMATED edge")
    p.add_argument("--earn-stake", action="store_true",
                   help="make each shape EARN its way up from --margin-pct "
                        "to --max-margin-pct as it builds a live record. Off "
                        "by default: the stake follows the setup in front of "
                        "it, not the record of the ones behind it")
    p.add_argument("--share-stakes", action="store_true",
                   help="when several setups fire at once, scale them all "
                        "down so the whole standing set fits. Off by "
                        "default: fill in potential order until free margin "
                        "runs out")
    p.add_argument("--max-notional-x", type=float, default=0.0,
                   help="ceiling on total position value as a multiple of "
                        "equity. 0 = none: how much to hold is decided by "
                        "each trade's potential")
    p.add_argument("--max-leverage", type=float, default=None,
                   help="cap leverage (default: solved per trade from the "
                        "stop distance and the hold)")
    p.add_argument("--poll-seconds", type=int, default=20)
    p.add_argument("--trades-csv", default=None,
                   help="append every closed trade to this file")
    p.add_argument("--testnet", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s")

    from pybit.unified_trading import HTTP

    from fp import bot as B
    from fp.live_book import LogicBook
    from fp.live_engine import Engine

    lb = LogicBook()
    eng = Engine()
    client = HTTP(testnet=args.testnet)

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")
                   if s.strip()]
    else:
        # The board has to include every coin the cross-sectional methods
        # rank against, not only the coins being traded -- dropping one
        # changes every other coin's rank.
        symbols = scoped_symbols()
    if not symbols:
        print("No symbols. Pass --symbols, or run python -m fp.train_engine "
              "to write data/scope.json.", file=sys.stderr)
        return 1

    fee = 2 * 0.00055
    print("=" * 74)
    print("  PAPER TRADING - virtual money, real Bybit prices")
    print("=" * 74)
    print(f"  Starting balance : ${args.equity:,.2f} (simulated)")
    print(f"  Coins scanned    : {len(symbols)}")
    print(f"    {', '.join(symbols)}")
    if lb.ok and lb.pairs:
        by = {}
        for p_ in lb.pairs:
            by.setdefault(p_["symbol"], []).append(p_)
        proven = int(lb.meta.get("proven", 0))
        print(f"  Logic            : {len(lb.pairs)} (coin, strategy) pairs, "
              f"{proven} PROVEN, "
              f"{len(lb.pairs) - proven} CANDIDATE")
        print(f"                     Each was profitable FORWARD in every "
              f"walk-forward fold it")
        print(f"                     traded in. A CANDIDATE was chosen by "
              f"looking across those")
        print(f"                     folds, so it is NOT independently "
              f"validated -- THIS RUN is")
        print(f"                     its test. The live record promotes what "
              f"pays and RETIRES")
        print(f"                     anything two standard errors below zero "
              f"on its own trades.")
        for symb in sorted(by):
            for p_ in sorted(by[symb], key=lambda r: -r["min_t"])[:6]:
                print(f"      {symb:<12} {p_['strategy']:<30} "
                      f"{100*p_['mean']:+.4f}%/trade  "
                      f"{p_['folds']} folds  stake {100*p_['stake']:.0f}%")
        print(f"  Entry            : when the strategy's state turns on, at "
              f"whatever the market is")
        print(f"  Exit             : when it turns off or flips. No target, "
              f"no stop, no time")
        print(f"                     limit -- the study that validated these "
              f"used none, and")
        print(f"                     adding one live would trade a different "
              f"rule.")
        print(f"  Stake            : floor {args.min_stake:.0f}% of live "
              f"equity while unproven, then that")
        print(f"                     pair's own half-Kelly on its LIVE "
              f"record, up to "
              f"{args.max_margin_pct:.0f}%")
        print(f"  Cost per round trip: {100*fee:.3f}% taker both sides, plus "
              f"live funding")
        print(f"  Price source     : Bybit "
              f"{'TESTNET' if args.testnet else 'MAINNET'} public API "
              f"(no key, no orders)")
        print("=" * 74)
        print("  Ctrl+C stops the bot and prints the session summary.")
        print("=" * 74)
    elif not eng.ok or not eng.models:
        print()
        print("  NO LOGIC IS LOADED.")
        print("  Neither fp/logic_book.json nor fp/engine_model.pkl holds")
        print("  anything that survived validation, so the bot will not open")
        print("  a position. Build one with:")
        print("      python -m fp.run_percoin    # every method, every coin")
        print("      python -m fp.run_engine     # the model, then train_engine")
        print("  An empty book means nothing was profitable FORWARD in every")
        print("  fold it traded in. That is a result, not a fault.")
        print("=" * 74)
    else:
        m = eng.meta
        print(f"  Logic            : {len(eng.models)} shape/side "
              f"combinations that SURVIVED the walk-forward")
        for (tp, sl, hold, side) in sorted(eng.models):
            print(f"      target {tp:.1f} sigma  stop {sl:.1f} sigma  "
                  f"hold <= {hold:>3}m  "
                  f"{'long ' if side > 0 else 'short'}")
        print(f"  Factors          : {len(eng.columns)} per bar -- price, "
              f"flow, state and cross-section")
        print(f"                     lookbacks {m.get('lookbacks')} minutes, "
              f"state windows {m.get('state_windows')}")
        print(f"  Trained on       : {m.get('train_rows', 0):,} rows, "
              f"{m.get('train_window', ['?', '?'])[0]}.."
              f"{m.get('train_window', ['?', '?'])[1]}")
        print(f"  Barriers         : target and stop scale with the "
              f"volatility at entry AND")
        print(f"                     with sqrt(hold), so break-even sits "
              f"near a coin flip")
        print(f"                     instead of the 94% a one-minute target "
              f"implies.")
        print(f"  Gate             : potential >= {eng.gate:.0f}/100, the "
              f"level the walk-forward")
        print(f"                     fixed before any result was read.")
        print(f"  Potential scale  : 100 x (p - p_be) / (1 - p_be). "
              f"0 = break-even,")
        print(f"                     100 = cannot lose by its own barriers. "
              f"p is the")
        print(f"                     calibrated win probability, LOWER-bounded "
              f"by its own")
        print(f"                     calibration sample -- sizing off a point "
              f"estimate is")
        print(f"                     how a backtest becomes a margin call.")
        if args.stake_curve == "linear":
            print(f"  Stake            : potential, read as a percent of "
                  f"equity, capped at "
                  f"{args.max_margin_pct:.0f}%.")
            print(f"      potential  20 -> 20%   40 -> 40%   60 -> 60%   "
                  f"80 -> 80%   100 -> ALL IN")
        else:
            print(f"  Stake            : potential^2/100 percent of equity, "
                  f"capped at {args.max_margin_pct:.0f}%.")
            print(f"      potential  20 ->  4%   40 -> 16%   60 -> 36%   "
                  f"80 -> 64%   100 -> ALL IN")
        print(f"    Half-Kelly and a ruin cap apply on top, so a big score on "
              f"a wide stop")
        print(f"    still cannot bet the account.")
        print(f"  Slots            : one position per COIN x SHAPE x SIDE, so "
              f"a coin can")
        print(f"                     carry several at once instead of being "
              f"locked by the")
        print(f"                     first one to fire.")
        print(f"  Cost per round trip: {100*fee:.3f}% of notional, taker both "
              f"sides, plus each")
        print(f"                     symbol's live funding rate off the "
              f"ticker feed.")
        print(f"  Price source     : Bybit "
              f"{'TESTNET' if args.testnet else 'MAINNET'} public API "
              f"(no key, no orders)")
        print("=" * 74)
        print("  Ctrl+C stops the bot and prints the session summary.")
        print("=" * 74)

    try:
        B.run(client, symbols, args.equity, args.max_positions,
              "net_TP3.0_SL1.5", 1, args.poll_seconds, args.trades_csv, fee,
              args.max_leverage, args.margin_pct / 100.0,
              args.max_notional_x, 1.0, False, None, "engine",
              True, "kelly", args.max_margin_pct / 100.0, False, 0.0,
              None, "book.json", False, False, args.earn_stake,
              args.stake_curve, args.share_stakes,
              "band", 2.0, 30, 0.05, args.min_stake / 100.0)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"\nStopped on an error: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
