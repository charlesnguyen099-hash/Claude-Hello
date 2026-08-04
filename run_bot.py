#!/usr/bin/env python3
"""Paper-trade the twelve-method voting logic on live Bybit data.

    pip install pandas numpy pybit
    python run_bot.py

No API key, no account, no order is ever placed. It reads Bybit's public
kline and ticker endpoints and keeps the $10 balance in memory.

    python run_bot.py --equity 10
    python run_bot.py --top 30 --max-positions 3
    python run_bot.py --min-votes 2 --exit net_TP3.0_SL1.5
    python run_bot.py --symbols BTCUSDT,ETHUSDT,SOLUSDT --trades-csv session.csv

Twelve methods vote on every 30-minute bar; a position opens only where
enough of them fire and agree on direction. Leverage is the file's
flexible sizing (17-100x from the setup's own stop distance). The exit is
fixed at entry — the file's best_exit_strategy column picks the winner
after the fact, which a live bot cannot do.

Ctrl+C prints closed trades (winners, losers, stop-outs, liquidations,
total profit, total loss), positions still open with the methods that
opened them, and the two combined.

Backtested on 2025-2026, every fixed exit averages between -0.20% and
-0.34% per trade. The only positive column in the file is the one that
chooses the exit in hindsight.
"""
from __future__ import annotations

import argparse
import logging
import sys

from wl import exits as X


def discover_symbols(client, top: int) -> list[str]:
    r = client.get_tickers(category="linear")
    rows = [x for x in r["result"]["list"] if x["symbol"].endswith("USDT")]
    rows.sort(key=lambda x: float(x.get("turnover24h") or 0), reverse=True)
    return [x["symbol"] for x in rows[:top]]


def main() -> int:
    p = argparse.ArgumentParser(
        description="Paper-trade the twelve-method voting logic on live Bybit data.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--equity", type=float, default=10.0)
    p.add_argument("--symbols", default=None, help="comma-separated; omit to use --top")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--max-positions", type=int, default=3)
    p.add_argument("--min-votes", type=int, default=1,
                   help="methods that must fire before a trade is taken")
    p.add_argument("--exit", default="net_TRAILING", choices=X.STRATEGIES,
                   help="exit strategy, fixed at entry (default: net_TRAILING)")
    p.add_argument("--poll-seconds", type=int, default=15)
    p.add_argument("--trades-csv", default=None)
    p.add_argument("--testnet", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    from pybit.unified_trading import HTTP

    from wl import bot as B
    from wl import methods as M

    client = HTTP(testnet=args.testnet)

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        try:
            symbols = discover_symbols(client, args.top)
        except Exception as exc:
            print(f"Could not list symbols from Bybit: {exc}", file=sys.stderr)
            return 1

    print("=" * 70)
    print("  PAPER TRADING — twelve-method voting, virtual money, real prices")
    print("=" * 70)
    print(f"  Starting balance : ${args.equity:,.2f} (simulated)")
    print(f"  Symbols scanned  : {len(symbols)}")
    print(f"    {', '.join(symbols[:12])}{' ...' if len(symbols) > 12 else ''}")
    print(f"  Max open at once : {args.max_positions}")
    print(f"  Methods          : {len(M.METHOD_NAMES)} voting on 30m bars")
    print(f"  Min votes to open: {args.min_votes} (and they must agree on direction)")
    print(f"  Exit strategy    : {args.exit} (fixed at entry)")
    print(f"  Leverage         : flexible {X.leverage_flexible(10.0):.0f}-"
          f"{X.leverage_flexible(0.1):.0f}x from the setup's stop distance")
    print(f"  Fee              : {X.FEE_ROUND_TRIP*100:.3f}% round trip")
    print(f"  Price source     : Bybit {'TESTNET' if args.testnet else 'MAINNET'} "
          f"public API (no key, no orders)")
    print("=" * 70)
    for m in M.METHOD_NAMES:
        print(f"    {m}")
    print("=" * 70)
    print("  Ctrl+C stops the bot and prints the session summary.")
    print("=" * 70)
    print()

    try:
        B.run(client, symbols, args.equity, args.max_positions, args.exit,
              args.min_votes, args.poll_seconds, args.trades_csv)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"\nStopped on an error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
