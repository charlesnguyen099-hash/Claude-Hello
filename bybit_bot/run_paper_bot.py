#!/usr/bin/env python3
"""Paper-trade the strategy on live Bybit market data with a virtual $10.

Pull the repo, install the requirements, run this. Nothing else to set up:
it reads Bybit's PUBLIC kline and ticker endpoints, which need no API key,
and it never places an order of any kind. The $10 is a number in memory.

    pip install -r requirements.txt
    python run_paper_bot.py

Stop it with Ctrl+C whenever you like. It prints a full session summary:
trades that closed (winners, losers, total profit, total loss), positions
still open at that moment (how many are winning, how many losing), and the
two combined.

Defaults to MAINNET market data on purpose. Bybit's testnet has its own
thin, synthetic order flow, so paper-trading against it would tell you
nothing about how the strategy behaves on real price action. No key is
sent either way, and no order is ever submitted, so reading mainnet
prices carries no risk to an account.

    python run_paper_bot.py --equity 10 --symbols BTCUSDT
    python run_paper_bot.py --symbols BTCUSDT,ETHUSDT,SOLUSDT
    python run_paper_bot.py --trades-csv session.csv

Read README.md before drawing conclusions from a green session. On the
two years of history in data/, this strategy loses money: -1.38% on 2026
and -4.69% on 2025. A profitable afternoon is a small sample, not an edge.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys

from bot import paper_trading
from bot.config import CONFIG


def main() -> int:
    p = argparse.ArgumentParser(
        description="Paper-trade on live Bybit data with virtual money.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--equity", type=float, default=10.0,
                   help="starting virtual balance in USDT (default: 10)")
    p.add_argument("--symbols", default="BTCUSDT",
                   help="comma-separated symbols (default: BTCUSDT)")
    p.add_argument("--max-positions", type=int, default=1,
                   help="how many positions may be open at once (default: 1)")
    p.add_argument("--poll-seconds", type=int, default=15,
                   help="how often to check open positions (default: 15)")
    p.add_argument("--trades-csv", default=None,
                   help="write every trade to this CSV on shutdown")
    p.add_argument("--testnet", action="store_true",
                   help="read testnet prices instead of real ones (not advised)")
    args = p.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        print("No symbols given.", file=sys.stderr)
        return 2

    config = dataclasses.replace(
        CONFIG,
        symbols=symbols,
        testnet=args.testnet,
        dry_run=True,               # belt and braces: never place an order
        max_concurrent_positions=args.max_positions,
    )

    print("=" * 62)
    print("  PAPER TRADING — virtual money, real Bybit prices")
    print("=" * 62)
    print(f"  Starting balance : ${args.equity:,.2f} (simulated)")
    print(f"  Symbols          : {', '.join(symbols)}")
    print(f"  Max open at once : {args.max_positions}")
    print(f"  Price source     : Bybit {'TESTNET' if args.testnet else 'MAINNET'} "
          f"public API (no key, no orders)")
    print(f"  Checking prices  : every {args.poll_seconds}s")
    print("=" * 62)
    print("  Entries are only taken when price action matches the strategy.")
    print("  It is normal for this to sit idle for hours — on 2026 data it")
    print("  entered 6 times in eight months. Idle is the strategy working,")
    print("  not the bot being broken.")
    print()
    print("  Press Ctrl+C to stop and print the session summary.")
    print("=" * 62)
    print()

    try:
        paper_trading.run(
            starting_equity=args.equity,
            poll_seconds=args.poll_seconds,
            trades_csv=args.trades_csv,
            config=config,
        )
    except KeyboardInterrupt:
        pass
    except Exception as exc:  # network down, Bybit rejecting, DNS, etc.
        print(f"\nStopped on an error: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("If this is a connection error, check that api.bybit.com is "
              "reachable from this machine.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
