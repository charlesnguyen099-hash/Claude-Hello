#!/usr/bin/env python3
"""Paper-trade the PureLogic rule on live Bybit data with a virtual $10.

    pip install pandas numpy pybit
    python run_pl_bot.py

No API key, no account, no order is ever placed. It reads Bybit's public
kline and ticker endpoints and keeps the balance in memory.

It scans every symbol given and opens a position only where the market
matches a cell the uploaded table had a verdict on. Symbols that never
match are simply never traded.

    python run_pl_bot.py --equity 10
    python run_pl_bot.py --symbols BTCUSDT,ETHUSDT,SOLUSDT --max-positions 3
    python run_pl_bot.py --top 30            # 30 highest-turnover perpetuals
    python run_pl_bot.py --trades-csv session.csv

Ctrl+C prints closed trades (winners, losers, liquidations, total profit,
total loss), positions still open at that moment, and the two combined.

Before reading a green session as proof: backtested on the 2025-2026 data
this rule came from, its edge before costs is -0.0030% (2025) and +0.0055%
(2026) per trade against a 0.210% round trip. It loses. A profitable hour
is a small sample.
"""
from __future__ import annotations

import argparse
import logging
import sys

TABLE = ("/root/.claude/uploads/2499e73f-5145-5c6f-b255-816732633901/"
         "70587060-Sheet16_PureLogic_MaxLeverage_MaxFee_24590.txt")


def discover_symbols(client, top: int) -> list[str]:
    """The `top` USDT perpetuals by 24h turnover."""
    r = client.get_tickers(category="linear")
    rows = [x for x in r["result"]["list"] if x["symbol"].endswith("USDT")]
    rows.sort(key=lambda x: float(x.get("turnover24h") or 0), reverse=True)
    return [x["symbol"] for x in rows[:top]]


def main() -> int:
    p = argparse.ArgumentParser(
        description="Paper-trade the PureLogic rule on live Bybit data.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--equity", type=float, default=10.0)
    p.add_argument("--symbols", default=None,
                   help="comma-separated; omit to use --top")
    p.add_argument("--top", type=int, default=20,
                   help="scan the N highest-turnover USDT perpetuals (default 20)")
    p.add_argument("--max-positions", type=int, default=3)
    p.add_argument("--leverage", type=int, default=None,
                   help="default: the highest that survives the measured "
                        "adverse move")
    p.add_argument("--poll-seconds", type=int, default=15)
    p.add_argument("--table", default=TABLE)
    p.add_argument("--trades-csv", default=None)
    p.add_argument("--testnet", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    from pybit.unified_trading import HTTP

    from pl import paper as P
    from pl import strategy as S

    client = HTTP(testnet=args.testnet)
    logic = S.PureLogic(args.table)
    leverage = args.leverage or S.safe_leverage()

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        try:
            symbols = discover_symbols(client, args.top)
        except Exception as exc:
            print(f"Could not list symbols from Bybit: {exc}", file=sys.stderr)
            return 1

    print("=" * 64)
    print("  PURELOGIC PAPER TRADING — virtual money, real Bybit prices")
    print("=" * 64)
    print(f"  Starting balance : ${args.equity:,.2f} (simulated)")
    print(f"  Symbols scanned  : {len(symbols)}")
    print(f"    {', '.join(symbols[:12])}{' ...' if len(symbols) > 12 else ''}")
    print(f"  Max open at once : {args.max_positions}")
    print(f"  Leverage         : {leverage}x")
    print(f"  Price source     : Bybit {'TESTNET' if args.testnet else 'MAINNET'} "
          f"public API (no key, no orders)")
    print("=" * 64)
    print(f"  Rule: {logic.describe()}")
    print("=" * 64)
    print("  A symbol is traded only when it matches a cell the table had a")
    print("  verdict on. Long idle stretches are the filter working.")
    print()
    print("  Ctrl+C stops the bot and prints the session summary.")
    print("=" * 64)
    print()

    try:
        P.run(client, logic, symbols, args.equity, leverage,
              args.max_positions, args.poll_seconds, args.trades_csv)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"\nStopped on an error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
