#!/usr/bin/env python3
"""Paper-trade the Sheet16_Logic_Final logic on live Bybit data.

    pip install pandas numpy pybit
    python run_bot.py

No API key, no account, no order is ever placed. It reads Bybit's public
kline and ticker endpoints and keeps the balance in memory.

    python run_bot.py --equity 10
    python run_bot.py --top 30 --max-positions 3
    python run_bot.py --symbols BTCUSDT,ETHUSDT,SOLUSDT
    python run_bot.py --trades-csv session.csv

A symbol is traded only where its market matches a cell the table had a
verdict on; symbols that never match are never traded. Ctrl+C prints the
session summary: closed trades (winners, losers, liquidations, total
profit, total loss), positions still open, and the two combined.

The table specifies 100x leverage and a 0.250% round-trip fee, and the bot
uses both verbatim rather than substituting safer numbers. At 100x, a 0.9%
adverse move liquidates the position — the table's own trades survive that
because they enter at the exact pivot (median MAE 0.030%), which a live
entry cannot do.
"""
from __future__ import annotations

import argparse
import logging
import sys

TABLE = ("/root/.claude/uploads/2499e73f-5145-5c6f-b255-816732633901/"
         "3c7b903e-Sheet16_Logic_Final_23214.txt")


def discover_symbols(client, top: int) -> list[str]:
    r = client.get_tickers(category="linear")
    rows = [x for x in r["result"]["list"] if x["symbol"].endswith("USDT")]
    rows.sort(key=lambda x: float(x.get("turnover24h") or 0), reverse=True)
    return [x["symbol"] for x in rows[:top]]


def main() -> int:
    p = argparse.ArgumentParser(
        description="Paper-trade the Sheet16_Logic_Final logic on live Bybit data.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--equity", type=float, default=10.0)
    p.add_argument("--symbols", default=None, help="comma-separated; omit to use --top")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--max-positions", type=int, default=3)
    p.add_argument("--poll-seconds", type=int, default=15)
    p.add_argument("--table", default=TABLE)
    p.add_argument("--trades-csv", default=None)
    p.add_argument("--testnet", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    from pybit.unified_trading import HTTP

    from pl import bot as B
    from pl import logic as L

    client = HTTP(testnet=args.testnet)
    logic = L.Logic(args.table)

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        try:
            symbols = discover_symbols(client, args.top)
        except Exception as exc:
            print(f"Could not list symbols from Bybit: {exc}", file=sys.stderr)
            return 1

    print("=" * 66)
    print("  PAPER TRADING — virtual money, real Bybit prices")
    print("=" * 66)
    print(f"  Starting balance : ${args.equity:,.2f} (simulated)")
    print(f"  Symbols scanned  : {len(symbols)}")
    print(f"    {', '.join(symbols[:12])}{' ...' if len(symbols) > 12 else ''}")
    print(f"  Max open at once : {args.max_positions}")
    print(f"  Leverage         : {L.LEVERAGE}x (from the table)")
    print(f"  Fee              : {L.FEE_ROUND_TRIP_PCT*100:.3f}% round trip "
          f"(from the table)")
    print(f"  Liquidation at   : {L.LIQUIDATION_MOVE*100:.2f}% adverse move")
    print(f"  Price source     : Bybit {'TESTNET' if args.testnet else 'MAINNET'} "
          f"public API (no key, no orders)")
    print("=" * 66)
    print(f"  {logic.describe()}")
    print("=" * 66)
    print("  Ctrl+C stops the bot and prints the session summary.")
    print("=" * 66)
    print()

    try:
        B.run(client, logic, symbols, args.equity, args.max_positions,
              args.poll_seconds, args.trades_csv)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"\nStopped on an error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
