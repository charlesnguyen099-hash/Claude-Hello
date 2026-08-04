#!/usr/bin/env python3
"""Paper-trade FINAL_Logic_PotentialScaledLeverage on live Bybit data, virtual $10.

    pip install pandas numpy pybit
    python run_bot.py

No API key, no account, no order is ever placed. It reads Bybit's public
kline and ticker endpoints and keeps the balance in memory.

    python run_bot.py --equity 10 --top 20 --max-positions 3
    python run_bot.py --min-votes 2 --exit net_TP2.0_SL1.5
    python run_bot.py --symbols BTCUSDT,ETHUSDT,SOLUSDT --trades-csv session.csv
    python run_bot.py --maker-fee --max-leverage 25

Twelve methods vote on each 30-minute bar; a position opens only where
enough of them fire and agree on direction. Leverage runs the file's full
potential chain: base = 28/atr14_pct clipped to 17-100x, potential score
= the volatility percentile, multiplier = score + 0.5, and the product
capped at the base. Exit defaults to TP3.0/SL1.5, the file's own choice
in 67.7% of its rows.

Ctrl+C prints closed trades (winners, losers, take-profits, stop-outs,
liquidations, total profit, total loss), positions still open with the
methods that opened them, and the two combined.

WHAT THE FILE'S COLUMNS DO AND DO NOT SUPPORT. Its final_direction is not
the methods' verdict: on the 19,717 rows where the twelve did reach a
consensus, final_direction agrees 9,930 times and reverses it 9,787 --
50.4% to 49.6%, a coin flip. It is whichever way the trade turned out to
work, so no bot can compute it. This one trades consensus_dir instead,
which is what the methods actually produce.
"""
from __future__ import annotations

import argparse
import logging
import sys

from fp import logic as L


def discover_symbols(client, top: int) -> list[str]:
    r = client.get_tickers(category="linear")
    rows = [x for x in r["result"]["list"] if x["symbol"].endswith("USDT")]
    rows.sort(key=lambda x: float(x.get("turnover24h") or 0), reverse=True)
    return [x["symbol"] for x in rows[:top]]


def main() -> int:
    p = argparse.ArgumentParser(
        description="Paper-trade FINAL_Logic_PotentialScaledLeverage on live Bybit data.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--equity", type=float, default=10.0)
    p.add_argument("--symbols", default=None, help="comma-separated; omit to use --top")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--max-positions", type=int, default=3)
    p.add_argument("--min-votes", type=int, default=1,
                   help="methods that must fire and agree before a trade")
    p.add_argument("--exit", default=L.DEFAULT_EXIT, choices=L.EXIT_STRATEGIES,
                   help=f"exit fixed at entry (default: {L.DEFAULT_EXIT})")
    p.add_argument("--max-leverage", type=float, default=None,
                   help="cap the potential-scaled leverage (default: no cap, "
                        "so the file's own 17-100x band is used as written)")
    p.add_argument("--maker-fee", action="store_true",
                   help="assume resting limit orders (0.040%% instead of 0.250%%)")
    p.add_argument("--poll-seconds", type=int, default=15)
    p.add_argument("--trades-csv", default=None)
    p.add_argument("--testnet", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    from pybit.unified_trading import HTTP

    from fp import bot as B
    from fp import methods as M

    client = HTTP(testnet=args.testnet)
    fee = L.MAKER_ROUND_TRIP if args.maker_fee else L.FEE_ROUND_TRIP

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        try:
            symbols = discover_symbols(client, args.top)
        except Exception as exc:
            print(f"Could not list symbols from Bybit: {exc}", file=sys.stderr)
            return 1

    lev_note = (f"potential-scaled {L.LEVERAGE_MIN:.0f}-{L.LEVERAGE_MAX:.0f}x"
                if args.max_leverage is None
                else f"potential-scaled, capped at {args.max_leverage:.0f}x")

    print("=" * 70)
    print("  PAPER TRADING — virtual money, real Bybit prices")
    print("=" * 70)
    print(f"  Starting balance : ${args.equity:,.2f} (simulated)")
    print(f"  Symbols scanned  : {len(symbols)}")
    print(f"    {', '.join(symbols[:12])}{' ...' if len(symbols) > 12 else ''}")
    print(f"  Max open at once : {args.max_positions}")
    print(f"  Methods          : {len(M.METHOD_NAMES)} voting on 30m bars")
    print(f"  Min votes to open: {args.min_votes} (and they must agree)")
    print(f"  Exit strategy    : {args.exit} (fixed at entry)")
    print(f"  Leverage         : {lev_note}")
    print(f"  Fee              : {fee*100:.3f}% round trip "
          f"({'maker, resting orders' if args.maker_fee else 'taker'})")
    print(f"  Price source     : Bybit {'TESTNET' if args.testnet else 'MAINNET'} "
          f"public API (no key, no orders)")
    print("=" * 70)
    for m in M.METHOD_NAMES:
        print(f"    {m}")
    print("=" * 70)
    print("  Leverage = min(lev_base x (potential_score + 0.5), lev_base),")
    print("  with lev_base = 28/atr14_pct clipped to 17-100x. The multiplier")
    print("  can only cut leverage, never raise it above what the stop allows.")
    print()
    print("  Direction comes from the methods' consensus. The file's own")
    print("  final_direction column agrees with that consensus 50.4% of the")
    print("  time -- it was chosen after the outcome, so it is not available.")
    print("=" * 70)
    print("  Ctrl+C stops the bot and prints the session summary.")
    print("=" * 70)
    print()

    try:
        B.run(client, symbols, args.equity, args.max_positions, args.exit,
              args.min_votes, args.poll_seconds, args.trades_csv, fee,
              args.max_leverage)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"\nStopped on an error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
