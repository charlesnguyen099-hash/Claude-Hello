#!/usr/bin/env python3
"""Paper-trade FINAL_Logic_PotentialScaledLeverage on live Bybit data, virtual $10.

    pip install pandas numpy pybit
    python run_bot.py

No API key, no account, no order is ever placed. It reads Bybit's public
kline and ticker endpoints and keeps the balance in memory.

    python run_bot.py                          # every USDT perpetual on Bybit
    python run_bot.py --top 50                 # only the 50 most liquid
    python run_bot.py --symbols BTCUSDT,ETHUSDT,SOLUSDT
    python run_bot.py --min-votes 2 --exit net_TP2.0_SL1.5
    python run_bot.py --maker-fee --max-leverage 25
    python run_bot.py --margin-pct 5 --max-positions 20

By default it scans EVERY USDT perpetual Bybit lists and trades any coin
whose methods fire and agree -- the scan runs the whole board rather than
stopping once a few positions are open, so a coin far down the list is as
tradeable as one near the top. Klines are fetched in parallel, otherwise
a full pass would take longer than the cycle it belongs to.

It runs continuously and shows continuously. Three threads: one keeps a
standing signal for every coin on the board, one re-prices every open
position each second so TP and SL fire promptly, and one prints the
capital and P&L dashboard on a fixed beat.

Nothing potential is missed. The methods read a 30-minute bar, so a
verdict cannot change until that bar closes -- klines are therefore
fetched once per coin per bar and the verdict is kept standing. The fill
loop then runs continuously over those standing signals and opens each
one the moment there is margin for it, so a signal raised while the book
was full is taken as soon as a position closes rather than lost. Polling
the exchange harder would not find more trades; it would only earn a
rate-limit ban, and a banned bot misses everything.

With no position cap, what limits the bot is free margin: each trade
takes --margin-pct of current equity, and no trade opens without it. You
get slightly fewer than 100/--margin-pct positions, because each entry
fee shrinks equity and so shrinks the next slice.

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
    """Every USDT perpetual, most liquid first. top=0 keeps all of them."""
    r = client.get_tickers(category="linear")
    rows = [x for x in r["result"]["list"] if x["symbol"].endswith("USDT")]
    rows.sort(key=lambda x: float(x.get("turnover24h") or 0), reverse=True)
    names = [x["symbol"] for x in rows]
    return names[:top] if top else names


def main() -> int:
    p = argparse.ArgumentParser(
        description="Paper-trade FINAL_Logic_PotentialScaledLeverage on live Bybit data.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--equity", type=float, default=10.0)
    p.add_argument("--symbols", default=None, help="comma-separated; omit to use --top")
    p.add_argument("--top", type=int, default=0,
                   help="scan only the N most liquid perpetuals "
                        "(default 0 = scan every USDT perpetual)")
    p.add_argument("--max-positions", type=int, default=0,
                   help="cap concurrent positions (default 0 = no cap; "
                        "free margin is the limit)")
    p.add_argument("--margin-pct", type=float, default=5.0,
                   help="percent of equity committed per trade (default 5). "
                        "At the file's leverage a trade that hits its stop "
                        "costs ~42%% of its margin, so this is ~2.1%% of the "
                        "account per trade")
    p.add_argument("--no-expectancy-gate", action="store_true",
                   help="take every consensus signal even where the expected "
                        "value after fees is negative (this is what a 12-hour "
                        "live session did, and it lost more in fees than the "
                        "whole drawdown)")
    p.add_argument("--assumed-win-rate", type=float, default=None,
                   help="win rate the expectancy gate assumes (default: the "
                        "rate this exit actually achieved on 2025+2026)")
    p.add_argument("--conviction-floor", type=float, default=L.CONVICTION_FLOOR,
                   help="smallest fraction of the file's leverage a setup the "
                        "methods barely agreed on may take (default "
                        f"{L.CONVICTION_FLOOR}; 1.0 turns the per-trade haircut "
                        "off). Measured equivalent to a flat deleveraging -- "
                        "see the note below")
    p.add_argument("--max-notional-x", type=float, default=10.0,
                   help="ceiling on total position value as a multiple of "
                        "equity (default 10; 0 disables). This is the only "
                        "bound on correlated risk -- crypto moves together, "
                        "so a full book is one bet, not many")
    p.add_argument("--sizing", default="kelly",
                   choices=["kelly", "potential", "flat"],
                   help="kelly (default): fraction of equity from the Kelly "
                        "criterion on the signal's own 95%% lower-bounded win "
                        "rate -- a rule with a real 70%% record takes 90%% of "
                        "the account, one at the measured 35.1%% takes nothing. "
                        "potential: base slice scaled by return per dollar of "
                        "margin. flat: same slice every trade")
    p.add_argument("--max-margin-pct", type=float,
                   default=100 * L.MAX_MARGIN_FRACTION,
                   help="hard ceiling on one trade's margin as a percent of "
                        f"equity (default {100*L.MAX_MARGIN_FRACTION:.0f})")
    p.add_argument("--flat-sizing", action="store_true",
                   help="give every trade the same margin. By default margin "
                        "scales 0.40-2.50x with the trade's expected return "
                        "per dollar of margin, which spans a factor of ten "
                        "across the ATR range because the fee does")
    p.add_argument("--signals", default="methods",
                   choices=["methods", "patterns"],
                   help="methods = the twelve voting rules on 30m bars; "
                        "patterns = the hard-coded lookup table on 15m bars "
                        "built by `python -m fp.patterns learn`, which "
                        "carries each shape's own record into the leverage")
    p.add_argument("--min-votes", type=int, default=1,
                   help="methods that must fire and agree before a trade")
    p.add_argument("--exit", default=L.DEFAULT_EXIT, choices=L.EXIT_STRATEGIES,
                   help=f"exit fixed at entry (default: {L.DEFAULT_EXIT})")
    p.add_argument("--max-leverage", type=float, default=None,
                   help="cap the potential-scaled leverage (default: no cap, "
                        "so the file's own 17-100x band is used as written)")
    p.add_argument("--entry", default="taker", choices=["maker", "taker"],
                   help="how the position is OPENED: maker 0.020%% (a resting "
                        "limit order, which may not fill) or taker 0.055%% (a "
                        "market order). The EXIT is always taker 0.055%% -- "
                        "TP and SL are conditional market orders and cannot "
                        "earn the maker rate")
    p.add_argument("--funding-rate", type=float, default=L.FUNDING_RATE_TYPICAL,
                   help="funding charged every 8h on notional (default "
                        f"{L.FUNDING_RATE_TYPICAL:.4f} = 0.010%%; a strong "
                        "trend pays 0.100%%). A TP3.0 trade lasts 7.11h on "
                        "average, so it meets 0.89 of these")
    p.add_argument("--slippage", type=float, default=0.0,
                   help="extra cost per side, as a fraction (0.0005 = 0.05%%). "
                        "Zero by default: ~$15 of notional does not move the "
                        "BTCUSDT spread")
    p.add_argument("--maker-fee", action="store_true",
                   help="shorthand for --entry maker")
    p.add_argument("--poll-seconds", type=int, default=5,
                   help="seconds between dashboard reprints (default 5). "
                        "Scanning and position management are continuous "
                        "and are not affected by this.")
    p.add_argument("--trades-csv", default=None)
    p.add_argument("--testnet", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    from pybit.unified_trading import HTTP

    from fp import bot as B
    from fp import methods as M

    client = HTTP(testnet=args.testnet)
    cost = L.round_trip_cost(args.exit, args.maker_fee or args.entry == "maker",
                             args.funding_rate, args.slippage)
    fee = cost["total"]

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
    print(f"  Symbols scanned  : {len(symbols)}"
          + ("  (every USDT perpetual)" if not args.top and not args.symbols else ""))
    print(f"    {', '.join(symbols[:12])}{' ...' if len(symbols) > 12 else ''}")
    print(f"  Max open at once : "
          + ("no cap (limited by free margin)" if not args.max_positions
             else str(args.max_positions)))
    print(f"  Margin per trade : {args.margin_pct:.1f}% of equity "
          f"(${args.equity * args.margin_pct / 100:.2f} at the start, "
          f"~{int(100 / args.margin_pct) - 1} positions max)")
    print(f"  Risk per trade   : ~{args.margin_pct * 0.42:.2f}% of the account "
          f"(a stop costs ~42% of the trade's margin)")
    mode = "flat" if args.flat_sizing else args.sizing
    if mode == "kelly":
        print(f"  Sizing           : Kelly, up to {args.max_margin_pct:.0f}% of "
              f"equity on one trade")
        print(f"    win rate -> stake   35.1% -> 0%    45% -> 32%    "
              f"55% -> 66%    70% -> 90%")
        print(f"    The win rate is a 95% LOWER bound on the signal's own live")
        print(f"    record, not its raw rate. A rule that won 5 of 5 is 56.6%,")
        print(f"    not 100% -- it takes about 100 straight wins to earn the")
        print(f"    ceiling. Kelly on a raw rate is how accounts die.")
        print(f"    A signal with no live record uses the measured 35.1%,")
        print(f"    where Kelly is 0%: it will not be staked at all.")
    elif mode == "potential":
        print(f"  Sizing           : base slice x {L.MARGIN_WEIGHT_MIN:.2f}-"
              f"{L.MARGIN_WEIGHT_MAX:.2f} by return per $ of margin")
    else:
        print(f"  Sizing           : flat, {args.margin_pct:.1f}% every trade")
    print(f"  Exposure ceiling : "
          + ("off -- total position value is unbounded" if not args.max_notional_x
             else f"{args.max_notional_x:.0f}x equity "
                  f"(${args.equity * args.max_notional_x:.2f} at the start)"))
    if args.signals == "patterns":
        from fp import patterns as P
        _lib = P.Library()
        _ok = sum(1 for x in _lib.patterns.values() if x.tradeable())
        _live = sum(1 for x in _lib.patterns.values() if x.n_live > 0)
        print(f"  Signals          : pattern library on {P.BAR_MINUTES}m bars, "
              f"{P.LOOKBACK}-candle lookback")
        print(f"                     {len(_lib.patterns)} entries, {_ok} tradeable, "
              f"{_live} with an out-of-sample record")
        if _live == 0:
            print("                     NOTHING PROVEN YET -- run "
                  "`python -m fp.patterns review` on a")
            print("                     day the library was not built from "
                  "before trusting it")
    else:
        print(f"  Methods          : {len(M.METHOD_NAMES)} voting on 30m bars")
    print(f"  Min votes to open: {args.min_votes} (and they must agree)")
    print(f"  Exit strategy    : {args.exit} (fixed at entry)")
    print(f"  Leverage         : {lev_note}")
    entry_kind = "maker (resting limit)" if (args.maker_fee or args.entry == "maker") else "taker (market)"
    print(f"  Cost per round trip: {fee*100:.3f}% of notional -- everything")
    print(f"    entry   {cost['entry']*100:.3f}%  {entry_kind}")
    print(f"    exit    {cost['exit']*100:.3f}%  taker -- TP/SL are market orders,")
    print(f"                     they cannot earn the maker rate")
    print(f"    funding {cost['funding']*100:.4f}%  {cost['funding_events']:.2f} "
          f"charges over a {cost['hold_hours']:.2f}h average hold")
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
    if args.conviction_floor < 1.0:
        print(f"  That score is the instrument's volatility rank -- identical")
        print(f"  for a long and a short on the same bar. The per-trade term")
        print(f"  is the vote: leverage is then cut to between "
              f"{args.conviction_floor:.2f} and 1.00 of")
        print("  it by how strongly the twelve agreed on THIS setup.")
        print()
        print("  Measured on 15,617 trades over 2025-2026, that agreement does")
        print("  NOT predict outcome (win rate 33.6% at one vote, 33.3% at")
        print("  five; correlation -0.0011, and it flips sign between years).")
        print("  The haircut is equivalent to a flat deleveraging of the same")
        print("  size, to within +0.027%. It is a risk preference, not an")
        print("  edge. --conviction-floor 1.0 turns it off.")
    else:
        print("  Per-trade conviction haircut: OFF (--conviction-floor 1.0).")
    print()
    print("  One further bound, also computed from ATR: leverage is held")
    print("  below 0.9 / (1.3 x 1.5 x atr14_pct), the point where the stop")
    print("  would sit past the liquidation price. Through the normal range")
    print("  it never binds -- only above ~2.7% ATR, where the 17x floor")
    print("  would otherwise make the trade die at 100% of margin instead")
    print("  of the 42% its stop was sized for.")
    print()
    print("  Direction comes from the methods' consensus. The file's own")
    print("  final_direction column agrees with that consensus 50.4% of the")
    print("  time -- it was chosen after the outcome, so it is not available.")
    print("=" * 70)
    print("  Every coin whose methods fire and agree is traded; the scan")
    print("  covers the whole board, not just the first matches.")
    print()
    print("  Runs continuously: signals are kept standing for every coin and")
    print("  filled the moment margin frees, positions are re-priced every")
    print(f"  second, and the dashboard reprints every {args.poll_seconds}s.")
    print(f"  Klines are refetched once per coin per {L.BAR_MINUTES}m bar --")
    print("  the verdict cannot change until the bar closes, and hammering")
    print("  the endpoint would only earn a rate-limit ban.")
    print("=" * 70)
    need = L.min_atr_for_edge(args.exit, fee, args.assumed_win_rate)
    p_win = (L.MEASURED_WIN_RATE.get(args.exit, 0.35)
             if args.assumed_win_rate is None else args.assumed_win_rate)
    print("  EXPECTANCY GATE   " + ("OFF" if args.no_expectancy_gate else "ON"))
    print(f"  A trade is taken only if p*TP*atr - (1-p)*SL*atr - fee > 0,")
    print(f"  with p = {p_win:.1%}, the rate {args.exit} actually achieved on")
    print(f"  BTCUSDT 30m over 2025-2026. At {fee*100:.3f}% that needs")
    print(f"  atr14_pct >= {need:.3f}%.")
    print()
    if need > 2.5:
        print("  WARNING. BTCUSDT 30m never reached that in two years (max")
        print("  2.42%), so at this fee the gate will refuse essentially")
        print("  every trade. That is the correct answer, not a fault: at")
        print("  taker fees this logic has no positive-expectancy trade, and")
        print("  no filter over ATR, votes or anything else changes it.")
        print("  Measured: -0.26%/trade (2025), -0.20% (2026) at EVERY ATR")
        print("  threshold. Use --maker-fee, or --no-expectancy-gate to")
        print("  trade anyway and watch the fee take the account.")
    else:
        print(f"  About a quarter of 30m bars clear {need:.3f}% ATR.")
    print("=" * 70)
    print("  Judging a session shorter than a day: stops resolve in a median")
    print("  5-7 bars (2.5-3.5h), targets in 11-12 (5.5-6h). Inside 12 hours")
    print("  ~90% of the stops have completed but only ~78-88% of the")
    print("  targets, so closed P&L reads worse than the position book is.")
    print("  Read 'incl. open positions', not 'realized only'.")
    print("=" * 70)
    print("  Measured on BTCUSDT 30m over 2025, 2026 and a held-out August:")
    print("  no exit is profitable in every period at any fee tier. Run")
    print("  python -m fp.research for the full walk-forward table.")
    print("=" * 70)
    print("  Ctrl+C stops the bot and prints the session summary.")
    print("=" * 70)
    print()

    try:
        B.run(client, symbols, args.equity, args.max_positions, args.exit,
              args.min_votes, args.poll_seconds, args.trades_csv, fee,
              args.max_leverage, args.margin_pct / 100.0, args.max_notional_x,
              args.conviction_floor, not args.no_expectancy_gate,
              args.assumed_win_rate, args.signals, not args.flat_sizing,
              "flat" if args.flat_sizing else args.sizing,
              args.max_margin_pct / 100.0)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"\nStopped on an error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
