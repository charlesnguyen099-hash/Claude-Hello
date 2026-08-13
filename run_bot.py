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

# OpenBLAS reserves a per-thread scratch buffer for as many threads as it
# thinks the machine has, and numpy/sklearn load it on import. Together
# with the bot's own scanner and manager threads that was enough to end a
# live run with
#
#     OpenBLAS error: Memory allocation still failed after 10 retries
#
# before a single bar was scanned. The model here is six small gradient
# boosters predicting one row at a time -- there is nothing to parallelise
# and nothing to lose by pinning the maths libraries to one thread each.
#
# This MUST run before numpy is imported by anything, so it sits above
# every other import in the file.
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")


import argparse
import logging
import sys

from fp import logic as L


def book_scope(book_file: str) -> list[str]:
    """The symbols the loaded book was actually validated on."""
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parent / "fp" / book_file
    try:
        return list(json.loads(p.read_text()).get("scope", []))
    except Exception:
        return []


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
    p.add_argument("--max-notional-x", type=float, default=0.0,
                   help="ceiling on total position value as a multiple of "
                        "equity. 0 by default = no ceiling: how much to hold "
                        "is decided by each trade's potential, not by a number "
                        "fixed in advance. Set it if you want a hard stop on "
                        "correlated exposure")
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
    p.add_argument("--signals", default="mtf",
                   choices=["mtf", "methods", "slow", "regime", "survivors",
                            "book"],
                   help="mtf (default): the multi-timeframe model. One model "
                        "for all symbols, reading 1m/5m/15m/30m/60m state at "
                        "once and predicting the NET return of a trade opened "
                        "now. It picks the side AND the barrier shape by "
                        "comparing every combination's predicted net, and "
                        "trades only where the walk-forward measured that "
                        "band of prediction actually paying. Walk-forward on "
                        "June: +0.23%%/trade at the 0.002 gate over 124 "
                        "independent trades, t about 1.8 -- promising, not "
                        "proven. book: trade the rules in --book-file, each with its "
                        "OWN entry, side, volatility-scaled target, stop and "
                        "time limit, on its own timeframe. This is what "
                        "fp/btc_book.py and fp/coin_book.py produce, and the "
                        "books are FITTED to the data they were built on. "
                        "survivors: ONLY the logics in fp/survivors.json, "
                        "each of which was tested on its own out of sample, "
                        "cleared a Bonferroni threshold for the number tested, "
                        "and beat a timing-rotation null. If that file is "
                        "empty no position is ever opened -- which is what "
                        "'trade only profitable logics' means when none are. "
                        "regime: direction from the logics that have paid in "
                        "the state this coin is in now, out of 2,602 built "
                        "from ~100 factors x 38 methods -- measured to lose. "
                        "slow: daily trend, position held until "
                        "the trend flips, leverage solved for including "
                        "volatility drag. methods: the twelve voting rules on "
                        "30m bars. This was the DEFAULT until it was measured "
                        "to lose -10.92%% and -6.53%% in two live sessions; it "
                        "is kept only for comparison and must now be asked "
                        "for by name")
    p.add_argument("--book-file", default="book.json",
                   help="which book to trade. The default book.json is every "
                        "rule from every study merged together (python -m "
                        "fp.book rebuilds it), and it is what a bare "
                        "`python run_bot.py` runs. The parts are still there "
                        "if you want one alone: btc_book.json, "
                        "coin_tiers.json, coin_book.json")
    p.add_argument("--mtf-gate", choices=("band", "probe"), default="band",
                   help="band (default): trade only bands MEASURED profitable "
                        "out of sample. None are, so the bot stands still -- "
                        "which is the honest reading, not a fault. probe: "
                        "trade every shape and side at --probe-pct to gather "
                        "a live record. Costs about 1.1%%/day of equity on "
                        "the measured baseline, capped by --probe-budget")
    p.add_argument("--probe-pct", type=float, default=2.0,
                   help="percent of equity a rule stakes while it is still "
                        "gathering its first --probe-n trades (default 2)")
    p.add_argument("--probe-n", type=int, default=30,
                   help="closed trades a rule needs before its own record "
                        "replaces the probe stake (default 30)")
    p.add_argument("--probe-budget", type=float, default=5.0,
                   help="percent of STARTING equity the probe programme may "
                        "lose in total before it stops opening new probes "
                        "(default 5). This is what bounds the experiment")
    p.add_argument("--earn-stake", action="store_true",
                   help="make each rule EARN its way up from --margin-pct to "
                        "--max-margin-pct as it builds a live record. Off by "
                        "default: the stake follows the setup in front of it, "
                        "not the record of the ones behind it")
    p.add_argument("--stake-curve", choices=("linear", "square"),
                   default="linear",
                   help="linear (default): stake%% = score, so a 60/100 setup "
                        "commits 60%% of equity. square: stake%% = score^2/100, "
                        "the conservative Kelly answer for an ESTIMATED edge")
    p.add_argument("--share-stakes", action="store_true",
                   help="when several setups fire at once, scale them all "
                        "down so the whole standing set fits. Off by default: "
                        "fill in score order, each taking the full stake its "
                        "score asked for, until free margin runs out")
    p.add_argument("--trust-book", action="store_true",
                   help="let the potential score run the full "
                        "--max-margin-pct from the first trade, instead of "
                        "earning up from --margin-pct as each rule builds a "
                        "live record. book.json is FITTED to its own data, so "
                        "this is the operator's call, not the default")
    p.add_argument("--book-anywhere", action="store_true",
                   help="let every book rule fire on every symbol scanned, "
                        "ignoring the symbols it was validated on. Off by "
                        "default: a rule proven on four symbols has evidence "
                        "for those four, and running it elsewhere is an "
                        "untested claim wearing tested numbers")
    p.add_argument("--min-votes", type=int, default=1,
                   help="methods that must fire and agree before a trade")
    p.add_argument("--exit", default=L.DEFAULT_EXIT, choices=L.EXIT_STRATEGIES,
                   help=f"exit fixed at entry (default: {L.DEFAULT_EXIT})")
    p.add_argument("--max-leverage", type=float, default=None,
                   help="cap the potential-scaled leverage (default: no cap, "
                        "so the file's own 17-100x band is used as written)")
    p.add_argument("--limit-entry", action="store_true",
                   help="assume the entry rests as a limit order and earns "
                        "the 0.020%% maker rate instead of paying 0.055%% "
                        "taker. Off by default because the bot sends market "
                        "orders, and a resting order may simply not fill. "
                        "The EXIT is taker either way -- TP and SL are "
                        "conditional market orders")
    p.add_argument("--slippage", type=float, default=0.0,
                   help="extra cost per side, as a fraction (0.0005 = 0.05%%). "
                        "Zero by default: ~$15 of notional does not move the "
                        "BTCUSDT spread")
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
    # Real costs, taken from the exchange. Entry and exit are both taker
    # unless --limit-entry, and funding is each symbol's own live rate
    # pulled from the ticker feed at trade time -- there is nothing to
    # choose here, and nothing to guess.
    cost = L.round_trip_cost(args.exit, args.limit_entry,
                             L.FUNDING_RATE_TYPICAL, args.slippage)
    fee = cost["total"]

    scope = book_scope(args.book_file) if args.signals == "book" else []
    if args.signals == "mtf":
        # The model was trained on these ten symbols. It carries no
        # per-rule scope, so the training universe IS the scope: firing
        # it on a symbol whose behaviour was never in the training set is
        # the same untested claim the book's scope gate exists to stop.
        import json as _j
        from pathlib import Path as _P
        try:
            scope = sorted(_j.loads(
                (_P(__file__).resolve().parent / "data" /
                 "mtf_scope.json").read_text()))
        except Exception:
            scope = book_scope(args.book_file)
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif scope and not args.top and not args.book_anywhere:
        # The book's rules were validated on these symbols and are scoped
        # to them, so scanning the other 680 perpetuals would fetch klines
        # for symbols no rule is allowed to fire on. --top or
        # --book-anywhere widens it deliberately.
        symbols = scope
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
    if args.signals not in ("book", "mtf"):
        print(f"  Margin per trade : {args.margin_pct:.1f}% of equity "
              f"(${args.equity * args.margin_pct / 100:.2f} at the start, "
              f"~{int(100 / args.margin_pct) - 1} positions max)")
        print(f"  Risk per trade   : ~{args.margin_pct * 0.42:.2f}% of the "
              f"account (a stop costs ~42% of the trade's margin)")
    else:
        # Both of these solve the stake per setup. A single "margin per
        # trade" number would be a fiction, and printing one next to a
        # 0-100% stake curve made the banner contradict itself.
        print(f"  Margin per trade : NOT fixed -- solved per setup, "
              f"0% to {args.max_margin_pct:.0f}% of equity")
    mode = "flat" if args.flat_sizing else args.sizing
    if args.signals == "mtf":
        import json as _j
        from pathlib import Path as _P
        try:
            _m = _j.loads((_P(__file__).resolve().parent / "fp" /
                           "mtf_model.json").read_text())
        except Exception:
            _m = {}
        print(f"  Signals          : multi-timeframe model, ONE model for all "
              f"symbols")
        print(f"                     views {_m.get('views')} minutes, "
              f"{len(_m.get('columns', []))} features, entry on "
              f"{_m.get('entry_minutes')}m bars")
        print(f"                     trained on {_m.get('train_rows', 0):,} rows, "
              f"{_m.get('train_window', ['?','?'])[0]}.."
              f"{_m.get('train_window', ['?','?'])[1]}")
        if args.mtf_gate == "probe":
            print(f"  Side and exit    : every barrier shape on BOTH sides "
                  f"gets its own slot, so a")
            print(f"                     symbol can carry six at once. The "
                  f"prediction only ORDERS")
            print(f"                     them -- out of sample it does not "
                  f"predict the outcome.")
        else:
            print(f"  Side and exit    : the model scores every barrier shape "
                  f"on BOTH sides and takes")
            print(f"                     the highest predicted net. Nothing "
                  f"about the trade is fixed.")
        try:
            _b = _j.loads((_P(__file__).resolve().parent / "fp" /
                           "mtf_bands.json").read_text())
        except Exception:
            _b = []
        print(f"  Bands OUT OF SAMPLE: refit on 05-31..07-20, scored on "
              f"07-20..08-14 -- bars")
        print(f"                     the model never saw. This replaces the "
              f"in-sample table.")
        for g in _b:
            lo = "-inf" if g["lo"] < -1 else f"{g['lo']:.3f}"
            hi = "+inf" if g["hi"] > 1 else f"{g['hi']:.3f}"
            mark = ("PAYS   " if g["edge_over_b"] > 0 and g["t"] >= 2.0
                    else "no    ")
            print(f"      {mark} {lo:>6}..{hi:<6} {g['edge_over_b']:+.3f} of a "
                  f"win, {g['trades']:>4} trades, t={g['t']:+.2f}")
        _pay = [g for g in _b if g["edge_over_b"] > 0 and g["t"] >= 2.0]
        print(f"  REFUTED          : the shipped gate claimed +0.406 of a win "
              f"at t=+2.76 on 34")
        print(f"                     in-sample trades. Out of sample the same "
              f"band paid +0.102")
        print(f"                     at t=+0.42, and trading it lost "
              f"0.678%/trade -- worse than")
        print(f"                     a random rotation of its own "
              f"predictions (p=0.995).")
        print(f"                     Rank correlation prediction vs outcome "
              f"over 24 OOS days:")
        print(f"                     -0.0096. There is no relationship. "
              f"Run python -m fp.verdict.")
        print(f"                     {len(_pay)} of {len(_b)} bands are "
              f"tradeable on that evidence.")
        if args.mtf_gate == "probe":
            print(f"  Gate             : PROBE. No in-sample number sizes "
                  f"anything. Every shape")
            print(f"                     and side trades {args.probe_pct:.1f}% "
                  f"of equity until it has")
            print(f"                     {args.probe_n} closed trades of its "
                  f"OWN, then its own record")
            print(f"                     -- and only that -- decides its "
                  f"stake.")
            print(f"                     A rule more than 2 standard errors "
                  f"below zero is RETIRED")
            print(f"                     and stops trading.")
            print(f"    COST: on the measured baseline every untimed barrier "
                  f"trade loses")
            print(f"    0.1136% of notional (6,140 independent trades, "
                  f"t = -13.9). Probing at")
            print(f"    {args.probe_pct:.1f}% and a few x leverage that is "
                  f"roughly -1.1%/day of equity.")
            print(f"    Probing STOPS once it has lost "
                  f"{args.probe_budget:.1f}% of the starting balance.")
            print(f"    You are paying that to find out whether the market "
                  f"has changed. It is")
            print(f"    a real cost and the data says it will probably not "
                  f"be repaid.")
        else:
            print(f"  Gate             : BAND (default). Only bands measured "
                  f"profitable trade.")
            print(f"                     Out of sample {len(_pay)} of "
                  f"{len(_b)} qualify, so THE BOT WILL NOT")
            print(f"                     OPEN ANYTHING. That is the honest "
                  f"reading of the evidence,")
            print(f"                     not a malfunction. Every entry rule "
                  f"tested on this data")
            print(f"                     loses about the fee: the model "
                  f"(rho -0.0096), six classic")
            print(f"                     effects (best t = +0.85 against a "
                  f"3.40 bar, python -m")
            print(f"                     fp.simple), and untimed entry "
                  f"(-0.1136%/trade).")
            print(f"                     --mtf-gate probe trades anyway, to "
                  f"gather live evidence,")
            print(f"                     at a measured cost of about "
                  f"-1.1%/day.")

    if args.signals == "book":
        # Measured, not guessed. See fp/aug_tiers.py.
        print("  MEASURED         : replayed on 2026-08-05..08-07 (34h, the")
        print("                     live session's own bars), the book's entry")
        print("                     timing scored WORSE than entering at a")
        print("                     random bar: -0.029%/trade, t = -4.60.")
        print("                     21% of its claims are refuted at that")
        print("                     window's real sigma and are not traded.")
        # Every book rule brings its own edge, its own target and its own
        # stop, so every rule solves its own stake. Half-Kelly, because the
        # edge is an estimate: full Kelly is optimal only when it is known.
        print(f"  Sizing           : each rule's own half-Kelly, from its own "
              f"measured edge")
        print(f"                     and its own barriers -- 0% up to "
              f"{args.max_margin_pct:.0f}% of equity on one trade")
        print(f"    A rule's claim is discounted by how much of the book's "
              f"edge has actually")
        print(f"    turned up live, and blended with that rule's own record "
              f"as it accumulates.")
        print(f"    One position PER RULE, so a 6-day daily rule no longer "
              f"locks its symbol.")
        print(f"  Potential scale  : 100 x edge/b -- the credible edge as a "
              f"share of what a win pays.")
        print(f"                     0 = break-even, 100 = cannot lose by its "
              f"own barriers.")
        if args.stake_curve == "linear":
            print(f"  Stake            : score, read as a percent of equity. "
                  f"Capped at {args.max_margin_pct:.0f}%.")
            print(f"      score  20 -> 20%     40 -> 40%     60 -> 60%     "
                  f"80 -> 80%    100 -> ALL IN")
        else:
            print(f"  Stake            : score^2/100 percent of equity, "
                  f"capped at {args.max_margin_pct:.0f}%.")
            print(f"      score  20 ->  4%     40 -> 16%     60 -> 36%     "
                  f"80 -> 64%    100 -> ALL IN")
        if args.earn_stake:
            print(f"    --earn-stake: a rule cannot exceed "
                  f"{args.margin_pct:.0f}% until it has a live record.")
        else:
            print(f"    Sizing reads ONLY the setup in front of it. A rule's "
                  f"own past does not")
            print(f"    shrink or grow its stake -- but a rule losing more "
                  f"than 2 standard errors")
            print(f"    below zero on its own record still stops trading "
                  f"entirely.")
    elif args.signals == "mtf":
        # mtf shares the book's potential score and stake curve; it does
        # NOT use the ATR ladder's Kelly-on-live-record, which is what
        # this chain used to fall through to and print.
        if args.mtf_gate == "probe":
            print(f"  Sizing           : {args.probe_pct:.1f}% while probing, "
                  f"then the rule's own live")
            print(f"                     record, 0% up to "
                  f"{args.max_margin_pct:.0f}% of equity")
        else:
            print(f"  Sizing           : the setup's own potential score, "
                  f"0% up to {args.max_margin_pct:.0f}% of equity")
        print(f"  Potential scale  : 100 x edge/b -- the credible edge as a "
              f"share of what a win")
        print(f"                     pays. 0 = break-even, 100 = cannot lose "
              f"by its own barriers.")
        if args.mtf_gate == "probe":
            print(f"                     The edge is the rule's OWN live mean. "
                  f"No other source.")
        if args.stake_curve == "linear":
            print(f"  Stake            : score, read as a percent of equity. "
                  f"Capped at {args.max_margin_pct:.0f}%.")
            print(f"      score  20 -> 20%     40 -> 40%     60 -> 60%     "
                  f"80 -> 80%    100 -> ALL IN")
        else:
            print(f"  Stake            : score^2/100 percent of equity, "
                  f"capped at {args.max_margin_pct:.0f}%.")
            print(f"      score  20 ->  4%     40 -> 16%     60 -> 36%     "
                  f"80 -> 64%    100 -> ALL IN")
        print(f"    Half-Kelly and a ruin cap apply on top, so a big score "
              f"on a wide stop")
        print(f"    still cannot bet the account.")
        if args.mtf_gate == "probe":
            print(f"    A rule DOES read its own past -- that is the only "
                  f"evidence there is. It")
            print(f"    does not read any other rule's, and no rule reads the "
                  f"in-sample table.")
        else:
            print(f"    Nothing reads the previous trade's result.")
        print(f"  Slots            : one position per SYMBOL x RULE, so a "
              f"symbol can carry")
        print(f"                     several shapes at once instead of being "
              f"locked by the first.")
    elif args.signals in ("survivors", "slow", "regime"):
        # These modes size flat on purpose: the potential already lives in
        # the leverage, and Kelly's win-rate input is the record of the
        # twelve-method exit -- a statistic about a different rule.
        print(f"  Sizing           : flat, {args.margin_pct:.1f}% every trade "
              f"(the rule's own leverage carries its potential)")
    elif mode == "kelly":
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
          + ("none -- position count, notional and leverage all follow the "
             "market" if not args.max_notional_x
             else f"{args.max_notional_x:.0f}x equity "
                  f"(${args.equity * args.max_notional_x:.2f} at the start)"))
    if args.signals == "book":
        import json as _json
        from pathlib import Path as _P
        bf = _P(__file__).resolve().parent / "fp" / args.book_file
        try:
            _b = _json.loads(bf.read_text())
            _rules = _b.get("logics", [])
        except Exception:
            _b, _rules = {}, []
        tfs = sorted({r.get("tf", "?") for r in _rules})
        longs = sum(1 for r in _rules if r.get("side") == "long")
        print(f"  Logic            : {args.book_file} — {len(_rules)} rules "
              f"on {', '.join(tfs) if tfs else 'nothing'} "
              f"({longs} long, {len(_rules)-longs} short)")
        print(f"    fitted on      : {_b.get('fitted_on', 'unknown')}")
        print(f"    Each rule brings its OWN entry, side, target and stop in")
        print(f"    units of the volatility at entry, and its own time limit.")
        print(f"    Nothing from the twelve-method design touches these trades")
        print(f"    -- fp/test_bot.py fails if it does.")
        if _b.get("scope"):
            if args.book_anywhere:
                print(f"    --book-anywhere: rules fire on ANY scanned symbol,")
                print(f"    including {len(symbols)} never used to validate them.")
            else:
                print(f"    Each rule fires ONLY on the symbols it was")
                print(f"    validated on. Use --book-anywhere to lift that.")
        if _b.get("in_sample"):
            print(f"    WARNING: this book is FITTED to the data above. It is")
            print(f"    what worked there, not a forecast. Forward performance")
            print(f"    is unknown until it runs on data it has never seen.")
        if not _rules:
            print(f"    EMPTY — no position will be opened. Build one with")
            print(f"    python -m fp.btc_book   or   python -m fp.coin_book")
    print(f"  Still enforced   : the stop must sit inside the liquidation "
          f"price, now")
    print(f"                     measured on the rule's OWN stop distance "
          f"rather than an")
    print(f"                     ATR multiple, and refused if it does not fit")
    if args.signals in ("slow", "regime"):
        from fp import slow as S
        print(f"  Signals          : slow trend, MA{S.TREND_LOOKBACK} on "
              f"{S.TREND_BAR_MINUTES//60}h bars")
        print(f"  Exit             : held until the trend flips -- no TP, no SL,")
        print(f"                     no re-entry. Turnover is what pays drag.")
        print(f"  Leverage         : solved per trade over {S.LEVERAGE_MIN:.0f}-"
              f"{S.LEVERAGE_MAX:.0f}x, maximising")
        print(f"                     move x L - costs x L - drag x L^2. A thin")
        print(f"                     move over a long hold solves to 0 and is")
        print(f"                     not taken.")
        print(f"    Why not 17-100x: on this data a PERFECTLY correct short")
        print(f"    returned +14.1% held as one position and -274.6% at 10x")
        print(f"    rebalanced. Drag scales with L squared. Run python -m fp.slow.")
    elif args.signals != "mtf":
        print(f"  Methods          : {len(M.METHOD_NAMES)} voting on 30m bars")
    if args.signals == "mtf":
        # No vote count, no fixed exit, no ATR leverage ladder: the model
        # picks the side, the barrier shape carries the exit, and each
        # trade solves its own leverage from its own stop distance.
        print(f"  Exit             : the barrier shape the model chose -- its "
              f"own target, stop")
        print(f"                     and time limit, in units of the "
              f"volatility at entry")
    else:
        print(f"  Min votes to open: {args.min_votes} (and they must agree)")
        print(f"  Exit strategy    : {args.exit} (fixed at entry)")
        print(f"  Leverage         : {lev_note}")
    entry_kind = "maker (resting limit)" if args.limit_entry else "taker (market order)"
    print(f"  Cost per round trip: {fee*100:.3f}% of notional -- everything")
    print(f"    entry   {cost['entry']*100:.3f}%  {entry_kind}")
    print(f"    exit    {cost['exit']*100:.3f}%  taker -- TP/SL are market orders,")
    print(f"                     they cannot earn the maker rate")
    print(f"    funding {cost['funding']*100:.4f}%  {cost['funding_events']:.2f} "
          f"charges over a {cost['hold_hours']:.2f}h average hold")
    print(f"                     -- but the LIVE per-symbol rate off the ticker")
    print(f"                     feed is what each trade is actually charged,")
    print(f"                     signed: a short COLLECTS positive funding")
    print(f"  Price source     : Bybit {'TESTNET' if args.testnet else 'MAINNET'} "
          f"public API (no key, no orders)")
    print("=" * 70)
    if args.signals == "mtf":
        # Everything below this point describes the twelve-method design:
        # its method list, its ATR leverage ladder and its expectancy gate.
        # mtf uses none of them, and printing them made a live banner
        # contradict itself twice in the same screen.
        print("  Ctrl+C stops the bot and prints the session summary.")
        print("=" * 70)
        return _run(args, client, symbols, fee)
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

    return _run(args, client, symbols, fee)


def _run(args, client, symbols, fee) -> int:
    """Start the bot. Split out so a mode with its own banner can skip the
    twelve-method sections and still launch by the same path."""
    try:
        B.run(client, symbols, args.equity, args.max_positions, args.exit,
              args.min_votes, args.poll_seconds, args.trades_csv, fee,
              args.max_leverage, args.margin_pct / 100.0, args.max_notional_x,
              args.conviction_floor, not args.no_expectancy_gate,
              args.assumed_win_rate, args.signals, not args.flat_sizing,
              "flat" if args.flat_sizing else args.sizing,
              args.max_margin_pct / 100.0, args.limit_entry, args.slippage,
              None, args.book_file, args.book_anywhere, args.trust_book,
              args.earn_stake, args.stake_curve, args.share_stakes,
              args.mtf_gate, args.probe_pct, args.probe_n,
              args.probe_budget / 100.0)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"\nStopped on an error: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
