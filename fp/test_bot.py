"""Tests for the paper-trading broker.

Run:  python -m fp.test_bot

These cover the four promises the bot makes, because each of them was
broken at some point in development and only a test catches it:

  * signals are refreshed once per 30m bar and not once per loop
    (the naive version made 125 kline calls a second and would be
    rate-limited off the exchange)
  * a symbol the methods pass over is not refetched forever
  * a signal blocked by margin stays standing and is filled the moment a
    slot frees -- that is what "miss no potential trade" means in practice
  * TP, SL and liquidation close at the right price with the right P&L,
    and the summary's counts add up

No network: every test drives a fake client.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from fp import bot as B
from fp import logic as L

BAR_MS = B.BAR_MS
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


def make_rows(n=400, base=100.0, drift=0.0, seed=0, newest_bar=None):
    """Bybit-shaped kline rows: newest first, aligned to the half hour."""
    r = np.random.default_rng(seed)
    close = base * np.exp(np.cumsum(r.normal(drift, 0.01, n)))
    spread = np.abs(r.normal(0, 0.004, n)) + 1e-4
    high, low = close * (1 + spread), close * (1 - spread)
    op = np.concatenate([[close[0]], close[:-1]])
    vol = np.abs(r.normal(1e6, 3e5, n))
    if newest_bar is None:
        newest_bar = (int(time.time() * 1000) // BAR_MS) * BAR_MS
    ts = newest_bar - np.arange(n)[::-1] * BAR_MS
    return [[str(ts[i]), f"{op[i]:.8f}", f"{high[i]:.8f}", f"{low[i]:.8f}",
             f"{close[i]:.8f}", f"{vol[i]:.2f}", "0"] for i in range(n)][::-1]


class FakeHTTP:
    def __init__(self, symbols, price=100.0, newest_bar=None):
        self.symbols = symbols
        self.rows = {s: make_rows(seed=i, base=price, newest_bar=newest_bar)
                     for i, s in enumerate(symbols)}
        self.last = dict.fromkeys(symbols, price)
        self.kline_calls = 0
        self.ticker_calls = 0

    def get_kline(self, category, symbol, interval, limit):
        self.kline_calls += 1
        return {"result": {"list": self.rows[symbol][:limit]}}

    def get_tickers(self, category, symbol=None):
        self.ticker_calls += 1
        if symbol:
            return {"result": {"list": [{"symbol": symbol,
                                         "lastPrice": str(self.last[symbol])}]}}
        return {"result": {"list": [{"symbol": s, "lastPrice": str(p),
                                     "turnover24h": "1000"}
                                    for s, p in self.last.items()]}}


def broker_for(symbols, client, **kw):
    opts = dict(equity=10.0, max_positions=0, exit_name=L.DEFAULT_EXIT,
                min_votes=1, fee=L.FEE_ROUND_TRIP, max_leverage=None,
                margin_pct=0.10, max_notional_x=0.0, conviction_floor=1.0,
                expectancy_gate=False)
    opts.update(kw)
    return B.Broker(client, symbols, **opts)


# --------------------------------------------------------------------------

def test_closed_bar_alignment():
    print("\nclosed_bar_ts")
    now = 1_800_000_000_000
    t = B.closed_bar_ts(now)
    check("aligned to the half hour", t % BAR_MS == 0, f"got {t}")
    check("is in the past", t < now)
    check("exactly one bar back", now - t <= 2 * BAR_MS and now - t > 0)
    # Anywhere inside a bar, the last CLOSED bar is the same one.
    inside = [B.closed_bar_ts(now + d) for d in (0, 1, BAR_MS // 2, BAR_MS - 1)]
    check("stable within a bar", len(set(inside)) == 1, str(inside))
    check("advances at the boundary", B.closed_bar_ts(now + BAR_MS) == t + BAR_MS)


def test_one_fetch_per_bar():
    print("\nklines are fetched once per bar, not once per pass")
    syms = [f"S{i}USDT" for i in range(12)]
    c = FakeHTTP(syms)
    b = broker_for(syms, c)

    b.refresh_signals(b.stale_symbols())
    first = c.kline_calls
    check("one call per symbol on the first pass", first == len(syms),
          f"{first} calls for {len(syms)} symbols")
    check("nothing stale afterwards", b.stale_symbols() == [],
          f"still stale: {b.stale_symbols()[:5]}")

    # 200 more scanner passes inside the same bar must cost nothing.
    for _ in range(200):
        for s in b.stale_symbols():
            b.refresh_signals([s])
    check("200 further passes make no new calls", c.kline_calls == first,
          f"{c.kline_calls - first} extra calls")

    # Roll the bar over: everything becomes stale again, exactly once.
    b.evaluated_bar = {s: v - BAR_MS for s, v in b.evaluated_bar.items()}
    check("bar rollover makes the board stale", len(b.stale_symbols()) == len(syms),
          f"{len(b.stale_symbols())} of {len(syms)}")
    b.refresh_signals(b.stale_symbols())
    check("rollover costs one call per symbol",
          c.kline_calls == 2 * len(syms), f"{c.kline_calls} total")


def test_unsignalled_symbols_not_refetched():
    """The bug that made the scanner refetch the same 24 symbols forever."""
    print("\nsymbols with no signal are still marked evaluated")
    syms = [f"S{i}USDT" for i in range(10)]
    c = FakeHTTP(syms)
    # min_votes above the number of methods, so nothing can ever signal.
    b = broker_for(syms, c, min_votes=99)

    b.refresh_signals(b.stale_symbols())
    check("no signals stand", len(b.signals) == 0, f"{len(b.signals)} stood")
    check("but all are marked evaluated", len(b.evaluated_bar) == len(syms))
    check("so none is stale", b.stale_symbols() == [],
          f"{len(b.stale_symbols())} would be refetched every pass")

    before = c.kline_calls
    for _ in range(50):
        b.refresh_signals(b.stale_symbols())
    check("50 passes make no new calls", c.kline_calls == before,
          f"{c.kline_calls - before} wasted calls")


def test_short_history_not_refetched():
    print("\nsymbols with too little history are not retried forever")
    syms = ["NEWUSDT"]
    c = FakeHTTP(syms)
    c.rows["NEWUSDT"] = make_rows(n=30)          # far below MIN_BARS
    b = broker_for(syms, c)
    b.refresh_signals(b.stale_symbols())
    check("no signal", "NEWUSDT" not in b.signals)
    check("not stale", b.stale_symbols() == [])
    before = c.kline_calls
    b.refresh_signals(b.stale_symbols())
    check("not refetched", c.kline_calls == before)


def test_blocked_signal_is_filled_when_margin_frees():
    print("\na signal blocked by margin stays standing and is filled later")
    syms = [f"S{i}USDT" for i in range(6)]
    c = FakeHTTP(syms)
    # 30% per trade, so only a few of the six fit at once. The exact number
    # is not fixed: each entry fee shrinks equity, and the slice is a
    # percentage of current equity, so the last slot is always a little
    # unaffordable. That is what a real exchange does too -- what matters
    # is that the signals it could not take are not thrown away.
    b = broker_for(syms, c, margin_pct=0.30)
    b.refresh_prices()
    for s in syms:
        b.signals[s] = B.Signal(bar_ts=B.closed_bar_ts(), direction=1,
                                atr_pct=1.0, votes=1, vote_margin=1, methods="X")

    opened, waiting = b.fill_standing()
    check("margin blocks some of them", 0 < opened < len(syms),
          f"opened {opened} of {len(syms)}")
    check("every signal is either opened or waiting",
          opened + waiting == len(syms), f"{opened} + {waiting}")
    check("signals were not consumed", len(b.signals) == len(syms),
          f"{len(b.signals)} left")
    check("no room reported", not b.has_room())
    check("margin is not oversubscribed", b.free_margin >= -1e-9,
          f"free {b.free_margin}")

    # Close one; the next standing signal must take the freed slot.
    held = list(b.open)
    b._close(held[0], b.open[held[0]].entry, "take_profit")
    opened2, _ = b.fill_standing()
    check("freed margin is reused immediately", opened2 == 1,
          f"opened {opened2}")
    check("the book is refilled to the same size", len(b.open) == opened,
          f"{len(b.open)} open, was {opened}")


def test_no_instant_reentry_after_stop():
    print("\na stopped-out symbol is not reopened by the same standing signal")
    syms = ["S0USDT"]
    c = FakeHTTP(syms)
    b = broker_for(syms, c)
    b.refresh_prices()
    bar = B.closed_bar_ts()
    b.signals["S0USDT"] = B.Signal(bar_ts=bar, direction=1, atr_pct=1.0,
                                   votes=1, vote_margin=1, methods="X")

    check("opens once", b.try_open("S0USDT") is True)
    pos = b.open["S0USDT"]
    b._close("S0USDT", pos.sl_price, "stop_loss")
    check("does not reopen on the same bar", b.try_open("S0USDT") is False)
    check("still no position", "S0USDT" not in b.open)

    # A new bar is a new decision, so it may trade again.
    b.signals["S0USDT"] = B.Signal(bar_ts=bar + BAR_MS, direction=1,
                                   atr_pct=1.0, votes=1, vote_margin=1, methods="X")
    check("reopens on the next bar", b.try_open("S0USDT") is True)


def test_take_profit_and_stop_prices():
    print("\nTP and SL fire at the right price with the right sign")
    for direction in (1, -1):
        side = "LONG" if direction > 0 else "SHORT"
        syms = ["S0USDT"]
        c = FakeHTTP(syms, price=100.0)
        b = broker_for(syms, c)
        b.refresh_prices()
        b.signals["S0USDT"] = B.Signal(bar_ts=B.closed_bar_ts(),
                                       direction=direction, atr_pct=1.0,
                                       votes=1, vote_margin=1, methods="X")
        b.try_open("S0USDT")
        pos = b.open["S0USDT"]
        tp_mult = L.TP_MULTIPLES[L.DEFAULT_EXIT]
        atr = 1.0 / 100 * pos.entry
        check(f"{side} tp is {tp_mult} ATR away",
              abs(pos.tp_price - (pos.entry + direction * tp_mult * atr)) < 1e-9)
        check(f"{side} sl is {L.SL_MULTIPLE} ATR away",
              abs(pos.sl_price - (pos.entry - direction * L.SL_MULTIPLE * atr)) < 1e-9)

        # Walk the price onto the target.
        c.last["S0USDT"] = pos.tp_price + direction * 0.5
        b.manage_all()
        check(f"{side} closes at the target", "S0USDT" not in b.open)
        t = b.closed[-1]
        check(f"{side} reason is take_profit", t.reason == "take_profit", t.reason)
        check(f"{side} target books a profit", t.pnl_usd > 0, f"{t.pnl_usd}")
        check(f"{side} exit price is the tp, not the overshoot",
              abs(t.exit_price - pos.tp_price) < 1e-9)

    for direction in (1, -1):
        side = "LONG" if direction > 0 else "SHORT"
        syms = ["S0USDT"]
        c = FakeHTTP(syms, price=100.0)
        b = broker_for(syms, c)
        b.refresh_prices()
        b.signals["S0USDT"] = B.Signal(bar_ts=B.closed_bar_ts(),
                                       direction=direction, atr_pct=1.0,
                                       votes=1, vote_margin=1, methods="X")
        b.try_open("S0USDT")
        pos = b.open["S0USDT"]
        c.last["S0USDT"] = pos.sl_price - direction * 0.5
        b.manage_all()
        t = b.closed[-1]
        check(f"{side} stops out", t.reason == "stop_loss", t.reason)
        check(f"{side} stop books a loss", t.pnl_usd < 0, f"{t.pnl_usd}")


def test_liquidation_caps_the_loss():
    print("\nliquidation costs the margin and no more")
    syms = ["S0USDT"]
    c = FakeHTTP(syms, price=100.0)
    b = broker_for(syms, c)
    b.refresh_prices()
    b.signals["S0USDT"] = B.Signal(bar_ts=B.closed_bar_ts(), direction=1,
                                   atr_pct=1.0, votes=1, vote_margin=1, methods="X")
    b.try_open("S0USDT")
    pos = b.open["S0USDT"]
    equity_before = b.equity
    # Gap straight through both levels; liquidation is checked first.
    c.last["S0USDT"] = pos.liq_price * 0.5
    b.manage_all()
    t = b.closed[-1]
    check("reason is liquidated", t.reason == "liquidated", t.reason)
    check("loss is exactly the margin", abs(t.pnl_usd + pos.margin) < 1e-9,
          f"{t.pnl_usd} vs -{pos.margin}")
    check("equity falls by the margin",
          abs((equity_before - b.equity) - pos.margin) < 1e-9)
    check("equity stays positive", b.equity > 0, f"{b.equity}")


def test_stop_always_sits_inside_liquidation():
    """The hole the solvency cap closes: above ~3.5% ATR the 17x floor put
    the stop PAST the liquidation price, so the trade died at 100% of
    margin instead of the 42% it was sized for."""
    print("\nthe stop is inside the liquidation price at every tradeable ATR")
    worst_ratio, worst_atr = 0.0, None
    checked = 0
    for atr_pct in [round(0.05 * i, 2) for i in range(1, 1200)]:
        chain = L.leverage_potential(atr_pct, None)
        if not chain["tradeable"]:
            continue
        checked += 1
        lev = chain["leverage"]
        sl_move = L.SL_MULTIPLE * atr_pct / 100.0
        liq_move = L.LIQ_MARGIN_FRACTION / lev
        ratio = sl_move / liq_move
        if ratio > worst_ratio:
            worst_ratio, worst_atr = ratio, atr_pct
    check(f"checked {checked} ATR levels from 0.05% to 60%", checked > 500)
    check("the stop always fires before liquidation", worst_ratio < 1.0,
          f"worst ratio {worst_ratio:.3f} at atr={worst_atr}%")
    check("with the intended safety buffer",
          abs(worst_ratio - 1 / L.SOLVENCY_BUFFER) < 0.01,
          f"{worst_ratio:.4f} vs {1/L.SOLVENCY_BUFFER:.4f}")

    # The file's chain must be untouched where it was already safe.
    for atr_pct in (0.10, 0.20, 0.30, 0.50, 0.80, 1.00, 1.65):
        chain = L.leverage_potential(atr_pct, None)
        base = L.lev_base(atr_pct)
        mult = L.potential_multiplier(L.potential_score(atr_pct))
        check(f"atr {atr_pct}% keeps the file's leverage",
              abs(chain["leverage"] - min(base * mult, base)) < 1e-9,
              f"{chain['leverage']} vs {min(base*mult, base)}")

    # And it must still be flexible, not clamped to one number.
    levs = {round(L.leverage_potential(a, None)["leverage"], 2)
            for a in (0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0)}
    check("leverage still varies with potential", len(levs) >= 7,
          f"only {len(levs)} distinct values: {sorted(levs)}")

    # Beyond the point where even 1x is unsound, the trade is skipped.
    syms = ["S0USDT"]
    c = FakeHTTP(syms, price=100.0)
    b = broker_for(syms, c)
    b.refresh_prices()
    b.signals["S0USDT"] = B.Signal(bar_ts=B.closed_bar_ts(), direction=1,
                                   atr_pct=90.0, votes=1, vote_margin=1, methods="X")
    check("a 90% ATR setup is refused", b.try_open("S0USDT") is False)
    check("and counted as unsolvent", b.skipped_unsolvent == 1,
          str(b.skipped_unsolvent))


def test_open_losses_reduce_free_margin():
    """Cash-only accounting let the bot keep opening at full size while its
    book was 25% underwater."""
    print("\nunrealized losses shrink the margin available")
    syms = [f"S{i}USDT" for i in range(20)]
    c = FakeHTTP(syms, price=100.0)
    b = broker_for(syms, c, margin_pct=0.10)
    b.refresh_prices()
    for s in syms:
        b.signals[s] = B.Signal(bar_ts=B.closed_bar_ts(), direction=1,
                                atr_pct=1.0, votes=1, vote_margin=1, methods="X")
    b.fill_standing()
    n_before = len(b.open)
    check("a book is open", n_before >= 5, f"{n_before}")
    check("equity_total equals cash when flat on the mark",
          abs(b.equity_total - b.equity) < 1e-6,
          f"{b.equity_total} vs {b.equity}")

    for s in list(b.open):
        c.last[s] = 99.0                     # 1% against, at ~28x
    b.refresh_prices()
    unreal = b.equity_total - b.equity
    check("the loss is visible in equity_total", unreal < -1.0, f"{unreal:.4f}")
    check("free margin absorbs it",
          b.free_margin < b.equity - b.committed_margin,
          f"free {b.free_margin:.4f}")
    check("and no new trade opens while underwater", not b.has_room(),
          f"free {b.free_margin:.4f} slice {b.slice_size():.4f}")

    opened, _ = b.fill_standing()
    check("fill_standing opens nothing", opened == 0, f"opened {opened}")

    # Recovering should restore capacity, so this is not a one-way ratchet.
    for s in list(b.open):
        c.last[s] = 101.0
    b.refresh_prices()
    check("a profitable book restores room", b.has_room(),
          f"free {b.free_margin:.4f} slice {b.slice_size():.4f}")


def test_exposure_ceiling():
    print("\ntotal notional is capped as a multiple of equity")
    syms = [f"S{i}USDT" for i in range(40)]
    c = FakeHTTP(syms, price=100.0)
    b = broker_for(syms, c, margin_pct=0.05, max_notional_x=10.0)
    b.refresh_prices()
    for s in syms:
        b.signals[s] = B.Signal(bar_ts=B.closed_bar_ts(), direction=1,
                                atr_pct=1.0, votes=1, vote_margin=1, methods="X")
    b.fill_standing()
    exposure = b.notional / b.equity_total
    check("notional stays under the ceiling", exposure <= 10.0 + 1e-6,
          f"{exposure:.2f}x")
    check("the ceiling is what stopped it, not margin",
          b.skipped_max_notional > 0 and b.free_margin > b.slice_size(),
          f"blocked {b.skipped_max_notional}, free {b.free_margin:.4f}")
    check("headroom is exhausted", b.notional_headroom() < b.slice_size() * 28,
          f"{b.notional_headroom():.4f}")

    # Without the ceiling the same board runs far hotter -- that is the point.
    b2 = broker_for(syms, FakeHTTP(syms, price=100.0), margin_pct=0.05,
                    max_notional_x=0.0)
    b2.refresh_prices()
    for s in syms:
        b2.signals[s] = B.Signal(bar_ts=B.closed_bar_ts(), direction=1,
                                 atr_pct=1.0, votes=1, vote_margin=1, methods="X")
    b2.fill_standing()
    check("uncapped exposure is much higher",
          b2.notional / b2.equity_total > exposure * 1.5,
          f"capped {exposure:.1f}x vs uncapped "
          f"{b2.notional / b2.equity_total:.1f}x")


def test_margin_never_oversubscribed():
    print("\nmargin cannot be spent twice")
    syms = [f"S{i}USDT" for i in range(40)]
    c = FakeHTTP(syms)
    b = broker_for(syms, c, margin_pct=0.10)
    b.refresh_prices()
    for s in syms:
        b.signals[s] = B.Signal(bar_ts=B.closed_bar_ts(), direction=1,
                                atr_pct=1.0, votes=1, vote_margin=1, methods="X")
    b.fill_standing()
    check("committed margin never exceeds equity",
          b.committed_margin <= b.equity + 1e-9,
          f"committed {b.committed_margin} vs equity {b.equity}")
    check("free margin is not negative", b.free_margin >= -1e-9,
          f"{b.free_margin}")
    check("about 1/margin_pct positions", len(b.open) <= 10,
          f"{len(b.open)} open")

    # And under concurrency, which is how the real scanner runs.
    import threading
    b2 = broker_for(syms, FakeHTTP(syms), margin_pct=0.10)
    b2.refresh_prices()
    for s in syms:
        b2.signals[s] = B.Signal(bar_ts=B.closed_bar_ts(), direction=1,
                                 atr_pct=1.0, votes=1, vote_margin=1, methods="X")
    threads = [threading.Thread(target=b2.fill_standing) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    check("8 concurrent fillers still respect margin",
          b2.committed_margin <= b2.equity + 1e-9 and len(b2.open) <= 10,
          f"{len(b2.open)} open, committed {b2.committed_margin:.4f}")


def test_summary_counts_add_up():
    print("\nthe shutdown summary reconciles")
    syms = [f"S{i}USDT" for i in range(8)]
    c = FakeHTTP(syms, price=100.0)
    b = broker_for(syms, c, margin_pct=0.10)
    b.refresh_prices()
    for s in syms:
        b.signals[s] = B.Signal(bar_ts=B.closed_bar_ts(), direction=1,
                                atr_pct=1.0, votes=1, vote_margin=1, methods="X")
    b.fill_standing()
    opened = list(b.open)
    # Close half at a profit, a quarter at a loss, leave the rest open.
    for s in opened[:3]:
        b._close(s, b.open[s].tp_price, "take_profit")
    for s in opened[3:5]:
        b._close(s, b.open[s].sl_price, "stop_loss")

    s = b.summary()
    check("closed = wins + losses",
          s["closed"] == s["closed_wins"] + s["closed_losses"])
    check("open = winning + losing",
          s["open"] == s["open_wins"] + s["open_losses"])
    check("total trades = closed + open",
          s["total_trades"] == s["closed"] + s["open"])
    check("total wins = closed wins + open winners",
          s["total_wins"] == s["closed_wins"] + s["open_wins"])
    check("total losses = closed losses + open losers",
          s["total_losses"] == s["closed_losses"] + s["open_losses"])
    check("realized = profit + loss",
          abs(s["realized"] - (s["gross_profit"] + s["gross_loss"])) < 1e-9)
    check("equity incl. open = cash + unrealized",
          abs(s["equity_incl_open"] - (s["equity"] + s["unrealized"])) < 1e-9)
    check("cash = start + realized - entry fees still committed",
          s["equity"] < s["starting_equity"] + s["gross_profit"] + 1e-9)
    check("open rows match the open count", len(s["open_rows"]) == s["open"])
    check("3 targets recorded", s["targets"] == 3, str(s["targets"]))
    check("2 stops recorded", s["stops"] == 2, str(s["stops"]))
    check("win rate matches the counts",
          abs(s["win_rate"] - 100 * s["closed_wins"] / s["closed"]) < 1e-9)
    # Both printers must survive whatever the numbers are.
    B.print_dashboard(b.snapshot())
    B.print_summary(s)


def test_fee_accounting():
    print("\nfees are charged on both legs and counted once")
    syms = ["S0USDT"]
    c = FakeHTTP(syms, price=100.0)
    b = broker_for(syms, c)
    b.refresh_prices()
    b.signals["S0USDT"] = B.Signal(bar_ts=B.closed_bar_ts(), direction=1,
                                   atr_pct=1.0, votes=1, vote_margin=1, methods="X")
    start = b.equity
    b.try_open("S0USDT")
    pos = b.open["S0USDT"]
    entry_fee = pos.margin * pos.leverage * b.fee / 2
    check("entry fee debited once",
          abs((start - b.equity) - entry_fee) < 1e-9,
          f"{start - b.equity} vs {entry_fee}")
    check("fees_paid tracks it", abs(b.fees_paid - entry_fee) < 1e-9)

    # Close flat: the whole result should be the round-trip fee.
    b._close("S0USDT", pos.entry, "take_profit")
    exit_fee = pos.qty * pos.entry * b.fee / 2
    check("a flat trade loses exactly the round trip",
          abs((start - b.equity) - (entry_fee + exit_fee)) < 1e-9,
          f"lost {start - b.equity:.6f}, round trip {entry_fee + exit_fee:.6f}")
    check("fees_paid is the round trip",
          abs(b.fees_paid - (entry_fee + exit_fee)) < 1e-9)


def test_leverage_chain_matches_logic():
    print("\nthe position uses the file's leverage chain unchanged")
    syms = ["S0USDT"]
    c = FakeHTTP(syms, price=100.0)
    b = broker_for(syms, c)
    b.refresh_prices()
    for atr_pct in (0.15, 0.30, 0.80, 2.0):
        b.open.clear()
        b.traded_bar.clear()
        b.signals["S0USDT"] = B.Signal(bar_ts=B.closed_bar_ts(), direction=1,
                                       atr_pct=atr_pct, votes=1, vote_margin=1,
                                       methods="X")
        b.try_open("S0USDT")
        pos = b.open["S0USDT"]
        chain = L.leverage_potential(atr_pct, None)
        check(f"atr {atr_pct}% -> lev {chain['leverage']:.1f}x",
              abs(pos.leverage - chain["leverage"]) < 1e-9,
              f"{pos.leverage} vs {chain['leverage']}")
        check(f"atr {atr_pct}% -> base kept",
              abs(pos.lev_base - chain["lev_base"]) < 1e-9)
        check(f"atr {atr_pct}% -> leverage never exceeds base",
              pos.leverage <= pos.lev_base + 1e-9)


def test_conviction_is_per_trade_and_only_cuts():
    print("\nthe conviction haircut belongs to the trade and can only cut")
    # The file's own score is a property of the INSTRUMENT: a long and a
    # short taken on the same bar get the same number. That was the whole
    # complaint, so check the new term actually separates them.
    atr = 0.5
    weak = L.leverage_potential(atr, None, votes=1, vote_margin=1)
    strong = L.leverage_potential(atr, None, votes=5, vote_margin=5)
    check("same ATR gives the same file score",
          abs(weak["potential_score"] - strong["potential_score"]) < 1e-12)
    check("but different leverage per trade",
          strong["leverage"] > weak["leverage"],
          f"weak {weak['leverage']:.1f}x strong {strong['leverage']:.1f}x")
    check("the strong one is the file's own leverage",
          abs(strong["leverage"] - L.leverage_potential(atr, None)["leverage"]) < 1e-9)

    # It must never raise leverage -- the first design did, which is why
    # this check exists.
    worst = 0.0
    for atr_pct in (0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.65, 2.0, 3.0, 5.0):
        plain = L.leverage_potential(atr_pct, None)["leverage"]
        for votes in range(1, 13):
            for vm in range(0, votes + 1):
                lev = L.leverage_potential(atr_pct, None, votes=votes,
                                           vote_margin=vm)["leverage"]
                worst = max(worst, lev - plain)
    check("never exceeds the file's leverage, over 10 ATRs x 91 vote combos",
          worst <= 1e-9, f"exceeded by {worst:.6f}")

    check("the haircut is bounded below by the floor",
          abs(L.conviction_haircut(1, 0) - L.CONVICTION_FLOOR) < 1e-12,
          str(L.conviction_haircut(1, 0)))
    check("and reaches exactly 1.0 at full agreement",
          abs(L.conviction_haircut(5, 5) - 1.0) < 1e-12)
    check("floor 1.0 disables it",
          abs(L.conviction_haircut(1, 1, floor=1.0) - 1.0) < 1e-12)
    check("a disabled haircut returns the file's chain unchanged",
          abs(L.leverage_potential(0.5, None, votes=1, vote_margin=1,
                                   conviction_floor=1.0)["leverage"]
              - L.leverage_potential(0.5, None)["leverage"]) < 1e-9)

    # And it must reach the broker, not just exist in logic.py.
    syms = ["S0USDT", "S1USDT"]
    c = FakeHTTP(syms, price=100.0)
    b = broker_for(syms, c, conviction_floor=L.CONVICTION_FLOOR)
    b.refresh_prices()
    bar = B.closed_bar_ts()
    b.signals["S0USDT"] = B.Signal(bar, 1, 0.5, 1, 1, "X")
    b.signals["S1USDT"] = B.Signal(bar, 1, 0.5, 5, 5, "X, Y, Z")
    b.try_open("S0USDT")
    b.try_open("S1USDT")
    check("the broker sizes the two differently",
          b.open["S1USDT"].leverage > b.open["S0USDT"].leverage,
          f"{b.open['S0USDT'].leverage:.1f}x vs {b.open['S1USDT'].leverage:.1f}x")
    check("and the weaker one takes less notional",
          b.open["S0USDT"].qty < b.open["S1USDT"].qty)
    check("solvency still holds for both",
          all(L.SL_MULTIPLE * 0.5 / 100 < 0.9 / p.leverage
              for p in b.open.values()))


def test_expectancy_gate():
    """A 12-hour live run lost 10.9%, and the fee bill was larger than the
    whole loss. This is the gate that refuses those trades."""
    print("\nnegative-expectancy trades are refused")
    check("expectancy is negative at taker for a typical ATR",
          L.expectancy(0.4, L.DEFAULT_EXIT, L.FEE_ROUND_TRIP) < 0,
          f"{L.expectancy(0.4, L.DEFAULT_EXIT, L.FEE_ROUND_TRIP):.6f}")
    check("and positive at maker for a high ATR",
          L.expectancy(1.5, L.DEFAULT_EXIT, L.MAKER_ROUND_TRIP) > 0,
          f"{L.expectancy(1.5, L.DEFAULT_EXIT, L.MAKER_ROUND_TRIP):.6f}")
    check("leverage cannot rescue it -- it is absent from the formula",
          L.expectancy(0.4, L.DEFAULT_EXIT, L.FEE_ROUND_TRIP)
          == L.expectancy(0.4, L.DEFAULT_EXIT, L.FEE_ROUND_TRIP))

    taker_need = L.min_atr_for_edge(L.DEFAULT_EXIT, L.FEE_ROUND_TRIP)
    maker_need = L.min_atr_for_edge(L.DEFAULT_EXIT, L.MAKER_ROUND_TRIP)
    check("taker needs an ATR BTCUSDT never reached in two years",
          taker_need > 2.42, f"needs {taker_need:.3f}%, two-year max was 2.42%")
    check("maker needs one a quarter of bars clear",
          0.3 < maker_need < 1.0, f"{maker_need:.3f}%")
    check("the gate's threshold is where expectancy crosses zero",
          abs(L.expectancy(taker_need, L.DEFAULT_EXIT, L.FEE_ROUND_TRIP)) < 1e-12)

    # The broker must actually apply it.
    syms = ["S0USDT"]
    c = FakeHTTP(syms, price=100.0)
    b = broker_for(syms, c, fee=L.FEE_ROUND_TRIP, expectancy_gate=True)
    b.refresh_prices()
    b.signals["S0USDT"] = B.Signal(B.closed_bar_ts(), 1, 0.4, 1, 1, "X")
    check("a typical taker trade is refused", b.try_open("S0USDT") is False)
    check("and counted", b.skipped_negative_ev == 1, str(b.skipped_negative_ev))

    b2 = broker_for(syms, FakeHTTP(syms, price=100.0), fee=L.MAKER_ROUND_TRIP,
                    expectancy_gate=True)
    b2.refresh_prices()
    b2.signals["S0USDT"] = B.Signal(B.closed_bar_ts(), 1, 1.5, 1, 1, "X")
    check("the same setup at maker fees is taken", b2.try_open("S0USDT") is True)

    b3 = broker_for(syms, FakeHTTP(syms, price=100.0), fee=L.FEE_ROUND_TRIP,
                    expectancy_gate=False)
    b3.refresh_prices()
    b3.signals["S0USDT"] = B.Signal(B.closed_bar_ts(), 1, 0.4, 1, 1, "X")
    check("the gate can be switched off", b3.try_open("S0USDT") is True)


def test_stale_excludes_open_positions():
    print("\nsymbols already holding a position are not rescored")
    syms = ["S0USDT", "S1USDT"]
    c = FakeHTTP(syms)
    b = broker_for(syms, c)
    b.refresh_prices()
    b.signals["S0USDT"] = B.Signal(bar_ts=B.closed_bar_ts(), direction=1,
                                   atr_pct=1.0, votes=1, vote_margin=1, methods="X")
    b.try_open("S0USDT")
    stale = b.stale_symbols()
    check("the open symbol is left out", "S0USDT" not in stale, str(stale))
    check("the other one is still due", "S1USDT" in stale, str(stale))


def main() -> int:
    print("=" * 70)
    print("fp.bot tests")
    print("=" * 70)
    for fn in (test_closed_bar_alignment,
               test_one_fetch_per_bar,
               test_unsignalled_symbols_not_refetched,
               test_short_history_not_refetched,
               test_blocked_signal_is_filled_when_margin_frees,
               test_no_instant_reentry_after_stop,
               test_take_profit_and_stop_prices,
               test_liquidation_caps_the_loss,
               test_stop_always_sits_inside_liquidation,
               test_open_losses_reduce_free_margin,
               test_exposure_ceiling,
               test_margin_never_oversubscribed,
               test_fee_accounting,
               test_leverage_chain_matches_logic,
               test_conviction_is_per_trade_and_only_cuts,
               test_expectancy_gate,
               test_stale_excludes_open_positions,
               test_summary_counts_add_up):
        fn()
    print("\n" + "=" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
    else:
        print("all checks passed")
    print("=" * 70)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
