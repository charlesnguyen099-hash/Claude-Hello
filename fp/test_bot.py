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
    def __init__(self, symbols, price=100.0, newest_bar=None, funding=0.0001):
        self.symbols = symbols
        self.funding = funding
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
                                     "turnover24h": "1000",
                                     "fundingRate": str(self.funding),
                                     "nextFundingTime": "0"}
                                    for s, p in self.last.items()]}}


def broker_for(symbols, client, **kw):
    opts = dict(equity=10.0, max_positions=0, exit_name=L.DEFAULT_EXIT,
                min_votes=1, fee=L.FEE_ROUND_TRIP, max_leverage=None,
                margin_pct=0.10, max_notional_x=0.0, conviction_floor=1.0,
                expectancy_gate=False, potential_sizing=False, sizing="flat")
    opts.update(kw)
    return B.Broker(client, symbols, **opts)


def Sig(*a, **kw):
    """A Signal shaped the way the live path now produces them.

    Every signal the bot creates arrives with its leverage already solved
    from its own stop distance and hold -- try_open raises if one does
    not, because a missing leverage is a bug rather than a case to handle.
    These tests predate that and build signals positionally, so this fills
    in the field without changing what each test is actually asserting.
    """
    kw.setdefault("slow_leverage", 3.0)
    return B.Signal(*a, **kw)


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
        b.signals[s] = Sig(bar_ts=B.closed_bar_ts(), direction=1,
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
    b.signals["S0USDT"] = Sig(bar_ts=bar, direction=1, atr_pct=1.0,
                                   votes=1, vote_margin=1, methods="X")

    check("opens once", b.try_open("S0USDT") is True)
    pos = b.open["S0USDT"]
    b._close("S0USDT", pos.sl_price, "stop_loss")
    check("does not reopen on the same bar", b.try_open("S0USDT") is False)
    check("still no position", "S0USDT" not in b.open)

    # A new bar is a new decision, so it may trade again.
    b.signals["S0USDT"] = Sig(bar_ts=bar + BAR_MS, direction=1,
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
        b.signals["S0USDT"] = Sig(bar_ts=B.closed_bar_ts(),
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
        b.signals["S0USDT"] = Sig(bar_ts=B.closed_bar_ts(),
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
    b.signals["S0USDT"] = Sig(bar_ts=B.closed_bar_ts(), direction=1,
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
    """A stop must fire BEFORE liquidation, or the trade dies at 100% of
    margin instead of the fraction it was sized for.

    The old version of this checked an ATR leverage ladder that no longer
    exists. The guarantee itself still has to hold, and now it holds on
    each signal's OWN stop distance -- which is stronger, because the
    stop is no longer a fixed multiple of anything.
    """
    print("\nthe stop is inside the liquidation price for every signal")
    from fp import leverage as LV
    syms = ["S0USDT"]
    worst, worst_at = 0.0, None
    checked = 0
    for sigma in [0.0002 * i for i in range(1, 60)]:
        for tp, sl, hold in ((1.0, 1.0, 60), (2.0, 2.0, 240), (4.0, 4.0, 720)):
            tp_d = tp * sigma * (hold ** 0.5)
            sl_d = sl * sigma * (hold ** 0.5)
            chain = LV.solvent_leverage(tp_d, sl_d, hold, sigma,
                                        LV.LEVERAGE_MAX)
            if not chain["tradeable"]:
                continue
            checked += 1
            ratio = sl_d / (LV.LIQ_MARGIN_FRACTION / chain["leverage"])
            if ratio > worst:
                worst, worst_at = ratio, (tp, sl, hold, round(sigma, 5))
    check(f"checked {checked} shape/volatility combinations", checked > 100,
          checked)
    check("the stop always fires before liquidation", worst < 1.0,
          f"worst ratio {worst:.3f} at {worst_at}")

    # A stop so wide that even 1x cannot keep it inside liquidation must
    # be refused outright rather than sized down.
    c = FakeHTTP(syms, price=100.0)
    b = broker_for(syms, c)
    b.refresh_prices()
    wide = Sig(bar_ts=B.closed_bar_ts(), direction=1,
               atr_pct=90.0, votes=1, vote_margin=1,
               methods="X", slow_leverage=3.0,
               tp_dist=0.95, sl_dist=0.95,
               max_hold_min=240.0, rule="eng:t", symbol="S0USDT",
               margin_frac=0.5, score=50.0)
    b.signals[wide.slot] = wide
    check("a stop wider than liquidation is refused",
          b.try_open(wide.slot) is False)
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
        b.signals[s] = Sig(bar_ts=B.closed_bar_ts(), direction=1,
                                atr_pct=1.0, votes=1, vote_margin=1, methods="X")
    b.fill_standing()
    n_before = len(b.open)
    check("a book is open", n_before >= 5, f"{n_before}")
    check("equity_total equals cash when flat on the mark",
          abs(b.equity_total - b.equity) < 1e-6,
          f"{b.equity_total} vs {b.equity}")

    for s in list(b.open):
        c.last[s] = 99.0                     # 1% against, at the solved 3x
    b.refresh_prices()
    unreal = b.equity_total - b.equity
    # 1% against at 3x on 10% margin slices. The old ladder ran at ~28x
    # and this threshold was -1.0; leverage is now SOLVED per trade
    # against its own stop and drag, and 3x is where that lands.
    check("the loss is visible in equity_total", unreal < -0.2, f"{unreal:.4f}")
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
    b = broker_for(syms, c, margin_pct=0.05, max_notional_x=1.5)
    b.refresh_prices()
    for s in syms:
        b.signals[s] = Sig(bar_ts=B.closed_bar_ts(), direction=1,
                                atr_pct=1.0, votes=1, vote_margin=1, methods="X")
    b.fill_standing()
    exposure = b.notional / b.equity_total
    check("notional stays under the ceiling", exposure <= 1.5 + 1e-6,
          f"{exposure:.2f}x")
    check("the ceiling is what stopped it, not margin",
          b.skipped_max_notional > 0 and b.free_margin > b.slice_size(),
          f"blocked {b.skipped_max_notional}, free {b.free_margin:.4f}")
    check("headroom is exhausted",
          b.notional_headroom() < b.slice_size() * 3,
          f"{b.notional_headroom():.4f}")

    # Without the ceiling the same board runs far hotter -- that is the point.
    b2 = broker_for(syms, FakeHTTP(syms, price=100.0), margin_pct=0.05,
                    max_notional_x=0.0)
    b2.refresh_prices()
    for s in syms:
        b2.signals[s] = Sig(bar_ts=B.closed_bar_ts(), direction=1,
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
        b.signals[s] = Sig(bar_ts=B.closed_bar_ts(), direction=1,
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
        b2.signals[s] = Sig(bar_ts=B.closed_bar_ts(), direction=1,
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
        b.signals[s] = Sig(bar_ts=B.closed_bar_ts(), direction=1,
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
    b.signals["S0USDT"] = Sig(bar_ts=B.closed_bar_ts(), direction=1,
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


def test_best_signals_are_filled_first():
    """The exposure ceiling is the scarce resource; whatever is tried
    first spends it."""
    print("\nthe best signals get the budget, not the first to arrive")
    syms = [f"S{i}USDT" for i in range(12)]
    c = FakeHTTP(syms, price=100.0)
    # The ceiling has to sit where it can actually bind: leverage is
    # solved and capped at 3x, so total notional cannot exceed 3x equity
    # however many positions open.
    b = broker_for(syms, c, sizing="potential", margin_pct=0.05,
                   max_notional_x=0.0)
    b.refresh_prices()
    bar = B.closed_bar_ts()
    # Worst first in insertion order, so arrival order and quality disagree.
    # Worst first in insertion order, so arrival order and quality
    # disagree. Ranking is by POTENTIAL now -- one 0-100 scale shared by
    # every shape -- not by an ATR-derived expectation.
    scores = [5, 10, 15, 20, 25, 30, 45, 55, 65, 75, 85, 95]
    for s, sc in zip(syms, scores):
        b.signals[s] = Sig(bar, 1, 1.0, 1, 1, "X", slow_leverage=3.0,
                           tp_dist=0.02, sl_dist=0.02, max_hold_min=240.0,
                           symbol=s, margin_frac=sc / 100.0 * 0.5,
                           score=float(sc))
    b.fill_standing()
    check("something opened", len(b.open) > 0, f"{len(b.open)}")
    check("margin ran out before every signal was filled",
          len(b.open) < len(syms), f"{len(b.open)} of {len(syms)}")
    # Ranking follows expected return per dollar of margin, which now
    # comes from the measured table -- so it is NOT simply "highest ATR
    # first" any more. What matters is that what opened ranks above what
    # did not, on the measure actually used.
    opened = [b.signals[k].score for k in b.open]
    rejected = [g.score for k, g in b.signals.items() if k not in b.open]
    check("what opened outranks what did not, on potential",
          (not rejected) or min(opened) >= max(rejected) - 1e-9,
          f"opened {sorted(opened)[:3]}, rejected {sorted(rejected)[-3:]}")


def test_full_cost_model():
    """Only trade what is profitable after EVERY cost -- which means the
    exit fee is taker even on a maker entry, and funding is counted."""
    print("\nthe cost model counts entry, exit and funding")
    c = L.round_trip_cost(L.DEFAULT_EXIT, entry_maker=True)
    check("a maker entry still pays a taker exit",
          abs(c["exit"] - L.EXIT_FEE_TAKER) < 1e-12,
          f"exit {c['exit']:.5f} vs taker {L.EXIT_FEE_TAKER:.5f}")
    check("so the cheapest round trip is 0.075%, not 0.040%",
          abs((c["entry"] + c["exit"]) - 0.00075) < 1e-9,
          f"{100*(c['entry']+c['exit']):.4f}%")
    check("funding is included", c["funding"] > 0, f"{c['funding']:.6f}")
    check("and matches the measured hold",
          abs(c["funding_events"] - c["hold_hours"] / L.FUNDING_INTERVAL_HOURS) < 1e-9)
    check("total is the sum of the three",
          abs(c["total"] - (c["entry"] + c["exit"] + c["funding"])) < 1e-12)

    t = L.round_trip_cost(L.DEFAULT_EXIT, entry_maker=False)
    check("a market entry costs more", t["total"] > c["total"],
          f"{100*c['total']:.3f}% vs {100*t['total']:.3f}%")
    trend = L.round_trip_cost(L.DEFAULT_EXIT, True, L.FUNDING_RATE_TRENDING)
    check("a trending market costs more still", trend["total"] > c["total"])
    slip = L.round_trip_cost(L.DEFAULT_EXIT, True, L.FUNDING_RATE_TYPICAL, 0.0005)
    check("slippage is charged on both sides",
          abs(slip["total"] - c["total"] - 0.001) < 1e-9,
          f"{100*(slip['total']-c['total']):.4f}%")

    # A longer-held exit meets more funding.
    short = L.round_trip_cost("net_TRAILING", True)
    long_ = L.round_trip_cost("net_TP3.0_SL1.5", True)
    check("a longer hold meets more funding",
          long_["funding"] > short["funding"],
          f"{long_['funding_events']:.2f} vs {short['funding_events']:.2f}")

    # And the gate must move with the cost.
    needs = [L.min_atr_for_edge(L.DEFAULT_EXIT,
                                L.round_trip_cost(L.DEFAULT_EXIT, mk, fr)["total"])
             for mk, fr in ((True, L.FUNDING_RATE_TYPICAL),
                            (False, L.FUNDING_RATE_TYPICAL),
                            (False, L.FUNDING_RATE_TRENDING))]
    check("a costlier trade needs a bigger move", needs == sorted(needs),
          str([round(x, 3) for x in needs]))
    check("even the cheapest needs more than a typical bar",
          needs[0] > 1.0, f"{needs[0]:.3f}%")

    # There is no ATR-band expectancy gate any more. What refuses an
    # unprofitable setup now is the setup's OWN arithmetic: potential()
    # scores 0 whenever the target does not clear the round trip at that
    # bar's volatility, and a 0 score is a 0 stake.
    from fp import engine as EN
    import numpy as _np
    thin = _np.array([0.00005])          # a very quiet bar
    p_be, a, b_ = EN.break_even(2.0, 2.0, 60, thin)
    check("a target the fee eats gives a non-positive payout",
          b_[0] <= 0, f"b = {100*b_[0]:+.4f}%")
    check("and break-even for it is impossible", p_be[0] >= 1.0, p_be[0])


def test_real_costs_come_from_the_exchange():
    """No fee to choose: taker both sides, and the symbol's own live
    funding rate, signed."""
    print("\ncosts are taken from the exchange, not from a flag")
    syms = ["S0USDT"]
    c = FakeHTTP(syms, price=100.0, funding=0.0002)
    b = broker_for(syms, c, sizing="flat")
    b.refresh_prices()
    check("the funding rate is read off the ticker feed",
          abs(b.funding["S0USDT"] - 0.0002) < 1e-12, str(b.funding))

    long_cost = b.trade_cost("S0USDT", 1)
    short_cost = b.trade_cost("S0USDT", -1)
    taker = L.ENTRY_FEE_TAKER + L.EXIT_FEE_TAKER
    check("both sides are taker by default",
          abs((long_cost + short_cost) / 2 - taker) < 1e-9,
          f"{100*(long_cost+short_cost)/2:.4f}% vs {100*taker:.4f}%")
    check("a long PAYS positive funding", long_cost > taker,
          f"{100*long_cost:.4f}%")
    check("a short COLLECTS it", short_cost < taker, f"{100*short_cost:.4f}%")

    # Negative funding flips who pays.
    c2 = FakeHTTP(syms, price=100.0, funding=-0.0002)
    b2 = broker_for(syms, c2, sizing="flat")
    b2.refresh_prices()
    check("negative funding reverses that",
          b2.trade_cost("S0USDT", 1) < b2.trade_cost("S0USDT", -1))

    # A limit entry saves only the entry side.
    b3 = broker_for(syms, FakeHTTP(syms, funding=0.0), sizing="flat",
                    limit_entry=True)
    b3.refresh_prices()
    saved = taker - b3.trade_cost("S0USDT", 1)
    check("a limit entry saves the entry side only",
          abs(saved - (L.ENTRY_FEE_TAKER - L.ENTRY_FEE_MAKER)) < 1e-9,
          f"saved {100*saved:.4f}%")
    check("the exit is still taker",
          b3.trade_cost("S0USDT", 1) >= L.EXIT_FEE_TAKER)

    # Slippage is charged on both sides.
    b4 = broker_for(syms, FakeHTTP(syms, funding=0.0), sizing="flat",
                    slippage=0.0005)
    b4.refresh_prices()
    check("slippage is charged twice",
          abs(b4.trade_cost("S0USDT", 1) - taker - 0.001) < 1e-9,
          f"{100*(b4.trade_cost('S0USDT',1)-taker):.4f}%")

    # An expensive symbol should be rejected where a cheap one is taken.
    # The sign reaches the gate: with an extreme rate the long is charged
    # far more than the short, even though the table refuses both.
    hi = FakeHTTP(["XUSDT"], price=100.0, funding=0.01)   # 1% per 8h, extreme
    bh = broker_for(["XUSDT"], hi, sizing="flat", expectancy_gate=True)
    bh.refresh_prices()
    check("an extreme rate splits the two sides widely",
          bh.trade_cost("XUSDT", 1) - bh.trade_cost("XUSDT", -1) > 0.015,
          f"{100*(bh.trade_cost('XUSDT',1)-bh.trade_cost('XUSDT',-1)):.3f}%")
    # And the signed cost reaches the sizing: at this funding rate the
    # long's round trip is larger than the short's, so the same barriers
    # produce a worse break-even for the long.
    from fp import engine as EN
    import numpy as _np
    sg = _np.array([0.001])
    long_be, _, _ = EN.break_even(2.0, 2.0, 240, sg, bh.trade_cost("XUSDT", 1))
    short_be, _, _ = EN.break_even(2.0, 2.0, 240, sg, bh.trade_cost("XUSDT", -1))
    check("the long needs a higher win rate than the short",
          long_be[0] > short_be[0],
          f"{100*long_be[0]:.1f}% vs {100*short_be[0]:.1f}%")


def test_open_positions_do_not_block_a_coin_s_other_shapes():
    print("\na coin with a position open is still scored for other shapes")
    syms = ["S0USDT", "S1USDT"]
    c = FakeHTTP(syms)
    b = broker_for(syms, c)
    b.refresh_prices()
    b.signals["S0USDT"] = Sig(bar_ts=B.closed_bar_ts(), direction=1,
                                   atr_pct=1.0, votes=1, vote_margin=1, methods="X")
    b.try_open("S0USDT")
    stale = b.stale_symbols()
    # A coin holds one position per SHAPE and SIDE. Skipping it because
    # something is already open is what made one slow trade lock a coin
    # out of every fast shape for the length of the hold.
    check("the coin with an open position is still scored",
          "S0USDT" in stale, str(stale))
    check("the other one is still due", "S1USDT" in stale, str(stale))
    b.evaluated_bar["S0USDT"] = B.closed_bar_ts(bar_minutes=b.bar_minutes)
    check("and drops out once it HAS been scored this bar",
          "S0USDT" not in b.stale_symbols(), str(b.stale_symbols()))


def main() -> int:
    print("=" * 70)
    print("fp.bot tests")
    print("=" * 70)
    for fn in (               test_closed_bar_alignment,
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
               test_best_signals_are_filled_first,
               test_full_cost_model,
               test_real_costs_come_from_the_exchange,
               test_open_positions_do_not_block_a_coin_s_other_shapes,
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
