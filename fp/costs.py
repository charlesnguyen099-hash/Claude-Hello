"""What a trade really costs, per coin, and how much leverage it may use.

A flat 0.110% round trip was wrong in two ways, and both matter.

1. IT IGNORED THE SPREAD. A market order does not fill at the close; it
   crosses the book. On BLESS the median one-minute high-low range is
   0.7110% -- two orders of magnitude wider than BTC's 0.0249% -- so the
   same "0.110% round trip" is roughly right for BTC and badly wrong for
   an illiquid alt. The spread is estimated here from the bars
   themselves with the Corwin-Schultz high-low estimator, which is the
   standard way to recover an effective spread when quotes are not
   available.

2. IT IGNORED LEVERAGE. The fee is charged on NOTIONAL, so at 10x the
   round trip costs ten times as much of the MARGIN behind it. A 0.11%
   round trip is 1.1% of capital at 10x. Returns scale the same way, so
   the RATIO is unchanged -- leverage does not make a losing edge win --
   but the drawdown does scale, and the break-even accuracy does not
   improve one basis point by adding leverage. round_trip() reports the
   cost per unit of notional; margin_cost() converts it.

LEVERAGE IS PER COIN. Bybit does not offer the same maximum on every
symbol: majors go to 100x, thin alts are capped far lower, and the cap
moves with the risk tier. The live bot reads leverageFilter.maxLeverage
from get_instruments_info at startup and never exceeds it. Studies here
use the conservative table below, because a backtest that assumes 50x on
a coin the exchange caps at 10x is fiction.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "data" / "instruments.json"

# Bybit VIP0, linear perpetuals. Both sides taker: a target or a stop is
# a market order when it fires and cannot earn the maker rate.
TAKER_PER_SIDE = 0.00055
FUNDING_PER_8H = 1e-4

# Used only when the exchange cannot be reached. Deliberately low: an
# assumed leverage the venue would refuse is a backtest that cannot be
# traded.
DEFAULT_MAX_LEVERAGE = 10.0


def corwin_schultz(high, low, window: int = 240) -> np.ndarray:
    """Effective spread per bar, from high-low ranges alone.

    Corwin & Schultz (2012): over two consecutive bars, the ratio of the
    combined range to the sum of the single-bar ranges separates the true
    volatility from the bid-ask bounce, because volatility scales with
    time and the spread does not.
    """
    h = pd.Series(np.asarray(high, dtype="float64"))
    l = pd.Series(np.asarray(low, dtype="float64"))
    h2 = pd.concat([h, h.shift(1)], axis=1).max(axis=1)
    l2 = pd.concat([l, l.shift(1)], axis=1).min(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = (np.log(h / l) ** 2 + np.log(h.shift(1) / l.shift(1)) ** 2)
        gamma = np.log(h2 / l2) ** 2
        k = 3.0 - 2.0 * np.sqrt(2.0)
        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
        s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    s = pd.Series(s).clip(lower=0.0)
    return s.rolling(window, min_periods=window // 4).median().bfill().values


def per_side(d: pd.DataFrame, taker: float = TAKER_PER_SIDE) -> np.ndarray:
    """Cost of ONE side, as a fraction of notional: fee plus half spread.

    Half the spread, because crossing takes you from the mid to the far
    touch -- and the same on the way out, so a round trip pays a full
    spread plus two fees.
    """
    spread = corwin_schultz(d["high"].values, d["low"].values)
    return taker + spread / 2.0


def round_trip(d: pd.DataFrame, hold_minutes: float = 0.0,
               taker: float = TAKER_PER_SIDE) -> np.ndarray:
    """Everything a trade pays, per unit of NOTIONAL."""
    return 2.0 * per_side(d, taker) + \
        (hold_minutes / 60.0 / 8.0) * FUNDING_PER_8H


def margin_cost(notional_cost, leverage) -> np.ndarray:
    """The same cost expressed against the MARGIN behind the trade.

    This is the number that surprises people: at 10x, a 0.11% round trip
    is 1.1% of the capital committed. It does not change whether an edge
    exists -- the winnings scale identically -- but it is what the
    account actually feels.
    """
    return np.asarray(notional_cost, dtype="float64") * float(leverage)


def break_even(win: float, cost) -> np.ndarray:
    """Accuracy a symmetric +/-win barrier needs, at this cost."""
    c = np.asarray(cost, dtype="float64")
    gain, loss = win - c, win + c
    return np.where(gain > 0, loss / (loss + gain), 1.0)


def max_leverage(symbol: str) -> float:
    """What Bybit actually allows on this symbol.

    Read from data/instruments.json, which the live bot refreshes from
    get_instruments_info at startup. Falls back low rather than high: a
    study that assumes more leverage than the venue permits is not a
    study of anything tradeable.
    """
    try:
        return float(json.loads(CACHE.read_text())[symbol]["max_leverage"])
    except Exception:
        return DEFAULT_MAX_LEVERAGE


# Bybit's common floor for a linear USDT perpetual, used only when the
# instruments cache has nothing for a symbol -- missing data should make
# an order request MORE conservative, never less.
DEFAULT_MIN_NOTIONAL = 5.0


def min_notional(symbol: str, price: float = 0.0) -> float:
    """The smallest position value Bybit will accept on this symbol.

    Two numbers on the exchange can bind: a minimum ORDER VALUE
    (min_notional directly) and a minimum ORDER QUANTITY (min_qty,
    which only becomes a dollar figure once multiplied by price). This
    returns whichever is larger, in dollars, so a caller has one number
    to compare margin x leverage against. Falls back to
    DEFAULT_MIN_NOTIONAL when the cache has nothing usable for this
    symbol, rather than falling back to zero -- a silent zero floor
    would let a trade below the exchange's real minimum through and
    fail at the order, not here.
    """
    try:
        row = json.loads(CACHE.read_text())[symbol]
        by_value = float(row.get("min_notional", 0) or 0)
        by_qty = float(row.get("min_qty", 0) or 0) * price
        best = max(by_value, by_qty)
        return best if best > 0 else DEFAULT_MIN_NOTIONAL
    except Exception:
        return DEFAULT_MIN_NOTIONAL


def refresh(client, symbols) -> dict:
    """Pull the real per-symbol limits and cache them.

    Also captures the fields a REAL order needs and min_notional()
    above was already reading but nothing ever wrote: qty_step and
    min_qty (lotSizeFilter) and tick_size (priceFilter). Without these
    an order quantity computed from margin/price is an arbitrary
    decimal Bybit will reject outright -- rounding to the exchange's
    own step is not optional once an order actually gets sent.
    """
    out = {}
    try:
        rows = client.get_instruments_info(
            category="linear")["result"]["list"]
    except Exception:
        return out
    want = set(symbols)
    for x in rows:
        s = x.get("symbol")
        if s not in want:
            continue
        lf = x.get("leverageFilter", {}) or {}
        lsf = x.get("lotSizeFilter", {}) or {}
        pf = x.get("priceFilter", {}) or {}
        out[s] = {"max_leverage": float(lf.get("maxLeverage", 1) or 1),
                  "min_leverage": float(lf.get("minLeverage", 1) or 1),
                  "qty_step": float(lsf.get("qtyStep", 0) or 0),
                  "min_qty": float(lsf.get("minOrderQty", 0) or 0),
                  "min_notional": float(lsf.get("minNotionalValue", 0)
                                        or 0),
                  "tick_size": float(pf.get("tickSize", 0) or 0)}
    if out:
        try:
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            CACHE.write_text(json.dumps(out, indent=1))
        except Exception:
            pass
    return out


def round_qty(symbol: str, qty: float, price: float = 0.0) -> float:
    """A quantity Bybit will actually accept for this symbol, or 0.0 if
    there is no real qty_step cached to round it against.

    Rounds DOWN to the exchange's qty_step -- rounding up could push
    a carefully-sized order over what margin actually covers -- and
    never below min_qty.

    REFUSES rather than guesses when qty_step is missing. An earlier
    version fell back to rounding to 6 decimals, meant to be
    "generous" -- fine enough to not matter -- for every linear USDT
    perp Bybit lists. That assumption breaks exactly on a symbol
    priced in the hundreds or thousands (SNDKUSDT, XAUUSDT, ...),
    where Bybit's real step is coarse (0.001, 0.01, whole units) and
    a 6-decimal quantity is not "generous", it is one the exchange
    does not recognise at all: real orders on SNDKUSDT were rejected
    over and over, first "number of contracts exceeds minimum limit
    allowed" (below min_notional after step-rounding, fixed below),
    then "Qty invalid" once that fix pushed the guessed-precision
    quantity to a value that still did not land on the exchange's
    real step. There is no fallback precision that is safe for every
    symbol; refusing to size the order is.

    Bybit binds on TWO separate floors: a quantity (min_qty, checked
    above) and a dollar value (min_notional). open_position() sizes
    the pre-rounding quantity to clear the second one -- but rounding
    DOWN to qty_step can cross back below it. Pass `price` and this
    rounds UP by one step instead when needed -- the smallest change
    that clears the floor Bybit actually enforces.
    """
    try:
        row = json.loads(CACHE.read_text())[symbol]
        step = float(row.get("qty_step", 0) or 0)
        floor_qty = float(row.get("min_qty", 0) or 0)
    except Exception:
        return 0.0
    if step <= 0:
        return 0.0
    n_steps = int(qty / step)
    rounded = max(n_steps * step, floor_qty)
    if price > 0 and rounded * price < min_notional(symbol, price):
        rounded += step
    return rounded
