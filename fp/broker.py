"""Real orders on Bybit -- money that leaves the paper-trading world.

Every other file in fp/ can be wrong and the worst that happens is a
number in a JSON report is wrong. This one is different: a bug here
places a REAL market order, with REAL leverage, against a REAL
account. It has been written as carefully as the rest of this system,
but it has NOT been exercised against a live Bybit account from this
session -- this sandbox cannot reach api.bybit.com at all (see
run_full_bot.py's ReplayFeed docstring). Test it on a small amount of
real capital before trusting it with more, the same way any of this
system's logic was proven on data before it was proven on money.

WHAT THIS DOES NOT DO. No conditional (stop/take-profit) orders are
placed on the exchange -- the exit logic (target, ratcheting stop,
trailing peak) is the SAME adaptive logic the paper account already
runs, computed fresh every scan cycle from the live bar, and an exit
becomes a real reduce-only MARKET order the instant that logic fires.
A resting conditional order cannot express "trail below the peak by
the model's own estimate of this setup's noise" -- that number moves
every bar. This means a position stays exposed between scans (up to
SCAN_SECONDS wide) with no order sitting on the book protecting it;
the scan loop closing it promptly is the only stop that exists.
"""
from __future__ import annotations

from fp import costs as C

CATEGORY = "linear"


class OrderError(RuntimeError):
    """A real order request Bybit rejected or could not be confirmed."""


class LiveBroker:
    """Thin wrapper over pybit's authenticated V5 client.

    Every method either returns the fill/position data a caller needs
    or raises OrderError -- never returns a half-successful result
    silently, because a caller (Account.open_position/close_position)
    that thinks a real order went through when it did not would keep
    trading a position that exists only in its own memory.
    """

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        from pybit.unified_trading import HTTP
        self.client = HTTP(testnet=testnet, api_key=api_key,
                           api_secret=api_secret)

    def wallet_equity(self) -> float:
        """Total USDT equity on the unified trading account right now.

        The one number that must never be allowed to drift from the
        exchange's own truth -- unlike paper trading, internal pnl
        arithmetic is a prediction of this number, not a substitute
        for reading it.
        """
        r = self.client.get_wallet_balance(accountType="UNIFIED",
                                           coin="USDT")
        rows = r.get("result", {}).get("list", [])
        if not rows:
            raise OrderError("empty wallet_balance response")
        try:
            return float(rows[0]["totalEquity"])
        except (KeyError, TypeError, ValueError) as exc:
            raise OrderError(f"could not read totalEquity: {exc}") from exc

    def open_positions(self) -> dict:
        """Symbol -> {side, qty, entry} for every position currently
        open on the real account, whether this bot opened it or not.

        Used at startup to refuse to run over positions the bot does
        not know the history of, and to reconcile if the process
        restarts mid-trade.
        """
        r = self.client.get_positions(category=CATEGORY, settleCoin="USDT")
        out = {}
        for row in r.get("result", {}).get("list", []):
            qty = float(row.get("size", 0) or 0)
            if qty <= 0:
                continue
            out[row["symbol"]] = {
                "side": 1 if row.get("side") == "Buy" else -1,
                "qty": qty,
                "entry": float(row.get("avgPrice", 0) or 0),
            }
        return out

    def set_leverage(self, symbol: str, leverage: float) -> None:
        lev = f"{leverage:g}"
        try:
            self.client.set_leverage(category=CATEGORY, symbol=symbol,
                                     buyLeverage=lev, sellLeverage=lev)
        except Exception as exc:
            # Bybit's own error code for "already set to this value" --
            # not a failure, the leverage IS what was asked for.
            if "110043" in str(exc):
                return
            raise OrderError(f"set_leverage({symbol}, {leverage}): "
                            f"{exc}") from exc

    def market_order(self, symbol: str, side: int, qty: float,
                     reduce_only: bool = False) -> dict:
        """A market order, IOC, one-way position mode. Returns the
        order response's result dict, or raises OrderError.

        qty must already be rounded to the symbol's step
        (fp.costs.round_qty) -- this function does not round, because
        a caller that sized a position and a caller that is closing an
        EXACT existing quantity need different rounding directions,
        and silently re-rounding here would let the two disagree.
        """
        if qty <= 0:
            raise OrderError(f"refusing a non-positive quantity: {qty}")
        try:
            r = self.client.place_order(
                category=CATEGORY, symbol=symbol,
                side="Buy" if side > 0 else "Sell",
                orderType="Market", qty=f"{qty}",
                timeInForce="IOC", reduceOnly=reduce_only,
                positionIdx=0,
            )
        except Exception as exc:
            raise OrderError(f"place_order({symbol}, side={side}, "
                            f"qty={qty}, reduce_only={reduce_only}): "
                            f"{exc}") from exc
        if r.get("retCode") not in (0, None):
            raise OrderError(f"place_order({symbol}) rejected: "
                            f"{r.get('retMsg')} (code {r.get('retCode')})")
        return r.get("result", {})

    def last_price(self, symbol: str) -> float:
        r = self.client.get_tickers(category=CATEGORY, symbol=symbol)
        rows = r.get("result", {}).get("list", [])
        if not rows:
            raise OrderError(f"no ticker for {symbol}")
        return float(rows[0]["lastPrice"])
