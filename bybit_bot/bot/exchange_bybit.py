"""Thin wrapper around Bybit's v5 unified-trading REST API (via pybit).

Every state-changing call (leverage, orders, stop management) is a no-op
that only logs when Config.dry_run is True — this is the default, and the
bot will not place a single real order until an operator explicitly sets
DRY_RUN=false with real API keys in .env.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from pybit.unified_trading import HTTP

from bot.config import Config

logger = logging.getLogger("bybit_bot.exchange")

INTERVAL_MAP = {"1m": "1", "15m": "15", "1h": "60"}


@dataclass
class InstrumentInfo:
    symbol: str
    tick_size: float
    qty_step: float
    min_order_qty: float
    max_leverage: float


class BybitExchange:
    def __init__(self, config: Config):
        self.config = config
        self.client = HTTP(
            testnet=config.testnet,
            api_key=config.api_key or None,
            api_secret=config.api_secret or None,
        )
        self._instrument_cache: dict[str, InstrumentInfo] = {}

    _MAX_KLINES_PER_CALL = 1000  # Bybit v5 kline endpoint hard cap per request

    def get_klines(self, symbol: str, timeframe: str, limit: int = 300) -> pd.DataFrame:
        """Fetches up to `limit` most-recent closed+forming candles,
        transparently paginating (walking backward with `end`) when
        `limit` exceeds Bybit's per-call cap — needed because the slower
        indicators (e.g. EMA span=315 on 1m bars) need several thousand
        bars of history to actually converge, not just to have a
        non-NaN value.
        """
        interval = INTERVAL_MAP[timeframe]
        chunks: list[pd.DataFrame] = []
        remaining = limit
        end_ms: int | None = None

        while remaining > 0:
            page_limit = min(remaining, self._MAX_KLINES_PER_CALL)
            kwargs = {
                "category": self.config.category,
                "symbol": symbol,
                "interval": interval,
                "limit": page_limit,
            }
            if end_ms is not None:
                kwargs["end"] = end_ms
            resp = self.client.get_kline(**kwargs)
            rows = resp["result"]["list"]
            if not rows:
                break
            df = pd.DataFrame(
                rows, columns=["ts", "open", "high", "low", "close", "volume", "turnover"]
            )
            df["ts"] = df["ts"].astype("int64")
            chunks.append(df)
            remaining -= len(rows)
            oldest_ts = df["ts"].min()
            end_ms = oldest_ts - 1
            if len(rows) < page_limit:
                break  # exchange has no more history than this

        if not chunks:
            return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

        combined = pd.concat(chunks, ignore_index=True).drop_duplicates(subset="ts")
        combined["datetime"] = pd.to_datetime(combined["ts"], unit="ms")
        for col in ("open", "high", "low", "close", "volume"):
            combined[col] = combined[col].astype(float)
        combined = combined.sort_values("datetime").tail(limit).reset_index(drop=True)
        return combined[["datetime", "open", "high", "low", "close", "volume"]]

    def get_last_price(self, symbol: str) -> float:
        """Public ticker endpoint — no API key needed. Used by paper
        trading to check open positions against live price between
        candle closes.
        """
        resp = self.client.get_tickers(category=self.config.category, symbol=symbol)
        return float(resp["result"]["list"][0]["lastPrice"])

    def get_instrument_info(self, symbol: str) -> InstrumentInfo:
        if symbol in self._instrument_cache:
            return self._instrument_cache[symbol]
        resp = self.client.get_instruments_info(category=self.config.category, symbol=symbol)
        item = resp["result"]["list"][0]
        info = InstrumentInfo(
            symbol=symbol,
            tick_size=float(item["priceFilter"]["tickSize"]),
            qty_step=float(item["lotSizeFilter"]["qtyStep"]),
            min_order_qty=float(item["lotSizeFilter"]["minOrderQty"]),
            max_leverage=float(item["leverageFilter"]["maxLeverage"]),
        )
        self._instrument_cache[symbol] = info
        return info

    def get_wallet_equity_usdt(self) -> float:
        if self.config.equity_override_usdt is not None:
            return self.config.equity_override_usdt
        if self.config.dry_run and not (self.config.api_key and self.config.api_secret):
            raise RuntimeError(
                "No API credentials and no EQUITY_OVERRIDE_USDT set. "
                "In dry-run without keys you must set EQUITY_OVERRIDE_USDT in .env "
                "to simulate an account size."
            )
        resp = self.client.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        coins = resp["result"]["list"][0]["coin"]
        return float(coins[0]["walletBalance"])

    def get_open_positions(self) -> list[dict]:
        if self.config.dry_run:
            return []
        resp = self.client.get_positions(category=self.config.category, settleCoin="USDT")
        return [p for p in resp["result"]["list"] if float(p.get("size", 0)) != 0]

    def set_leverage(self, symbol: str, leverage: int) -> None:
        if self.config.dry_run:
            logger.info("[DRY_RUN] would set leverage %sx on %s", leverage, symbol)
            return
        try:
            self.client.set_leverage(
                category=self.config.category,
                symbol=symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage),
            )
        except Exception as exc:  # Bybit errors if leverage already set to this value
            logger.debug("set_leverage(%s, %s) no-op/error: %s", symbol, leverage, exc)

    def round_qty(self, symbol: str, qty: float) -> float:
        info = self.get_instrument_info(symbol)
        step = info.qty_step
        rounded = (qty // step) * step
        return max(rounded, 0.0)

    def place_market_entry_with_stop(
        self, symbol: str, side: str, qty: float, stop_price: float
    ) -> dict | None:
        """side: 'long' or 'short'. Places a market order then attaches the
        stop-loss to the resulting position via set_trading_stop.
        """
        order_side = "Buy" if side == "long" else "Sell"
        if self.config.dry_run:
            logger.info(
                "[DRY_RUN] would MARKET %s %s qty=%s then set stopLoss=%s",
                order_side, symbol, qty, stop_price,
            )
            return None

        order = self.client.place_order(
            category=self.config.category,
            symbol=symbol,
            side=order_side,
            orderType="Market",
            qty=str(qty),
            reduceOnly=False,
        )
        self.client.set_trading_stop(
            category=self.config.category,
            symbol=symbol,
            stopLoss=str(stop_price),
            positionIdx=0,
        )
        return order

    def update_stop_loss(self, symbol: str, stop_price: float) -> None:
        if self.config.dry_run:
            logger.info("[DRY_RUN] would update stopLoss on %s to %s", symbol, stop_price)
            return
        self.client.set_trading_stop(
            category=self.config.category,
            symbol=symbol,
            stopLoss=str(stop_price),
            positionIdx=0,
        )

    def close_position_market(self, symbol: str, side: str, qty: float) -> None:
        """side is the side of the OPEN position; closing sends the
        opposite market order with reduceOnly=True.
        """
        close_side = "Sell" if side == "long" else "Buy"
        if self.config.dry_run:
            logger.info("[DRY_RUN] would MARKET close %s %s qty=%s", close_side, symbol, qty)
            return
        self.client.place_order(
            category=self.config.category,
            symbol=symbol,
            side=close_side,
            orderType="Market",
            qty=str(qty),
            reduceOnly=True,
        )
