"""
Bybit API Client Wrapper — dùng pybit v5
Lấy dữ liệu trực tiếp từ Bybit, không lưu local (tiết kiệm storage)
"""

import time
import logging
from functools import wraps
from typing import Optional

import pandas as pd
from pybit.unified_trading import HTTP

import config

logger = logging.getLogger(__name__)

# Cache instrument info để không gọi API lặp lại (reset mỗi lần khởi động)
_instrument_cache: dict[str, dict] = {}


def retry(attempts: int = 3, delay: float = 2.0):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            for i in range(attempts):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    if i == attempts - 1:
                        raise
                    logger.warning(f"Retry {i+1}/{attempts} for {fn.__name__}: {str(e).encode('ascii','replace').decode()}")
                    time.sleep(delay * (2 ** i))
        return wrapper
    return decorator


class BybitClient:
    def __init__(self):
        self.session = HTTP(
            testnet=config.TESTNET,
            api_key=config.API_KEY,
            api_secret=config.API_SECRET,
        )

    # ── Market Data ──────────────────────────────────────────────────────────

    @retry()
    def get_tickers(self) -> list[dict]:
        """Lấy tất cả ticker linear perpetual."""
        resp = self.session.get_tickers(category="linear")
        return resp["result"]["list"]

    @retry()
    def get_klines(self, symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
        """
        Lấy nến OHLCV từ Bybit API — không lưu local.
        Bybit trả về dữ liệu theo thứ tự mới nhất trước.
        """
        resp = self.session.get_kline(
            category="linear",
            symbol=symbol,
            interval=interval,
            limit=limit,
        )
        rows = resp["result"]["list"]
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
        df = df.astype({
            "timestamp": "int64",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    @retry()
    def get_orderbook(self, symbol: str, limit: int = 25) -> dict:
        resp = self.session.get_orderbook(category="linear", symbol=symbol, limit=limit)
        return resp["result"]

    # ── Account ───────────────────────────────────────────────────────────────

    @retry()
    def get_wallet_balance(self) -> float:
        """Trả về tổng equity USDT."""
        resp = self.session.get_wallet_balance(accountType="UNIFIED")
        coins = resp["result"]["list"][0]["coin"]
        for c in coins:
            if c["coin"] == "USDT":
                return float(c["equity"])
        return 0.0

    @retry()
    def get_positions(self) -> list[dict]:
        """Lấy tất cả vị thế đang mở."""
        resp = self.session.get_positions(category="linear", settleCoin="USDT")
        return [p for p in resp["result"]["list"] if float(p["size"]) > 0]

    @retry()
    def get_open_orders(self, symbol: Optional[str] = None) -> list[dict]:
        params = {"category": "linear", "settleCoin": "USDT"}
        if symbol:
            params["symbol"] = symbol
        resp = self.session.get_open_orders(**params)
        return resp["result"]["list"]

    # ── Trading ───────────────────────────────────────────────────────────────

    @retry()
    def set_leverage(self, symbol: str, leverage: int):
        try:
            self.session.set_leverage(
                category="linear",
                symbol=symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage),
            )
        except Exception as e:
            if "leverage not modified" in str(e).lower():
                pass
            else:
                raise

    @retry()
    def place_order(
        self,
        symbol: str,
        side: str,          # "Buy" | "Sell"
        qty: float,
        order_type: str = "Market",
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        reduce_only: bool = False,
    ) -> dict:
        params = dict(
            category="linear",
            symbol=symbol,
            side=side,
            orderType=order_type,
            qty=str(qty),
            timeInForce="GoodTillCancel",
            reduceOnly=reduce_only,
            positionIdx=0,  # one-way mode
        )
        if sl:
            params["stopLoss"] = str(round(sl, 6))
            params["slTriggerBy"] = "MarkPrice"
        if tp:
            params["takeProfit"] = str(round(tp, 6))
            params["tpTriggerBy"] = "MarkPrice"

        resp = self.session.place_order(**params)
        return resp["result"]

    @retry()
    def close_position(self, symbol: str, side: str, qty: float) -> dict:
        close_side = "Sell" if side == "Buy" else "Buy"
        return self.place_order(symbol, close_side, qty, reduce_only=True)

    @retry()
    def cancel_all_orders(self, symbol: str):
        self.session.cancel_all_orders(category="linear", symbol=symbol)

    @retry()
    def set_trading_stop(self, symbol: str, side: str, trailing_stop: float):
        """Đặt trailing stop cho vị thế."""
        try:
            self.session.set_trading_stop(
                category="linear",
                symbol=symbol,
                trailingStop=str(round(trailing_stop, 6)),
                positionIdx=0,
            )
        except Exception as e:
            logger.warning(f"Trailing stop failed for {symbol}: {e}")

    @retry()
    def get_instrument_info(self, symbol: str) -> dict:
        if symbol in _instrument_cache:
            return _instrument_cache[symbol]
        resp = self.session.get_instruments_info(category="linear", symbol=symbol)
        info = resp["result"]["list"][0]
        _instrument_cache[symbol] = info
        return info

    def get_max_leverage(self, symbol: str) -> int:
        """Lấy leverage tối đa Bybit cho phép với symbol này."""
        try:
            info = self.get_instrument_info(symbol)
            max_lev = int(float(info["leverageFilter"]["maxLeverage"]))
            return min(max_lev, config.MAX_LEVERAGE)
        except Exception as e:
            logger.warning(f"Cannot get max leverage for {symbol}: {e}")
            return config.DEFAULT_LEVERAGE

    def get_min_order_usdt(self, symbol: str) -> float:
        """Lấy giá trị lệnh tối thiểu (USDT) của symbol."""
        try:
            info = self.get_instrument_info(symbol)
            min_qty   = float(info["lotSizeFilter"]["minOrderQty"])
            # Lấy giá hiện tại để tính notional tối thiểu
            tickers = self.session.get_tickers(category="linear", symbol=symbol)
            price = float(tickers["result"]["list"][0]["lastPrice"])
            return min_qty * price
        except Exception:
            return 1.0
