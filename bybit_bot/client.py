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
    def update_stop_loss(self, symbol: str, sl_price: float):
        """Cap nhat SL cho vi the dang mo (dung de doi SL ve break-even)."""
        try:
            self.session.set_trading_stop(
                category="linear",
                symbol=symbol,
                stopLoss=str(round(sl_price, 6)),
                slTriggerBy="MarkPrice",
                positionIdx=0,
            )
        except Exception as e:
            logger.warning(f"Update SL failed for {symbol}: {e}")

    @retry()
    def update_take_profit(self, symbol: str, tp_price: float):
        """Cap nhat TP cho vi the dang mo (dung de chuyen tu TP1 sang TP2)."""
        try:
            self.session.set_trading_stop(
                category="linear",
                symbol=symbol,
                takeProfit=str(round(tp_price, 6)),
                tpTriggerBy="MarkPrice",
                positionIdx=0,
            )
        except Exception as e:
            logger.warning(f"Update TP failed for {symbol}: {e}")

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

    def get_closed_pnl(self, symbols: list[str]) -> dict[str, float]:
        """Lay closed PnL cua cac symbol vua dong lenh (trong 5 phut gan nhat).
        Tra ve {symbol: pnl} cho cac symbol co trong danh sach."""
        result = {}
        try:
            resp = self.session.get_closed_pnl(
                category="linear",
                limit=50,
            )
            for item in resp["result"]["list"]:
                sym = item.get("symbol", "")
                if sym in symbols:
                    pnl = float(item.get("closedPnl", 0))
                    if sym not in result:  # lay lenh gan nhat
                        result[sym] = pnl
        except Exception as e:
            logger.debug(f"get_closed_pnl error: {e}")
        return result

