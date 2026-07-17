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
        Lay nen OHLCV tu Bybit API — khong luu local.
        Bybit tra ve du lieu theo thu tu moi nhat truoc.
        """
        resp = self.session.get_kline(
            category="linear",
            symbol=symbol,
            interval=interval,
            limit=min(limit, 1000),
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

    def get_klines_paginated(self, symbol: str, interval: str, total_limit: int = 2000) -> pd.DataFrame:
        """
        Lay nhieu trang nen OHLCV bang cach phan trang nguoc ve qua khu.
        Bybit max 1000 nen/call -> de lay 2000 nen can 2 calls.
        Dung cho 1m signal timeframe de co du lich su indicator (EMA, VWAP, v.v.).
        """
        MAX_PER_CALL = 1000
        all_rows: list = []
        end_ts: int | None = None   # phan trang: end timestamp cho call tiep theo
        remaining = total_limit

        while remaining > 0:
            limit = min(remaining, MAX_PER_CALL)
            params: dict = dict(category="linear", symbol=symbol, interval=interval, limit=limit)
            if end_ts is not None:
                params["end"] = end_ts
            try:
                resp = self.session.get_kline(**params)
                rows = resp["result"]["list"]
            except Exception as e:
                logger.warning(f"get_klines_paginated {symbol} {interval}: {str(e).encode('ascii','replace').decode()}")
                break
            if not rows:
                break
            all_rows.extend(rows)
            remaining -= len(rows)
            if len(rows) < limit:
                break   # het data
            # Nen cu nhat trong batch nay: timestamp rows[-1][0] (Bybit sort moi truoc)
            end_ts = int(rows[-1][0]) - 1  # -1ms de tranh trung lap
            if remaining <= 0:
                break

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
        df = df.astype({
            "timestamp": "int64",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
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
        limit_price: Optional[float] = None,  # Dat khi dung Limit/IOC — rounds theo tick
        tick_size: float = 0.0,               # Can thiet de round limit_price chinh xac
    ) -> dict:
        # Limit order: dung IOC (fill ngay hoac huy — tranh lenh treo)
        # Market order: GTC (standard)
        if order_type == "Limit" and limit_price is not None:
            time_in_force = "IOC"
        else:
            time_in_force = "GoodTillCancel"

        params = dict(
            category="linear",
            symbol=symbol,
            side=side,
            orderType=order_type,
            qty=str(qty),
            timeInForce=time_in_force,
            reduceOnly=reduce_only,
            positionIdx=0,  # one-way mode
        )
        if order_type == "Limit" and limit_price is not None:
            lp = self.round_to_tick(limit_price, tick_size) if tick_size > 0 else round(limit_price, 6)
            params["price"] = str(lp)

        if sl:
            # Round SL theo tick size neu co
            sl_rounded = self.round_to_tick(sl, tick_size) if tick_size > 0 else round(sl, 6)
            params["stopLoss"] = str(sl_rounded)
            params["slTriggerBy"] = "MarkPrice"
        if tp:
            tp_rounded = self.round_to_tick(tp, tick_size) if tick_size > 0 else round(tp, 6)
            params["takeProfit"] = str(tp_rounded)
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

    def get_current_price(self, symbol: str) -> float:
        """Lay gia mark price hien tai (real-time, khong cache) de check stale signal."""
        try:
            resp = self.session.get_tickers(category="linear", symbol=symbol)
            items = resp["result"]["list"]
            if items:
                return float(items[0].get("markPrice", 0))
        except Exception as e:
            logger.debug(f"get_current_price {symbol}: {e}")
        return 0.0

    def get_bid_ask(self, symbol: str) -> tuple[float, float]:
        """Lay best bid/ask hien tai. Return (bid, ask), (0,0) neu loi."""
        try:
            resp = self.session.get_tickers(category="linear", symbol=symbol)
            items = resp["result"]["list"]
            if items:
                bid = float(items[0].get("bid1Price", 0))
                ask = float(items[0].get("ask1Price", 0))
                return bid, ask
        except Exception as e:
            logger.debug(f"get_bid_ask {symbol}: {e}")
        return 0.0, 0.0

    @staticmethod
    def round_to_tick(price: float, tick_size: float) -> float:
        """Round price xuong boi so gan nhat cua tick_size (floor)."""
        import math
        if tick_size <= 0:
            return price
        ticks = math.floor(price / tick_size)
        result = round(ticks * tick_size, 10)
        # Trim floating point noise
        decimals = len(str(tick_size).rstrip("0").split(".")[-1]) if "." in str(tick_size) else 0
        return round(result, decimals)

    def get_order_status(self, symbol: str, order_id: str) -> str:
        """Kiem tra trang thai lenh (Filled / Cancelled / PartiallyFilled / ...).
        Dung de xac nhan IOC Limit order co duoc fill hay bi huy."""
        try:
            resp = self.session.get_order_history(
                category="linear",
                symbol=symbol,
                orderId=order_id,
                limit=1,
            )
            items = resp["result"]["list"]
            if items:
                return items[0].get("orderStatus", "Unknown")
        except Exception as e:
            logger.debug(f"get_order_status {symbol} {order_id}: {e}")
        return "Unknown"

    def verify_position_sl(self, symbol: str) -> tuple[bool, float]:
        """Xac nhan vi the co SL dang hoat dong. Return (has_sl, sl_price)."""
        try:
            resp = self.session.get_positions(category="linear", symbol=symbol)
            for p in resp["result"]["list"]:
                if float(p.get("size", 0)) > 0:
                    sl = float(p.get("stopLoss", 0))
                    return sl > 0, sl
        except Exception as e:
            logger.debug(f"verify_position_sl {symbol}: {e}")
        return False, 0.0

    def get_today_pnl(self) -> float:
        """Tong realized PnL hom nay (UTC 00:00 den gio hien tai).
        Am = dang lo trong ngay, duong = dang co lai."""
        from datetime import datetime, timezone
        today_start_ms = int(
            datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000
        )
        total = 0.0
        try:
            resp = self.session.get_closed_pnl(
                category="linear",
                startTime=today_start_ms,
                limit=200,
            )
            for item in resp["result"]["list"]:
                total += float(item.get("closedPnl", 0))
        except Exception as e:
            logger.debug(f"get_today_pnl error: {e}")
        return total

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

