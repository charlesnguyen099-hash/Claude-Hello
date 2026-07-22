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


def _sf(val, default: float = 0.0) -> float:
    """Safe float: Bybit tra ve '' (empty string) cho field khong co gia tri.
    float('') crash -> dung ham nay thay cho moi float(x.get(...)) tren Bybit data."""
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def retry(attempts: int = 3, delay: float = 2.0):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            for i in range(attempts):
                try:
                    return fn(*args, **kwargs)
                except ValueError:
                    # ValueError = gia/tham so khong hop le — retry cung khong giai quyet duoc
                    raise
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
                return _sf(c.get("equity", 0))
        return 0.0

    @retry()
    def get_positions(self) -> list[dict]:
        """Lấy tất cả vị thế đang mở."""
        resp = self.session.get_positions(category="linear", settleCoin="USDT")
        return [p for p in resp["result"]["list"] if _sf(p.get("size")) > 0]

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

    @retry(attempts=1)
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
            if lp <= 0:
                raise ValueError(f"place_order: limit_price={lp} invalid (<=0), abort")
            params["price"] = str(lp)

        # Dung is not None thay vi truthy check — sl=0.0 la falsy nhung la gia hop le
        if sl is not None or tp is not None:
            params["tpslMode"] = "Full"
        if sl is not None and sl > 0:
            sl_rounded = self.round_to_tick(sl, tick_size) if tick_size > 0 else round(sl, 6)
            if sl_rounded > 0:
                params["stopLoss"]    = str(sl_rounded)
                params["slTriggerBy"] = "MarkPrice"
        if tp is not None and tp > 0:
            tp_rounded = self.round_to_tick(tp, tick_size) if tick_size > 0 else round(tp, 6)
            if tp_rounded > 0:
                params["takeProfit"]  = str(tp_rounded)
                params["tpTriggerBy"] = "LastPrice"

        resp = self.session.place_order(**params)
        # Bybit v5 create-order response CHI tra orderId/orderLinkId — KHONG echo SL/TP.
        # (Check cu doc result["stopLoss"] luon rong → bao loi "KHONG SET SL/TP" sai.
        #  Viec xac nhan SL/TP thuc te do verify_position_tp_sl dam nhiem sau khi fill.)
        if params.get("stopLoss") or params.get("takeProfit"):
            logger.info(
                f"place_order {params['symbol']}: submitted with "
                f"SL={params.get('stopLoss','-')} TP={params.get('takeProfit','-')}"
            )
        return resp["result"]

    @retry()
    def close_position(self, symbol: str, side: str, qty: float) -> dict:
        close_side = "Sell" if side == "Buy" else "Buy"
        return self.place_order(symbol, close_side, qty, reduce_only=True)

    def _tick_round(self, price: float, tick_size: float) -> str:
        """Round price theo tick_size va tra ve string cho Bybit API.
        Neu tick_size=0: tu dong lay tu cache instrument info.
        Dam bao LUON tra ve gia hop le — tranh Bybit reject SL/TP vi sai decimal."""
        if tick_size > 0:
            return str(self.round_to_tick(price, tick_size))
        # fallback: xac dinh so decimal tu do lon gia
        if price >= 10:
            return str(round(price, 2))
        elif price >= 1:
            return str(round(price, 4))
        elif price >= 0.01:
            return str(round(price, 6))
        elif price >= 0.0001:
            return str(round(price, 8))
        else:
            return str(round(price, 10))

    @retry(attempts=5, delay=1.0)
    def set_sl_tp(self, symbol: str, sl_price: float, tp_price: float, tick_size: float = 0.0):
        """Set CA HAI SL va TP tren position (position-level).
        tpslMode='Full' bat buoc khi set ca hai cung luc tren Bybit V5.
        CRITICAL: tpslMode=Full voi chi 1 gia tri se XOA gia tri con lai.
        -> Tu dong lay gia tri hien tai tu exchange neu caller truyen 0."""
        # Auto-fetch missing value from exchange to avoid clearing it with tpslMode=Full
        if sl_price <= 0 or tp_price <= 0:
            try:
                resp = self.session.get_positions(category="linear", symbol=symbol)
                for p in resp["result"]["list"]:
                    if _sf(p.get("size")) > 0:
                        if sl_price <= 0:
                            sl_price = _sf(p.get("stopLoss"))
                        if tp_price <= 0:
                            tp_price = _sf(p.get("takeProfit"))
                        break
            except Exception:
                pass

        if sl_price <= 0 and tp_price <= 0:
            logger.warning(f"set_sl_tp {symbol}: ca SL va TP deu = 0 (kể cả exchange), skip")
            return

        if tick_size <= 0:
            try:
                info = self.get_instrument_info(symbol)
                tick_size = float(info["priceFilter"]["tickSize"])
            except Exception:
                tick_size = 0.0

        params: dict = dict(
            category="linear",
            symbol=symbol,
            tpslMode="Full",
            slTriggerBy="MarkPrice",
            tpTriggerBy="LastPrice",
            positionIdx=0,
        )
        if sl_price > 0:
            sl_rounded = self._tick_round(sl_price, tick_size)
            if float(sl_rounded) > 0:   # guard: sau khi round khong duoc = 0 (scientific notation bug)
                params["stopLoss"] = sl_rounded
            else:
                logger.warning(f"set_sl_tp {symbol}: sl_price={sl_price} round->0 (tick={tick_size}) — skip SL")
        if tp_price > 0:
            tp_rounded = self._tick_round(tp_price, tick_size)
            if float(tp_rounded) > 0:
                params["takeProfit"] = tp_rounded
            else:
                logger.warning(f"set_sl_tp {symbol}: tp_price={tp_price} round->0 (tick={tick_size}) — skip TP")

        # Neu khong co ca hai sau khi guard, skip luon (khong gui request vo nghia)
        if "stopLoss" not in params and "takeProfit" not in params:
            logger.warning(f"set_sl_tp {symbol}: ca SL va TP deu round ve 0 — skip request")
            return

        logger.info(f"set_sl_tp {symbol}: SL={params.get('stopLoss','(none)')} TP={params.get('takeProfit','(none)')} tick={tick_size}")
        # ErrCode -> no retry: gia khong hop le, retry cung that bai
        _NO_RETRY_CODES = {110084, 110085, 110043, 10001}
        try:
            resp = self.session.set_trading_stop(**params)
        except Exception as e:
            err_str = str(e).encode("ascii", "replace").decode()
            # ErrCode 34040 "not modified": gia tri da duoc set, khong can thay doi -> success
            if "34040" in str(e):
                logger.info(f"set_sl_tp {symbol}: 34040 not modified (SL/TP da dung gia tri nay) — OK")
                return
            raise
        ret_code = resp.get("retCode", -1)
        if ret_code != 0:
            msg = resp.get("retMsg", "")
            logger.error(f"set_sl_tp {symbol} FAILED retCode={ret_code} msg={msg} SL={params.get('stopLoss')} TP={params.get('takeProfit')}")
            print(f"[SL/TP ERROR] {symbol} retCode={ret_code} {msg}", flush=True)
            if ret_code in _NO_RETRY_CODES:
                # Gia sai huong hoac invalid — retry cung khong co ich, raise de caller xu ly
                raise ValueError(f"set_trading_stop price invalid retCode={ret_code}: {msg}")
            raise RuntimeError(f"set_trading_stop failed retCode={ret_code}: {msg}")

    def update_stop_loss(self, symbol: str, sl_price: float, tick_size: float = 0.0):
        """Cap nhat SL — lay TP hien tai tu exchange va goi set_sl_tp voi ca hai gia tri.
        tpslMode=Full: KHONG duoc set chi 1 gia tri vi Bybit se XOA gia tri con lai."""
        try:
            resp = self.session.get_positions(category="linear", symbol=symbol)
            tp_current = 0.0
            for p in resp["result"]["list"]:
                if _sf(p.get("size")) > 0:
                    tp_current = _sf(p.get("takeProfit"))
                    break
        except Exception:
            tp_current = 0.0
        self.set_sl_tp(symbol, sl_price, tp_current, tick_size=tick_size)

    def update_take_profit(self, symbol: str, tp_price: float, tick_size: float = 0.0):
        """Cap nhat TP — lay SL hien tai tu exchange va goi set_sl_tp voi ca hai gia tri.
        tpslMode=Full: KHONG duoc set chi 1 gia tri vi Bybit se XOA gia tri con lai."""
        try:
            resp = self.session.get_positions(category="linear", symbol=symbol)
            sl_current = 0.0
            for p in resp["result"]["list"]:
                if _sf(p.get("size")) > 0:
                    sl_current = _sf(p.get("stopLoss"))
                    break
        except Exception:
            sl_current = 0.0
        self.set_sl_tp(symbol, sl_current, tp_price, tick_size=tick_size)

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
            max_lev = int(_sf(info.get("leverageFilter", {}).get("maxLeverage", config.MAX_LEVERAGE), config.MAX_LEVERAGE))
            return min(max_lev, config.MAX_LEVERAGE)
        except Exception as e:
            logger.warning("Cannot get max leverage for %s: %s", symbol, str(e).encode("ascii", "replace").decode())
            return config.DEFAULT_LEVERAGE

    def get_current_price(self, symbol: str) -> float:
        """Lay gia mark price hien tai (real-time, khong cache) de check stale signal."""
        try:
            resp = self.session.get_tickers(category="linear", symbol=symbol)
            items = resp["result"]["list"]
            if items:
                return _sf(items[0].get("markPrice"))
        except Exception as e:
            logger.debug(f"get_current_price {symbol}: {e}")
        return 0.0

    def get_bid_ask(self, symbol: str) -> tuple[float, float]:
        """Lay best bid/ask tu orderbook (chinh xac hon tickers API)."""
        try:
            resp = self.session.get_orderbook(category="linear", symbol=symbol, limit=1)
            data = resp["result"]
            bids = data.get("b", [])
            asks = data.get("a", [])
            if bids and asks:
                bid = _sf(bids[0][0])
                ask = _sf(asks[0][0])
                if bid > 0 and ask > 0:
                    return bid, ask
        except Exception as e:
            logger.debug(f"get_bid_ask orderbook {symbol}: {e}")
        # Fallback: tickers markPrice
        try:
            resp  = self.session.get_tickers(category="linear", symbol=symbol)
            items = resp["result"]["list"]
            if items:
                mark = _sf(items[0].get("markPrice"))
                if mark > 0:
                    return mark * 0.9999, mark * 1.0001
        except Exception as e:
            logger.debug(f"get_bid_ask tickers {symbol}: {e}")
        return 0.0, 0.0

    @staticmethod
    def round_to_tick(price: float, tick_size: float, ceil: bool = False) -> float:
        """Round price theo tick_size.
        ceil=False (default): floor — dung cho TP, LONG SL (di xa khoi entry)
        ceil=True: ceiling — dung cho SHORT SL (phai o TREN entry, floor lam chat SL)
        """
        import math
        if tick_size <= 0:
            return price
        ticks = math.ceil(price / tick_size) if ceil else math.floor(price / tick_size)
        result = round(ticks * tick_size, 10)
        # Fix: str(1e-05)="1e-05" khong chua "." -> decimals=0 -> round(0.0722,0)=0.0
        # Dung format fixed-decimal de xu ly scientific notation chinh xac
        tick_str = f"{tick_size:.10f}".rstrip("0")
        decimals = len(tick_str.split(".")[-1]) if "." in tick_str else 0
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
                if _sf(p.get("size")) > 0:
                    sl = _sf(p.get("stopLoss"))
                    return sl > 0, sl
        except Exception as e:
            logger.debug(f"verify_position_sl {symbol}: {e}")
        return False, 0.0

    def verify_position_tp_sl(self, symbol: str) -> tuple[bool, float, bool, float]:
        """Xac nhan vi the co ca SL va TP dang hoat dong.
        Return (has_sl, sl_price, has_tp, tp_price)."""
        try:
            resp = self.session.get_positions(category="linear", symbol=symbol)
            for p in resp["result"]["list"]:
                if _sf(p.get("size")) > 0:
                    sl = _sf(p.get("stopLoss"))
                    tp = _sf(p.get("takeProfit"))
                    return sl > 0, sl, tp > 0, tp
        except Exception as e:
            logger.debug(f"verify_position_tp_sl {symbol}: {e}")
        return False, 0.0, False, 0.0

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
                total += _sf(item.get("closedPnl"))
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
                    pnl = _sf(item.get("closedPnl"))
                    if sym not in result:  # lay lenh gan nhat
                        result[sym] = pnl
        except Exception as e:
            logger.debug(f"get_closed_pnl error: {e}")
        return result

