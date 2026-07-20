"""
Market Scanner — Tat ca coin co TRENDING (khong gioi han so luong).

Dinh nghia TRENDING (4 tieu chi, score 0-100):
  1. Price Momentum  (40%): % thay doi gia 24h — coin dang tang/giam manh
  2. Volume Surge    (30%): volume so voi nguong toi thieu — dong tien dang do vao
  3. Liquidity       (20%): spread nho = thanh khoan tot = de vao/ra
  4. Volume size     (10%): tong volume 24h — coin dang active

Tra ve TAT CA coin qua filter, sap xep theo trending_score giam dan.
Coin score cao nhat duoc phan tich truoc trong moi tick.
Loai tru: coin pump/dump > 25% 24h (move da xong), leverage token, volume < 100K,
          coin nho (vol<10M) khong co momentum (|chg|<1%) hoac spread > 0.5%.
"""

import logging
from dataclasses import dataclass

import pandas as pd

import config
from client import BybitClient, _sf

logger = logging.getLogger(__name__)


@dataclass
class SymbolInfo:
    symbol: str
    volume_usdt_24h: float
    price_change_pct: float   # abs(24h change %)
    price_change_raw: float   # signed 24h change %
    last_price: float
    bid_ask_spread_pct: float
    volume_ratio: float       # volume / min threshold (surge proxy)
    trending_score: float = 0.0


class MarketScanner:
    def __init__(self, client: BybitClient):
        self.client = client
        self.trending_symbols: list[str] = []   # tat ca coin trending, sap xep theo score
        self.volume_map: dict[str, float] = {}

    def scan(self) -> list[str]:
        """Tra ve TAT CA symbols co trending, sap xep score cao nhat truoc."""
        try:
            tickers = self.client.get_tickers()
        except Exception as e:
            logger.error(f"Scanner failed to get tickers: {e}")
            return self.trending_symbols

        symbols: list[SymbolInfo] = []
        for t in tickers:
            sym = t.get("symbol", "")
            if not sym.endswith("USDT"):
                continue
            # Loai leverage token va stable pairs (1000PEPE, 1000BONK la coin that — KHONG loai)
            if any(x in sym for x in ["3LUSDT", "3SUSDT", "UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT", "USDC", "BUSD", "TUSD"]):
                continue

            try:
                vol     = _sf(t.get("turnover24h"))
                raw_chg = _sf(t.get("price24hPcnt")) * 100
                price   = _sf(t.get("lastPrice"))
                bid     = _sf(t.get("bid1Price")) or price * 0.999
                ask     = _sf(t.get("ask1Price")) or price * 1.001
                spread  = (ask - bid) / price * 100 if price > 0 else 99

                vol_surge = vol / max(config.MIN_VOLUME_USDT_24H, 1)

                if vol < config.MIN_VOLUME_USDT_24H or price <= 0:
                    continue

                # Quality gate cho coin nho (vol < 10M):
                # phai co momentum ro rang va spread du chat luong
                if vol < 10_000_000:
                    if abs(raw_chg) < 1.0:   # gia phai di chuyen it nhat 1% trong 24h
                        continue
                    if spread > 0.5:          # spread > 0.5% = kho execute, bo qua
                        continue

                # Loai coin pump/dump xong (>25% trong 24h — momentum can)
                if abs(raw_chg) > 25:
                    logger.debug(f"Scanner skip {sym}: 24h change={raw_chg:.1f}% > 25%")
                    continue

                symbols.append(SymbolInfo(
                    symbol=sym,
                    volume_usdt_24h=vol,
                    price_change_pct=abs(raw_chg),
                    price_change_raw=raw_chg,
                    last_price=price,
                    bid_ask_spread_pct=spread,
                    volume_ratio=vol_surge,
                ))
            except (ValueError, TypeError):
                continue

        if not symbols:
            logger.warning("No symbols passed filter — keeping previous list")
            return self.trending_symbols

        df = pd.DataFrame([s.__dict__ for s in symbols])
        n  = len(df)

        # TRENDING SCORE (0-100): rank-based, coin co score cao nhat duoc xu ly truoc
        df["r_momentum"] = df["price_change_pct"].rank(ascending=False)
        df["s_momentum"] = (1 - (df["r_momentum"] - 1) / n) * 40   # 0-40

        df["r_volume"] = df["volume_ratio"].rank(ascending=False)
        df["s_volume"] = (1 - (df["r_volume"] - 1) / n) * 30   # 0-30

        df["r_liq"] = df["bid_ask_spread_pct"].rank(ascending=True)
        df["s_liq"] = (1 - (df["r_liq"] - 1) / n) * 20   # 0-20

        df["r_vol24"] = df["volume_usdt_24h"].rank(ascending=False)
        df["s_vol24"] = (1 - (df["r_vol24"] - 1) / n) * 10   # 0-10

        df["trending_score"] = df["s_momentum"] + df["s_volume"] + df["s_liq"] + df["s_vol24"]

        # Sap xep giam dan theo trending_score — tat ca coin, khong cat gioi han
        sorted_df = df.sort_values("trending_score", ascending=False)
        # BTC và ETH luôn được scan trước (dù momentum 24h thấp, chúng drive toàn thị trường)
        _pinned = [s for s in ("BTCUSDT", "ETHUSDT") if s in sorted_df["symbol"].values]
        _rest   = [s for s in sorted_df["symbol"].tolist() if s not in _pinned]
        self.trending_symbols = _pinned + _rest
        self.volume_map = dict(zip(df["symbol"], df["volume_usdt_24h"]))

        top5_info = ", ".join(
            f"{r['symbol']}({r['price_change_pct']:.1f}%,{r['trending_score']:.0f}pt)"
            for _, r in sorted_df.head(5).iterrows()
        )
        logger.info(
            f"Scanner: {n} coins eligible (sorted by trend score) | "
            f"Top5: {top5_info}"
        )
        return self.trending_symbols
