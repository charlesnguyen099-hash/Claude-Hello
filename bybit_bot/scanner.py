"""
Market Scanner — 2 danh sach song song:
  1. top_volume: top N theo volume 24h (thanh khoan cao nhat) — dung cho priority
  2. top_trending: top 20 theo trending score — coin dang duoc chu y nhat
     Trending score = vol_rank×0.4 + volatility_rank×0.4 + liquidity_rank×0.2
     (volatility = abs(price_change_24h%), liquidity = 1/spread)
"""

import logging
from dataclasses import dataclass

import pandas as pd

import config
from client import BybitClient

logger = logging.getLogger(__name__)

TRENDING_TOP_N = 20   # so coin trong trending list


@dataclass
class SymbolInfo:
    symbol: str
    volume_usdt_24h: float
    price_change_pct: float
    last_price: float
    bid_ask_spread_pct: float
    score: float = 0.0


class MarketScanner:
    def __init__(self, client: BybitClient):
        self.client = client
        self.trending_symbols: list[str] = []   # top20 trending (cap nhat moi gio)

    def scan(self) -> list[str]:
        """Tra ve danh sach top N symbols theo volume (priority list)."""
        try:
            tickers = self.client.get_tickers()
        except Exception as e:
            logger.error(f"Scanner failed to get tickers: {e}")
            return []

        symbols: list[SymbolInfo] = []
        for t in tickers:
            sym = t.get("symbol", "")
            if not sym.endswith("USDT"):
                continue
            if any(x in sym for x in ["UP", "DOWN", "BULL", "BEAR", "1000"]):
                continue

            try:
                vol    = float(t.get("turnover24h", 0))
                change = abs(float(t.get("price24hPcnt", 0)) * 100)
                price  = float(t.get("lastPrice", 0))
                bid    = float(t.get("bid1Price", price * 0.999))
                ask    = float(t.get("ask1Price", price * 1.001))
                spread = (ask - bid) / price * 100 if price > 0 else 99

                if vol < config.MIN_VOLUME_USDT_24H or price <= 0:
                    continue

                # Loai coin da pump/dump qua manh trong 24h (move da xong, vao muon)
                # AKE +324%, BILL -45% — cac truong hop nay xac suat reverse cao, khong nen trade theo trend
                raw_change = float(t.get("price24hPcnt", 0)) * 100  # signed (+ = up, - = down)
                if abs(raw_change) > 25:
                    logger.debug(f"Scanner skip {sym}: 24h change={raw_change:.1f}% (>25%, move exhausted)")
                    continue

                symbols.append(SymbolInfo(
                    symbol=sym,
                    volume_usdt_24h=vol,
                    price_change_pct=change,
                    last_price=price,
                    bid_ask_spread_pct=spread,
                ))
            except (ValueError, TypeError):
                continue

        if not symbols:
            logger.warning("No symbols passed filter")
            return []

        df = pd.DataFrame([s.__dict__ for s in symbols])

        # --- TOP VOLUME LIST (priority coins) ---
        top_vol = df.nlargest(config.TOP_N_SYMBOLS, "volume_usdt_24h")
        result  = top_vol["symbol"].tolist()
        self.volume_map: dict[str, float] = dict(
            zip(top_vol["symbol"], top_vol["volume_usdt_24h"])
        )

        # --- TRENDING LIST ---
        # Chi tinh trong tap coin du vol toi thieu (da loc o tren)
        # Rank tung tieu chi (1 = tot nhat), tinh weighted score
        df["vol_rank"]        = df["volume_usdt_24h"].rank(ascending=False)
        df["volatility_rank"] = df["price_change_pct"].rank(ascending=False)
        # Liquidity: spread cang nho cang tot → rank ascending=True
        df["liquidity_rank"]  = df["bid_ask_spread_pct"].rank(ascending=True)

        n = len(df)
        # Normalize rank ve [0,1]: 0 = tot nhat, 1 = kem nhat
        df["trending_score"] = (
            (df["vol_rank"]        / n) * 0.4 +
            (df["volatility_rank"] / n) * 0.4 +
            (df["liquidity_rank"]  / n) * 0.2
        )

        # Loc them: chi lay coin co vol >= 10M de dam bao thanh khoan
        df_trending = df[df["volume_usdt_24h"] >= 10_000_000].copy()
        top_trend   = df_trending.nsmallest(TRENDING_TOP_N, "trending_score")
        self.trending_symbols = top_trend["symbol"].tolist()

        logger.info(
            f"Scanner: top_volume={len(result)} | "
            f"trending top{TRENDING_TOP_N}: {self.trending_symbols[:5]}..."
        )
        return result
