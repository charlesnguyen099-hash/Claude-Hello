"""
Market Scanner — tự động chọn top 50 cặp linear tốt nhất theo:
  1. Volume 24h (thanh khoản cao)
  2. Biến động 24h (cơ hội profit)
  3. Spread thấp (chi phí giao dịch thấp)
  Score = Volume_rank × 0.5 + Volatility_rank × 0.4 + Liquidity_rank × 0.1
"""

import logging
from dataclasses import dataclass

import pandas as pd

import config
from client import BybitClient

logger = logging.getLogger(__name__)


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

    def scan(self) -> list[str]:
        """Trả về danh sách top N symbols được xếp hạng tốt nhất."""
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
            # Loại bỏ các cặp leverage token, index
            if any(x in sym for x in ["UP", "DOWN", "BULL", "BEAR", "1000"]):
                continue

            try:
                vol = float(t.get("turnover24h", 0))
                change = abs(float(t.get("price24hPcnt", 0)) * 100)
                price = float(t.get("lastPrice", 0))
                bid = float(t.get("bid1Price", price * 0.999))
                ask = float(t.get("ask1Price", price * 1.001))
                spread = (ask - bid) / price * 100 if price > 0 else 99

                if vol < config.MIN_VOLUME_USDT_24H or price <= 0:
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

        # Chuẩn hoá rank 0-1 (rank cao = tốt hơn)
        df["vol_rank"]    = df["volume_usdt_24h"].rank(pct=True)
        df["chg_rank"]    = df["price_change_pct"].rank(pct=True)
        df["spread_rank"] = (1 - df["bid_ask_spread_pct"].rank(pct=True))  # spread nhỏ = tốt

        df["score"] = (
            df["vol_rank"]    * 0.50 +
            df["chg_rank"]    * 0.40 +
            df["spread_rank"] * 0.10
        )

        top = df.nlargest(config.TOP_N_SYMBOLS, "score")
        result = top["symbol"].tolist()

        logger.info(
            f"Scanner selected {len(result)} symbols. "
            f"Top 5: {result[:5]}"
        )
        return result
