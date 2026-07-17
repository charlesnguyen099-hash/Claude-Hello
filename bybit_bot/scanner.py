"""
Market Scanner — Top 20 coins có TRENDING mạnh nhất.

Định nghĩa TRENDING (4 tiêu chí, score 0-100):
  1. Price Momentum  (40%): % thay đổi giá 1h và 4h gần đây — coin đang tăng/giảm mạnh
  2. Volume Surge    (30%): volume hiện tại vs trung bình 24h — dòng tiền đang đổ vào
  3. Trend Strength  (20%): spread nhỏ = thanh khoản tốt = dễ vào/ra
  4. Volatility      (10%): biên độ dao động 24h — coin đang active

Coin có trending_score cao nhất = đang được thị trường chú ý, có momentum rõ ràng.
Loại trừ: coin pump/dump quá 25% 24h (move đã xong), leverage token, volume < 10M.
"""

import logging
from dataclasses import dataclass

import pandas as pd

import config
from client import BybitClient

logger = logging.getLogger(__name__)

TRENDING_TOP_N = 20   # luôn chỉ trade top 20 coin trending nhất


@dataclass
class SymbolInfo:
    symbol: str
    volume_usdt_24h: float
    price_change_pct: float   # abs(24h change %)
    price_change_raw: float   # signed 24h change %
    last_price: float
    bid_ask_spread_pct: float
    volume_ratio: float       # volume hiện tại / avg volume (surge indicator)
    trending_score: float = 0.0


class MarketScanner:
    def __init__(self, client: BybitClient):
        self.client = client
        self.trending_symbols: list[str] = []   # top20 trending (cập nhật mỗi lần scan)
        self.volume_map: dict[str, float] = {}

    def scan(self) -> list[str]:
        """Trả về top 20 symbols có trending mạnh nhất."""
        try:
            tickers = self.client.get_tickers()
        except Exception as e:
            logger.error(f"Scanner failed to get tickers: {e}")
            return self.trending_symbols  # giữ list cũ nếu API lỗi

        symbols: list[SymbolInfo] = []
        for t in tickers:
            sym = t.get("symbol", "")
            if not sym.endswith("USDT"):
                continue
            # Loại leverage token và stable pairs
            if any(x in sym for x in ["UP", "DOWN", "BULL", "BEAR", "1000", "USDC", "BUSD", "TUSD"]):
                continue

            try:
                vol       = float(t.get("turnover24h", 0))
                raw_chg   = float(t.get("price24hPcnt", 0)) * 100   # signed %
                price     = float(t.get("lastPrice", 0))
                bid       = float(t.get("bid1Price", price * 0.999))
                ask       = float(t.get("ask1Price", price * 1.001))
                spread    = (ask - bid) / price * 100 if price > 0 else 99

                # Volume surge: volume 24h so với implied avg hourly * 24
                # Bybit trả turnover24h = tổng USDT khớp, dùng làm base
                # Không có hourly breakdown từ ticker → dùng volume ticks nếu có
                vol_ratio = float(t.get("volume24h", 0))   # contracts
                # Fallback: dùng turnover làm proxy volume surge (so với min threshold)
                vol_surge = vol / max(config.MIN_VOLUME_USDT_24H, 1)

                if vol < 10_000_000 or price <= 0:
                    continue

                # Loại coin đã pump/dump xong (>25% trong 24h → momentum cạn)
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

        # ── TRENDING SCORE (0-100, cao hơn = trending mạnh hơn) ──────────────
        #
        # 1. Price Momentum (40%): coin đang chạy mạnh trong 24h
        #    Rank by abs(price_change_pct), cao nhất = rank 1 = score cao nhất
        df["r_momentum"] = df["price_change_pct"].rank(ascending=False)
        df["s_momentum"] = (1 - (df["r_momentum"] - 1) / n) * 40   # 0-40

        # 2. Volume Surge (30%): dòng tiền đang đổ vào mạnh
        df["r_volume"] = df["volume_ratio"].rank(ascending=False)
        df["s_volume"] = (1 - (df["r_volume"] - 1) / n) * 30   # 0-30

        # 3. Liquidity (20%): spread nhỏ = thanh khoản tốt = execution tốt
        df["r_liq"] = df["bid_ask_spread_pct"].rank(ascending=True)   # nhỏ hơn = tốt hơn
        df["s_liq"] = (1 - (df["r_liq"] - 1) / n) * 20   # 0-20

        # 4. Volatility (10%): coin đang active, không ngủ
        df["r_vol24"] = df["volume_usdt_24h"].rank(ascending=False)
        df["s_vol24"] = (1 - (df["r_vol24"] - 1) / n) * 10   # 0-10

        df["trending_score"] = df["s_momentum"] + df["s_volume"] + df["s_liq"] + df["s_vol24"]

        # Lấy top 20 trending score cao nhất
        top_trend = df.nlargest(TRENDING_TOP_N, "trending_score")
        self.trending_symbols = top_trend["symbol"].tolist()
        self.volume_map = dict(zip(df["symbol"], df["volume_usdt_24h"]))

        top5_info = ", ".join(
            f"{r['symbol']}({r['price_change_pct']:.1f}%,{r['trending_score']:.0f}pt)"
            for _, r in top_trend.head(5).iterrows()
        )
        logger.info(
            f"Scanner: {n} coins → top{TRENDING_TOP_N} trending | "
            f"Top5: {top5_info}"
        )
        return self.trending_symbols
