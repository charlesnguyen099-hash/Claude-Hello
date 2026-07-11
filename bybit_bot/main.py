"""
Bybit Futures Auto Trading Bot — Main Entry Point
Chạy 24/7, tự động quét top 50 symbol, chọn strategy tốt nhất, quản lý rủi ro.

Usage:
    export BYBIT_API_KEY="your_key"
    export BYBIT_API_SECRET="your_secret"
    export BYBIT_TESTNET="true"    # Bỏ dòng này khi dùng mainnet
    python main.py
"""

import logging
import sys
import time
import traceback
from datetime import datetime, timezone

import config
from bot_logger import setup_logging, BotLogger
from client import BybitClient
from executor import Executor
from risk_manager import RiskManager
from scanner import MarketScanner
from selector import StrategySelector

setup_logging()
logger = logging.getLogger(__name__)


class TradingBot:
    def __init__(self):
        logger.info("="*60)
        logger.info("Bybit Auto Trading Bot starting...")
        logger.info(f"Mode: {'TESTNET' if config.TESTNET else '⚠️  MAINNET'}")
        logger.info(f"Top N symbols: {config.TOP_N_SYMBOLS}")
        logger.info(f"Max positions: {config.MAX_OPEN_POSITIONS}")
        logger.info(f"Risk per trade: {config.ACCOUNT_RISK_PCT*100:.1f}%")
        logger.info("="*60)

        self.client    = BybitClient()
        self.scanner   = MarketScanner(self.client)
        self.selector  = StrategySelector()
        self.risk_mgr  = RiskManager(self.client)
        self.bot_logger = BotLogger()
        self.executor  = Executor(self.client, self.risk_mgr, self.bot_logger)

        self.symbols: list[str] = []
        self.last_scan_ts: float = 0

    # ── Main loop ────────────────────────────────────────────────────────────

    def run(self):
        while True:
            try:
                self._tick()
            except KeyboardInterrupt:
                logger.info("Bot stopped by user.")
                sys.exit(0)
            except Exception as e:
                logger.error(f"Unhandled error in main loop: {e}\n{traceback.format_exc()}")

            time.sleep(config.LOOP_INTERVAL_SEC)

    def _tick(self):
        now = time.time()

        # ── 1. Cập nhật danh sách symbol mỗi 1 giờ ──────────────────────────
        if now - self.last_scan_ts >= config.SCAN_INTERVAL_SEC:
            logger.info("Scanning top symbols...")
            self.symbols = self.scanner.scan()
            self.last_scan_ts = now
            if not self.symbols:
                logger.warning("No symbols found, retrying next cycle")
                return

        # ── 2. Lấy trạng thái tài khoản ─────────────────────────────────────
        try:
            equity = self.client.get_wallet_balance()
            open_positions = self.client.get_positions()
        except Exception as e:
            logger.error(f"Failed to get account state: {e}")
            return

        logger.info(
            f"[TICK] {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')} | "
            f"Equity={equity:.2f} USDT | "
            f"Open positions={len(open_positions)}/{config.MAX_OPEN_POSITIONS}"
        )

        # ── 3. Quản lý vị thế đang mở (trailing stop, TP2) ──────────────────
        if open_positions:
            self.executor.manage_open_positions(open_positions)

        # ── 4. Quét từng symbol để tìm tín hiệu mới ─────────────────────────
        for symbol in self.symbols:
            try:
                self._process_symbol(symbol, equity, open_positions)
            except Exception as e:
                logger.debug(f"Error processing {symbol}: {e}")
            time.sleep(0.1)  # Rate limit: ~10 symbols/giây

    def _process_symbol(self, symbol: str, equity: float, open_positions: list[dict]):
        """Phân tích 1 symbol và ra quyết định giao dịch."""
        # Lấy nến từ Bybit API (không lưu local)
        df_signal = self.client.get_klines(symbol, config.TIMEFRAMES["signal"], config.CANDLE_LIMIT)
        df_trend  = self.client.get_klines(symbol, config.TIMEFRAMES["trend"],  80)
        df_macro  = self.client.get_klines(symbol, config.TIMEFRAMES["macro"],  50)

        if df_signal.empty or len(df_signal) < 50:
            return

        # Chọn strategy tốt nhất cho symbol này
        strategy, _ = self.selector.select(symbol, df_signal, df_trend, df_macro)
        if strategy is None:
            return

        # Sinh signal
        signal = strategy.generate_signal(df_signal, df_trend, df_macro)
        if signal.direction == 0:
            return

        logger.info(
            f"{symbol} [{strategy.name}] → "
            f"{'LONG' if signal.direction==1 else 'SHORT'} "
            f"strength={signal.strength:.2f} | {signal.reason}"
        )

        # Thực thi lệnh
        self.executor.execute_signal(symbol, signal, equity, open_positions)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    bot = TradingBot()
    bot.run()
