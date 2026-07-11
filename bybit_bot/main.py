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
import threading
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
        logger.info(f"Mode: {'TESTNET' if config.TESTNET else 'MAINNET (LIVE)'}")
        logger.info(f"Top N symbols: {config.TOP_N_SYMBOLS}")
        logger.info(f"Max positions: {config.MAX_OPEN_POSITIONS}")
        logger.info(f"Risk per trade: {config.SL_MAX_LOSS_PCT*100:.1f}% capital per trade")
        logger.info("="*60)

        self.client    = BybitClient()
        self.scanner   = MarketScanner(self.client)
        self.selector  = StrategySelector()
        self.risk_mgr  = RiskManager(self.client)
        self.bot_logger = BotLogger()
        self.executor  = Executor(self.client, self.risk_mgr, self.bot_logger)

        self.symbols: list[str] = []
        self.last_scan_ts: float = 0
        self._ranking_thread: threading.Thread = None
        self._ranking_lock = threading.Lock()

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

        # ── 1. Cập nhật symbols mỗi 1 giờ, ranking chạy nền ─────────────────
        if now - self.last_scan_ts >= config.SCAN_INTERVAL_SEC:
            logger.info("Scanning top symbols...")
            raw_symbols = self.scanner.scan()
            if not raw_symbols:
                logger.warning("No symbols found, retrying next cycle")
                return
            self.last_scan_ts = now

            # Lần đầu: dùng thứ tự scanner (volume/volatility) để trade ngay
            if not self.symbols:
                self.symbols = raw_symbols
                logger.info(f"Initial symbols loaded: {len(self.symbols)}, top 5: {self.symbols[:5]}")

            # Ranking chạy nền — không block bot trading
            is_running = self._ranking_thread and self._ranking_thread.is_alive()
            if not is_running:
                t = threading.Thread(
                    target=self._rank_by_expectancy_bg,
                    args=(raw_symbols,),
                    daemon=True,
                )
                self._ranking_thread = t
                t.start()
                logger.info("Expectancy ranking started in background...")

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

        # ── 4. Quét từng symbol — thấy signal là trade ngay, không chờ hết vòng
        signals_found = 0
        for symbol in self.symbols:
            # Cập nhật lại open_positions sau mỗi lệnh mới
            try:
                open_positions = self.client.get_positions()
                equity         = self.client.get_wallet_balance()
            except Exception:
                pass

            if len(open_positions) >= config.MAX_OPEN_POSITIONS:
                logger.info(f"[SCAN STOP] Max positions reached, waiting next tick")
                break

            try:
                result = self._process_symbol(symbol, equity, open_positions)
                if result == "signal":
                    signals_found += 1
            except Exception as e:
                logger.debug(f"Error processing {symbol}: {e}")
            time.sleep(0.05)

    def _rank_by_expectancy_bg(self, symbols: list[str]):
        """
        Ranking + trading đồng thời:
        - Vừa backtest từng symbol
        - Nếu có strategy tốt thì kiểm tra signal và trade ngay
        - Sau khi xong toàn bộ thì cập nhật thứ tự symbols cho vòng sau
        """
        scores: list[tuple[float, str]] = []

        for symbol in symbols:
            try:
                df   = self.client.get_klines(symbol, config.TIMEFRAMES["signal"], config.CANDLE_LIMIT_SIGNAL)
                df_t = self.client.get_klines(symbol, config.TIMEFRAMES["trend"],  config.CANDLE_LIMIT_TREND)
                df_m = self.client.get_klines(symbol, config.TIMEFRAMES["macro"],  config.CANDLE_LIMIT_MACRO)

                if df.empty or len(df) < 50:
                    scores.append((0.0, symbol))
                    continue

                strategy, result = self.selector.select(symbol, df, df_t, df_m)
                expectancy = result.expectancy if result else 0.0
                scores.append((expectancy, symbol))

                # Co strategy tot -> kiem tra signal va trade ngay
                if strategy and result and result.expectancy > 0:
                    try:
                        equity         = self.client.get_wallet_balance()
                        open_positions = self.client.get_positions()
                        pos_symbols    = {p["symbol"] for p in open_positions}

                        # Skip nếu đã có vị thế trên symbol này
                        if symbol in pos_symbols:
                            continue

                        if len(open_positions) < config.MAX_OPEN_POSITIONS:
                            df_scalp = self.client.get_klines(symbol, config.TIMEFRAMES["scalp"], config.CANDLE_LIMIT_SCALP)
                            signal   = strategy.generate_signal(df, df_t, df_m)

                            if signal.direction == 0 and len(df_scalp) >= 50:
                                signal = strategy.generate_signal(df_scalp, df, df_t)

                            if signal.direction != 0 and signal.strength >= config.MIN_SIGNAL_STRENGTH:
                                signal.symbol = symbol
                                logger.info(
                                    f"{symbol} [{strategy.name}] -> "
                                    f"{'LONG' if signal.direction==1 else 'SHORT'} "
                                    f"strength={signal.strength:.2f} | {signal.reason}"
                                )
                                self.executor.execute_signal(symbol, signal, equity, open_positions)
                            else:
                                logger.debug(f"[BG] {symbol} [{strategy.name}]: no signal after strategy found")
                    except Exception as e:
                        logger.debug(f"Trade attempt failed {symbol}: {e}")

            except Exception:
                scores.append((0.0, symbol))
            time.sleep(0.05)

        # Cap nhat thu tu symbols theo expectancy cho vong scan tiep theo
        scores.sort(key=lambda x: x[0], reverse=True)
        ranked = [s for _, s in scores]
        with self._ranking_lock:
            self.symbols = ranked

        top = " | ".join(f"{s}({e:.3f}%)" for e, s in scores[:10] if e > 0)
        logger.info(f"[RANK DONE] Top 10: {top}")

    def _process_symbol(self, symbol: str, equity: float, open_positions: list[dict]) -> str:
        """Phân tích 1 symbol và ra quyết định giao dịch."""
        # Skip ngay nếu đã có vị thế trên symbol này — không tốn API call
        pos_symbols = {p["symbol"] for p in open_positions}
        if symbol in pos_symbols:
            return "has_position"

        # Lay nen tu Bybit API — toi da co the de khong bo lo signal nao
        df_scalp  = self.client.get_klines(symbol, config.TIMEFRAMES["scalp"],  config.CANDLE_LIMIT_SCALP)
        df_signal = self.client.get_klines(symbol, config.TIMEFRAMES["signal"], config.CANDLE_LIMIT_SIGNAL)
        df_trend  = self.client.get_klines(symbol, config.TIMEFRAMES["trend"],  config.CANDLE_LIMIT_TREND)
        df_macro  = self.client.get_klines(symbol, config.TIMEFRAMES["macro"],  config.CANDLE_LIMIT_MACRO)

        if df_signal.empty or len(df_signal) < 50:
            return "no_data"

        # Chon strategy tot nhat cho symbol nay
        strategy, result = self.selector.select(symbol, df_signal, df_trend, df_macro)
        if strategy is None:
            return "no_strategy"

        # Thu signal tren 15m truoc, neu khong co thi thu 5m
        signal = strategy.generate_signal(df_signal, df_trend, df_macro)

        if signal.direction == 0 and len(df_scalp) >= 50:
            signal = strategy.generate_signal(df_scalp, df_signal, df_trend)

        if signal.direction == 0:
            logger.debug(f"{symbol} [{strategy.name}]: no signal (direction=0)")
            return "no_signal"

        if signal.strength < config.MIN_SIGNAL_STRENGTH:
            logger.debug(f"{symbol} [{strategy.name}]: signal too weak ({signal.strength:.2f} < {config.MIN_SIGNAL_STRENGTH})")
            return "no_signal"

        signal.symbol = symbol

        logger.info(
            f"{symbol} [{strategy.name}] -> "
            f"{'LONG' if signal.direction==1 else 'SHORT'} "
            f"strength={signal.strength:.2f} | {signal.reason}"
        )

        # Kiểm tra equity đủ không (tối thiểu 5 USDT)
        if equity < 5:
            logger.warning(f"Equity quá thấp ({equity:.2f} USDT) — cần nạp thêm tiền để vào lệnh")
            return "low_equity"

        # Thực thi lệnh
        self.executor.execute_signal(symbol, signal, equity, open_positions)
        return "signal"


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    bot = TradingBot()
    bot.run()
