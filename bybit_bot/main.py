"""
Bybit Futures Auto Trading Bot — Main Entry Point
Chay 24/7, quet top symbols theo thu tu Bybit, thu tat ca strategies, trade ngay khi co signal.

Usage:
    export BYBIT_API_KEY="your_key"
    export BYBIT_API_SECRET="your_secret"
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
from strategies import ALL_STRATEGIES

setup_logging()
logger = logging.getLogger(__name__)


class TradingBot:
    def __init__(self):
        logger.info("="*60)
        logger.info("Bybit Auto Trading Bot starting...")
        logger.info(f"Mode: {'TESTNET' if config.TESTNET else 'MAINNET (LIVE)'}")
        logger.info(f"Top N symbols: {config.TOP_N_SYMBOLS}")
        logger.info(f"Max positions: {config.MAX_OPEN_POSITIONS}")
        logger.info(f"Strategies: {[s.name for s in ALL_STRATEGIES]}")
        logger.info("="*60)

        self.client     = BybitClient()
        self.scanner    = MarketScanner(self.client)
        self.risk_mgr   = RiskManager(self.client)
        self.bot_logger = BotLogger()
        self.executor   = Executor(self.client, self.risk_mgr, self.bot_logger)

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
                logger.error(f"Unhandled error: {e}\n{traceback.format_exc()}")

            time.sleep(config.LOOP_INTERVAL_SEC)

    def _tick(self):
        now = time.time()

        # Refresh danh sach symbols moi gio — giu nguyen thu tu Bybit (volume cao nhat truoc)
        if now - self.last_scan_ts >= config.SCAN_INTERVAL_SEC:
            logger.info("Scanning top symbols...")
            symbols = self.scanner.scan()
            if symbols:
                self.symbols = symbols
                self.last_scan_ts = now
                logger.info(f"Symbols updated: {len(self.symbols)}, top 5: {self.symbols[:5]}")
            elif not self.symbols:
                logger.warning("No symbols found, retrying next cycle")
                return

        # Lay trang thai tai khoan
        try:
            equity         = self.client.get_wallet_balance()
            open_positions = self.client.get_positions()
        except Exception as e:
            logger.error(f"Failed to get account state: {str(e).encode('ascii','replace').decode()}")
            return

        logger.info(
            f"[TICK] {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')} | "
            f"Equity={equity:.2f} USDT | "
            f"Open={len(open_positions)}/{config.MAX_OPEN_POSITIONS}"
        )

        # Quan ly vi the dang mo
        if open_positions:
            self.executor.manage_open_positions(open_positions)

        # Quet tung symbol theo thu tu Bybit — trade ngay khi co signal
        pos_symbols = {p["symbol"] for p in open_positions}

        for symbol in self.symbols:

            if symbol in pos_symbols:
                continue

            try:
                traded = self._process_symbol(symbol, equity, open_positions)
                if traded:
                    # Cap nhat lai sau khi trade
                    try:
                        open_positions = self.client.get_positions()
                        equity         = self.client.get_wallet_balance()
                        pos_symbols    = {p["symbol"] for p in open_positions}
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"Error {symbol}: {str(e).encode('ascii','replace').decode()}")

            time.sleep(0.05)

    def _trend_direction(self, df) -> int:
        """1h trend: +1 up, -1 down, 0 sideways."""
        from strategies.base import compute_ema
        if len(df) < 50:
            return 0
        close = df["close"]
        ema20 = compute_ema(close, 20).iloc[-1]
        ema50 = compute_ema(close, 50).iloc[-1]
        price = close.iloc[-1]
        if price > ema20 > ema50:
            return 1
        if price < ema20 < ema50:
            return -1
        return 0

    def _process_symbol(self, symbol: str, equity: float, open_positions: list[dict]) -> bool:
        """Can >= 2 strategies dong thuan, scale qty theo do manh. Tra True neu da trade."""
        df_scalp  = self.client.get_klines(symbol, config.TIMEFRAMES["scalp"],  config.CANDLE_LIMIT_SCALP)
        df_signal = self.client.get_klines(symbol, config.TIMEFRAMES["signal"], config.CANDLE_LIMIT_SIGNAL)
        df_trend  = self.client.get_klines(symbol, config.TIMEFRAMES["trend"],  config.CANDLE_LIMIT_TREND)
        df_macro  = self.client.get_klines(symbol, config.TIMEFRAMES["macro"],  config.CANDLE_LIMIT_MACRO)

        if df_signal.empty or len(df_signal) < 50:
            return False

        # ATR filter: bo qua symbol bien dong qua nho
        from strategies.base import compute_atr, compute_rsi
        atr   = compute_atr(df_signal).iloc[-1]
        price = df_signal["close"].iloc[-1]
        if price > 0 and atr / price < config.MIN_ATR_PCT:
            return False

        # RSI va VWAP lam nen tang phan tich
        rsi_now = compute_rsi(df_signal["close"]).iloc[-1]
        from strategies.vwap_volume import compute_vwap
        vwap_now  = compute_vwap(df_signal).iloc[-1]
        vwap_dist = (price - vwap_now) / vwap_now

        # Momentum confirmation: 2 nen lien tiep gan nhat phai cung chieu voi signal
        # Tranh bi danh lua boi 1 nen spike lon roi dao chieu ngay
        opens  = df_signal["open"]
        closes = df_signal["close"]
        c1_bull = closes.iloc[-1] > opens.iloc[-1]  # nen cuoi xanh
        c2_bull = closes.iloc[-2] > opens.iloc[-2]  # nen truoc xanh
        c1_bear = closes.iloc[-1] < opens.iloc[-1]  # nen cuoi do
        c2_bear = closes.iloc[-2] < opens.iloc[-2]  # nen truoc do
        short_term_up   = c1_bull and c2_bull  # 2 nen xanh lien tiep
        short_term_down = c1_bear and c2_bear  # 2 nen do lien tiep

        # Xac dinh mode: REVERSAL (tai dinh/day) hay MOMENTUM (giua xu huong)
        is_reversal = rsi_now < 30 or rsi_now > 70
        reversal_dir = 1 if rsi_now < 30 else (-1 if rsi_now > 70 else 0)

        # 1h macro trend
        macro_trend = self._trend_direction(df_trend)

        long_signals  = []
        short_signals = []

        for strategy in ALL_STRATEGIES:
            try:
                sig = strategy.generate_signal(df_signal, df_trend, df_macro)
                if sig.direction == 0 and len(df_scalp) >= 50:
                    sig = strategy.generate_signal(df_scalp, df_signal, df_trend)

                if sig.direction == 0 or sig.strength < config.MIN_SIGNAL_STRENGTH:
                    continue

                # Nen spike: neu nen truoc (nen -2) lon hon 2x ATR thi la spike, bo qua
                prev_candle_size = abs(closes.iloc[-2] - opens.iloc[-2])
                is_spike = prev_candle_size > atr * 2.0
                if is_spike and not is_reversal:
                    continue

                # Long chi khi 2 nen lien tiep xanh (momentum xac nhan)
                if sig.direction == 1 and not short_term_up and not is_reversal:
                    continue
                # Short chi khi 2 nen lien tiep do (momentum xac nhan)
                if sig.direction == -1 and not short_term_down and not is_reversal:
                    continue

                if sig.direction == 1 and macro_trend >= 0:
                    long_signals.append(sig)
                elif sig.direction == -1 and macro_trend <= 0:
                    short_signals.append(sig)
            except Exception:
                continue

        # REVERSAL trade: RSI cuc doan + 2 nen xac nhan dao chieu thuc su
        # RSI>70 trong uptrend manh khong phai reversal — phai co 2 nen nguoc chieu
        if is_reversal and reversal_dir != 0:
            reversal_confirmed = (
                (reversal_dir == 1  and short_term_up)   or   # RSI<30: phai co 2 nen xanh (boc day)
                (reversal_dir == -1 and short_term_down)       # RSI>70: phai co 2 nen do  (quay dau giam)
            )
            reversal_signals = long_signals if reversal_dir == 1 else short_signals
            if len(reversal_signals) >= config.MIN_CONSENSUS and reversal_confirmed:
                signals = reversal_signals
                best = max(signals, key=lambda s: s.strength)
                best.strength = min(0.95, best.strength + 0.15)
                best.consensus = len(signals)
                best.symbol    = symbol
                names = "+".join(s.strategy_name for s in signals)
                rsi_label = f"RSI={rsi_now:.0f}({'OVERSOLD' if reversal_dir==1 else 'OVERBOUGHT'})"
                logger.info(
                    f"{symbol} [REVERSAL {rsi_label}] [{names}] -> "
                    f"{'LONG' if best.direction==1 else 'SHORT'} "
                    f"strength={best.strength:.2f} | {best.reason}"
                )
                self.executor.execute_signal(symbol, best, equity, open_positions)
                return True

        # MOMENTUM trade: can >= MIN_CONSENSUS strategies dong thuan
        if len(long_signals) >= config.MIN_CONSENSUS:
            signals = long_signals
        elif len(short_signals) >= config.MIN_CONSENSUS:
            signals = short_signals
        else:
            return False

        # Signal manh nhat lam base, set consensus de risk_manager scale qty
        best = max(signals, key=lambda s: s.strength)
        best.consensus = len(signals)
        best.symbol    = symbol
        names = "+".join(s.strategy_name for s in signals)

        logger.info(
            f"{symbol} [{names}] consensus={len(signals)} -> "
            f"{'LONG' if best.direction==1 else 'SHORT'} "
            f"strength={best.strength:.2f} | {best.reason}"
        )

        self.executor.execute_signal(symbol, best, equity, open_positions)
        return True


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    bot = TradingBot()
    bot.run()
