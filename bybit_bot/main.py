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

    def _micro_trend(self, df) -> int:
        """
        Phan tich micro-trend tren 1m voi nhieu nen nhat co the.
        Xet: EMA alignment, body-weighted momentum, volume momentum.
        Tra +1 (up), -1 (down), 0 (khong ro rang / sideways).
        Can it nhat 50 nen de phan tich.
        """
        from strategies.base import compute_ema
        if df is None or df.empty or len(df) < 50:
            return 0

        close  = df["close"]
        open_  = df["open"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]
        n      = len(df)

        # 1. EMA stack: EMA9 > EMA21 > EMA50 = uptrend, nguoc lai = downtrend
        ema9  = compute_ema(close, 9)
        ema21 = compute_ema(close, 21)
        ema50 = compute_ema(close, min(50, n - 1))
        price = close.iloc[-1]
        e9    = ema9.iloc[-1]
        e21   = ema21.iloc[-1]
        e50   = ema50.iloc[-1]

        ema_bull = price > e9 > e21 > e50
        ema_bear = price < e9 < e21 < e50
        ema_score = 1 if ema_bull else (-1 if ema_bear else 0)

        # 2. Body-weighted momentum: xet 20 nen gan nhat
        #    Moi nen dong gop theo body size x direction (lon hon = quan trong hon)
        recent = min(20, n)
        bodies = (close.iloc[-recent:] - open_.iloc[-recent:])
        body_momentum = bodies.sum()  # duong = bullish, am = bearish
        # Chuan hoa theo ATR
        atr_approx = (high.iloc[-recent:] - low.iloc[-recent:]).mean()
        body_score = 0
        if atr_approx > 0:
            norm = body_momentum / (atr_approx * recent)
            if norm > 0.15:
                body_score = 1
            elif norm < -0.15:
                body_score = -1

        # 3. Volume momentum: so sanh volume nen xanh vs nen do trong 30 nen gan nhat
        recent_v = min(30, n)
        bull_vol = volume.iloc[-recent_v:][close.iloc[-recent_v:] > open_.iloc[-recent_v:]].sum()
        bear_vol = volume.iloc[-recent_v:][close.iloc[-recent_v:] < open_.iloc[-recent_v:]].sum()
        total_vol = bull_vol + bear_vol
        vol_score = 0
        if total_vol > 0:
            bull_ratio = bull_vol / total_vol
            if bull_ratio > 0.60:
                vol_score = 1
            elif bull_ratio < 0.40:
                vol_score = -1

        # 4. Slope EMA9: huong chuyen dong EMA9 trong 5 nen gan nhat
        slope_score = 0
        if len(ema9) >= 6:
            slope = (ema9.iloc[-1] - ema9.iloc[-6]) / (ema9.iloc[-6] + 1e-9)
            if slope > 0.001:
                slope_score = 1
            elif slope < -0.001:
                slope_score = -1

        # 5. Higher highs / Lower lows: 10 nen gan nhat
        hh_ll_score = 0
        if n >= 10:
            highs10 = high.iloc[-10:]
            lows10  = low.iloc[-10:]
            # Higher highs va higher lows = uptrend
            if highs10.iloc[-1] > highs10.iloc[-5] and lows10.iloc[-1] > lows10.iloc[-5]:
                hh_ll_score = 1
            # Lower highs va lower lows = downtrend
            elif highs10.iloc[-1] < highs10.iloc[-5] and lows10.iloc[-1] < lows10.iloc[-5]:
                hh_ll_score = -1

        # Tong hop: can >= 3/5 yeu to dong thuan
        total = ema_score + body_score + vol_score + slope_score + hh_ll_score
        if total >= 3:
            return 1
        if total <= -3:
            return -1
        return 0

    def _process_symbol(self, symbol: str, equity: float, open_positions: list[dict]) -> bool:
        """Can >= 2 strategies dong thuan, scale qty theo do manh. Tra True neu da trade."""
        df_micro  = self.client.get_klines(symbol, config.TIMEFRAMES["micro"],  config.CANDLE_LIMIT_MICRO)
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

        # RSI cho reversal detection
        rsi_now = compute_rsi(df_signal["close"]).iloc[-1]

        # Momentum confirmation (15m): 2 nen lien tiep gan nhat phai cung chieu voi signal
        opens  = df_signal["open"]
        closes = df_signal["close"]
        c1_bull = closes.iloc[-1] > opens.iloc[-1]
        c2_bull = closes.iloc[-2] > opens.iloc[-2]
        c1_bear = closes.iloc[-1] < opens.iloc[-1]
        c2_bear = closes.iloc[-2] < opens.iloc[-2]
        short_term_up   = c1_bull and c2_bull
        short_term_down = c1_bear and c2_bear

        # Spike: nen vua dong [-1] lon hon 2x ATR
        last_candle_size = abs(closes.iloc[-1] - opens.iloc[-1])
        is_spike = last_candle_size > atr * 2.0

        # 1m micro-trend: phan tich toan dien 1000 nen 1m (EMA, body momentum, volume, slope, HH/LL)
        micro = self._micro_trend(df_micro)
        micro_up   = (micro == 1)
        micro_down = (micro == -1)

        # 5m hard trend filter (tinh truoc, ap dung sau khi collect signals)
        scalp_trend = self._micro_trend(df_scalp)

        # Xac dinh mode: REVERSAL hay MOMENTUM
        is_reversal  = rsi_now < 30 or rsi_now > 70
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

                # Bo qua sau spike lon (ca reversal cung phai cho spike qua di)
                if is_spike:
                    continue

                # Long chi khi 2 nen xanh lien tiep (momentum xac nhan)
                if sig.direction == 1 and not short_term_up and not is_reversal:
                    continue
                # Short chi khi 2 nen do lien tiep (momentum xac nhan)
                if sig.direction == -1 and not short_term_down and not is_reversal:
                    continue

                # Reversal bypass macro filter — bat day/dinh du macro nguoc
                if is_reversal:
                    if sig.direction == 1:
                        long_signals.append(sig)
                    elif sig.direction == -1:
                        short_signals.append(sig)
                else:
                    if sig.direction == 1 and macro_trend >= 0:
                        long_signals.append(sig)
                    elif sig.direction == -1 and macro_trend <= 0:
                        short_signals.append(sig)
            except Exception:
                continue

        # 5m hard trend filter: ap dung sau khi collect du signals
        # scalp_trend=1 (5m uptrend) -> chi Long, cam Short
        # scalp_trend=-1 (5m downtrend) -> chi Short, cam Long
        # scalp_trend=0 (sideways) -> cho phep ca hai chieu
        if scalp_trend == 1:
            short_signals = []
            logger.debug(f"{symbol}: 5m uptrend — short signals blocked")
        elif scalp_trend == -1:
            long_signals = []
            logger.debug(f"{symbol}: 5m downtrend — long signals blocked")

        # REVERSAL trade: RSI cuc doan + 2 nen 15m + 3 nen 1m xac nhan dao chieu + >= MIN_CONSENSUS
        if is_reversal and reversal_dir != 0:
            reversal_confirmed = (
                (reversal_dir == 1  and short_term_up   and micro_up)   or
                (reversal_dir == -1 and short_term_down and micro_down)
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

        best = max(signals, key=lambda s: s.strength)

        # 1m micro-trend phai cung chieu voi signal — tranh entry khi gia dang di nguoc
        if best.direction == 1 and not micro_up:
            logger.debug(f"{symbol}: LONG signal but 1m micro trend not up — skip")
            return False
        if best.direction == -1 and not micro_down:
            logger.debug(f"{symbol}: SHORT signal but 1m micro trend not down — skip")
            return False

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
