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
from strategies import ALL_STRATEGIES, BREAKOUT_STRATEGY

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
        self.last_full_scan_ts: float = 0  # lan cuoi check 180 con lai
        self.volume_map: dict[str, float] = {}  # symbol -> 24h volume USDT

        # Post-loss tracking: symbol -> timestamp dong lenh lo
        # Trong 5 phut sau lo, can consensus >= MIN_CONSENSUS+1 de vao lai
        self._recent_loss_ts: dict[str, float] = {}
        self.executor.on_loss_callback = self._on_symbol_loss
        # Track positions de detect SL/TP hit boi exchange (khong qua executor)
        self._prev_pos_symbols: set[str] = set()

    def _on_symbol_loss(self, symbol: str):
        self._recent_loss_ts[symbol] = time.time()
        logger.info(f"{symbol}: post-loss cooldown started (5 min higher consensus)")

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
                self.volume_map = getattr(self.scanner, "volume_map", {})
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
            # Refresh lai sau khi manage — co the co lenh vua dong (SL/TP hit)
            # De bot co the re-enter ngay trong cung tick nay
            try:
                open_positions = self.client.get_positions()
                equity         = self.client.get_wallet_balance()
            except Exception:
                pass

        pos_symbols = {p["symbol"] for p in open_positions}

        # Detect position dong boi exchange (SL/TP hit) — khong qua executor._close_position
        # Neu symbol vua co position ma gio mat → check closed PnL → neu lo thi fire post-loss
        closed_by_exchange = self._prev_pos_symbols - pos_symbols
        if closed_by_exchange:
            try:
                closed_pnl = self.client.get_closed_pnl(list(closed_by_exchange))
                for symbol, pnl in closed_pnl.items():
                    if pnl < 0:
                        self._on_symbol_loss(symbol)
                        logger.info(f"{symbol}: SL/TP hit by exchange, pnl={pnl:.4f} → post-loss filter")
            except Exception as e:
                logger.debug(f"get_closed_pnl error: {e}")
        self._prev_pos_symbols = pos_symbols

        # Top 20: check moi tick (moi 15 giay) — bat breakout nhanh
        top20   = self.symbols[:config.TOP_FOCUS_COUNT]
        # Con lai: chi check moi FULL_SCAN_INTERVAL giay
        do_full = (now - self.last_full_scan_ts) >= config.FULL_SCAN_INTERVAL
        rest    = self.symbols[config.TOP_FOCUS_COUNT:] if do_full else []
        if do_full:
            self.last_full_scan_ts = now

        scan_list = top20 + rest
        if rest:
            logger.info(f"[TICK] Full scan: top20 + {len(rest)} remaining symbols")
        else:
            logger.info(f"[TICK] Fast scan: top20 only")

        for symbol in scan_list:

            if symbol in pos_symbols:
                continue

            # Volume filter cho non-top20: bo qua coin nho thanh khoan thap
            is_top20 = symbol in top20
            if not is_top20:
                vol = self.volume_map.get(symbol, 0)
                if vol < config.MIN_VOLUME_NON_TOP20:
                    logger.debug(
                        f"{symbol}: skip non-top20 (vol={vol/1e6:.1f}M < {config.MIN_VOLUME_NON_TOP20/1e6:.0f}M)"
                    )
                    continue

            try:
                traded = self._process_symbol(symbol, equity, open_positions, is_top20)
                if traded:
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

    def _micro_entry_analysis(self, df_micro, direction: int, is_top20: bool, is_reversal: bool = False) -> bool:
        """
        Phan tich toan bo 1m candles de xac dinh timing entry.
        5 yeu to: EMA alignment, momentum 3 nen, volume, exhaustion, micro structure.
        Top20 (300 nen): can score >= 3/5. Non-top20 (30 nen): can score >= 2/4.
        Tra True = timing tot, False = nen cho.
        """
        from strategies.base import compute_ema, compute_atr
        if df_micro is None or df_micro.empty:
            return False  # khong co data → khong trade
        n = len(df_micro)
        if n < 5:
            return False  # qua it data → khong trade

        close  = df_micro["close"]
        open_  = df_micro["open"]
        high   = df_micro["high"]
        low    = df_micro["low"]
        volume = df_micro["volume"]

        atr_1m = compute_atr(df_micro).iloc[-1]
        if atr_1m == 0:
            return False  # gia bat dong → khong trade

        price = close.iloc[-1]
        score = 0

        # Factor 1: EMA alignment — price / EMA9 / EMA21 phai xep hang dung chieu
        if n >= 21:
            ema9  = compute_ema(close, 9)
            ema21 = compute_ema(close, 21)
            e9, e21 = ema9.iloc[-1], ema21.iloc[-1]
            if direction == 1 and price > e9 > e21:
                score += 1
            elif direction == -1 and price < e9 < e21:
                score += 1
            else:
                score -= 1

        # Factor 2: Momentum — 2/3 nen gan nhat phai cung chieu
        bodies_3 = close.iloc[-3:].values - open_.iloc[-3:].values
        bull3 = sum(1 for b in bodies_3 if b > 0)
        bear3 = sum(1 for b in bodies_3 if b < 0)
        if direction == 1 and bull3 >= 2:
            score += 1
        elif direction == -1 and bear3 >= 2:
            score += 1
        else:
            score -= 1

        # Factor 3: Volume binh thuong — khong phai spike va khong qua nho
        if n >= 10:
            vol_ma = volume.rolling(10).mean().iloc[-1]
            if vol_ma > 0:
                ratio = volume.iloc[-1] / vol_ma
                if 0.5 <= ratio <= 4.0:   # volume hop le
                    score += 1
                elif ratio > 6.0:          # spike volume — co the dang o dinh/day
                    score -= 1

        # Factor 4: Khong exhausted — nen hien tai khong qua nho sau loat nen lon (pause signal)
        if n >= 4:
            prev_body_avg = abs(close.iloc[-4:-1].values - open_.iloc[-4:-1].values).mean()
            curr_body     = abs(close.iloc[-1] - open_.iloc[-1])
            if prev_body_avg > 0:
                ratio_body = curr_body / prev_body_avg
                if ratio_body > 0.3:    # nen hien tai co noi luc
                    score += 1
                elif ratio_body < 0.15: # doji / spinning top — exhaustion signal
                    score -= 1

        # Factor 5: Micro structure — HH+HL (long) hoac LH+LL (short) trong 5 nen gan nhat
        if n >= 8:
            h5 = high.iloc[-5:].values
            l5 = low.iloc[-5:].values
            if direction == 1:
                if h5[-1] > h5[-3] and l5[-1] > l5[-3]:
                    score += 1
                elif h5[-1] < h5[-3] and l5[-1] < l5[-3]:
                    score -= 1
            else:
                if h5[-1] < h5[-3] and l5[-1] < l5[-3]:
                    score += 1
                elif h5[-1] > h5[-3] and l5[-1] > l5[-3]:
                    score -= 1

        # Factor 6: Dual range check — 100 nen (xu huong trung han) + 20 nen (local bounce/dip)
        # HARD BLOCK 100-candle: tranh long o top 25% / short o bottom 25% cua 100 phut qua
        # HARD BLOCK 20-candle:  tranh long o top 20% / short o bottom 20% cua 20 phut qua
        #   (bat duoc "short o day local" khi 100-candle range cho thay midrange nhung thuc te dang bounce)
        # Ngoai le: is_reversal=True (RSI cuc doan xac nhan) → skip range block
        _range_window = min(100, n)
        if _range_window >= 20:
            high_rng = high.iloc[-_range_window:].max()
            low_rng  = low.iloc[-_range_window:].min()
            rng = high_rng - low_rng
            if rng > 0:
                range_pos = (price - low_rng) / rng
                if not is_reversal:
                    if direction == 1 and range_pos > 0.90:
                        logger.debug(f"micro_entry: HARD BLOCK long — 100c range_pos={range_pos:.2f} > 0.90")
                        return False
                    if direction == -1 and range_pos < 0.10:
                        logger.debug(f"micro_entry: HARD BLOCK short — 100c range_pos={range_pos:.2f} < 0.10")
                        return False
                # Bonus cho entry o vung an toan
                if direction == 1 and range_pos < 0.55:
                    score += 1
                elif direction == -1 and range_pos > 0.45:
                    score += 1

        # 20-candle local range: check them de tranh short o day local / long o dinh local
        # Bat cac truong hop 100-candle cho thay midrange nhung local dang o extreme
        _local_window = min(20, n)
        if _local_window >= 10 and not is_reversal:
            local_high = high.iloc[-_local_window:].max()
            local_low  = low.iloc[-_local_window:].min()
            local_rng  = local_high - local_low
            if local_rng > 0:
                local_pos = (price - local_low) / local_rng
                if direction == 1 and local_pos > 0.90:
                    logger.debug(f"micro_entry: HARD BLOCK long — 20c local_pos={local_pos:.2f} > 0.90 (local top)")
                    return False
                if direction == -1 and local_pos < 0.10:
                    logger.debug(f"micro_entry: HARD BLOCK short — 20c local_pos={local_pos:.2f} < 0.10 (local bottom)")
                    return False

        # Factor 7: Momentum deceleration — nen gan day nho manh so voi nen truoc
        # Tranh vao lenh khi momentum dang kiet suc (sap dao chieu)
        if n >= 8:
            recent_body = abs(close.iloc[-3:-1].values - open_.iloc[-3:-1].values).mean()
            prev_body   = abs(close.iloc[-8:-3].values - open_.iloc[-8:-3].values).mean()
            if prev_body > 0:
                decel = recent_body / prev_body
                if decel < 0.35:    # Momentum giam > 65% — dang dung lai / dao chieu
                    score -= 1
                elif decel > 0.60:  # Momentum on dinh
                    score += 1

        threshold = 3  # tat ca coin: can >= 3/7 factors (non-top20 riskier, khong giam nhe hon)
        ok = score >= threshold
        if not ok:
            logger.debug(
                f"micro_entry_analysis: dir={direction} score={score} threshold={threshold} "
                f"n={n} top20={is_top20} -> skip"
            )
        return ok

    def _process_symbol(self, symbol: str, equity: float, open_positions: list[dict], is_top20: bool = False) -> bool:
        """Can >= 2 strategies dong thuan, scale qty theo do manh. Tra True neu da trade."""
        micro_limit = config.CANDLE_LIMIT_MICRO if is_top20 else config.CANDLE_LIMIT_MICRO_SMALL
        df_micro  = self.client.get_klines(symbol, config.TIMEFRAMES["micro"], micro_limit)
        df_scalp  = self.client.get_klines(symbol, config.TIMEFRAMES["scalp"],  config.CANDLE_LIMIT_SCALP)
        df_signal = self.client.get_klines(symbol, config.TIMEFRAMES["signal"], config.CANDLE_LIMIT_SIGNAL)
        df_trend  = self.client.get_klines(symbol, config.TIMEFRAMES["trend"],  config.CANDLE_LIMIT_TREND)
        df_macro  = self.client.get_klines(symbol, config.TIMEFRAMES["macro"],  config.CANDLE_LIMIT_MACRO)

        if df_signal.empty or len(df_signal) < 50:
            return False

        # ATR filter: bo qua symbol bien dong qua nho
        from strategies.base import compute_atr, compute_rsi, compute_adx
        atr   = compute_atr(df_signal).iloc[-1]
        price = df_signal["close"].iloc[-1]
        if price > 0 and atr / price < config.MIN_ATR_PCT:
            return False

        # ADX filter: bo qua khi thi truong sideway (ADX < MIN_ADX)
        import math as _math
        adx = compute_adx(df_signal).iloc[-1]
        if _math.isnan(adx) or adx < config.MIN_ADX:
            logger.debug(f"{symbol}: skip — ADX={adx:.1f} < {config.MIN_ADX} (sideway)")
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

        # Spike filter: nen hien tai HOAC bat ky nen nao trong 5 nen gan nhat > 2x ATR
        # Neu co spike dump -> khong short them; spike pump -> khong long them
        spike_lookback = 5
        recent_bodies  = closes.iloc[-spike_lookback:].values - opens.iloc[-spike_lookback:].values
        last_candle_size = abs(recent_bodies[-1])
        is_spike = last_candle_size > atr * 2.0

        # Post-spike direction block (tat ca coin)
        spike_was_dump = any(b < -atr * 2.0 for b in recent_bodies)
        spike_was_pump = any(b >  atr * 2.0 for b in recent_bodies)

        # 1m micro-trend — chay cho tat ca coin (non-top20 gio du 120 nen)
        micro = self._micro_trend(df_micro)
        micro_up   = (micro == 1)
        micro_down = (micro == -1)

        scalp_trend = self._micro_trend(df_scalp) if is_top20 else 0  # 5m trend cho post-spike check

        # [CHONG LO] Spike check tren 1m — direction-aware
        # Pump spike -> block LONG (khong mua dinh), nhung cho phep SHORT (ban dinh la hop le)
        # Dump spike -> block SHORT (khong ban day), nhung cho phep LONG (mua day la hop le)
        # Ca 2 cung xuat hien -> thi truong loan, skip tat ca
        _micro_spike_dump = False
        _micro_spike_pump = False
        if not df_micro.empty and len(df_micro) >= 5:
            _micro_atr    = compute_atr(df_micro).iloc[-1]
            _micro_bodies = (df_micro["close"].iloc[-5:].values - df_micro["open"].iloc[-5:].values)
            _micro_spike_dump = any(b < -_micro_atr * 2.0 for b in _micro_bodies)
            _micro_spike_pump = any(b >  _micro_atr * 2.0 for b in _micro_bodies)
            if _micro_spike_dump and _micro_spike_pump:
                logger.debug(f"{symbol}: skip — 1m spike ca 2 chieu (thi truong loan)")
                return False

        # BREAKOUT: chi top20
        if is_top20 and df_micro is not None and not df_micro.empty and len(df_micro) >= 30:
            bo_sig = BREAKOUT_STRATEGY.generate_signal(df_micro, df_scalp, df_signal)
            if bo_sig.direction != 0:
                # 5m khong duoc nguoc chieu — cho phep sideways
                bo_ok = (
                    (bo_sig.direction == 1  and scalp_trend >= 0) or
                    (bo_sig.direction == -1 and scalp_trend <= 0)
                )
                # 1m micro-trend cung phai xac nhan
                micro_ok = (bo_sig.direction == 1 and micro_up) or (bo_sig.direction == -1 and micro_down)
                post_spike_ok = not (bo_sig.direction == -1 and spike_was_dump and scalp_trend != -1) and \
                                not (bo_sig.direction == 1  and spike_was_pump and scalp_trend != 1)
                if bo_ok and micro_ok and not is_spike and post_spike_ok:
                    # BREAKOUT phai qua range check — tranh long o dinh / short o day
                    if not self._micro_entry_analysis(df_micro, bo_sig.direction, is_top20):
                        logger.debug(f"{symbol}: BREAKOUT skip — range/micro_entry block")
                    else:
                        bo_sig.symbol    = symbol
                        bo_sig.consensus = 1
                        logger.info(
                            f"{symbol} [BREAKOUT TOP20] -> "
                            f"{'LONG' if bo_sig.direction==1 else 'SHORT'} "
                            f"strength={bo_sig.strength:.2f} | {bo_sig.reason}"
                        )
                        self.executor.execute_signal(symbol, bo_sig, equity, open_positions)
                        return True

        # Xac dinh mode: REVERSAL hay MOMENTUM
        # Nguong 35/65 dong bo voi sustained_trend va bollinger — bat duoc reversal som hon
        is_reversal  = rsi_now < 35 or rsi_now > 65
        reversal_dir = 1 if rsi_now < 35 else (-1 if rsi_now > 65 else 0)

        # 1h macro trend
        macro_trend = self._trend_direction(df_trend)

        long_signals  = []
        short_signals = []

        for strategy in ALL_STRATEGIES:
            try:
                # SustainedTrendStrategy chi chay cho top20
                if strategy.name == "sustained_trend" and not is_top20:
                    continue

                sig = strategy.generate_signal(df_signal, df_trend, df_macro)
                # Scalp fallback: thu 5m neu 15m khong co signal
                # Skip VWAP (window 96x15m=24h, tren 5m cho ra 8h — sai)
                # Pass df_macro de giu 4h context khong bi mat
                if sig.direction == 0 and len(df_scalp) >= 50 and strategy.name != "vwap_volume":
                    sig = strategy.generate_signal(df_scalp, df_signal, df_macro)

                if sig.direction == 0 or sig.strength < config.MIN_SIGNAL_STRENGTH:
                    continue

                # Bo qua neu nen hien tai la spike
                if is_spike:
                    continue

                # Post-spike block: chi ap dung khi 5m CHUA xac nhan trend cung chieu
                # Neu 5m da xac nhan downtrend (scalp_trend==-1) thi dump la phan cua trend -> cho phep short
                # Neu 5m da xac nhan uptrend  (scalp_trend== 1) thi pump la phan cua trend -> cho phep long
                if sig.direction == -1 and spike_was_dump and scalp_trend != -1:
                    continue
                if sig.direction == 1 and spike_was_pump and scalp_trend != 1:
                    continue

                # Long chi khi 2 nen xanh lien tiep (momentum xac nhan)
                # TOP_PRIORITY (BTC/ETH/SOL/BNB/XRP): bo qua yeu cau nay, dung 1m micro trend thay the
                is_priority = symbol in config.TOP_PRIORITY
                if sig.direction == 1 and not short_term_up and not is_reversal and not is_priority:
                    continue
                # Short chi khi 2 nen do lien tiep (momentum xac nhan)
                if sig.direction == -1 and not short_term_down and not is_reversal and not is_priority:
                    continue

                # Reversal bypass macro filter — bat day/dinh du macro nguoc
                if is_reversal:
                    if sig.direction == 1:
                        long_signals.append(sig)
                    elif sig.direction == -1:
                        short_signals.append(sig)
                else:
                    # macro_trend == 1: uptrend → long ok
                    # macro_trend ==-1: downtrend → short ok
                    # macro_trend == 0: sideways → cho phep nhung can consensus cao hon (xu ly sau)
                    if sig.direction == 1 and macro_trend >= 0:
                        long_signals.append(sig)
                    elif sig.direction == -1 and macro_trend <= 0:
                        short_signals.append(sig)
            except Exception:
                continue

        if False:  # 5m hard filter da bo — 83h qua dai, miss nhieu lenh ngan han
            pass
            logger.debug(f"{symbol}: 5m downtrend — long signals blocked")

        # Post-loss check: tinh truoc khi dung cho ca reversal va momentum
        post_loss = (time.time() - self._recent_loss_ts.get(symbol, 0)) < 300
        if post_loss:
            logger.debug(f"{symbol}: post-loss 5min active → consensus+1 required")

        # REVERSAL trade: RSI cuc doan + 2 nen 15m + 1m micro xac nhan dao chieu + >= MIN_CONSENSUS
        if is_reversal and reversal_dir != 0:
            # Direction-aware spike: long sau pump spike va short sau dump spike deu nguy hiem
            reversal_spike_blocked = (
                (reversal_dir == 1  and _micro_spike_pump) or
                (reversal_dir == -1 and _micro_spike_dump)
            )
            if not reversal_spike_blocked:
                reversal_confirmed = (
                    (reversal_dir == 1  and short_term_up)   or
                    (reversal_dir == -1 and short_term_down)
                ) and self._micro_entry_analysis(df_micro, reversal_dir, is_top20, is_reversal=True)
                reversal_signals = long_signals if reversal_dir == 1 else short_signals
                reversal_min = config.MIN_CONSENSUS + (1 if post_loss else 0)
                if len(reversal_signals) >= reversal_min and reversal_confirmed:
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
        # Neu 1h sideways (macro_trend==0): yeu cau them 1 consensus de tranh tin hieu gia
        # Neu vua lo lenh tren symbol nay trong 5 phut truoc: yeu cau consensus+1 (post-loss filter)
        sideways_1h = (macro_trend == 0)
        # Cap o MIN_CONSENSUS+1 de tranh yeu cau 5 consensus (qua hiem, bot ngung trade)
        extra = min(1, (1 if sideways_1h else 0) + (1 if post_loss else 0))
        required_consensus = config.MIN_CONSENSUS + extra

        if len(long_signals) >= required_consensus:
            signals = long_signals
        elif len(short_signals) >= required_consensus:
            signals = short_signals
        else:
            return False

        best = max(signals, key=lambda s: s.strength)

        # Direction-aware 1m spike filter:
        # Pump spike -> block LONG (khong mua dinh), SHORT van duoc phep (ban dinh tot)
        # Dump spike -> block SHORT (khong ban day), LONG van duoc phep (mua day tot)
        if _micro_spike_pump and best.direction == 1:
            logger.debug(f"{symbol}: skip — 1m pump spike, khong long")
            return False
        if _micro_spike_dump and best.direction == -1:
            logger.debug(f"{symbol}: skip — 1m dump spike, khong short")
            return False

        # 1m micro trend confirmation — tat ca coin (micro trend phai cung chieu hoac neutral)
        # Tranh trade khi 1m dang nguoc chieu hoan toan voi signal
        if best.direction == 1 and micro_down:
            logger.debug(f"{symbol}: skip — 1m micro trend BEARISH vs LONG signal")
            return False
        if best.direction == -1 and micro_up:
            logger.debug(f"{symbol}: skip — 1m micro trend BULLISH vs SHORT signal")
            return False

        # 1m micro entry timing: apply cho TAT CA coin voi phan tich day du 5 yeu to
        if not self._micro_entry_analysis(df_micro, best.direction, is_top20):
            logger.debug(f"{symbol}: skip — 1m micro entry timing not confirmed (score too low)")
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
