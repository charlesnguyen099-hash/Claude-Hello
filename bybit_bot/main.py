"""
Bybit Futures Auto Trading Bot — Main Entry Point
Chay 24/7, quet top symbols theo thu tu Bybit, thu tat ca strategies, trade ngay khi co signal.

Usage:
    export BYBIT_API_KEY="your_key"
    export BYBIT_API_SECRET="your_secret"
    python main.py
"""

import logging
import math
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
from strategies.base import compute_ema, compute_atr, compute_rsi, compute_adx

setup_logging()
logger = logging.getLogger(__name__)


class TradingBot:
    def __init__(self):
        logger.info("="*60)
        logger.info("Bybit Auto Trading Bot starting...")
        logger.info(f"Mode: {'TESTNET' if config.TESTNET else 'MAINNET (LIVE)'}")
        logger.info(f"Top N symbols: {config.TOP_N_SYMBOLS}")
        logger.info(f"Max positions: unlimited")
        logger.info(f"Strategies: {[s.name for s in ALL_STRATEGIES]}")
        logger.info("="*60)

        self.client     = BybitClient()
        self.scanner    = MarketScanner(self.client)
        self.risk_mgr   = RiskManager(self.client)
        self.bot_logger = BotLogger()
        self.executor   = Executor(self.client, self.risk_mgr, self.bot_logger)

        self.symbols: list[str] = []
        self.last_scan_ts: float = 0

        # Post-loss tracking: symbol -> timestamp dong lenh lo
        # Trong 5 phut sau lo, can consensus >= MIN_CONSENSUS+1 de vao lai
        self._recent_loss_ts: dict[str, float] = {}
        self.executor.on_loss_callback = self._on_symbol_loss
        # Track positions de detect SL/TP hit boi exchange (khong qua executor)
        self._prev_pos_symbols: set[str] = set()
        # BTC global trend: +1 uptrend, -1 downtrend, 0 sideways (cap nhat moi tick)
        self.btc_trend: int = 0
        self.btc_trend_4h: int = 0

    def _on_symbol_loss(self, symbol: str, side: str = ""):
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

            # Khi co vi the mo: check position management moi 3s trong 15s window
            # De khong miss breakeven/partial-close trong spike ngan (AKE pattern)
            try:
                positions_now = self.client.get_positions()
                if positions_now:
                    for _ in range(4):
                        time.sleep(3)
                        try:
                            positions_now = self.client.get_positions()
                            if positions_now:
                                self.executor.manage_open_positions(positions_now)
                        except Exception:
                            pass
                    # Remaining time in 15s window already elapsed (4×3=12s)
                    time.sleep(3)
                else:
                    time.sleep(config.LOOP_INTERVAL_SEC)
            except Exception:
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
            f"Open={len(open_positions)}"
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

        top10 = set(self.symbols[:10])

        # Cap nhat BTC global trend moi tick (ca 1h va 4h)
        try:
            df_btc_1h = self.client.get_klines("BTCUSDT", config.TIMEFRAMES["trend"], 100)
            df_btc_4h = self.client.get_klines("BTCUSDT", config.TIMEFRAMES["macro"],  100)
            if not df_btc_1h.empty and len(df_btc_1h) >= 50:
                self.btc_trend = self._trend_direction(df_btc_1h)
            if not df_btc_4h.empty and len(df_btc_4h) >= 50:
                self.btc_trend_4h = self._trend_direction(df_btc_4h)
        except Exception:
            pass

        # Priority list: top10 only
        priority_set = top10

        # Scan list: priority first, then trending-only coins (khong lap)
        trending_only = [s for s in getattr(self.scanner, "trending_symbols", []) if s not in priority_set]
        scan_list = list(dict.fromkeys(list(priority_set) + trending_only))

        logger.info(
            f"[TICK] Focus scan: {len(scan_list)} symbols "
            f"(priority={len(priority_set)}, trending_only={len(trending_only)}) | "
            f"BTC_1h={'UP' if self.btc_trend==1 else 'DOWN' if self.btc_trend==-1 else 'SIDE'} "
            f"BTC_4h={'UP' if self.btc_trend_4h==1 else 'DOWN' if self.btc_trend_4h==-1 else 'SIDE'}"
        )

        for symbol in scan_list:

            if symbol in pos_symbols:
                continue

            is_priority = symbol in priority_set
            try:
                traded = self._process_symbol(symbol, equity, open_positions, is_priority)
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
        """+1 up, -1 down, 0 sideways. Pass df_trend for 1h or df_macro for 4h."""
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

    def _micro_entry_analysis(self, df_micro, direction: int, is_reversal: bool = False) -> bool:
        """
        Phan tich toan bo 1m candles de xac dinh timing entry.
        7 yeu to: EMA, momentum, volume, body size, micro structure, range, deceleration.
        Tat ca coin: can score >= 3/7. Tra True = timing tot, False = nen cho.
        """
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
        # HARD BLOCK 20-candle:  tranh long o top 30% / short o bottom 30% cua 20 phut qua
        #   (bat duoc "short o day local" khi 100-candle range cho thay midrange nhung thuc te dang bounce)
        #   SKHYNIX 21:24 = 25.7% → 20c block; SOL/ADA 22:15 = 0-6% → block
        # Ngoai le: is_reversal=True (RSI cuc doan xac nhan) → skip range block
        _range_window = min(100, n)
        if _range_window >= 20:
            high_rng = high.iloc[-_range_window:].max()
            low_rng  = low.iloc[-_range_window:].min()
            rng = high_rng - low_rng
            if rng > 0:
                range_pos = (price - low_rng) / rng
                if not is_reversal:
                    if direction == 1 and range_pos > 0.75:
                        logger.debug(f"micro_entry: HARD BLOCK long — 100c range_pos={range_pos:.2f} > 0.75")
                        return False
                    if direction == -1 and range_pos < 0.25:
                        logger.debug(f"micro_entry: HARD BLOCK short — 100c range_pos={range_pos:.2f} < 0.25")
                        return False
                # Bonus cho entry o vung an toan
                if direction == 1 and range_pos < 0.45:
                    score += 1
                elif direction == -1 and range_pos > 0.55:
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
                if direction == 1 and local_pos > 0.70:
                    logger.debug(f"micro_entry: HARD BLOCK long — 20c local_pos={local_pos:.2f} > 0.70 (local top)")
                    return False
                if direction == -1 and local_pos < 0.30:
                    logger.debug(f"micro_entry: HARD BLOCK short — 20c local_pos={local_pos:.2f} < 0.30 (local bottom)")
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

        threshold = 3
        ok = score >= threshold
        if not ok:
            logger.debug(f"micro_entry_analysis: dir={direction} score={score}/{threshold} n={n} -> skip")
        return ok

    def _process_symbol(self, symbol: str, equity: float, open_positions: list[dict], is_priority: bool = False) -> bool:
        """Phan tich symbol, chay tat ca filter va strategy, tra True neu da trade."""
        df_micro  = self.client.get_klines(symbol, config.TIMEFRAMES["micro"], config.CANDLE_LIMIT_MICRO)
        df_scalp  = self.client.get_klines(symbol, config.TIMEFRAMES["scalp"],  config.CANDLE_LIMIT_SCALP)
        df_signal = self.client.get_klines(symbol, config.TIMEFRAMES["signal"], config.CANDLE_LIMIT_SIGNAL)
        df_trend  = self.client.get_klines(symbol, config.TIMEFRAMES["trend"],  config.CANDLE_LIMIT_TREND)
        df_macro  = self.client.get_klines(symbol, config.TIMEFRAMES["macro"],  config.CANDLE_LIMIT_MACRO)

        if df_signal.empty or len(df_signal) < 50:
            return False

        # ATR filter: bo qua symbol bien dong qua nho
        atr   = compute_atr(df_signal).iloc[-1]
        price = df_signal["close"].iloc[-1]
        if price > 0 and atr / price < config.MIN_ATR_PCT:
            return False

        # ADX filter: top10 dung nguong thap hon (15 vs 20) — coin lon trend smoother
        min_adx = 15 if is_priority else config.MIN_ADX
        adx = compute_adx(df_signal).iloc[-1]
        if math.isnan(adx) or adx < min_adx:
            logger.debug(f"{symbol}: skip — ADX={adx:.1f} < {min_adx} (sideway)")
            return False

        # 24h directional move filter: tranh chase sau khi coin da pump/dump > 20% trong 24h
        # Coin up > 20%  → block LONG momentum (move da xong, late entry); SHORT reversal van ok
        # Coin down > 20% → block SHORT momentum; LONG reversal van ok
        # Tinh tu df_signal: close[-1] vs close 96 nen 15m truoc (~24h)
        _block_long_24h  = False
        _block_short_24h = False
        if len(df_signal) >= 96:
            _ref_24h = df_signal["close"].iloc[-96]
            if _ref_24h > 0:
                _change_24h = (df_signal["close"].iloc[-1] - _ref_24h) / _ref_24h * 100
                if _change_24h > 20:
                    _block_long_24h = True
                    logger.debug(f"{symbol}: 24h change=+{_change_24h:.1f}% → block LONG (pump exhausted)")
                elif _change_24h < -20:
                    _block_short_24h = True
                    logger.debug(f"{symbol}: 24h change={_change_24h:.1f}% → block SHORT (dump exhausted)")

        # RSI cho reversal detection
        rsi_now = compute_rsi(df_signal["close"]).iloc[-1]

        # 1h range position: block long o TOP 82% / short o BOTTOM 18% cua 20-candle 1h range
        # LAB Long: vao o 85th pct → block; ADA Short: vao o 12th pct → block; DOGE Short: 15th pct → block
        # Khong ap dung cho REVERSAL (reversal chinh xac la vao o cac cuc doan nay)
        _h1_block_long  = False
        _h1_block_short = False
        if not df_trend.empty and len(df_trend) >= 20:
            h1_high = df_trend["high"].iloc[-20:].max()
            h1_low  = df_trend["low"].iloc[-20:].min()
            h1_rng  = h1_high - h1_low
            if h1_rng > 0:
                h1_pos = (price - h1_low) / h1_rng
                if h1_pos > 0.82:
                    _h1_block_long = True
                    logger.debug(f"{symbol}: 1h range_pos={h1_pos:.2f} > 0.82 → block LONG (1h top)")
                elif h1_pos < 0.18:
                    _h1_block_short = True
                    logger.debug(f"{symbol}: 1h range_pos={h1_pos:.2f} < 0.18 → block SHORT (1h bottom)")

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
        spike_lookback = 10
        recent_bodies  = closes.iloc[-spike_lookback:].values - opens.iloc[-spike_lookback:].values
        last_candle_size = abs(recent_bodies[-1])
        is_spike = last_candle_size > atr * 2.0

        # Post-spike direction block (tat ca coin)
        spike_was_dump = any(b < -atr * 2.0 for b in recent_bodies)
        spike_was_pump = any(b >  atr * 2.0 for b in recent_bodies)

        micro = self._micro_trend(df_micro)
        micro_up   = (micro == 1)
        micro_down = (micro == -1)

        scalp_trend = self._micro_trend(df_scalp)  # 5m trend cho post-spike check

        # [CHONG LO] Spike check tren 1m — direction-aware
        # Pump spike -> block LONG (khong mua dinh), nhung cho phep SHORT (ban dinh la hop le)
        # Dump spike -> block SHORT (khong ban day), nhung cho phep LONG (mua day la hop le)
        # Ca 2 cung xuat hien -> thi truong loan, skip tat ca
        _micro_spike_dump = False
        _micro_spike_pump = False
        if not df_micro.empty and len(df_micro) >= 15:
            _micro_atr    = compute_atr(df_micro).iloc[-1]
            _micro_bodies = (df_micro["close"].iloc[-15:].values - df_micro["open"].iloc[-15:].values)
            # Nguong 1.5x ATR (giam tu 2.0x): bat pump/dump vua duoi 2x ATR trong 15 nen
            _micro_spike_dump = any(b < -_micro_atr * 1.5 for b in _micro_bodies)
            _micro_spike_pump = any(b >  _micro_atr * 1.5 for b in _micro_bodies)
            # Current forming candle: block neu body 1m hien tai >= 0.4% (mid-pump/dump entry)
            # Bat cac truong hop vao lenh DANG GIUA pump — candle chua dong nen 2x ATR chua dat
            # SOXL/NEAR/HYPE/SNDK: gia tang 0.7-1.8% trong candle dang hinh thanh → block LONG
            _curr_open  = df_micro["open"].iloc[-1]
            _curr_close = df_micro["close"].iloc[-1]
            if _curr_open > 0:
                _curr_body_pct = (_curr_close - _curr_open) / _curr_open
                if _curr_body_pct > 0.004 and not _micro_spike_pump:   # +0.4% body → pump flag
                    _micro_spike_pump = True
                    logger.debug(f"{symbol}: forming 1m candle body +{_curr_body_pct*100:.2f}% → pump flag (mid-pump)")
                elif _curr_body_pct < -0.004 and not _micro_spike_dump: # -0.4% body → dump flag
                    _micro_spike_dump = True
                    logger.debug(f"{symbol}: forming 1m candle body {_curr_body_pct*100:.2f}% → dump flag (mid-dump)")
            if _micro_spike_dump and _micro_spike_pump:
                logger.debug(f"{symbol}: skip — 1m spike ca 2 chieu (thi truong loan)")
                return False
            # Cumulative net move: 15-candle lookback, 0.8% threshold
            # Bat ca dump bat dau tu 15 phut truoc (truoc chi bat 10 phut)
            _close_15_ago = df_micro["close"].iloc[-15]
            _micro_price  = df_micro["close"].iloc[-1]
            if _close_15_ago > 0:
                _net_move = (_micro_price - _close_15_ago) / _close_15_ago
                if _net_move < -0.008 and not _micro_spike_dump:   # net drop > 0.8% → dump flag
                    _micro_spike_dump = True
                    logger.debug(f"{symbol}: cumulative net dump {_net_move*100:.1f}% in 15 candles → dump flag")
                elif _net_move > 0.008 and not _micro_spike_pump:  # net pump > 0.8% → pump flag
                    _micro_spike_pump = True
                    logger.debug(f"{symbol}: cumulative net pump {_net_move*100:.1f}% in 15 candles → pump flag")

            # RSI 1m: oversold (< 35) → dump flag (tranh short o day);
            # overbought (> 65) → pump flag (tranh long o dinh)
            # Nguong 35/65 dong bo voi nguong reversal detection cua 15m
            if len(df_micro) >= 14:
                _micro_rsi = compute_rsi(df_micro["close"]).iloc[-1]
                if _micro_rsi < 35 and not _micro_spike_pump:
                    _micro_spike_dump = True
                    logger.debug(f"{symbol}: 1m RSI={_micro_rsi:.1f} oversold → dump flag (tranh short o day)")
                elif _micro_rsi > 65 and not _micro_spike_dump:
                    _micro_spike_pump = True
                    logger.debug(f"{symbol}: 1m RSI={_micro_rsi:.1f} overbought → pump flag (tranh long o dinh)")

            # Consecutive candles block: 5 nen lien tiep cung chieu = momentum extended
            # Tranh long sau 5 nen xanh lien tiep (dang o dinh), short sau 5 nen do (dang o day)
            if len(df_micro) >= 5:
                _micro_c = df_micro["close"].iloc[-5:].values
                _micro_o = df_micro["open"].iloc[-5:].values
                _all_green = all(_micro_c[i] > _micro_o[i] for i in range(5))
                _all_red   = all(_micro_c[i] < _micro_o[i] for i in range(5))
                if _all_green and not _micro_spike_pump:
                    _micro_spike_pump = True
                    logger.debug(f"{symbol}: 5 consecutive green 1m candles → pump flag (extended run)")
                if _all_red and not _micro_spike_dump:
                    _micro_spike_dump = True
                    logger.debug(f"{symbol}: 5 consecutive red 1m candles → dump flag (extended run)")

            # 30-candle extended check: bat dump/pump xay ra 15-30 phut truoc (ngoai window 15c)
            # ADA/DOGE: dump tu 30 phut truoc, gia on dinh o day → 15c miss nhung 30c bat duoc
            # WLD: dump trong 10 phut, 30c net drop > 1.2% → block short
            if len(df_micro) >= 30 and not _micro_spike_dump and not _micro_spike_pump:
                _close_30_ago = df_micro["close"].iloc[-30]
                if _close_30_ago > 0:
                    _net_move_30 = (_micro_price - _close_30_ago) / _close_30_ago
                    if _net_move_30 < -0.012:
                        _micro_spike_dump = True
                        logger.debug(f"{symbol}: 30c net dump {_net_move_30*100:.1f}% → dump flag")
                    elif _net_move_30 > 0.012:
                        _micro_spike_pump = True
                        logger.debug(f"{symbol}: 30c net pump {_net_move_30*100:.1f}% → pump flag")

            # Drop-from-high / Rise-from-low (30c): tranh short sau khi gia da roi >= 0.25% tu dinh
            # va tranh long sau khi gia da tang >= 0.25% tu day — move da xong roi, vao late
            # SKHY entry 172.18 vs high 172.42 = 0.14% drop → < 0.5% cu miss → ha xuong 0.25%
            # NEAR entry 2.0788 vs high 2.0817 = 0.14% drop → tuong tu
            # SNDK entry 1600.53 vs high 1607.74 = 0.45% drop → < 0.5% cu miss → bat duoc voi 0.25%
            if len(df_micro) >= 30:
                _high_30c = df_micro["high"].iloc[-30:].max()
                _low_30c  = df_micro["low"].iloc[-30:].min()
                if _high_30c > 0 and not _micro_spike_dump:
                    _drop_from_high = (_high_30c - _micro_price) / _high_30c
                    if _drop_from_high > 0.0025:  # gia da roi >= 0.25% tu dinh 30c
                        _micro_spike_dump = True
                        logger.debug(f"{symbol}: 30c drop-from-high {_drop_from_high*100:.2f}% → dump flag (late short)")
                if _low_30c > 0 and not _micro_spike_pump:
                    _rise_from_low = (_micro_price - _low_30c) / _low_30c
                    if _rise_from_low > 0.0025:  # gia da tang >= 0.25% tu day 30c
                        _micro_spike_pump = True
                        logger.debug(f"{symbol}: 30c rise-from-low {_rise_from_low*100:.2f}% → pump flag (late long)")

            # Range position in 30c: bottom 30% → dump flag; top 30% → pump flag
            # Nang len tu 35%/65% → 30%/70% de bat them truong hop price chua dat extreme nhung da gan dinh/day
            if len(df_micro) >= 30 and _micro_price > 0:
                _h30 = df_micro["high"].iloc[-30:].max()
                _l30 = df_micro["low"].iloc[-30:].min()
                _rng30 = _h30 - _l30
                if _rng30 > 0:
                    _pos30 = (_micro_price - _l30) / _rng30  # 0=at low, 1=at high
                    if _pos30 < 0.30 and not _micro_spike_dump:
                        _micro_spike_dump = True
                        logger.debug(f"{symbol}: price in bottom {_pos30*100:.0f}% of 30c range → dump flag (near 30c low, block short)")
                    elif _pos30 > 0.70 and not _micro_spike_pump:
                        _micro_spike_pump = True
                        logger.debug(f"{symbol}: price in top {(1-_pos30)*100:.0f}% of 30c range → pump flag (near 30c high, block long)")

            # Re-check sau extended filters: ca 2 flag co the duoc set boi cac check phia tren
            # (vi du: dump flag boi drop-from-high + pump flag boi rise-from-low trong ranging market)
            if _micro_spike_dump and _micro_spike_pump:
                logger.debug(f"{symbol}: skip — dual spike flag after extended checks (ranging/choppy 1m)")
                return False

        # 1h macro trend va 4h macro trend — can truoc BREAKOUT de tranh NameError
        macro_trend = self._trend_direction(df_trend)
        macro_4h    = self._trend_direction(df_macro)

        # Post-loss filter — tinh som de ap dung cho ca BREAKOUT va momentum
        post_loss = (time.time() - self._recent_loss_ts.get(symbol, 0)) < 300
        if post_loss:
            logger.debug(f"{symbol}: post-loss 5min active → consensus+1 / BREAKOUT blocked")

        # BREAKOUT: chay cho tat ca scan_list, skip neu post_loss
        if not post_loss and df_micro is not None and not df_micro.empty and len(df_micro) >= 30:
            bo_sig = BREAKOUT_STRATEGY.generate_signal(df_micro, df_scalp, df_signal)
            if bo_sig.direction != 0:
                # 5m khong duoc nguoc chieu — cho phep sideways
                bo_ok = (
                    (bo_sig.direction == 1  and scalp_trend >= 0) or
                    (bo_sig.direction == -1 and scalp_trend <= 0)
                )
                # 1m micro-trend cung phai xac nhan
                micro_ok = (bo_sig.direction == 1 and micro_up) or (bo_sig.direction == -1 and micro_down)
                # 15m spike block
                post_spike_ok = not (bo_sig.direction == -1 and spike_was_dump and scalp_trend != -1) and \
                                not (bo_sig.direction == 1  and spike_was_pump and scalp_trend != 1)
                # 1m micro spike direction-aware (dong bo voi momentum path)
                micro_spike_ok = not (_micro_spike_pump and bo_sig.direction == 1) and \
                                 not (_micro_spike_dump and bo_sig.direction == -1)
                # BTC global trend filter cho BREAKOUT — dong bo voi momentum path
                _bo_btc_1h = self.btc_trend
                _bo_btc_4h = self.btc_trend_4h
                _bo_coin_bear = (macro_trend == -1 and macro_4h == -1)
                _bo_coin_bull = (macro_trend ==  1 and macro_4h ==  1)
                if symbol == "BTCUSDT":
                    bo_btc_ok = not (_bo_btc_1h == 1  and bo_sig.direction == -1) and \
                                not (_bo_btc_1h == -1 and bo_sig.direction == 1)
                else:
                    # Hard block BREAKOUT nguoc BTC chi khi coin KHONG co xu huong doc lap
                    bo_btc_ok = not (_bo_btc_1h == 1  and _bo_btc_4h == 1  and bo_sig.direction == -1 and not _bo_coin_bear) and \
                                not (_bo_btc_1h == -1 and _bo_btc_4h == -1 and bo_sig.direction == 1  and not _bo_coin_bull)
                # 1h range block cho BREAKOUT — EVAA type: pump spike len top 1h range
                bo_h1_ok = not (_h1_block_long and bo_sig.direction == 1) and \
                           not (_h1_block_short and bo_sig.direction == -1)
                # 24h exhaustion block cho BREAKOUT
                bo_24h_ok = not (_block_long_24h and bo_sig.direction == 1) and \
                            not (_block_short_24h and bo_sig.direction == -1)
                # [FIX] macro_4h + macro_trend alignment — dong bo voi momentum path
                # BREAKOUT truoc day khong check 1h/4h trend, co the trade nguoc trend chinh
                bo_trend_ok = (
                    (bo_sig.direction == 1  and macro_trend >= 0 and macro_4h >= 0) or
                    (bo_sig.direction == -1 and macro_trend <= 0 and macro_4h <= 0)
                )
                if bo_ok and micro_ok and not is_spike and post_spike_ok and micro_spike_ok and bo_btc_ok and bo_h1_ok and bo_24h_ok and bo_trend_ok:
                    # BREAKOUT phai qua range check — tranh long o dinh / short o day
                    if not self._micro_entry_analysis(df_micro, bo_sig.direction):
                        logger.debug(f"{symbol}: BREAKOUT skip — range/micro_entry block")
                    else:
                        bo_sig.symbol    = symbol
                        bo_sig.consensus = 1
                        logger.info(
                            f"{symbol} [BREAKOUT TOP20] -> "
                            f"{'LONG' if bo_sig.direction==1 else 'SHORT'} "
                            f"strength={bo_sig.strength:.2f} | {bo_sig.reason}"
                        )
                        self.executor.execute_signal(symbol, bo_sig, equity, open_positions, is_priority=is_priority)
                        return True

        # Xac dinh mode: REVERSAL hay MOMENTUM
        # Nguong 35/65 dong bo voi sustained_trend va bollinger — bat duoc reversal som hon
        is_reversal  = rsi_now < 35 or rsi_now > 65
        reversal_dir = 1 if rsi_now < 35 else (-1 if rsi_now > 65 else 0)

        long_signals  = []
        short_signals = []

        for strategy in ALL_STRATEGIES:
            try:
                sig = strategy.generate_signal(df_signal, df_trend, df_macro)
                # Scalp fallback: thu 5m neu 15m khong co signal
                # Skip VWAP (window 96x15m=24h, tren 5m cho ra 8h — sai)
                # Giu nguyen df_trend (1h) va df_macro (4h) — de khong lam hong trend filter trong strategy
                if sig.direction == 0 and len(df_scalp) >= 50 and strategy.name != "vwap_volume":
                    sig = strategy.generate_signal(df_scalp, df_trend, df_macro)

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

                # 24h exhaustion: block momentum trade cung chieu move da xay ra (reversal van ok)
                if sig.direction == 1 and _block_long_24h and not is_reversal:
                    continue
                if sig.direction == -1 and _block_short_24h and not is_reversal:
                    continue

                # Long chi khi 2 nen xanh lien tiep (momentum xac nhan)
                # top10 priority: bo qua yeu cau nay, dung 1m micro trend thay the
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
                    # Ca 1h VA 4h phai khong oppose huong trade
                    # macro_trend (1h) >= 0 va macro_4h (4h) >= 0 → long ok
                    # macro_trend (1h) <= 0 va macro_4h (4h) <= 0 → short ok
                    # Neu 4h bullish ma 1h neutral → khong short (4h la trend chinh)
                    # Neu 4h bearish ma 1h neutral → khong long (4h la trend chinh)
                    long_ok  = (macro_trend >= 0) and (macro_4h >= 0)
                    short_ok = (macro_trend <= 0) and (macro_4h <= 0)
                    if sig.direction == 1 and long_ok:
                        long_signals.append(sig)
                    elif sig.direction == -1 and short_ok:
                        short_signals.append(sig)
            except Exception:
                continue

        # REVERSAL trade: RSI cuc doan + 2 nen 15m + 1m micro xac nhan dao chieu + >= MIN_CONSENSUS
        if is_reversal and reversal_dir != 0:
            # Direction-aware spike: long sau pump spike va short sau dump spike deu nguy hiem
            reversal_spike_blocked = (
                (reversal_dir == 1  and _micro_spike_pump) or
                (reversal_dir == -1 and _micro_spike_dump)
            )
            if not reversal_spike_blocked:
                # Micro trend hard block: khong reversal khi 1m dang chay nguoc chieu manh
                reversal_micro_ok = not (reversal_dir == 1 and micro_down) and not (reversal_dir == -1 and micro_up)
                reversal_confirmed = reversal_micro_ok and (
                    (reversal_dir == 1  and short_term_up)   or
                    (reversal_dir == -1 and short_term_down)
                ) and self._micro_entry_analysis(df_micro, reversal_dir, is_reversal=True)
                reversal_signals = long_signals if reversal_dir == 1 else short_signals
                reversal_base = config.MIN_CONSENSUS if is_priority else config.MIN_CONSENSUS_TRENDING
                # Deep trend guard: neu ca 1h VA 4h deu oppose reversal direction
                # (vi du: BILL -45% — 1h bearish + 4h bearish → can them +1 consensus)
                # Tranh catch the falling knife khi trend lon duoc xac nhan tren nhieu TF
                reversal_deep_opposed = (
                    (reversal_dir == 1  and macro_trend == -1 and macro_4h == -1) or
                    (reversal_dir == -1 and macro_trend ==  1 and macro_4h ==  1)
                )
                reversal_min = reversal_base + (1 if post_loss else 0) + (1 if reversal_deep_opposed else 0)
                if reversal_deep_opposed:
                    logger.debug(
                        f"{symbol}: reversal deep-trend guard +1 consensus "
                        f"(1h={'UP' if macro_trend==1 else 'DOWN'}, "
                        f"4h={'UP' if macro_4h==1 else 'DOWN'}, "
                        f"reversal={'LONG' if reversal_dir==1 else 'SHORT'}) "
                        f"→ need {reversal_min}/{len(ALL_STRATEGIES)}"
                    )
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
                    self.executor.execute_signal(symbol, best, equity, open_positions, is_priority=is_priority)
                    return True

        # RSI EXTREME GUARD: neu RSI 15m oversold (< 35) thi xoa short signals o MOMENTUM path
        # Reversal path da xu ly o tren; neu reversal khong du consensus thi KHONG duoc short them vao oversold
        # Tuong tu: RSI > 65 xoa long signals (khong long vao overbought)
        if rsi_now < 35:
            short_signals = []
        elif rsi_now > 65:
            long_signals = []

        # BTC GLOBAL TREND FILTER — HARD BLOCK khi ca 1h VA 4h BTC cung chieu
        # Neu BTC 1h+4h BULLISH → xoa het SHORT signals (tat ca coin, ke ca priority SOL/ETH)
        # Neu BTC 1h+4h BEARISH → xoa het LONG signals
        # Ngoai le: REVERSAL signal (RSI cuc doan) — reversal co the di nguoc BTC
        # Ngoai le: BTCUSDT chinh no — tu xu ly theo trend chinh no
        # Day la nguyen nhan chinh khien bot short SOL/WLD/ZEC/ADA khi BTC dang pump
        btc_trend    = self.btc_trend
        btc_trend_4h = self.btc_trend_4h

        if symbol == "BTCUSDT":
            # BTC tu xu ly: hard block nguoc trend 1h chinh no
            if btc_trend == 1:
                short_signals = []
                logger.debug("BTCUSDT: clear SHORT — BTC 1h UP")
            elif btc_trend == -1:
                long_signals = []
                logger.debug("BTCUSDT: clear LONG — BTC 1h DOWN")
        else:
            # Altcoin: phan tich BTC alignment de quyet dinh hard/soft block
            btc_strongly_bull = (btc_trend == 1  and btc_trend_4h == 1)
            btc_strongly_bear = (btc_trend == -1 and btc_trend_4h == -1)

            # Coin co xu huong doc lap nguoc BTC (ca 1h VA 4h cua chinh coin do)
            # Vi du: BTC bull nhung coin rieng dang bearish 1h+4h → co the cho phep short
            # Yeu cau them consensus cao hon (xu ly o phan consensus ben duoi)
            coin_independently_bear = (macro_trend == -1 and macro_4h == -1)
            coin_independently_bull = (macro_trend ==  1 and macro_4h ==  1)

            # Hard block: BTC strongly opposes AND coin khong co xu huong doc lap nguoc lai
            # Neu coin co xu huong doc lap → cho phep nhung se cap cao consensus
            if btc_strongly_bull and not is_reversal and not coin_independently_bear:
                short_signals = []
                logger.debug(f"{symbol}: BTC 1h+4h BULLISH, coin not independently bearish → block SHORT")
            if btc_strongly_bear and not is_reversal and not coin_independently_bull:
                long_signals = []
                logger.debug(f"{symbol}: BTC 1h+4h BEARISH, coin not independently bullish → block LONG")

        # BTC alignment flags cho consensus adjustment
        btc_strongly_bull = (btc_trend == 1  and btc_trend_4h == 1)   if symbol != "BTCUSDT" else False
        btc_strongly_bear = (btc_trend == -1 and btc_trend_4h == -1)  if symbol != "BTCUSDT" else False
        coin_independently_bear = (macro_trend == -1 and macro_4h == -1)
        coin_independently_bull = (macro_trend ==  1 and macro_4h ==  1)

        # Soft penalty cho non-priority khi BTC 1 TF nguoc (chua confirm 2/2)
        btc_opposes_long  = (btc_trend == -1 and symbol != "BTCUSDT" and not is_priority and btc_trend_4h != -1)
        btc_opposes_short = (btc_trend ==  1 and symbol != "BTCUSDT" and not is_priority and btc_trend_4h != 1)

        # MOMENTUM trade
        sideways_1h = (macro_trend == 0)

        if is_priority:
            # Priority (top10): base = MIN_CONSENSUS = 4
            # BTC cung chieu (bonus) → giam 1 → 3 (bat nhieu co hoi hon)
            # Coin diverge nguoc BTC → tang 2 → 6 (can xac nhan cao)
            base = config.MIN_CONSENSUS
            extra = 1 if post_loss else 0
            btc_long_bonus  = 1 if btc_strongly_bull else 0
            btc_short_bonus = 1 if btc_strongly_bear else 0
            diverge_long_penalty  = 2 if (btc_strongly_bear and coin_independently_bull)  else 0
            diverge_short_penalty = 2 if (btc_strongly_bull and coin_independently_bear) else 0
            required_long  = max(2, min(7, base + extra - btc_long_bonus  + diverge_long_penalty))
            required_short = max(2, min(7, base + extra - btc_short_bonus + diverge_short_penalty))
        else:
            # Trending non-priority: base = MIN_CONSENSUS_TRENDING = 5
            # BTC cung chieu → giam 1 → 4 (non-priority de vao hon khi trend ro)
            # Coin diverge nguoc BTC → tang 2 → 7 (rat kho vao, can gan tat ca strategies)
            base = config.MIN_CONSENSUS_TRENDING
            extra = (1 if sideways_1h else 0) + (1 if post_loss else 0)
            btc_long_bonus  = 1 if btc_strongly_bull else 0
            btc_short_bonus = 1 if btc_strongly_bear else 0
            diverge_long_penalty  = 2 if (btc_strongly_bear and coin_independently_bull)  else 0
            diverge_short_penalty = 2 if (btc_strongly_bull and coin_independently_bear) else 0
            required_long  = max(2, min(7, base + extra - btc_long_bonus  + diverge_long_penalty + (1 if btc_opposes_long  else 0)))
            required_short = max(2, min(7, base + extra - btc_short_bonus + diverge_short_penalty + (1 if btc_opposes_short else 0)))

        # TOP10 PRIORITY: 2 trong 2 Tier-1 strategy (supertrend + vwap_volume) dong thuan -> trade
        # Tier-1 bypass: KHONG bi chan boi BTC filter — top10 coin lon co momentum rieng
        # post_loss KHONG ap dung cho Tier1 — tin hieu Tier1 du manh de vao lai ngay
        TIER1 = {"supertrend", "vwap_volume"}
        tier1_long  = sum(1 for s in long_signals  if s.strategy_name in TIER1)
        tier1_short = sum(1 for s in short_signals if s.strategy_name in TIER1)

        # Tier1 bypass: chi khi ca 2 Tier1 cung chieu (2/2) va KHONG conflict
        # Neu conflict (1 long + 1 short): KHONG skip toan bo — van cho consensus check chay
        # [FIX] Tier1 bypass phai ton trong macro_4h alignment — tranh bypass trong reversal mode
        # khi signals vao tu reversal branch (khong co macro check)
        tier1_bypass_long  = (is_priority and tier1_long >= 2 and tier1_short == 0
                              and macro_trend >= 0 and macro_4h >= 0
                              and not btc_strongly_bear)
        tier1_bypass_short = (is_priority and tier1_short >= 2 and tier1_long == 0
                              and macro_trend <= 0 and macro_4h <= 0
                              and not btc_strongly_bull)

        if tier1_bypass_long:
            signals = long_signals
            logger.info(f"{symbol}: [TOP10 TIER1] 2/2 Tier-1 LONG — bypass consensus")
        elif tier1_bypass_short:
            signals = short_signals
            logger.info(f"{symbol}: [TOP10 TIER1] 2/2 Tier-1 SHORT — bypass consensus")
        elif len(long_signals) >= required_long:
            signals = long_signals
        elif len(short_signals) >= required_short:
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

        # 1h range hard block (MOMENTUM path only — REVERSAL duoc phep o cuc doan)
        # Block long o top 82% / short o bottom 18% cua 20-candle 1h range
        if _h1_block_long and best.direction == 1:
            logger.debug(f"{symbol}: skip — price at 1h range top (>82%), block MOMENTUM LONG")
            return False
        if _h1_block_short and best.direction == -1:
            logger.debug(f"{symbol}: skip — price at 1h range bottom (<18%), block MOMENTUM SHORT")
            return False

        # 1m micro trend confirmation — tat ca coin (micro trend phai cung chieu hoac neutral)
        # Tranh trade khi 1m dang nguoc chieu hoan toan voi signal
        if best.direction == 1 and micro_down:
            logger.debug(f"{symbol}: skip — 1m micro trend BEARISH vs LONG signal")
            return False
        if best.direction == -1 and micro_up:
            logger.debug(f"{symbol}: skip — 1m micro trend BULLISH vs SHORT signal")
            return False

        # 1m EMA alignment check (MOMENTUM path) — bat cac truong hop _micro_trend tra ve 0 (neutral)
        # do chi dat 2/5 factors thay vi 3/5, nhung EMA9 vs EMA21 dang nguoc chieu ro rang
        # EMA9 < EMA21: 1m bearish alignment → tranh long; EMA9 > EMA21: 1m bullish → tranh short
        if len(df_micro) >= 21:
            _e9  = compute_ema(df_micro["close"], 9).iloc[-1]
            _e21 = compute_ema(df_micro["close"], 21).iloc[-1]
            if best.direction == 1 and _e9 < _e21:
                logger.debug(f"{symbol}: skip — 1m EMA9({_e9:.4f}) < EMA21({_e21:.4f}), bearish micro, block LONG")
                return False
            if best.direction == -1 and _e9 > _e21:
                logger.debug(f"{symbol}: skip — 1m EMA9({_e9:.4f}) > EMA21({_e21:.4f}), bullish micro, block SHORT")
                return False

        # EMA50 pullback filter (15m): chi enter khi gia GAN EMA50, khong chase khi da extended xa
        # Uptrend LONG: price nen bounce tu EMA50 (support), khong phai cach EMA50 qua xa
        # Downtrend SHORT: price nen tu EMA50 (resistance) xuong, khong phai da qua extended
        # Muc 2.5x ATR_15m: cho phep price o tren/duoi EMA50 mot chut (momentum), nhung khong qua xa
        if len(df_signal) >= 50:
            _ema50_15m = compute_ema(df_signal["close"], 50).iloc[-1]
            _atr_15m   = compute_atr(df_signal, config.ATR_PERIOD).iloc[-1]
            _p15 = df_signal["close"].iloc[-1]
            if _ema50_15m > 0 and _atr_15m > 0:
                _ema50_dist = _p15 - _ema50_15m  # + = above, - = below
                if best.direction == 1 and _ema50_dist > 2.5 * _atr_15m:
                    logger.debug(
                        f"{symbol}: skip LONG — price {_ema50_dist/_ema50_15m*100:.1f}% above 15m EMA50 "
                        f"({_ema50_dist/(_atr_15m+1e-9):.1f}x ATR, too extended)"
                    )
                    return False
                if best.direction == -1 and _ema50_dist < -2.5 * _atr_15m:
                    logger.debug(
                        f"{symbol}: skip SHORT — price {-_ema50_dist/_ema50_15m*100:.1f}% below 15m EMA50 "
                        f"({-_ema50_dist/(_atr_15m+1e-9):.1f}x ATR, too extended)"
                    )
                    return False

        # 1m micro entry timing: apply cho TAT CA coin voi phan tich day du 5 yeu to
        if not self._micro_entry_analysis(df_micro, best.direction):
            logger.debug(f"{symbol}: skip — 1m micro entry timing not confirmed (score too low)")
            return False

        # TOP_PRIORITY: cross-check voi 1h frame — tranh trade khi 1h nguoc chieu hoan toan
        if is_priority and len(df_trend) >= 50:
            h1_confirms = 0
            h1_opposes  = 0
            for strategy in ALL_STRATEGIES:
                try:
                    if strategy.name == "sustained_trend":
                        continue
                    # df=1h candles, df_trend=1h (itself as trend ref), df_macro=4h
                    sig_1h = strategy.generate_signal(df_trend, df_trend, df_macro)
                    if sig_1h.direction == best.direction and sig_1h.strength >= config.MIN_SIGNAL_STRENGTH:
                        h1_confirms += 1
                    elif sig_1h.direction == -best.direction and sig_1h.strength >= config.MIN_SIGNAL_STRENGTH:
                        h1_opposes += 1
                except Exception:
                    continue
            logger.info(
                f"{symbol}: [1H CROSS-REF] {'LONG' if best.direction==1 else 'SHORT'} — "
                f"confirm={h1_confirms} oppose={h1_opposes}"
            )
            if h1_confirms == 0 and h1_opposes > 0:
                logger.debug(f"{symbol}: TOP_PRIORITY skip — 1h opposes signal, no 1h confirmation")
                return False

        best.consensus = len(signals)
        best.symbol    = symbol
        names = "+".join(s.strategy_name for s in signals)

        logger.info(
            f"{symbol} [{names}] consensus={len(signals)} -> "
            f"{'LONG' if best.direction==1 else 'SHORT'} "
            f"strength={best.strength:.2f} | {best.reason}"
        )

        self.executor.execute_signal(symbol, best, equity, open_positions, is_priority=is_priority)
        return True


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    bot = TradingBot()
    bot.run()
