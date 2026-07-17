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

        # Track positions de detect SL/TP hit boi exchange (khong qua executor)
        self._prev_pos_symbols: set[str] = set()
        # BTC global trend: +1 uptrend, -1 downtrend, 0 sideways (cap nhat moi tick)
        self.btc_trend: int = 0
        self.btc_trend_4h: int = 0

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
                logger.info(f"Symbols updated: {len(self.symbols)}, top 10: {self.symbols[:10]}")
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

        # Detect position dong boi exchange (SL/TP hit) — xoa executor state de tranh stale
        closed_by_exchange = self._prev_pos_symbols - pos_symbols
        if closed_by_exchange:
            for sym in closed_by_exchange:
                self.executor.clear_position_state(sym)
                logger.info(f"{sym}: position closed by exchange (SL/TP hit) — state cleared")
        self._prev_pos_symbols = pos_symbols

        # Cap nhat BTC global trend TRUOC cap check — tranh BTC trend stale khi at max positions
        try:
            df_btc_1h = self.client.get_klines("BTCUSDT", config.TIMEFRAMES["trend"], 100)
            df_btc_4h = self.client.get_klines("BTCUSDT", config.TIMEFRAMES["macro"],  100)
            if not df_btc_1h.empty and len(df_btc_1h) >= 50:
                self.btc_trend = self._trend_direction(df_btc_1h)
            if not df_btc_4h.empty and len(df_btc_4h) >= 50:
                self.btc_trend_4h = self._trend_direction(df_btc_4h)
        except Exception:
            pass

        # Hard cap: khong mo them lenh neu da dat MAX_OPEN_POSITIONS
        if len(open_positions) >= config.MAX_OPEN_POSITIONS:
            logger.info(
                f"[TICK] Max positions ({config.MAX_OPEN_POSITIONS}) reached — skip new entries"
            )
            return

        # Chi trade TOP 10 coins theo volume (chat luong cao nhat, thanh khoan tot nhat)
        # Bo trending list hoan toan — tap trung 100% capacity vao 10 coin chinh
        top10 = self.symbols[:10]
        priority_set = set(top10)
        scan_list = top10  # tat ca top10 deu la priority

        logger.info(
            f"[TICK] TOP10 scan: {scan_list} | "
            f"BTC_1h={'UP' if self.btc_trend==1 else 'DOWN' if self.btc_trend==-1 else 'SIDE'} "
            f"BTC_4h={'UP' if self.btc_trend_4h==1 else 'DOWN' if self.btc_trend_4h==-1 else 'SIDE'}"
        )

        # BTC/ETH direction map: de check correlation truoc khi mo lenh moi
        # {symbol: "Long" | "Short"} cho cac position dang mo
        _pos_side_map = {p["symbol"]: p.get("side", "") for p in open_positions}

        for symbol in scan_list:

            if symbol in pos_symbols:
                continue

            is_priority = True  # tat ca top10 deu la priority
            try:
                traded = self._process_symbol(
                    symbol, equity, open_positions, is_priority,
                    btc_eth_side_map=_pos_side_map
                )
                if traded:
                    try:
                        open_positions = self.client.get_positions()
                        equity         = self.client.get_wallet_balance()
                        pos_symbols    = {p["symbol"] for p in open_positions}
                        _pos_side_map  = {p["symbol"]: p.get("side", "") for p in open_positions}
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
            return False  # khong co data -> khong trade
        n = len(df_micro)
        if n < 5:
            return False  # qua it data -> khong trade

        close  = df_micro["close"]
        open_  = df_micro["open"]
        high   = df_micro["high"]
        low    = df_micro["low"]
        volume = df_micro["volume"]

        atr_1m = compute_atr(df_micro).iloc[-1]
        if atr_1m == 0:
            return False  # gia bat dong -> khong trade

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

        # Factor 2: Momentum — it nhat 1/3 nen gan nhat phai cung chieu (giam tu 2/3)
        # 2/3 qua chat cho breakout moi bat dau: nen dao chieu chi co 1 nen cung chieu
        # Chi penalty -1 khi CA 3 nen deu nguoc chieu (ro rang counter-momentum)
        bodies_3 = close.iloc[-3:].values - open_.iloc[-3:].values
        bull3 = sum(1 for b in bodies_3 if b > 0)
        bear3 = sum(1 for b in bodies_3 if b < 0)
        if direction == 1 and bull3 >= 2:
            score += 1  # strong momentum
        elif direction == -1 and bear3 >= 2:
            score += 1  # strong momentum
        elif direction == 1 and bear3 == 3:
            score -= 1  # ca 3 nen do khi muon long = bad
        elif direction == -1 and bull3 == 3:
            score -= 1  # ca 3 nen xanh khi muon short = bad

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

        # Factor 6: Range check — 100c (xu huong trung han) + 20c (local)
        # HARD BLOCK chi khi cuc ki cuc doan (dang o top/bottom 10% of range)
        # Nguong cu 75%/25% (100c) va 65%/35% (20c) qua chat — block het trend-following entries:
        #   Trong downtrend, price luon o bottom 25% cua 100c -> ALL SHORT blocked
        #   Trong uptrend, price luon o top 25% cua 100c -> ALL LONG blocked
        # Muc 90%/10% chi block khi THUC SU da cham cuc (exhaustion zone)
        # Ngoai le: is_reversal=True -> skip range block (reversal chinh xac la vao o cuc doan)
        _range_window = min(100, n)
        if _range_window >= 20:
            high_rng = high.iloc[-_range_window:].max()
            low_rng  = low.iloc[-_range_window:].min()
            rng = high_rng - low_rng
            if rng > 0:
                range_pos = (price - low_rng) / rng
                if not is_reversal:
                    if direction == 1 and range_pos > 0.90:  # chi block khi o top 10% (tang tu 75%)
                        logger.debug(f"micro_entry: HARD BLOCK long — 100c range_pos={range_pos:.2f} > 0.90")
                        return False
                    if direction == -1 and range_pos < 0.10:  # chi block khi o bottom 10% (tang tu 25%)
                        logger.debug(f"micro_entry: HARD BLOCK short — 100c range_pos={range_pos:.2f} < 0.10")
                        return False
                # Bonus cho entry o vung an toan (range 30%-70%)
                if direction == 1 and range_pos < 0.45:
                    score += 1
                elif direction == -1 and range_pos > 0.55:
                    score += 1

        # 20-candle local range: chi block khi thuc su o cuc doan nho (top/bottom 15%)
        # Nguong cu 65%/35% qua chat — block moi breakout DOWN (luon o bottom 20c sau breakdown)
        _local_window = min(20, n)
        if _local_window >= 10 and not is_reversal:
            local_high = high.iloc[-_local_window:].max()
            local_low  = low.iloc[-_local_window:].min()
            local_rng  = local_high - local_low
            if local_rng > 0:
                local_pos = (price - local_low) / local_rng
                if direction == 1 and local_pos > 0.88:  # tang tu 0.65 -> 0.88
                    logger.debug(f"micro_entry: HARD BLOCK long — 20c local_pos={local_pos:.2f} > 0.88")
                    return False
                if direction == -1 and local_pos < 0.12:  # giam tu 0.35 -> 0.12
                    logger.debug(f"micro_entry: HARD BLOCK short — 20c local_pos={local_pos:.2f} < 0.12")
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

    def _process_symbol(self, symbol: str, equity: float, open_positions: list[dict], is_priority: bool = False, btc_eth_side_map: dict | None = None) -> bool:
        """Phan tich symbol, chay tat ca filter va strategy, tra True neu da trade."""
        # Init gradual trend flags truoc block 30c de tranh NameError neu df_micro < 30 candles
        _is_gradual_uptrend   = False
        _is_gradual_downtrend = False

        df_micro  = self.client.get_klines(symbol, config.TIMEFRAMES["micro"], config.CANDLE_LIMIT_MICRO)
        df_scalp  = self.client.get_klines(symbol, config.TIMEFRAMES["scalp"],  config.CANDLE_LIMIT_SCALP)
        df_signal = self.client.get_klines(symbol, config.TIMEFRAMES["signal"], config.CANDLE_LIMIT_SIGNAL)
        df_trend  = self.client.get_klines(symbol, config.TIMEFRAMES["trend"],  config.CANDLE_LIMIT_TREND)
        df_macro  = self.client.get_klines(symbol, config.TIMEFRAMES["macro"],  config.CANDLE_LIMIT_MACRO)

        if df_signal.empty or len(df_signal) < 50:
            return False

        # Large-cap flag: BTC va ETH co % volatility thap hon altcoin (~0.5x)
        # Tat ca percentage thresholds (spike, pump/dump detection) can duoc scale xuong
        # De tranh ETH/BTC bi bo qua (ATR% nho) hoac vao lenh tai dinh/day chua duoc bat
        _is_largecap = symbol in {"BTCUSDT", "ETHUSDT"}
        # Scale factor cho tat ca % threshold: largecap dung 0.5x
        _sp = 0.5 if _is_largecap else 1.0

        # ATR filter: bo qua symbol bien dong qua nho
        # Large-cap: 0.2% (BTC ATR% ~0.3-0.4%, ETH ~0.3%), altcoin: 0.4%
        atr   = compute_atr(df_signal).iloc[-1]
        price = df_signal["close"].iloc[-1]
        _min_atr_pct = config.MIN_ATR_PCT * _sp
        if price > 0 and atr / price < _min_atr_pct:
            return False

        # Top5 mode: tat ca deu la priority, dung config.MIN_ADX cho tat ca (22)
        min_adx = config.MIN_ADX
        adx = compute_adx(df_signal).iloc[-1]
        if math.isnan(adx) or adx < min_adx:
            logger.debug(f"{symbol}: skip — ADX={adx:.1f} < {min_adx} (sideway)")
            return False

        # 24h directional move filter: tranh chase sau khi coin da pump/dump > 20% trong 24h
        # Coin up > 20%  -> block LONG momentum (move da xong, late entry); SHORT reversal van ok
        # Coin down > 20% -> block SHORT momentum; LONG reversal van ok
        # HARD SKIP: abs > 30% -> skip TOAN BO (AKEUSDT +39%: ca SHORT reversal cung nguy hiem)
        # Scanner da skip o >25% nhung self.symbols la cache cu (1h) -> coin co the pump them
        # Tinh tu df_signal: close[-1] vs close 96 nen 15m truoc (~24h)
        _block_long_24h  = False
        _block_short_24h = False
        _change_24h = 0.0
        if len(df_signal) >= 96:
            _ref_24h = df_signal["close"].iloc[-96]
            if _ref_24h > 0:
                _change_24h = (df_signal["close"].iloc[-1] - _ref_24h) / _ref_24h * 100
                if abs(_change_24h) > 30:
                    logger.info(f"{symbol}: 24h change={_change_24h:.1f}% > 30% -> HARD SKIP (extreme move)")
                    return False
                if _change_24h > 20:
                    _block_long_24h = True
                    logger.debug(f"{symbol}: 24h change=+{_change_24h:.1f}% -> block LONG (pump exhausted)")
                elif _change_24h < -20:
                    _block_short_24h = True
                    logger.debug(f"{symbol}: 24h change={_change_24h:.1f}% -> block SHORT (dump exhausted)")

        # RSI cho reversal detection
        rsi_now = compute_rsi(df_signal["close"]).iloc[-1]

        # Lay live mark price mot lan cho range position checks — tranh dung 15m close (stale up to 14m)
        # Tai day la diem dau tien co du context de goi API (sau spike filter da pass)
        # Reuse cho _live_check_price trong 30c block de tranh second API call
        _range_live_price = self.client.get_current_price(symbol)
        _range_price = _range_live_price if _range_live_price > 0 else price

        # 1h range position: block long o TOP 75% / short o BOTTOM 25% cua 20-candle 1h range
        # Khong ap dung cho REVERSAL (reversal chinh xac la vao o cac cuc doan nay)
        _h1_block_long  = False
        _h1_block_short = False
        if not df_trend.empty and len(df_trend) >= 20:
            h1_high = df_trend["high"].iloc[-20:].max()
            h1_low  = df_trend["low"].iloc[-20:].min()
            h1_rng  = h1_high - h1_low
            if h1_rng > 0:
                h1_pos = (_range_price - h1_low) / h1_rng
                if h1_pos > 0.75:
                    _h1_block_long = True
                    logger.debug(f"{symbol}: 1h range_pos={h1_pos:.2f} > 0.75 -> block LONG (1h top)")
                elif h1_pos < 0.25:
                    _h1_block_short = True
                    logger.debug(f"{symbol}: 1h range_pos={h1_pos:.2f} < 0.25 -> block SHORT (1h bottom)")

        # 2h 1m range: block SHORT khi gia o bottom 20% cua range 120 nen 1m (2 gio)
        # Block LONG khi o top 80%
        # ONDOUSDT/SKHYNIXUSDT/XAGUSDT pattern: price dump 2h truoc, then bot vao SHORT o day -> loss
        # 120c = 2h 1m candles = du dai de bat dump xay ra truoc 30-60 phut
        _m2h_block_long  = False
        _m2h_block_short = False
        _m2h_pos = 0.5  # default mid-range (used also in reversal extreme block below)
        if not df_micro.empty and len(df_micro) >= 120:
            _m2h_high = df_micro["high"].iloc[-120:].max()
            _m2h_low  = df_micro["low"].iloc[-120:].min()
            _m2h_rng  = _m2h_high - _m2h_low
            if _m2h_rng > 0:
                _m2h_pos = (_range_price - _m2h_low) / _m2h_rng
                # 80%/20% uniform for all coins — live price ensures accuracy.
                # Gradual-trend exception in 30c/60c/5m blocks handles "allow LONG in uptrend";
                # this guard prevents chasing near 2h range extremes regardless of coin size.
                _m2h_top_thresh = 0.80
                _m2h_bot_thresh = 0.20
                if _m2h_pos < _m2h_bot_thresh:
                    _m2h_block_short = True
                    logger.debug(f"{symbol}: 2h 1m range_pos={_m2h_pos:.2f} < {_m2h_bot_thresh} -> block SHORT (2h bottom)")
                elif _m2h_pos > _m2h_top_thresh:
                    _m2h_block_long = True
                    logger.debug(f"{symbol}: 2h 1m range_pos={_m2h_pos:.2f} > {_m2h_top_thresh} -> block LONG (2h top)")

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
        _live_check_price = 0.0  # init truoc; se duoc set tu get_current_price trong block 30c
        # _micro_price: init truoc block de tranh NameError khi df_micro co < 15 candles
        _micro_price = df_micro["close"].iloc[-1] if not df_micro.empty else 0.0
        if not df_micro.empty and len(df_micro) >= 15:
            _micro_atr    = compute_atr(df_micro).iloc[-1]
            _micro_bodies = (df_micro["close"].iloc[-15:].values - df_micro["open"].iloc[-15:].values)
            # Nguong 1.5x ATR (giam tu 2.0x): bat pump/dump vua duoi 2x ATR trong 15 nen
            _micro_spike_dump = any(b < -_micro_atr * 1.5 for b in _micro_bodies)
            _micro_spike_pump = any(b >  _micro_atr * 1.5 for b in _micro_bodies)
            # Current forming candle: block neu body 1m hien tai >= 0.4% (mid-pump/dump entry)
            # Bat cac truong hop vao lenh DANG GIUA pump — candle chua dong nen 2x ATR chua dat
            # SOXL/NEAR/HYPE/SNDK: gia tang 0.7-1.8% trong candle dang hinh thanh -> block LONG
            _curr_open  = df_micro["open"].iloc[-1]
            _curr_close = df_micro["close"].iloc[-1]
            if _curr_open > 0:
                _curr_body_pct = (_curr_close - _curr_open) / _curr_open
                if _curr_body_pct > 0.003 * _sp and not _micro_spike_pump:
                    _micro_spike_pump = True
                    logger.debug(f"{symbol}: forming 1m candle body +{_curr_body_pct*100:.2f}% -> pump flag (mid-pump)")
                elif _curr_body_pct < -0.003 * _sp and not _micro_spike_dump:
                    _micro_spike_dump = True
                    logger.debug(f"{symbol}: forming 1m candle body {_curr_body_pct*100:.2f}% -> dump flag (mid-dump)")
            if _micro_spike_dump and _micro_spike_pump:
                logger.debug(f"{symbol}: skip — 1m spike ca 2 chieu (thi truong loan)")
                return False
            # Cumulative net move: 15-candle lookback, 0.8% threshold
            # Bat ca dump bat dau tu 15 phut truoc (truoc chi bat 10 phut)
            _close_15_ago = df_micro["close"].iloc[-15]
            _micro_price  = df_micro["close"].iloc[-1]
            if _close_15_ago > 0:
                _net_move = (_micro_price - _close_15_ago) / _close_15_ago
                if _net_move < -0.008 * _sp and not _micro_spike_dump:
                    _micro_spike_dump = True
                    logger.debug(f"{symbol}: cumulative net dump {_net_move*100:.1f}% in 15 candles -> dump flag")
                elif _net_move > 0.008 * _sp and not _micro_spike_pump:
                    _micro_spike_pump = True
                    logger.debug(f"{symbol}: cumulative net pump {_net_move*100:.1f}% in 15 candles -> pump flag")

            # RSI 1m: chi block khi CUC DOAN that su (< 20 hoac > 80)
            # Nguong 35/65 cu qua rong: trong downtrend 1m RSI thuong 25-40 -> block het SHORT
            # 20/80: chi bat truong hop panic dump/pump that su (gap xuong, margin call cascade)
            if len(df_micro) >= 14:
                _micro_rsi = compute_rsi(df_micro["close"]).iloc[-1]
                if _micro_rsi < 20 and not _micro_spike_pump:
                    _micro_spike_dump = True
                    logger.debug(f"{symbol}: 1m RSI={_micro_rsi:.1f} extreme oversold -> dump flag")
                elif _micro_rsi > 80 and not _micro_spike_dump:
                    _micro_spike_pump = True
                    logger.debug(f"{symbol}: 1m RSI={_micro_rsi:.1f} extreme overbought -> pump flag")

            # Consecutive candles block: largecap 6 nen (8 lien tiep tren BTC/ETH rat hiem)
            # Altcoin: 8 nen lien tiep = exhaustion / dao chieu
            _consec_n = 6 if _is_largecap else 8
            if len(df_micro) >= _consec_n:
                _micro_c = df_micro["close"].iloc[-_consec_n:].values
                _micro_o = df_micro["open"].iloc[-_consec_n:].values
                _all_green = all(_micro_c[i] > _micro_o[i] for i in range(_consec_n))
                _all_red   = all(_micro_c[i] < _micro_o[i] for i in range(_consec_n))
                if _all_green and not _micro_spike_pump:
                    _micro_spike_pump = True
                    logger.debug(f"{symbol}: {_consec_n} consecutive green 1m candles -> pump flag (exhaustion)")
                if _all_red and not _micro_spike_dump:
                    _micro_spike_dump = True
                    logger.debug(f"{symbol}: {_consec_n} consecutive red 1m candles -> dump flag (exhaustion)")

            # 30-candle extended check: bat dump/pump xay ra 15-30 phut truoc (ngoai window 15c)
            # ADA/DOGE: dump tu 30 phut truoc, gia on dinh o day -> 15c miss nhung 30c bat duoc
            # WLD: dump trong 10 phut, 30c net drop > 1.2% -> block short
            if len(df_micro) >= 30 and not _micro_spike_dump and not _micro_spike_pump:
                _close_30_ago = df_micro["close"].iloc[-30]
                if _close_30_ago > 0:
                    _net_move_30 = (_micro_price - _close_30_ago) / _close_30_ago
                    if _net_move_30 < -0.012 * _sp:
                        _micro_spike_dump = True
                        logger.debug(f"{symbol}: 30c net dump {_net_move_30*100:.1f}% -> dump flag")
                    elif _net_move_30 > 0.012 * _sp:
                        _micro_spike_pump = True
                        logger.debug(f"{symbol}: 30c net pump {_net_move_30*100:.1f}% -> pump flag")

            # Drop-from-high / Rise-from-low (30c): chi block khi da di >= 0.60% (spike that su)
            # 0.20% cu qua nho — trong downtrend bat ky 30c nao cung co drop > 0.20% -> block het SHORT
            # 0.60% = dich chuyen that su, price da di xa khoi vung vao lenh tot
            # BUGFIX: dung live mark price thay vi _micro_price (last closed candle) de bat forming-candle dump
            # Truoc: neu dump xay ra trong forming candle (chua dong), _micro_price = gia truoc dump -> miss
            # Sau: _live_check_price = max(live_price, _micro_price) -> bat ca hai truong hop
            if len(df_micro) >= 30:
                _high_30c = df_micro["high"].iloc[-30:].max()
                _low_30c  = df_micro["low"].iloc[-30:].min()
                # Reuse live price tu _range_live_price (da fetch o tren); fallback ve _micro_price
                _live_check_price = _range_live_price if _range_live_price > 0 else _micro_price

                # Phan biet spike vs gradual trend dua tren ty le nen theo chieu:
                # Spike: 1-3 nen khong lo, phan lon cac nen con lai flat
                # Trend: >= 50% nen trong 30c la nen cung chieu -> la trend that su, khong block
                _30c_closes = df_micro["close"].iloc[-30:].values
                _30c_opens  = df_micro["open"].iloc[-30:].values
                _n_green_30 = sum(1 for i in range(30) if _30c_closes[i] > _30c_opens[i])
                _n_red_30   = sum(1 for i in range(30) if _30c_closes[i] < _30c_opens[i])
                _is_gradual_uptrend   = _n_green_30 >= 18  # >= 60% nen xanh = uptrend ro rang (18+18>30 -> not both true)
                _is_gradual_downtrend = _n_red_30   >= 18  # >= 60% nen do  = downtrend ro rang

                if _high_30c > 0 and not _micro_spike_dump:
                    _drop_from_high = (_high_30c - _live_check_price) / _high_30c
                    # Spike: 0.60% threshold; Gradual downtrend (>=50% red candles): raise to 2.0%
                    # Downtrend that su -> cho phep vao SHORT, chi block khi drop THAT SU nhanh (spike)
                    _dump_threshold = 0.0200 * _sp if _is_gradual_downtrend else 0.0060 * _sp
                    if _drop_from_high > _dump_threshold:
                        _micro_spike_dump = True
                        logger.debug(f"{symbol}: 30c drop-from-high {_drop_from_high*100:.2f}% > {_dump_threshold*100:.2f}% (live={_live_check_price:.4f}) -> dump flag")
                if _low_30c > 0 and not _micro_spike_pump:
                    _rise_from_low = (_live_check_price - _low_30c) / _low_30c
                    # Spike: 0.60% threshold; Gradual uptrend (>=50% green candles): raise to 2.0%
                    # Uptrend that su -> cho phep vao LONG, chi block khi rise THAT SU nhanh (spike)
                    _pump_threshold = 0.0200 * _sp if _is_gradual_uptrend else 0.0060 * _sp
                    if _rise_from_low > _pump_threshold:
                        _micro_spike_pump = True
                        logger.debug(f"{symbol}: 30c rise-from-low {_rise_from_low*100:.2f}% > {_pump_threshold*100:.2f}% -> pump flag")

            # 30c range position block da DUOC XOA:
            # _pos30 < 0.35 -> dump flag: SAI trong downtrend (price luon o bottom 35% -> block het SHORT)
            # _pos30 > 0.65 -> pump flag: SAI trong uptrend (price luon o top 35% -> block het LONG)
            # Hay de macro trend + ADX + consensus xu ly phan nay

            # Re-check sau extended filters: ca 2 flag co the duoc set boi cac check phia tren
            # (vi du: dump flag boi drop-from-high + pump flag boi rise-from-low trong ranging market)
            if _micro_spike_dump and _micro_spike_pump:
                logger.debug(f"{symbol}: skip — dual spike flag after extended checks (ranging/choppy 1m)")
                return False

        # Drop-from-high / Rise-from-low (60c): spike threshold 1.0%; gradual trend threshold 2.5%
        # Ap dung logic phan biet spike vs trend tuong tu 30c block
        if not df_micro.empty and len(df_micro) >= 60:
            _high_60c = df_micro["high"].iloc[-60:].max()
            _low_60c  = df_micro["low"].iloc[-60:].min()
            # Reuse _live_check_price (set trong block 30c); fallback ve _range_price (live hoac 15m close)
            _lcp60 = _live_check_price if _live_check_price > 0 else _range_price
            # Phan biet spike vs gradual trend qua 60 nen 1m
            _60c_closes = df_micro["close"].iloc[-60:].values
            _60c_opens  = df_micro["open"].iloc[-60:].values
            _n_green_60 = sum(1 for i in range(60) if _60c_closes[i] > _60c_opens[i])
            _n_red_60   = sum(1 for i in range(60) if _60c_closes[i] < _60c_opens[i])
            _is_grad_up_60   = _n_green_60 >= 36  # >= 60% nen xanh = uptrend ro rang (36+36>60 -> not both true)
            _is_grad_down_60 = _n_red_60   >= 36  # >= 60% nen do  = downtrend ro rang
            if _high_60c > 0 and not _micro_spike_dump:
                _drop_60 = (_high_60c - _lcp60) / _high_60c
                _dump_thr_60 = 0.0250 * _sp if _is_grad_down_60 else 0.0100 * _sp
                if _drop_60 > _dump_thr_60:
                    _micro_spike_dump = True
                    logger.debug(f"{symbol}: 60c drop-from-high {_drop_60*100:.2f}% > {_dump_thr_60*100:.2f}% -> dump flag")
            if _low_60c > 0 and not _micro_spike_pump:
                _rise_60 = (_lcp60 - _low_60c) / _low_60c
                _pump_thr_60 = 0.0250 * _sp if _is_grad_up_60 else 0.0100 * _sp
                if _rise_60 > _pump_thr_60:
                    _micro_spike_pump = True
                    logger.debug(f"{symbol}: 60c rise-from-low {_rise_60*100:.2f}% > {_pump_thr_60*100:.2f}% -> pump flag")

        # 5m / 1h extended pump-dump check (12 x 5m candles = 1 gio)
        # Bat pump/dump SPIKE xay ra trong 1h qua ma 30c/60c 1m miss
        # Ap dung logic phan biet spike vs trend tuong tu 30c/60c block
        if not df_scalp.empty and len(df_scalp) >= 12:
            _s_price  = df_scalp["close"].iloc[-1]
            _s12_low  = df_scalp["low"].iloc[-12:].min()
            _s12_high = df_scalp["high"].iloc[-12:].max()
            # Dung live price cho check 5m/1h (reuse tu block 30c); fallback ve _range_price (live hoac 15m close)
            _lcp_5m = _live_check_price if _live_check_price > 0 else _range_price
            # Phan biet spike vs gradual trend qua 12 nen 5m (1 gio)
            _s12_closes = df_scalp["close"].iloc[-12:].values
            _s12_opens  = df_scalp["open"].iloc[-12:].values
            _n_green_5m = sum(1 for i in range(12) if _s12_closes[i] > _s12_opens[i])
            _n_red_5m   = sum(1 for i in range(12) if _s12_closes[i] < _s12_opens[i])
            _is_grad_up_5m   = _n_green_5m >= 8  # >= 67% nen xanh = uptrend ro rang (8+8>12 -> not both true)
            _is_grad_down_5m = _n_red_5m   >= 8  # >= 67% nen do  = downtrend ro rang
            if _s12_low > 0 and not _micro_spike_pump:
                _rise_1h = (_lcp_5m - _s12_low) / _s12_low
                # Spike threshold: 1.5% (alt) / 0.75% (largecap)
                # Gradual uptrend: raise to 3.0% (alt) / 1.5% (largecap) — cho phep trend-following LONG
                _pump_thr_5m = 0.0300 * _sp if _is_grad_up_5m else 0.0150 * _sp
                if _rise_1h > _pump_thr_5m:
                    _micro_spike_pump = True
                    logger.debug(f"{symbol}: 5m 1h rise-from-low {_rise_1h*100:.2f}% > {_pump_thr_5m*100:.2f}% -> pump flag")
            if _s12_high > 0 and not _micro_spike_dump:
                _drop_1h = (_s12_high - _lcp_5m) / _s12_high
                _dump_thr_5m = 0.0300 * _sp if _is_grad_down_5m else 0.0150 * _sp
                if _drop_1h > _dump_thr_5m:
                    _micro_spike_dump = True
                    logger.debug(f"{symbol}: 5m 1h drop-from-high {_drop_1h*100:.2f}% > {_dump_thr_5m*100:.2f}% -> dump flag")

        # 1h macro trend va 4h macro trend — can truoc BREAKOUT de tranh NameError
        macro_trend = self._trend_direction(df_trend)
        macro_4h    = self._trend_direction(df_macro)

        # BREAKOUT: chay cho tat ca scan_list
        if df_micro is not None and not df_micro.empty and len(df_micro) >= 30:
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
                # Yeu cau it nhat 1 TF xac nhan trend — tranh breakout trong double-sideways
                bo_trend_ok = (
                    (bo_sig.direction == 1  and (macro_trend + macro_4h) >= 1) or
                    (bo_sig.direction == -1 and (macro_trend + macro_4h) <= -1)
                )
                # 2h 1m range block cho BREAKOUT — tranh short o day / long o dinh 2h
                bo_m2h_ok = not (_m2h_block_short and bo_sig.direction == -1) and \
                            not (_m2h_block_long  and bo_sig.direction == 1)
                if bo_ok and micro_ok and not is_spike and post_spike_ok and micro_spike_ok and bo_btc_ok and bo_h1_ok and bo_24h_ok and bo_trend_ok and bo_m2h_ok:
                    # BREAKOUT phai qua range check — tranh long o dinh / short o day
                    if not self._micro_entry_analysis(df_micro, bo_sig.direction):
                        logger.debug(f"{symbol}: BREAKOUT skip — range/micro_entry block")
                    else:
                        bo_sig.symbol    = symbol
                        bo_sig.consensus = 1
                        logger.info(
                            f"{symbol} [BREAKOUT TOP10] -> "
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

        n_15m_valid = 0  # so strategies co signal hop le tren 15m
        for strategy in ALL_STRATEGIES:
            try:
                sig = strategy.generate_signal(df_signal, df_trend, df_macro)
                # Dem signal 15m hop le
                if sig.direction != 0 and sig.strength >= config.MIN_SIGNAL_STRENGTH:
                    n_15m_valid += 1
                # Scalp fallback: chi thu 5m neu 15m chet VA thi truong da co it nhat 1 signal 15m hop le
                # Tranh pure 5m-only consensus trong thi truong sideways chet 15m
                # Skip VWAP (window 96x15m=24h, tren 5m cho ra 8h — sai)
                if sig.direction == 0 and n_15m_valid >= 1 and len(df_scalp) >= 50 and strategy.name != "vwap_volume":
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
                    # Yeu cau it nhat 1 TF (1h hoac 4h) xac nhan trend — tranh trade trong double-sideways
                    # sum >= 1: it nhat 1 trong 2 TF la uptrend -> long ok
                    # sum <= -1: it nhat 1 trong 2 TF la downtrend -> short ok
                    # sum = 0 (0+0 hoac 1+(-1) conflict): block het — khong trade MOMENTUM
                    long_ok  = (macro_trend + macro_4h) >= 1
                    short_ok = (macro_trend + macro_4h) <= -1
                    # 5m alignment pre-filter: khong dem signal khi 5m nguoc chieu (ALTCOIN ONLY)
                    # BTC/ETH (largecap): 5m corrections trong 1h trend la BINH THUONG (buy dip / sell bounce)
                    # -> khong apply cho largecap, dung 1m micro check (micro_up/down + EMA9/21) thay the
                    # Altcoin: 5m bounce trong 1h downtrend = timing xau cho SHORT -> bo qua
                    if _is_largecap:
                        scalp_allows_short = True
                        scalp_allows_long  = True
                    else:
                        scalp_allows_short = scalp_trend != 1   # 5m khong bullish
                        scalp_allows_long  = scalp_trend != -1  # 5m khong bearish
                    if sig.direction == 1 and long_ok and scalp_allows_long:
                        long_signals.append(sig)
                    elif sig.direction == -1 and short_ok and scalp_allows_short:
                        short_signals.append(sig)
            except Exception:
                continue

        # REVERSAL trade: RSI cuc doan + 2 nen 15m + 1m micro xac nhan dao chieu + >= MIN_CONSENSUS
        if is_reversal and reversal_dir != 0:
            # 5m 1h range position: reversal long chi hop le khi price van o BOTTOM 60% cua 1h range
            # Tranh "catch dead cat bounce" khi price da phuc hoi nhieu tu day 1h
            # Tuong tu: reversal short chi hop le khi price o TOP 60% cua 1h range
            _rev_1h_blocked = False
            if not df_scalp.empty and len(df_scalp) >= 12:
                _s12_hi = df_scalp["high"].iloc[-12:].max()
                _s12_lo = df_scalp["low"].iloc[-12:].min()
                _s12_rng = _s12_hi - _s12_lo
                if _s12_rng > 0:
                    _s12_pos = (_range_price - _s12_lo) / _s12_rng
                    if reversal_dir == 1 and _s12_pos > 0.60:
                        _rev_1h_blocked = True
                        logger.debug(f"{symbol}: reversal LONG blocked — 1h range_pos={_s12_pos:.2f} > 0.60 (not near bottom)")
                    elif reversal_dir == -1 and _s12_pos < 0.40:
                        _rev_1h_blocked = True
                        logger.debug(f"{symbol}: reversal SHORT blocked — 1h range_pos={_s12_pos:.2f} < 0.40 (not near top)")

            # 30c 1m range: check gia co phai da bounce/dump TRUOC KHI entry hay chua
            # SNDKUSDT pattern: RSI < 35 (chua recover) nhung price da bounce 89% tu day 30c (1513->1535)
            # AKEUSDT 19:24 pattern: dump 0.0009030 -> bounce len 0.0009631 = 69.6% of 30c range
            #   threshold 0.70 miss (69.6% < 70%) -> vao LONG o gan dinh bounce -> price dao chieu -> SL hit
            # Giam tu 0.70 -> 0.65: them dem buffer de bat cac truong hop bounce 65-70%
            # Tuong tu: giam SHORT threshold tu 0.30 -> 0.35
            if not _rev_1h_blocked and not df_micro.empty and len(df_micro) >= 30:
                _r30_hi = df_micro["high"].iloc[-30:].max()
                _r30_lo = df_micro["low"].iloc[-30:].min()
                _r30_rng = _r30_hi - _r30_lo
                if _r30_rng > 0:
                    _r30_pos = (_range_price - _r30_lo) / _r30_rng
                    if reversal_dir == 1 and _r30_pos > 0.65:
                        _rev_1h_blocked = True
                        logger.debug(
                            f"{symbol}: reversal LONG blocked — 30c 1m range_pos={_r30_pos:.2f} > 0.65 "
                            f"(price already bounced, reversal stale)"
                        )
                    elif reversal_dir == -1 and _r30_pos < 0.35:
                        _rev_1h_blocked = True
                        logger.debug(
                            f"{symbol}: reversal SHORT blocked — 30c 1m range_pos={_r30_pos:.2f} < 0.35 "
                            f"(price already dumped, reversal stale)"
                        )

            # 2h extreme block cho REVERSAL: tranh reversal LONG o dinh 2h / SHORT o day 2h
            # SKHYUSDT pattern: RSI < 35 nhung price da o top 85% of 2h range -> bad long reversal
            # MUUSDT pattern: RSI > 65 nhung price da o bottom 15% of 2h range -> bad short reversal
            # Nguong 85%/15% ketat hon momentum (80%/20%) vi reversal can price THUC SU o cuc doan chinh xac
            if not _rev_1h_blocked:
                if reversal_dir == 1 and _m2h_pos > 0.85:
                    _rev_1h_blocked = True
                    logger.debug(
                        f"{symbol}: reversal LONG blocked — 2h range_pos={_m2h_pos:.2f} > 0.85 "
                        f"(price at 2h top, dangerous reversal long)"
                    )
                elif reversal_dir == -1 and _m2h_pos < 0.15:
                    _rev_1h_blocked = True
                    logger.debug(
                        f"{symbol}: reversal SHORT blocked — 2h range_pos={_m2h_pos:.2f} < 0.15 "
                        f"(price at 2h bottom, dangerous reversal short)"
                    )

            # Direction-aware spike: long sau pump spike va short sau dump spike deu nguy hiem
            reversal_spike_blocked = (
                (reversal_dir == 1  and (_micro_spike_pump or _rev_1h_blocked)) or
                (reversal_dir == -1 and (_micro_spike_dump or _rev_1h_blocked))
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
                # Deep trend guard: neu ca 1h VA 4h deu oppose reversal direction -> +1 consensus
                reversal_deep_opposed = (
                    (reversal_dir == 1  and macro_trend == -1 and macro_4h == -1) or
                    (reversal_dir == -1 and macro_trend ==  1 and macro_4h ==  1)
                )
                reversal_min = reversal_base + (1 if reversal_deep_opposed else 0)
                if reversal_deep_opposed:
                    logger.debug(
                        f"{symbol}: reversal deep-trend guard +1 consensus "
                        f"(1h={'UP' if macro_trend==1 else 'DOWN'}, "
                        f"4h={'UP' if macro_4h==1 else 'DOWN'}, "
                        f"reversal={'LONG' if reversal_dir==1 else 'SHORT'}) "
                        f"-> need {reversal_min}/{len(ALL_STRATEGIES)}"
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

        # RE-APPLY MACRO FILTER sau reversal path (tranh signal leak)
        # Khi is_reversal=True, signals duoc collect KHONG co macro filter (de bat counter-trend)
        # Neu reversal khong du consensus -> phai loc lai truoc khi MOMENTUM path chay
        # Tranh truong hop: BTC bearish + reversal fail -> MOMENTUM van long voi signals chua filter
        _long_ok_macro  = (macro_trend + macro_4h) >= 1
        _short_ok_macro = (macro_trend + macro_4h) <= -1
        if is_reversal:
            long_signals  = [s for s in long_signals  if _long_ok_macro]
            short_signals = [s for s in short_signals if _short_ok_macro]

        # RSI EXTREME GUARD: neu RSI 15m oversold (< 35) thi xoa short signals o MOMENTUM path
        # Reversal path da xu ly o tren; neu reversal khong du consensus thi KHONG duoc short them vao oversold
        # Tuong tu: RSI > 65 xoa long signals (khong long vao overbought)
        if rsi_now < 35:
            short_signals = []
        elif rsi_now > 65:
            long_signals = []

        # BTC GLOBAL TREND FILTER — HARD BLOCK khi ca 1h VA 4h BTC cung chieu
        # Neu BTC 1h+4h BULLISH -> xoa het SHORT signals (tat ca coin, ke ca priority SOL/ETH)
        # Neu BTC 1h+4h BEARISH -> xoa het LONG signals
        # Ngoai le: REVERSAL signal (RSI cuc doan) — reversal co the di nguoc BTC
        # Ngoai le: BTCUSDT chinh no — tu xu ly theo trend chinh no
        # Day la nguyen nhan chinh khien bot short SOL/WLD/ZEC/ADA khi BTC dang pump
        btc_trend    = self.btc_trend
        btc_trend_4h = self.btc_trend_4h

        if symbol == "BTCUSDT":
            # BTC: chi SHORT khi CA 1h VA 4h deu bear (tranh short dip tam thoi trong uptrend)
            # Chi LONG khi CA 1h VA 4h deu bull
            # BTC/ETH co xu huong V-shape bounce sau dip ngan: chi 1h bear la khong du de short
            # Dung separate if (khong elif) de ca 2 co the true dong thoi (conflict -> no trade)
            if btc_trend == 1 or btc_trend_4h == 1:
                short_signals = []
                logger.debug("BTCUSDT: clear SHORT — BTC 1h or 4h UP (V-shape bounce risk)")
            if btc_trend == -1 or btc_trend_4h == -1:
                long_signals = []
                logger.debug("BTCUSDT: clear LONG — BTC 1h or 4h DOWN")
        elif symbol == "ETHUSDT":
            # ETH: tuong tu BTC — chi SHORT khi ca 1h VA 4h ETH deu bear
            # macro_trend = ETH 1h, macro_4h = ETH 4h
            if macro_trend == 1 or macro_4h == 1:
                short_signals = []
                logger.debug("ETHUSDT: clear SHORT — ETH 1h or 4h UP (V-shape bounce risk)")
            if macro_trend == -1 or macro_4h == -1:
                long_signals = []
                logger.debug("ETHUSDT: clear LONG — ETH 1h or 4h DOWN")
        else:
            # Altcoin: phan tich BTC alignment de quyet dinh hard/soft block
            btc_strongly_bull = (btc_trend == 1  and btc_trend_4h == 1)
            btc_strongly_bear = (btc_trend == -1 and btc_trend_4h == -1)

            # Coin co xu huong doc lap nguoc BTC (ca 1h VA 4h cua chinh coin do)
            # Vi du: BTC bull nhung coin rieng dang bearish 1h+4h -> co the cho phep short
            # Yeu cau them consensus cao hon (xu ly o phan consensus ben duoi)
            coin_independently_bear = (macro_trend == -1 and macro_4h == -1)
            coin_independently_bull = (macro_trend ==  1 and macro_4h ==  1)

            # Hard block: BTC strongly opposes AND coin khong co xu huong doc lap nguoc lai
            # Neu coin co xu huong doc lap -> cho phep nhung se cap cao consensus
            if btc_strongly_bull and not is_reversal and not coin_independently_bear:
                short_signals = []
                logger.debug(f"{symbol}: BTC 1h+4h BULLISH, coin not independently bearish -> block SHORT")
            if btc_strongly_bear and not is_reversal and not coin_independently_bull:
                long_signals = []
                logger.debug(f"{symbol}: BTC 1h+4h BEARISH, coin not independently bullish -> block LONG")

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

        both_sideways = (macro_trend == 0 and macro_4h == 0)  # ca 2 TF sideways = thi truong ranging

        if is_priority:
            # Priority (top10): base = MIN_CONSENSUS = 5
            # BTC cung chieu (bonus) -> giam 1 -> 4
            # Coin diverge nguoc BTC -> tang 2 -> 7
            # both_sideways (+1): ca 1h VA 4h sideways -> thi truong ranging, can them xac nhan
            base = config.MIN_CONSENSUS
            extra = (1 if both_sideways else 0)
            btc_long_bonus  = 1 if btc_strongly_bull else 0
            btc_short_bonus = 1 if btc_strongly_bear else 0
            diverge_long_penalty  = 2 if (btc_strongly_bear and coin_independently_bull)  else 0
            diverge_short_penalty = 2 if (btc_strongly_bull and coin_independently_bear) else 0
            required_long  = max(2, min(7, base + extra - btc_long_bonus  + diverge_long_penalty))
            required_short = max(2, min(7, base + extra - btc_short_bonus + diverge_short_penalty))
        else:
            # Non-priority: base = MIN_CONSENSUS_TRENDING = 5
            base = config.MIN_CONSENSUS_TRENDING
            extra = (1 if sideways_1h else 0)
            btc_long_bonus  = 1 if btc_strongly_bull else 0
            btc_short_bonus = 1 if btc_strongly_bear else 0
            diverge_long_penalty  = 2 if (btc_strongly_bear and coin_independently_bull)  else 0
            diverge_short_penalty = 2 if (btc_strongly_bull and coin_independently_bear) else 0
            required_long  = max(2, min(7, base + extra - btc_long_bonus  + diverge_long_penalty + (1 if btc_opposes_long  else 0)))
            required_short = max(2, min(7, base + extra - btc_short_bonus + diverge_short_penalty + (1 if btc_opposes_short else 0)))

        # TOP10 PRIORITY: 2 trong 2 Tier-1 strategy (supertrend + vwap_volume) dong thuan -> trade
        # Tier-1 bypass: KHONG bi chan boi BTC filter — top10 coin lon co momentum rieng
        # Tier1 bypass: 2/2 strategies dong thuan, khong bi chan boi bat ky extra filter nao
        TIER1 = {"supertrend", "vwap_volume"}
        tier1_long  = sum(1 for s in long_signals  if s.strategy_name in TIER1)
        tier1_short = sum(1 for s in short_signals if s.strategy_name in TIER1)

        # Tier1 bypass: chi khi ca 2 Tier1 cung chieu (2/2) va KHONG conflict
        # Neu conflict (1 long + 1 short): KHONG skip toan bo — van cho consensus check chay
        # [FIX] Tier1 bypass phai ton trong macro_4h alignment — tranh bypass trong reversal mode
        # khi signals vao tu reversal branch (khong co macro check)
        tier1_bypass_long  = (is_priority and tier1_long >= 2 and tier1_short == 0
                              and (macro_trend + macro_4h) >= 1  # it nhat 1 timeframe xac nhan uptrend
                              and not btc_strongly_bear)
        tier1_bypass_short = (is_priority and tier1_short >= 2 and tier1_long == 0
                              and (macro_trend + macro_4h) <= -1  # it nhat 1 timeframe xac nhan downtrend
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

        # BTC/ETH CORRELATION BLOCK: block neu pair kia da co position CUNG CHIEU
        # BTC va ETH correlated manh -> ca 2 cung SHORT = double loss khi bounce
        # Cho phep nguoc chieu (BTC long + ETH short = hedging, khac strategy)
        if btc_eth_side_map and symbol in ("BTCUSDT", "ETHUSDT"):
            pair = "ETHUSDT" if symbol == "BTCUSDT" else "BTCUSDT"
            pair_side = btc_eth_side_map.get(pair, "")
            # side tu Bybit: "Buy" = Long, "Sell" = Short
            signal_side = "Buy" if best.direction == 1 else "Sell"
            if pair_side == signal_side:
                logger.info(
                    f"{symbol}: skip — BTC/ETH correlation: {pair} already {pair_side}, "
                    f"block {signal_side} to avoid double correlated exposure"
                )
                return False

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
        # Block long o top 75% / short o bottom 25% cua 20-candle 1h range
        if _h1_block_long and best.direction == 1:
            logger.debug(f"{symbol}: skip — price at 1h range top (>75%), block MOMENTUM LONG")
            return False
        if _h1_block_short and best.direction == -1:
            logger.debug(f"{symbol}: skip — price at 1h range bottom (<25%), block MOMENTUM SHORT")
            return False

        # 2h 1m range block: tranh SHORT khi gia o bottom 20% range 2h
        # ONDOUSDT/SKHYNIXUSDT/XAGUSDT: dump xay ra truoc do, bot vao SHORT o day -> bounce -> loss
        if _m2h_block_short and best.direction == -1:
            logger.debug(f"{symbol}: skip — price at 2h 1m range bottom (<20%), block MOMENTUM SHORT")
            return False
        if _m2h_block_long and best.direction == 1:
            logger.debug(f"{symbol}: skip — price at 2h 1m range top (>80%), block MOMENTUM LONG")
            return False

        # 30c 1m range position: block MOMENTUM khi gia o top/bottom 25% cua range 30 phut
        # AKEUSDT pattern: pump tu 0.0009201 len 0.0009818 = 77% of 30c range -> bad long entry
        # Exception: gradual trend (>= 60% candles same direction) -> cho phep trend-following
        # La lap phong thu thu 3 (sau spike flag va 2h range block) — catch edge cases slip qua
        if not df_micro.empty and len(df_micro) >= 30:
            _r30m_high = df_micro["high"].iloc[-30:].max()
            _r30m_low  = df_micro["low"].iloc[-30:].min()
            _r30m_rng  = _r30m_high - _r30m_low
            if _r30m_rng > 0:
                _r30m_pos = (_range_price - _r30m_low) / _r30m_rng
                if best.direction == 1 and _r30m_pos > 0.75 and not _is_gradual_uptrend:
                    logger.debug(
                        f"{symbol}: skip LONG — 30c range_pos={_r30m_pos:.2f} > 0.75 "
                        f"(not gradual uptrend, bad entry timing)"
                    )
                    return False
                if best.direction == -1 and _r30m_pos < 0.25 and not _is_gradual_downtrend:
                    logger.debug(
                        f"{symbol}: skip SHORT — 30c range_pos={_r30m_pos:.2f} < 0.25 "
                        f"(not gradual downtrend, bad entry timing)"
                    )
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
        # EMA9 < EMA21: 1m bearish alignment -> tranh long; EMA9 > EMA21: 1m bullish -> tranh short
        if len(df_micro) >= 21:
            _e9  = compute_ema(df_micro["close"], 9).iloc[-1]
            _e21 = compute_ema(df_micro["close"], 21).iloc[-1]
            if best.direction == 1 and _e9 < _e21:
                logger.debug(f"{symbol}: skip — 1m EMA9({_e9:.4f}) < EMA21({_e21:.4f}), bearish micro, block LONG")
                return False
            if best.direction == -1 and _e9 > _e21:
                logger.debug(f"{symbol}: skip — 1m EMA9({_e9:.4f}) > EMA21({_e21:.4f}), bullish micro, block SHORT")
                return False

        # 5m (scalp) trend alignment: hard block MOMENTUM ALTCOIN khi 5m nguoc chieu signal
        # BTC/ETH largecap: 5m correction la binh thuong trong 1h trend — la entry tot (buy dip)
        #   -> KHONG block largecap, de 1m micro check (micro_up/down + EMA9/21) xu ly
        # ETHUSDT 19:30 altcoin pattern: 1h bearish, 5m bounce -> SHORT timing xau -> SL hit
        # BREAKOUT da check scalp_trend rieng; REVERSAL khong block (5m bounce tai day la ok)
        if not _is_largecap:
            if best.direction == -1 and scalp_trend == 1:
                logger.debug(
                    f"{symbol}: skip SHORT — 5m BULLISH (scalp_trend=1) vs 1h DOWN, altcoin bounce, cho 5m roll over"
                )
                return False
            if best.direction == 1 and scalp_trend == -1:
                logger.debug(
                    f"{symbol}: skip LONG — 5m BEARISH (scalp_trend=-1) vs 1h UP, altcoin pullback, cho 5m recover"
                )
                return False

        # EMA50 pullback filter (15m): noi long len 4.0x ATR (tu 2.0x)
        # 2.0x ATR qua chat — trong trending market (ADX > 25), price co the gap EMA50 3-5x ATR
        # voi leverage thap hon (20x thay vi 100x), loss tu EMA50 extended trade nho hon
        # Chi block khi THUC SU qua extended: 4x ATR (khoang 2-3% cho most coins)
        if len(df_signal) >= 50:
            _ema50_15m = compute_ema(df_signal["close"], 50).iloc[-1]
            _atr_15m   = compute_atr(df_signal, config.ATR_PERIOD).iloc[-1]
            _p15 = df_signal["close"].iloc[-1]
            if _ema50_15m > 0 and _atr_15m > 0:
                _ema50_dist = _p15 - _ema50_15m  # + = above, - = below
                if best.direction == 1 and _ema50_dist > 4.0 * _atr_15m:
                    logger.debug(
                        f"{symbol}: skip LONG — price {_ema50_dist/_ema50_15m*100:.1f}% above 15m EMA50 "
                        f"({_ema50_dist/(_atr_15m+1e-9):.1f}x ATR, too extended)"
                    )
                    return False
                if best.direction == -1 and _ema50_dist < -4.0 * _atr_15m:
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
