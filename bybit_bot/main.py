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
        # Rotate batch cho rest coins (ngoai top20): moi tick quet 1 batch
        self._rest_batch_idx: int = 0
        REST_BATCH_SIZE = 20  # quet 20 coin/tick tu phan con lai

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
                logger.info(f"Symbols updated: {len(self.symbols)}, top 20: {self.symbols[:20]}")
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
        # btc_trend    = 15m trend direction (nhanh, bat flip som)
        # btc_trend_4h = 1h trend direction  (chac chan hon, xac nhan xu huong lon)
        try:
            df_btc_15m = self.client.get_klines("BTCUSDT", config.TIMEFRAMES["trend"], 100)
            df_btc_1h  = self.client.get_klines("BTCUSDT", config.TIMEFRAMES["macro"],  100)
            if not df_btc_15m.empty and len(df_btc_15m) >= 50:
                self.btc_trend = self._trend_direction(df_btc_15m)
            if not df_btc_1h.empty and len(df_btc_1h) >= 50:
                self.btc_trend_4h = self._trend_direction(df_btc_1h)
        except Exception:
            pass

        # =====================================================================
        # TEST TRADE — XOA SAU KHI XAC NHAN BOT PULL CODE DUOC
        # Force trade BTCUSDT theo btc_trend de kiem tra bot co chay code moi
        # =====================================================================
        _btc_in_pos = any(p["symbol"] == "BTCUSDT" for p in open_positions)
        if not _btc_in_pos and self.btc_trend != 0:
            try:
                from strategies.base import Signal
                _test_dir = self.btc_trend  # 1=Long, -1=Short theo 15m trend
                _df_btc15 = self.client.get_klines("BTCUSDT", config.TIMEFRAMES["trend"], 20)
                _btc_price = float(_df_btc15["close"].iloc[-1]) if not _df_btc15.empty else 0.0
                _btc_atr   = compute_atr(_df_btc15, 14).iloc[-1] if not _df_btc15.empty and len(_df_btc15) >= 14 else _btc_price * 0.005
                _test_sig = Signal(
                    direction=_test_dir,
                    strength=0.85,
                    strategy_name="TEST_FORCE",
                    entry_price=_btc_price,
                    atr=_btc_atr,
                    reason="TEST: forced BTC trade to verify bot pulls new code",
                    symbol="BTCUSDT",
                    consensus=3,
                )
                logger.warning(f"[TEST] Forcing BTCUSDT {'LONG' if _test_dir==1 else 'SHORT'} @ {_btc_price:.2f} — DELETE AFTER CONFIRM")
                self.executor.execute_signal("BTCUSDT", _test_sig, equity, open_positions, is_priority=True)
            except Exception as _e:
                logger.error(f"[TEST] Force trade failed: {_e}")
        # =====================================================================
        # END TEST TRADE
        # =====================================================================

        # Hard cap: khong mo them lenh neu da dat MAX_OPEN_POSITIONS
        if len(open_positions) >= config.MAX_OPEN_POSITIONS:
            logger.info(
                f"[TICK] Max positions ({config.MAX_OPEN_POSITIONS}) reached — skip new entries"
            )
            return

        # Top 20 (trending cao nhat): phan tich moi tick
        # Phan con lai: rotate theo batch 20 coin/tick (tranh miss nhung van nhanh)
        REST_BATCH_SIZE = 20
        top20 = self.symbols[:20]
        rest  = self.symbols[20:]
        if rest:
            _start = self._rest_batch_idx % len(rest)
            _end   = _start + REST_BATCH_SIZE
            rest_batch = (rest + rest)[_start:_end]  # wrap around
            self._rest_batch_idx = (_start + REST_BATCH_SIZE) % len(rest)
        else:
            rest_batch = []
        scan_list    = top20 + rest_batch
        priority_set = set(top20)

        logger.info(
            f"[TICK] Scan {len(scan_list)} coins (top20 + {len(rest_batch)} rest batch) | "
            f"BTC_1h={'UP' if self.btc_trend==1 else 'DOWN' if self.btc_trend==-1 else 'SIDE'} "
            f"BTC_4h={'UP' if self.btc_trend_4h==1 else 'DOWN' if self.btc_trend_4h==-1 else 'SIDE'}"
        )

        # BTC/ETH direction map: de check correlation truoc khi mo lenh moi
        # {symbol: "Long" | "Short"} cho cac position dang mo
        _pos_side_map = {p["symbol"]: p.get("side", "") for p in open_positions}

        for symbol in scan_list:

            if symbol in pos_symbols:
                continue

            is_priority = symbol in priority_set
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

        threshold = 2
        ok = score >= threshold
        if not ok:
            logger.debug(f"micro_entry_analysis: dir={direction} score={score}/{threshold} n={n} -> skip")
        return ok

    def _process_symbol(self, symbol: str, equity: float, open_positions: list[dict], is_priority: bool = False, btc_eth_side_map: dict | None = None) -> bool:
        """Phan tich symbol, chay tat ca filter va strategy, tra True neu da trade."""
        # Init gradual trend flags — se duoc tinh chinh xac sau khi co df_micro
        _is_gradual_uptrend   = False
        _is_gradual_downtrend = False

        # Timeframe moi: signal=1m (paginated, 2000c), scalp=5m, trend=15m, macro=1h
        # df_micro = df_signal (1m) — dung lai ten cu de khong phai doi het code ben duoi
        df_signal = self.client.get_klines_paginated(symbol, config.TIMEFRAMES["signal"], config.CANDLE_LIMIT_SIGNAL)
        df_scalp  = self.client.get_klines(symbol, config.TIMEFRAMES["scalp"],  config.CANDLE_LIMIT_SCALP)
        df_trend  = self.client.get_klines(symbol, config.TIMEFRAMES["trend"],  config.CANDLE_LIMIT_TREND)
        df_macro  = self.client.get_klines(symbol, config.TIMEFRAMES["macro"],  config.CANDLE_LIMIT_MACRO)
        df_micro  = df_signal   # alias: 1m data dung cho micro analysis (khong fetch them)

        if df_signal.empty or len(df_signal) < 50:
            return False

        # Volatility bucket cho spike/pump-dump thresholds:
        # largecap  (BTC/ETH):              _sp = 0.50 — ATR% ~0.3-0.4%, threshold rat thap
        # midcap    (SOL/XRP/HYPE/BNB/..): _sp = 0.75 — volatility trung binh, giua 2 nhom
        # altcoin   (phan con lai):          _sp = 1.00 — volatility cao nhat
        _LARGECAP = {"BTCUSDT", "ETHUSDT"}
        _MIDCAP   = {"SOLUSDT", "XRPUSDT", "HYPEUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT", "TRXUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT"}
        _is_largecap = symbol in _LARGECAP
        _sp = 0.50 if symbol in _LARGECAP else (0.75 if symbol in _MIDCAP else 1.0)

        # ATR filter: bo qua symbol bien dong qua nho
        # Large-cap: 0.2% (BTC ATR% ~0.3-0.4%, ETH ~0.3%), altcoin: 0.4%
        atr   = compute_atr(df_signal).iloc[-1]
        price = df_signal["close"].iloc[-1]
        _min_atr_pct = config.MIN_ATR_PCT * _sp
        if price > 0 and atr / price < _min_atr_pct:
            return False

        # Dung 15m ADX thay vi 1m ADX: 1m ADX qua nhay, thuong < 18 vao buoi sang/
        # Asian session du trend 15m+1h co that. 15m ADX on dinh hon, phan anh trend chuan xac.
        min_adx = config.MIN_ADX
        adx = compute_adx(df_trend).iloc[-1] if not df_trend.empty and len(df_trend) >= 20 else float("nan")
        if math.isnan(adx) or adx < min_adx:
            logger.debug(f"{symbol}: skip — 15m ADX={adx:.1f} < {min_adx} (sideway)")
            return False

        # 24h directional move filter: tranh chase sau khi coin da pump/dump > 20% trong 24h
        # Dung df_trend (15m): 96 nen x 15m = 24h (chinh xac, khong bi anh huong boi doi sang 1m)
        # df_signal la 1m: 96 nen 1m = 96 phut, khong dung cho 24h check
        _block_long_24h  = False
        _block_short_24h = False
        _change_24h = 0.0
        if len(df_trend) >= 96:
            _ref_24h = df_trend["close"].iloc[-96]
            if _ref_24h > 0:
                _change_24h = (df_trend["close"].iloc[-1] - _ref_24h) / _ref_24h * 100
                if abs(_change_24h) > 30:
                    logger.info(f"{symbol}: 24h change={_change_24h:.1f}% > 30% -> HARD SKIP (extreme move)")
                    return False
                if _change_24h > 20:
                    _block_long_24h = True
                    logger.debug(f"{symbol}: 24h change=+{_change_24h:.1f}% -> block LONG (pump exhausted)")
                elif _change_24h < -20:
                    _block_short_24h = True
                    logger.debug(f"{symbol}: 24h change={_change_24h:.1f}% -> block SHORT (dump exhausted)")

        # RSI cho reversal detection — dung df_trend (15m) thay df_signal (1m)
        # RSI 1m qua nhay, oversold/overbought xuyen hien lien tuc trong noise binh thuong
        # RSI 15m on dinh hon, chi bat reversal that su
        rsi_now = compute_rsi(df_trend["close"]).iloc[-1] if not df_trend.empty and len(df_trend) >= 14 else 50.0

        # Lay live mark price mot lan cho range position checks — tranh dung 15m close (stale up to 14m)
        # Tai day la diem dau tien co du context de goi API (sau spike filter da pass)
        # Reuse cho _live_check_price trong 30c block de tranh second API call
        _range_live_price = self.client.get_current_price(symbol)
        _range_price = _range_live_price if _range_live_price > 0 else price

        # Tinh macro_trend TRUOC range blocks — can de relax threshold cho priority coins trong trend
        macro_trend = self._trend_direction(df_trend)   # 15m direction
        macro_4h    = self._trend_direction(df_macro)   # 1h direction
        # ATR tu 15m cho SL/TP — 1m ATR qua nho, noise hit SL lien tuc
        _atr_for_sl = compute_atr(df_trend, config.ATR_PERIOD).iloc[-1] if not df_trend.empty and len(df_trend) >= config.ATR_PERIOD else 0.0

        # 15m range position: block long o TOP / short o BOTTOM cua 20-candle 15m range
        # Priority coins trong confirmed trend duoc phep entry o top/bottom hon (85/15 thay vi 75/25)
        # Vi trong uptrend BTC, price hop le o top 75-90% cua range 5h
        _h1_block_long  = False
        _h1_block_short = False
        if not df_trend.empty and len(df_trend) >= 20:
            h1_high = df_trend["high"].iloc[-20:].max()
            h1_low  = df_trend["low"].iloc[-20:].min()
            h1_rng  = h1_high - h1_low
            if h1_rng > 0:
                h1_pos = (_range_price - h1_low) / h1_rng
                _h1_long_thresh  = 0.85 if (is_priority and macro_trend >= 1) else 0.75
                _h1_short_thresh = 0.15 if (is_priority and macro_trend <= -1) else 0.25
                if h1_pos > _h1_long_thresh:
                    _h1_block_long = True
                    logger.debug(f"{symbol}: 15m range_pos={h1_pos:.2f} > {_h1_long_thresh} -> block LONG (15m top)")
                elif h1_pos < _h1_short_thresh:
                    _h1_block_short = True
                    logger.debug(f"{symbol}: 15m range_pos={h1_pos:.2f} < {_h1_short_thresh} -> block SHORT (15m bottom)")

        # 2h 1m range: block SHORT khi gia o bottom 20% cua range 120 nen 1m (2 gio)
        # Block LONG khi o top 80%
        # Priority coins trong confirmed trend: relax den 88%/12%
        _m2h_block_long  = False
        _m2h_block_short = False
        _m2h_pos = 0.5  # default mid-range (used also in reversal extreme block below)
        if not df_micro.empty and len(df_micro) >= 120:
            _m2h_high = df_micro["high"].iloc[-120:].max()
            _m2h_low  = df_micro["low"].iloc[-120:].min()
            _m2h_rng  = _m2h_high - _m2h_low
            if _m2h_rng > 0:
                _m2h_pos = (_range_price - _m2h_low) / _m2h_rng
                _m2h_top_thresh = 0.88 if (is_priority and macro_trend >= 1) else 0.80
                _m2h_bot_thresh = 0.12 if (is_priority and macro_trend <= -1) else 0.20
                if _m2h_pos < _m2h_bot_thresh:
                    _m2h_block_short = True
                    logger.debug(f"{symbol}: 2h 1m range_pos={_m2h_pos:.2f} < {_m2h_bot_thresh} -> block SHORT (2h bottom)")
                elif _m2h_pos > _m2h_top_thresh:
                    _m2h_block_long = True
                    logger.debug(f"{symbol}: 2h 1m range_pos={_m2h_pos:.2f} > {_m2h_top_thresh} -> block LONG (2h top)")

        # Momentum confirmation (15m): it nhat 2/3 nen gan nhat cung chieu voi signal
        # 2-consecutive (c1 AND c2) qua chat: breakout candle c1=green, c2=red (consolidation) bi block
        # 2-of-3 cho phep mot nen pullback trong xu huong — van du xac nhan momentum
        opens  = df_signal["open"]
        closes = df_signal["close"]
        _3c_bodies = [closes.iloc[-i] - opens.iloc[-i] for i in range(1, 4)]
        _n_bull3 = sum(1 for b in _3c_bodies if b > 0)
        _n_bear3 = sum(1 for b in _3c_bodies if b < 0)
        short_term_up   = _n_bull3 >= 2   # 2 trong 3 nen xanh
        short_term_down = _n_bear3 >= 2   # 2 trong 3 nen do

        # Spike filter tren 1m signal: nen lon bat thuong > 2x ATR (ATR 1m)
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
        _live_check_price = 0.0
        _micro_price = df_micro["close"].iloc[-1] if not df_micro.empty else 0.0

        # Tinh gradual trend flags TRUOC spike gate — can cho ca hai nhanh (spike va non-spike)
        # Bug: neu tinh ben trong spike gate, khi spike flag da set truoc, block skip -> flag sai = False
        if not df_micro.empty and len(df_micro) >= 30:
            _30c_c = df_micro["close"].iloc[-30:].values
            _30c_o = df_micro["open"].iloc[-30:].values
            _n_grn = sum(1 for i in range(30) if _30c_c[i] > _30c_o[i])
            _n_red = sum(1 for i in range(30) if _30c_c[i] < _30c_o[i])
            _is_gradual_uptrend   = _n_grn >= 18
            _is_gradual_downtrend = _n_red >= 18

        if not df_micro.empty and len(df_micro) >= 15:
            _micro_atr    = compute_atr(df_micro).iloc[-1]
            _micro_bodies = (df_micro["close"].iloc[-15:].values - df_micro["open"].iloc[-15:].values)
            # Nguong 1.5x ATR (giam tu 2.0x): bat pump/dump vua duoi 2x ATR trong 15 nen
            _micro_spike_dump = any(b < -_micro_atr * 1.5 for b in _micro_bodies)
            _micro_spike_pump = any(b >  _micro_atr * 1.5 for b in _micro_bodies)
            # Current forming candle: block neu body 1m hien tai >= 0.4% (mid-pump/dump entry)
            # Bat cac truong hop vao lenh DANG GIUA pump — candle chua dong nen 2x ATR chua dat
            # SOXL/NEAR/HYPE/SNDK: gia tang 0.7-1.8% trong candle dang hinh thanh -> block LONG
            # Forming candle body check da xoa: body % block entry dung luc momentum manh nhat
            # 1.5x ATR spike check (ben tren) du de bat candle bat thuong that su
            if _micro_spike_dump and _micro_spike_pump:
                logger.debug(f"{symbol}: 1m spike ca 2 chieu (ranging) — flags cleared, let consensus decide")
                _micro_spike_dump = False
                _micro_spike_pump = False
            # Cumulative net move: 15-candle lookback, 0.8% threshold
            # Bat ca dump bat dau tu 15 phut truoc (truoc chi bat 10 phut)
            _close_15_ago = df_micro["close"].iloc[-15]
            _micro_price  = df_micro["close"].iloc[-1]
            if _close_15_ago > 0:
                _net_move = (_micro_price - _close_15_ago) / _close_15_ago
                # scalp_trend bypass: neu 5m xac nhan cung chieu -> la trend, khong phai spike
                if _net_move < -0.008 * _sp and not _micro_spike_dump and scalp_trend != -1:
                    _micro_spike_dump = True
                    logger.debug(f"{symbol}: cumulative net dump {_net_move*100:.1f}% in 15 candles -> dump flag")
                elif _net_move > 0.008 * _sp and not _micro_spike_pump and scalp_trend != 1:
                    _micro_spike_pump = True
                    logger.debug(f"{symbol}: cumulative net pump {_net_move*100:.1f}% in 15 candles -> pump flag")

            # RSI 1m: chi block khi CUC DOAN va 5m KHONG xac nhan trend cung chieu
            # Trong downtrend BTC: RSI 1m < 20 la BINH THUONG (trend manh), khong phai exhaustion
            # Chi set dump flag khi scalp_trend KHONG phai bearish (tuc la dump nay la spike, khong phai trend)
            if len(df_micro) >= 14:
                _micro_rsi = compute_rsi(df_micro["close"]).iloc[-1]
                if _micro_rsi < 20 and not _micro_spike_pump and scalp_trend != -1:
                    _micro_spike_dump = True
                    logger.debug(f"{symbol}: 1m RSI={_micro_rsi:.1f} extreme oversold (5m not bearish) -> dump flag")
                elif _micro_rsi > 80 and not _micro_spike_dump and scalp_trend != 1:
                    _micro_spike_pump = True
                    logger.debug(f"{symbol}: 1m RSI={_micro_rsi:.1f} extreme overbought (5m not bullish) -> pump flag")

            # Consecutive candles block: largecap 10 nen (6 green 1m candles binh thuong trong BTC uptrend)
            # Altcoin: 8 nen lien tiep = exhaustion / dao chieu
            # scalp_trend bypass: neu 5m xac nhan cung chieu -> la trend that, khong phai exhaustion
            _consec_n = 10 if _is_largecap else 8
            if len(df_micro) >= _consec_n:
                _micro_c = df_micro["close"].iloc[-_consec_n:].values
                _micro_o = df_micro["open"].iloc[-_consec_n:].values
                _all_green = all(_micro_c[i] > _micro_o[i] for i in range(_consec_n))
                _all_red   = all(_micro_c[i] < _micro_o[i] for i in range(_consec_n))
                if _all_green and scalp_trend != 1 and not _micro_spike_pump:
                    _micro_spike_pump = True
                    logger.debug(f"{symbol}: {_consec_n} consecutive green 1m candles -> pump flag (exhaustion)")
                if _all_red and scalp_trend != -1 and not _micro_spike_dump:
                    _micro_spike_dump = True
                    logger.debug(f"{symbol}: {_consec_n} consecutive red 1m candles -> dump flag (exhaustion)")

            # 30c net move: chi block khi TOAN BO 30 phut co move lon bat thuong
            # 0.6% da xoa vi qua nho — trong trend binh thuong 30c la 0.5-2%
            # Chi giu lai 15c cumulative (da check phia tren) va forming candle
            _live_check_price = _range_live_price if _range_live_price > 0 else _micro_price

            # Dual spike (ca pump va dump trong 30c) = ranging/choppy -> skip REVERSAL nhung KHONG block momentum
            # Macro trend + consensus se tu loc momentum trong ranging market
            if _micro_spike_dump and _micro_spike_pump:
                logger.debug(f"{symbol}: dual spike flag (ranging 1m) — skip reversal path only")

        # macro_trend / macro_4h / _atr_for_sl da tinh TRUOC range blocks (tren)

        # Strong trend flags — dung cho 30c range bypass va cac check sau
        _strong_bull = is_priority and macro_trend >= 1 and macro_4h >= 1
        _strong_bear = is_priority and macro_trend <= -1 and macro_4h <= -1

        # BREAKOUT: chay cho tat ca scan_list — su dung 1m signal data
        if df_signal is not None and not df_signal.empty and len(df_signal) >= 30:
            bo_sig = BREAKOUT_STRATEGY.generate_signal(df_signal, df_scalp, df_trend)
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
                # Chi can 1h xac nhan trend la du cho breakout
                bo_trend_ok = (
                    (bo_sig.direction == 1  and macro_trend >= 1) or
                    (bo_sig.direction == -1 and macro_trend <= -1) or
                    (bo_sig.direction == 1  and macro_4h >= 1) or
                    (bo_sig.direction == -1 and macro_4h <= -1)
                )
                # 2h 1m range block cho BREAKOUT — tranh short o day / long o dinh 2h
                bo_m2h_ok = not (_m2h_block_short and bo_sig.direction == -1) and \
                            not (_m2h_block_long  and bo_sig.direction == 1)
                if bo_ok and micro_ok and not is_spike and post_spike_ok and micro_spike_ok and bo_btc_ok and bo_h1_ok and bo_24h_ok and bo_trend_ok and bo_m2h_ok:
                    # BREAKOUT phai qua range check — tranh long o dinh / short o day
                    if not self._micro_entry_analysis(df_signal, bo_sig.direction):
                        logger.debug(f"{symbol}: BREAKOUT skip — range/micro_entry block")
                    else:
                        bo_sig.symbol    = symbol
                        bo_sig.consensus = 1
                        # ATR override: dung 15m ATR cho SL/TP (1m ATR qua nho)
                        if _atr_for_sl > 0:
                            bo_sig.atr = _atr_for_sl
                        logger.info(
                            f"{symbol} [BREAKOUT] -> "
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

        n_15m_valid = 0  # so strategies co signal hop le
        for strategy in ALL_STRATEGIES:
            try:
                # Strategies chay tren 15m (df_trend) — duoc thiet ke cho timeframe nay
                # 1m (df_signal) chi dung cho entry timing (spike, micro_trend, range checks)
                # Chay tren 1m: EMA/MACD/Ichimoku/sustained_trend rat hiem fire -> 0 consensus
                sig = strategy.generate_signal(df_trend, df_macro, df_macro)
                # Fallback: thu 5m neu 15m khong co signal (VWAP skip: window 5m sai)
                if sig.direction == 0 and len(df_scalp) >= 50 and strategy.name != "vwap_volume":
                    sig = strategy.generate_signal(df_scalp, df_trend, df_macro)
                if sig.direction != 0 and sig.strength >= config.MIN_SIGNAL_STRENGTH:
                    n_15m_valid += 1

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
                    # Chi can 1 trong 2 TF xac nhan — sum=0 (1h+4h conflict) van ok neu 1h confirm
                    long_ok  = macro_trend >= 1 or macro_4h >= 1
                    short_ok = macro_trend <= -1 or macro_4h <= -1
                    # Early trend entry: cho phep SHORT/LONG khi 1m + 5m da confirm du 15m chua flip
                    # BTC break down 06:00: 1m bearish + 5m bearish nhung 15m EMA chua cross -> miss het move
                    # Neu ca micro (1m) va scalp (5m) deu bearish va coin la priority -> cho short du macro chua -1
                    if is_priority and sig.direction == -1 and not short_ok:
                        if micro_down and scalp_trend == -1:
                            short_ok = True   # 1m+5m da confirm breakdown truoc 15m
                    if is_priority and sig.direction == 1 and not long_ok:
                        if micro_up and scalp_trend == 1:
                            long_ok = True    # 1m+5m da confirm breakout truoc 15m
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
                    # ATR override: dung 15m ATR cho SL/TP (1m ATR qua nho)
                    if _atr_for_sl > 0:
                        best.atr = _atr_for_sl
                    names = "+".join(s.strategy_name for s in signals)
                    rsi_label = f"RSI15m={rsi_now:.0f}({'OVERSOLD' if reversal_dir==1 else 'OVERBOUGHT'})"
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
        _long_ok_macro  = macro_trend >= 1 or macro_4h >= 1
        _short_ok_macro = macro_trend <= -1 or macro_4h <= -1
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
            # BTC: block SHORT chi khi CA 1h VA 4h deu bullish (BOTH)
            # Dung AND thay OR: cho phep short khi 1h da dao chieu du 4h chua flip
            # 4h EMA50 mat hang ngay moi flip -> AND cho phep bat early downtrend
            if btc_trend == 1 and btc_trend_4h == 1:
                short_signals = []
                logger.debug("BTCUSDT: clear SHORT — BTC 1h AND 4h both UP")
            if btc_trend == -1 and btc_trend_4h == -1:
                long_signals = []
                logger.debug("BTCUSDT: clear LONG — BTC 1h AND 4h both DOWN")
        elif symbol == "ETHUSDT":
            if macro_trend == 1 and macro_4h == 1:
                short_signals = []
                logger.debug("ETHUSDT: clear SHORT — ETH 1h AND 4h both UP")
            if macro_trend == -1 and macro_4h == -1:
                long_signals = []
                logger.debug("ETHUSDT: clear LONG — ETH 1h AND 4h both DOWN")
        else:
            # Altcoin: phan tich BTC alignment de quyet dinh hard/soft block
            btc_strongly_bull = (btc_trend == 1  and btc_trend_4h == 1)
            btc_strongly_bear = (btc_trend == -1 and btc_trend_4h == -1)

            # Coin co xu huong doc lap nguoc BTC (ca 1h VA 4h cua chinh coin do)
            # Vi du: BTC bull nhung coin rieng dang bearish 1h+4h -> co the cho phep short
            # Yeu cau them consensus cao hon (xu ly o phan consensus ben duoi)
            # OR: neu 1 trong 2 TF da xac nhan xu huong doc lap la du
            # Vi du: MNTUSDT co 15m bullish du 1h chua flip -> van cho phep LONG vs BTC bear
            coin_independently_bear = (macro_trend == -1 or macro_4h == -1)
            coin_independently_bull = (macro_trend ==  1 or macro_4h ==  1)

            # Hard block: BTC strongly opposes AND coin khong co xu huong doc lap
            # Chi block khi coin CUNG CHIEU voi BTC move (khong co divergence)
            # Cho phep coin co xu huong doc lap (coin_independently_bull/bear) di nguoc BTC
            if btc_strongly_bull and not is_reversal and not coin_independently_bear:
                short_signals = []
                logger.debug(f"{symbol}: BTC 1h+4h BULLISH, coin not independently bearish -> block SHORT")
            if btc_strongly_bear and not is_reversal and not coin_independently_bull:
                long_signals = []
                logger.debug(f"{symbol}: BTC 1h+4h BEARISH, coin not independently bullish -> block LONG")
            # Neu coin doc lap nguoc BTC: van cho phep nhung can consensus cao hon (xu ly o phan duoi)

        # BTC alignment flags cho consensus adjustment
        btc_strongly_bull = (btc_trend == 1  and btc_trend_4h == 1)   if symbol != "BTCUSDT" else False
        btc_strongly_bear = (btc_trend == -1 and btc_trend_4h == -1)  if symbol != "BTCUSDT" else False
        coin_independently_bear = (macro_trend == -1 or macro_4h == -1)
        coin_independently_bull = (macro_trend ==  1 or macro_4h ==  1)

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
            extra = 0
            btc_long_bonus  = 1 if btc_strongly_bull else 0
            btc_short_bonus = 1 if btc_strongly_bear else 0
            diverge_long_penalty  = 2 if (btc_strongly_bear and coin_independently_bull)  else 0
            diverge_short_penalty = 2 if (btc_strongly_bull and coin_independently_bear) else 0
            required_long  = max(2, min(7, base + extra - btc_long_bonus  + diverge_long_penalty))
            required_short = max(2, min(7, base + extra - btc_short_bonus + diverge_short_penalty))
        else:
            # Non-priority: base = MIN_CONSENSUS_TRENDING = 5
            base = config.MIN_CONSENSUS_TRENDING
            extra = 0
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
            logger.info(f"{symbol}: [TIER1] 2/2 Tier-1 LONG — bypass consensus")
        elif tier1_bypass_short:
            signals = short_signals
            logger.info(f"{symbol}: [TIER1] 2/2 Tier-1 SHORT — bypass consensus")
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

        # 30c range block da xoa: redundant — da co 15m range + 2h range blocks

        # 1m micro trend confirmation — tat ca coin (micro trend phai cung chieu hoac neutral)
        # Tranh trade khi 1m dang nguoc chieu hoan toan voi signal
        if best.direction == 1 and micro_down:
            logger.debug(f"{symbol}: skip — 1m micro trend BEARISH vs LONG signal")
            return False
        if best.direction == -1 and micro_up:
            logger.debug(f"{symbol}: skip — 1m micro trend BULLISH vs SHORT signal")
            return False

        # 1m EMA9/21 check da xoa: redundant — micro_trend da dung EMA lam 1 trong 5 factors

        # 5m hard block da xoa: "sell the bounce" la chien thuat tot trong downtrend
        # 5m bounce trong 1h downtrend = timing SHORT tot, khong phai xau
        # Giu lai reset cho micro_spike_dump khi 5m da xac nhan
        if not _is_largecap:
            if best.direction == -1 and scalp_trend == -1 and (macro_trend + macro_4h) <= -1:
                _micro_spike_dump = False

        # EMA50 pullback filter: dung df_trend (15m) — 1m EMA50 qua nhay, thuong bien dong qua lon
        # Chi block khi THUC SU qua extended: 4x ATR(15m)
        if not df_trend.empty and len(df_trend) >= 50:
            _ema50_15m = compute_ema(df_trend["close"], 50).iloc[-1]
            _atr_15m   = _atr_for_sl if _atr_for_sl > 0 else compute_atr(df_trend, config.ATR_PERIOD).iloc[-1]
            _p15 = df_trend["close"].iloc[-1]
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

        # micro_entry_analysis da xoa: redundant sau consensus 4/7 + range blocks + micro_trend check

        best.consensus = len(signals)
        best.symbol    = symbol
        # ATR override: dung 15m ATR cho SL/TP — 1m ATR qua nho (noise se hit SL lien tuc)
        if _atr_for_sl > 0:
            best.atr = _atr_for_sl
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
