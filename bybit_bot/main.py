"""
Bybit Futures Auto Trading Bot - Main Entry Point
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
from client import BybitClient, _sf
from executor import Executor
from risk_manager import RiskManager
from scanner import MarketScanner
from strategies import ALL_STRATEGIES, BREAKOUT_STRATEGY
from strategies.base import Signal, compute_ema, compute_atr, compute_rsi, compute_adx

setup_logging()
logger = logging.getLogger(__name__)


class TradingBot:
    def __init__(self):
        logger.info("="*60)
        logger.info("Bybit Auto Trading Bot starting...")
        # === VERSION BANNER - de XAC NHAN dang chay code MOI (khong phai code cu) ===
        # Neu ban KHONG thay dong nay khi khoi dong -> bot dang chay code CU, PHAI restart.
        logger.info(">>> TREND-GUARD v5 : high-conviction + exhaustion 30/70 + TP 12-25% <<<")
        logger.info(">>> Gates: TRUE-DIR(unanimity) + EXHAUSTION + IMM-momentum tren CA 3 path <<<")
        print(">>> [TREND-GUARD v5] high-conviction mode ACTIVE - long-top/short-bottom BLOCKED <<<", flush=True)
        logger.info(f"Mode: {'TESTNET' if config.TESTNET else 'MAINNET (LIVE)'}")
        logger.info(f"TP range: {config.TP_ROI_MIN*100:.0f}%-{config.TP_ROI_MAX*100:.0f}% ROI | SL={config.SL_TP_RATIO:.0f}xTP")
        logger.info(f"Scan budget per tick: TOP{config.TOP20_COUNT}={config.SCAN_BUDGET_TOP20_SEC}s + REST={config.SCAN_BUDGET_REST_SEC}s")
        logger.info("Max positions: unlimited (limited by equity & market opportunity)")
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
        self.btc_trend: int = 0      # EMA(100/250) tren 1m ~ medium trend (~1.5h)
        self.btc_trend_4h: int = 0   # EMA(300/600) tren 1m ~ macro trend (~5h, ten giu nguyen de tranh refactor lon)
        self.btc_trend_fast: int = 0 # EMA(20/50) tren 1m ~ short-term trend (~20min) - bat bounce/dip BTC nhanh
        self.df_btc = None           # BTC 1m data cho correlation check
        # Daily loss guard
        self._equity_day_date: str = ""
        # Per-symbol cooldown: tranh re-analyze cung coin trong SYMBOL_COOLDOWN_SEC
        self._last_analyzed: dict[str, float] = {}

    # -- Main loop ------------------------------------------------------------

    def run(self):
        while True:
            try:
                self._tick()
            except KeyboardInterrupt:
                logger.info("Bot stopped by user.")
                sys.exit(0)
            except Exception as e:
                logger.error(f"Unhandled error: {e}\n{traceback.format_exc()}")

            # Quick position check giua cac tick - bat breakeven/TP spike som
            try:
                positions_now = self.client.get_positions()
                if positions_now:
                    time.sleep(1)
                    try:
                        self.executor.manage_positions(self.client.get_positions())
                    except Exception as _e:
                        logger.error(f"manage_positions error: {str(_e).encode('ascii','replace').decode()}")
            except Exception as _e:
                logger.error(f"get_positions error (inter-tick): {str(_e).encode('ascii','replace').decode()}")

            # Minimal pause (rate limit protection)
            time.sleep(config.LOOP_INTERVAL_SEC)

    def _tick(self):
        now = time.time()

        # Cap nhat danh sach trending moi SCAN_INTERVAL_SEC giay
        if now - self.last_scan_ts >= config.SCAN_INTERVAL_SEC:
            symbols = self.scanner.scan()
            if symbols:
                self.symbols = symbols
                self.last_scan_ts = now
                logger.info(f"Trending list updated: {len(self.symbols)} coins")
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

        _today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if _today != self._equity_day_date:
            self._equity_day_date = _today
            logger.info(f"[DAILY] New day - equity: {equity:.2f} USDT")
        _daily_pnl_pct = 0.0
        try:
            _today_realized_pnl = self.client.get_today_pnl()
            _daily_pnl_pct = _today_realized_pnl / equity if equity > 0 else 0
            logger.debug(f"[DAILY PnL] today={_daily_pnl_pct*100:.2f}%")
        except Exception:
            pass

        logger.info(
            f"[TICK] {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')} | "
            f"Equity={equity:.2f} USDT | "
            f"Open={len(open_positions)} | "
            f"DayPnL={_daily_pnl_pct*100:+.2f}%"
        )

        # Quan ly vi the dang mo
        if open_positions:
            self.executor.manage_positions(open_positions)
            # DYNAMIC EXIT: phan tich lien tuc -> chot loi khi market quay dau,
            # cat lo som khi trend nguoc han (khong cho SL/TP chet)
            self._dynamic_manage_positions(open_positions)
            # Refresh lai sau khi manage - co the co lenh vua dong (SL/TP hit)
            # De bot co the re-enter ngay trong cung tick nay
            try:
                open_positions = self.client.get_positions()
                equity         = self.client.get_wallet_balance()
            except Exception:
                pass

        pos_symbols = {p["symbol"] for p in open_positions}

        # Detect position dong boi exchange (SL/TP hit) - xoa executor state de tranh stale
        closed_by_exchange = self._prev_pos_symbols - pos_symbols
        if closed_by_exchange:
            for sym in closed_by_exchange:
                self.executor.clear_position_state(sym)
                # Chi xoa cooldown neu da du 30s ke tu lan trade cuoi
                # Tranh truong hop exchange cham ghi nhan lenh moi -> bi coi la "closed" -> double entry
                _last_trade = self._last_analyzed.get(sym, 0)
                _age = time.time() - _last_trade
                if _age > 30:
                    self._last_analyzed.pop(sym, None)
                    logger.info(f"{sym}: position closed by exchange (SL/TP hit) - cooldown cleared, re-entry allowed")
                else:
                    logger.info(f"{sym}: position closed by exchange - keeping cooldown ({_age:.0f}s < 30s, guard against stale API)")
        self._prev_pos_symbols = pos_symbols

        # Cap nhat BTC global trend TRUOC cap check - tranh BTC trend stale khi at max positions
        # btc_trend    = 15m trend direction (nhanh, bat flip som)
        # btc_trend_4h = 1h trend direction  (chac chan hon, xac nhan xu huong lon)
        try:
            df_btc = self.client.get_klines("BTCUSDT", "1", 1000)
            if not df_btc.empty and len(df_btc) >= 605:  # EMA600 can it nhat 605 nen
                self.df_btc = df_btc
                # EMA(20/50) tren 1m ~ ~20min/50min trend - nhanh, bat bounce/dip BTC
                self.btc_trend_fast = self._trend_direction(df_btc, fast=20, slow=50)
                # EMA(100/250) tren 1m ~ EMA(20/50) tren 5m - medium trend BTC
                self.btc_trend    = self._trend_direction(df_btc, fast=100, slow=250)
                # EMA(300/600) tren 1m ~ EMA(20/40) tren 15m - macro trend BTC
                self.btc_trend_4h = self._trend_direction(df_btc, fast=300, slow=600)
        except Exception:
            pass



        # Xu ly TAT CA coin trending, coin score cao nhat truoc
        # Dung time budget: uu tien top 20 truoc, phan con lai sau
        logger.info(
            f"[TICK] Scan {len(self.symbols)} coins (top{config.TOP20_COUNT}+rest) | "
            f"open={len(open_positions)} | "
            f"BTC_fast={'UP' if self.btc_trend_fast==1 else 'DOWN' if self.btc_trend_fast==-1 else 'SIDE'} "
            f"BTC_mid={'UP' if self.btc_trend==1 else 'DOWN' if self.btc_trend==-1 else 'SIDE'} "
            f"BTC_macro={'UP' if self.btc_trend_4h==1 else 'DOWN' if self.btc_trend_4h==-1 else 'SIDE'}"
        )

        _pos_side_map = {p["symbol"] for p in open_positions}
        _pos_side_map = {p["symbol"]: p.get("side", "") for p in open_positions}
        _now          = time.time()

        # KHONG dung phanh daily-loss: chan lo phai o LOGIC VAO LENH (sat market, dung trend),
        # khong phai o phanh dem PnL/dem so lenh. Chi giu gate potential + free-margin ben duoi.
        # MAX CONCURRENT: 0 = khong gioi han (potential + free-margin tu dieu tiet so lenh)
        if config.MAX_CONCURRENT_POSITIONS > 0 and len(open_positions) >= config.MAX_CONCURRENT_POSITIONS:
            logger.info(f"[RISK] Da co {len(open_positions)}/{config.MAX_CONCURRENT_POSITIONS} lenh - khong mo them, cho lenh dong")
            return

        # Coin de phan tich: 0 = tat ca (da sort theo trend score), >0 = cat top-N.
        # Khong cat cung -> khong bo lo lenh tiem nang; budget+cooldown tu dieu tiet.
        if config.TOP_TRADE_COUNT > 0:
            top_trade = self.symbols[:config.TOP_TRADE_COUNT]
        else:
            top_trade = self.symbols

        def _run_scan(symbols: list[str], cooldown: float, budget: float, label: str) -> bool:
            """Chay scan cho 1 nhom symbols. Tra ve equity_exhausted."""
            nonlocal open_positions, equity, pos_symbols, _pos_side_map
            _start          = time.time()
            _exec_overhead  = 0.0   # tong thoi gian execute trade + refresh state - khong tinh vao budget
            _analyzed       = 0
            _traded         = 0
            for symbol in symbols:
                # Budget chi tinh thoi gian SCAN/ANALYSIS, khong tinh thoi gian execute trade
                _scan_elapsed = (time.time() - _start) - _exec_overhead
                if _scan_elapsed > budget:
                    logger.debug(f"[TICK] {label} budget hit after {_analyzed} analyzed, {_traded} traded")
                    break
                if symbol in pos_symbols:
                    continue
                # Dung ngay khi da du so lenh (0 = khong gioi han)
                if config.MAX_CONCURRENT_POSITIONS > 0 and len(pos_symbols) >= config.MAX_CONCURRENT_POSITIONS:
                    break
                _last = self._last_analyzed.get(symbol, 0)
                if _now - _last < cooldown:
                    continue
                _analyzed += 1
                self._last_analyzed[symbol] = _now
                try:
                    traded = self._process_symbol(
                        symbol, equity, open_positions, is_priority=True,
                        btc_eth_side_map=_pos_side_map,
                    )
                    if traded:
                        _traded += 1
                        # Hard cooldown 60s sau khi trade - khong cho re-enter bat ke top20 hay rest
                        self._last_analyzed[symbol] = time.time() + (60 - min(cooldown, 60))
                        pos_symbols.add(symbol)
                        _t0 = time.time()
                        try:
                            open_positions = self.client.get_positions()
                            equity         = self.client.get_wallet_balance()
                            pos_symbols    = {p["symbol"] for p in open_positions}
                            _pos_side_map  = {p["symbol"]: p.get("side", "") for p in open_positions}
                        except Exception:
                            pass
                        _exec_overhead += time.time() - _t0
                        # Giu lai symbol trong pos_symbols du exchange chua ghi nhan kip
                        pos_symbols.add(symbol)
                        if equity <= 0:
                            return True
                except Exception as e:
                    logger.warning(f"Error processing {symbol}: {str(e).encode('ascii','replace').decode()}")
            logger.debug(f"[TICK] {label}: analyzed {_analyzed}/{len(symbols)} coins, traded {_traded}")
            return False

        # CHI 1 PASS tren TOP COIN - khong scan phan con lai (khong rai rac ra coin nho)
        _run_scan(top_trade, config.TOP20_COOLDOWN_SEC, config.SCAN_BUDGET_TOP20_SEC, "TOP")

    def _trend_direction(self, df, fast: int = 20, slow: int = 50) -> int:
        """+1 up, -1 down, 0 sideways.
        Tren 1m data dung EMA dai hon de tinh trend tuong duong cac TF cao hon:
          fast=100, slow=250 -> tuong duong EMA20/50 tren 5m (~1.7h/~4h)
          fast=300, slow=600 -> tuong duong EMA20/40 tren 15m (~5h/~10h)
        """
        if len(df) < slow + 5:
            return 0
        close = df["close"]
        ema_f = compute_ema(close, fast).iloc[-1]
        ema_s = compute_ema(close, slow).iloc[-1]
        price = close.iloc[-1]
        if price > ema_f > ema_s:
            return 1
        if price < ema_f < ema_s:
            return -1
        return 0

    def _dynamic_manage_positions(self, open_positions: list[dict]):
        """Quan ly lenh CHU DONG bang phan tich lien tuc (khong cho SL/TP chet).
        - Dang LOI + market quay dau nguoc -> chot ngay (khoa loi, khong de thanh lo).
        - Dang LO + trend lon nguoc han (khong phuc hoi) -> dong luc LO IT NHAT (bounce),
          hoac cat cung khi lo qua sau -> khong cho cham SL banh chanh.
        SL/TP tren san van con nguyen lam luoi an toan cuoi cung."""
        if not getattr(config, "DYN_EXIT_ENABLE", True) or not open_positions:
            return
        for pos in open_positions:
            symbol = pos["symbol"]
            side   = pos.get("side", "")
            if side not in ("Buy", "Sell"):
                continue
            pos_dir = 1 if side == "Buy" else -1

            # Cho lenh 'tho' - khong dong theo nhieu ngan han ngay sau khi vao
            _created = int(pos.get("createdTime", 0) or 0) / 1000
            if _created > 0 and (time.time() - _created) < config.DYN_MIN_HOLD_SEC:
                continue

            # PnL ROI (tren margin)
            _pos_val = max(_sf(pos.get("positionValue", 0)), 0.0)
            _lev     = max(1.0, _sf(pos.get("leverage", 10.0)))
            _margin  = _pos_val / _lev if _lev > 0 else 0.0
            _upnl    = _sf(pos.get("unrealisedPnl", 0))
            if _margin <= 0:
                continue
            pnl_roi = _upnl / _margin

            # Phan tich hien tai cua chinh coin nay
            try:
                df = self.client.get_klines_paginated(symbol, "1", 700)
            except Exception as e:
                logger.debug(f"[DYN-EXIT] {symbol}: fetch klines failed: {e!r}")
                continue
            if df is None or len(df) < 260:
                continue

            macro_trend = self._trend_direction(df, fast=100, slow=250)
            macro_4h    = self._trend_direction(df, fast=300, slow=600)
            imm         = self._immediate_momentum(df, sp=1.0)
            # macro_dir: dung cho dynamic exit.
            # Lay macro_trend lam chieu chinh; chi reset ve 0 khi macro_4h XUNG DOT (ngược chiều).
            # macro_4h == 0 (trung tinh): van tin macro_trend.
            macro_dir   = macro_trend if (macro_trend != 0 and macro_4h != -macro_trend) else 0

            # Nguong chot loi phai TRU phi DONG lenh (theo ROI = exit_fee * leverage) +
            # buffer funding -> chot la LOI RONG that su, khong hoa/lo vi phi o don bay cao.
            # unrealisedPnl cua Bybit la GROSS (chua tru phi dong) nen phai cong nguong len.
            _exit_cost_roi = (config.EXIT_FEE + config.FUNDING_FEE_BUFFER) * _lev
            _lock_thresh   = max(config.DYN_PROFIT_LOCK_ROI, _exit_cost_roi + 0.02)  # +2% net toi thieu

            # 1) PROFIT-LOCK: dang loi (sau phi dong) ma market quay dau nguoc -> chot ngay
            if pnl_roi >= _lock_thresh:
                _turned = (imm == -pos_dir) or (macro_dir == -pos_dir)
                if _turned:
                    logger.warning(
                        f"[DYN-EXIT] {symbol} {side}: LOCK PROFIT roi=+{pnl_roi*100:.0f}% "
                        f"| market quay dau (imm={imm} macro={macro_dir} pos={pos_dir}) -> chot ngay"
                    )
                    self.executor._close_position(pos)
                    continue

            # 2) SMART CUT-LOSS: dang lo + trend lon nguoc han -> khong the phuc hoi
            if pnl_roi < 0 and macro_dir == -pos_dir:
                if imm == pos_dir:
                    # co bounce nguoc ve phia minh = luc LO IT NHAT -> dong ngay
                    logger.warning(
                        f"[DYN-EXIT] {symbol} {side}: CUT on bounce roi={pnl_roi*100:.0f}% "
                        f"| trend nguoc (macro={macro_dir}) + bounce (imm={imm}) -> dong luc lo it nhat"
                    )
                    self.executor._close_position(pos)
                    continue
                if pnl_roi <= -config.DYN_HARD_CUT_ROI:
                    # lo qua sau, khong co bounce -> cat luon, khong cho cham SL banh chanh
                    logger.warning(
                        f"[DYN-EXIT] {symbol} {side}: HARD CUT roi={pnl_roi*100:.0f}% "
                        f"| trend nguoc + khong bounce -> cat, chan lo them"
                    )
                    self.executor._close_position(pos)
                    continue

    def _btc_correlated(self, df_coin, lookback: int = 100, threshold: float = 0.5) -> bool:
        """True neu coin co rolling correlation voi BTC >= threshold (neo theo BTC).
        False = coin chay doc lap, bo qua BTC filter."""
        if self.df_btc is None or df_coin is None or df_coin.empty:
            return True  # khong co data -> assume correlated (an toan hon)
        n = min(lookback, len(df_coin), len(self.df_btc))
        if n < 30:
            return True
        coin_ret = df_coin["close"].iloc[-n:].pct_change().dropna()
        btc_ret  = self.df_btc["close"].iloc[-n:].pct_change().dropna()
        min_len  = min(len(coin_ret), len(btc_ret))
        if min_len < 20:
            return True
        corr = coin_ret.iloc[-min_len:].corr(btc_ret.iloc[-min_len:])
        if corr != corr:  # NaN
            return True
        return abs(corr) >= threshold

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

    def _volume_confirmed_trend(self, df, sp: float = 1.0) -> tuple[int, float]:
        """QUAN TOA XAC DINH TREND - ket hop VOLUME (trong so lon) + cau truc gia.
        Tra ve (direction, strength) - direction: +1 up, -1 down, 0 khong ro;
        strength 0.0-1.0 (do manh/ro cua trend, dung cho TP/SL scaling + priority).

        Volume la yeu to QUYET DINH - trend that su phai co dong tien xac nhan:
          - OBV (On-Balance Volume) slope: dong tien tich luy theo huong nao
          - Directional volume: volume nen xanh vs do (ai dang thang the)
          - Volume expansion: volume tang dan = trend con fuel (khong phai kiet suc)
          - Price structure: EMA9/21/50 stack + higher-highs/lower-lows
          - Net move: gia da di bao nhieu % (xac nhan co move that)

        Dung cho MASTER AUTHORITY: neu strong+clear -> mo lenh THEO huong nay,
        flip moi signal nguoc chieu. Giai quyet HBAR (up+vol -> short) va HYPE (down -> long)."""
        if df is None or df.empty or len(df) < 60:
            return 0, 0.0

        close  = df["close"]
        open_  = df["open"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]
        n = len(df)
        price = close.iloc[-1]

        # -- 1. OBV slope (dong tien tich luy) - trong so cao nhat --------------
        _chg = close.diff().fillna(0.0)
        _dir_sign = _chg.apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))
        _obv = (_dir_sign * volume).cumsum()
        obv_score = 0.0
        if n >= 30:
            _obv_now  = _obv.iloc[-1]
            _obv_past = _obv.iloc[-30]
            _obv_rng  = float(abs(_obv.iloc[-30:]).max()) + 1e-9
            _obv_slope = (_obv_now - _obv_past) / _obv_rng
            if   _obv_slope >  0.15: obv_score =  1.0
            elif _obv_slope < -0.15: obv_score = -1.0
            else: obv_score = _obv_slope / 0.15   # tuyen tinh trong vung yeu

        # -- 2. Directional volume (buy vs sell) 20c ---------------------------
        _rv = min(20, n)
        _bull_v = volume.iloc[-_rv:][close.iloc[-_rv:] > open_.iloc[-_rv:]].sum()
        _bear_v = volume.iloc[-_rv:][close.iloc[-_rv:] < open_.iloc[-_rv:]].sum()
        _tot_v  = _bull_v + _bear_v
        dvol_score = 0.0
        if _tot_v > 0:
            _br = _bull_v / _tot_v
            dvol_score = (_br - 0.5) / 0.15   # 0.65->+1, 0.35->-1
            dvol_score = max(-1.0, min(1.0, dvol_score))

        # -- 3. Volume expansion (trend con fuel?) -----------------------------
        vexp = 0.0
        if n >= 40:
            _v_recent = volume.iloc[-10:].mean()
            _v_prior  = volume.iloc[-40:-10].mean()
            if _v_prior > 0:
                _ratio = _v_recent / _v_prior
                vexp = min(1.0, max(0.0, (_ratio - 1.0) / 0.5))   # +50% vol = full expansion

        # -- 4. Price structure: EMA stack -------------------------------------
        e9  = compute_ema(close, 9).iloc[-1]
        e21 = compute_ema(close, 21).iloc[-1]
        e50 = compute_ema(close, min(50, n - 1)).iloc[-1]
        if   price > e9 > e21 > e50: struct_score =  1.0
        elif price < e9 < e21 < e50: struct_score = -1.0
        elif price > e21 and e9 > e21: struct_score =  0.5
        elif price < e21 and e9 < e21: struct_score = -0.5
        else: struct_score = 0.0

        # -- 5. Higher-highs/lower-lows 15c ------------------------------------
        hhll = 0.0
        if n >= 15:
            _hr = high.iloc[-8:].max();  _hp = high.iloc[-15:-8].max()
            _lr = low.iloc[-8:].min();   _lp = low.iloc[-15:-8].min()
            if   _hr > _hp and _lr > _lp: hhll =  1.0
            elif _hr < _hp and _lr < _lp: hhll = -1.0

        # -- 6. Net move 30c (co move that su khong) ---------------------------
        netmove = 0.0
        if n >= 30 and close.iloc[-30] > 0:
            _nm = (price - close.iloc[-30]) / close.iloc[-30]
            netmove = max(-1.0, min(1.0, _nm / (0.010 * sp)))   # 1%*sp = full

        # -- TONG HOP: volume nhom (OBV + dvol) la chu dao --------------------
        _vol_core = obv_score * 0.5 + dvol_score * 0.5     # -1..1
        _struct_core = struct_score * 0.5 + hhll * 0.3 + netmove * 0.2  # -1..1

        # Direction - GATE CHAT de tranh doc nham chop/nhieu la trend:
        #   1. Volume core manh (>0.30): OBV + directional volume dong thuan
        #   2. EMA stack DAY DU (struct_score == +/-1: price>e9>e21>e50): trend that
        #   3. Net move THAT (abs>=0.35 = >=0.35%*sp): co di chuyen, khong phai dung im
        #   4. HH/LL khong nguoc chieu (>=0 cho up, <=0 cho down)
        #   5. Struct core tong hop manh (>0.35)
        if (_vol_core > 0.30 and struct_score >= 1.0 and abs(netmove) >= 0.35
                and hhll >= 0.0 and _struct_core > 0.35):
            direction = 1
        elif (_vol_core < -0.30 and struct_score <= -1.0 and abs(netmove) >= 0.35
                and hhll <= 0.0 and _struct_core < -0.35):
            direction = -1
        else:
            return 0, 0.0

        # Strength ~ do lon 2 core, ti le voi MOVE THUC (move nho khong the "manh")
        # va boost boi volume expansion (trend con fuel). Chop move nho -> strength thap.
        _mag = (abs(_vol_core) + abs(_struct_core)) / 2.0
        strength = min(1.0, _mag * (0.55 + 0.45 * abs(netmove)) * (1.0 + 0.2 * vexp))
        return direction, round(strength, 3)

    def _exhaustion_check(self, df_micro, direction: int, price: float,
                          rsi_now: float, sp: float, vol_locked: bool) -> tuple[int, float, str]:
        """Phan biet KIET SUC (chan) vs TREND TIEP DIEN (cho) tai cuc doan range.
        Ket qua (new_direction, tp_override, reason): new_direction=0 -> BLOCK.

        Nguyen tac: KHONG mua DINH kiet suc, KHONG ban DAY kiet suc - NHUNG van cho
        trend tiep dien (downtrend dang do -> short OK; uptrend deu -> long OK).
        Phan biet bang 2 dau hieu KIET SUC tai cuc doan:
          1. OVER-EXTENSION: gia cach EMA21 qua xa (>3 ATR) = parabol/qua da (BZ pump doc)
          2. DAO CHIEU: nen dao chieu (bounce tai day / reject tai dinh) hoac RSI cuc doan
        Tai TOP >=80%:
          - long ma (reject HOAC over-extended up) -> KIET -> flip short (neu reject) / block
          - long ma dang len DEU (immediate up, khong over-ext, khong reject) -> CHO (trend)
        Tai BOTTOM <=20%: doi xung.
        vol_locked: khong flip (tranh whipsaw) nhung van block khi kiet."""
        # Cua so 45 nen (~45 phut) - do CUC DOAN CUC BO, khong phai range 2h.
        # 2h range lam gia trong trend luon peg o dinh/day -> chan het continuation.
        # 45 nen: pullback/bounce dua gia ve GIUA cua so (trade duoc), con parabol/vertical
        # thi gia o cuc doan cua 45 nen (block BZ). Coin moi list >=30 nen van duoc bao ve.
        if df_micro is None or df_micro.empty or len(df_micro) < 30:
            return direction, 0.0, "pass"
        _nw = min(45, len(df_micro))
        hi = df_micro["high"].iloc[-_nw:].max()
        lo = df_micro["low"].iloc[-_nw:].min()
        rng = hi - lo
        if rng <= 0 or lo <= 0 or (rng / lo) < 0.006 * sp:
            return direction, 0.0, "pass-narrow"
        p   = price if price > 0 else df_micro["close"].iloc[-1]
        pos = (p - lo) / rng
        c = df_micro["close"].iloc[-1]; o = df_micro["open"].iloc[-1]
        h = df_micro["high"].iloc[-1];  l = df_micro["low"].iloc[-1]
        rngc = max(h - l, 1e-12); body = c - o
        lower_wick = (min(o, c) - l) / rngc
        upper_wick = (h - max(o, c)) / rngc
        # Dau hieu dao chieu tai cuc doan (de FLIP thay vi chi block)
        reject = (body < 0) or (upper_wick > 0.5) or (rsi_now > 68)
        bounce = (body > 0) or (lower_wick > 0.5) or (rsi_now < 32)
        # QUY TAC: KHONG trade trong 25% CUC DOAN (dinh/day) - "dinh hoac gan dinh, day hoac gan day".
        # Trend vao lenh o vung giua (25-75%): long tren pullback, short tren bounce - entry dep hon,
        # khong bao gio mua sat/gan dinh / ban sat/gan day. Tai cuc doan: flip neu dao chieu, else skip.
        # KHONG long trong 25% TREN cua range, KHONG short trong 25% DUOI (dinh/gan-dinh,
        # day/gan-day). Vung giua 25-75%: long tren pullback / short tren bounce - VAN bat
        # duoc trend continuation (trong uptrend gia thuong o phan tren, 30/70 qua chat).
        # Tai cuc doan: flip neu co dao chieu ro, khong thi SKIP.
        # QUY TAC TUYET DOI (user lap nhieu lan): KHONG short o DAY hoac GAN DAY,
        # KHONG long o DINH hoac GAN DINH. Vung cuc doan = 25% tren/duoi cua so 45 nen.
        #  - HARD extreme (>=90% / <=10%): blow-off/vertical -> block, flip neu co dao chieu.
        #  - SOFT extreme (75-90% / 10-25% = "gan dinh/gan day"): LUON block huong nguy hiem
        #    (short gan day / long gan dinh). Neu co dao chieu ro -> flip; khong thi SKIP.
        #    KHAC ban cu: truoc chi block SOFT khi co bounce/reject -> lot "short gan day" khi
        #    nen con do/RSI chua qua ban -> bounce sau do -> LO. Gio gan day/gan dinh = CAM.
        HARD_TOP, HARD_BOT = 0.90, 0.10
        SOFT_TOP, SOFT_BOT = 0.75, 0.25
        if direction == 1:
            if pos >= HARD_TOP:
                if reject and not vol_locked:
                    return -1, 0.10, f"flip LONG->SHORT blow-off@dinh {pos:.0%}"
                return 0, 0.0, f"BLOCK LONG@blow-off-dinh {pos:.0%}"
            if pos >= SOFT_TOP:   # gan dinh -> KHONG long (du co reject hay khong)
                if reject and not vol_locked:
                    return -1, 0.10, f"flip LONG->SHORT reject@gan-dinh {pos:.0%}"
                return 0, 0.0, f"BLOCK LONG@gan-dinh {pos:.0%} (khong long gan dinh)"
        if direction == -1:
            if pos <= HARD_BOT:
                if bounce and not vol_locked:
                    return 1, 0.10, f"flip SHORT->LONG blow-off@day {pos:.0%}"
                return 0, 0.0, f"BLOCK SHORT@blow-off-day {pos:.0%}"
            if pos <= SOFT_BOT:   # gan day -> KHONG short (du co bounce hay khong)
                if bounce and not vol_locked:
                    return 1, 0.10, f"flip SHORT->LONG bounce@gan-day {pos:.0%}"
                return 0, 0.0, f"BLOCK SHORT@gan-day {pos:.0%} (khong short gan day)"
        return direction, 0.0, f"pass@{pos:.0%}"

    def _immediate_momentum(self, df, sp: float = 1.0, n: int = 7) -> int:
        """Chieu di chuyen NGAY LUC NAY (n nen gan nhat) - nhanh hon EMA/vwt.
        +1 dang len, -1 dang xuong, 0 di ngang. Dung de KHONG trade nguoc move hien tai
        (khong short khi dang bounce len, khong long khi dang do xuong)."""
        if df is None or df.empty or len(df) < n + 1:
            return 0
        r = df.iloc[-n:]
        c0 = r["close"].iloc[0]; c1 = r["close"].iloc[-1]
        if c0 <= 0:
            return 0
        net = (c1 - c0) / c0
        greens = int((r["close"] > r["open"]).sum())
        reds   = int((r["close"] < r["open"]).sum())
        thr = 0.0025 * sp   # 0.25%*sp qua n nen = move that su
        if net > thr and greens > reds:
            return 1
        if net < -thr and reds > greens:
            return -1
        return 0

    def _true_direction(self, macro_trend: int, macro_4h: int, vwt_dir: int,
                        vwt_str: float, imm: int) -> int:
        """TREND THUC - can bang: DUNG chieu + du xac nhan, nhung KHONG qua chat (van co lenh).
        Chieu = IMMEDIATE momentum (huong gia dang di NGAY LUC NAY) - khong bao gio trade
        nguoc move hien tai. Sau do loc:
          - imm == 0 (khong co move ro): KHONG trade (choppy).
          - volume MANH nguoc move (vwt = -D, str>=0.45): day la pullback/bounce nguoc -> skip.
          - CA 2 macro nguoc move VA volume khong xac nhan: nguoc trend lon -> skip.
          - Can XAC NHAN: volume cung chieu (str>=0.40) HOAC it nhat 1 macro cung chieu.
        Bat duoc trend TRE (macro EMA con lag) qua volume+immediate, van tranh trade sai chieu."""
        if imm == 0:
            return 0
        D = imm
        # Volume manh nguoc chieu move hien tai -> countertrend bounce/pullback -> cho
        if vwt_dir == -D and vwt_str >= 0.45:
            return 0
        # Ca 2 macro nguoc chieu VA volume khong xac nhan D -> nguoc trend lon that su
        if macro_trend == -D and macro_4h == -D and not (vwt_dir == D and vwt_str >= 0.45):
            return 0
        # Can it nhat 1 xac nhan: volume cung chieu HOAC 1 macro cung chieu
        if (vwt_dir == D and vwt_str >= 0.40) or macro_trend == D or macro_4h == D:
            return D
        return 0   # imm don le khong co backing (volume/macro) -> co the la noise -> skip

    def _find_local_extrema(self, df, window: int = 5):
        """
        Tim local peaks va troughs tren 1m data.
        peak:   high[i] la cao nhat trong [i-window .. i+window]
        trough: low[i]  la thap nhat trong [i-window .. i+window]
        Tra ve: (peaks_idx, troughs_idx, last_peak_price, last_trough_price,
                 candles_since_peak, candles_since_trough)
        """
        if df is None or df.empty or len(df) < window * 2 + 1:
            return [], [], 0.0, 0.0, 999, 999
        highs = df["high"].values
        lows  = df["low"].values
        n = len(highs)
        peaks, troughs = [], []
        for i in range(window, n - window):
            if highs[i] == max(highs[i-window:i+window+1]):
                peaks.append(i)
            if lows[i] == min(lows[i-window:i+window+1]):
                troughs.append(i)
        last_peak_idx    = peaks[-1]   if peaks   else 0
        last_trough_idx  = troughs[-1] if troughs else 0
        last_peak_price  = highs[last_peak_idx]  if peaks   else highs.max()
        last_trough_price = lows[last_trough_idx] if troughs else lows.min()
        candles_since_peak   = n - 1 - last_peak_idx
        candles_since_trough = n - 1 - last_trough_idx
        return peaks, troughs, last_peak_price, last_trough_price, candles_since_peak, candles_since_trough

    def _micro_entry_analysis(self, df_micro, direction: int, is_reversal: bool = False, strong_trend: bool = False) -> bool:
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

        # Factor 1: EMA alignment - price / EMA9 / EMA21 phai xep hang dung chieu
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

        # Factor 2: Momentum - it nhat 1/3 nen gan nhat phai cung chieu (giam tu 2/3)
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

        # Factor 3: Volume binh thuong - khong phai spike va khong qua nho
        if n >= 10:
            vol_ma = volume.rolling(10).mean().iloc[-1]
            if vol_ma > 0:
                ratio = volume.iloc[-1] / vol_ma
                if 0.5 <= ratio <= 4.0:   # volume hop le
                    score += 1
                elif ratio > 6.0:          # spike volume - co the dang o dinh/day
                    score -= 1

        # Factor 4: Khong exhausted - nen hien tai khong qua nho sau loat nen lon (pause signal)
        if n >= 4:
            prev_body_avg = abs(close.iloc[-4:-1].values - open_.iloc[-4:-1].values).mean()
            curr_body     = abs(close.iloc[-1] - open_.iloc[-1])
            if prev_body_avg > 0:
                ratio_body = curr_body / prev_body_avg
                if ratio_body > 0.3:    # nen hien tai co noi luc
                    score += 1
                elif ratio_body < 0.15: # doji / spinning top - exhaustion signal
                    score -= 1

        # Factor 5: Micro structure - HH+HL (long) hoac LH+LL (short) trong 5 nen gan nhat
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

        # Factor 6: Range check - 3 muc: 30c (30 phut), 100c (1.7h), 20c (local)
        # Nguong du chat de tranh du dinh / du day trong momentum path
        # is_reversal=True -> skip block (reversal vao chinh xac o cuc doan)
        #
        # 30-candle (30 phut): block LONG neu o top 70%, block SHORT neu o bottom 30%
        _w30 = min(30, n)
        if _w30 >= 15 and not is_reversal:
            _h30 = high.iloc[-_w30:].max()
            _l30 = low.iloc[-_w30:].min()
            _r30 = _h30 - _l30
            if _r30 > 0:
                _p30 = (price - _l30) / _r30
                if direction == 1 and _p30 > 0.92:
                    logger.debug(f"micro_entry: BLOCK long - 30c range_pos={_p30:.2f} > 0.92 (du dinh 30 phut)")
                    return False
                if direction == -1 and _p30 < 0.08:
                    logger.debug(f"micro_entry: BLOCK short - 30c range_pos={_p30:.2f} < 0.08 (du day 30 phut)")
                    return False

        # 100-candle (~1.7h): block LONG neu o top, block SHORT neu o bottom
        # strong_trend=True: BTC/coin dang uptrend manh (macro+scalp+micro confirm) -> skip range block
        # BTC tang tu 63678->66937: sau moi TP, range_pos>88% -> block re-entry -> miss tiep theo
        _range_window = min(100, n)
        if _range_window >= 20:
            high_rng = high.iloc[-_range_window:].max()
            low_rng  = low.iloc[-_range_window:].min()
            rng = high_rng - low_rng
            if rng > 0:
                range_pos = (price - low_rng) / rng
                if not is_reversal and not strong_trend:
                    if direction == 1 and range_pos > 0.88:
                        logger.debug(f"micro_entry: BLOCK long - 100c range_pos={range_pos:.2f} > 0.88")
                        return False
                    if direction == -1 and range_pos < 0.12:
                        logger.debug(f"micro_entry: BLOCK short - 100c range_pos={range_pos:.2f} < 0.12")
                        return False
                # Bonus cho entry o vung an toan (range 30%-70%)
                if direction == 1 and range_pos < 0.45:
                    score += 1
                elif direction == -1 and range_pos > 0.55:
                    score += 1

        # 20-candle local range: block LONG top, block SHORT bottom
        # strong_trend: trong uptrend manh, price lien tuc lam dinh moi -> local_pos luon > 90% -> skip
        _local_window = min(20, n)
        if _local_window >= 10 and not is_reversal and not strong_trend:
            local_high = high.iloc[-_local_window:].max()
            local_low  = low.iloc[-_local_window:].min()
            local_rng  = local_high - local_low
            if local_rng > 0:
                local_pos = (price - local_low) / local_rng
                if direction == 1 and local_pos > 0.90:
                    logger.debug(f"micro_entry: BLOCK long - 20c local_pos={local_pos:.2f} > 0.90")
                    return False
                if direction == -1 and local_pos < 0.10:
                    logger.debug(f"micro_entry: BLOCK short - 20c local_pos={local_pos:.2f} < 0.10")
                    return False

        # Factor 7: Momentum deceleration - nen gan day nho manh so voi nen truoc
        # Tranh vao lenh khi momentum dang kiet suc (sap dao chieu)
        if n >= 8:
            recent_body = abs(close.iloc[-3:-1].values - open_.iloc[-3:-1].values).mean()
            prev_body   = abs(close.iloc[-8:-3].values - open_.iloc[-8:-3].values).mean()
            if prev_body > 0:
                decel = recent_body / prev_body
                if decel < 0.35:    # Momentum giam > 65% - dang dung lai / dao chieu
                    score -= 1
                elif decel > 0.60:  # Momentum on dinh
                    score += 1

        threshold = 2
        ok = score >= threshold
        if not ok:
            logger.debug(f"micro_entry_analysis: dir={direction} score={score}/{threshold} n={n} -> skip")
        return ok

    def _process_symbol(self, symbol: str, equity: float, open_positions: list[dict], is_priority: bool = True, btc_eth_side_map: dict | None = None) -> bool:
        """Phan tich symbol, chay tat ca filter va strategy, tra True neu da trade."""
        # Init gradual trend flags - se duoc tinh chinh xac sau khi co df_micro
        _is_gradual_uptrend   = False
        _is_gradual_downtrend = False
        # Init AEQ-12 bypass flags - dung o AEQ-EXTREMA ca khi block AEQ-12 bi skip
        # (bug cu: NameError khi _range_live_price <= 0 -> coin bi bo phan tich am tham)
        _aeq12_bypass_long  = False
        _aeq12_bypass_short = False

        # Fetch 1 lan duy nhat: 2000 nen 1m = ~33h data chi tiet
        # Tat ca df_ reuse cung bo data nay - khong co API call thua
        # Trend/macro duoc tinh bang EMA dai hon tren 1m (chinh xac hon multi-TF)
        df_signal = self.client.get_klines_paginated(symbol, "1", config.CANDLE_LIMIT_SIGNAL)
        df_scalp  = df_signal
        df_trend  = df_signal
        df_macro  = df_signal
        df_micro  = df_signal

        if df_signal.empty or len(df_signal) < 50:
            return False

        # Volatility bucket cho spike/pump-dump thresholds:
        # largecap  (BTC/ETH):              _sp = 0.50 - ATR% ~0.3-0.4%, threshold rat thap
        # midcap    (SOL/XRP/HYPE/BNB/..): _sp = 0.75 - volatility trung binh, giua 2 nhom
        # altcoin   (phan con lai):          _sp = 1.00 - volatility cao nhat
        _LARGECAP = {"BTCUSDT", "ETHUSDT"}
        _MIDCAP   = {"SOLUSDT", "XRPUSDT", "HYPEUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT", "TRXUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT"}
        _is_largecap = symbol in _LARGECAP
        _sp = 0.50 if symbol in _LARGECAP else (0.75 if symbol in _MIDCAP else 1.0)

        # STOCK TOKEN BLACKLIST: cac coin nay la tokenized stocks/ETF, theo NASDAQ chu khong theo BTC
        # BTC trend filter ap dung cho crypto - stock token co dynamic hoan toan khac biet
        # -> Skip hoan toan de tranh trade nhung coin co logic rieng ma bot khong hieu
        _STOCK_TOKENS = {
            "SKHYNIXUSDT", "SKHYUSDT", "MRVLUSDT", "MUUSDT", "SOXLUSDT", "SOXXUSDT",
            "INTCUSDT", "NVDAUSDT", "AMDUSDT", "TSMUSDT", "MSFTUSDT", "TSLAUSDT",
            "AAPLUSDT", "GOOGLAUSDT", "GOOGLUSDT", "AMZNUSDT", "METAUSDT", "TSLAAUSDT",
            "TSLLUSDT", "COINUSDT", "ESPORTSUSDT", "SPCXUSDT", "PLTRUSDT", "HOODUSDT",
            "MSTRUSDT", "IONQUSDT", "PENGSTOCKUSDT", "APPSTOCKUSDT", "PANWUSDT", "ARKKUSDT",
        }
        # Heuristic an toan: chi ten chua "STOCK" (PENGSTOCK, APPSTOCK...) - khong dung
        # suffix-match vi nhieu crypto that ket thuc bang X (AVAX, PYTH...) se bi chan oan.
        if symbol in _STOCK_TOKENS or "STOCK" in symbol:
            logger.debug(f"{symbol}: skip - stock/ETF token, follows NASDAQ not crypto")
            return False

        # ATR filter: bo qua symbol bien dong qua nho
        # Large-cap: 0.2% (BTC ATR% ~0.3-0.4%, ETH ~0.3%), altcoin: 0.4%
        atr   = compute_atr(df_signal).iloc[-1]
        price = df_signal["close"].iloc[-1]
        _min_atr_pct = config.MIN_ATR_PCT * _sp
        if price > 0 and atr / price < _min_atr_pct:
            return False

        # ADX tren 1m: 2000 nen du sample de ADX on dinh, MIN_ADX=12 phu hop cho 1m
        min_adx = config.MIN_ADX
        adx = compute_adx(df_signal).iloc[-1] if not df_signal.empty and len(df_signal) >= 20 else float("nan")
        if math.isnan(adx) or adx < min_adx:
            logger.debug(f"{symbol}: skip - 1m ADX={adx:.1f} < {min_adx} (sideway)")
            return False

        # 24h directional move filter: tranh chase sau khi coin da pump/dump > 20% trong 24h
        # 1440 nen 1m = 1440 phut = 24h chinh xac (chinh xac hon 96x15m vi du lieu 1m granular)
        _W24H = 1440
        _block_long_24h  = False
        _block_short_24h = False
        _change_24h = 0.0
        if len(df_signal) >= _W24H:
            _ref_24h = df_signal["close"].iloc[-_W24H]
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

        # RSI tren 1m: 2000 nen du de tinh on dinh, chinh xac hon RSI 15m (granular hon)
        rsi_now = compute_rsi(df_signal["close"]).iloc[-1] if not df_signal.empty and len(df_signal) >= 14 else 50.0
        # Clamp RSI de tranh NaN/Inf anh huong den reversal logic
        if not (0 <= rsi_now <= 100):
            rsi_now = 50.0

        # Lay live mark price mot lan cho range position checks - tranh dung 15m close (stale up to 14m)
        # Tai day la diem dau tien co du context de goi API (sau spike filter da pass)
        # Reuse cho _live_check_price trong 30c block de tranh second API call
        _range_live_price = self.client.get_current_price(symbol)
        _range_price = _range_live_price if _range_live_price > 0 else price

        # Tinh macro_trend tu 1m data voi EMA dai hon - chinh xac hon, khong tre candle dong:
        #   macro_trend: EMA(100/250) tren 1m ~ EMA(20/50) tren 5m  = medium trend ~1.7h/~4h
        #   macro_4h:    EMA(300/600) tren 1m ~ EMA(20/40) tren 15m = long trend   ~5h/~10h
        macro_trend = self._trend_direction(df_signal, fast=100, slow=250)
        macro_4h    = self._trend_direction(df_signal, fast=300, slow=600)
        # ATR: dung ATR(50) tren 1m thay vi ATR(14) - on dinh hon, it bi anh huong boi spike
        _atr_for_sl = compute_atr(df_signal, 50).iloc[-1] if not df_signal.empty and len(df_signal) >= 50 else 0.0

        # BTC correlation: coin chay theo BTC thi ap dung BTC filter, khong thi phan tich doc lap
        # BTCUSDT/ETHUSDT luon correlated (dung filter rieng). Stock token da bi skip truoc.
        _btc_filter_on = symbol in ("BTCUSDT", "ETHUSDT") or self._btc_correlated(df_signal)
        logger.debug(f"{symbol}: btc_correlated={_btc_filter_on}")

        # Range block ~5h: 300 nen 1m = 300 phut = 5h (tuong duong 20 nen 15m cu)
        # Priority coins trong confirmed trend duoc phep entry o top/bottom hon (85/15 thay vi 75/25)
        _h1_block_long  = False
        _h1_block_short = False
        _W5H = 300   # 300 x 1m = 5h window
        # B3 fix: define before 2h block which uses these regardless of 5h data length
        _btc_bear_rng = self.btc_trend == -1 and self.btc_trend_4h == -1
        _btc_bull_rng = self.btc_trend == 1  and self.btc_trend_4h == 1
        if not df_signal.empty and len(df_signal) >= _W5H:
            h1_high = df_signal["high"].iloc[-_W5H:].max()
            h1_low  = df_signal["low"].iloc[-_W5H:].min()
            h1_rng  = h1_high - h1_low
            if h1_rng > 0:
                h1_pos = (_range_price - h1_low) / h1_rng
                _h1_long_thresh  = 0.70 if (is_priority and macro_trend >= 1) else 0.60
                _h1_short_thresh = 0.30 if (is_priority and macro_trend <= -1) else 0.40
                if h1_pos > _h1_long_thresh:
                    # BTC bull chi bypass o vung giua (60-85%), extreme top (>85%) luon block
                    if not _btc_bull_rng or h1_pos > 0.85:
                        _h1_block_long = True
                        logger.debug(f"{symbol}: 5h range_pos={h1_pos:.2f} > {_h1_long_thresh} -> block LONG (5h top)")
                elif h1_pos < _h1_short_thresh:
                    # BTC bear chi bypass o vung giua (15-40%), extreme bottom (<15%) luon block
                    if not _btc_bear_rng or h1_pos < 0.15:
                        _h1_block_short = True
                        logger.debug(f"{symbol}: 5h range_pos={h1_pos:.2f} < {_h1_short_thresh} -> block SHORT (5h bottom)")

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
                _m2h_top_thresh = 0.72 if (is_priority and macro_trend >= 1) else 0.65
                _m2h_bot_thresh = 0.28 if (is_priority and macro_trend <= -1) else 0.35
                if _m2h_pos < _m2h_bot_thresh:
                    _m2h_block_short = True
                    logger.debug(f"{symbol}: 2h 1m range_pos={_m2h_pos:.2f} < {_m2h_bot_thresh} -> block SHORT (2h bottom)")
                elif _m2h_pos > _m2h_top_thresh:
                    _m2h_block_long = True
                    logger.debug(f"{symbol}: 2h 1m range_pos={_m2h_pos:.2f} > {_m2h_top_thresh} -> block LONG (2h top)")

        # Momentum confirmation (15m): it nhat 2/3 nen gan nhat cung chieu voi signal
        # 2-consecutive (c1 AND c2) qua chat: breakout candle c1=green, c2=red (consolidation) bi block
        # 2-of-3 cho phep mot nen pullback trong xu huong - van du xac nhan momentum
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

        # VOLUME-CONFIRMED TREND - tinh SOM de dung cho ca consensus reduction
        # (uu tien coin trend ro) VA master authority (flip signal nguoc chieu ben duoi)
        _vwt_dir, _vwt_str = self._volume_confirmed_trend(df_micro, _sp)

        # Strong trend flags - dung som cho EMA250, RSI guard, AEQ-4 bypass
        # Ca 3 TF (macro/macro_4h/scalp) cung chieu = trend that su, khong phai spike
        _strong_trend_up = (macro_trend == 1  and macro_4h == 1  and scalp_trend == 1)
        _strong_trend_dn = (macro_trend == -1 and macro_4h == -1 and scalp_trend == -1)

        # [CHONG LO] Spike check tren 1m - direction-aware
        # Pump spike -> block LONG (khong mua dinh), nhung cho phep SHORT (ban dinh la hop le)
        # Dump spike -> block SHORT (khong ban day), nhung cho phep LONG (mua day la hop le)
        # Ca 2 cung xuat hien -> thi truong loan, skip tat ca
        _micro_spike_dump = False
        _micro_spike_pump = False
        _dual_spike       = False
        _live_check_price = 0.0
        _micro_price = df_micro["close"].iloc[-1] if not df_micro.empty else 0.0

        # Tinh gradual trend flags TRUOC spike gate - can cho ca hai nhanh (spike va non-spike)
        # Bug: neu tinh ben trong spike gate, khi spike flag da set truoc, block skip -> flag sai = False
        if not df_micro.empty and len(df_micro) >= 30:
            _30c_c = df_micro["close"].iloc[-30:].values
            _30c_o = df_micro["open"].iloc[-30:].values
            _n_grn = sum(1 for i in range(30) if _30c_c[i] > _30c_o[i])
            _n_red = sum(1 for i in range(30) if _30c_c[i] < _30c_o[i])
            # Gradual = nhieu nen nho cung chieu, KHONG phai spike lon dot ngot
            # Neu co nen nao > 3x average body -> day la spike/pump, KHONG phai gradual
            # ZEC pump 1.4% trong 10 phut: co nen spike lon -> _no_spike_30 = False -> gradual = False
            _30c_bodies = abs(_30c_c - _30c_o)
            _avg_body_30 = _30c_bodies.mean()
            _max_body_30 = _30c_bodies.max()
            _no_spike_30 = (_max_body_30 < _avg_body_30 * 3.0) if _avg_body_30 > 0 else True
            # Net rise/fall trong 30 nen: OPN tang 4.6% gradual (khong spike) van phai bi AEQ-12
            # Neu net move > 1.2% trong 30 nen (30 phut) -> exhaustion, KHONG bypass AEQ-12
            _30c_net_rise = (_30c_c[-1] - _30c_c[0]) / _30c_c[0] if _30c_c[0] > 0 else 0
            _30c_net_fall = (_30c_c[0] - _30c_c[-1]) / _30c_c[0] if _30c_c[0] > 0 else 0
            _is_gradual_uptrend   = _n_grn >= 18 and _no_spike_30 and _30c_net_rise < 0.012
            _is_gradual_downtrend = _n_red >= 18 and _no_spike_30 and _30c_net_fall < 0.012

        # POST-PEAK / POST-TROUGH DECLINE: EMA(100/250) lag 30-60 phut sau khi gia qua dinh/day
        # -> macro_trend van = 1 nhung coin thuc te da dung tang va dang giam sustained
        # Block LONG neu: giam > 1% tu dinh 60c, dinh >= 15c truoc, 5m chua confirm uptrend
        # Ngoai le: scalp_trend == 1 (5m EMA xac nhan uptrend van tiep tuc -> cho phep LONG)
        _post_peak_decline_long  = False
        _post_trough_rise_short  = False
        _ppd_drop = 0.0
        _ppd_rise = 0.0
        _ppd_hi_age = 0
        _ppd_lo_age = 0
        _spike_dump_60c = False   # co nen dump don le > 1.5% trong 60c
        _spike_pump_60c = False   # co nen pump don le > 1.5% trong 60c
        if not df_micro.empty and len(df_micro) >= 60:
            _ppd_price  = df_micro["close"].iloc[-1]
            _ppd_60h    = df_micro["high"].iloc[-60:].max()
            _ppd_60l    = df_micro["low"].iloc[-60:].min()
            _ppd_hi_idx = int(df_micro["high"].iloc[-60:].values.argmax())
            _ppd_lo_idx = int(df_micro["low"].iloc[-60:].values.argmin())
            _ppd_hi_age = 59 - _ppd_hi_idx   # nen cu hon = so lon hon
            _ppd_lo_age = 59 - _ppd_lo_idx
            _ppd_drop   = (_ppd_60h - _ppd_price) / _ppd_60h if _ppd_60h > 0 else 0
            _ppd_rise   = (_ppd_price - _ppd_60l) / _ppd_60l if _ppd_60l > 0 else 0
            _ppd_thresh = 0.010 * _sp   # 1.0% altcoin, 0.5% BTC/ETH

            # Spike detection (% based, khong dung ATR vi chua tinh tai day)
            # CHILLGUY pattern: dump -3.6% trong 1 nen 1m -> price bounce 1.6% -> SHORT sai
            # Dung 1.5% cho altcoin (0.75% cho BTC/ETH) - spike bat thuong, khong phai move binh thuong
            _ppd_spike_th = 0.015 * _sp
            _ppd_60c_cl = df_micro["close"].iloc[-60:].values
            _ppd_60c_op = df_micro["open"].iloc[-60:].values
            _ppd_60c_bp = (_ppd_60c_cl - _ppd_60c_op) / (_ppd_60c_op + 1e-12)
            _spike_dump_60c = any(b < -_ppd_spike_th for b in _ppd_60c_bp)
            _spike_pump_60c = any(b >  _ppd_spike_th for b in _ppd_60c_bp)

            # Spike bounce block: sau spike dump -> price bounce = dump exhausted
            # Block SHORT du scalp_trend = -1 (EMA lag sau spike dump)
            _ppd_dump_bounce = _spike_dump_60c and _ppd_lo_age >= 15 and _ppd_rise > _ppd_thresh
            _ppd_pump_fade   = _spike_pump_60c and _ppd_hi_age >= 15 and _ppd_drop > _ppd_thresh

            if (_ppd_hi_age >= 15 and _ppd_drop > _ppd_thresh and (scalp_trend != 1  or _ppd_pump_fade)):
                _post_peak_decline_long = True
                logger.debug(f"{symbol}: post-peak-decline - {_ppd_drop*100:.1f}% below 60c high ({_ppd_hi_age}c ago) spike_pump={_spike_pump_60c}")
            if (_ppd_lo_age >= 15 and _ppd_rise > _ppd_thresh and (scalp_trend != -1 or _ppd_dump_bounce)):
                _post_trough_rise_short = True
                logger.debug(f"{symbol}: post-trough-rise - {_ppd_rise*100:.1f}% above 60c low ({_ppd_lo_age}c ago) spike_dump={_spike_dump_60c}")

        # == EMERGING TREND DETECTOR - HBAR pattern fix ===========================
        # HBAR loss -23%: gia tao day 2h (0.07069) roi di len BEN VUNG (nen xanh lien
        # tiep, higher lows, reclaim MA) nhung rise moi 0.66% < nguong ppd 1.0% -> khong
        # co bao ve -> flip "at 10c peak" LONG->SHORT ngay CHAN uptrend moi -> nguoc trend.
        # Nhan dien TREND DANG HINH THANH bang CAU TRUC gia (khong chi dua % move):
        #   1. Day/dinh 60c co tuoi >= 8 nen (khong phai bounce 1-2 nen)
        #   2. Da di 0.5%-2.5% (*_sp) tu day/dinh - co move that su NHUNG van con SOM
        #   3. Gia reclaim EMA21 VA EMA50 1m (cau truc gia da doi phe)
        #   4. EMA9 vs EMA21 cung chieu (momentum ngan han xac nhan)
        #   5. Higher lows / lower highs (10c vs 10c truoc) - di chuyen bac thang
        #   6. >= 7/12 nen cung chieu (gradual - khong phai 1 spike don le)
        # Khi detect: (a) moi flip/block NGUOC chieu trend nay bi chan,
        #             (b) signal nguoc chieu bi flip THEO trend (EMERGING override),
        #             (c) scenario engine tu tao entry theo trend neu strategies im lang
        _emerging_uptrend   = False
        _emerging_downtrend = False
        if not df_micro.empty and len(df_micro) >= 60:
            _emg_close = df_micro["close"]
            _emg_ema9  = _emg_close.ewm(span=9,  adjust=False).mean().iloc[-1]
            _emg_ema21 = _emg_close.ewm(span=21, adjust=False).mean().iloc[-1]
            _emg_ema50 = _emg_close.ewm(span=50, adjust=False).mean().iloc[-1]
            _emg_price = _range_live_price if _range_live_price > 0 else _emg_close.iloc[-1]
            _emg_low_recent  = df_micro["low"].iloc[-10:].min()
            _emg_low_prior   = df_micro["low"].iloc[-20:-10].min()
            _emg_high_recent = df_micro["high"].iloc[-10:].max()
            _emg_high_prior  = df_micro["high"].iloc[-20:-10].max()
            _emg_12c = df_micro.iloc[-12:]
            _emg_grn = int((_emg_12c["close"] > _emg_12c["open"]).sum())
            _emg_red = int((_emg_12c["close"] < _emg_12c["open"]).sum())

            _emerging_uptrend = (
                _ppd_lo_age >= 8
                and 0.005 * _sp <= _ppd_rise <= 0.025 * _sp
                and _emg_price > _emg_ema21 and _emg_price > _emg_ema50
                and _emg_ema9 > _emg_ema21
                and _emg_low_recent >= _emg_low_prior
                and _emg_grn >= 7
                and not _spike_pump_60c
            )
            _emerging_downtrend = (
                _ppd_hi_age >= 8
                and 0.005 * _sp <= _ppd_drop <= 0.025 * _sp
                and _emg_price < _emg_ema21 and _emg_price < _emg_ema50
                and _emg_ema9 < _emg_ema21
                and _emg_high_recent <= _emg_high_prior
                and _emg_red >= 7
                and not _spike_dump_60c
            )
            if _emerging_uptrend:
                logger.debug(f"{symbol}: EMERGING UPTREND - +{_ppd_rise*100:.2f}% tu day 60c ({_ppd_lo_age}c), higher lows, EMA9>21, price>EMA50")
            if _emerging_downtrend:
                logger.debug(f"{symbol}: EMERGING DOWNTREND - -{_ppd_drop*100:.2f}% tu dinh 60c ({_ppd_hi_age}c), lower highs, EMA9<21, price<EMA50")

            # Cau truc emerging PHU NHAN ket luan post-peak/trough nguoc chieu:
            # da reclaim EMA + higher lows = khong con "sustained decline" (va nguoc lai)
            if _emerging_uptrend and _post_peak_decline_long:
                _post_peak_decline_long = False
            if _emerging_downtrend and _post_trough_rise_short:
                _post_trough_rise_short = False

        if not df_micro.empty and len(df_micro) >= 15:
            _micro_atr    = compute_atr(df_micro).iloc[-1]
            _micro_bodies = (df_micro["close"].iloc[-15:].values - df_micro["open"].iloc[-15:].values)
            # Nguong 1.5x ATR (giam tu 2.0x): bat pump/dump vua duoi 2x ATR trong 15 nen
            _micro_spike_dump = any(b < -_micro_atr * 1.5 for b in _micro_bodies)
            _micro_spike_pump = any(b >  _micro_atr * 1.5 for b in _micro_bodies)
            # Current forming candle: block neu body 1m hien tai >= 0.4% (mid-pump/dump entry)
            # Bat cac truong hop vao lenh DANG GIUA pump - candle chua dong nen 2x ATR chua dat
            # SOXL/NEAR/HYPE/SNDK: gia tang 0.7-1.8% trong candle dang hinh thanh -> block LONG
            # Forming candle body check da xoa: body % block entry dung luc momentum manh nhat
            # 1.5x ATR spike check (ben tren) du de bat candle bat thuong that su
            # Dual spike: GIU CA 2 FLAG - ranging market = block moi momentum entry
            # (truoc day xoa flag de "let consensus decide" - nhung consensus khong loc duoc ranging)
            if _micro_spike_dump and _micro_spike_pump:
                logger.debug(f"{symbol}: dual spike (ranging 1m) - keep both flags, block all momentum")
            # Cumulative net move: 15-candle lookback, 0.8% threshold
            # Bat ca dump bat dau tu 15 phut truoc (truoc chi bat 10 phut)
            _close_15_ago = df_micro["close"].iloc[-15]
            _micro_price  = df_micro["close"].iloc[-1]
            if _close_15_ago > 0:
                _net_move = (_micro_price - _close_15_ago) / _close_15_ago
                # scalp_trend bypass: neu 5m xac nhan cung chieu -> la trend, khong phai spike
                # Tang nguong 0.8% -> 1.5%: 0.8% qua nho, block het cac coin dang trend binh thuong
                # 15 phut move 1.5% la spike/exhaustion, 0.8% la move thuong trong trend manh
                if _net_move < -0.015 * _sp and not _micro_spike_dump and scalp_trend != -1:
                    _micro_spike_dump = True
                    logger.debug(f"{symbol}: cumulative net dump {_net_move*100:.1f}% in 15 candles -> dump flag")
                elif _net_move > 0.015 * _sp and not _micro_spike_pump and scalp_trend != 1:
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
            # 0.6% da xoa vi qua nho - trong trend binh thuong 30c la 0.5-2%
            # Chi giu lai 15c cumulative (da check phia tren) va forming candle
            _live_check_price = _range_live_price if _range_live_price > 0 else _micro_price

            # Ghi nhan trang thai dual spike sau khi tat ca flags da duoc tinh
            _dual_spike = _micro_spike_dump and _micro_spike_pump

        # macro_trend / macro_4h / _atr_for_sl da tinh TRUOC range blocks (tren)

        # Strong trend flags - dung cho 30c range bypass va cac check sau
        # BREAKOUT: chay cho tat ca scan_list - su dung 1m signal data
        # (TAT theo config.ENABLE_BREAKOUT_PATH - chi trade trend-following)
        if config.ENABLE_BREAKOUT_PATH and df_signal is not None and not df_signal.empty and len(df_signal) >= 30:
            bo_sig = BREAKOUT_STRATEGY.generate_signal(df_signal, df_scalp, df_trend)
            if bo_sig.direction != 0:
                # 5m khong duoc nguoc chieu - cho phep sideways
                bo_ok = (
                    (bo_sig.direction == 1  and scalp_trend >= 0) or
                    (bo_sig.direction == -1 and scalp_trend <= 0)
                )
                # 1m micro-trend cung phai xac nhan
                micro_ok = (bo_sig.direction == 1 and micro_up) or (bo_sig.direction == -1 and micro_down)
                # 15m spike block
                post_spike_ok = not (bo_sig.direction == -1 and spike_was_dump and scalp_trend != -1) and \
                                not (bo_sig.direction == 1  and spike_was_pump and scalp_trend != 1)
                # BTC global trend filter cho BREAKOUT - dong bo voi momentum path
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
                # 1h range block cho BREAKOUT - EVAA type: pump spike len top 1h range
                bo_h1_ok = not (_h1_block_long and bo_sig.direction == 1) and \
                           not (_h1_block_short and bo_sig.direction == -1)
                # 24h exhaustion block cho BREAKOUT
                bo_24h_ok = not (_block_long_24h and bo_sig.direction == 1) and \
                            not (_block_short_24h and bo_sig.direction == -1)
                # Chi can 1h xac nhan trend la du cho breakout
                # Top5: cho phep breakout khi 1m confirm du 15m/1h chua flip
                # (tranh miss move nhu BTC bounce voi volume surge)
                _bo_macro_not_both_contra_long  = not (macro_trend == -1 and macro_4h == -1)
                _bo_macro_not_both_contra_short = not (macro_trend ==  1 and macro_4h ==  1)
                bo_trend_ok = (
                    (bo_sig.direction == 1  and macro_trend >= 1) or
                    (bo_sig.direction == -1 and macro_trend <= -1) or
                    (bo_sig.direction == 1  and macro_4h >= 1) or
                    (bo_sig.direction == -1 and macro_4h <= -1) or
                    (is_priority and bo_sig.direction == 1  and micro_up   and _bo_macro_not_both_contra_long) or
                    (is_priority and bo_sig.direction == -1 and micro_down and _bo_macro_not_both_contra_short)
                )
                # AEQ-12 cho BREAKOUT: fresh spike top/bottom - spike don le khong phai breakout that
                # BREAKOUT that: gia tang dan, break qua resistance co tich luy
                # Spike gia: 1-3 nen tang dot ngot len dinh -> KHONG phai breakout -> block
                bo_aeq12_ok = True
                if not df_micro.empty and len(df_micro) >= 15 and _range_live_price > 0:
                    _bo_low_30c  = df_micro["low"].iloc[-30:].min()  if len(df_micro) >= 30 else df_micro["low"].min()
                    _bo_high_30c = df_micro["high"].iloc[-30:].max() if len(df_micro) >= 30 else df_micro["high"].max()
                    _bo_high_10c = df_micro["high"].iloc[-10:].max()
                    _bo_low_10c  = df_micro["low"].iloc[-10:].min()
                    if _bo_low_30c > 0 and _bo_high_10c > 0 and _bo_high_30c > 0 and _bo_low_10c > 0:
                        _bo_ext_up   = (_range_live_price - _bo_low_30c) / _bo_low_30c
                        _bo_ext_down = (_bo_high_30c - _range_live_price) / _bo_high_30c
                        _bo_at_peak   = (_range_live_price / _bo_high_10c) >= 0.97
                        _bo_at_trough = (_range_live_price / _bo_low_10c)  <= 1.03
                        _bo_ext_thresh = 0.006   # 0.6% flat cho tat ca coin
                        _bo_grad_up = _is_gradual_uptrend   and scalp_trend == 1 and macro_trend == 1
                        _bo_grad_dn = _is_gradual_downtrend and scalp_trend == -1 and macro_trend == -1
                        if bo_sig.direction == 1 and _bo_at_peak and _bo_ext_up > _bo_ext_thresh and not _bo_grad_up:
                            bo_aeq12_ok = False
                            logger.debug(f"{symbol} [BREAKOUT] AEQ-12 block LONG: {_bo_ext_up*100:.1f}% above 30c low at 10c peak")
                        if bo_sig.direction == -1 and _bo_at_trough and _bo_ext_down > _bo_ext_thresh and not _bo_grad_dn:
                            bo_aeq12_ok = False
                            logger.debug(f"{symbol} [BREAKOUT] AEQ-12 block SHORT: {_bo_ext_down*100:.1f}% below 30c high at 10c trough")
                # AEQ-MULTIHR cho BREAKOUT: old 4h peak/trough - price o dinh pump nhieu gio
                bo_aeq_mh_ok = True
                if not df_micro.empty and len(df_micro) >= 120 and _range_live_price > 0:
                    _bo_n_mh    = min(240, len(df_micro))
                    _bo_high_mh = df_micro["high"].iloc[-_bo_n_mh:].max()
                    _bo_low_mh  = df_micro["low"].iloc[-_bo_n_mh:].min()
                    if _bo_high_mh > 0 and _bo_low_mh > 0 and _bo_high_mh > _bo_low_mh:
                        _bo_ext_up_mh   = (_range_live_price - _bo_low_mh)  / _bo_low_mh
                        _bo_ext_down_mh = (_bo_high_mh - _range_live_price) / _bo_high_mh
                        _bo_at_mh_peak   = 0.975 <= (_range_live_price / _bo_high_mh) <= 1.005
                        _bo_at_mh_trough = 0.995 <= (_range_live_price / _bo_low_mh)  <= 1.025
                        _bo_mh_thresh    = 0.015 * _sp
                        _bo_mh_peak_idx   = int(df_micro["high"].iloc[-_bo_n_mh:].values.argmax())
                        _bo_mh_trough_idx = int(df_micro["low"].iloc[-_bo_n_mh:].values.argmin())
                        _bo_peak_old   = _bo_mh_peak_idx   < (_bo_n_mh - 10)
                        _bo_trough_old = _bo_mh_trough_idx < (_bo_n_mh - 10)
                        if bo_sig.direction == 1 and _bo_ext_up_mh > _bo_mh_thresh and _bo_at_mh_peak and _bo_peak_old:
                            bo_aeq_mh_ok = False
                            logger.debug(f"{symbol} [BREAKOUT] AEQ-MULTIHR block LONG: at {_bo_n_mh}c peak {_bo_ext_up_mh*100:.1f}%")
                        if bo_sig.direction == -1 and _bo_ext_down_mh > _bo_mh_thresh and _bo_at_mh_trough and _bo_trough_old:
                            bo_aeq_mh_ok = False
                            logger.debug(f"{symbol} [BREAKOUT] AEQ-MULTIHR block SHORT: at {_bo_n_mh}c trough {_bo_ext_down_mh*100:.1f}%")
                # BREAKOUT theo dinh nghia la break khoi 2h/30c range top/bottom
                # -> KHONG ap dung 2h va 30c range block cho BREAKOUT (chung se block chinh xac diem breakout)
                # Chi giu 5h range block (macro overextension, dai han hon)
                bo_m2h_ok = True
                bo_30c_ok = True
                # Spike trong confirmed trend = sustained move, khong phai isolated spike
                _bo_pump_in_trend = micro_up and (_is_gradual_uptrend or (scalp_trend == 1 and macro_trend == 1))
                _bo_dump_in_trend = micro_down and (_is_gradual_downtrend or (scalp_trend == -1 and macro_trend == -1))
                micro_spike_ok = not (_micro_spike_pump and bo_sig.direction == 1 and not _bo_pump_in_trend) and \
                                 not (_micro_spike_dump and bo_sig.direction == -1 and not _bo_dump_in_trend)
                # [AEQ-VOL] BREAKOUT: volume pressure check - same logic as momentum path
                bo_vol_ok = True
                if not df_micro.empty and len(df_micro) >= 30:
                    _bo_vn = min(20, len(df_micro))
                    _bo_df_vs = df_micro.iloc[-_bo_vn:]
                    _bo_buy_vol  = _bo_df_vs.loc[_bo_df_vs["close"] > _bo_df_vs["open"], "volume"].sum()
                    _bo_sell_vol = _bo_df_vs.loc[_bo_df_vs["close"] < _bo_df_vs["open"], "volume"].sum()
                    _bo_tot = _bo_buy_vol + _bo_sell_vol
                    if _bo_tot > 0:
                        _bo_buy_press = _bo_buy_vol / _bo_tot
                        _bo_vol_base  = df_micro["volume"].iloc[-50:-20].mean() if len(df_micro) >= 50 else df_micro["volume"].mean()
                        _bo_vol_now   = df_micro["volume"].iloc[-20:].mean()
                        _bo_vol_weak  = _bo_vol_now < _bo_vol_base * 0.70
                        if bo_sig.direction == 1 and _m2h_pos > 0.60:
                            _bo_thresh = 0.45 if _bo_vol_weak else 0.38
                            if _bo_buy_press < _bo_thresh:
                                bo_vol_ok = False
                                logger.debug(f"{symbol} [BREAKOUT] AEQ-VOL block LONG: buy_pressure={_bo_buy_press:.0%} at 2h top {_m2h_pos:.0%}")
                        if bo_sig.direction == -1 and _m2h_pos < 0.40:
                            _bo_thresh = 0.55 if _bo_vol_weak else 0.62
                            if _bo_buy_press > _bo_thresh:
                                bo_vol_ok = False
                                logger.debug(f"{symbol} [BREAKOUT] AEQ-VOL block SHORT: buy_pressure={_bo_buy_press:.0%} at 2h bot {_m2h_pos:.0%}")
                # VOL-TREND guard cho BREAKOUT: breakout NGUOC volume-trend manh = false breakout
                # (break len nhung OBV+volume dang DOWN manh -> bull trap). Block, khong trade.
                _bo_vwt_ok = not (_vwt_dir != 0 and _vwt_str >= 0.45 and bo_sig.direction != _vwt_dir)
                if not _bo_vwt_ok:
                    logger.info(f"{symbol} [BREAKOUT] block: nguoc vol-trend manh (vwt={_vwt_dir} str={_vwt_str:.2f})")
                # EXHAUSTION: khong long dinh/gan dinh, khong short day/gan day - ke ca breakout
                # (BZ pattern: pump len 91.88 gan dinh 91.93 -> mua dinh -> lo). Chan neu sai phia.
                _bo_ez_price = _range_live_price if _range_live_price > 0 else (
                    df_micro["close"].iloc[-1] if not df_micro.empty else 0.0)
                _bo_ez_dir, _, _bo_ez_reason = self._exhaustion_check(
                    df_micro, bo_sig.direction, _bo_ez_price, rsi_now, _sp, vol_locked=True)
                _bo_ez_ok = (_bo_ez_dir == bo_sig.direction)
                if not _bo_ez_ok:
                    logger.info(f"{symbol} [BREAKOUT] block exhaustion: {_bo_ez_reason}")
                # Immediate momentum phai CUNG chieu breakout (khong long khi dang roi, short khi dang len)
                _bo_imm = self._immediate_momentum(df_micro, _sp)
                _bo_imm_ok = not ((bo_sig.direction == 1 and _bo_imm == -1) or (bo_sig.direction == -1 and _bo_imm == 1))
                if not _bo_imm_ok:
                    logger.info(f"{symbol} [BREAKOUT] block: immediate momentum nguoc chieu (imm={_bo_imm})")
                # Trend chu dao khong duoc nguoc chieu breakout (breakout gia = pha vo THEO trend)
                _bo_true = self._true_direction(macro_trend, macro_4h, _vwt_dir, _vwt_str, _bo_imm)
                _bo_true_ok = not (_bo_true != 0 and _bo_true != bo_sig.direction)
                if not _bo_true_ok:
                    logger.info(f"{symbol} [BREAKOUT] block: trend chu dao {_bo_true} nguoc breakout {bo_sig.direction}")
                # ANTI-CHOP: breakout chi hop le khi co TREND RO (ADX>=20), khong pha vo gia trong chop
                _bo_adx_ok = not (math.isnan(adx) or adx < 20.0)
                if not _bo_adx_ok:
                    logger.info(f"{symbol} [BREAKOUT] block: ADX={adx:.1f} < 20 (chop, khong phai breakout that)")
                if bo_ok and micro_ok and not is_spike and post_spike_ok and micro_spike_ok and bo_btc_ok and bo_h1_ok and bo_24h_ok and bo_trend_ok and bo_m2h_ok and bo_30c_ok and bo_aeq12_ok and bo_aeq_mh_ok and bo_vol_ok and _bo_vwt_ok and _bo_ez_ok and _bo_imm_ok and _bo_true_ok and _bo_adx_ok:
                    # micro_entry_analysis da xoa: BREAKOUT theo dinh nghia la break qua range
                    # -> range check trong _micro_entry_analysis se HARD BLOCK moi breakout hop le
                    # Da co: bo_h1_ok, bo_m2h_ok, bo_trend_ok, micro_ok thay the
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
        # 35/65 qua nhat - RSI 35-50 la binh thuong trong downtrend, khong phai reversal point
        # Dung 30/70: reversal chi khi RSI thuc su cuc doan (oversold/overbought ro rang)
        is_reversal  = rsi_now < 30 or rsi_now > 70
        reversal_dir = 1 if rsi_now < 30 else (-1 if rsi_now > 70 else 0)
        _is_range_rev = False  # triggered by 2h range extreme, not RSI

        # Range-extreme reversal: gia o day/dinh 2h range + micro momentum bat dau xoay chieu
        # BTC co the giam 0.85% ma RSI chi ve 35-40 (khong du RSI<30) nhung van la day range
        # -> bat lenh LONG o day, SHORT o dinh ma khong can RSI extreme
        # Dieu kien: _m2h_pos < 0.18 (bot 18%) hoac > 0.82 (top 82%), micro da xoay chieu
        if not is_reversal:
            if _m2h_pos < 0.18 and micro_up and scalp_trend >= 0:
                _is_range_rev = True
                is_reversal   = True
                reversal_dir  = 1
                logger.debug(f"{symbol}: range-extreme LONG trigger - 2h pos={_m2h_pos:.2f} < 0.18, micro_up")
            elif _m2h_pos > 0.82 and micro_down and scalp_trend <= 0:
                _is_range_rev = True
                is_reversal   = True
                reversal_dir  = -1
                logger.debug(f"{symbol}: range-extreme SHORT trigger - 2h pos={_m2h_pos:.2f} > 0.82, micro_down")

        # ====================== LO HONG NGHIEM TRONG (FIX) ======================
        # Reversal path DA TAT (ENABLE_REVERSAL_PATH=False) NHUNG is_reversal van =True khi
        # RSI cuc doan (RSI<30 = gan DAY, RSI>70 = gan DINH). Khi do coin roi xuong MOMENTUM
        # path, ma CA HAI lop bao ve deu co dieu kien 'not is_reversal':
        #   - Gate trend-following (macro-align + ADX + pullback + imm==T + breakout)
        #   - FINAL exhaustion guard (khong short day / khong long dinh)
        # -> BI BO QUA HET -> short o day (RSI<30) / long o dinh (RSI>70) chay thang ra lenh.
        # Day CHINH LA nguyen nhan 'short o day'. Reversal tat -> is_reversal KHONG duoc bypass:
        # ep ve momentum binh thuong de di qua DAY DU gate + exhaustion.
        if not config.ENABLE_REVERSAL_PATH:
            is_reversal   = False
            _is_range_rev = False
            reversal_dir  = 0

        long_signals  = []
        short_signals = []

        # Define BTC alignment flags BEFORE strategy loop
        # Chi ap dung BTC filter cho coin correlated voi BTC
        if symbol not in ("BTCUSDT", "ETHUSDT") and _btc_filter_on:
            btc_strongly_bull = (self.btc_trend == 1  and self.btc_trend_4h == 1)
            btc_strongly_bear = (self.btc_trend == -1 and self.btc_trend_4h == -1)
            # Sum <= -1: it nhat 1 TF bear va TF kia khong bull (vd: -1,0 hoac -1,-1)
            # Truoc day yeu cau ca 2 TF == -1 -> coin -1,0 bi BTC bull xoa short oan
            coin_independently_bull = (macro_trend + macro_4h) >= 1
            coin_independently_bear = (macro_trend + macro_4h) <= -1
        else:
            btc_strongly_bull = False
            btc_strongly_bear = False
            coin_independently_bull = False
            coin_independently_bear = False

        n_15m_valid = 0  # so strategies co signal hop le
        for strategy in ALL_STRATEGIES:
            try:
                # Tat ca strategies chay tren 1m data (2000 nen)
                # EMA dai hon (EMA9/21/50/100/250) tinh trend chuan xac hon multi-TF
                sig = strategy.generate_signal(df_signal, df_signal, df_signal)
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

                # Filter short_term_up/down (2/3 nen 1m) da xoa:
                # Strategies chay tren 15m, nen 1m co the do trong pullback binh thuong
                # Thay the boi scalp_allows_long/short (5m alignment) o phan ben duoi

                # Reversal bypass macro filter - bat day/dinh du macro nguoc
                if is_reversal:
                    if sig.direction == 1:
                        long_signals.append(sig)
                    elif sig.direction == -1:
                        short_signals.append(sig)
                else:
                    # Yeu cau TONG 2 TF phai net positive/negative:
                    # macro_trend + macro_4h >= 1: it nhat 1 TF up va TF kia khong bearish
                    # macro_trend=-1 + macro_4h=1 = 0 -> CONFLICT -> khong trade (USUSDT pattern)
                    # macro_trend=0  + macro_4h=1 = 1 -> ok (long-term up, short-term sideways)
                    # macro_trend=1  + macro_4h=0 = 1 -> ok (medium-term up, long-term sideways)
                    # macro_trend=1  + macro_4h=1 = 2 -> manh nhat
                    long_ok  = (macro_trend + macro_4h) >= 1
                    short_ok = (macro_trend + macro_4h) <= -1
                    # BTC strongly bear -> SHORT tat ca coin khong co xu huong doc lap UP
                    # BTC strongly bull -> LONG tat ca coin khong co xu huong doc lap DOWN
                    # BTC fast bounce (EMA20/50 UP) -> LONG ok du mid/macro BTC con DOWN
                    # (EMA coin chua kip flip nhung BTC da xac dinh xu huong ro -> trade theo BTC)
                    _btc_fast_bounce_ok = (self.btc_trend_fast == 1)
                    _btc_fast_dump_ok   = (self.btc_trend_fast == -1)
                    # BTC signal chi CONFIRM them cho coin da lean cung chieu - KHONG tao signal tu so khong
                    # Coin flat (0,0): BTC bearish KHONG du de short - coin phai tu no lean bearish truoc
                    # _coin_leans_bear: it nhat 1 TF bearish va TF kia khong bullish (sum <= -1)
                    # _coin_leans_bull: it nhat 1 TF bullish va TF kia khong bearish  (sum >= 1)
                    _coin_leans_bear = (macro_trend + macro_4h) <= -1
                    _coin_leans_bull = (macro_trend + macro_4h) >= 1
                    if btc_strongly_bear and not coin_independently_bull and _coin_leans_bear:
                        short_ok = True
                    if btc_strongly_bull and not coin_independently_bear and _coin_leans_bull:
                        long_ok = True
                    # Fast bounce: cho phep LONG/SHORT theo BTC fast move
                    # Coin lean cung chieu: ok (coin da co xu huong)
                    # Coin flat (0,0): chi ok neu 5m DA confirm cung chieu (scalp_trend == +/-1)
                    _btc_fast_coin_ok_bull = _coin_leans_bull or (scalp_trend == 1  and not coin_independently_bear)
                    _btc_fast_coin_ok_bear = _coin_leans_bear or (scalp_trend == -1 and not coin_independently_bull)
                    if _btc_fast_bounce_ok and not coin_independently_bear and _btc_fast_coin_ok_bull:
                        long_ok = True
                    if _btc_fast_dump_ok and not coin_independently_bull and _btc_fast_coin_ok_bear:
                        short_ok = True
                    # Early trend entry: cho phep SHORT/LONG khi 1m + 5m da confirm du 15m chua flip
                    # Guard: KHONG bypass khi CA HAI TF (macro_trend + macro_4h) deu oppose -
                    # dual-TF confirmed downtrend = khong phai "early flip", la downtrend that
                    if is_priority and sig.direction == -1 and not short_ok:
                        _macro_not_both_bull = not (macro_trend == 1 and macro_4h == 1)
                        if micro_down and scalp_trend == -1 and _macro_not_both_bull:
                            short_ok = True
                    if is_priority and sig.direction == 1 and not long_ok:
                        _macro_not_both_bear = not (macro_trend == -1 and macro_4h == -1)
                        if micro_up and scalp_trend == 1 and _macro_not_both_bear:
                            long_ok = True
                    # Micro-only entry: bat early trend khi ca 2 TF sideways (0,0) nhung 1m+5m ro chieu
                    # Dieu kien: ca 2 TF phai khong bearish/bullish (khong co conflict hoac downtrend)
                    # scalp >= 0: 5m neutral hoac cung chieu deu ok (early trend 5m chua flip)
                    if is_priority and sig.direction == 1 and not long_ok:
                        _macro_neither_bear = macro_trend >= 0 and macro_4h >= 0
                        if micro_up and scalp_trend >= 0 and _macro_neither_bear:
                            long_ok = True
                    if is_priority and sig.direction == -1 and not short_ok:
                        _macro_neither_bull = macro_trend <= 0 and macro_4h <= 0
                        if micro_down and scalp_trend <= 0 and _macro_neither_bull:
                            short_ok = True
                    # 5m alignment pre-filter: khong dem signal khi 5m nguoc chieu (ALTCOIN ONLY)
                    # BTC/ETH (largecap): 5m corrections trong 1h trend la BINH THUONG (buy dip / sell bounce)
                    # -> khong apply cho largecap, dung 1m micro check (micro_up/down + EMA9/21) thay the
                    # Altcoin: 5m bounce trong 1h downtrend = timing xau cho SHORT -> bo qua
                    if _is_largecap:
                        scalp_allows_short = True
                        scalp_allows_long  = True
                    else:
                        # Altcoin: 5m uu tien confirm nhung co nhieu bypass de khong bo miss lenh tot
                        # 1. 5m bearish/bullish xac nhan -> ok
                        # 2. 1m + 15m cung chieu -> ok (early trend, 5m chua flip)
                        # 3. Ca 15m VA 1h cung chieu -> ok (ca 2 TF lon xac nhan, 5m neutral chap nhan)
                        _micro_short_ok  = micro_down and macro_trend <= -1
                        _micro_long_ok   = micro_up   and macro_trend >= 1
                        _both_tf_bear    = macro_trend <= -1 and macro_4h <= -1
                        _both_tf_bull    = macro_trend >= 1  and macro_4h >= 1
                        # BTC strongly bear -> SHORT ok du scalp bounce tam (coin chua kip flip)
                        # BTC strongly bull -> LONG ok du scalp dip tam
                        # BTC fast bounce/dump -> bypass scalp filter theo huong ngan han
                        _btc_bear_scalp_ok    = btc_strongly_bear and not coin_independently_bull and _coin_leans_bear
                        _btc_bull_scalp_ok    = btc_strongly_bull and not coin_independently_bear and _coin_leans_bull
                        _btc_fast_long_scalp  = _btc_fast_bounce_ok and not coin_independently_bear and _btc_fast_coin_ok_bull
                        _btc_fast_short_scalp = _btc_fast_dump_ok  and not coin_independently_bull and _btc_fast_coin_ok_bear
                        # Micro-only bypass: khi long_ok/short_ok duoc set boi micro-only path
                        # (ca 2 TF sideways nhung 1m+5m confirm) thi scalp_allows phai nhat quan
                        # Bug cu: long_ok=True nhung scalp_allows_long=False vi scalp_trend=0, macro=0
                        _micro_only_long  = is_priority and micro_up   and macro_trend >= 0 and macro_4h >= 0 and scalp_trend >= 0
                        _micro_only_short = is_priority and micro_down and macro_trend <= 0 and macro_4h <= 0 and scalp_trend <= 0
                        scalp_allows_short = (scalp_trend == -1) or _micro_short_ok or _both_tf_bear or _btc_bear_scalp_ok or _btc_fast_short_scalp or _micro_only_short
                        scalp_allows_long  = (scalp_trend ==  1) or _micro_long_ok  or _both_tf_bull or _btc_bull_scalp_ok or _btc_fast_long_scalp  or _micro_only_long
                    if sig.direction == 1 and long_ok and scalp_allows_long:
                        long_signals.append(sig)
                    elif sig.direction == -1 and short_ok and scalp_allows_short:
                        short_signals.append(sig)
            except Exception:
                continue

        # REVERSAL trade: RSI cuc doan + 2 nen 15m + 1m micro xac nhan dao chieu + >= MIN_CONSENSUS
        # (TAT theo config.ENABLE_REVERSAL_PATH - bat dao chieu de bat dao roi -> tat de an toan)
        if config.ENABLE_REVERSAL_PATH and is_reversal and reversal_dir != 0:
            # 1h range (60 nen 1m = 60 phut): reversal long chi hop le khi price o BOTTOM 60% range
            # Tranh "catch dead cat bounce" khi price da phuc hoi nhieu tu day 1h
            _W1H = 60
            _rev_1h_blocked = False
            if not df_signal.empty and len(df_signal) >= _W1H:
                _s12_hi = df_signal["high"].iloc[-_W1H:].max()
                _s12_lo = df_signal["low"].iloc[-_W1H:].min()
                _s12_rng = _s12_hi - _s12_lo
                if _s12_rng > 0:
                    _s12_pos = (_range_price - _s12_lo) / _s12_rng
                    # LONG reversal chi hop le o bottom 1h (< 40%)
                    # SHORT reversal chi hop le o top 1h (> 60%)
                    if reversal_dir == 1 and _s12_pos > 0.40:
                        _rev_1h_blocked = True
                        logger.debug(f"{symbol}: reversal LONG blocked - 1h range_pos={_s12_pos:.2f} > 0.40 (not near 1h bottom)")
                    elif reversal_dir == -1 and _s12_pos < 0.60:
                        _rev_1h_blocked = True
                        logger.debug(f"{symbol}: reversal SHORT blocked - 1h range_pos={_s12_pos:.2f} < 0.60 (not near 1h top)")

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
                            f"{symbol}: reversal LONG blocked - 30c 1m range_pos={_r30_pos:.2f} > 0.65 "
                            f"(price already bounced, reversal stale)"
                        )
                    elif reversal_dir == -1 and _r30_pos < 0.35:
                        _rev_1h_blocked = True
                        logger.debug(
                            f"{symbol}: reversal SHORT blocked - 30c 1m range_pos={_r30_pos:.2f} < 0.35 "
                            f"(price already dumped, reversal stale)"
                        )

            # 2h extreme block cho REVERSAL:
            # LONG reversal hop le o BOTTOM 2h (< 35%) - tranh LONG khi price o mid/top 2h
            # SHORT reversal hop le o TOP 2h (> 65%) - tranh SHORT khi price o mid/bottom 2h
            # (Bo logic cu block LONG o dinh va SHORT o day - nguoc chieu reversal)
            if not _rev_1h_blocked:
                if reversal_dir == 1 and _m2h_pos > 0.50:
                    _rev_1h_blocked = True
                    logger.debug(
                        f"{symbol}: reversal LONG blocked - 2h range_pos={_m2h_pos:.2f} > 0.50 "
                        f"(price not at 2h bottom, reversal long invalid)"
                    )
                elif reversal_dir == -1 and _m2h_pos < 0.50:
                    _rev_1h_blocked = True
                    logger.debug(
                        f"{symbol}: reversal SHORT blocked - 2h range_pos={_m2h_pos:.2f} < 0.50 "
                        f"(price not at 2h top, reversal short invalid)"
                    )

            # Reversal spike block:
            # - LONG sau dump spike la NGUY HIEM (falling knife) -> block
            # - SHORT sau pump spike la HOP LE (ban dinh) -> KHONG block
            # - Range reversal: dump spike TAO RA day range -> LONG van ok (spike = diem dao chieu)
            reversal_spike_blocked = (
                (reversal_dir == 1  and ((_micro_spike_dump and not _is_range_rev) or _rev_1h_blocked)) or
                (reversal_dir == -1 and ((_micro_spike_pump and not _is_range_rev) or _rev_1h_blocked))
            )
            # Post-peak block cho reversal: EMA lag -> RSI oversold trong downtrend = false reversal
            # MAGMAUSDT pattern: RSI < 30 sau 40 phut giam, nhung gia van o tren day 2h thuc su
            # Chi cho phep neu price o extreme 2h bottom (< 20%) - day that su khong phai EMA lag
            if not reversal_spike_blocked and _post_peak_decline_long and reversal_dir == 1 and _m2h_pos >= 0.20:
                reversal_spike_blocked = True
                logger.info(
                    f"{symbol}: reversal LONG blocked - post-peak decline {_ppd_drop*100:.1f}% "
                    f"({_ppd_hi_age}c ago), 2h_pos={_m2h_pos:.2f} not at extreme bottom"
                )
            if not reversal_spike_blocked and _post_trough_rise_short and reversal_dir == -1 and _m2h_pos <= 0.80:
                reversal_spike_blocked = True
                logger.info(
                    f"{symbol}: reversal SHORT blocked - post-trough rise {_ppd_rise*100:.1f}% "
                    f"({_ppd_lo_age}c ago), 2h_pos={_m2h_pos:.2f} not at extreme top"
                )
            if not reversal_spike_blocked:
                # RSI slope check: reversal chi hop le khi RSI dang THUC SU dao chieu
                # RSI < 30 nhung van dang giam = falling knife; phai tang 3 nen lien tiep moi vao
                # RSI > 70 nhung van dang tang = dang pump; phai giam 3 nen lien tiep moi short
                _rsi_series = compute_rsi(df_signal["close"])
                _rsi_slope_ok = True
                if len(_rsi_series) >= 5:
                    _rsi_3ago = _rsi_series.iloc[-4]
                    _rsi_now2 = _rsi_series.iloc[-1]
                    if reversal_dir == 1 and _rsi_now2 < _rsi_3ago:
                        _rsi_slope_ok = False
                        logger.debug(f"{symbol}: reversal LONG blocked - RSI chua turn ({_rsi_now2:.1f} < {_rsi_3ago:.1f}, van giam)")
                    elif reversal_dir == -1 and _rsi_now2 > _rsi_3ago:
                        _rsi_slope_ok = False
                        logger.debug(f"{symbol}: reversal SHORT blocked - RSI chua turn ({_rsi_now2:.1f} > {_rsi_3ago:.1f}, van tang)")

                # reversal_micro_ok da xoa: o DINH micro_up=True, o DAY micro_down=True
                # => block reversal SHORT o dinh va LONG o day - nguoc y muon
                # Thay vao: chi can RSI slope turn + short_term confirm la du
                reversal_confirmed = _rsi_slope_ok and (
                    (reversal_dir == 1  and short_term_up)   or
                    (reversal_dir == -1 and short_term_down)
                ) and self._micro_entry_analysis(df_micro, reversal_dir, is_reversal=True)
                reversal_signals = long_signals if reversal_dir == 1 else short_signals
                reversal_base = config.MIN_CONSENSUS if is_priority else config.MIN_CONSENSUS_TRENDING
                # Deep trend guard: neu ca 1h VA 4h deu oppose reversal direction -> block neu khong o extreme 2h
                reversal_deep_opposed = (
                    (reversal_dir == 1  and macro_trend == -1 and macro_4h == -1) or
                    (reversal_dir == -1 and macro_trend ==  1 and macro_4h ==  1)
                )
                # DASH pattern: SHORT reversal khi macro UP (1+1=2) nhung price chi o 55% 2h range
                # -> pullback trong uptrend = KHONG PHAI DINH THAT -> block
                # Chi cho phep reversal chong macro khi price o EXTREME that su (>75% / <25% 2h range)
                if reversal_deep_opposed:
                    _rev_extreme_2h = (
                        (reversal_dir == -1 and _m2h_pos > 0.75) or
                        (reversal_dir == 1  and _m2h_pos < 0.25)
                    )
                    if not _rev_extreme_2h:
                        reversal_spike_blocked = True
                        logger.info(
                            f"{symbol}: reversal {'SHORT' if reversal_dir==-1 else 'LONG'} blocked "
                            f"- macro strongly {'UP' if macro_trend==1 else 'DOWN'} "
                            f"but 2h_pos={_m2h_pos:.2f} not at extreme (need {'<0.25' if reversal_dir==1 else '>0.75'})"
                        )
                reversal_min = reversal_base + (1 if reversal_deep_opposed else 0)
                # Range reversal: vi tri 2h extreme la xac nhan manh -> giam yeu cau 1 signal
                if _is_range_rev:
                    reversal_min = max(2, reversal_min - 1)
                if reversal_deep_opposed and not reversal_spike_blocked:
                    logger.debug(
                        f"{symbol}: reversal deep-trend guard +1 consensus "
                        f"(1h={'UP' if macro_trend==1 else 'DOWN'}, "
                        f"4h={'UP' if macro_4h==1 else 'DOWN'}, "
                        f"reversal={'LONG' if reversal_dir==1 else 'SHORT'}) "
                        f"-> need {reversal_min}/{len(ALL_STRATEGIES)}"
                    )
                # VOL-TREND guard cho REVERSAL: reversal la counter-trend tai RSI extreme
                # (mua day RSI<30 / ban dinh RSI>70). Nhung neu volume-trend VERY STRONG
                # nguoc chieu (locked-level 0.62) -> khong phai diem dao chieu ma la trend
                # manh dang chay -> catch dao chieu = bat dao roi (falling knife). Block.
                _rev_vwt_block = (_vwt_dir != 0 and _vwt_str >= 0.62 and reversal_dir != _vwt_dir)
                if _rev_vwt_block:
                    logger.info(
                        f"{symbol} [REVERSAL] block: volume-trend VERY STRONG nguoc chieu "
                        f"(vwt={_vwt_dir} str={_vwt_str:.2f} vs rev={reversal_dir}) - khong bat dao roi"
                    )
                if len(reversal_signals) >= reversal_min and reversal_confirmed and not _rev_vwt_block:
                    signals = reversal_signals
                    best = max(signals, key=lambda s: s.strength)
                    best.strength = min(0.95, best.strength + 0.15)
                    best.consensus = len(signals)
                    best.symbol    = symbol
                    # ATR override: dung 15m ATR cho SL/TP (1m ATR qua nho)
                    if _atr_for_sl > 0:
                        best.atr = _atr_for_sl
                    # TP size cho reversal: _is_range_rev (2h extreme) -> medium TP
                    # RSI-based reversal: depends on how extreme the RSI is
                    if _is_range_rev:
                        best.tp_roi_override = 0.12  # 12% ROI - range reversal, medium
                    elif rsi_now < 20 or rsi_now > 80:
                        best.tp_roi_override = 0.0   # RSI cuc doan manh -> TP lon (potential scaling)
                    else:
                        best.tp_roi_override = 0.12  # RSI vua du -> TP trung binh
                    # EXHAUSTION guard cho REVERSAL: reversal LONG phai o DAY (RSI<30), reversal
                    # SHORT phai o DINH (RSI>70) - dung nguyen tac. Guard chan truong hop nghich ly
                    # (reversal LONG lai o dinh / reversal SHORT lai o day) neu co.
                    _rv_ez_price = _range_live_price if _range_live_price > 0 else (
                        df_micro["close"].iloc[-1] if not df_micro.empty else 0.0)
                    _rv_ez_dir, _, _rv_ez_reason = self._exhaustion_check(
                        df_micro, best.direction, _rv_ez_price, rsi_now, _sp, vol_locked=True)
                    if _rv_ez_dir != best.direction:
                        logger.info(f"{symbol} [REVERSAL] block exhaustion: {_rv_ez_reason} - skip")
                        return False
                    # KHONG BAT DAO ROI: reversal LONG chi khi da co dau hieu XOAY (gia khong con
                    # roi manh), reversal SHORT khi khong con tang manh. APLD/RDW: RSI<30 giua
                    # downtrend manh -> mua dip -> dao roi -> lo. Doi immediate momentum khong con nguoc.
                    _rv_imm = self._immediate_momentum(df_micro, _sp)
                    if best.direction == 1 and _rv_imm == -1:
                        logger.info(f"{symbol} [REVERSAL] block LONG - gia dang roi manh (imm=-1), khong bat dao roi")
                        return False
                    if best.direction == -1 and _rv_imm == 1:
                        logger.info(f"{symbol} [REVERSAL] block SHORT - gia dang tang manh (imm=1), khong ban dinh dang len")
                        return False
                    # TREND CHU DAO: reversal la counter-trend, nhung neu trend chu dao con MANH
                    # nguoc chieu (macro + volume deu chong) -> dead-cat bounce, KHONG mua/ban.
                    # KORU: long tren bounce nho giua downtrend manh (macro down + vwt down) -> block.
                    # Reversal CHI hop le khi trend da suy yeu (true_dir trung tinh) - day/dinh THAT.
                    _rv_true = self._true_direction(macro_trend, macro_4h, _vwt_dir, _vwt_str, _rv_imm)
                    if _rv_true != 0 and _rv_true != best.direction:
                        logger.info(
                            f"{symbol} [REVERSAL] block - trend chu dao {_rv_true} con manh nguoc chieu "
                            f"reversal {best.direction} (macro={macro_trend}/{macro_4h} vwt={_vwt_dir}:{_vwt_str:.2f}) - dead-cat"
                        )
                        return False
                    names = "+".join(s.strategy_name for s in signals)
                    rsi_label = f"RSI15m={rsi_now:.0f}({'OVERSOLD<30' if reversal_dir==1 else 'OVERBOUGHT>70'})"
                    logger.info(
                        f"{symbol} [REVERSAL {rsi_label}] [{names}] -> "
                        f"{'LONG' if best.direction==1 else 'SHORT'} "
                        f"strength={best.strength:.2f} TP_override={best.tp_roi_override*100:.0f}% | {best.reason}"
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

        # RSI EXTREME GUARD: chi xoa signals khi RSI THUC SU cuc doan (25/75 thay vi 35/65)
        # RSI 35 qua som - RSI 35-50 la binh thuong trong downtrend, khong phai oversold
        # RSI < 25 moi la oversold that su (chi xay ra khi dump manh bat thuong)
        # RSI > 75 moi la overbought that su
        # RSI extreme guard: 75->80 (BTC uptrend RSI bam sat 75-85 suot gio)
        # Strong trend bypass: trong uptrend xac nhan, RSI cao la binh thuong
        if rsi_now < 25 and not _strong_trend_dn:
            short_signals = []
        elif rsi_now > 80 and not _strong_trend_up:
            long_signals = []

        # BTC GLOBAL TREND FILTER - HARD BLOCK khi ca 1h VA 4h BTC cung chieu
        # Neu BTC 1h+4h BULLISH -> xoa het SHORT signals (tat ca coin, ke ca priority SOL/ETH)
        # Neu BTC 1h+4h BEARISH -> xoa het LONG signals
        # Ngoai le: REVERSAL signal (RSI cuc doan) - reversal co the di nguoc BTC
        # Ngoai le: BTCUSDT chinh no - tu xu ly theo trend chinh no
        # Day la nguyen nhan chinh khien bot short SOL/WLD/ZEC/ADA khi BTC dang pump
        btc_trend    = self.btc_trend
        btc_trend_4h = self.btc_trend_4h

        btc_trend_fast = self.btc_trend_fast  # EMA(20/50) 1m ~20min trend - bat bounce/dip nhanh

        if symbol == "BTCUSDT":
            # BTCUSDT: symmetric fast-trend exception cho ca LONG va SHORT
            # Block SHORT khi mid+macro UP, NHUNG cho phep SHORT neu fast da flip DOWN (dang dump ngan han)
            if btc_trend == 1 and btc_trend_4h == 1 and btc_trend_fast != -1:
                short_signals = []
                logger.debug("BTCUSDT: clear SHORT - BTC mid+macro UP, fast not dumping")
            # Block LONG khi mid+macro DOWN, NHUNG cho phep LONG neu fast da flip UP (dang bounce ngan han)
            if btc_trend == -1 and btc_trend_4h == -1 and btc_trend_fast != 1:
                long_signals = []
                logger.debug("BTCUSDT: clear LONG - BTC mid+macro DOWN, fast not bouncing")
        elif symbol == "ETHUSDT":
            if macro_trend == 1 and macro_4h == 1:
                short_signals = []
                logger.debug("ETHUSDT: clear SHORT - ETH 1h AND 4h both UP")
            if macro_trend == -1 and macro_4h == -1:
                long_signals = []
                logger.debug("ETHUSDT: clear LONG - ETH 1h AND 4h both DOWN")
        else:
            # Altcoin: phan tich BTC alignment de quyet dinh hard/soft block
            # Chi ap dung neu coin correlated voi BTC
            if _btc_filter_on:
                btc_strongly_bull = (btc_trend == 1  and btc_trend_4h == 1)
                btc_strongly_bear = (btc_trend == -1 and btc_trend_4h == -1)
                coin_independently_bear = (macro_trend == -1 and macro_4h == -1)
                coin_independently_bull = (macro_trend ==  1 and macro_4h ==  1)
                btc_fast_bounce = (btc_trend_fast == 1)
                btc_fast_dump   = (btc_trend_fast == -1)
                if btc_strongly_bull and not is_reversal and not coin_independently_bear and not btc_fast_dump:
                    short_signals = []
                    logger.debug(f"{symbol}: BTC mid+macro BULLISH (fast not dumping), coin not independently bearish -> block SHORT")
                if btc_strongly_bear and not is_reversal and not coin_independently_bull and not btc_fast_bounce:
                    long_signals = []
                    logger.debug(f"{symbol}: BTC mid+macro BEARISH (fast not bouncing), coin not independently bullish -> block LONG")
            else:
                # Coin doc lap: khong ap dung BTC hard block, phan tich theo indicator rieng
                btc_strongly_bull = False
                btc_strongly_bear = False
                coin_independently_bull = False
                coin_independently_bear = False
                logger.debug(f"{symbol}: BTC uncorrelated -> skip BTC hard block, analyze independently")

        # BTC alignment flags cho consensus adjustment - chi dung neu coin correlated
        if _btc_filter_on and symbol != "BTCUSDT":
            btc_strongly_bull = (btc_trend == 1  and btc_trend_4h == 1)
            btc_strongly_bear = (btc_trend == -1 and btc_trend_4h == -1)
            coin_independently_bear = (macro_trend == -1 and macro_4h == -1)
            coin_independently_bull = (macro_trend ==  1 and macro_4h ==  1)
        else:
            btc_strongly_bull = False
            btc_strongly_bear = False
            coin_independently_bear = False
            coin_independently_bull = False

        # Soft penalty cho non-priority khi BTC 1 TF nguoc (chua confirm 2/2)
        btc_opposes_long  = (btc_trend == -1 and symbol != "BTCUSDT" and not is_priority and btc_trend_4h != -1)
        btc_opposes_short = (btc_trend ==  1 and symbol != "BTCUSDT" and not is_priority and btc_trend_4h != 1)

        # MOMENTUM trade
        sideways_1h = (macro_trend == 0)

        both_sideways = (macro_trend == 0 and macro_4h == 0)  # ca 2 TF sideways = thi truong ranging

        # BTC CHI LA THAM KHAO - khong duoc de BTC de "thuc luc" cua coin:
        #   - Bonus khi BTC CUNG chieu: giam 1 consensus (BTC xac nhan them = tham khao ho tro)
        #   - KHONG penalty khi coin diverge nguoc BTC: coin da co xu huong doc lap
        #     duoc confirm ca 2 TF (coin_independently_*) = thuc luc rieng da chung minh,
        #     trade theo chieu cua chinh coin - khong bat coin phai "xin phep" BTC
        #     (penalty cu +1 consensus lam miss lenh tiem nang tren coin trend doc lap)
        # VOLUME-TREND BONUS: coin co trend VOLUME ro (strength cao) -> giam consensus
        # yeu cau theo dung huong trend -> coin lon/trend ro de vao lenh hon (dung yeu cau
        # cua user: coin lon trend ro phai duoc trade, khong bo qua). Chi giam BEN trend.
        _vwt_long_bonus  = 0
        _vwt_short_bonus = 0
        if _vwt_dir == 1:
            _vwt_long_bonus  = 2 if _vwt_str >= 0.62 else (1 if _vwt_str >= 0.45 else 0)
        elif _vwt_dir == -1:
            _vwt_short_bonus = 2 if _vwt_str >= 0.62 else (1 if _vwt_str >= 0.45 else 0)

        if is_priority:
            # Priority: base = MIN_CONSENSUS, BTC cung chieu -> giam 1
            base = config.MIN_CONSENSUS
            extra = 0
            btc_long_bonus  = 1 if btc_strongly_bull else 0
            btc_short_bonus = 1 if btc_strongly_bear else 0
            required_long  = max(config.MIN_CONSENSUS, min(7, base + extra - btc_long_bonus  - _vwt_long_bonus))
            required_short = max(config.MIN_CONSENSUS, min(7, base + extra - btc_short_bonus - _vwt_short_bonus))
        else:
            # Non-priority: base = MIN_CONSENSUS_TRENDING
            base = config.MIN_CONSENSUS_TRENDING
            extra = 0
            btc_long_bonus  = 1 if btc_strongly_bull else 0
            btc_short_bonus = 1 if btc_strongly_bear else 0
            required_long  = max(config.MIN_CONSENSUS, min(7, base + extra - btc_long_bonus  - _vwt_long_bonus  + (1 if btc_opposes_long  else 0)))
            required_short = max(config.MIN_CONSENSUS, min(7, base + extra - btc_short_bonus - _vwt_short_bonus + (1 if btc_opposes_short else 0)))

        # TOP10 PRIORITY: 2 trong 2 Tier-1 strategy (supertrend + vwap_volume) dong thuan -> trade
        # Tier-1 bypass: KHONG bi chan boi BTC filter - top10 coin lon co momentum rieng
        # Tier1 bypass: 2/2 strategies dong thuan, khong bi chan boi bat ky extra filter nao
        TIER1 = {"supertrend", "vwap_volume"}
        tier1_long  = sum(1 for s in long_signals  if s.strategy_name in TIER1)
        tier1_short = sum(1 for s in short_signals if s.strategy_name in TIER1)

        # Tier1 bypass: chi khi ca 2 Tier1 cung chieu (2/2) va KHONG conflict
        # Neu conflict (1 long + 1 short): KHONG skip toan bo - van cho consensus check chay
        # [FIX] Tier1 bypass phai ton trong macro_4h alignment - tranh bypass trong reversal mode
        # khi signals vao tu reversal branch (khong co macro check)
        # Tier1 bypass: 2/2 strategies tier1 dong thuan
        # (0,0) flat coin: cho phep bypass neu KHONG co TF nao nguoc chieu (macro_trend >= 0 AND macro_4h >= 0)
        # Tranh bypass khi co TF dang chong lai (vd: -1,1 hoac 1,-1 = conflict)
        tier1_bypass_long  = (is_priority and tier1_long >= 2 and tier1_short == 0
                              and macro_trend >= 0 and macro_4h >= 0  # khong TF nao bearish
                              and not btc_strongly_bear)
        tier1_bypass_short = (is_priority and tier1_short >= 2 and tier1_long == 0
                              and macro_trend <= 0 and macro_4h <= 0  # khong TF nao bullish
                              and not btc_strongly_bull)

        _tier1_active   = False
        _scenario_entry = False   # entry tu SCENARIO ENGINE (khong qua strategy consensus)
        _scenario_name  = ""
        if tier1_bypass_long:
            signals = long_signals
            _tier1_active = True
            logger.info(f"{symbol}: [TIER1] 2/2 Tier-1 LONG - bypass consensus")
        elif tier1_bypass_short:
            signals = short_signals
            _tier1_active = True
            logger.info(f"{symbol}: [TIER1] 2/2 Tier-1 SHORT - bypass consensus")
        elif len(long_signals) >= required_long:
            signals = long_signals
        elif len(short_signals) >= required_short:
            signals = short_signals
        else:
            # SCENARIO ENGINE - TAT theo config.ENABLE_SCENARIO_PATH (chi trade trend-following)
            if not config.ENABLE_SCENARIO_PATH:
                return False
            # ANTI-CHOP cho SCENARIO: chi bat scenario khi co TREND RO (ADX>=20).
            if math.isnan(adx) or adx < 20.0:
                logger.debug(f"{symbol}: scenario skip - ADX={adx:.1f} < 20 (chop, khong trend)")
                return False
            # == SCENARIO ENGINE - bat lenh tiem nang khi strategies im lang ======
            # Strategies (EMA/RSI-based) co lag co huu - nhieu setup tiem nang RO RANG
            # tren cau truc gia khong duoc strategy nao bao (HBAR: uptrend moi tu day
            # nhung EMA dai van bearish -> 0 signal -> miss lenh win). Danh gia truc tiep
            # cac kich ban xac suat cao; MOI kich ban tu chua timing + position analysis:
            #   S1/S2 EMERGING TREND: trend moi hinh thanh tu day/dinh (HBAR = S1 LONG)
            #   S3/S4 BREAKOUT: pha vo 2h range + volume xac nhan, chua chay xa (khong chase)
            #   S5/S6 PULLBACK: hoi ve vung EMA21-50 trong trend da xac nhan roi resume
            #   S7/S8 RANGE EXTREME: cham day/dinh 2h range du rong -> mean revert ve giua
            _sc_dir      = 0
            _sc_strength = 0.0
            _sc_tp       = 0.0
            _sc_name     = ""
            if not df_micro.empty and len(df_micro) >= 120:
                _sc_close  = df_micro["close"]
                _sc_price  = _range_live_price if _range_live_price > 0 else _sc_close.iloc[-1]
                _sc_last_green = _sc_close.iloc[-1] > df_micro["open"].iloc[-1]
                _sc_last_red   = _sc_close.iloc[-1] < df_micro["open"].iloc[-1]
                _sc_hi_prior = df_micro["high"].iloc[-120:-3].max()   # range TRUOC 3 nen: break phai MOI
                _sc_lo_prior = df_micro["low"].iloc[-120:-3].min()
                _sc_hi_full  = df_micro["high"].iloc[-120:].max()
                _sc_lo_full  = df_micro["low"].iloc[-120:].min()
                _sc_rng      = _sc_hi_full - _sc_lo_full
                _sc_pos      = (_sc_price - _sc_lo_full) / _sc_rng if _sc_rng > 0 else 0.5
                _sc_rng_pct  = _sc_rng / _sc_lo_full if _sc_lo_full > 0 else 0.0
                _sc_vol3     = df_micro["volume"].iloc[-3:].mean()
                _sc_vol30    = df_micro["volume"].iloc[-30:-3].mean()
                _sc_vol_surge = (_sc_vol3 / _sc_vol30) if _sc_vol30 > 0 else 0.0
                _sc_bod5_pct = 0.0
                if _sc_price > 0:
                    _sc_bod5_pct = float(abs(df_micro["close"].iloc[-5:].values
                                             - df_micro["open"].iloc[-5:].values).max()) / _sc_price

                # S1/S2 - EMERGING TREND (uu tien cao nhat - chinh la HBAR pattern)
                if _emerging_uptrend and not _block_long_24h:
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.62, 0.0, "sc_emerging_up"
                elif _emerging_downtrend and not _block_short_24h:
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = -1, 0.62, 0.0, "sc_emerging_down"
                # S3/S4 - BREAKOUT 2h range + volume >= 1.5x, break 0.1-0.7% (khong chase),
                # khong co nen spike > 2%*_sp trong 5c, macro lon khong chong lai
                elif (_sc_hi_prior > 0 and _sc_hi_prior * 1.001 < _sc_price < _sc_hi_prior * 1.007
                      and _sc_vol_surge >= 1.5 and _sc_bod5_pct <= 0.020 * _sp
                      and macro_4h >= 0 and not _block_long_24h and _sc_last_green):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.65, 0.0, "sc_breakout_up"
                elif (_sc_lo_prior > 0 and _sc_lo_prior * 0.993 < _sc_price < _sc_lo_prior * 0.999
                      and _sc_vol_surge >= 1.5 and _sc_bod5_pct <= 0.020 * _sp
                      and macro_4h <= 0 and not _block_short_24h and _sc_last_red):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = -1, 0.65, 0.0, "sc_breakout_down"
                # S5/S6 - PULLBACK CONTINUATION: trend da xac nhan, gia hoi ve vung
                # EMA21-EMA50 1m roi co nen resume; RSI trung tinh (khong extreme)
                elif ((macro_trend == 1 and macro_4h >= 0) or (_is_gradual_uptrend and scalp_trend == 1)) \
                        and _emg_ema50 > 0 and _emg_ema50 * 0.999 <= _sc_price <= _emg_ema21 * 1.0015 \
                        and _sc_last_green and 35 <= rsi_now <= 65 and not _block_long_24h:
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.60, 0.0, "sc_pullback_up"
                elif ((macro_trend == -1 and macro_4h <= 0) or (_is_gradual_downtrend and scalp_trend == -1)) \
                        and _emg_ema21 > 0 and _emg_ema21 * 0.9985 <= _sc_price <= _emg_ema50 * 1.001 \
                        and _sc_last_red and 35 <= rsi_now <= 65 and not _block_short_24h:
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = -1, 0.60, 0.0, "sc_pullback_down"
                # S7/S8 - RANGE EXTREME mean-revert: range du rong (>=1.2%*_sp),
                # gia cham day/dinh (<=12% / >=88%), khong co trend manh/emerging NGUOC chieu,
                # nen cuoi xac nhan quay dau. TP nho - an giua range roi thoat
                elif (_sc_rng_pct >= 0.012 * _sp and _sc_pos <= 0.12
                      and not _strong_trend_dn and not _emerging_downtrend
                      and _sc_last_green and not _block_long_24h):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.58, 0.12, "sc_range_bottom"
                elif (_sc_rng_pct >= 0.012 * _sp and _sc_pos >= 0.88
                      and not _strong_trend_up and not _emerging_uptrend
                      and _sc_last_red and not _block_short_24h):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = -1, 0.58, 0.12, "sc_range_top"

            if _sc_dir == 0:
                return False
            _scenario_entry = True
            _scenario_name  = _sc_name
            signals = [Signal(
                direction=_sc_dir, strength=_sc_strength, strategy_name=_sc_name,
                entry_price=(_range_live_price if _range_live_price > 0 else price),
                atr=atr, reason=f"scenario:{_sc_name}", symbol=symbol,
                tp_roi_override=_sc_tp,
            )]
            logger.info(
                f"{symbol}: [SCENARIO] {_sc_name} -> {'LONG' if _sc_dir == 1 else 'SHORT'} "
                f"(strategies im lang, cau truc gia tu xac nhan)"
            )

        best = max(signals, key=lambda s: s.strength)

        def _block(reason: str) -> bool:
            """Log block reason at INFO if tier1 active, else DEBUG."""
            if _tier1_active:
                logger.info(f"{symbol}: [TIER1-BLOCKED] {reason}")
            else:
                logger.debug(f"{symbol}: {reason}")
            return False

        # True khi flip direction tai extrema/range extreme (peak/trough)
        # Dung de bypass micro_entry_analysis va AEQ-PUMP sau khi da quyet dinh flip
        _direction_flipped = False

        # ======================================================================
        # |  MASTER VOLUME-CONFIRMED TREND AUTHORITY - QUAN TOA TOI CAO         |
        # ======================================================================
        # Nguyen nhan HBAR (up+vol -> short) va HYPE (down -> long): signal sai chieu
        # van duoc trade vi khong co "quan toa" cuoi cung xac dinh trend bang VOLUME.
        # _volume_confirmed_trend ket hop OBV + directional volume + expansion +
        # cau truc gia -> khi STRONG va CLEAR, day la chan ly, flip moi thu nguoc chieu.
        #
        # 3 muc do:
        #   strength >= 0.62 (VERY STRONG): flip BAT KY signal nguoc chieu. Khoa
        #     _vol_trend_locked -> khong flip nguoc lai duoc nua (chong dao chieu ngu).
        #   strength >= 0.45 (STRONG): flip signal nguoc chieu (chua khoa cung).
        #   strength <  0.45: chi tham khao, khong ep.
        # TP scale theo strength: trend cang ro -> TP cang lon (an dam theo trend).
        # (_vwt_dir/_vwt_str da tinh som o tren - dung lai, khong goi 2 lan)
        _vol_trend_locked  = False
        if _vwt_dir != 0 and _vwt_str >= 0.45 and not is_reversal:
            # TP theo tiem nang trend: strength 0.45->0.20 ROI, 1.0->0.50 ROI
            _vwt_tp = 0.14 + (_vwt_str - 0.45) / 0.55 * 0.11
            _vwt_tp = max(0.12, min(0.25, _vwt_tp))
            if best.direction != _vwt_dir:
                # Signal NGUOC volume-trend -> flip THEO volume-trend (dung chieu that su)
                logger.info(
                    f"{symbol}: [VOL-TREND MASTER] flip {'LONG' if best.direction==1 else 'SHORT'}"
                    f"->{'LONG' if _vwt_dir==1 else 'SHORT'} | strength={_vwt_str:.2f} "
                    f"(OBV+vol+structure xac nhan {'UP' if _vwt_dir==1 else 'DOWN'}), TP={_vwt_tp*100:.0f}%"
                )
                best.direction = _vwt_dir
                best.tp_roi_override = _vwt_tp
                _direction_flipped = True
                if _vwt_str >= 0.62:
                    _vol_trend_locked = True
            else:
                # Signal CUNG chieu volume-trend -> boost strength (lenh tiem nang cao)
                # -> potential cao hon -> TP + von lon hon (TP/SL theo do tiem nang)
                best.strength = min(1.0, best.strength + 0.15 * _vwt_str)
                if _vwt_str >= 0.62:
                    _vol_trend_locked = True
                logger.debug(f"{symbol}: vol-trend CONFIRMS {'LONG' if _vwt_dir==1 else 'SHORT'} str={_vwt_str:.2f} -> boost")

        # BTC/ETH CORRELATION BLOCK: block neu pair kia da co position CUNG CHIEU
        # BTC va ETH correlated manh -> ca 2 cung SHORT = double loss khi bounce
        # Cho phep nguoc chieu (BTC long + ETH short = hedging, khac strategy)
        if btc_eth_side_map and symbol in ("BTCUSDT", "ETHUSDT"):
            pair = "ETHUSDT" if symbol == "BTCUSDT" else "BTCUSDT"
            pair_side = btc_eth_side_map.get(pair, "")
            # side tu Bybit: "Buy" = Long, "Sell" = Short
            signal_side = "Buy" if best.direction == 1 else "Sell"
            if pair_side == signal_side:
                return _block(f"BTC/ETH correlation: {pair} already {pair_side}, block {signal_side}")

        # Volume pressure filter
        if not df_micro.empty and len(df_micro) >= 5:
            _vbars = df_micro.iloc[-5:]
            _green_vol = _vbars.loc[_vbars["close"] >= _vbars["open"], "volume"].sum()
            _red_vol   = _vbars.loc[_vbars["close"] <  _vbars["open"], "volume"].sum()
            _total_vol = _green_vol + _red_vol
            if _total_vol > 0:
                _buy_ratio = _green_vol / _total_vol
                if best.direction == -1 and _buy_ratio >= 0.75:
                    return _block(f"skip SHORT - 5-bar volume pressure BUY {_buy_ratio*100:.0f}%")
                if best.direction == 1 and _buy_ratio <= 0.25:
                    return _block(f"skip LONG - 5-bar volume pressure SELL {(1-_buy_ratio)*100:.0f}%")

        # Dual spike (ranging market): block tat ca momentum entry
        if _dual_spike:
            return _block("skip - dual spike (ranging 1m choppy market), no momentum entry")

        # 1m spike filter - cho phep khi spike la phan cua confirmed uptrend/downtrend
        # Spike trong trend = sustained move (BTC pump 1%+ trong uptrend), khong phai isolated spike
        # QUAN TRONG: scalp_trend==1 don doc KHONG du - no co the bi push boi chinh cai spike do.
        # Phai co THEM macro_trend==1 (15m EMA) xac nhan trend ton tai truoc spike.
        _pump_spike_in_trend = micro_up and (_is_gradual_uptrend or _emerging_uptrend
                                             or (scalp_trend == 1 and macro_trend == 1))
        _dump_spike_in_trend = micro_down and (_is_gradual_downtrend or _emerging_downtrend
                                               or (scalp_trend == -1 and macro_trend == -1))
        # Breakout scenario: nen breakout > 1.5x ATR la BINH THUONG (da co volume + close
        # tren range confirm) - khong block nhu isolated spike
        if (_micro_spike_pump and best.direction == 1 and not _pump_spike_in_trend
                and _scenario_name != "sc_breakout_up"):
            return _block("skip - 1m pump spike (not in confirmed uptrend), no long")
        if (_micro_spike_dump and best.direction == -1 and not _dump_spike_in_trend
                and _scenario_name != "sc_breakout_down"):
            return _block("skip - 1m dump spike (not in confirmed downtrend), no short")

        # 1h range block (5h range extreme) -> flip direction thay vi block
        # Dinh/day 5h range = vi tri cuoi xu huong lon -> dao chieu, TP trung binh (co the tao dinh/day moi)
        # Emerging trend/scenario: gia len dinh 5h TRONG uptrend moi = breakout, KHONG flip nguoc
        if _h1_block_long and best.direction == 1 and not _emerging_uptrend and not _scenario_entry:
            best.direction = -1
            best.tp_roi_override = 0.15  # 15% ROI: dinh 5h tuong doi lon, TP medium
            _direction_flipped = True
            logger.info(f"{symbol}: 5h range flip LONG->SHORT at 5h top, TP=15%")
        if _h1_block_short and best.direction == -1 and not _emerging_downtrend and not _scenario_entry:
            best.direction = 1
            best.tp_roi_override = 0.15
            _direction_flipped = True
            logger.info(f"{symbol}: 5h range flip SHORT->LONG at 5h bottom, TP=15%")

        # 2h range block
        # Exception: gradual trend (>=18/30 nen cung chieu) + scalp xac nhan -> day/dinh 2h la DIEM BO QUA
        # BTC tang lien tuc 25 phut tao ra dinh 2h moi = gradual uptrend, khong phai pump da can kiet
        # Emerging trend: uptrend moi day gia len dinh 2h = trend dang chay, khong flip nguoc
        # Scenario entry: da tu phan tich vi tri (breakout tren dinh la chu dich) - khong flip
        _m2h_grad_bypass_long  = (_is_gradual_uptrend   and scalp_trend == 1  and macro_trend == 1) \
                                 or _emerging_uptrend or _scenario_entry
        _m2h_grad_bypass_short = (_is_gradual_downtrend and scalp_trend == -1 and macro_trend == -1) \
                                 or _emerging_downtrend or _scenario_entry
        if _m2h_block_short and best.direction == -1 and not _m2h_grad_bypass_short:
            # Day 2h range: flip SHORT->LONG, TP 12% (day lon, co the tao day moi nhung TP nho du co loi)
            best.direction = 1
            best.tp_roi_override = 0.12
            _direction_flipped = True
            logger.info(f"{symbol}: 2h range flip SHORT->LONG at 2h bottom ({_m2h_pos:.0%}), TP=12%")
        if _m2h_block_long and best.direction == 1 and not _m2h_grad_bypass_long:
            # Dinh 2h range: flip LONG->SHORT, TP 12% (dinh lon, co the tao dinh moi nhung TP nho du co loi)
            best.direction = -1
            best.tp_roi_override = 0.12
            _direction_flipped = True
            logger.info(f"{symbol}: 2h range flip LONG->SHORT at 2h top ({_m2h_pos:.0%}), TP=12%")

        # POST-PEAK / POST-TROUGH: EMA(100/250) lag sau khi gia qua dinh/day
        # MAGMAUSDT pattern: EMA con bullish nhung coin da giam 1%+ trong 40+ phut -> LONG = sai chieu
        # FLIP thay vi block: gia dang giam sustained sau dinh -> SHORT la lenh dung chieu
        # (nguyen tac: khong bo lenh tiem nang, doi chieu de trade theo trend thuc te)
        # Guard vi tri 2h range: KHONG short khi gia DA o day range (<30%) - short day la muon;
        # tuong tu KHONG long khi gia da o dinh range (>70%). Scenario entry tu phan tich - bo qua.
        if not _direction_flipped and not _scenario_entry:
            if best.direction == 1 and _post_peak_decline_long and _m2h_pos > 0.30:
                best.direction = -1
                best.tp_roi_override = 0.10
                _direction_flipped = True
                logger.info(
                    f"{symbol}: POST-PEAK flip LONG->SHORT - {_ppd_drop*100:.1f}% below 60c high "
                    f"({_ppd_hi_age}c ago, EMA lag) scalp={scalp_trend}, TP=10%"
                )
            elif best.direction == -1 and _post_trough_rise_short and _m2h_pos < 0.70:
                best.direction = 1
                best.tp_roi_override = 0.10
                _direction_flipped = True
                logger.info(
                    f"{symbol}: POST-TROUGH flip SHORT->LONG - {_ppd_rise*100:.1f}% above 60c low "
                    f"({_ppd_lo_age}c ago, EMA lag) scalp={scalp_trend}, TP=10%"
                )

        # == EMERGING TREND OVERRIDE - HBAR fix ===================================
        # Signal NGUOC chieu voi trend dang hinh thanh -> flip THEO trend:
        #   SHORT khi uptrend moi bat dau = ban ngay chan song len (HBAR -23%) -> doi thanh LONG
        #   LONG khi downtrend moi bat dau = mua ngay chan song xuong -> doi thanh SHORT
        # 24h exhausted (da pump/dump >20%): khong flip theo - block han de khong chase
        if not _direction_flipped and not _scenario_entry:
            if best.direction == -1 and _emerging_uptrend:
                if _block_long_24h:
                    return _block("skip SHORT - emerging uptrend (khong flip: 24h pump exhausted)")
                best.direction = 1
                best.tp_roi_override = 0.12
                _direction_flipped = True
                logger.info(
                    f"{symbol}: EMERGING-UP flip SHORT->LONG - trend moi tu day 60c "
                    f"(+{_ppd_rise*100:.1f}%, {_ppd_lo_age}c, higher lows + reclaim EMA), TP=12%"
                )
            elif best.direction == 1 and _emerging_downtrend:
                if _block_short_24h:
                    return _block("skip LONG - emerging downtrend (khong flip: 24h dump exhausted)")
                best.direction = -1
                best.tp_roi_override = 0.12
                _direction_flipped = True
                logger.info(
                    f"{symbol}: EMERGING-DOWN flip LONG->SHORT - trend moi tu dinh 60c "
                    f"(-{_ppd_drop*100:.1f}%, {_ppd_hi_age}c, lower highs + mat EMA), TP=12%"
                )

        # == SHORT-TERM TREND CONFIRMATION - 3 CAP DO ==========================
        #
        # Can nguyen lenh ngu: strategy chay tren 15m thay tin hieu (EMA lag)
        # nhung thuc te 1m dang GIAM RO RANG -> Long vao downtrend -> SL ngay
        #
        # Cap 1 - HARD BLOCK: 1m bearish (micro=-1) cho Long, bullish cho Short
        #   Khong co ngoai le. Gia dang giam = khong Long. Don gian vay thoi.
        #   (JASMY/MUSDT/HUMA deu vao day)
        #
        # Cap 2-3 - SOFT BLOCK: ca {1m, 5m} deu khong xac nhan
        #   Ngoai le: macro STRONG (ca 15m VA 1h cung chieu) -> pullback entry trong trend
        #   -> Cho phep Long khi 1m dang nghi (neutral) neu macro ro rang UP

        # --- Cap 1: HARD BLOCK (co ngoai le BTC/macro alignment) ---
        # Ngoai le: BTC strongly bear + coin macro bear -> SHORT trong micro bounce = ban dinh bounce hop le
        # Ngoai le: BTC strongly bull + coin macro bull -> LONG trong micro dip = mua day pullback hop le
        _btc_bear_short_ok = btc_strongly_bear and (macro_trend <= -1 or macro_4h <= -1)
        _btc_bull_long_ok  = btc_strongly_bull and (macro_trend >= 1  or macro_4h >= 1)

        # PUMP EXHAUSTION SHORT: khi LONG bi HARD BLOCK vi micro_down,
        # nhung gia vua pump (o phan tren 2h range) -> flip sang SHORT thay vi bo qua.
        # Day la "trade short va trade tre hon mot chut" - micro_down = xac nhan reversal bat dau.
        # Lenh DA flip / scenario entry: khong ap dung hard block + khong flip lan 2
        # (double-flip bug: POST-TROUGH flip SHORT->LONG roi PUMP-EXH flip lai LONG->SHORT
        #  = quay ve chieu ma phan tich truoc do da ket luan la SAI)
        _pump_exhaustion_flip = False
        if (best.direction == 1 and micro_down and not _btc_bull_long_ok
                and not _direction_flipped and not _scenario_entry):
            _pump_exh_30 = 0.0
            if not df_micro.empty and len(df_micro) >= 30:
                _p30  = df_micro["close"].iloc[-30]
                _pnow = df_micro["close"].iloc[-1]
                _pump_exh_30 = (_pnow - _p30) / _p30 if _p30 > 0 else 0.0
            _can_pump_exh_short = (
                _pump_exh_30 > 0.005        # +0.5% trong 30 nen = co pump xay ra truoc do
                and _m2h_pos > 0.55         # price o nua tren cua 2h range (vung dinh)
                and not _is_gradual_uptrend # khong phai uptrend lien tuc (do la continuation)
                and not (macro_trend == 1 and macro_4h == 1)  # khong co macro bull manh
                and len(signals) >= 2       # consensus >= 2
            )
            if _can_pump_exh_short:
                best.direction = -1
                _pump_exhaustion_flip = True
                logger.info(
                    f"{symbol}: PUMP-EXHAUSTION flip LONG->SHORT | "
                    f"pump30={_pump_exh_30*100:.1f}% m2h={_m2h_pos:.0%} "
                    f"is_gradual_up={_is_gradual_uptrend} scalp={scalp_trend}"
                )
                # Do NOT return - continue with direction=-1
            else:
                return _block(
                    f"HARD BLOCK LONG - 1m BEARISH (micro=-1, gia dang giam) "
                    f"| 5m={scalp_trend} 15m={macro_trend} 1h={macro_4h}"
                )
        # DUMP EXHAUSTION LONG: doi xung voi pump_exhaustion_flip
        # Khi SHORT bi HARD BLOCK vi micro_up nhung gia vua dump xuong day -> flip sang LONG
        _dump_exhaustion_flip = False
        if (best.direction == -1 and micro_up and not _btc_bear_short_ok
                and not _direction_flipped and not _scenario_entry):
            _dump_exh_30 = 0.0
            if not df_micro.empty and len(df_micro) >= 30:
                _p30d  = df_micro["close"].iloc[-30]
                _pnowd = df_micro["close"].iloc[-1]
                _dump_exh_30 = (_p30d - _pnowd) / _p30d if _p30d > 0 else 0.0
            _can_dump_exh_long = (
                _dump_exh_30 > 0.005         # da dump > 0.5% trong 30 nen
                and _m2h_pos < 0.45          # price o nua duoi cua 2h range (vung day)
                and not _is_gradual_downtrend
                and not (macro_trend == -1 and macro_4h == -1)
                and len(signals) >= 2
            )
            if _can_dump_exh_long:
                best.direction = 1
                _dump_exhaustion_flip = True
                logger.info(
                    f"{symbol}: DUMP-EXHAUSTION flip SHORT->LONG | "
                    f"dump30={_dump_exh_30*100:.1f}% m2h={_m2h_pos:.0%} scalp={scalp_trend}"
                )
            else:
                return _block(
                    f"HARD BLOCK SHORT - 1m BULLISH (micro=+1, gia dang tang) "
                    f"| 5m={scalp_trend} 15m={macro_trend} 1h={macro_4h}"
                )
        elif best.direction == -1 and micro_up and _btc_bear_short_ok:
            pass  # BTC bear context: SHORT khi micro bounce la hop le

        # --- Cap 2-3: SOFT BLOCK khi khong co TF ngan nao xac nhan ---
        # Macro STRONG = ca 15m VA 1h cung chieu -> pullback entry ok
        _strong_macro_bull = (macro_trend == 1  and macro_4h == 1)
        _strong_macro_bear = (macro_trend == -1 and macro_4h == -1)
        # Partial macro: macro_4h alone + scalp confirm du de cho phep entry
        _partial_macro_bull = macro_4h == 1  and scalp_trend == 1
        _partial_macro_bear = macro_4h == -1 and scalp_trend == -1
        _has_st_long  = (micro == 1  or scalp_trend == 1)
        _has_st_short = (micro == -1 or scalp_trend == -1)

        # Flip/scenario: da co phan tich cau truc rieng - khong doi hoi TF confirm (EMA lag)
        if not is_reversal and not _direction_flipped and not _scenario_entry:
            if best.direction == 1 and not _has_st_long and not _strong_macro_bull and not _partial_macro_bull and not _btc_bull_long_ok:
                return _block(
                    f"skip LONG - khong co TF ngan han xac nhan va macro khong manh "
                    f"(1m={micro}, 5m={scalp_trend}, 15m={macro_trend}, 1h={macro_4h})"
                )
            if best.direction == -1 and not _has_st_short and not _strong_macro_bear and not _partial_macro_bear and not _btc_bear_short_ok:
                return _block(
                    f"skip SHORT - khong co TF ngan han xac nhan va macro khong manh "
                    f"(1m={micro}, 5m={scalp_trend}, 15m={macro_trend}, 1h={macro_4h})"
                )

        if not _is_largecap:
            if best.direction == -1 and scalp_trend == -1 and (macro_trend + macro_4h) <= -1:
                _micro_spike_dump = False

        # EMA250 pullback filter tren 1m (EMA250 ~ EMA50 tren 5m = medium-term mean)
        if not df_signal.empty and len(df_signal) >= 255:
            _ema250_1m = compute_ema(df_signal["close"], 250).iloc[-1]
            _atr_1m    = _atr_for_sl if _atr_for_sl > 0 else compute_atr(df_signal, 50).iloc[-1]
            _p1m = df_signal["close"].iloc[-1]
            if _ema250_1m > 0 and _atr_1m > 0:
                _ema250_dist = _p1m - _ema250_1m
                # Strong trend bypass: BTC tang 5% -> price >4xATR khoi EMA250 la binh thuong
                if best.direction == 1 and _ema250_dist > 4.0 * _atr_1m and not _strong_trend_up:
                    return _block(f"skip LONG - price {_ema250_dist/_ema250_1m*100:.1f}% above 1m EMA250 ({_ema250_dist/(_atr_1m+1e-9):.1f}x ATR)")
                if best.direction == -1 and _ema250_dist < -4.0 * _atr_1m and not _strong_trend_dn:
                    return _block(f"skip SHORT - price {-_ema250_dist/_ema250_1m*100:.1f}% below 1m EMA250 ({-_ema250_dist/(_atr_1m+1e-9):.1f}x ATR)")

        # [AEQ-1] ATR spike
        if not df_micro.empty and len(df_micro) >= 35:
            _atr_ser_1m = compute_atr(df_micro, 14)
            _atr_cur_1m = _atr_ser_1m.iloc[-1]
            _atr_avg_1m = _atr_ser_1m.iloc[-21:-1].mean()
            if _atr_avg_1m > 0 and _atr_cur_1m > _atr_avg_1m * 2.5:
                _atr_strong_trend = is_priority and abs(macro_trend + macro_4h) >= 2
                if not _atr_strong_trend:
                    return _block(f"skip - ATR spike {_atr_cur_1m:.4f} > 2.5x avg {_atr_avg_1m:.4f}")

        # [AEQ-2] RSI divergence
        if not df_micro.empty and len(df_micro) >= 30:
            _rsi_1m_ser = compute_rsi(df_micro["close"], 14)
            _price_1m   = df_micro["close"]
            _p_early    = _price_1m.iloc[-20:-8]
            _p_recent   = _price_1m.iloc[-8:-1]
            _r_early    = _rsi_1m_ser.iloc[-20:-8]
            _r_recent   = _rsi_1m_ser.iloc[-8:-1]
            if len(_p_early) >= 5 and len(_p_recent) >= 5:
                if best.direction == 1 and _p_recent.max() > _p_early.max() and _r_recent.max() < _r_early.max() - 8:
                    return _block(f"skip LONG - bearish divergence RSI {_r_recent.max():.1f} < {_r_early.max():.1f}")
                if best.direction == -1 and _p_recent.min() < _p_early.min() and _r_recent.min() > _r_early.min() + 8:
                    return _block(f"skip SHORT - bullish divergence RSI {_r_recent.min():.1f} > {_r_early.min():.1f}")

        # [AEQ-3] Wick rejection
        if not df_micro.empty and len(df_micro) >= 3:
            _lc     = df_micro.iloc[-2]
            _lc_rng = _lc["high"] - _lc["low"]
            if _lc_rng > 0:
                _up_wick = _lc["high"] - max(_lc["open"], _lc["close"])
                _dn_wick = min(_lc["open"], _lc["close"]) - _lc["low"]
                if best.direction == 1 and _up_wick / _lc_rng > 0.85:
                    return _block(f"skip LONG - 1m wick rejection {_up_wick/_lc_rng*100:.0f}%")
                if best.direction == -1 and _dn_wick / _lc_rng > 0.85:
                    return _block(f"skip SHORT - 1m wick rejection {_dn_wick/_lc_rng*100:.0f}%")

        # [AEQ-4] Last 15m net body conflict - dung 15 nen 1m gan nhat (= 15 phut, tuong duong 1 nen 15m)
        # Tong body 15 nen 1m < -1.5x ATR = net bearish pressure manh khi muon long
        # [AEQ-4] Net body conflict - bypass khi strong trend (pullback trong uptrend la binh thuong)
        if not df_signal.empty and len(df_signal) >= 20 and _atr_for_sl > 0 and not is_reversal:
            _net_15m_body = (df_signal["close"].iloc[-15:] - df_signal["open"].iloc[-15:]).sum()
            if best.direction == 1 and _net_15m_body < -1.5 * _atr_for_sl and not _strong_trend_up:
                return _block(f"skip LONG - 15m net body strongly bearish ({_net_15m_body:.4f})")
            if best.direction == -1 and _net_15m_body > 1.5 * _atr_for_sl and not _strong_trend_dn:
                return _block(f"skip SHORT - 15m net body strongly bullish ({_net_15m_body:.4f})")

        # [AEQ-5a] Candle color: da xoa - qua chat, xu ly boi _micro_entry_analysis score
        # [AEQ-5b] EMA20 slope: da xoa - duplicate voi micro_up/down check

        # [AEQ-6] Funding period - CHI block khi trend YEU/khong ro
        # Funding fee ~0.01-0.05% = rat nho so voi move cua trend manh. Chan ca lenh
        # trend ro chi vi funding = bo lo lenh l"i lon. -> bypass khi vol-trend manh cung chieu
        # hoac scenario/flip (da co trend xac nhan). Thu hep window: 8->5 phut moi ben.
        _utc_now_f  = datetime.now(timezone.utc)
        _f_hour     = _utc_now_f.hour
        _f_min      = _utc_now_f.minute
        _near_funding_pre  = (_f_hour % 8 == 7 and _f_min >= 55)
        _near_funding_post = (_f_hour % 8 == 0 and _f_min <= 3)
        _funding_bypass = (
            (_vwt_dir == best.direction and _vwt_str >= 0.45)  # vol-trend manh cung chieu
            or _direction_flipped or _scenario_entry            # da co trend/scenario xac nhan
        )
        if (_near_funding_pre or _near_funding_post) and not _funding_bypass:
            return _block(f"skip - near funding window {_f_hour:02d}:{_f_min:02d} UTC (trend yeu)")

        # [AEQ-7] Volume near-zero
        if not df_micro.empty and len(df_micro) >= 20:
            _vol_r = df_micro["volume"].iloc[-5:].mean()
            _vol_p = df_micro["volume"].iloc[-15:-5].mean()
            if _vol_p > 0 and _vol_r < _vol_p * 0.20:
                return _block(f"skip - volume near-zero {_vol_r:.0f} < 20% of {_vol_p:.0f}")

        # [AEQ-8] Body deceleration near-zero - chi block khi HOAN TOAN dead (< 5%)
        # 10% qua nho: brief pause truoc breakout bi block oan
        if not df_micro.empty and len(df_micro) >= 15:
            _bd_r = abs(df_micro["close"].iloc[-4:-1] - df_micro["open"].iloc[-4:-1]).mean()
            _bd_p = abs(df_micro["close"].iloc[-11:-4] - df_micro["open"].iloc[-11:-4]).mean()
            if _bd_p > 0 and _bd_r < _bd_p * 0.05:
                return _block(f"skip - candle bodies near-zero {_bd_r:.4f} < 5% of {_bd_p:.4f}")

        # [AEQ-10] Stochastic extreme on 5m - chi block khi CUC DOAN (92/8)
        # 85/15 qua chat: trong uptrend manh, stochastic bam sat 80-95 lien tuc
        # Chi block khi > 92 hoac < 8 (thuc su exhaustion), va macro KHONG confirm
        if not df_scalp.empty and len(df_scalp) >= 14:
            _slo_low14 = df_scalp["low"].iloc[-14:].min()
            _slo_rng   = df_scalp["high"].iloc[-14:].max() - _slo_low14
            if _slo_rng > 0:
                _stoch_k      = ((df_scalp["close"].iloc[-1] - _slo_low14) / _slo_rng) * 100
                _macro_confirm = (macro_trend == best.direction and macro_4h == best.direction)
                # Emerging trend / breakout scenario: gia di len tu day 8+ nen -> stoch(14)
                # luon > 92 (dinh nghia cua rise) - block o day se giet chinh HBAR pattern.
                # Emerging/breakout da co cau truc xac nhan (higher lows, volume) - bo qua stoch
                _stoch_bypass_long  = _emerging_uptrend   or _scenario_name in ("sc_breakout_up", "sc_emerging_up")
                _stoch_bypass_short = _emerging_downtrend or _scenario_name in ("sc_breakout_down", "sc_emerging_down")
                if best.direction == 1 and _stoch_k > 92 and not _macro_confirm and not _stoch_bypass_long:
                    return _block(f"skip LONG - 5m Stochastic overbought K={_stoch_k:.1f}")
                if best.direction == -1 and _stoch_k < 8 and not _macro_confirm and not _stoch_bypass_short:
                    return _block(f"skip SHORT - 5m Stochastic oversold K={_stoch_k:.1f}")

        # [AEQ-11] Flat/ranging at top or bottom of 2h range: tranh Long khi gia flat o dinh (distribution)
        # ONDO pattern: price flat 30+ min o top range -> distribution zone -> Long bi SL
        # std < 0.15% cua mean = flat (price khong di chuyen dang ke trong 20 nen gan nhat)
        # FLIP thay vi block khi o CUC DOAN that su (>=80% / <=20% cua 2h range):
        #   flat o dinh = distribution -> SHORT dung chieu; flat o day = accumulation -> LONG dung chieu
        #   (nguyen tac: dinh phai short, day phai long - khong bo lenh tiem nang)
        # Vung giua (50-80% / 20-50%): van block - chua du gan dinh/day de flip tu tin
        if not is_reversal and not df_micro.empty and len(df_micro) >= 20:
            _close20  = df_micro["close"].iloc[-20:]
            _std20    = _close20.std()
            _mean20   = _close20.mean()
            if _mean20 > 0 and (_std20 / _mean20) < 0.0015:  # std < 0.15% = flat range
                if best.direction == 1 and _m2h_pos > 0.50 and not _scenario_entry:
                    if _m2h_pos >= 0.80 and not _direction_flipped:
                        best.direction = -1
                        best.tp_roi_override = 0.10
                        _direction_flipped = True
                        logger.info(
                            f"{symbol}: AEQ-11 flip LONG->SHORT - flat at 2h top ({_m2h_pos:.0%}), "
                            f"distribution zone, TP=10%"
                        )
                    else:
                        return _block(
                            f"skip LONG - flat at 2h top ({_m2h_pos:.0%}), "
                            f"std={_std20/_mean20*100:.3f}% (distribution zone)"
                        )
                elif best.direction == -1 and _m2h_pos < 0.50 and not _scenario_entry:
                    if _m2h_pos <= 0.20 and not _direction_flipped:
                        best.direction = 1
                        best.tp_roi_override = 0.10
                        _direction_flipped = True
                        logger.info(
                            f"{symbol}: AEQ-11 flip SHORT->LONG - flat at 2h bottom ({_m2h_pos:.0%}), "
                            f"accumulation zone, TP=10%"
                        )
                    else:
                        return _block(
                            f"skip SHORT - flat at 2h bottom ({_m2h_pos:.0%}), "
                            f"std={_std20/_mean20*100:.3f}% (accumulation zone)"
                        )

        # [AEQ-12] Pump-top / dump-bottom prevention (du dinh / du day toan dien)
        #
        # Bug cu (da xoa): dung 30c MEAN lam baseline -> mean bi keo len boi chinh cai pump
        # -> _move_from_mean nho -> khong trigger. Phai dung 30c LOW.
        #
        # 3 dieu kien CUNG XAY RA de block:
        #   1. Gia da tang X% khoi 30c LOW (co pump xay ra)
        #   2. Gia dang sat dinh 10c gan nhat (at the peak, not a pullback entry)
        #   3. KHONG co full multi-TF trend (5m + 15m + 1h deu phai bullish)
        #
        # Exception: ca 3 TF (scalp/macro/macro_4h) phai cung chieu = trend that su, ton tai truoc pump
        # BREAKOUT ap dung cung rule nhu cac strategy khac - single-candle spike van bi block
        _is_breakout_signal = any(s.strategy_name == "breakout" for s in signals)
        if (not df_micro.empty and len(df_micro) >= 15 and _range_live_price > 0
                and not is_reversal):
            _low_30c  = df_micro["low"].iloc[-30:].min()  if len(df_micro) >= 30 else df_micro["low"].min()
            _high_30c = df_micro["high"].iloc[-30:].max() if len(df_micro) >= 30 else df_micro["high"].max()
            _high_10c = df_micro["high"].iloc[-10:].max()
            _low_10c  = df_micro["low"].iloc[-10:].min()
            if _low_30c > 0 and _high_30c > 0 and _high_10c > 0 and _low_10c > 0:
                _ext_up   = (_range_live_price - _low_30c)  / _low_30c   # % above 30c low
                _ext_down = (_high_30c - _range_live_price) / _high_30c  # % below 30c high
                # "At peak" = within 3% of recent 10c high (not buying a dip, buying the spike top)
                _at_10c_peak   = (_range_live_price / _high_10c) >= 0.97
                _at_10c_trough = (_range_live_price / _low_10c)  <= 1.03
                _ext_thresh = 0.006   # 0.6% flat cho tat ca coin
                # Exception: gradual trend = gia tang DAN (>=18/30 nen xanh), KHONG phai spike dot ngot
                # BTC pattern: tang lien tuc 25 phut, moi nen xanh nho -> gradual uptrend -> cho phep LONG
                # Spike: 1-3 nen tang vot len dinh trong it phut -> phai block
                # Dieu kien bypass: gradual trend PHAI duoc xac nhan boi scalp_trend (5m) cung chieu
                # AEQ-12 bypass: phai co CA 3 dieu kien: gradual (khong spike) + 5m + 15m confirm
                # Truoc day chi can scalp_trend -> ZEC spike 1.4% bypass duoc -> LONG o dinh
                # Gio phai them macro_trend == 1: EMA100/250 tren 1m can nhieu gio moi flip
                # -> spike 10p KHONG the co macro_trend=1 -> bypass KHONG hoat dong voi spike
                _aeq12_bypass_long  = _is_gradual_uptrend   and scalp_trend == 1 and macro_trend == 1
                _aeq12_bypass_short = _is_gradual_downtrend and scalp_trend == -1 and macro_trend == -1

                # Volume exhaustion override: du bypass active, neu volume dang giam o peak/trough
                # -> move dang kiet suc -> force flip (SLX SHORT at trough: volume giam, dump het hoi)
                if not df_micro.empty and len(df_micro) >= 15:
                    _vol5  = df_micro["volume"].iloc[-5:].mean()
                    _vol15 = df_micro["volume"].iloc[-15:-5].mean()
                    _vol_exhausted = (_vol5 < _vol15 * 0.65) if _vol15 > 0 else False
                else:
                    _vol_exhausted = False

                # Emerging uptrend: "10c peak" chi la buoc tien cua trend moi (HBAR) - KHONG flip SHORT
                # _direction_flipped/_scenario_entry: da flip/da phan tich - khong flip lan 2 (double-flip bug)
                if (best.direction == 1 and _at_10c_peak and _ext_up > _ext_thresh
                        and not _direction_flipped and not _scenario_entry and not _emerging_uptrend):
                    # Bypass chi hop le neu KHONG co volume exhaustion tai dinh
                    _bypass_ok = _aeq12_bypass_long and not _vol_exhausted
                    if not _bypass_ok:
                        best.direction = -1
                        best.tp_roi_override = 0.08
                        _direction_flipped = True
                        logger.info(
                            f"{symbol}: AEQ-12 flip LONG->SHORT at 10c peak "
                            f"({_ext_up*100:.1f}% above 30c low, scalp={scalp_trend}, vol_exhausted={_vol_exhausted}), TP=8%"
                        )
                if (best.direction == -1 and _at_10c_trough and _ext_down > _ext_thresh
                        and not _direction_flipped and not _scenario_entry and not _emerging_downtrend):
                    # Bypass chi hop le neu KHONG co volume exhaustion tai day
                    _bypass_ok = _aeq12_bypass_short and not _vol_exhausted
                    if not _bypass_ok:
                        best.direction = 1
                        best.tp_roi_override = 0.08
                        _direction_flipped = True
                        logger.info(
                            f"{symbol}: AEQ-12 flip SHORT->LONG at 10c trough "
                            f"({_ext_down*100:.1f}% below 30c high, scalp={scalp_trend}, vol_exhausted={_vol_exhausted}), TP=8%"
                        )

        # [AEQ-EXTREMA] Local peak/trough detection tren 1m - chinh xac hon AEQ-12 don gian
        # Tim local extrema trong 60 nen gan nhat, xac dinh price dang o dinh hay day that su
        # Flip direction neu signal nguoc chieu extrema: peak + LONG -> SHORT, trough + SHORT -> LONG
        if not is_reversal and not df_micro.empty and len(df_micro) >= 15:
            _ex_window = 4  # window 4 nen moi ben = 9 nen tong de xac dinh peak/trough
            _ex_lookback = min(60, len(df_micro))
            df_ex = df_micro.iloc[-_ex_lookback:]
            _, _, _ex_peak_p, _ex_trough_p, _ex_since_peak, _ex_since_trough = \
                self._find_local_extrema(df_ex, window=_ex_window)
            _ex_price = _range_live_price if _range_live_price > 0 else df_micro["close"].iloc[-1]
            # "Near peak": within 1% of last local peak, peak made 3-30 candles ago (not too fresh, not too old)
            _ex_near_peak   = (_ex_price >= _ex_peak_p * 0.990) and (3 <= _ex_since_peak <= 30)
            _ex_near_trough = (_ex_price <= _ex_trough_p * 1.010) and (3 <= _ex_since_trough <= 30)

            if (_ex_near_peak and best.direction == 1 and not _aeq12_bypass_long
                    and not _direction_flipped and not _scenario_entry and not _emerging_uptrend):
                # Gia gan dinh local 1m, signal muon LONG -> co kha nang SHORT tot hon
                # Flip neu: price da tang du (>0.5% tu 10c low) va scalp khong phai bullish manh
                _ex_from_trough = (_ex_price - _ex_trough_p) / _ex_trough_p if _ex_trough_p > 0 else 0
                if _ex_from_trough > 0.005 and scalp_trend != 1:
                    best.direction = -1
                    best.tp_roi_override = 0.10
                    _direction_flipped = True
                    logger.info(
                        f"{symbol}: EXTREMA flip LONG->SHORT at local 1m peak "
                        f"price={_ex_price:.6f} peak={_ex_peak_p:.6f} +{_ex_from_trough*100:.2f}% "
                        f"(peak {_ex_since_peak}c ago), TP=10%"
                    )
                elif _ex_from_trough > 0.008:
                    # scalp bullish nhung van o dinh local: flip LONG->SHORT voi TP nho
                    best.direction = -1
                    best.tp_roi_override = 0.08
                    _direction_flipped = True
                    logger.info(
                        f"{symbol}: EXTREMA flip LONG->SHORT (scalp=up but at peak {_ex_since_peak}c ago, "
                        f"+{_ex_from_trough*100:.2f}%), TP=8%"
                    )

            elif (_ex_near_trough and best.direction == -1 and not _aeq12_bypass_short
                    and not _direction_flipped and not _scenario_entry and not _emerging_downtrend):
                # Gia gan day local 1m, signal muon SHORT -> co kha nang LONG tot hon
                _ex_from_peak = (_ex_peak_p - _ex_price) / _ex_peak_p if _ex_peak_p > 0 else 0
                if _ex_from_peak > 0.005 and scalp_trend != -1:
                    best.direction = 1
                    best.tp_roi_override = 0.10
                    _direction_flipped = True
                    logger.info(
                        f"{symbol}: EXTREMA flip SHORT->LONG at local 1m trough "
                        f"price={_ex_price:.6f} trough={_ex_trough_p:.6f} -{_ex_from_peak*100:.2f}% "
                        f"(trough {_ex_since_trough}c ago), TP=10%"
                    )
                elif _ex_from_peak > 0.008:
                    # scalp bearish nhung van o day local: flip SHORT->LONG voi TP nho
                    best.direction = 1
                    best.tp_roi_override = 0.08
                    _direction_flipped = True
                    logger.info(
                        f"{symbol}: EXTREMA flip SHORT->LONG (scalp=down but at trough {_ex_since_trough}c ago, "
                        f"-{_ex_from_peak*100:.2f}%), TP=8%"
                    )

        # -- AEQ-MULTIHR: Multi-hour range check (240c ~ 4h on 1m data) ---------
        # AEQ-12 chi nhin 30c (~30 phut) - khong phat hien "dang o dinh cua pump nhieu gio"
        # SUI pattern: pump len 0.7739 luc 11:00, bot van LONG tai 0.769 luc 13:53 (sau 3h)
        # Fix: neu gia dang gan dinh 240c VA dinh do duoc tao ra > 10 candles truoc
        #      -> dang o vung dinh (khong phai fresh breakout) -> block LONG / block SHORT
        # Khong co _full_up_trend exception: du trend align, vao LONG sat dinh 4h = timing xau
        if (not df_micro.empty and len(df_micro) >= 120 and _range_live_price > 0
                and not is_reversal):
            _n_mh     = min(240, len(df_micro))
            _high_mh  = df_micro["high"].iloc[-_n_mh:].max()
            _low_mh   = df_micro["low"].iloc[-_n_mh:].min()
            if _high_mh > 0 and _low_mh > 0 and _high_mh > _low_mh:
                _ext_up_mh   = (_range_live_price - _low_mh) / _low_mh
                _ext_down_mh = (_high_mh - _range_live_price) / _high_mh
                # "At peak": within 2.5% BELOW 4h high, but NOT above it (would be breakout)
                _at_mh_peak   = 0.975 <= (_range_live_price / _high_mh) <= 1.005
                _at_mh_trough = 0.995 <= (_range_live_price / _low_mh)  <= 1.025
                _mh_thresh = 0.015 * _sp   # 0.75% BTC/ETH, 1.125% midcap, 1.5% altcoin
                # Peak/trough "old" = made > 10 candles ago -> not a fresh current breakout
                _mh_peak_idx   = int(df_micro["high"].iloc[-_n_mh:].values.argmax())
                _mh_trough_idx = int(df_micro["low"].iloc[-_n_mh:].values.argmin())
                _mh_peak_is_old   = _mh_peak_idx   < (_n_mh - 10)
                _mh_trough_is_old = _mh_trough_idx < (_n_mh - 10)

                # Guards: khong flip lenh DA flip/scenario (double-flip), khong flip nguoc emerging trend
                if (best.direction == 1 and _ext_up_mh > _mh_thresh and _at_mh_peak and _mh_peak_is_old
                        and not _direction_flipped and not _scenario_entry and not _emerging_uptrend):
                    # Dinh lon 4h (major peak): flip LONG->SHORT voi TP lon (dao chieu lon)
                    best.direction = -1
                    best.tp_roi_override = 0.0  # 0 = large TP via potential scaling (dinh lon = TP lon)
                    _direction_flipped = True
                    logger.info(
                        f"{symbol}: MULTIHR flip LONG->SHORT at {_n_mh}c major peak "
                        f"({_ext_up_mh*100:.1f}% above {_n_mh}c low, {_n_mh-_mh_peak_idx}c ago), TP=large"
                    )
                elif (best.direction == -1 and _ext_down_mh > _mh_thresh and _at_mh_trough and _mh_trough_is_old
                        and not _direction_flipped and not _scenario_entry and not _emerging_downtrend):
                    # Day lon 4h (major trough): flip SHORT->LONG voi TP lon
                    best.direction = 1
                    best.tp_roi_override = 0.0  # large TP
                    _direction_flipped = True
                    logger.info(
                        f"{symbol}: MULTIHR flip SHORT->LONG at {_n_mh}c major trough "
                        f"({_ext_down_mh*100:.1f}% below {_n_mh}c high, {_n_mh-_mh_trough_idx}c ago), TP=large"
                    )

        # [AEQ-VOL] Volume pressure: xac nhan momentum con du fuel hay da can kiet
        # Neu gia o dinh/day nhung volume khong confirm -> exhaustion -> block
        # Logic:
        #   buy_pressure  = buy_vol / (buy_vol + sell_vol) tren 20 nen 1m gan nhat
        #   vol_declining = vol 20c gan nhat < 70% vol 30c truoc do (fuel dang can)
        # LONG block: price o top 60% range 2h MA buy_pressure < 0.40 (sellers chiem uu)
        # SHORT block: price o bot 40% range 2h MA buy_pressure > 0.60 (buyers chiem uu)
        # Tang cu khi ca 2: o extreme VA volume declining -> nguong that chat hon (0.45/0.55)
        if not is_reversal and not df_micro.empty and len(df_micro) >= 30:
            _vn = min(20, len(df_micro))
            _df_vs = df_micro.iloc[-_vn:]
            _buy_vol  = _df_vs.loc[_df_vs["close"] > _df_vs["open"], "volume"].sum()
            _sell_vol = _df_vs.loc[_df_vs["close"] < _df_vs["open"], "volume"].sum()
            _tot_vol  = _buy_vol + _sell_vol
            if _tot_vol > 0:
                _buy_press = _buy_vol / _tot_vol
                # Vol trend: recent 20c vs prior 30c
                _vol_base = df_micro["volume"].iloc[-50:-20].mean() if len(df_micro) >= 50 else df_micro["volume"].mean()
                _vol_now  = df_micro["volume"].iloc[-20:].mean()
                _vol_weak = _vol_now < _vol_base * 0.70  # recent vol < 70% baseline

                # LONG: neu price o top 2h range (>60%) ma sellers dang chiem uu -> flip SHORT
                # Volume xac nhan sellers -> SHORT voi TP nho (co the tao dinh moi nhung SHORT co loi)
                # Guards: khong flip lenh da flip/scenario; khong flip nguoc EMERGING uptrend
                # (HBAR: gia len tu day -> pos vuot 0.60 som -> flip nguoc = ban chan song len)
                if (best.direction == 1 and _m2h_pos > 0.60
                        and not _direction_flipped and not _scenario_entry and not _emerging_uptrend):
                    _thresh = 0.45 if _vol_weak else 0.38
                    if _buy_press < _thresh:
                        best.direction = -1
                        best.tp_roi_override = 0.08  # 8% ROI: volume ko manh, uncertain
                        _direction_flipped = True
                        logger.info(
                            f"{symbol}: AEQ-VOL flip LONG->SHORT: buy_pressure={_buy_press:.0%} < {_thresh:.0%} "
                            f"at 2h top {_m2h_pos:.0%} (vol_weak={_vol_weak}), TP=8%"
                        )
                # SHORT: neu price o bot 2h range (<40%) ma buyers dang chiem uu -> flip LONG
                # Volume xac nhan buyers -> LONG voi TP nho
                if (best.direction == -1 and _m2h_pos < 0.40
                        and not _direction_flipped and not _scenario_entry and not _emerging_downtrend):
                    _thresh = 0.55 if _vol_weak else 0.62
                    if _buy_press > _thresh:
                        best.direction = 1
                        best.tp_roi_override = 0.08  # 8% ROI
                        _direction_flipped = True
                        logger.info(
                            f"{symbol}: AEQ-VOL flip SHORT->LONG: buy_pressure={_buy_press:.0%} > {_thresh:.0%} "
                            f"at 2h bot {_m2h_pos:.0%} (vol_weak={_vol_weak}), TP=8%"
                        )

        # ======================================================================
        # |  RE-ASSERT MASTER VOLUME-TREND LOCK - chot chan cuoi cung           |
        # ======================================================================
        # Neu volume-trend VERY STRONG (locked): TUYET DOI khong flip nguoc lai.
        # Bat ky flip site nao o tren (AEQ-VOL/MULTIHR/EXTREMA...) lo dao chieu
        # nguoc volume-trend manh -> khoi phuc lai dung chieu + TP theo trend.
        # Day la nguyen nhan HBAR: uptrend manh nhung "at peak" flip -> short -> lo.
        if _vol_trend_locked and best.direction != _vwt_dir:
            logger.info(
                f"{symbol}:  VOL-TREND LOCK re-assert -> khoi phuc {'LONG' if _vwt_dir==1 else 'SHORT'} "
                f"(mot flip site da dao nguoc trend manh str={_vwt_str:.2f})"
            )
            best.direction = _vwt_dir
            _vwt_tp2 = 0.14 + (_vwt_str - 0.40) / 0.60 * 0.11
            best.tp_roi_override = max(0.12, min(0.25, _vwt_tp2))
            _direction_flipped = True

        # MOMENTUM GATE: LUON goi micro_entry_analysis cho tat ca momentum trade
        # Tranh vao lenh khi 1m dang di nguoc chieu (JASMY Long trong downtrend, v.v.)
        # Tier1 bypass KHONG duoc mien kieu tra nay - timing xau van la timing xau du consensus cao
        # Direction flip (tai extrema/range): bypass gate - da co range/extrema analysis lam timing
        # Flip entry la counter-trend, micro_entry_analysis se reject do EMA/momentum nguoc chieu
        _strong_trend = (best.direction == 1 and _strong_trend_up) or (best.direction == -1 and _strong_trend_dn)

        # Scenario entry: kich ban da tu chua timing analysis (resume candle, break+volume, ...)
        if not _direction_flipped and not _scenario_entry:
            if not self._micro_entry_analysis(df_micro, best.direction, is_reversal=False, strong_trend=_strong_trend):
                return _block(f"skip - MOMENTUM micro_entry_analysis rejected (consensus={len(signals)})")

        # [AEQ-PUMP] Live price vs 5-candle average: tranh du dinh / du day
        # Block 4 truong hop:
        #   1. SHORT vao giua pump dang chay (SHORT qua som)
        #   2. LONG vao giua dump dang chay (LONG qua som)
        #   3. LONG khi gia DA pump roi (du dinh - PHAUSDT pattern)
        #   4. SHORT khi gia DA dump roi (du day)
        # Nguong: _sp-scale -> largecap 0.15%, altcoin 0.3% - scaled by volatility class
        if _range_live_price > 0 and not df_micro.empty and len(df_micro) >= 6:
            _avg_5c = df_micro["close"].iloc[-6:-1].mean()
            if _avg_5c > 0:
                _live_move_pct = (_range_live_price - _avg_5c) / _avg_5c
                _pump_thresh = 0.008 if _sp < 1.0 else 0.010   # 0.8% largecap+midcap, 1.0% altcoin
                if best.direction == -1 and _live_move_pct > _pump_thresh:
                    # Ngoai le: pump exhaustion flip hoac direction flip tai extrema hoac scenario
                    # Direction flip tai dinh: gia dang cao = dung dieu kien SHORT -> khong block
                    if not _pump_exhaustion_flip and not _direction_flipped and not _scenario_entry:
                        return _block(
                            f"skip SHORT - live {_live_move_pct*100:.2f}% above 5c avg "
                            f"(gia dang pump, Short qua som)"
                        )
                if best.direction == 1 and _live_move_pct < -_pump_thresh:
                    # Direction flip tai day: gia dang thap = dung dieu kien LONG -> khong block
                    if not _direction_flipped and not _scenario_entry:
                        return _block(
                            f"skip LONG - live {_live_move_pct*100:.2f}% below 5c avg "
                            f"(gia dang dump, Long qua som)"
                        )
                # Block du dinh / du day: gia da di xa roi moi vao theo
                # Ngoai le: dang trong confirmed trend HOAC emerging trend (trend moi - HBAR)
                _long_in_trend  = micro_up   and (_is_gradual_uptrend   or _emerging_uptrend
                                                  or (scalp_trend == 1  and macro_trend == 1))
                _short_in_trend = micro_down and (_is_gradual_downtrend or _emerging_downtrend
                                                  or (scalp_trend == -1 and macro_trend == -1))
                if (best.direction == 1 and _live_move_pct > _pump_thresh and not _long_in_trend
                        and not _direction_flipped and not _scenario_entry):
                    return _block(
                        f"skip LONG - live {_live_move_pct*100:.2f}% above 5c avg "
                        f"(gia da pump, Long du dinh)"
                    )
                if (best.direction == -1 and _live_move_pct < -_pump_thresh and not _short_in_trend
                        and not _direction_flipped and not _scenario_entry):
                    return _block(
                        f"skip SHORT - live {_live_move_pct*100:.2f}% below 5c avg "
                        f"(gia da dump, Short du day)"
                    )

        best.consensus = len(signals)
        best.symbol    = symbol
        # ATR override: dung 15m ATF cho SL/TP - 1m ATR qua nho (noise se hit SL lien tuc)
        if _atr_for_sl > 0:
            best.atr = _atr_for_sl

        # DYNAMIC TP theo kich thuoc dinh/day:
        # Short-term extrema (dinh/day ngan han, co the tao dinh/day moi):
        #   -> TP nho: du bu phi san + loi nho -> dong lenh nhanh, bat lenh tiep theo
        # Major extrema (dinh/day lon cua 2h/4h range):
        #   -> TP binh thuong/lon
        #
        # Fee break-even: ROUND_TRIP_FEE x leverage = % roi tren margin
        # Leverage tra ve tu get_max_leverage (se goi lai trong compute_trade)
        # De don gian, uoc tinh leverage = 50x -> fee_roi ~ 5.5%, safe TP = fee + 5% = ~10%
        # Voi leverage thap hon (20x), fee_roi = 2.2% -> safe TP = 7-8%
        # -> Dung 8% ROI lam TP nho (an toan voi moi leverage tu 10x tro len)
        _tp_small  = 0.08   # 8% ROI - TP nho cho short-term extrema
        _tp_medium = 0.15   # 15% ROI - TP trung binh
        _tp_large  = 0.0    # 0 = dung potential scaling binh thuong (12-60%)

        # Xac dinh kich thuoc extrema tu cac bien da tinh truoc do
        # _ex_near_peak/_ex_near_trough: da xac dinh trong AEQ-EXTREMA block
        # _m2h_pos: vi tri trong 2h range
        # _ext_up_mh/_ext_down_mh: % tu day/dinh 4h (da tinh trong AEQ-MULTIHR)
        _is_major_peak   = False
        _is_major_trough = False
        try:
            _is_major_peak   = _m2h_pos > 0.75 or (_ext_up_mh   > 0.02 and _at_mh_peak)
            _is_major_trough = _m2h_pos < 0.25 or (_ext_down_mh  > 0.02 and _at_mh_trough)
        except Exception:
            pass  # bien chua duoc tinh (phan AEQ-MULTIHR chua chay)

        _is_short_term_extrema = False
        try:
            _is_short_term_extrema = (_ex_near_peak and best.direction == -1) or \
                                     (_ex_near_trough and best.direction == 1)
        except Exception:
            pass

        # Dynamic TP chi ap dung cho trade KHONG phai direction flip / scenario entry
        # Direction flip da set tp_roi_override rieng tai diem flip - khong override lai
        # Scenario entry da chon TP theo kich ban (range extreme=12%, con lai=potential scaling)
        # (AEQ-12=8%, 2h range=12%, 5h range=15%, MULTIHR=large, AEQ-VOL=8%, EXTREMA=8%)
        if not _direction_flipped and not _scenario_entry:
            if _is_short_term_extrema and not _is_major_peak and not _is_major_trough:
                best.tp_roi_override = _tp_small
                logger.info(f"{symbol}: short-term extrema -> TP={_tp_small*100:.0f}% ROI (small, fast)")
            elif _is_major_peak or _is_major_trough:
                best.tp_roi_override = 0.0  # dung scaling binh thuong (lon)
                logger.info(f"{symbol}: major extrema -> TP scaled by potential (large)")
            elif _pump_exhaustion_flip or _dump_exhaustion_flip:
                best.tp_roi_override = _tp_medium
                logger.info(f"{symbol}: exhaustion flip -> TP={_tp_medium*100:.0f}% ROI (medium)")
            # else: default potential scaling

        # ======================================================================
        # |  TRUE-DIRECTION GATE - chi trade khi trend RO + DUNG chieu           |
        # ======================================================================
        # REU: long khi trend quay xuong -> lo. MORPHO: short trong range choppy -> lo.
        # ZRO/ADA: short khi gia dang bounce len -> lo. Nguyen nhan: trade nguoc trend thuc
        # / trade trong choppy khong co trend. Gate nay:
        #   - Tinh TREND THUC tu macro + immediate momentum + volume (dong thuan co trong so)
        #   - Choppy (khong trend ro): SKIP momentum (khong danh bac trong range)
        #   - Signal nguoc trend thuc: FLIP ve dung chieu (bat lenh tiem nang dung huong)
        # MOI LENH deu phai qua gate nay (KE CA da flip huong): gate tu set best.direction = _T
        # (trend da xac minh anti-lag) -> khong tin huong signal/flip lung tung. Day la CACH DAM BAO
        # 'biet huong nao dung': huong = trend macro + fast EMA + cau truc 20 nen dong thuan, khong
        # phai do 1 trong 13 diem flip quyet dinh roi lot qua. Neu khong co trend xac lap -> KHONG trade.
        # Mien DUY NHAT: reversal path (da tat) va scenario (da tat) -> thuc te LUON chay gate nay.
        if not is_reversal and not _scenario_entry:
            _imm = self._immediate_momentum(df_micro, _sp)

            # ================================================================
            # TREND GATE v3 - 10000-SCENARIO SCORING FRAMEWORK
            # ================================================================
            # Kien truc: Fast EMA (20/50) la TIN HIEU CHU DAO (phan ung 20-50p).
            # Macro (100/250 / 300/600) la BO LOC BAI TRU - block khi nguoc Fast,
            # khong phai nguon quyet dinh chieu chinh.
            #
            # 8 bien trang thai thi truong:
            #   F (fast EMA 20/50), M1 (macro 100/250), M2 (macro 300/600),
            #   I (imm 7 nen), P (vi tri range), ADX, V (volume), S (HH/HL structure)
            # 3^5 x 7 x 4 x 3 = 20,412 to hop -> xu ly bang scoring engine.
            #
            # Loi goc KAITO: F=0 (crossing), M1=1 (lag), M2=0, I=1 (bounce) -> LONG SAI.
            #   Nguyen nhan: cu chi block F == -_T, khong block F == 0.
            #   Fix: khi F=0 -> CA HAI macro (M1 va M2) phai dong thuan,
            #   M2=0 (neutral) KHONG DU khi F=0 (khong co xac nhan thu 3).
            # ================================================================

            _fast_tr = self._trend_direction(df_micro, fast=20, slow=50)

            # --- BUOC 1: XAC DINH CHIEU TRADE (_T) ----------------------------
            # Fast EMA la tin hieu nhanh nhat co the tin cay (EMA 20/50 ~ 20-50 phut lag).
            # Macro (EMA 100-600) la boi canh lich su - block khi roi ro nguoc, khong phai chu.
            if _fast_tr != 0:
                # F co chieu ro rang: lay lam _T chu dao
                _T = _fast_tr
                # Macro KHONG DUOC NGUOC chieu F.
                # M1 nguoc: gia da di ra khoi macro -> conflict nguy hiem
                # M2 nguoc: 4h macro chong Fast -> 2 TF lon deu phan doi -> block
                if macro_trend == -_T:
                    logger.info(f"{symbol}: TREND skip - M1={macro_trend} nguoc fast={_fast_tr} (trend dang dao chieu)")
                    return _block("skip - macro nguoc fast EMA (trend dao chieu)")
                if macro_4h == -_T:
                    logger.info(f"{symbol}: TREND skip - M2={macro_4h} nguoc fast={_fast_tr} (4h trend chong fast)")
                    return _block("skip - macro_4h nguoc fast EMA (4h conflict)")
            else:
                # F=0: EMA20 ~ EMA50, dang crossing hoac sideway.
                # Khi F=0, CA HAI macro phai dong thuan (M1 == M2).
                # M1=1 + M2=0 + F=0 + I=1 = KAITO bug: chi co 1 macro, fast chua xac nhan
                # -> KHONG DU de xac dinh xu huong - nguy co vao lenh trong transition.
                if macro_trend != 0 and macro_trend == macro_4h:
                    # Ca hai macro dong thuan -> dung lam _T (do tin thap: F chua confirm)
                    _T = macro_trend
                else:
                    _msg = (f"fast=0 macro={macro_trend}/{macro_4h} "
                            f"({'M1=M2=0' if macro_trend==0 else 'M1!=M2, khong du'})")
                    logger.info(f"{symbol}: TREND skip - F=0, {_msg}")
                    return _block("skip - fast EMA neutral, khong du xac nhan huong (F=0 transition risk)")

            # --- BUOC 2: ADX - trend phai du manh de trade --------------------
            if math.isnan(adx) or adx < 20.0:
                logger.info(f"{symbol}: TREND skip - ADX={adx:.1f}<20 (chop/sideway)")
                return _block("skip - ADX<20 (chop)")

            # --- BUOC 3: VOLUME khong duoc nguoc trend xac lap ----------------
            if _vwt_dir == -_T and _vwt_str >= 0.45:
                logger.info(f"{symbol}: TREND skip - volume nguoc (vwt={_vwt_dir}:{_vwt_str:.2f} T={_T})")
                return _block("skip - volume nguoc trend")

            # --- BUOC 4: EMA STACK CHECK (them moi - bao ve khi F=0) ----------
            # Khi F=0 (EMA20~EMA50, transition), gia co the da vao "vung sai":
            #   LONG ma gia < EMA9 < EMA21 < EMA50 = bearish stack -> SKIP
            #   SHORT ma gia > EMA9 > EMA21 > EMA50 = bullish stack -> SKIP
            # Chi apply khi F=0 vi khi F!=0, EMA20/50 stack da phan anh trong _T roi.
            if _fast_tr == 0:
                _e9s  = compute_ema(df_micro["close"], 9).iloc[-1]
                _e21s = compute_ema(df_micro["close"], 21).iloc[-1]
                _e50s = compute_ema(df_micro["close"], min(50, len(df_micro) - 1)).iloc[-1]
                _ps   = df_micro["close"].iloc[-1]
                _bearish_stack = _ps < _e9s and _e9s < _e21s   # price below EMA9 & EMA9 below EMA21
                _bullish_stack = _ps > _e9s and _e9s > _e21s
                if _T == 1 and _bearish_stack:
                    logger.info(f"{symbol}: TREND skip LONG - EMA stack bearish khi F=0 (p<e9<e21, gia trong vung giam)")
                    return _block("skip LONG - EMA stack bearish khi F=0")
                if _T == -1 and _bullish_stack:
                    logger.info(f"{symbol}: TREND skip SHORT - EMA stack bullish khi F=0 (p>e9>e21, gia trong vung tang)")
                    return _block("skip SHORT - EMA stack bullish khi F=0")

            # --- BUOC 5: EXTENSION - entry tren pullback, khong chase ----------
            _ema21m = compute_ema(df_micro["close"], 21).iloc[-1]
            _atrm = _atr_for_sl if _atr_for_sl > 0 else float(
                (df_micro["high"].iloc[-14:] - df_micro["low"].iloc[-14:]).mean())
            _price_now = _range_live_price if _range_live_price > 0 else df_micro["close"].iloc[-1]
            _ext = (_price_now - _ema21m) / _atrm if _atrm > 0 else 0.0
            if _T == 1 and _ext > 1.5:
                logger.info(f"{symbol}: TREND skip LONG - extended {_ext:.1f}x ATR tren EMA21 (chase dinh)")
                return _block("skip LONG - gia extended tren EMA (chase dinh)")
            if _T == -1 and _ext < -1.5:
                logger.info(f"{symbol}: TREND skip SHORT - extended {_ext:.1f}x ATR duoi EMA21 (chase day)")
                return _block("skip SHORT - gia extended duoi EMA (chase day)")

            # --- BUOC 6: IMMEDIATE MOMENTUM phai CUNG CHIEU _T ----------------
            # imm == 0 (chop) HOAC imm == -_T (nguoc) -> KHONG vao.
            # Chi trade khi gia DANG di DUNG HUONG NGAY LUC NAY.
            if _imm != _T:
                logger.info(f"{symbol}: TREND skip - imm={_imm} != T={_T} (chua resume hoac nguoc)")
                return _block("skip - imm chua xac nhan trend (chop hoac nguoc)")

            # --- BUOC 7: CANDLE STRUCTURE (HH/HL) ----------------------------
            # Kiem tra 15 nen gan nhat: HH+HL = bullish, LH+LL = bearish.
            # Neu structure NGUOC chieu _T va F=0 (fast chua confirm): block.
            # Khi F!=0 (fast confirm), structure check chi la bonus cho confidence.
            _hh_ll = 0
            if len(df_micro) >= 15:
                _hr = df_micro["high"].iloc[-8:].max();  _hp = df_micro["high"].iloc[-15:-8].max()
                _lr = df_micro["low"].iloc[-8:].min();   _lp = df_micro["low"].iloc[-15:-8].min()
                if   _hr > _hp and _lr > _lp: _hh_ll = 1    # higher highs + higher lows
                elif _hr < _hp and _lr < _lp: _hh_ll = -1   # lower highs + lower lows
            if _fast_tr == 0 and _hh_ll == -_T:
                logger.info(f"{symbol}: TREND skip - candle structure (HH_LL={_hh_ll}) nguoc T={_T} khi F=0")
                return _block("skip - HH/HL candle structure nguoc trend (F=0)")

            # --- BUOC 8: ANTI-BREAKOUT COUNTER-TREND (bao ve nguoc breakout) --
            # Short khi gia dang break DINH 10 nen = short vao rally -> block.
            # Long khi gia dang break DAY 10 nen = long vao dump -> block.
            _N_brk = 10
            if len(df_micro) > _N_brk + 1:
                _prior_high = float(df_micro["high"].iloc[-(_N_brk + 1):-1].max())
                _prior_low  = float(df_micro["low"].iloc[-(_N_brk + 1):-1].min())
                _cur_close  = float(df_micro["close"].iloc[-1])
                if _T == -1 and _cur_close > _prior_high:
                    logger.info(f"{symbol}: TREND skip SHORT - close {_cur_close:.6f} pha dinh 10 nen {_prior_high:.6f} (breakout len)")
                    return _block("skip SHORT - gia break dinh 10 nen (rally nguoc)")
                if _T == 1 and _cur_close < _prior_low:
                    logger.info(f"{symbol}: TREND skip LONG - close {_cur_close:.6f} pha day 10 nen {_prior_low:.6f} (breakdown xuong)")
                    return _block("skip LONG - gia break day 10 nen (dump nguoc)")

            # ================================================================
            # DA QUA CA 8 GATE - TINH DIEM DO TIN CAY (CONFIDENCE 0-10)
            # Quy tac: moi yeu to dong thuan voi _T cho them diem.
            # Diem quyet dinh TP (bao nhieu % ROI chot loi) va von (bao nhieu % equity).
            # ================================================================
            # Lop 1 - EMA alignment (toi da 6 diem):
            #   fast_tr == _T: +2 (luon dung vi gate buoc 1 da check)
            #   M1     == _T: +2 (macro 100/250 xac nhan)
            #   M2     == _T: +2 (macro 300/600 xac nhan)
            # Lop 2 - Supplement (toi da 4 diem):
            #   volume xac nhan va manh: +1
            #   ADX >= 30 (trend rat manh): +1
            #   HH/HL candle structure dong thuan: +1
            #   Entry gap EMA21 <= 0.5 ATR (entry dep): +1
            _conf = 0
            if _fast_tr == _T:                              _conf += 2   # fast EMA xac nhan
            if macro_trend == _T:                           _conf += 2   # M1 xac nhan
            if macro_4h == _T:                              _conf += 2   # M2 xac nhan
            if _vwt_dir == _T and _vwt_str >= 0.40:        _conf += 1   # volume xac nhan
            if not math.isnan(adx) and adx >= 30:          _conf += 1   # trend rat manh
            if _hh_ll == _T:                                _conf += 1   # structure dong thuan
            if abs(_ext) <= 0.5:                            _conf += 1   # entry gan EMA21
            # _conf: 0-10

            # Nguong toi thieu: can it nhat 3 diem
            # (vi du: fast=+2 + 1 supplement = 3 -> TP nho nhat, von nho nhat)
            if _conf < 3:
                logger.info(f"{symbol}: TREND skip - confidence={_conf}<3 (qua it tin hieu dong thuan)")
                return _block("skip - confidence < 3 (khong du tin hieu dong thuan)")

            # TP scale theo confidence: 12% (conf=0) -> 25% (conf=10)
            # conf=3 -> 15.9%, conf=5 -> 18.5%, conf=7 -> 21.1%, conf=10 -> 25%
            _tp_by_conf = 0.12 + (_conf / 10.0) * 0.13
            _tp_by_conf = max(config.TP_ROI_MIN, min(config.TP_ROI_MAX, _tp_by_conf))

            # Conviction cho capital: conf/10 -> potential -> capital_pct
            # conf=3 -> 0.30 (5% equity), conf=5 -> 0.50, conf=7 -> 0.70, conf=10 -> 1.0 (90%)
            _conv = _conf / 10.0

            # Direction = _T da xac lap qua toan bo gate
            if best.direction != _T:
                logger.info(f"{symbol}: TREND override -> {'LONG' if _T==1 else 'SHORT'} (signal={best.direction})")
            best.direction = _T
            best.tp_roi_override = round(_tp_by_conf, 4)
            best.strength = max(best.strength, min(1.0, _conv))

            logger.info(
                f"{symbol}: GATE PASS | {'L' if _T==1 else 'S'} | "
                f"F={_fast_tr} M1={macro_trend} M2={macro_4h} I={_imm} | "
                f"ADX={adx:.0f} ext={_ext:.2f} HH_LL={_hh_ll} vwt={_vwt_dir}:{_vwt_str:.2f} | "
                f"conf={_conf}/10 TP={_tp_by_conf*100:.1f}% conv={_conv:.2f}"
            )

        # ======================================================================
        # |  FINAL EXHAUSTION-ZONE GUARD - QUYET DINH CUOI CUNG tai CUC DOAN     |
        # ======================================================================
        # Nguyen tac cot loi (user lap lai nhieu lan): "o DINH phai SHORT, o DAY phai LONG".
        # ZEC pattern: downtrend manh -> vol-trend ep SHORT, NHUNG gia o 19% range (sat day)
        # -> bounce ngay sau -> LO. Short o day / long o dinh = nguoc mean-reversion tai cuc doan.
        #
        # Guard chay CUOI CUNG (sau ca vol-trend master + moi flip) -> tieng noi quyet dinh.
        # QUY TAC TUYET DOI: KHONG long dinh/gan dinh (>=80%), KHONG short day/gan day (<=20%).
        # KHONG ngoai le breakout - pump len dinh moi van la mua dinh (BZ 91.88 gan dinh 91.93).
        if not is_reversal:
            _ez_price = _range_live_price if _range_live_price > 0 else (
                df_micro["close"].iloc[-1] if not df_micro.empty else 0.0)
            _ez_newdir, _ez_tp, _ez_reason = self._exhaustion_check(
                df_micro, best.direction, _ez_price, rsi_now, _sp, _vol_trend_locked)
            if _ez_newdir == 0:
                logger.info(f"{symbol}: EXHAUSTION {_ez_reason} - {'BLESS' if best.direction==1 else 'ZEC'} pattern, skip")
                return _block(f"skip - exhaustion {_ez_reason}")
            if _ez_newdir != best.direction:
                logger.info(f"{symbol}: [EXHAUSTION] {_ez_reason} - flip sang {'LONG' if _ez_newdir==1 else 'SHORT'}")
                best.direction = _ez_newdir
                best.tp_roi_override = _ez_tp

        names = "+".join(s.strategy_name for s in signals)
        _tp_log = f" TP_override={best.tp_roi_override*100:.0f}%" if best.tp_roi_override > 0 else ""

        # ENTRY DECISION LOG - hien thi CHINH XAC vi sao vao lenh + tat ca gate values.
        # Neu 1 lenh sai chieu, dong log nay cho biet dung path/gia tri nao -> fix trung dich.
        try:
            _dbg_pos = 0.0
            if not df_micro.empty and len(df_micro) >= 120:
                _dh = df_micro["high"].iloc[-120:].max(); _dl = df_micro["low"].iloc[-120:].min()
                if _dh > _dl:
                    _dbg_pos = ((_range_live_price if _range_live_price > 0 else df_micro["close"].iloc[-1]) - _dl) / (_dh - _dl)
            logger.info(
                f"{symbol}: [ENTRY-DECISION] {'LONG' if best.direction==1 else 'SHORT'} | "
                f"pos_in_range={_dbg_pos*100:.0f}% | macro={macro_trend}/{macro_4h} "
                f"vwt={_vwt_dir}:{_vwt_str:.2f} imm={self._immediate_momentum(df_micro,_sp)} | "
                f"path={'reversal' if is_reversal else ('scenario:'+_scenario_name if _scenario_entry else 'momentum')} "
                f"flipped={_direction_flipped}"
            )
        except Exception:
            pass

        logger.info(
            f"{symbol} [{names}] consensus={len(signals)} -> "
            f"{'LONG' if best.direction==1 else 'SHORT'} "
            f"strength={best.strength:.2f}{_tp_log} | {best.reason}"
        )

        self.executor.execute_signal(symbol, best, equity, open_positions, is_priority=is_priority)
        return True


# -- Entry point ---------------------------------------------------------------

if __name__ == "__main__":
    bot = TradingBot()
    bot.run()
