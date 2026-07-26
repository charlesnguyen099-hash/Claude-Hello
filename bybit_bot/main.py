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


def compute_macd(close, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD = EMA(fast) - EMA(slow). Signal = EMA(signal) of MACD. Histogram = MACD - Signal.
    Tra ve (macd_val, signal_val, histogram) tai nen cuoi cung. NaN neu khong du du lieu."""
    if close is None or len(close) < slow + signal:
        return float("nan"), float("nan"), float("nan")
    macd_line = compute_ema(close, fast) - compute_ema(close, slow)
    sig_line  = compute_ema(macd_line, signal)
    return float(macd_line.iloc[-1]), float(sig_line.iloc[-1]), float((macd_line - sig_line).iloc[-1])


def _candle_pattern(df, n: int = 3) -> int:
    """Phan tich pattern nen cuoi cung (va n nen truoc).
    Tra ve: +1 bullish pattern, -1 bearish pattern, 0 khong ro.
    Bat: Engulfing, Pin Bar, Hammer/Shooting Star, Marubozu.
    Dung cho GATE V4: pattern xac nhan la 1 trong 5 dau hieu confluence."""
    if df is None or df.empty or len(df) < 2:
        return 0
    o1 = float(df["open"].iloc[-1]);  c1 = float(df["close"].iloc[-1])
    h1 = float(df["high"].iloc[-1]); l1 = float(df["low"].iloc[-1])
    o2 = float(df["open"].iloc[-2]); c2 = float(df["close"].iloc[-2])
    rng1 = max(h1 - l1, 1e-12)
    body1 = abs(c1 - o1)
    low_wick1  = (min(o1, c1) - l1) / rng1
    high_wick1 = (h1 - max(o1, c1)) / rng1
    body_ratio1 = body1 / rng1

    # Bullish Engulfing: nen hien tai xanh nuot toan bo nen do truoc
    if c1 > o1 and c2 < o2 and c1 >= o2 and o1 <= c2:
        return 1
    # Bearish Engulfing: nen hien tai do nuot toan bo nen xanh truoc
    if c1 < o1 and c2 > o2 and c1 <= o2 and o1 >= c2:
        return -1
    # Hammer (bullish): bong duoi dai (>55% range), than nho, o duoi range
    if low_wick1 >= 0.55 and body_ratio1 <= 0.35 and (min(o1,c1) - l1) > (h1 - max(o1,c1)):
        return 1
    # Shooting Star (bearish): bong tren dai, than nho, o tren range
    if high_wick1 >= 0.55 and body_ratio1 <= 0.35 and (h1 - max(o1,c1)) > (min(o1,c1) - l1):
        return -1
    # Pin Bar Bullish: bong duoi rat dai (>65%), than rat nho
    if low_wick1 >= 0.65 and body_ratio1 <= 0.20:
        return 1
    # Pin Bar Bearish: bong tren rat dai (>65%), than rat nho
    if high_wick1 >= 0.65 and body_ratio1 <= 0.20:
        return -1
    # Marubozu Bullish: than chiem >80% range, rat it bong
    if c1 > o1 and body_ratio1 >= 0.80:
        return 1
    # Marubozu Bearish
    if c1 < o1 and body_ratio1 >= 0.80:
        return -1
    return 0


def _volume_ratio(df, lookback: int = 20) -> float:
    """Tinh ty le volume trung binh 3 nen cuoi / volume trung binh lookback nen.
    > 2.0 = spike manh; 1.5-2.0 = cao; 0.7-1.5 = binh thuong; < 0.7 = thap."""
    if df is None or df.empty or len(df) < lookback:
        return 1.0
    vol = df["volume"]
    avg_recent = vol.iloc[-3:].mean()
    avg_base   = vol.iloc[-lookback:-3].mean()
    if avg_base <= 0:
        return 1.0
    return avg_recent / avg_base


def _ema_state(df, lookback_cross: int = 6) -> str:
    """Phan loai trang thai EMA theo 6 category cua 10000 HQ Scenarios.
    Returns: bullish_stack | bearish_stack | golden_cross | death_cross |
             price_cross_up | price_cross_down | neutral"""
    if df is None or len(df) < 10:
        return "neutral"
    close = df["close"]
    price = float(close.iloc[-1])
    e20 = compute_ema(close, 20); e50 = compute_ema(close, 50)
    e20v = float(e20.iloc[-1]); e50v = float(e50.iloc[-1])

    # Full stack (uu tien: 90%+ heavy)
    if len(df) >= 200:
        e200v = float(compute_ema(close, 200).iloc[-1])
        if price > e20v > e50v > e200v: return "bullish_stack"
        if price < e20v < e50v < e200v: return "bearish_stack"

    # Golden/Death cross: EMA20 cat EMA50 trong lookback_cross nen gan nhat
    _lb = min(lookback_cross, len(df) - 1)
    e20p = float(e20.iloc[-_lb]); e50p = float(e50.iloc[-_lb])
    if e20p < e50p and e20v >= e50v: return "golden_cross"
    if e20p > e50p and e20v <= e50v: return "death_cross"

    # Price cat EMA20
    pricep = float(close.iloc[-_lb])
    if pricep < e20p and price >= e20v: return "price_cross_up"
    if pricep > e20p and price <= e20v: return "price_cross_down"
    return "neutral"


def _macd_state(hist: float, prev_hist: float) -> str:
    """Phan loai MACD theo 8 category cua 10000 HQ Scenarios.
    Returns: bull_divergence | bear_divergence | bull_crossover | bear_crossover |
             hist_up | hist_down | above0_bull | below0_bear | neutral"""
    if math.isnan(hist): return "neutral"
    ph = prev_hist if not math.isnan(prev_hist) else hist
    # Crossover (hai dau hieu manh nhat theo Excel)
    if ph < 0 and hist > 0: return "bull_crossover"
    if ph > 0 and hist < 0: return "bear_crossover"
    # Divergence: histogram nguoc voi nen 0 nhung dang dao chieu
    # "MACD below 0 & bullish div": hist < 0 nhung dang tang (momentum reversing)
    if hist < 0 and hist > ph: return "bull_divergence"
    # "MACD above 0 & bearish div": hist > 0 nhung dang giam
    if hist > 0 and hist < ph: return "bear_divergence"
    # Histogram tang/giam ro rang
    if hist > 0 and hist >= ph: return "hist_up"
    if hist < 0 and hist <= ph: return "hist_down"
    if hist > 0: return "above0_bull"
    if hist < 0: return "below0_bear"
    return "neutral"


def _rsi_zone(rsi: float, df=None) -> str:
    """Phan loai RSI zone theo 6 category cua 10000 HQ Scenarios.
    Returns: oversold | near_oversold | overbought | near_overbought |
             bull_divergence | bear_divergence | neutral"""
    if math.isnan(rsi): return "neutral"
    # Extreme zones (uu tien check truoc divergence)
    if rsi < 30: return "oversold"
    if rsi < 40: return "near_oversold"
    if rsi > 70: return "overbought"
    if rsi > 60: return "near_overbought"
    # Divergence detection: price vs RSI trong 20 nen — yeu cau bien dong gia >= 1%
    # va RSI phai o vung ho tro huong divergence (khong chi neutral)
    if df is not None and len(df) >= 20:
        _price_now = float(df["close"].iloc[-1])
        _price_20  = float(df["close"].iloc[-20])
        _chg = (_price_now - _price_20) / max(abs(_price_20), 1e-12)
        # Bullish div: price giam >= 1% nhung RSI van >= 45 (RSI khong confirm suy yeu)
        if _chg < -0.010 and 45 <= rsi <= 60: return "bull_divergence"
        # Bearish div: price tang >= 1% nhung RSI van <= 55 (RSI khong confirm suc manh)
        if _chg > 0.010 and 40 <= rsi <= 55: return "bear_divergence"
    return "neutral"


# ===========================================================================
# NEW UTILITY FUNCTIONS — 50K SCENARIO ADDITIONS
# ===========================================================================

def compute_macd_series(close, fast=12, slow=26, signal=9):
    """Returns (macd_line, sig_line, histogram) as pd.Series for divergence detection."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig_line  = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, sig_line, macd_line - sig_line


def _detect_divergence_50k(price_series, indicator_series, lookback=30, tol=0.003):
    """Detect regular and hidden divergence between price and indicator.
    Hidden Bull: price HL + indicator LL = trend continuation signal (#1 in 50K 95-98% tier).
    Returns dict with regular_bull/bear, hidden_bull/bear keys."""
    result = {"regular_bull": False, "regular_bear": False,
              "hidden_bull": False,  "hidden_bear": False}
    try:
        if len(price_series) < lookback or len(indicator_series) < lookback:
            return result
        ps  = price_series.dropna().iloc[-lookback:]
        ind = indicator_series.dropna().iloc[-lookback:]
        if len(ps) < lookback // 2 or len(ind) < lookback // 2:
            return result
        half = len(ps) // 2
        p_p_low  = float(ps.iloc[:half].min());  p_r_low  = float(ps.iloc[half:].min())
        p_p_high = float(ps.iloc[:half].max());  p_r_high = float(ps.iloc[half:].max())
        i_p_low  = float(ind.iloc[:half].min()); i_r_low  = float(ind.iloc[half:].min())
        i_p_high = float(ind.iloc[:half].max()); i_r_high = float(ind.iloc[half:].max())
        base_p = max(abs(p_p_low), 1e-12); base_ph = max(abs(p_p_high), 1e-12)
        base_il = max(abs(i_p_low),  1e-12); base_ih = max(abs(i_p_high), 1e-12)
        # Regular bull: price LL + indicator HL (classic oversold div)
        # For negative indicator values, use absolute-distance comparison (not ratio — ratio inverts sign)
        if i_p_low >= 0:
            _reg_bull_ind = i_r_low > i_p_low * (1 + tol)
        else:
            _reg_bull_ind = i_r_low > i_p_low + abs(i_p_low) * tol  # i_r_low less negative than i_p_low
        result["regular_bull"] = (p_r_low < p_p_low * (1 - tol)) and _reg_bull_ind
        # Regular bear: price HH + indicator LH
        result["regular_bear"] = (p_r_high > p_p_high * (1 + tol)) and (i_r_high < i_p_high * (1 - tol))
        # Hidden bull: price HL + indicator LL (#1 signal in 95-98% tier)
        result["hidden_bull"]  = (p_r_low  > p_p_low  * (1 + tol)) and (i_r_low  < i_p_low  * (1 - tol) or (i_p_low > 0 and i_r_low < i_p_low - i_p_low * tol))
        # Hidden bear: price LH + indicator HH
        result["hidden_bear"]  = (p_r_high < p_p_high * (1 - tol)) and (i_r_high > i_p_high * (1 + tol))
    except Exception:
        pass
    return result


def _ema_state_v2(df, lookback_cross=6):
    """Expanded EMA state from 50K scenarios — includes EMA50/200 cross, bounce, pullback.
    Returns (category, direction) where direction: 1=bull, -1=bear, 0=neutral."""
    if df is None or len(df) < 10:
        return "neutral", 0
    close = df["close"]
    price = float(close.iloc[-1])
    e20 = compute_ema(close, 20); e50 = compute_ema(close, 50)
    e20v = float(e20.iloc[-1]);   e50v = float(e50.iloc[-1])
    _lb  = min(lookback_cross, len(df) - 1)
    e20p = float(e20.iloc[-_lb]); e50p_lb = float(e50.iloc[-_lb])
    pricep = float(close.iloc[-_lb])

    if len(df) >= 200:
        e200  = compute_ema(close, 200)
        e200v = float(e200.iloc[-1]); e200p = float(e200.iloc[-_lb])
        e50p2 = float(e50.iloc[-_lb])

        # Full bull/bear stack (highest conviction)
        if price > e20v > e50v > e200v: return "bullish_stack", 1
        if price < e20v < e50v < e200v: return "bearish_stack", -1

        # EMA50/200 Classic Golden/Death Cross — 469+409=878 in 95-98% tier
        if e50p2 < e200p and e50v >= e200v: return "ema50_200_golden", 1
        if e50p2 > e200p and e50v <= e200v: return "ema50_200_death",  -1

        # Price crossing EMA200 (major signal — 3393+3344 in 50K)
        if pricep < e200p and price >= e200v: return "price_cross_ema200_up",   1
        if pricep > e200p and price <= e200v: return "price_cross_ema200_down", -1

        # Pullback to EMA20 in trend (Price > EMA50 > EMA200, touching EMA20)
        if price > e50v > e200v and price <= e20v * 1.008:
            return "pullback_ema20_bull", 1
        if price < e50v < e200v and price >= e20v * 0.992:
            return "pullback_ema20_bear", -1

        # EMA200 bounce/rejection (within 0.6%)
        if e200v > 0 and abs(price - e200v) / e200v < 0.006:
            if price >= e200v and e50v > e200v: return "price_bounce_ema200",  1
            if price <= e200v and e50v < e200v: return "price_reject_ema200", -1

        # EMA50 bounce/rejection (within 0.6%)
        if e50v > 0 and abs(price - e50v) / e50v < 0.006:
            if price >= e50v and e20v > e50v: return "price_bounce_ema50",  1
            if price <= e50v and e20v < e50v: return "price_reject_ema50", -1

    # EMA20/50 Golden/Death Cross (original)
    if e20p < e50p_lb and e20v >= e50v: return "golden_cross", 1
    if e20p > e50p_lb and e20v <= e50v: return "death_cross",  -1

    # Price crossing EMA20
    if pricep < e20p and price >= e20v: return "price_cross_up",   1
    if pricep > e20p and price <= e20v: return "price_cross_down", -1

    return "neutral", 0


def _stoch_state_50k(k, d, k_prev=None, d_prev=None):
    """Classify stochastic state per 50K scenarios.
    Returns category string."""
    if math.isnan(k) or math.isnan(d):
        return "neutral"
    cross_up   = k_prev is not None and not math.isnan(k_prev) and k_prev < d and k >= d
    cross_down = k_prev is not None and not math.isnan(k_prev) and k_prev > d and k <= d
    if k < 20 and cross_up:   return "oversold_cross_up"      # 449 in 95-98%
    if k > 80 and cross_down: return "overbought_cross_down"  # 465 in 95-98%
    if k < 20 and k > d:      return "embedded_oversold_cross"  # just crossed from embedded oversold
    if k > 80 and k < d:      return "embedded_overbought_cross"
    if k < 20:                return "oversold"
    if k > 80:                return "overbought"
    if 20 <= k <= 50:         return "rising_midzone"
    if 50 <  k <= 80:         return "upper_midzone"   # K=50-80 trong uptrend = embedded strength
    return "neutral"


def _session_score_50k():
    """Session timing score per 50K scenario distribution.
    Returns (category, score). London-NY Overlap = 31% of 95-98% tier."""
    h = datetime.utcnow().hour
    if 13 <= h < 17: return "london_ny_overlap", 12  # 1129/3650 = 31%
    if 12 <= h < 13: return "ny_premarket",       9
    if 8  <= h < 12: return "london_open",         9  # 646+641/3650 = 35% combined London window
    if 17 <= h < 21: return "ny_regular",          8  # 623/3650 = 17%
    if 0  <= h < 8:  return "asian",               5
    return "off_hours", 4


def _candle_pattern_v2(df, n: int = 3) -> int:
    """Expanded candle pattern detection — adds 50K patterns: Three White Soldiers,
    Three Black Crows, Tweezer Top/Bottom, Bullish/Bearish Harami, Piercing Line, Dark Cloud.
    Returns: 1=bullish, -1=bearish, 0=neutral"""
    # First check original patterns
    result = _candle_pattern(df, n)
    if result != 0:
        return result
    if df is None or len(df) < 4:
        return 0
    o = df["open"]; h = df["high"]; l = df["low"]; c = df["close"]
    try:
        # Three White Soldiers: 3 consecutive bullish candles with higher closes
        if (c.iloc[-1] > c.iloc[-2] > c.iloc[-3] and
            c.iloc[-1] > o.iloc[-1] and c.iloc[-2] > o.iloc[-2] and c.iloc[-3] > o.iloc[-3]):
            return 1
        # Three Black Crows: 3 consecutive bearish candles with lower closes
        if (c.iloc[-1] < c.iloc[-2] < c.iloc[-3] and
            c.iloc[-1] < o.iloc[-1] and c.iloc[-2] < o.iloc[-2] and c.iloc[-3] < o.iloc[-3]):
            return -1
        # Tweezer Bottom: two candles with same low, second bullish
        tw_tol = (h.iloc[-1] - l.iloc[-1]) * 0.1 + 1e-12
        if abs(l.iloc[-1] - l.iloc[-2]) < tw_tol and c.iloc[-1] > o.iloc[-1]:
            return 1
        # Tweezer Top: two candles with same high, second bearish
        if abs(h.iloc[-1] - h.iloc[-2]) < tw_tol and c.iloc[-1] < o.iloc[-1]:
            return -1
        # Bullish Harami: large bearish candle then small bullish inside
        prev_bear = c.iloc[-2] < o.iloc[-2]
        curr_inside = o.iloc[-1] > c.iloc[-2] and c.iloc[-1] < o.iloc[-2]
        if prev_bear and curr_inside and c.iloc[-1] > o.iloc[-1]:
            return 1
        # Bearish Harami: large bullish candle then small bearish inside
        prev_bull = c.iloc[-2] > o.iloc[-2]
        curr_inside_b = o.iloc[-1] < c.iloc[-2] and c.iloc[-1] > o.iloc[-2]
        if prev_bull and curr_inside_b and c.iloc[-1] < o.iloc[-1]:
            return -1
        # Piercing Line: bearish then bullish that closes > midpoint of previous
        if (c.iloc[-2] < o.iloc[-2] and c.iloc[-1] > o.iloc[-1] and
                o.iloc[-1] < c.iloc[-2] and c.iloc[-1] > (o.iloc[-2] + c.iloc[-2]) / 2):
            return 1
        # Dark Cloud Cover: bullish then bearish that closes < midpoint of previous
        if (c.iloc[-2] > o.iloc[-2] and c.iloc[-1] < o.iloc[-1] and
                o.iloc[-1] > c.iloc[-2] and c.iloc[-1] < (o.iloc[-2] + c.iloc[-2]) / 2):
            return -1
    except Exception:
        pass
    return 0


class TradingBot:
    def __init__(self):
        logger.info("="*60)
        logger.info("Bybit Auto Trading Bot starting...")
        # === VERSION BANNER - de XAC NHAN dang chay code MOI (khong phai code cu) ===
        # Neu ban KHONG thay dong nay khi khoi dong -> bot dang chay code CU, PHAI restart.
        logger.info(">>> SCENARIO GATE v6+ : 50K HQ Scenario Matcher — 3-tier TP, dynamic exit <<<")
        logger.info(">>> 10-factor confluence scoring (0-100) — trade khi score>=50 — Excel-aligned <<<")
        print(">>> [SCENARIO GATE v6+] 50K HQ Scenario Matcher ACTIVE — 95-98% / 90-95% / 85-90% tiers <<<", flush=True)
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
        # Dynamic TP: TP price da duoc nang theo momentum (chi tang, khong giam)
        self._dyn_tp_raised: dict[str, float] = {}

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
                self._dyn_tp_raised.pop(sym, None)
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
            f"[TICK] Scan {len(self.symbols)} coins ({sum(1 for s in self.symbols if self.scanner.volume_map.get(s,0)>=config.HIGH_VOL_THRESHOLD)}high+{sum(1 for s in self.symbols if self.scanner.volume_map.get(s,0)<config.HIGH_VOL_THRESHOLD)}rest) | "
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
        if config.TOP_TRADE_COUNT > 0:
            top_trade = self.symbols[:config.TOP_TRADE_COUNT]
        else:
            top_trade = self.symbols

        # Tach 2 nhom theo volume:
        #   high_vol (>= HIGH_VOL_THRESHOLD = 10M): scan tan suat cao (cooldown ngan, budget lon)
        #   rest     (<  HIGH_VOL_THRESHOLD)       : scan tan suat thap hon (cooldown dai, budget nho)
        # Scanner da sort tat ca theo trending_score -> thu tu uu tien van theo score.
        _vmap      = self.scanner.volume_map
        high_vol   = [s for s in top_trade if _vmap.get(s, 0) >= config.HIGH_VOL_THRESHOLD]
        rest_vol   = [s for s in top_trade if _vmap.get(s, 0) <  config.HIGH_VOL_THRESHOLD]

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
                    import traceback as _tb
                    logger.warning(f"Error processing {symbol}: {str(e).encode('ascii','replace').decode()}\n{_tb.format_exc()}")
            logger.debug(f"[TICK] {label}: analyzed {_analyzed}/{len(symbols)} coins, traded {_traded}")
            return False

        # 2 pass: high-vol truoc (tan suat cao), rest sau (tan suat thap hon)
        _run_scan(high_vol,  config.TOP20_COOLDOWN_SEC,  config.SCAN_BUDGET_TOP20_SEC, "HIGH-VOL")
        _run_scan(rest_vol,  config.SYMBOL_COOLDOWN_SEC, config.SCAN_BUDGET_REST_SEC,  "REST")

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

            # Nguong chot loi phai TRU TOAN BO phi khu hoi (vao+ra+funding) theo ROI.
            # unrealisedPnl cua Bybit la GROSS (chua tru phi nao) nen phai dat nguong tren tong phi.
            # TOTAL_ROUND_TRIP_COST = entry_fee + exit_fee + funding_buffer -> dam bao moi exit
            # deu cover du phi -> khong bao gio chot loi ma hoa ra lo sau phi.
            _exit_cost_roi = config.TOTAL_ROUND_TRIP_COST * _lev   # entry+exit+funding (toan bo phi)
            _lock_thresh   = max(config.DYN_PROFIT_LOCK_ROI, _exit_cost_roi + 0.05)  # +5% net buffer (tranh close dang lo sau phi)

            # 1) PROFIT-LOCK: dang loi (sau phi dong) ma market quay dau nguoc -> chot ngay
            if pnl_roi >= _lock_thresh:
                _turned = (imm == -pos_dir) or (macro_dir == -pos_dir)
                if _turned:
                    logger.warning(
                        f"[DYN-EXIT] {symbol} {side}: LOCK PROFIT roi=+{pnl_roi*100:.0f}% "
                        f"| market quay dau (imm={imm} macro={macro_dir} pos={pos_dir}) -> chot ngay"
                    )
                    self.executor._close_position(pos)
                    self._dyn_tp_raised.pop(symbol, None)
                    continue

            # 1b) SOFT PROFIT LOCK: co loi nho nhung BOTH imm+macro quay nguoc -> dong som.
            # Bat case vao lenh giua dao dong (mid-oscillation entry): gia di nguoc ngay sau entry,
            # co chut loi gross nhung se mat het sau phi neu tiep tuc xau -> dong khi con loi.
            if getattr(config, "DYN_TP_ENABLE", True) and _exit_cost_roi < pnl_roi < _lock_thresh:
                if imm == -pos_dir and macro_dir == -pos_dir:
                    logger.warning(
                        f"[DYN-TP] {symbol} {side}: SOFT LOCK roi=+{pnl_roi*100:.2f}% "
                        f"| imm={imm} macro={macro_dir} ca hai quay nguoc -> dong truoc khi mat loi"
                    )
                    self.executor._close_position(pos)
                    self._dyn_tp_raised.pop(symbol, None)
                    continue

            # 1c) DYNAMIC TP RAISE: momentum van manh cung chieu lenh -> nang TP tren san.
            # TP chi duoc tang (trailing theo huong loi), khong bao gio HA xuong.
            # Chi nang TP khi da CO LOI NET (pnl_roi > phi khu hoi) - khong nang khi con lo.
            if getattr(config, "DYN_TP_RAISE_ENABLE", True) and pnl_roi > _exit_cost_roi:
                if imm == pos_dir and macro_dir == pos_dir:
                    _entry_px = _sf(pos.get("avgPrice", 0))
                    _cur_tp   = self.executor._tp_price.get(symbol, 0.0)
                    if _entry_px > 0 and _cur_tp > 0:
                        _tick      = self.executor._tick_size.get(symbol, 0.0)
                        _tp_ceil   = getattr(config, "DYN_TP_CEIL_ROI", 0.20)
                        _tp_ceil_px = _entry_px + pos_dir * _tp_ceil * _entry_px / _lev
                        _mark      = float(df["close"].iloc[-1])
                        _step      = getattr(config, "DYN_TP_RAISE_STEP", 0.30)
                        _remaining = (_tp_ceil_px - _mark) * pos_dir
                        if _remaining > 0:
                            _new_tp = _mark + pos_dir * _remaining * _step
                            _prev   = self._dyn_tp_raised.get(symbol, _cur_tp)
                            # TP chi duoc nang: Long TP phai tang, Short TP phai giam
                            _raised = (pos_dir == 1 and _new_tp > _prev + 1e-9) or \
                                      (pos_dir == -1 and _new_tp < _prev - 1e-9)
                            if _raised:
                                _new_tp_r = self.client.round_to_tick(_new_tp, _tick) if _tick > 0 else round(_new_tp, 6)
                                try:
                                    self.client.update_take_profit(symbol, _new_tp_r, tick_size=_tick)
                                    self._dyn_tp_raised[symbol] = _new_tp_r
                                    self.executor._tp_price[symbol] = _new_tp_r
                                    logger.info(
                                        f"[DYN-TP] {symbol} {side}: RAISE TP {_cur_tp:.6f} -> {_new_tp_r:.6f} "
                                        f"roi=+{pnl_roi*100:.1f}% imm={imm} macro={macro_dir}"
                                    )
                                except Exception as _dtp_err:
                                    logger.debug(f"[DYN-TP] {symbol}: raise TP failed: {_dtp_err!r}")

            # 2) SMART CUT-LOSS: dang lo + trend lon nguoc han -> khong the phuc hoi
            if pnl_roi < -_exit_cost_roi and macro_dir == -pos_dir:
                if imm == pos_dir:
                    # co bounce nguoc ve phia minh = luc LO IT NHAT -> dong ngay
                    logger.warning(
                        f"[DYN-EXIT] {symbol} {side}: CUT on bounce roi={pnl_roi*100:.0f}% "
                        f"| trend nguoc (macro={macro_dir}) + bounce (imm={imm}) -> dong luc lo it nhat"
                    )
                    self.executor._close_position(pos)
                    self._dyn_tp_raised.pop(symbol, None)
                    continue
                if pnl_roi <= -config.DYN_HARD_CUT_ROI:
                    # lo qua sau, khong co bounce -> cat luon, khong cho cham SL banh chanh
                    logger.warning(
                        f"[DYN-EXIT] {symbol} {side}: HARD CUT roi={pnl_roi*100:.0f}% "
                        f"| trend nguoc + khong bounce -> cat, chan lo them"
                    )
                    self.executor._close_position(pos)
                    self._dyn_tp_raised.pop(symbol, None)
                    continue

            # 2b) IMM-CUT: macro van bullish nhung immediate momentum dao nguoc manh + lo dang sau
            # Bat case coin uptrend (macro=1) nhung gia dang dump ngan han (imm=-1):
            # macro khong flip nen SMART CUT khong chay -> can IMM-CUT rieng.
            # Chi cat khi da du lo (tranh cat nham khi noise): 15% ROI = ~1.5% price o 10x.
            if pnl_roi <= -0.15 and imm == -pos_dir:
                logger.warning(
                    f"[DYN-EXIT] {symbol} {side}: IMM-CUT roi={pnl_roi*100:.0f}% "
                    f"| imm={imm} nguoc chieu + lo sau (macro={macro_dir} van cung chieu nhung gia dang giam) -> cat"
                )
                self.executor._close_position(pos)
                self._dyn_tp_raised.pop(symbol, None)
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
        # body phai co nghia (>= 20% range cua nen) de tranh doji nho kich hoat flip
        _body_meaningful = abs(body) / rngc >= 0.20
        reject = (_body_meaningful and body < 0) or (upper_wick > 0.5) or (rsi_now > 68)
        bounce = (_body_meaningful and body > 0) or (lower_wick > 0.5) or (rsi_now < 32)
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

    def _quality_gate(self, direction: int, df_micro, rsi_now: float,
                      sc_pos: float, sp: float,
                      is_reversal: bool = False,
                      is_breakout: bool = False,
                      pos_24h: float = 0.5,
                      hi_24h: float = 0.0,
                      lo_24h: float = 0.0,
                      strong_trend: bool = False) -> tuple[bool, str]:
        """
        Universal pre-entry quality gate - ap dung cho TAT CA entry paths.
        Returns (True, "") neu OK, (False, reason) neu bi block.

        QG-1: RSI extreme - khong short oversold (<25), khong long overbought (>75)
        QG-2: 2h range position - khong short <15% range, khong long >85% range
        QG-3: 30m high/low proximity - khong short trong 0.25% cua 30m low (va nguoc lai)
        QG-4: Immediate momentum conflict - khong short khi bounce, khong long khi dump
        QG-5: EMA21 overstretch - khong short khi da qua xa duoi EMA21 (va nguoc lai)
        QG-6: 3-candle body direction strongly against entry
        """
        if df_micro is None or df_micro.empty or len(df_micro) < 21:
            return True, ""

        price = float(df_micro["close"].iloc[-1])

        # QG-1: RSI extreme
        if direction == -1 and rsi_now < 25:
            return False, f"QG1:RSI={rsi_now:.0f}<25(oversold->block SHORT)"
        if direction == 1 and rsi_now > 75:
            return False, f"QG1:RSI={rsi_now:.0f}>75(overbought->block LONG)"

        # QG-2: 2h range position + 24h range position (skip breakout)
        if not is_breakout:
            if direction == -1 and sc_pos < 0.15:
                return False, f"QG2a:pos={sc_pos*100:.0f}%<15%(2h bottom->block SHORT)"
            if direction == 1 and sc_pos > 0.85:
                return False, f"QG2a:pos={sc_pos*100:.0f}%>85%(2h top->block LONG)"
            # 24h range position - thiet lap kep voi _block_long_24h, bat case borderline
            # strong_trend: relax threshold (strong bull trend cho phep long len toi 92% 24h range)
            _qg2b_long_thresh  = 0.92 if strong_trend else 0.80
            _qg2b_short_thresh = 0.08 if strong_trend else 0.20
            _qg2c_margin       = 0.05 if strong_trend else 0.025
            if direction == 1 and pos_24h > _qg2b_long_thresh:
                return False, f"QG2b:24h_pos={pos_24h*100:.0f}%>{_qg2b_long_thresh*100:.0f}%(gan dinh 24h->block LONG)"
            if direction == -1 and pos_24h < _qg2b_short_thresh:
                return False, f"QG2b:24h_pos={pos_24h*100:.0f}%<{_qg2b_short_thresh*100:.0f}%(gan day 24h->block SHORT)"
            # Distance from 24h absolute high/low
            if direction == 1 and hi_24h > 0 and price >= hi_24h * (1 - _qg2c_margin):
                return False, f"QG2c:within {_qg2c_margin*100:.0f}% of 24h high={hi_24h:.6g}(->block LONG)"
            if direction == -1 and lo_24h > 0 and price <= lo_24h * (1 + _qg2c_margin):
                return False, f"QG2c:within {_qg2c_margin*100:.0f}% of 24h low={lo_24h:.6g}(->block SHORT)"

        _hf = getattr(config, "HIGH_FREQ_MODE", False)

        # QG-3: 30m high/low proximity (skip breakout; skip in HF mode)
        # HF mode: gia trong trend LUON gan 30m high/low -> QG3 block het moi entry trend
        # -> bypass de bat momentum move (trend alignment gate xu ly sai trend)
        if not _hf and not is_breakout and len(df_micro) >= 30:
            _lo30 = float(df_micro["low"].iloc[-30:].min())
            _hi30 = float(df_micro["high"].iloc[-30:].max())
            _qg3_pct = 0.005 * sp  # 0.25% largecap, 0.375% midcap, 0.5% altcoin
            if direction == -1 and _lo30 > 0 and price <= _lo30 * (1 + _qg3_pct):
                return False, f"QG3:within {_qg3_pct*100:.2f}% of 30m low={_lo30:.6g}(->block SHORT)"
            if direction == 1 and _hi30 > 0 and price >= _hi30 * (1 - _qg3_pct):
                return False, f"QG3:within {_qg3_pct*100:.2f}% of 30m high={_hi30:.6g}(->block LONG)"

        # QG-4: Immediate momentum - chi block khi gia DANG GIAM MANH THUC SU (PEOPLEUSDT pattern)
        # HF mode: threshold cao hon (5x) tranh block micro-pullback trong trend
        # trend_confirmed: bypass hoan toan (pullback trong confirmed trend = ok)
        # Chi block: gia giam NHANH va MANH lien tuc (khong phai dip 1-2 nen)
        if not is_reversal and not trend_confirmed and len(df_micro) >= 10:
            _c5  = float(df_micro["close"].iloc[-5:].mean())
            _c10 = float(df_micro["close"].iloc[-10:-5].mean())
            if _c10 > 0:
                _imm = (_c5 - _c10) / _c10
                # HF: nguong 0.5% largecap / 0.75% midcap / 1% altcoin - chi bat PEOPLEUSDT kieu giam 6 phut lien
                # Non-HF: nguong 0.1% cu
                _thresh = 0.010 * sp if _hf else 0.002 * sp
                if direction == -1 and _imm > _thresh:
                    return False, f"QG4:imm_mom=+{_imm*100:.3f}%(bouncing->block SHORT)"
                if direction == 1 and _imm < -_thresh:
                    return False, f"QG4:imm_mom={_imm*100:.3f}%(falling->block LONG)"

        # QG-5: EMA21 overstretch (HF: loosen to 3%/6% so only extreme cases block)
        if len(df_micro) >= 21:
            _ema21 = float(compute_ema(df_micro["close"], 21).iloc[-1])
            if _ema21 > 0:
                _stretch = (price - _ema21) / _ema21
                if _hf:
                    _lim = 0.06 * sp  # HF: 3% BTC, 6% altcoin - chi block khi that su qua gian
                else:
                    _lim = (0.025 if is_breakout else 0.015) * sp
                if direction == -1 and _stretch < -_lim:
                    return False, f"QG5:price {_stretch*100:.2f}% below EMA21(overstretched->block SHORT)"
                if direction == 1 and _stretch > _lim:
                    return False, f"QG5:price {_stretch*100:.2f}% above EMA21(overstretched->block LONG)"

        # QG-6: Last 3 candle bodies strongly against direction (skip reversal; bypass in confirmed trend)
        if not trend_confirmed and not is_reversal and len(df_micro) >= 4:
            _bodies = (df_micro["close"].iloc[-3:].values
                       - df_micro["open"].iloc[-3:].values)
            _ranges = (df_micro["high"].iloc[-3:].values
                       - df_micro["low"].iloc[-3:].values)
            _total_range = float(_ranges.sum())
            if _total_range > 0:
                _body_ratio = float(_bodies.sum()) / _total_range
                if direction == -1 and _body_ratio > 0.55:
                    return False, f"QG6:3c bullish ratio={_body_ratio:.2f}(->block SHORT)"
                if direction == 1 and _body_ratio < -0.55:
                    return False, f"QG6:3c bearish ratio={_body_ratio:.2f}(->block LONG)"

        return True, ""

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

        # Volatility bucket dua tren VOLUME THUC TE tu scanner (khong chi BTC/ETH hardcode):
        # high_vol (>= 10M USDT/24h): coin lon, ATR% nho hon, dung nguong mem hon
        # mid_vol  (1M-10M):          volatility trung binh
        # low_vol  (< 1M):            altcoin volatility cao, dung nguong chat hon
        _coin_vol = self.scanner.volume_map.get(symbol, 0)
        _LARGECAP = {"BTCUSDT", "ETHUSDT"}   # giu cho risk_manager ATR mult compat
        _MIDCAP   = {"SOLUSDT", "XRPUSDT", "HYPEUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT", "TRXUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT"}
        _is_largecap = symbol in _LARGECAP
        # _is_high_vol: bat ca coin >= 10M USDT volume - khong chi BTC/ETH
        _is_high_vol = _coin_vol >= config.HIGH_VOL_THRESHOLD   # 10M
        _sp = 0.50 if symbol in _LARGECAP else (0.75 if (symbol in _MIDCAP or _is_high_vol) else 1.0)

        # 2h range position - tinh som de dung o quality gate cho tat ca paths
        # (se duoc tinh lai chinh xac hon trong scenario block, nhung gia tri nay dung cho gate)
        _sc_pos = 0.5
        if len(df_signal) >= 120:
            _qg_hi = float(df_signal["high"].iloc[-120:].max())
            _qg_lo = float(df_signal["low"].iloc[-120:].min())
            _qg_rng = _qg_hi - _qg_lo
            if _qg_rng > 0:
                _sc_pos = (float(df_signal["close"].iloc[-1]) - _qg_lo) / _qg_rng

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

        # --- Post-loss / Post-win cooldown & flip-guard --------------------------------
        import time as _time_mod
        _now_ts = _time_mod.time()
        _loss_cooldown_sec = getattr(config, "LOSS_COOLDOWN_SEC", 7200)
        _win_cooldown_sec  = getattr(config, "WIN_COOLDOWN_SEC",  120)
        _flip_cooldown_sec = getattr(config, "FLIP_COOLDOWN_SEC", 7200)
        _loss_ts = self.executor._loss_cooldown.get(symbol, 0.0)
        if _now_ts - _loss_ts < _loss_cooldown_sec:
            _rem = int(_loss_ts + _loss_cooldown_sec - _now_ts)
            logger.debug(f"{symbol}: skip - loss cooldown ({_rem}s remaining)")
            return False
        # Win cooldown: sau dong lenh loi, momentum da can - tranh re-entry ngay (PRLUSDT pattern)
        _win_ts = self.executor._win_cooldown.get(symbol, 0.0)
        if _now_ts - _win_ts < _win_cooldown_sec:
            _rem = int(_win_ts + _win_cooldown_sec - _now_ts)
            logger.debug(f"{symbol}: skip - win cooldown ({_rem}s remaining)")
            return False
        # ------------------------------------------------------------------------------

        # 24h directional move filter: tranh chase sau khi coin da pump/dump trong 24h
        # 1440 nen 1m = 1440 phut = 24h chinh xac (chinh xac hon 96x15m vi du lieu 1m granular)
        _W24H = 1440
        _block_long_24h  = False
        _block_short_24h = False
        _change_24h = 0.0
        _24h_pos    = 0.5   # vi tri gia trong 24h range [0..1], 1=dinh, 0=day
        _24h_hi     = 0.0
        _24h_lo     = 0.0
        _24h_rng    = 0.0
        if len(df_signal) >= _W24H:
            _ref_24h = df_signal["close"].iloc[-_W24H]
            if _ref_24h > 0:
                _change_24h = (df_signal["close"].iloc[-1] - _ref_24h) / _ref_24h * 100
                if abs(_change_24h) > 30:
                    logger.info(f"{symbol}: 24h change={_change_24h:.1f}% > 30% -> HARD SKIP (extreme move)")
                    return False
                # Nguong block 15%: se duoc RELAX sau khi macro_trend duoc tinh (xem TREND-RELAX bên dưới)
                # Nhung van set truoc de _24h_pos check ben duoi co the override
                if _change_24h > 15:
                    _block_long_24h = True
                    logger.debug(f"{symbol}: 24h change=+{_change_24h:.1f}% -> block LONG (pump exhausted)")
                elif _change_24h < -15:
                    _block_short_24h = True
                    logger.debug(f"{symbol}: 24h change={_change_24h:.1f}% -> block SHORT (dump exhausted)")
            # Vi tri gia trong 24h high/low range: quan trong hon % change vi bat duoc truong hop
            # coin da pump tu truoc 24h nhung hien tai van o dinh (nhu HUSDT +24% multi-day).
            # HUSDT: 24h change chi +9.75% (duoi nguong 20%) nhung price o 98% 24h range = dinh tuyet doi.
            _24h_hi = float(df_signal["high"].iloc[-_W24H:].max())
            _24h_lo = float(df_signal["low"].iloc[-_W24H:].min())
            _24h_rng = _24h_hi - _24h_lo
            _cur_p24  = float(df_signal["close"].iloc[-1])
            if _24h_rng > 0:
                _24h_pos = (_cur_p24 - _24h_lo) / _24h_rng
                # Giam nguong tu 88%/12% xuong 82%/18% de bat case nhu PEOPLEUSDT
                # (bot vao long tai 88.1% cua 24h range - chi vuot nguong cu 0.1%)
                if _24h_pos > 0.82:
                    _block_long_24h = True
                    logger.debug(f"{symbol}: 24h range pos={_24h_pos:.0%} > 82% -> block LONG (gan dinh 24h)")
                elif _24h_pos < 0.18:
                    _block_short_24h = True
                    logger.debug(f"{symbol}: 24h range pos={_24h_pos:.0%} < 18% -> block SHORT (gan day 24h)")

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

        # _atrm: ATR tren df_micro (1m) - se duoc ghi de chinh xac hon trong scenario block.
        # Khoi tao som tranh UnboundLocalError khi scenario block bi skip (df_micro qua ngan).
        _atrm = _atr_for_sl if _atr_for_sl > 0 else 0.001

        # STRONG SUSTAINED TREND detection (QUSDT-pattern: pump +8%+ trong 24h, tat ca EMA aligned
        # va EMA21 van dang tang / giam nhanh = trend chua ket thuc, khong phai spike da xong).
        # Dung de: (1) block SHORT trong strong bull trend, (2) uu tien LONG entry pullback-to-EMA.
        _strong_bull_trend = False
        _strong_bear_trend = False
        if len(df_signal) >= 200:
            _strd_e21    = float(compute_ema(df_signal["close"], 21).iloc[-1])
            _strd_e50    = float(compute_ema(df_signal["close"], 50).iloc[-1])
            _strd_e100   = float(compute_ema(df_signal["close"], 100).iloc[-1])
            _strd_e21_p  = float(compute_ema(df_signal["close"], 21).iloc[-21])  # 21 nen truoc
            if _strd_e21_p > 0:
                _strd_accel = (_strd_e21 - _strd_e21_p) / _strd_e21_p
                # Bull: ca 3 EMA aligned up + EMA21 tang + 24h change > 3%
                # Nguong 8% cu qua cao: BTC/ETH thuong chi tang 2-5%/ngay -> never trigger
                # Giam xuong 3% de bat largecap trong normal bull run
                # High-vol (>= 10M): 24h tang cham hon altcoin -> nguong thap hon
                _bull_24h_min = 3.0 if _is_high_vol else 5.0
                _bear_24h_max = -3.0 if _is_high_vol else -5.0
                _bull_accel_min = 0.001 if _is_high_vol else 0.003
                if (macro_trend == 1 and macro_4h == 1
                        and _change_24h > _bull_24h_min
                        and _strd_e21 > _strd_e50 > _strd_e100
                        and _strd_accel > _bull_accel_min):
                    _strong_bull_trend = True
                    logger.debug(f"{symbol}: STRONG_BULL_TREND detected (24h={_change_24h:.1f}% accel={_strd_accel*100:.2f}%)")
                elif (macro_trend == -1 and macro_4h == -1
                        and _change_24h < _bear_24h_max
                        and _strd_e21 < _strd_e50 < _strd_e100
                        and _strd_accel < -_bull_accel_min):
                    _strong_bear_trend = True
                    logger.debug(f"{symbol}: STRONG_BEAR_TREND detected (24h={_change_24h:.1f}% accel={_strd_accel*100:.2f}%)")

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
                _h1_long_thresh  = 0.80 if (is_priority and macro_trend >= 1) else 0.70
                _h1_short_thresh = 0.20 if (is_priority and macro_trend <= -1) else 0.30
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

        # TREND RELAXATION: khi CA HAI macro TF xac nhan trend, relax range blocks qua strict.
        # Trong uptrend manh, gia LIEN TUC o top of range -> block_long_24h/h1/2h luon True
        # -> bot khong trade gi ca du trend ro rang. Thuc te: price o top = uptrend khoe,
        # KHONG phai pump exhausted (exhausted chi khi 24h change > 22% hoac o sat tuyet dinh).
        # High-vol coins (>= 10M): macro_4h (EMA 300/600 on 1m) qua cham flip khi trend moi bat dau
        # -> dung chi macro_trend (EMA 100/250) cho ca coin lon de bat trend som hon
        # Altcoin nho: van can ca 2 TF de tranh false signal
        _all_tfs_bull = (macro_trend == 1 and macro_4h == 1) or (_is_high_vol and macro_trend == 1)
        _all_tfs_bear = (macro_trend == -1 and macro_4h == -1) or (_is_high_vol and macro_trend == -1)
        if _all_tfs_bull and _block_long_24h:
            # Chi giu block khi THUC SU spike extreme: pump >22% trong 24h HOAC o 93%+ range
            if _change_24h <= 22.0 and _24h_pos <= 0.93:
                _block_long_24h = False
                logger.debug(f"{symbol}: TREND-RELAX block_long_24h cleared (macro=bull pos={_24h_pos:.0%} chg={_change_24h:.1f}%)")
        if _all_tfs_bear and _block_short_24h:
            if _change_24h >= -22.0 and _24h_pos >= 0.07:
                _block_short_24h = False
                logger.debug(f"{symbol}: TREND-RELAX block_short_24h cleared (macro=bear pos={_24h_pos:.0%} chg={_change_24h:.1f}%)")

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
                and not _spike_dump_60c   # tranh dead-cat bounce sau dump spike
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

            # Consecutive candles block: high-vol (>=10M) 10 nen (BTC/SOL uptrend co 6+ green 1m binh thuong)
            # Altcoin nho: 8 nen lien tiep = exhaustion / dao chieu
            # scalp_trend bypass: neu 5m xac nhan cung chieu -> la trend that, khong phai exhaustion
            _consec_n = 10 if _is_high_vol else 8
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
                    _bo_last = self.executor._last_direction.get(symbol)
                    if _bo_last is not None:
                        import time as _tm_bo
                        _bo_ldir, _bo_lts = _bo_last
                        if _tm_bo.time() - _bo_lts < _flip_cooldown_sec and _bo_ldir != bo_sig.direction:
                            logger.info(f"{symbol}: FLIP-GUARD block BREAKOUT {'LONG' if bo_sig.direction==1 else 'SHORT'}")
                            return False
                    _bo_qg_ok, _bo_qg_msg = self._quality_gate(
                        bo_sig.direction, df_micro, rsi_now, _sc_pos, _sp, is_breakout=True,
                        pos_24h=_24h_pos, hi_24h=_24h_hi, lo_24h=_24h_lo,
                        strong_trend=(_strong_bull_trend and bo_sig.direction == 1) or (_strong_bear_trend and bo_sig.direction == -1))
                    if not _bo_qg_ok:
                        logger.info(f"{symbol}: [QUALITY-GATE] BREAKOUT blocked - {_bo_qg_msg}")
                        return False
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

                # Strong trend guard: trong strong bull/bear trend, block SHORT/LONG tru khi RSI cuc cao/thap
                if sig.direction == -1 and _strong_bull_trend and rsi_now < 78:
                    logger.debug(f"{symbol}: strong-trend guard block SHORT in STRONG_BULL_TREND (RSI={rsi_now:.1f})")
                    continue
                if sig.direction == 1 and _strong_bear_trend and rsi_now > 22:
                    logger.debug(f"{symbol}: strong-trend guard block LONG in STRONG_BEAR_TREND (RSI={rsi_now:.1f})")
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
                    # 5m alignment pre-filter: khong dem signal khi 5m nguoc chieu (ALTCOIN NHO ONLY)
                    # High-vol (>= 10M): 5m corrections trong 1h trend la BINH THUONG (buy dip / sell bounce)
                    # -> khong apply cho high-vol, dung 1m micro check (micro_up/down + EMA9/21) thay the
                    # Altcoin nho: 5m bounce trong 1h downtrend = timing xau cho SHORT -> bo qua
                    if _is_high_vol:
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
                    _rv_last = self.executor._last_direction.get(symbol)
                    if _rv_last is not None:
                        import time as _tm_rv
                        _rv_ldir, _rv_lts = _rv_last
                        if _tm_rv.time() - _rv_lts < _flip_cooldown_sec and _rv_ldir != best.direction:
                            logger.info(f"{symbol}: FLIP-GUARD block REVERSAL {'LONG' if best.direction==1 else 'SHORT'}")
                            return False
                    _rv_qg_ok, _rv_qg_msg = self._quality_gate(
                        best.direction, df_micro, rsi_now, _sc_pos, _sp, is_reversal=True,
                        pos_24h=_24h_pos, hi_24h=_24h_hi, lo_24h=_24h_lo,
                        strong_trend=(_strong_bull_trend and best.direction == 1) or (_strong_bear_trend and best.direction == -1))
                    if not _rv_qg_ok:
                        logger.info(f"{symbol}: [QUALITY-GATE] REVERSAL blocked - {_rv_qg_msg}")
                        return False
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
            # ANTI-CHOP cho SCENARIO: HF mode dung nguong thap hon (6), normal >= 14
            _hf_mode = getattr(config, "HIGH_FREQ_MODE", False)
            _sc_adx_min = 6.0 if _hf_mode else 14.0
            if math.isnan(adx) or adx < _sc_adx_min:
                logger.debug(f"{symbol}: scenario skip - ADX={adx:.1f} < {_sc_adx_min} (chop)")
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
            _fast_tr_pre = 0  # default; overwritten at line ~3800 before gate section
            if not df_micro.empty and len(df_micro) >= 120:
                _sc_close  = df_micro["close"]
                _sc_price  = _range_live_price if _range_live_price > 0 else _sc_close.iloc[-1]
                # _atrm: ATR cho tinh toan trong scenario engine (S53-S84 dung de scale move/wick)
                _atrm = float(compute_atr(df_micro, 14).iloc[-1]) if len(df_micro) >= 14 else (_atr_for_sl if _atr_for_sl > 0 else 0.001)
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

                # Immediate momentum: close trung binh 5 nen cuoi vs 5 nen truoc do
                # > 0 = nen cuoi dang tang (bounce), < 0 = dang giam
                _imm_close5  = df_micro["close"].iloc[-5:].mean()
                _imm_close10 = df_micro["close"].iloc[-10:-5].mean()
                _imm_mom     = (_imm_close5 - _imm_close10) / _imm_close10 if _imm_close10 > 0 else 0.0

                # S1/S2 - EMERGING TREND (uu tien cao nhat - chinh la HBAR pattern)
                # _sc_pos guard: tranh long o gan dinh 2h range (>82%) hoac short o gan day (< 18%)
                # _imm_mom guard: tranh short khi gia dang bounce (>+0.05%) hoac long khi dang giam (<-0.05%)
                if _emerging_uptrend and not _block_long_24h and _sc_pos < 0.82 and _imm_mom > -0.0005:
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.62, 0.0, "sc_emerging_up"
                elif _emerging_downtrend and not _block_short_24h and _sc_pos > 0.18 and _imm_mom < 0.0005:
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
                # GUARD THEM (TNSR pattern): loai bo pullback sau pump/dump manh:
                #   1. volume khong bi collapse: _sc_vol_surge >= 0.35
                #   2. khong co pump/dump lon truoc do trong 30c (> 2.5% so EMA21)
                elif ((macro_trend == 1 and macro_4h >= 0) or (_is_gradual_uptrend and scalp_trend == 1)) \
                        and _emg_ema50 > 0 and _emg_ema50 * 0.999 <= _sc_price <= _emg_ema21 * 1.0015 \
                        and _sc_last_green and 35 <= rsi_now <= 65 and not _block_long_24h \
                        and _sc_vol_surge >= 0.35 \
                        and (_emg_ema21 <= 0 or df_micro["high"].iloc[-30:].max() <= _emg_ema21 * 1.025):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.60, 0.0, "sc_pullback_up"
                elif ((macro_trend == -1 and macro_4h <= 0) or (_is_gradual_downtrend and scalp_trend == -1)) \
                        and _emg_ema21 > 0 and _emg_ema21 * 0.9985 <= _sc_price <= _emg_ema50 * 1.001 \
                        and _sc_last_red and 35 <= rsi_now <= 65 and not _block_short_24h \
                        and _sc_vol_surge >= 0.35 \
                        and (_emg_ema21 <= 0 or df_micro["low"].iloc[-30:].min() >= _emg_ema21 * 0.975):
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
                # S8b/S8c - STRONG TREND PULLBACK: trong xu huong manh (>8% 24h, EMA aligned),
                # cho gia keo ve EMA21 roi vao long/short theo xu huong. Bypass _block_long/short_24h
                # vi 24h_pos se cao (~70-92%) trong strong bull. RSI cho ve 30-65 truoc khi entry.
                elif _strong_bull_trend and _sc_dir == 0:
                    _near_ema21 = (_emg_ema21 > 0
                                   and _emg_ema21 * 0.997 <= _sc_price <= _emg_ema21 * 1.008)
                    # imm_mom phai duong ro rang (khong vao khi gia dang roi) + 2 nen cuoi xanh
                    _pb_long_2green = (len(df_micro) >= 2
                                       and df_micro["close"].iloc[-1] > df_micro["open"].iloc[-1]
                                       and df_micro["close"].iloc[-2] > df_micro["open"].iloc[-2])
                    if (_near_ema21 and _pb_long_2green and 30 <= rsi_now <= 65
                            and _24h_pos < 0.88 and _sc_pos <= 0.65 and _imm_mom > 0.0003):
                        _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.70, 0.0, "sc_trend_pullback_long"
                elif _strong_bear_trend and _sc_dir == 0:
                    _near_ema21_bear = (_emg_ema21 > 0
                                        and _emg_ema21 * 0.992 <= _sc_price <= _emg_ema21 * 1.003)
                    _pb_short_2red = (len(df_micro) >= 2
                                      and df_micro["close"].iloc[-1] < df_micro["open"].iloc[-1]
                                      and df_micro["close"].iloc[-2] < df_micro["open"].iloc[-2])
                    if (_near_ema21_bear and _pb_short_2red and 35 <= rsi_now <= 70
                            and _24h_pos > 0.12 and _sc_pos >= 0.35 and _imm_mom < -0.0003):
                        _sc_dir, _sc_strength, _sc_tp, _sc_name = -1, 0.70, 0.0, "sc_trend_pullback_short"

            # ================================================================
            # S9-S40: EXTENDED SCENARIO ENGINE
            # Moi nhom S9+ chay tuan tu (if _sc_dir == 0) sau khi S1-S8 that bai.
            # Moi nhom co guard chong du-dinh: RSI, price_pos, macro alignment.
            # ================================================================

            # S9/S10: EMA21/50 GOLDEN/DEATH CROSS tren 1m
            # Golden: EMA21 vua vuot EMA50 tu duoi len (trong 6 nen), xac nhan macro >= 0
            # Anti-du-dinh: price khong cach cross point > 1.5%, RSI < 75
            if _sc_dir == 0 and len(df_micro) >= 55:
                _e21s = compute_ema(df_micro["close"], 21)
                _e50s = compute_ema(df_micro["close"], 50)
                _x_up  = _e21s.iloc[-1] > _e50s.iloc[-1] and any(
                    _e21s.iloc[i] <= _e50s.iloc[i] for i in range(-7, -1))
                _x_dn  = _e21s.iloc[-1] < _e50s.iloc[-1] and any(
                    _e21s.iloc[i] >= _e50s.iloc[i] for i in range(-7, -1))
                _e50v   = _e50s.iloc[-1]
                _dist50 = abs(_sc_price - _e50v) / _e50v if _e50v > 0 else 1.0
                if (_x_up and not _block_long_24h and 30 < rsi_now < 75
                        and macro_4h >= 0 and _sc_vol_surge >= 0.9 and _dist50 < 0.015
                        and _sc_pos < 0.82):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.63, 0.0, "sc_ema_golden_cross"
                elif (_x_dn and not _block_short_24h and 25 < rsi_now < 70
                        and macro_4h <= 0 and _sc_vol_surge >= 0.9 and _dist50 < 0.015
                        and _sc_pos > 0.18):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = -1, 0.63, 0.0, "sc_ema_death_cross"

            # S11/S12: 4 TIMEFRAME ALIGNMENT (micro + scalp + macro + macro_4h tat ca dong long)
            # Day la signal manh nhat: tat ca khung thoi gian cung chieu
            if _sc_dir == 0:
                _4tf_long  = (micro_up and scalp_trend == 1 and macro_trend == 1 and macro_4h == 1
                              and not _block_long_24h and 35 < rsi_now < 78
                              and _sc_vol_surge >= 0.7 and _sc_pos < 0.82)
                _4tf_short = (micro_down and scalp_trend == -1 and macro_trend == -1 and macro_4h == -1
                              and not _block_short_24h and 22 < rsi_now < 65
                              and _sc_vol_surge >= 0.7 and _sc_pos > 0.18)
                if _4tf_long:
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.70, 0.0, "sc_4tf_aligned_long"
                elif _4tf_short:
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = -1, 0.70, 0.0, "sc_4tf_aligned_short"

            # S13/S14: MOMENTUM BURST - 5+ nen lien tiep cung chieu, volume tang dan
            # Anti-du-dinh: _sc_pos < 0.85 (khong o dinh range), RSI < 78
            if _sc_dir == 0 and len(df_micro) >= 12:
                _mb8c = df_micro["close"].iloc[-8:].values
                _mb8o = df_micro["open"].iloc[-8:].values
                _mb8v = df_micro["volume"].iloc[-8:].values
                _mb_cg = 0; _mb_cr = 0
                for _mbi in range(7, -1, -1):
                    if _mb8c[_mbi] > _mb8o[_mbi]: _mb_cg += 1
                    else: break
                for _mbi in range(7, -1, -1):
                    if _mb8c[_mbi] < _mb8o[_mbi]: _mb_cr += 1
                    else: break
                _mb_vacc = _mb8v[-3:].mean() >= _mb8v[-6:-3].mean() * 0.85
                if (_mb_cg >= 5 and not _block_long_24h and rsi_now < 78 and _mb_vacc
                        and macro_trend >= 0 and _sc_vol_surge >= 0.7 and _sc_pos < 0.85):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.62, 0.0, "sc_momentum_burst_long"
                elif (_mb_cr >= 5 and not _block_short_24h and rsi_now > 22 and _mb_vacc
                        and macro_trend <= 0 and _sc_vol_surge >= 0.7 and _sc_pos > 0.15):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = -1, 0.62, 0.0, "sc_momentum_burst_short"

            # S15/S16: BOLLINGER SQUEEZE RELEASE
            # BB cang dat (width thap) roi mo rong dot ngot + break band = breakout tin cy cao
            if _sc_dir == 0 and len(df_micro) >= 35:
                _bb_c   = df_micro["close"]
                _bb_m   = _bb_c.rolling(20).mean()
                _bb_s   = _bb_c.rolling(20).std()
                _bb_w   = (2 * _bb_s / _bb_m.replace(0, float("nan")))
                _bb_now = float(_bb_w.iloc[-1]) if not _bb_w.iloc[-1:].isna().any() else 1.0
                _bb_min = float(_bb_w.iloc[-20:-1].min()) if len(_bb_w) >= 20 else 1.0
                _bb_up  = float((_bb_m + 2 * _bb_s).iloc[-1])
                _bb_lo  = float((_bb_m - 2 * _bb_s).iloc[-1])
                _sq_was = _bb_min < 0.025   # was squeezed
                _sq_now = _bb_now > _bb_min * 1.4  # now expanding
                if (_sq_was and _sq_now and _sc_price >= _bb_up * 0.997 and not _block_long_24h
                        and rsi_now < 80 and macro_trend >= 0 and _sc_vol_surge >= 1.2
                        and _sc_pos < 0.88):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.64, 0.0, "sc_bb_squeeze_up"
                elif (_sq_was and _sq_now and _sc_price <= _bb_lo * 1.003 and not _block_short_24h
                        and rsi_now > 20 and macro_trend <= 0 and _sc_vol_surge >= 1.2
                        and _sc_pos > 0.12):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = -1, 0.64, 0.0, "sc_bb_squeeze_down"

            # S17/S18: VWAP RECLAIM - gia cắt qua VWAP voi volume
            # VWAP tinh tren 200 nen = session anchor, moc quan trong
            if _sc_dir == 0 and len(df_micro) >= 50:
                _vw_n = min(200, len(df_micro))
                _vw_df = df_micro.iloc[-_vw_n:]
                _vw_den = float(_vw_df["volume"].sum())
                _vwap_v = float((_vw_df["close"] * _vw_df["volume"]).sum()) / _vw_den if _vw_den > 0 else 0.0
                if _vwap_v > 0:
                    _vw_above_now  = _sc_price > _vwap_v
                    _vw_above_p4   = float(df_micro["close"].iloc[-5]) > _vwap_v
                    _vw_dist       = abs(_sc_price - _vwap_v) / _vwap_v
                    _just_cross_up = _vw_above_now and not _vw_above_p4
                    _just_cross_dn = not _vw_above_now and _vw_above_p4
                    if (_just_cross_up and not _block_long_24h and rsi_now < 72 and macro_trend >= 0
                            and _sc_vol_surge >= 1.1 and _vw_dist < 0.010 and _sc_pos < 0.80):
                        _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.62, 0.0, "sc_vwap_reclaim_long"
                    elif (_just_cross_dn and not _block_short_24h and rsi_now > 28 and macro_trend <= 0
                            and _sc_vol_surge >= 1.1 and _vw_dist < 0.010 and _sc_pos > 0.20):
                        _sc_dir, _sc_strength, _sc_tp, _sc_name = -1, 0.62, 0.0, "sc_vwap_reclaim_short"

            # S19/S20: RSI RECOVERY TRONG TREND (RSI dip roi bat len trong uptrend)
            # Pattern: uptrend -> RSI pullback < 45 -> RSI bat len >= 48 = entry tot
            if _sc_dir == 0 and len(df_micro) >= 25:
                _rsr = compute_rsi(df_micro["close"], 14)
                _rsr_now  = float(_rsr.iloc[-1])
                _rsr_min  = float(_rsr.iloc[-12:-1].min())
                _rsr_prev = float(_rsr.iloc[-4])
                _rsr_rising  = _rsr_now > _rsr_prev + 1.5
                _rsr_falling = _rsr_now < _rsr_prev - 1.5
                if (macro_trend == 1 and macro_4h >= 0 and _rsr_min < 45 and _rsr_now >= 48
                        and _rsr_rising and _sc_pos < 0.75 and not _block_long_24h
                        and _sc_vol_surge >= 0.6):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.61, 0.0, "sc_rsi_recovery_long"
                elif (macro_trend == -1 and macro_4h <= 0 and _rsr_min > 55 and _rsr_now <= 52
                        and _rsr_falling and _sc_pos > 0.25 and not _block_short_24h
                        and _sc_vol_surge >= 0.6):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = -1, 0.61, 0.0, "sc_rsi_recovery_short"

            # S21/S22: EMA100/200 LEVEL BOUNCE (key support/resistance tren 1m)
            # Price cham EMA100 hoac EMA200 + nen xac nhan quay dau = entry chinh xac
            if _sc_dir == 0 and len(df_micro) >= 205:
                _el_e100 = float(compute_ema(df_micro["close"], 100).iloc[-1])
                _el_e200 = float(compute_ema(df_micro["close"], 200).iloc[-1])
                _el_atr  = float(compute_atr(df_micro, 14).iloc[-1])
                _el_tol  = _el_atr * 0.8
                _at_e100_sup = abs(_sc_price - _el_e100) < _el_tol and macro_trend >= 0
                _at_e200_sup = abs(_sc_price - _el_e200) < _el_tol and macro_trend >= 0
                _at_e100_res = abs(_sc_price - _el_e100) < _el_tol and macro_trend <= 0
                _at_e200_res = abs(_sc_price - _el_e200) < _el_tol and macro_trend <= 0
                _lc2 = df_micro.iloc[-2]
                _lc2_hammer  = (_lc2["close"] > _lc2["open"] and
                                 (min(_lc2["close"], _lc2["open"]) - _lc2["low"])
                                 > abs(_lc2["close"] - _lc2["open"]) * 1.5)
                _lc2_star    = (_lc2["close"] < _lc2["open"] and
                                 (_lc2["high"] - max(_lc2["close"], _lc2["open"]))
                                 > abs(_lc2["close"] - _lc2["open"]) * 1.5)
                if ((_at_e100_sup or _at_e200_sup) and _lc2_hammer and not _block_long_24h
                        and rsi_now < 70 and _sc_vol_surge >= 0.8):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.63, 0.0, "sc_ema_level_bounce_long"
                elif ((_at_e100_res or _at_e200_res) and _lc2_star and not _block_short_24h
                        and rsi_now > 30 and _sc_vol_surge >= 0.8):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = -1, 0.63, 0.0, "sc_ema_level_bounce_short"

            # S23/S24: THREE WHITE SOLDIERS / THREE BLACK CROWS
            # 3 nen lien tiep day du than (body > 60% range), moi nen cao/thap hon nen truoc
            if _sc_dir == 0 and len(df_micro) >= 10:
                _tw_c = df_micro["close"].iloc[-4:-1].values
                _tw_o = df_micro["open"].iloc[-4:-1].values
                _tw_h = df_micro["high"].iloc[-4:-1].values
                _tw_l = df_micro["low"].iloc[-4:-1].values
                _tw_v = df_micro["volume"].iloc[-4:-1].values
                _tw_avg = float(df_micro["volume"].iloc[-10:-4].mean()) if len(df_micro) >= 10 else 1.0
                _tw_3g  = all(_tw_c[i] > _tw_o[i] for i in range(3))
                _tw_3r  = all(_tw_c[i] < _tw_o[i] for i in range(3))
                _tw_cli = _tw_c[0] < _tw_c[1] < _tw_c[2]
                _tw_drp = _tw_c[0] > _tw_c[1] > _tw_c[2]
                _tw_bod = [abs(_tw_c[i] - _tw_o[i]) / ((_tw_h[i] - _tw_l[i]) + 1e-9) for i in range(3)]
                _tw_fbody = all(b > 0.55 for b in _tw_bod)
                _tw_vok   = _tw_v.mean() >= _tw_avg * 0.65
                if (_tw_3g and _tw_cli and _tw_fbody and _tw_vok and not _block_long_24h
                        and rsi_now < 78 and macro_trend >= 0 and _sc_pos < 0.85):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.63, 0.0, "sc_three_soldiers"
                elif (_tw_3r and _tw_drp and _tw_fbody and _tw_vok and not _block_short_24h
                        and rsi_now > 22 and macro_trend <= 0 and _sc_pos > 0.15):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = -1, 0.63, 0.0, "sc_three_crows"

            # S25/S26: INSIDE BAR CONSOLIDATION BREAK
            # 7 nen trong tam range chat (< 0.8% hoac < 1.5x ATR), sau do break pha vo voi vol
            if _sc_dir == 0 and len(df_micro) >= 15:
                _ib_hi = float(df_micro["high"].iloc[-8:-1].max())
                _ib_lo = float(df_micro["low"].iloc[-8:-1].min())
                _ib_rng = _ib_hi - _ib_lo
                _ib_atr = float(compute_atr(df_micro, 14).iloc[-1]) if len(df_micro) >= 14 else 0.0
                _ib_tight = ((_ib_rng / _sc_price < 0.008 if _sc_price > 0 else False)
                             or (_ib_atr > 0 and _ib_rng < _ib_atr * 1.5))
                _ib_break_up = _sc_price > _ib_hi * 1.001 and _sc_last_green
                _ib_break_dn = _sc_price < _ib_lo * 0.999 and _sc_last_red
                if (_ib_tight and _ib_break_up and not _block_long_24h and rsi_now < 80
                        and _sc_vol_surge >= 1.3 and macro_trend >= 0 and _sc_pos < 0.86):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.64, 0.0, "sc_inside_bar_break_up"
                elif (_ib_tight and _ib_break_dn and not _block_short_24h and rsi_now > 20
                        and _sc_vol_surge >= 1.3 and macro_trend <= 0 and _sc_pos > 0.14):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = -1, 0.64, 0.0, "sc_inside_bar_break_down"

            # S27/S28: BULLISH/BEARISH ENGULFING tai muc gia co y nghia
            # Nen hien tai bao tron nen truoc (body lon hon) + volume xac nhan
            if _sc_dir == 0 and len(df_micro) >= 8:
                _eg_c1 = float(df_micro["close"].iloc[-2]); _eg_o1 = float(df_micro["open"].iloc[-2])
                _eg_c0 = float(df_micro["close"].iloc[-1]); _eg_o0 = float(df_micro["open"].iloc[-1])
                _eg_bh1 = max(_eg_c1, _eg_o1); _eg_bl1 = min(_eg_c1, _eg_o1)
                _eg_bh0 = max(_eg_c0, _eg_o0); _eg_bl0 = min(_eg_c0, _eg_o0)
                _eg_body1 = abs(_eg_c1 - _eg_o1); _eg_body0 = abs(_eg_c0 - _eg_o0)
                _eg_bull = (_eg_c0 > _eg_o0 and _eg_bh0 > _eg_bh1 and _eg_bl0 < _eg_bl1
                            and _eg_body0 > _eg_body1 * 1.2 and _eg_c1 < _eg_o1)
                _eg_bear = (_eg_c0 < _eg_o0 and _eg_bh0 > _eg_bh1 and _eg_bl0 < _eg_bl1
                            and _eg_body0 > _eg_body1 * 1.2 and _eg_c1 > _eg_o1)
                _eg_vok  = float(df_micro["volume"].iloc[-1]) >= float(df_micro["volume"].iloc[-6:-1].mean()) * 1.2
                if (_eg_bull and _eg_vok and not _block_long_24h and rsi_now < 72 and macro_trend >= 0
                        and _sc_pos < 0.82):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.61, 0.0, "sc_bull_engulf"
                elif (_eg_bear and _eg_vok and not _block_short_24h and rsi_now > 28 and macro_trend <= 0
                        and _sc_pos > 0.18):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = -1, 0.61, 0.0, "sc_bear_engulf"

            # S29/S30: MACD ZERO LINE CROSS (MACD cat duong 0 - xac nhan doi pha)
            # MACD cat len/xuong 0 = moment doi phe manh nhat, cho phep entry som
            if _sc_dir == 0 and len(df_micro) >= 45:
                _mz_line, _mz_sig, _mz_hist = compute_macd_series(df_micro["close"])
                _mz_cross_up = float(_mz_line.iloc[-1]) > 0 and float(_mz_line.iloc[-5]) < 0
                _mz_cross_dn = float(_mz_line.iloc[-1]) < 0 and float(_mz_line.iloc[-5]) > 0
                _mz_rising   = float(_mz_hist.iloc[-1]) > float(_mz_hist.iloc[-2])
                _mz_falling  = float(_mz_hist.iloc[-1]) < float(_mz_hist.iloc[-2])
                if (_mz_cross_up and _mz_rising and not _block_long_24h and rsi_now < 72
                        and macro_trend >= 0 and _sc_vol_surge >= 0.8 and _sc_pos < 0.82 and adx >= 12):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.61, 0.0, "sc_macd_zero_cross_long"
                elif (_mz_cross_dn and _mz_falling and not _block_short_24h and rsi_now > 28
                        and macro_trend <= 0 and _sc_vol_surge >= 0.8 and _sc_pos > 0.18 and adx >= 12):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = -1, 0.61, 0.0, "sc_macd_zero_cross_short"

            # S31/S32: HAMMER / SHOOTING STAR tai vung ho tro/khang cu
            # Hammer (wick duoi dai) tai day range, Shooting star (wick tren dai) tai dinh
            if _sc_dir == 0 and len(df_micro) >= 8:
                _hs_lc = df_micro.iloc[-2]
                _hs_rng = float(_hs_lc["high"]) - float(_hs_lc["low"])
                if _hs_rng > 0:
                    _hs_body   = abs(float(_hs_lc["close"]) - float(_hs_lc["open"]))
                    _hs_lo_wk  = min(float(_hs_lc["close"]), float(_hs_lc["open"])) - float(_hs_lc["low"])
                    _hs_hi_wk  = float(_hs_lc["high"]) - max(float(_hs_lc["close"]), float(_hs_lc["open"]))
                    _hs_hammer = (_hs_body > 0 and _hs_lo_wk >= _hs_body * 2.0
                                  and _hs_hi_wk < _hs_body * 0.6 and _sc_pos < 0.42)
                    _hs_star   = (_hs_body > 0 and _hs_hi_wk >= _hs_body * 2.0
                                  and _hs_lo_wk < _hs_body * 0.6 and _sc_pos > 0.58)
                    _hs_cfg    = float(df_micro["close"].iloc[-1]) > float(df_micro["open"].iloc[-1])
                    _hs_cfr    = float(df_micro["close"].iloc[-1]) < float(df_micro["open"].iloc[-1])
                    if (_hs_hammer and _hs_cfg and not _block_long_24h and rsi_now < 65 and macro_4h >= 0
                            and _sc_vol_surge >= 0.8):
                        _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.60, 0.0, "sc_hammer_bounce"
                    elif (_hs_star and _hs_cfr and not _block_short_24h and rsi_now > 35 and macro_4h <= 0
                            and _sc_vol_surge >= 0.8):
                        _sc_dir, _sc_strength, _sc_tp, _sc_name = -1, 0.60, 0.0, "sc_shooting_star"

            # S33/S34: DOUBLE BOTTOM / DOUBLE TOP (W-pattern / M-pattern)
            # Hai day/dinh xap xi nhau trong 60 nen, day thu 2 co RSI cao hon (bull div)
            if _sc_dir == 0 and len(df_micro) >= 65:
                _db_lo1 = float(df_micro["low"].iloc[-60:-35].min())
                _db_lo2 = float(df_micro["low"].iloc[-30:].min())
                _db_hi1 = float(df_micro["high"].iloc[-60:-35].max())
                _db_hi2 = float(df_micro["high"].iloc[-30:].max())
                _db_mid_hi = float(df_micro["high"].iloc[-45:-15].max())  # swing high giua 2 day
                _db_mid_lo = float(df_micro["low"].iloc[-45:-15].min())   # swing low giua 2 dinh
                _db_rsi = compute_rsi(df_micro["close"], 14)
                _db_lo1_i = int(df_micro["low"].iloc[-60:-35].argmin())
                _db_lo2_i = int(df_micro["low"].iloc[-30:].argmin())
                _db_rsi1  = float(_db_rsi.iloc[-60 + _db_lo1_i]) if len(_db_rsi) >= 60 else 50.0
                _db_rsi2  = float(_db_rsi.iloc[-30 + _db_lo2_i]) if len(_db_rsi) >= 30 else 50.0
                _db_match  = _db_lo1 > 0 and abs(_db_lo2 - _db_lo1) / _db_lo1 < 0.006
                _dt_match  = _db_hi1 > 0 and abs(_db_hi2 - _db_hi1) / _db_hi1 < 0.006
                _db_rsi_bul = _db_rsi2 > _db_rsi1 + 2
                _db_break_up   = _sc_price >= _db_mid_hi * 0.999 and _sc_last_green
                _db_break_down = _sc_price <= _db_mid_lo * 1.001 and _sc_last_red
                if (_db_match and _db_rsi_bul and _db_break_up and not _block_long_24h
                        and rsi_now < 72 and macro_4h >= 0 and _sc_vol_surge >= 1.0):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.64, 0.0, "sc_double_bottom"
                elif (_dt_match and not _db_rsi_bul and _db_break_down and not _block_short_24h
                        and rsi_now > 28 and macro_4h <= 0 and _sc_vol_surge >= 1.0):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = -1, 0.64, 0.0, "sc_double_top"

            # S35/S36: VOLUME ACCELERATION BURST (volume tang dan + cung chieu)
            # 4 nen lien tiep: moi nen volume lon hon nen truoc, tat ca cung chieu
            if _sc_dir == 0 and len(df_micro) >= 12:
                _va_v = df_micro["volume"].iloc[-5:].values
                _va_c = df_micro["close"].iloc[-5:].values
                _va_o = df_micro["open"].iloc[-5:].values
                _va_avg = float(df_micro["volume"].iloc[-15:-5].mean()) if len(df_micro) >= 15 else 0.0
                _va_green = all(_va_c[i] > _va_o[i] for i in range(5))
                _va_red   = all(_va_c[i] < _va_o[i] for i in range(5))
                _va_acc   = (_va_v[1] > _va_v[0] and _va_v[2] > _va_v[1] and _va_v[3] > _va_v[2])
                _va_hi    = _va_avg > 0 and _va_v[-3:].mean() >= _va_avg * 1.4
                if (_va_green and _va_acc and _va_hi and not _block_long_24h and rsi_now < 78
                        and macro_trend >= 0 and _sc_pos < 0.84):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.65, 0.0, "sc_vol_acceleration_long"
                elif (_va_red and _va_acc and _va_hi and not _block_short_24h and rsi_now > 22
                        and macro_trend <= 0 and _sc_pos > 0.16):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = -1, 0.65, 0.0, "sc_vol_acceleration_short"

            # S37/S38: TREND CHANNEL BOUNCE (EMA50 +/- 2xATR = Keltner channel)
            # Gia cham day kenh roi quay dau = mean-revert trong trend
            if _sc_dir == 0 and len(df_micro) >= 60:
                _kc_e50 = float(compute_ema(df_micro["close"], 50).iloc[-1])
                _kc_atr = float(compute_atr(df_micro, 14).iloc[-1])
                _kc_lo  = _kc_e50 - 2.0 * _kc_atr
                _kc_hi  = _kc_e50 + 2.0 * _kc_atr
                _kc_at_lo = _sc_price <= _kc_lo * 1.005
                _kc_at_hi = _sc_price >= _kc_hi * 0.995
                if (_kc_at_lo and _sc_last_green and macro_trend >= 0 and not _block_long_24h
                        and rsi_now < 65 and _sc_vol_surge >= 0.7):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.60, 0.0, "sc_channel_bounce_long"
                elif (_kc_at_hi and _sc_last_red and macro_trend <= 0 and not _block_short_24h
                        and rsi_now > 35 and _sc_vol_surge >= 0.7):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = -1, 0.60, 0.0, "sc_channel_bounce_short"

            # S39/S40: MACRO TREND CONTINUATION (ca 3 TF vung chac + pullback ket thuc)
            # 3 TF cao cung chieu + micro bat dau quay theo = entry pullback chinh xac
            if _sc_dir == 0:
                _mc_long  = (scalp_trend == 1 and macro_trend == 1 and macro_4h == 1
                             and micro_up and not _block_long_24h and 38 < rsi_now < 75
                             and _sc_vol_surge >= 0.7 and _sc_pos < 0.80 and adx >= 16)
                _mc_short = (scalp_trend == -1 and macro_trend == -1 and macro_4h == -1
                             and micro_down and not _block_short_24h and 25 < rsi_now < 62
                             and _sc_vol_surge >= 0.7 and _sc_pos > 0.20 and adx >= 16)
                if _mc_long:
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.67, 0.0, "sc_macro_cont_long"
                elif _mc_short:
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = -1, 0.67, 0.0, "sc_macro_cont_short"

            # S41/S42: STOCHASTIC CROSS trong vung trung tinh (20-80)
            # Stoch K cat duong D, khong o vung cuc doan = signal on dinh
            if _sc_dir == 0 and len(df_micro) >= 20:
                _stk_lo = float(df_micro["low"].iloc[-14:].min())
                _stk_rn = float(df_micro["high"].iloc[-14:].max()) - _stk_lo
                _stk_k  = ((float(df_micro["close"].iloc[-1]) - _stk_lo) / _stk_rn * 100) if _stk_rn > 0 else 50
                _stk_k2 = ((float(df_micro["close"].iloc[-3]) - _stk_lo) / _stk_rn * 100) if _stk_rn > 0 else 50
                _stk_d  = (_stk_k + _stk_k2 + ((float(df_micro["close"].iloc[-5]) - _stk_lo) / _stk_rn * 100 if _stk_rn > 0 else 50)) / 3
                _stk_cross_up = _stk_k > _stk_d and _stk_k2 <= _stk_d and 20 < _stk_k < 75
                _stk_cross_dn = _stk_k < _stk_d and _stk_k2 >= _stk_d and 25 < _stk_k < 80
                if (_stk_cross_up and macro_trend >= 0 and not _block_long_24h and rsi_now < 72
                        and _sc_vol_surge >= 0.7 and _sc_pos < 0.80):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.60, 0.0, "sc_stoch_cross_long"
                elif (_stk_cross_dn and macro_trend <= 0 and not _block_short_24h and rsi_now > 28
                        and _sc_vol_surge >= 0.7 and _sc_pos > 0.20):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = -1, 0.60, 0.0, "sc_stoch_cross_short"

            # S43/S44: POST-FLAT LAUNCH (range compression then explosive move)
            # Gia o vung rat flat (std < 0.08%) roi bat ngo tang/giam manh = thoat khoi range
            if _sc_dir == 0 and len(df_micro) >= 20:
                _pf_c10 = df_micro["close"].iloc[-12:-2]
                _pf_std = float(_pf_c10.std())
                _pf_mn  = float(_pf_c10.mean())
                _pf_flat = _pf_mn > 0 and _pf_std / _pf_mn < 0.0008
                _pf_cur_body = abs(float(df_micro["close"].iloc[-1]) - float(df_micro["open"].iloc[-1]))
                _pf_avg_body = float(abs(df_micro["close"].iloc[-12:-2] - df_micro["open"].iloc[-12:-2]).mean())
                _pf_burst = _pf_cur_body > _pf_avg_body * 2.5 if _pf_avg_body > 0 else False
                if (_pf_flat and _pf_burst and _sc_last_green and not _block_long_24h and rsi_now < 78
                        and _sc_vol_surge >= 1.5 and macro_trend >= 0 and _sc_pos < 0.84):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.64, 0.0, "sc_post_flat_launch_long"
                elif (_pf_flat and _pf_burst and _sc_last_red and not _block_short_24h and rsi_now > 22
                        and _sc_vol_surge >= 1.5 and macro_trend <= 0 and _sc_pos > 0.16):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = -1, 0.64, 0.0, "sc_post_flat_launch_short"

            # S45/S46: OBV (ON-BALANCE VOLUME) TREND BREAK
            # OBV tang len qua dinh truoc = mua tich luy, gia se theo sau
            if _sc_dir == 0 and len(df_micro) >= 50:
                _obv = (df_micro["volume"] * df_micro["close"].diff().apply(
                    lambda x: 1 if x > 0 else (-1 if x < 0 else 0))).cumsum()
                _obv_now  = float(_obv.iloc[-1])
                _obv_peak = float(_obv.iloc[-30:-5].max())
                _obv_trou = float(_obv.iloc[-30:-5].min())
                _obv_rising  = _obv_now > _obv_peak and float(_obv.iloc[-5]) > float(_obv.iloc[-15])
                _obv_falling = _obv_now < _obv_trou and float(_obv.iloc[-5]) < float(_obv.iloc[-15])
                if (_obv_rising and macro_trend >= 0 and not _block_long_24h and rsi_now < 75
                        and _sc_vol_surge >= 0.8 and _sc_pos < 0.82):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.62, 0.0, "sc_obv_breakout_long"
                elif (_obv_falling and macro_trend <= 0 and not _block_short_24h and rsi_now > 25
                        and _sc_vol_surge >= 0.8 and _sc_pos > 0.18):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = -1, 0.62, 0.0, "sc_obv_breakout_short"

            # S47/S48: HIGHER HIGH / LOWER LOW STRUCTURE (market structure confirmation)
            # 3 HH lien tiep (moi dinh cao hon) = uptrend da xac nhan; 3 LL = downtrend
            if _sc_dir == 0 and len(df_micro) >= 30:
                _hh_h = [float(df_micro["high"].iloc[-25:-15].max()),
                         float(df_micro["high"].iloc[-15:-8].max()),
                         float(df_micro["high"].iloc[-8:].max())]
                _ll_l = [float(df_micro["low"].iloc[-25:-15].min()),
                         float(df_micro["low"].iloc[-15:-8].min()),
                         float(df_micro["low"].iloc[-8:].min())]
                _3hh = _hh_h[0] < _hh_h[1] < _hh_h[2]
                _3ll = _ll_l[0] > _ll_l[1] > _ll_l[2]
                if (_3hh and micro_up and not _block_long_24h and rsi_now < 75 and macro_trend >= 0
                        and _sc_vol_surge >= 0.7 and _sc_pos < 0.82):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.63, 0.0, "sc_higher_highs_long"
                elif (_3ll and micro_down and not _block_short_24h and rsi_now > 25 and macro_trend <= 0
                        and _sc_vol_surge >= 0.7 and _sc_pos > 0.18):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = -1, 0.63, 0.0, "sc_lower_lows_short"

            # S49/S50: RSI MIDZONE MOMENTUM (RSI cat 50 xac nhan doi phe)
            # RSI cat len/xuong 50 = doi phe ro rang, khong co lag EMA
            if _sc_dir == 0 and len(df_micro) >= 20:
                _rm_rsi = compute_rsi(df_micro["close"], 14)
                _rm_now = float(_rm_rsi.iloc[-1])
                _rm_p5  = float(_rm_rsi.iloc[-6])
                _rm_x_up = _rm_now > 50 and _rm_p5 < 50 and _rm_now < 72
                _rm_x_dn = _rm_now < 50 and _rm_p5 > 50 and _rm_now > 28
                if (_rm_x_up and macro_trend >= 0 and not _block_long_24h and _sc_vol_surge >= 0.8
                        and _sc_pos < 0.80 and adx >= 14):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.61, 0.0, "sc_rsi_50_cross_long"
                elif (_rm_x_dn and macro_trend <= 0 and not _block_short_24h and _sc_vol_surge >= 0.8
                        and _sc_pos > 0.20 and adx >= 14):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = -1, 0.61, 0.0, "sc_rsi_50_cross_short"

            # S51/S52: DEAD-CAT BOUNCE REJECTION (bounce that bai trong downtrend -> SHORT)
            # Pattern BNCUSDT: downtrend -> bounce len -> bounce mat dong luc -> tiep tuc xuong.
            # Dieu kien SHORT:
            #   1. Prior downtrend: fast EMA bearish (EMA20 < EMA50) + macro <= 0
            #   2. Bounce xay ra: price da tang >= 0.5% trong 15-30 nen (dead-cat len)
            #   3. Bounce that bai: RSI bat bounce roi quay xuong (RSI peak 48-68, gio < peak-5)
            #   4. Volume bounce yeu hon volume dump truoc do (bounce khong co conviction)
            #   5. Price chua vuot EMA50 (con nam duoi EMA trung han)
            # Dieu kien LONG (doi xung - bear-trap bounce rejection):
            #   Uptrend -> pullback -> pullback mat dong luc -> tiep tuc len.
            if _sc_dir == 0 and len(df_micro) >= 35:
                _dc_ema20 = float(compute_ema(df_micro["close"], 20).iloc[-1])
                _dc_ema50 = float(compute_ema(df_micro["close"], 50).iloc[-1])
                _dc_rsi_s = compute_rsi(df_micro["close"], 14)
                _dc_rsi_now  = float(_dc_rsi_s.iloc[-1])
                _dc_rsi_peak = float(_dc_rsi_s.iloc[-15:-1].max())  # peak RSI trong 15 nen
                _dc_rsi_declining = _dc_rsi_now < _dc_rsi_peak - 5  # RSI da quay xuong tu peak

                # Bounce metric: gia da tang bao nhieu % tu day 20 nen qua
                _dc_lo20 = float(df_micro["low"].iloc[-20:].min())
                _dc_hi20 = float(df_micro["high"].iloc[-20:].max())
                _dc_bounce_pct = (_sc_price - _dc_lo20) / _dc_lo20 if _dc_lo20 > 0 else 0.0
                _dc_pullbk_pct = (_dc_hi20 - _sc_price) / _dc_hi20 if _dc_hi20 > 0 else 0.0

                # Volume check: volume trung binh 5 nen cuoi < volume 10 nen truoc (bounce yeu)
                _dc_vol_now = float(df_micro["volume"].iloc[-5:].mean()) if len(df_micro) >= 5 else 0
                _dc_vol_pre = float(df_micro["volume"].iloc[-20:-5].mean()) if len(df_micro) >= 20 else 0
                _dc_vol_fade = _dc_vol_pre > 0 and _dc_vol_now < _dc_vol_pre * 0.9  # vol giam

                # SHORT: downtrend (fast bearish + macro <=0) + bounce len roi mat dong luc
                _dc_prior_down = _dc_ema20 < _dc_ema50 and macro_trend <= 0
                _dc_short_ok = (
                    _dc_prior_down
                    and _dc_bounce_pct >= 0.004    # bounce len >= 0.4% tu day
                    and _dc_rsi_declining          # RSI da quay xuong tu peak bounce
                    and 35 < _dc_rsi_now < 60      # RSI vung trung (khong oversold, khong mua manh)
                    and _dc_vol_fade               # volume bounce yeu dan
                    and _sc_price < _dc_ema50 * 1.002  # price chua vuot EMA50 (bounce yeu)
                    and not _block_short_24h
                    and _sc_pos > 0.25             # khong o day mut
                )
                # LONG: uptrend (fast bullish + macro >=0) + pullback roi mat dong luc
                _dc_prior_up = _dc_ema20 > _dc_ema50 and macro_trend >= 0
                _dc_long_ok = (
                    _dc_prior_up
                    and _dc_pullbk_pct >= 0.004    # pullback xuong >= 0.4% tu dinh
                    and _dc_rsi_declining          # RSI tiep tuc giam trong pullback
                    and 40 < _dc_rsi_now < 65      # RSI khong oversold
                    and _dc_vol_fade               # volume pullback yeu
                    and _sc_price > _dc_ema50 * 0.998  # price con tren EMA50 (pullback nong)
                    and not _block_long_24h
                    and _sc_pos < 0.75
                )
                if _dc_short_ok and _sc_vol_surge >= 0.6 and adx >= 12:
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = -1, 0.66, 0.0, "sc_dead_cat_reject_short"
                elif _dc_long_ok and _sc_vol_surge >= 0.6 and adx >= 12:
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.66, 0.0, "sc_bear_trap_reject_long"

            # S53/S54: TREND RESUMPTION AFTER CONSOLIDATION
            # Sau khi trend manh, gia sideway 10-20 nen roi pha ra cung chieu -> tiep tuc trend.
            # Khac S39 (macro continuation): S53 tap trung vao pattern sideway -> pha ra (breakout nho).
            # Dieu kien: macro trend ro rang + 10 nen vua roi sideway (ATR nho, range hep)
            #            + nen hien tai pha ra cung chieu voi volume tang.
            if _sc_dir == 0 and len(df_micro) >= 25 and _atrm > 0:
                _tr_hi10 = float(df_micro["high"].iloc[-12:-2].max())
                _tr_lo10 = float(df_micro["low"].iloc[-12:-2].min())
                _tr_range10 = (_tr_hi10 - _tr_lo10) / _atrm  # range 10 nen tinh bang ATR
                _tr_sideway = _tr_range10 < 1.5              # range hep = dang consolidate
                _tr_break_up   = _sc_price > _tr_hi10 and _sc_last_green
                _tr_break_down = _sc_price < _tr_lo10 and _sc_last_red
                _tr_vol_ok = _sc_vol_surge >= 1.2           # volume pha ra phai cao hon binh thuong
                if (_tr_sideway and _tr_break_up and macro_trend >= 1 and scalp_trend >= 1
                        and not _block_long_24h and 35 < rsi_now < 72 and _tr_vol_ok
                        and _sc_pos < 0.82 and adx >= 14):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.67, 0.0, "sc_trend_resume_long"
                elif (_tr_sideway and _tr_break_down and macro_trend <= -1 and scalp_trend <= -1
                        and not _block_short_24h and 28 < rsi_now < 65 and _tr_vol_ok
                        and _sc_pos > 0.18 and adx >= 14):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = -1, 0.67, 0.0, "sc_trend_resume_short"

            # ==============================================================
            # S55/S56: POST-SPIKE EXHAUSTION SHORT / POST-DUMP EXHAUSTION LONG
            # Single candle > 3 ATR, 2-5 nen sau price reversing + vol fade -> short.
            # AWEUSDT pattern: spike 1 nen xong dao chieu ngay. Bot phai bat SHORT chu khong LONG.
            if _sc_dir == 0 and len(df_micro) >= 10 and _atrm > 0:
                _ps_window = df_micro.iloc[-8:-1]   # 7 nen gan nhat
                _ps_spike_idx = -1
                _ps_spike_dir = 0
                for _pi in range(len(_ps_window)):
                    _pc = _ps_window.iloc[_pi]
                    _p_rng = float(_pc["high"]) - float(_pc["low"])
                    _p_body = float(_pc["close"]) - float(_pc["open"])
                    if _p_rng > 3.0 * _atrm:
                        _ps_spike_idx = _pi
                        _ps_spike_dir = 1 if _p_body > 0 else -1
                if _ps_spike_idx >= 0:
                    _ps_spike_c   = _ps_window.iloc[_ps_spike_idx]
                    _ps_spike_hi  = float(_ps_spike_c["high"])
                    _ps_spike_lo  = float(_ps_window["low"].min())
                    _ps_spike_rng = _ps_spike_hi - _ps_spike_lo
                    _ps_vol_spike = float(_ps_spike_c["volume"])
                    _ps_vol_after = float(df_micro["volume"].iloc[-4:].mean()) if len(df_micro) >= 4 else 0
                    _ps_vol_fade  = _ps_vol_after < _ps_vol_spike * 0.6   # vol sau spike < 60% vol spike
                    _ps_pos       = (_sc_price - _ps_spike_lo) / _ps_spike_rng if _ps_spike_rng > 0 else 0.5
                    # Pump spike + price dang giam tu dinh spike -> SHORT
                    if (_ps_spike_dir == 1 and _ps_pos > 0.45 and _ps_vol_fade
                            and _sc_last_red and not _block_short_24h
                            and _fast_tr_pre != 1 and 30 < rsi_now < 72):
                        _sc_dir, _sc_strength, _sc_tp, _sc_name = -1, 0.68, 0.0, "sc_post_spike_short"
                    # Dump spike + price dang phuc hoi tu day spike -> LONG
                    elif (_ps_spike_dir == -1 and _ps_pos < 0.55 and _ps_vol_fade
                            and _sc_last_green and not _block_long_24h
                            and _fast_tr_pre != -1 and 28 < rsi_now < 70):
                        _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.68, 0.0, "sc_post_dump_long"

            # S57/S58: RSI DIVERGENCE
            # Price new high + RSI thap hon peak truoc -> bearish divergence -> SHORT.
            # Price new low + RSI cao hon trough truoc -> bullish divergence -> LONG.
            if _sc_dir == 0 and len(df_micro) >= 40:
                _div_rsi  = compute_rsi(df_micro["close"], 14)
                _div_rsi_now  = float(_div_rsi.iloc[-1])
                _div_price_hi20 = float(df_micro["high"].iloc[-20:].max())
                _div_price_hi40 = float(df_micro["high"].iloc[-40:-20].max())
                _div_price_lo20 = float(df_micro["low"].iloc[-20:].min())
                _div_price_lo40 = float(df_micro["low"].iloc[-40:-20].min())
                _div_rsi_hi20   = float(_div_rsi.iloc[-20:].max())
                _div_rsi_hi40   = float(_div_rsi.iloc[-40:-20].max())
                _div_rsi_lo20   = float(_div_rsi.iloc[-20:].min())
                _div_rsi_lo40   = float(_div_rsi.iloc[-40:-20].min())
                # Bearish divergence: price new high, RSI lower high
                _div_bear = (_div_price_hi20 > _div_price_hi40 * 1.001
                             and _div_rsi_hi20 < _div_rsi_hi40 - 5
                             and _div_rsi_now > 55 and _sc_pos > 0.65)
                # Bullish divergence: price new low, RSI higher low
                _div_bull = (_div_price_lo20 < _div_price_lo40 * 0.999
                             and _div_rsi_lo20 > _div_rsi_lo40 + 5
                             and _div_rsi_now < 45 and _sc_pos < 0.35)
                if _div_bear and not _block_short_24h and _sc_vol_surge >= 0.5 and adx >= 12:
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = -1, 0.65, 0.0, "sc_rsi_bearish_div"
                elif _div_bull and not _block_long_24h and _sc_vol_surge >= 0.5 and adx >= 12:
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.65, 0.0, "sc_rsi_bullish_div"

            # S59/S60: EMA COMPRESSION BREAKOUT
            # EMA20 va EMA50 sat nhau (< 0.3% cach nhau) = gia dang ngo -> pha ra = co hoi lon.
            if _sc_dir == 0 and len(df_micro) >= 55:
                _ec_ema20 = float(compute_ema(df_micro["close"], 20).iloc[-1])
                _ec_ema50 = float(compute_ema(df_micro["close"], 50).iloc[-1])
                _ec_gap   = abs(_ec_ema20 - _ec_ema50) / _ec_ema50 if _ec_ema50 > 0 else 1.0
                _ec_compress = _ec_gap < 0.003   # EMA20/50 sat nhau < 0.3%
                _ec_break_up   = _sc_price > max(_ec_ema20, _ec_ema50) * 1.001 and _sc_last_green
                _ec_break_down = _sc_price < min(_ec_ema20, _ec_ema50) * 0.999 and _sc_last_red
                if _ec_compress and _sc_vol_surge >= 1.3:
                    if (_ec_break_up and macro_trend >= 0 and not _block_long_24h
                            and 38 < rsi_now < 70 and _sc_pos < 0.80 and adx >= 14):
                        _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.66, 0.0, "sc_ema_compress_breakup"
                    elif (_ec_break_down and macro_trend <= 0 and not _block_short_24h
                            and 30 < rsi_now < 62 and _sc_pos > 0.20 and adx >= 14):
                        _sc_dir, _sc_strength, _sc_tp, _sc_name = -1, 0.66, 0.0, "sc_ema_compress_breakdn"

            # S61/S62: CONSECUTIVE WICK REJECTION (repeated rejection at S/R)
            # 3+ nen co upper/lower wick dai tai cung muc gia -> S/R manh.
            # Wicks dai = buyers/sellers co mat nhung bi ap dao -> quay dau.
            if _sc_dir == 0 and len(df_micro) >= 8 and _atrm > 0:
                _wr_c5 = df_micro.iloc[-6:-1]
                _wr_upper_wicks = [(float(r["high"]) - max(float(r["close"]), float(r["open"]))) / _atrm
                                   for _, r in _wr_c5.iterrows()]
                _wr_lower_wicks = [(min(float(r["close"]), float(r["open"])) - float(r["low"])) / _atrm
                                   for _, r in _wr_c5.iterrows()]
                _wr_upper_cnt = sum(1 for w in _wr_upper_wicks if w > 0.5)   # wick > 0.5 ATR
                _wr_lower_cnt = sum(1 for w in _wr_lower_wicks if w > 0.5)
                _wr_hi5 = float(_wr_c5["high"].max())
                _wr_lo5 = float(_wr_c5["low"].min())
                # Upper wick rejections: price repeatedly rejected at top -> SHORT
                if (_wr_upper_cnt >= 3 and _sc_last_red
                        and _sc_price > _wr_hi5 * 0.995  # price van gan dinh bi reject
                        and not _block_short_24h and rsi_now > 45
                        and _sc_vol_surge >= 0.5 and adx >= 10):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = -1, 0.64, 0.0, "sc_wick_reject_short"
                # Lower wick rejections: price repeatedly finds buyers at bottom -> LONG
                elif (_wr_lower_cnt >= 3 and _sc_last_green
                        and _sc_price < _wr_lo5 * 1.005  # price van gan day tim duoc nguoi mua
                        and not _block_long_24h and rsi_now < 55
                        and _sc_vol_surge >= 0.5 and adx >= 10):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.64, 0.0, "sc_wick_reject_long"

            # S63/S64: LIQUIDATION CASCADE REVERSAL
            # Rapid 3-candle move > 4 ATR (cascade liquidation) + volume spike + dung lai ->
            # khi cascades done, bounce manh nguoc lai (shorts/longs da bi thanh ly het).
            if _sc_dir == 0 and len(df_micro) >= 10 and _atrm > 0:
                _lc_c3     = df_micro.iloc[-4:-1]
                _lc_move   = (float(_lc_c3["high"].max()) - float(_lc_c3["low"].min())) / _atrm
                _lc_vol3   = float(_lc_c3["volume"].mean())
                _lc_vol_bg = float(df_micro["volume"].iloc[-20:-4].mean()) if len(df_micro) >= 20 else _lc_vol3
                _lc_vol_spike = _lc_vol3 > _lc_vol_bg * 2.5  # cascade = volume boc len 2.5x
                _lc_price_dir = 1 if float(_lc_c3["close"].iloc[-1]) > float(_lc_c3["open"].iloc[0]) else -1
                # Cascade xuong + gia on dinh (nen cuoi nho hon) -> LONG bounce
                _lc_last_body_abs = abs(float(df_micro["close"].iloc[-2]) - float(df_micro["open"].iloc[-2])) / _atrm
                _lc_stabilize = _lc_last_body_abs < 0.8  # nen cuoi nho = dang on dinh
                if _lc_move >= 4.0 and _lc_vol_spike and _lc_stabilize:
                    if (_lc_price_dir == -1 and not _block_long_24h
                            and rsi_now < 38 and _sc_pos < 0.35 and _sc_last_green):
                        _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.69, 0.0, "sc_liq_cascade_long"
                    elif (_lc_price_dir == 1 and not _block_short_24h
                            and rsi_now > 62 and _sc_pos > 0.65 and _sc_last_red):
                        _sc_dir, _sc_strength, _sc_tp, _sc_name = -1, 0.69, 0.0, "sc_liq_cascade_short"

            # S65/S66: EMA STACK RECOVERY / DETERIORATION
            # EMA20 > EMA50 > EMA100 (bullish stack confirmed) + price > stack = tren da len manh.
            # EMA20 < EMA50 < EMA100 (bearish stack confirmed) + price < stack = tren da xuong manh.
            # Chi trade khi stack moi hinh thanh (< 15 nen truc tiep stack day du).
            if _sc_dir == 0 and len(df_micro) >= 110:
                _es_ema20  = compute_ema(df_micro["close"], 20)
                _es_ema50  = compute_ema(df_micro["close"], 50)
                _es_ema100 = compute_ema(df_micro["close"], 100)
                _es_e20v   = float(_es_ema20.iloc[-1])
                _es_e50v   = float(_es_ema50.iloc[-1])
                _es_e100v  = float(_es_ema100.iloc[-1])
                # Stack bullish: EMA20 > EMA50 > EMA100
                _es_bull_stack = _es_e20v > _es_e50v > _es_e100v
                _es_bear_stack = _es_e20v < _es_e50v < _es_e100v
                # Kiem tra stack moi hinh thanh (8 nen truoc chua co)
                _es_e20_8 = float(_es_ema20.iloc[-8])
                _es_e50_8 = float(_es_ema50.iloc[-8])
                _es_e100_8 = float(_es_ema100.iloc[-8])
                _es_bull_new = _es_bull_stack and not (_es_e20_8 > _es_e50_8 > _es_e100_8)
                _es_bear_new = _es_bear_stack and not (_es_e20_8 < _es_e50_8 < _es_e100_8)
                if _es_bull_new and _sc_price > _es_e20v and not _block_long_24h:
                    if _sc_vol_surge >= 1.0 and 38 < rsi_now < 72 and _sc_pos < 0.85 and adx >= 14:
                        _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.68, 0.0, "sc_ema_stack_bull"
                elif _es_bear_new and _sc_price < _es_e20v and not _block_short_24h:
                    if _sc_vol_surge >= 1.0 and 28 < rsi_now < 62 and _sc_pos > 0.15 and adx >= 14:
                        _sc_dir, _sc_strength, _sc_tp, _sc_name = -1, 0.68, 0.0, "sc_ema_stack_bear"

            # S67/S68: FAILED BREAKDOWN / FAILED BREAKOUT (bull/bear trap)
            # Price breaks key level nhung KHONG sustain -> reverse nhanh = bull/bear trap.
            # Failed breakdown: pha day nhung dong lai tren day (nen co long lower wick) -> LONG.
            # Failed breakout: pha dinh nhung dong lai duoi dinh (nen co long upper wick) -> SHORT.
            if _sc_dir == 0 and len(df_micro) >= 20 and _atrm > 0:
                _fb_lo20  = float(df_micro["low"].iloc[-20:-2].min())   # day 20 nen (tru 2 nen cuoi)
                _fb_hi20  = float(df_micro["high"].iloc[-20:-2].max())  # dinh 20 nen
                _fb_prev_l = float(df_micro["low"].iloc[-2])   # nen truoc
                _fb_prev_h = float(df_micro["high"].iloc[-2])
                _fb_cur_c  = float(df_micro["close"].iloc[-1])
                _fb_cur_o  = float(df_micro["open"].iloc[-1])
                # Failed breakdown: nen truoc pha day (low < _fb_lo20) nhung close tren day -> LONG
                _fb_break_dn = _fb_prev_l < _fb_lo20 * 0.999  # pha day that su
                _fb_recover  = _fb_cur_c > _fb_lo20 and _fb_cur_c > _fb_cur_o  # phuc hoi len tren day
                # Failed breakout: nen truoc pha dinh nhung close duoi dinh -> SHORT
                _fb_break_up = _fb_prev_h > _fb_hi20 * 1.001  # pha dinh that su
                _fb_reject   = _fb_cur_c < _fb_hi20 and _fb_cur_c < _fb_cur_o  # bi day xuong duoi dinh
                if (_fb_break_dn and _fb_recover and not _block_long_24h
                        and rsi_now < 50 and _sc_pos < 0.45 and _sc_vol_surge >= 0.8 and adx >= 10):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.67, 0.0, "sc_failed_breakdown_long"
                elif (_fb_break_up and _fb_reject and not _block_short_24h
                        and rsi_now > 50 and _sc_pos > 0.55 and _sc_vol_surge >= 0.8 and adx >= 10):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = -1, 0.67, 0.0, "sc_failed_breakout_short"

            # S69/S70: MOMENTUM DECELERATION (luc cang mat dan -> dao chieu)
            # 3 nen lien tiep: body[0] > body[1] > body[2] = momentum giam dan
            # Sau khi da di nhieu -> sap het nang luong -> trade nguoc chieu.
            if _sc_dir == 0 and len(df_micro) >= 6 and _atrm > 0:
                _md_b1 = float(df_micro["close"].iloc[-4]) - float(df_micro["open"].iloc[-4])
                _md_b2 = float(df_micro["close"].iloc[-3]) - float(df_micro["open"].iloc[-3])
                _md_b3 = float(df_micro["close"].iloc[-2]) - float(df_micro["open"].iloc[-2])
                # 3 nen tang giam dan -> mat da tang -> SHORT
                _md_bull_decel = (_md_b1 > _atrm * 0.3 and _md_b2 > 0 and _md_b3 > 0
                                  and _md_b1 > _md_b2 > _md_b3)
                # 3 nen giam giam dan -> mat da giam -> LONG
                _md_bear_decel = (_md_b1 < -_atrm * 0.3 and _md_b2 < 0 and _md_b3 < 0
                                  and _md_b1 < _md_b2 < _md_b3)
                if (_md_bull_decel and not _block_short_24h
                        and rsi_now > 58 and _sc_pos > 0.65 and _sc_vol_surge >= 0.5 and adx >= 12):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = -1, 0.63, 0.0, "sc_momentum_decel_short"
                elif (_md_bear_decel and not _block_long_24h
                        and rsi_now < 42 and _sc_pos < 0.35 and _sc_vol_surge >= 0.5 and adx >= 12):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.63, 0.0, "sc_momentum_decel_long"

            # S71/S72: PRICE-VOLUME DIVERGENCE (accumulation / distribution detect)
            # Volume tang nhung gia it bien dong = ai do dang tich luy/xa hang im lang.
            if _sc_dir == 0 and len(df_micro) >= 20 and _atrm > 0:
                _pvd_vol10  = float(df_micro["volume"].iloc[-10:].mean())
                _pvd_vol_bg = float(df_micro["volume"].iloc[-30:-10].mean()) if len(df_micro) >= 30 else _pvd_vol10
                _pvd_vol_up = _pvd_vol10 > _pvd_vol_bg * 1.3  # volume tang 30%
                _pvd_rng10  = (float(df_micro["high"].iloc[-10:].max()) -
                               float(df_micro["low"].iloc[-10:].min())) / _atrm
                _pvd_tight  = _pvd_rng10 < 1.2  # range hep trong khi volume tang = tich luy/xa hang
                _pvd_trend  = float(df_micro["close"].iloc[-1]) - float(df_micro["close"].iloc[-10])
                # Volume cao + gia khong tang (khong ro uptrend) + macro bullish = tich luy -> LONG
                if (_pvd_vol_up and _pvd_tight and macro_trend >= 1
                        and _pvd_trend >= 0 and not _block_long_24h
                        and 35 < rsi_now < 62 and _sc_pos < 0.65 and adx >= 10):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.62, 0.0, "sc_accum_breakout_long"
                elif (_pvd_vol_up and _pvd_tight and macro_trend <= -1
                        and _pvd_trend <= 0 and not _block_short_24h
                        and 38 < rsi_now < 65 and _sc_pos > 0.35 and adx >= 10):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = -1, 0.62, 0.0, "sc_distrib_breakdn_short"

            # S73/S74: ATR EXPANSION BREAKOUT (volatility squeeze -> explosion)
            # ATR5 << ATR20: thi truong ngu -> bung no. Trade huong bung no khi xac nhan.
            if _sc_dir == 0 and len(df_micro) >= 25 and _atrm > 0:
                _ae_atr5  = float(compute_atr(df_micro, 5).iloc[-1])  if len(df_micro) >= 5  else _atrm
                _ae_atr20 = float(compute_atr(df_micro, 20).iloc[-1]) if len(df_micro) >= 20 else _atrm
                _ae_expanding = _ae_atr5 > _ae_atr20 * 1.5  # ATR hien tai gian ra manh
                _ae_was_tight = _ae_atr20 < _atrm * 0.85    # truoc do ATR nho hon binh thuong
                if _ae_expanding and _ae_was_tight and _sc_vol_surge >= 1.4:
                    if (_sc_last_green and macro_trend >= 0 and not _block_long_24h
                            and 35 < rsi_now < 72 and _sc_pos < 0.82 and adx >= 15):
                        _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.68, 0.0, "sc_atr_expand_long"
                    elif (_sc_last_red and macro_trend <= 0 and not _block_short_24h
                            and 28 < rsi_now < 65 and _sc_pos > 0.18 and adx >= 15):
                        _sc_dir, _sc_strength, _sc_tp, _sc_name = -1, 0.68, 0.0, "sc_atr_expand_short"

            # S75/S76: 24H RANGE TOP SHORT / BOTTOM LONG (HUSDT pattern)
            # Coin o 90%+ cua 24h range + volume giam + macro da len nhieu -> SHORT.
            # Day la nghich dao cua du dinh: thay vi block long, ta BAT SHORT o dinh ngay.
            if _sc_dir == 0 and _24h_rng > 0:
                _rp_pos = _24h_pos   # da tinh o tren: vi tri trong 24h range
                _rp_vol_fade = _sc_vol_surge < 0.8  # volume hien tai thap hon binh thuong
                # SHORT khi gia o dinh ngay + volume fade + da tang nhieu
                if (_rp_pos > 0.90 and _rp_vol_fade and _change_24h > 5
                        and not _block_short_24h and rsi_now > 55
                        and scalp_trend >= 0 and _sc_last_red and adx >= 10):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = -1, 0.67, 0.12, "sc_24h_top_short"
                # LONG khi gia o day ngay + volume fade + da giam nhieu
                elif (_rp_pos < 0.10 and _rp_vol_fade and _change_24h < -5
                        and not _block_long_24h and rsi_now < 45
                        and scalp_trend <= 0 and _sc_last_green and adx >= 10):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.67, 0.12, "sc_24h_bottom_long"

            # S77/S78: MULTI-DAY PUMP EXHAUSTION SHORT / DUMP EXHAUSTION LONG
            # Coin da tang > 12% trong 72h va hien tai dang dao chieu -> SHORT.
            if _sc_dir == 0 and len(df_signal) >= 4320:
                _me_ref72 = float(df_signal["close"].iloc[-4320])
                _me_cur   = float(df_signal["close"].iloc[-1])
                if _me_ref72 > 0:
                    _me_ch72 = (_me_cur - _me_ref72) / _me_ref72 * 100
                    _me_rsi_declining = rsi_now < float(compute_rsi(df_signal["close"], 14).iloc[-10:-1].max()) - 8
                    # Pump > 12% trong 3 ngay + RSI dang quay xuong + price o dinh = SHORT
                    if (_me_ch72 > 12 and _me_rsi_declining and _24h_pos > 0.75
                            and not _block_short_24h and _sc_last_red
                            and macro_trend <= 0 and _sc_vol_surge >= 0.5 and adx >= 12):
                        _sc_dir, _sc_strength, _sc_tp, _sc_name = -1, 0.68, 0.12, "sc_multiday_pump_exhaust"
                    # Dump > 12% trong 3 ngay + RSI dang phuc hoi + price o day = LONG
                    elif (_me_ch72 < -12 and not _me_rsi_declining and _24h_pos < 0.25
                            and not _block_long_24h and _sc_last_green
                            and macro_trend >= 0 and _sc_vol_surge >= 0.5 and adx >= 12):
                        _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.68, 0.12, "sc_multiday_dump_exhaust"

            # S79/S80: VOLUME DECLINING AT EXTREME PRICE (distribution / accumulation confirm)
            # Volume liên tuc giam 5 nen lien tiep trong khi gia o dinh/day -> xa hang / gom hang xong.
            if _sc_dir == 0 and len(df_micro) >= 8:
                _vd_vols = df_micro["volume"].iloc[-6:-1].values
                _vd_declining = all(_vd_vols[i] > _vd_vols[i+1] for i in range(len(_vd_vols)-1))
                _vd_rising = all(_vd_vols[i] < _vd_vols[i+1] for i in range(len(_vd_vols)-1))
                if _vd_declining and _24h_pos > 0.82 and not _block_short_24h:
                    if rsi_now > 55 and _sc_last_red and adx >= 10:
                        _sc_dir, _sc_strength, _sc_tp, _sc_name = -1, 0.65, 0.0, "sc_vol_fade_at_top"
                elif _vd_declining and _24h_pos < 0.18 and not _block_long_24h:
                    if rsi_now < 45 and _sc_last_green and adx >= 10:
                        _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.65, 0.0, "sc_vol_fade_at_bottom"

            # S81/S82: HIGHER HIGH LOWER RSI = BEARISH DIV (confirm tren 1m chi tiet hon S57)
            # S57 dung 20/40 nen, S81 dung 10/20 nen (ngau chuan hon cho scalp 1m)
            if _sc_dir == 0 and len(df_micro) >= 22:
                _hd_rsi  = compute_rsi(df_micro["close"], 14)
                _hd_hi10 = float(df_micro["high"].iloc[-10:].max())
                _hd_hi20 = float(df_micro["high"].iloc[-20:-10].max())
                _hd_lo10 = float(df_micro["low"].iloc[-10:].min())
                _hd_lo20 = float(df_micro["low"].iloc[-20:-10].min())
                _hd_rsi10_max = float(_hd_rsi.iloc[-10:].max())
                _hd_rsi20_max = float(_hd_rsi.iloc[-20:-10].max())
                _hd_rsi10_min = float(_hd_rsi.iloc[-10:].min())
                _hd_rsi20_min = float(_hd_rsi.iloc[-20:-10].min())
                # Bearish div: price high moi nhung RSI thap hon
                if (_hd_hi10 > _hd_hi20 * 1.0005 and _hd_rsi10_max < _hd_rsi20_max - 4
                        and not _block_short_24h and rsi_now > 52 and _sc_pos > 0.62
                        and _sc_last_red and _sc_vol_surge >= 0.4 and adx >= 10):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = -1, 0.64, 0.0, "sc_bearish_div_short"
                # Bullish div: price low moi nhung RSI cao hon
                elif (_hd_lo10 < _hd_lo20 * 0.9995 and _hd_rsi10_min > _hd_rsi20_min + 4
                        and not _block_long_24h and rsi_now < 48 and _sc_pos < 0.38
                        and _sc_last_green and _sc_vol_surge >= 0.4 and adx >= 10):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.64, 0.0, "sc_bullish_div_long"

            # S83/S84: SWING HIGH/LOW REJECTION (Wyckoff re-test)
            # Price test lai swing high/low truoc do va bi day nguoc lai -> S/R confirmed.
            if _sc_dir == 0 and len(df_micro) >= 30 and _atrm > 0:
                # Tim swing high 10-25 nen truoc
                _sh_window = df_micro.iloc[-26:-3]
                _sh_swing_hi = float(_sh_window["high"].max())
                _sh_swing_lo = float(_sh_window["low"].min())
                _sh_cur_hi = float(df_micro["high"].iloc[-2])
                _sh_cur_lo = float(df_micro["low"].iloc[-2])
                _sh_close  = float(df_micro["close"].iloc[-1])
                # Re-test swing high + bi reject (touch nhung dong lai duoi) -> SHORT
                _sh_test_hi = (_sh_cur_hi >= _sh_swing_hi * 0.998 and
                               _sh_close < _sh_swing_hi * 0.998 and
                               float(df_micro["open"].iloc[-1]) < _sh_swing_hi)
                # Re-test swing low + bi bounce (touch nhung dong lai tren) -> LONG
                _sh_test_lo = (_sh_cur_lo <= _sh_swing_lo * 1.002 and
                               _sh_close > _sh_swing_lo * 1.002 and
                               float(df_micro["open"].iloc[-1]) > _sh_swing_lo)
                if (_sh_test_hi and not _block_short_24h and rsi_now > 48
                        and _sc_vol_surge >= 0.5 and _sc_last_red and adx >= 10):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = -1, 0.66, 0.0, "sc_swing_hi_reject"
                elif (_sh_test_lo and not _block_long_24h and rsi_now < 52
                        and _sc_vol_surge >= 0.5 and _sc_last_green and adx >= 10):
                    _sc_dir, _sc_strength, _sc_tp, _sc_name = 1, 0.66, 0.0, "sc_swing_lo_bounce"

            if _sc_dir == 0:
                return False

            # VOLUME COLLAPSE GUARD (chay cho TAT CA scenario paths S1-S50):
            # Neu volume hien tai < 20% trung binh 30c -> post-pump/dump exhaustion.
            # Bat ky scenario nao vao luc nay cung la fake: khong co ai giao dich nua.
            # TNSR pattern: vol 6.56K vs MA10 116K = 5.6% -> phai bi chan o day.
            if _sc_vol_surge < 0.20:
                logger.info(
                    f"{symbol}: [SCENARIO] {_sc_name} BLOCKED - volume collapse "
                    f"(vol_surge={_sc_vol_surge:.2f} < 0.20, post-pump/dump exhaustion)")
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

        # Volume pressure filter (skip in HF mode)
        if not getattr(config, "HIGH_FREQ_MODE", False) and not df_micro.empty and len(df_micro) >= 5:
            _vbars = df_micro.iloc[-5:]
            _green_vol = _vbars.loc[_vbars["close"] >= _vbars["open"], "volume"].sum()
            _red_vol   = _vbars.loc[_vbars["close"] <  _vbars["open"], "volume"].sum()
            _total_vol = _green_vol + _red_vol
            if _total_vol > 0:
                _buy_ratio = _green_vol / _total_vol
                if best.direction == -1 and _buy_ratio >= 0.90:
                    return _block(f"skip SHORT - 5-bar volume pressure BUY {_buy_ratio*100:.0f}%")
                if best.direction == 1 and _buy_ratio <= 0.10:
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

        # 1h range block (5h range extreme) -> BLOCK (khong flip: flip gay sai chieu)
        # Dinh/day 5h range trong trend khong xac nhan -> block hoan toan, khong dao chieu
        if _h1_block_long and best.direction == 1 and not _emerging_uptrend and not _scenario_entry:
            if not _all_tfs_bull:
                return _block(f"5h range top ({_h1_pos:.0%}) block LONG - trend chua xac nhan, skip")
        if _h1_block_short and best.direction == -1 and not _emerging_downtrend and not _scenario_entry:
            if not _all_tfs_bear:
                return _block(f"5h range bottom ({_h1_pos:.0%}) block SHORT - trend chua xac nhan, skip")

        # 2h range block
        # Exception: gradual trend (>=18/30 nen cung chieu) + scalp xac nhan -> day/dinh 2h la DIEM BO QUA
        # BTC tang lien tuc 25 phut tao ra dinh 2h moi = gradual uptrend, khong phai pump da can kiet
        # Emerging trend: uptrend moi day gia len dinh 2h = trend dang chay, khong flip nguoc
        # Scenario entry: da tu phan tich vi tri (breakout tren dinh la chu dich) - khong flip
        # _all_tfs_bull/bear bypass: cho phep skip flip trong confirmed trend
        # NHUNG chi bypass khi KHONG o extreme position (tren 85% hoac duoi 15% 2h range)
        # Extreme bottom trong downtrend = nguong bounce cao, van flip SHORT->LONG (AKEUSDT pattern)
        # Extreme top trong uptrend = nguong pullback cao, van flip LONG->SHORT
        _not_2h_extreme_top = _m2h_pos < 0.85   # khong o extreme dinh 2h
        _not_2h_extreme_bot = _m2h_pos > 0.15   # khong o extreme day 2h
        _m2h_grad_bypass_long  = (_is_gradual_uptrend   and scalp_trend == 1  and macro_trend == 1) \
                                 or _emerging_uptrend or _scenario_entry \
                                 or (_all_tfs_bull and _not_2h_extreme_top)
        _m2h_grad_bypass_short = (_is_gradual_downtrend and scalp_trend == -1 and macro_trend == -1) \
                                 or _emerging_downtrend or _scenario_entry \
                                 or (_all_tfs_bear and _not_2h_extreme_bot)
        if _m2h_block_short and best.direction == -1 and not _m2h_grad_bypass_short:
            return _block(f"2h range bottom ({_m2h_pos:.0%}) block SHORT - skip, khong flip sai chieu")
        if _m2h_block_long and best.direction == 1 and not _m2h_grad_bypass_long:
            return _block(f"2h range top ({_m2h_pos:.0%}) block LONG - skip, khong flip sai chieu")

        # POST-PEAK / POST-TROUGH: EMA(100/250) lag sau khi gia qua dinh/day
        # MAGMAUSDT pattern: EMA con bullish nhung coin da giam 1%+ trong 40+ phut -> LONG = sai chieu
        # FLIP thay vi block: gia dang giam sustained sau dinh -> SHORT la lenh dung chieu
        # (nguyen tac: khong bo lenh tiem nang, doi chieu de trade theo trend thuc te)
        # Guard vi tri 2h range: KHONG short khi gia DA o day range (<30%) - short day la muon;
        # tuong tu KHONG long khi gia da o dinh range (>70%). Scenario entry tu phan tich - bo qua.
        if not _direction_flipped and not _scenario_entry:
            if best.direction == 1 and _post_peak_decline_long and _m2h_pos > 0.30:
                return _block(
                    f"POST-PEAK block LONG - {_ppd_drop*100:.1f}% below 60c high "
                    f"({_ppd_hi_age}c ago) scalp={scalp_trend}"
                )
            elif best.direction == -1 and _post_trough_rise_short and _m2h_pos < 0.70:
                return _block(
                    f"POST-TROUGH block SHORT - {_ppd_rise*100:.1f}% above 60c low "
                    f"({_ppd_lo_age}c ago) scalp={scalp_trend}"
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
        # Pre-compute high_conviction for early bypass (full compute again in gate section)
        _fast_tr_pre = self._trend_direction(df_micro, fast=20, slow=50) if not df_micro.empty and len(df_micro) >= 55 else 0
        _T_pre = best.direction  # fallback to signal direction
        _vrat_pre = (df_micro["volume"].iloc[-5:].mean() / (df_micro["volume"].iloc[-20:-5].mean() + 1e-9)
                     if not df_micro.empty and len(df_micro) >= 20 else 0.0)
        _high_conviction = (
            _fast_tr_pre == best.direction and macro_trend == best.direction and macro_4h == best.direction
            and adx >= 25 and _vrat_pre >= 1.5
        )

        # PUMP EXHAUSTION SHORT: khi LONG bi HARD BLOCK vi micro_down
        # TREND ALIGNMENT GATE (chay ca trong HF mode):
        # Block khi CA 2 macro TF deu NGUOC CHIEU signal -> sai trend ro rang.
        # Chi block khi CA HAI (macro_trend va macro_4h) nguoc, tranh block khi chi 1 TF conflict.
        # VD: LONG khi macro_trend=-1 va macro_4h=-1 = dang trong downtrend ro rang -> block.
        _hf_mode = getattr(config, "HIGH_FREQ_MODE", False)
        if _hf_mode:
            # Block khi CA 2 macro TF deu NGUOC va fast EMA cung chua flip:
            # - macro_trend=-1 AND macro_4h=-1 AND fast_tr=-1: downtrend ro rang ca ngan+dai han -> block LONG
            # - Neu fast_tr==+1 (da flip): emerging uptrend, cho phep (EMA dai lag, EMAs 20/50 da bullish)
            _macro_strongly_against_long  = (macro_trend == -1 and macro_4h == -1 and _fast_tr_pre == -1)
            _macro_strongly_against_short = (macro_trend ==  1 and macro_4h ==  1 and _fast_tr_pre ==  1)
            if best.direction == 1 and _macro_strongly_against_long:
                return _block(
                    f"HF TREND-ALIGN BLOCK LONG: fast={_fast_tr_pre} macro={macro_trend} macro4h={macro_4h} "
                    f"tat ca bearish -> sai trend ro rang (adx={adx:.0f})"
                )
            if best.direction == -1 and _macro_strongly_against_short:
                return _block(
                    f"HF TREND-ALIGN BLOCK SHORT: fast={_fast_tr_pre} macro={macro_trend} macro4h={macro_4h} "
                    f"tat ca bullish -> sai trend ro rang (adx={adx:.0f})"
                )

        _pump_exhaustion_flip = False
        if (not _hf_mode and best.direction == 1 and micro_down and not _btc_bull_long_ok
                and not _direction_flipped):
            _pump_exh_30 = 0.0
            if not df_micro.empty and len(df_micro) >= 30:
                _p30  = df_micro["close"].iloc[-30]
                _pnow = df_micro["close"].iloc[-1]
                _pump_exh_30 = (_pnow - _p30) / _p30 if _p30 > 0 else 0.0
            _can_pump_exh_short = (
                _pump_exh_30 > 0.005        # +0.5% trong 30 nen = co pump xay ra truoc do
                and _m2h_pos > 0.65         # price phai o vung DINH THAT SU (>65%, ketat hon 55%)
                and not _is_gradual_uptrend # khong phai uptrend lien tuc (do la continuation)
                and not (macro_trend == 1 and macro_4h == 1)  # khong co macro bull manh
                and _fast_tr_pre != 1       # fast EMA KHONG bullish = pump khong co nen tang
                                            # Neu fast=+1 = EMA20>EMA50 = uptrend con manh
                                            # -> KHONG flip short, tranh short giua uptrend dang chay
                and len(signals) >= 2       # consensus >= 2
                and not _scenario_entry     # scenario tu xu ly rieng ben duoi
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
            elif _high_conviction:
                pass  # HIGH-CONVICTION: F+M1+M2 cung chieu + ADX>=25 + vol>=1.5x -> cho phep entry
            elif _scenario_entry:
                # SCENARIO LONG khi micro bearish: chi cho phep neu dung o day that su.
                # Khong the LONG giua downtrend chi vi scenario fire - LAUSDT pattern.
                # Dieu kien cho phep: o vung day 2h (m2h<0.30) VA fast_tr khong bearish manh.
                # Neu fast_tr==-1 VA khong o day = trend dang xuong, scenario LONG la sai chieu.
                _sc_long_ok = _m2h_pos < 0.40 and _fast_tr_pre != -1
                if not _sc_long_ok:
                    return _block(
                        f"HARD BLOCK LONG (scenario) - micro bearish + fast={_fast_tr_pre} "
                        f"+ 2h_pos={_m2h_pos:.0%} > 40% (khong o day thuc su)"
                    )
            else:
                return _block(
                    f"HARD BLOCK LONG - 1m BEARISH (micro=-1, gia dang giam) "
                    f"| 5m={scalp_trend} 15m={macro_trend} 1h={macro_4h}"
                )
        # DUMP EXHAUSTION LONG: doi xung voi pump_exhaustion_flip
        _dump_exhaustion_flip = False
        if (not _hf_mode and best.direction == -1 and micro_up and not _btc_bear_short_ok
                and not _direction_flipped):
            _dump_exh_30 = 0.0
            if not df_micro.empty and len(df_micro) >= 30:
                _p30d  = df_micro["close"].iloc[-30]
                _pnowd = df_micro["close"].iloc[-1]
                _dump_exh_30 = (_p30d - _pnowd) / _p30d if _p30d > 0 else 0.0
            _can_dump_exh_long = (
                _dump_exh_30 > 0.005         # da dump > 0.5% trong 30 nen
                and _m2h_pos < 0.35          # price phai o vung day THAT SU (<35%, ketat hon 45%)
                and not _is_gradual_downtrend
                and not (macro_trend == -1 and macro_4h == -1)
                and _fast_tr_pre != -1       # fast EMA KHONG bearish = bounce co nen tang
                                             # Neu fast=-1 = EMA20<EMA50 = xu huong van xuong
                                             # -> bounce chi la dead-cat, KHONG flip sang LONG
                and len(signals) >= 2
                and not _scenario_entry
            )
            if _can_dump_exh_long:
                best.direction = 1
                _dump_exhaustion_flip = True
                logger.info(
                    f"{symbol}: DUMP-EXHAUSTION flip SHORT->LONG | "
                    f"dump30={_dump_exh_30*100:.1f}% m2h={_m2h_pos:.0%} scalp={scalp_trend}"
                )
            elif _high_conviction:
                pass  # HIGH-CONVICTION: F+M1+M2 cung chieu + ADX>=25 + vol>=1.5x -> cho phep entry
            elif _scenario_entry:
                # SCENARIO SHORT khi micro bullish: chi cho phep neu dung o dinh that su.
                _sc_short_ok = _m2h_pos > 0.60 and _fast_tr_pre != 1
                if not _sc_short_ok:
                    return _block(
                        f"HARD BLOCK SHORT (scenario) - micro bullish + fast={_fast_tr_pre} "
                        f"+ 2h_pos={_m2h_pos:.0%} < 60% (khong o dinh thuc su)"
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

        if not _is_high_vol:
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
        if not df_signal.empty and len(df_signal) >= 20 and _atr_for_sl > 0 and not is_reversal and not _high_conviction:
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
                _stoch_bypass_long  = _emerging_uptrend   or _scenario_name in ("sc_breakout_up", "sc_emerging_up", "sc_trend_pullback_long")
                _stoch_bypass_short = _emerging_downtrend or _scenario_name in ("sc_breakout_down", "sc_emerging_down", "sc_trend_pullback_short")
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
            if _mean20 > 0 and (_std20 / _mean20) < 0.0010:  # std < 0.10% = truly flat range (was 0.15%)
                if best.direction == 1 and _m2h_pos > 0.50 and not _scenario_entry:
                    return _block(
                        f"skip LONG - flat at 2h top ({_m2h_pos:.0%}), "
                        f"std={_std20/_mean20*100:.3f}% (distribution zone)"
                    )
                elif best.direction == -1 and _m2h_pos < 0.50 and not _scenario_entry:
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

                # LONG: neu price o top 2h range (>75%) ma sellers dang chiem uu -> flip SHORT
                # Nang tu 0.60 len 0.75: 0.60 qua thap, lam flip nhieu lenh o giua range
                # Guards: khong flip lenh da flip/scenario; khong flip nguoc EMERGING uptrend
                if (best.direction == 1 and _m2h_pos > 0.75
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
                # SHORT: neu price o bot 2h range (<25%) ma buyers dang chiem uu -> flip LONG
                # Ha tu 0.40 xuong 0.25: doi xung voi tren
                if (best.direction == -1 and _m2h_pos < 0.25
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
        # UNIVERSAL ABSOLUTE BLOCKS - KHONG CO EXCEPTION NAO (ke ca scenario/flip)
        # Chay cho TAT CA lenh truoc khi vao gate. Ngan cac truong hop "ngu ngoc" ma
        # tat ca cac guard AEQ/scenario/flip deu co the bo qua.
        # ======================================================================

        # --- UNIVERSAL BLOCK A: CHASING LARGE RECENT MOVE (vao cuoi song lon) ---
        # 30c range >= 3 ATR + price o top 80% (LONG) hoac bottom 20% (SHORT) = du dinh/du day.
        # Gap lon nhat: scenario entry va direction_flipped bypass tat ca AEQ guards ngoai tru
        # HARD BLOCK micro. Guard nay dong lai gap do - khong co bypass nao ca.
        # Ngoai le DUY NHAT:
        #   1. HIGH_CONVICTION (F+M1+M2+ADX25+vol1.5x) = breakout momentum that su
        #   2. Full trend + moderate move: ca 3 TF cung chieu (fast/scalp/macro) + move < 5 ATR
        #      = continuation trong confirmed trend, khong phai chasing spike
        if not _high_conviction and len(df_micro) >= 32 and _atrm > 0:
            _ub_r30_hi  = float(df_micro["high"].iloc[-30:].max())
            _ub_r30_lo  = float(df_micro["low"].iloc[-30:].min())
            _ub_r30_rng = _ub_r30_hi - _ub_r30_lo
            _ub_move_atr = _ub_r30_rng / _atrm
            if _ub_move_atr >= 3.0 and _ub_r30_rng > 0:
                _ub_cur_p = float(df_micro["close"].iloc[-1])
                _ub_pos   = (_ub_cur_p - _ub_r30_lo) / _ub_r30_rng
                # Volume ratio tinh truc tiep (khong dung _sc_vol_surge co the chua set)
                _ub_vrat = (df_micro["volume"].iloc[-5:].mean() /
                            (df_micro["volume"].iloc[-20:-5].mean() + 1e-9)
                            if len(df_micro) >= 20 else 1.0)
                _ub_full_bull = (_fast_tr_pre == 1  and scalp_trend == 1  and macro_trend >= 1)
                _ub_full_bear = (_fast_tr_pre == -1 and scalp_trend == -1 and macro_trend <= -1)
                _ub_mod_move  = _ub_move_atr < 5.0  # khong phai extreme spike

                _trade_dir = best.direction   # dung direction hien tai (co the da flip)
                if _trade_dir == 1 and _ub_pos > 0.80:
                    # LONG khi gia o top 80% cua 30c range co 3+ ATR move
                    _ub_ok = (_ub_full_bull and _ub_mod_move) or (_ub_pos >= 0.95 and _ub_vrat >= 1.5)
                    if not _ub_ok:
                        return _block(
                            f"skip LONG - 30c move {_ub_move_atr:.1f}ATR at top {_ub_pos:.0%} "
                            f"(du dinh: fast={_fast_tr_pre} scalp={scalp_trend} macro={macro_trend})"
                        )
                if _trade_dir == -1 and _ub_pos < 0.20:
                    # SHORT khi gia o bottom 20% cua 30c range co 3+ ATR move
                    _ub_ok = (_ub_full_bear and _ub_mod_move) or (_ub_pos <= 0.05 and _ub_vrat >= 1.5)
                    if not _ub_ok:
                        return _block(
                            f"skip SHORT - 30c move {_ub_move_atr:.1f}ATR at bottom {_ub_pos:.0%} "
                            f"(du day: fast={_fast_tr_pre} scalp={scalp_trend} macro={macro_trend})"
                        )

        # --- UNIVERSAL BLOCK B: CANDLE EXHAUSTION (vao cuoi song - ap dung CA scenario) ---
        # Phien ban nang cap cua BLOCK 4 (BLOCK 4 chi chay cho non-scenario trong gate).
        # Scenario entry co the vao CUOI nen lon vi BLOCK 4 bi skip.
        # Guard nay dong lai gap do.
        if not _high_conviction and _atrm > 0 and len(df_micro) >= 5:
            _ub_c3      = df_micro.iloc[-4:-1]
            _ub_bodies  = (_ub_c3["close"] - _ub_c3["open"]).values
            _ub_run_up  = sum(float(b) for b in _ub_bodies if b > 0) / _atrm
            _ub_run_dn  = sum(-float(b) for b in _ub_bodies if b < 0) / _atrm
            _ub_last    = float(df_micro["close"].iloc[-2]) - float(df_micro["open"].iloc[-2])
            _ub_last_g  = max(0.0, _ub_last) / _atrm
            _ub_last_r  = max(0.0, -_ub_last) / _atrm
            _trade_dir  = best.direction
            if _trade_dir == 1 and (_ub_run_up >= 1.5 or _ub_last_g >= 2.0):
                return _block(
                    f"skip LONG (universal) - 3c pump {_ub_run_up:.1f}ATR last={_ub_last_g:.1f}ATR (du dinh)"
                )
            if _trade_dir == -1 and (_ub_run_dn >= 1.5 or _ub_last_r >= 2.0):
                return _block(
                    f"skip SHORT (universal) - 3c dump {_ub_run_dn:.1f}ATR last={_ub_last_r:.1f}ATR (du day)"
                )

        # --- UNIVERSAL BLOCK C: SINGLE-CANDLE SPIKE (bat AWEUSDT pattern) ---
        # BLOCK B chi kiem 3 nen body lien tiep. Neu spike xay ra 1 nen > 3 ATR
        # trong 8 nen gan nhat va gia hien tai van o trong vung spike -> du dinh/du day.
        # Truong hop dien hinh: spike xong, price retrace 1-2 nen, bot vao cung chieu spike.
        _trade_dir = best.direction
        if not _high_conviction and _atrm > 0 and len(df_micro) >= 10:
            _ubc_window = df_micro.iloc[-9:-1]   # 8 nen gan nhat (tru nen hien tai)
            for _ubc_i in range(len(_ubc_window)):
                _ubc_c   = _ubc_window.iloc[_ubc_i]
                _ubc_rng = float(_ubc_c["high"]) - float(_ubc_c["low"])
                if _ubc_rng > 3.0 * _atrm:
                    _ubc_hi = float(_ubc_c["high"])
                    _ubc_lo = float(_ubc_c["low"])
                    _ubc_cur = float(df_micro["close"].iloc[-1])
                    _ubc_pos = (_ubc_cur - _ubc_lo) / _ubc_rng if _ubc_rng > 0 else 0.5
                    # Long khi gia van o top 55% cua spike range (bao gom retrace) -> du dinh spike
                    if _trade_dir == 1 and _ubc_pos > 0.55:
                        _ub_full_bull_c = (_fast_tr_pre == 1 and scalp_trend == 1 and macro_trend >= 1)
                        if not _ub_full_bull_c:
                            return _block(
                                f"skip LONG - spike {_ubc_rng/_atrm:.1f}ATR (nen {8-_ubc_i} truoc) "
                                f"price tai {_ubc_pos:.0%} spike zone (du dinh spike don doc)"
                            )
                    # Short khi gia van o bottom 45% cua spike range -> du day spike
                    if _trade_dir == -1 and _ubc_pos < 0.45:
                        _ub_full_bear_c = (_fast_tr_pre == -1 and scalp_trend == -1 and macro_trend <= -1)
                        if not _ub_full_bear_c:
                            return _block(
                                f"skip SHORT - dump spike {_ubc_rng/_atrm:.1f}ATR (nen {8-_ubc_i} truoc) "
                                f"price tai {_ubc_pos:.0%} spike zone (du day spike don doc)"
                            )
                    break   # chi kiem spike lon nhat / dau tien

        # --- UNIVERSAL BLOCK D: EMA OVER-EXTENSION (chase qua xa EMA) ---
        # Price > 4% tren EMA20 khi LONG, hoac > 4% duoi EMA20 khi SHORT ->
        # da di qua xa, mean-revert kha nang cao, khong co do tro san.
        if not _high_conviction and len(df_micro) >= 25:
            _ubd_ema20 = float(compute_ema(df_micro["close"], 20).iloc[-1])
            _ubd_cur   = float(df_micro["close"].iloc[-1])
            if _ubd_ema20 > 0:
                _ubd_ext = (_ubd_cur - _ubd_ema20) / _ubd_ema20
                if _trade_dir == 1 and _ubd_ext > 0.04:
                    return _block(
                        f"skip LONG - price {_ubd_ext*100:.1f}% tren EMA20 (qua xa, du dinh xa EMA)"
                    )
                if _trade_dir == -1 and _ubd_ext < -0.04:
                    return _block(
                        f"skip SHORT - price {abs(_ubd_ext)*100:.1f}% duoi EMA20 (qua xa, du day xa EMA)"
                    )

        # --- UNIVERSAL BLOCK E: RSI EXTREME AT ENTRY ---
        # RSI > 78 khi LONG (overbought extreme), RSI < 22 khi SHORT (oversold extreme).
        # Vung nay mean-revert manh, ty le thanh cong rat thap.
        if not _high_conviction:
            _ube_rsi = rsi_now   # rsi_now da duoc tinh o tren
            if _trade_dir == 1 and _ube_rsi > 78:
                return _block(f"skip LONG - RSI={_ube_rsi:.0f} > 78 (overbought extreme, du dinh RSI)")
            if _trade_dir == -1 and _ube_rsi < 22:
                return _block(f"skip SHORT - RSI={_ube_rsi:.0f} < 22 (oversold extreme, du day RSI)")

        # --- UNIVERSAL BLOCK F: VOLUME SPIKE REVERSAL INDICATOR ---
        # Spike candle trong 3 nen vua qua co volume >> binh thuong + nen hien tai dao chieu.
        # Dau hieu: ai do xa hang (selling into pump) hoac mua vao (buying into dump) = reversal.
        if not _high_conviction and len(df_micro) >= 10 and _atrm > 0:
            _ubf_c3_vols = df_micro["volume"].iloc[-4:-1].values   # 3 nen truoc
            _ubf_bg_vol  = float(df_micro["volume"].iloc[-20:-4].mean()) if len(df_micro) >= 20 else 1.0
            _ubf_max_vol_idx = int(_ubf_c3_vols.argmax())
            _ubf_max_vol = float(_ubf_c3_vols[_ubf_max_vol_idx])
            if _ubf_bg_vol > 0 and _ubf_max_vol > _ubf_bg_vol * 3.0:   # volume spike > 3x
                _ubf_spike_c  = df_micro.iloc[-4 + _ubf_max_vol_idx]
                _ubf_spike_body = float(_ubf_spike_c["close"]) - float(_ubf_spike_c["open"])
                _ubf_cur_body   = float(df_micro["close"].iloc[-2]) - float(df_micro["open"].iloc[-2])
                # Spike candle tang manh, nen tiep theo dao chieu: ai do dang xa hang
                if _ubf_spike_body > _atrm * 1.0 and _ubf_cur_body < -_atrm * 0.3:
                    if _trade_dir == 1:  # tiep tuc LONG sau khi da co dau hieu dao chieu -> block
                        return _block(
                            f"skip LONG - vol spike {_ubf_max_vol/_ubf_bg_vol:.1f}x + reversal candle "
                            f"(xa hang trong pump, du dinh vol spike)"
                        )
                # Spike candle giam manh, nen tiep theo dao chieu: ai do dang mua vao
                if _ubf_spike_body < -_atrm * 1.0 and _ubf_cur_body > _atrm * 0.3:
                    if _trade_dir == -1:  # tiep tuc SHORT sau khi co dau hieu reversal -> block
                        return _block(
                            f"skip SHORT - vol spike {_ubf_max_vol/_ubf_bg_vol:.1f}x + reversal candle "
                            f"(mua vao trong dump, du day vol spike)"
                        )

        # --- UNIVERSAL BLOCK H: 24H RANGE TOP/BOTTOM (HUSDT pattern) ---
        # _block_long/short_24h duoc set khi price > 88% / < 12% cua 24h range.
        # Nhung cac guard AEQ / scenario co the bypass flag do.
        # Universal block nay dam bao KHONG lenh nao vao long o dinh ngay / short o day ngay.
        # Ngoai le duy nhat: _high_conviction (F+M1+M2 aligned + ADX>=25 + vol>=1.5x).
        _trade_dir = best.direction
        if not _high_conviction and _24h_rng > 0 and _24h_hi > 0:
            _ubh_cur = float(df_micro["close"].iloc[-1]) if not df_micro.empty else 0.0
            _ubh_pos = (_ubh_cur - _24h_lo) / _24h_rng if _ubh_cur > 0 else _24h_pos
            if _trade_dir == 1 and _ubh_pos > 0.88:
                return _block(
                    f"skip LONG - price {_ubh_pos:.0%} of 24h range "
                    f"(24hH={_24h_hi:.6f} 24hL={_24h_lo:.6f}) (du dinh tuyet doi ngay - HUSDT pattern)"
                )
            if _trade_dir == -1 and _ubh_pos < 0.12:
                return _block(
                    f"skip SHORT - price {_ubh_pos:.0%} of 24h range "
                    f"(24hH={_24h_hi:.6f} 24hL={_24h_lo:.6f}) (du day tuyet doi ngay)"
                )

        # --- UNIVERSAL BLOCK I: MULTI-DAY PUMP AT ENTRY ---
        # Coin pump > 15% trong 72h (3 ngay) + hien tai o top 75% range = exhausted.
        # HUSDT: pump tu 0.054160 (3 ngay truoc) len 0.067220 = +24% -> long o top la ngu.
        if not _high_conviction and len(df_signal) >= 4320:
            _W72H = 4320  # 72h = 4320 nen 1m
            _ref_72h = float(df_signal["close"].iloc[-_W72H])
            _cur_72h = float(df_signal["close"].iloc[-1])
            if _ref_72h > 0:
                _change_72h = (_cur_72h - _ref_72h) / _ref_72h * 100
                if _change_72h > 15 and _24h_pos > 0.75 and _trade_dir == 1:
                    return _block(
                        f"skip LONG - 72h pump={_change_72h:.1f}% + 24h pos={_24h_pos:.0%} "
                        f"(coin da tang nhieu ngay, entry o gan dinh multi-day)"
                    )
                if _change_72h < -15 and _24h_pos < 0.25 and _trade_dir == -1:
                    return _block(
                        f"skip SHORT - 72h dump={_change_72h:.1f}% + 24h pos={_24h_pos:.0%} "
                        f"(coin da giam nhieu ngay, entry o gan day multi-day)"
                    )

        # --- UNIVERSAL BLOCK G: CHOPPY RANGE NO-TRADE ZONE ---
        # 20 nen gia dao dong qua lai khong co huong ro (range < 1.5 ATR) + ADX thap ->
        # moi signal trong vung nay deu rat de bi stop out boi range chop.
        if not _high_conviction and len(df_micro) >= 22 and _atrm > 0:
            _ubg_rng = (float(df_micro["high"].iloc[-20:].max()) -
                        float(df_micro["low"].iloc[-20:].min())) / _atrm
            if _ubg_rng < 1.5 and adx < 18:   # range rat hep + ADX yeu = choppy
                _ubg_full_trend = (abs(_fast_tr_pre) + abs(scalp_trend) + abs(macro_trend)) >= 3
                if not _ubg_full_trend:
                    return _block(
                        f"skip - choppy no-trend zone: 20c range={_ubg_rng:.1f}ATR ADX={adx:.0f} "
                        f"(khong co trend ro, de bi chop ra)"
                    )

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
            # UNIFIED SCORING GATE v6 - PURE SCORING ENGINE
            # ================================================================
            # Kien truc moi: TAT CA tin hieu (EMA, macro, MACD, RSI, vol, imm,
            # extension, fresh-move, candle) deu la SCORE COMPONENTS.
            # Chi giu 3 absolute block toi thieu:
            #   1. ADX < 8 (thi truong chet hoan toan, khong co xu huong)
            #   2. RSI 15/85 (extreme exhaustion, bounce/dump sap xay ra chac chan)
            #   3. Fresh move > 3.0 ATR/10n (spike dien cuong - khong ai theo kip)
            # Moi thu con lai -> diem cong/tru trong UNIFIED SCORE.
            # Nguong vao lenh: score >= 50 (tong hop nhieu tin hieu dong thuan).
            # KAITO bug duoc xu ly bang penalty M1-ngươc (-15) chu khong phai hard block.
            # ================================================================

            _fast_tr = self._trend_direction(df_micro, fast=20, slow=50)

            # --- XAC DINH _T (huong trade) - linh hoat ---
            # Uu tien: F != 0 -> lay F. F=0 -> lay M1. M1=0 -> lay M2.
            # Neu tat ca = 0 -> skip (khong co tin hieu huong nao).
            if _fast_tr != 0:
                _T = _fast_tr
            elif macro_trend != 0:
                _T = macro_trend
            elif macro_4h != 0:
                _T = macro_4h
            else:
                logger.info(f"{symbol}: GATE skip - F=M1=M2=0, thi truong sideway hoan toan")
                return _block("skip - F=M1=M2=0 (no direction signal)")

            # --- ABSOLUTE BLOCK 0: VOLUME COLLAPSE (khong co ai trade) ---
            _vrat_early = _volume_ratio(df_micro)
            if _vrat_early < 0.25:
                return _block(f"skip - volume collapse {_vrat_early:.2f}x (no liquidity)")

            # --- ABSOLUTE BLOCK 1: ADX < 8 (thi truong chet) ---
            _atrm = _atr_for_sl if _atr_for_sl > 0 else float(
                (df_micro["high"].iloc[-14:] - df_micro["low"].iloc[-14:]).mean())
            if math.isnan(adx) or adx < 8.0:
                logger.info(f"{symbol}: GATE skip - ADX={adx:.1f}<8 (thi truong chet flat)")
                return _block("skip - ADX<8 (dead flat)")

            # --- ABSOLUTE BLOCK 1b: RANGING MARKET (TNSRUSDT-type) ---
            # Ranging = ADX thap + khong co cau truc HH/HL + volume thap.
            # Khong co scenario nao trong 50K xay ra trong flat range. Block.
            _hh_ll_early = 0
            if len(df_micro) >= 15:
                _hr_e = df_micro["high"].iloc[-8:].max();  _hp_e = df_micro["high"].iloc[-15:-8].max()
                _lr_e = df_micro["low"].iloc[-8:].min();   _lp_e = df_micro["low"].iloc[-15:-8].min()
                if   _hr_e > _hp_e and _lr_e > _lp_e: _hh_ll_early = 1
                elif _hr_e < _hp_e and _lr_e < _lp_e: _hh_ll_early = -1
            # Ranging block: cần TẤT CẢ 3 điều kiện cùng lúc (tránh block trend mới hình thành)
            # ADX rất thấp (<12) + hoàn toàn không có structure + volume collapse nhẹ (<1.2x)
            if adx < 12 and _hh_ll_early == 0 and _vrat_early < 1.2:
                return _block(f"skip - ranging market ADX={adx:.0f}<12 no-structure vol={_vrat_early:.2f}x")

            # --- ABSOLUTE BLOCK 2: RSI EXTREME 15/85 ---
            if not math.isnan(rsi_now):
                if _T == -1 and rsi_now < 15:
                    return _block(f"skip SHORT - RSI {rsi_now:.0f}<15 (extreme oversold)")
                if _T == 1 and rsi_now > 85:
                    return _block(f"skip LONG - RSI {rsi_now:.0f}>85 (extreme overbought)")
                # ANTI-DU-DINH: RSI >= 78 + gia o dinh range 2h = du dinh long / short o day
                # Khong high_conviction: se ban o day / mua o dinh - rui ro cao, bo qua
                if not _high_conviction:
                    if _T == 1 and rsi_now >= 78 and _m2h_pos >= 0.82:
                        return _block(f"skip LONG - RSI {rsi_now:.0f}>=78 + 2h top {_m2h_pos:.0%} (du dinh)")
                    if _T == -1 and rsi_now <= 22 and _m2h_pos <= 0.18:
                        return _block(f"skip SHORT - RSI {rsi_now:.0f}<=22 + 2h bottom {_m2h_pos:.0%} (short o day)")

            # --- ABSOLUTE BLOCK 3: FRESH MOVE EXTREME (>3.0 ATR) ---
            _live_p = _range_live_price if _range_live_price > 0 else float(df_micro["close"].iloc[-1])
            _mv10_abs = 0.0
            if _atrm > 0 and len(df_micro) >= 11:
                _ref10 = float(df_micro["close"].iloc[-11])
                _mv10_abs = (_live_p - _ref10) / _atrm
                if _T == -1 and _mv10_abs < -3.0:
                    return _block(f"skip SHORT - fresh dump {_mv10_abs:.2f}ATR/10n (spike dien cuong)")
                if _T == 1 and _mv10_abs > 3.0:
                    return _block(f"skip LONG - fresh pump {_mv10_abs:.2f}ATR/10n (spike dien cuong)")

            # --- ABSOLUTE BLOCK 4: CANDLE EXHAUSTION (vao cuoi song - du day/du dinh) ---
            # Block khi 3 nen gan nhat da chay MANH cung chieu muon vao -> van cuoi song.
            # Vi du LQTYUSDT: cay nen do lon drop 4+ ATR, bot SHORT cuoi nen = du day.
            # Nguong: 3c bodies + single last candle. High-conviction bypass (F+M1+M2+ADX25+vol).
            if _atrm > 0 and len(df_micro) >= 5 and not _high_conviction:
                _c3 = df_micro.iloc[-4:-1]
                _c3_bodies = (_c3["close"] - _c3["open"]).values
                _run_up   = sum(float(b) for b in _c3_bodies if b > 0) / _atrm  # ATR of green bodies
                _run_down = sum(-float(b) for b in _c3_bodies if b < 0) / _atrm  # ATR of red bodies
                _last_body_raw = float(df_micro["close"].iloc[-2]) - float(df_micro["open"].iloc[-2])
                _last_green_atr = max(0.0, _last_body_raw) / _atrm
                _last_red_atr   = max(0.0, -_last_body_raw) / _atrm
                # LONG khi gia vua pump = du dinh; SHORT khi gia vua dump = du day
                if _T == 1 and (_run_up >= 1.5 or _last_green_atr >= 2.0):
                    return _block(
                        f"skip LONG - 3c pump {_run_up:.1f}ATR last={_last_green_atr:.1f}ATR (du dinh)"
                    )
                if _T == -1 and (_run_down >= 1.5 or _last_red_atr >= 2.0):
                    return _block(
                        f"skip SHORT - 3c dump {_run_down:.1f}ATR last={_last_red_atr:.1f}ATR (du day short)"
                    )

            # ================================================================
            # UNIFIED SCORE - tat ca tin hieu deu la diem
            # Tong diem toi da ly thuyet: ~130+ nhung clamp 0-100
            # ================================================================

            # S1: FAST EMA alignment (20 pts)
            if _fast_tr == _T:       _s_fast = 20
            elif _fast_tr == 0:      _s_fast = 0    # F neutral: khong bonus, khong penalty
            else:                    _s_fast = -15  # F nguoc: penalty (KAITO risk)

            # S2: MACRO alignment - M1 (15 pts)
            if macro_trend == _T:    _s_m1 = 15
            elif macro_trend == 0:   _s_m1 = 0
            else:                    _s_m1 = -12   # M1 nguoc: penalty nang (trend nguoc lon)

            # S3: MACRO M2/4h alignment (10 pts)
            if macro_4h == _T:       _s_m2 = 10
            elif macro_4h == 0:      _s_m2 = 0
            else:                    _s_m2 = -8

            # S4: ADX strength bonus
            if not math.isnan(adx):
                if adx >= 25:        _s_adx = 10
                elif adx >= 15:      _s_adx = 5
                else:                _s_adx = -5   # ADX 8-15: choppy penalty
            else:                    _s_adx = 0

            # S5: Volume direction — require current vol >= 50% avg for BOTH bonus and penalty
            # to avoid stale/noise OBV signals in low-volume consolidation
            _vol_dir_active = _vrat_early >= 0.50
            if _vwt_dir == _T and _vwt_str >= 0.4 and _vol_dir_active:    _s_vol_dir = 8
            elif _vwt_dir == -_T and _vwt_str >= 0.4 and _vol_dir_active: _s_vol_dir = -10
            else:                                                            _s_vol_dir = 0

            # S6: RSI zone penalty/bonus (30/70 range)
            if not math.isnan(rsi_now):
                if _T == -1 and rsi_now < 30:   _s_rsi_zone = -12  # SHORT trong oversold
                elif _T == 1 and rsi_now > 70:  _s_rsi_zone = -12  # LONG trong overbought
                elif _T == -1 and rsi_now < 40: _s_rsi_zone = -5   # SHORT near oversold
                elif _T == 1 and rsi_now > 60:  _s_rsi_zone = -5   # LONG near overbought
                else:                            _s_rsi_zone = 0
            else:                                _s_rsi_zone = 0

            # S7: Extension penalty (chase prevention)
            _price_now = _range_live_price if _range_live_price > 0 else float(df_micro["close"].iloc[-1])
            _ema21m = compute_ema(df_micro["close"], 21).iloc[-1]
            _ext = (_price_now - _ema21m) / _atrm if _atrm > 0 else 0.0
            _ext_dir = _ext * _T   # duong = extended cung chieu _T (chase)
            if _ext_dir > 2.5:      _s_ext = -20
            elif _ext_dir > 1.5:    _s_ext = -10
            elif _ext_dir > 0.8:    _s_ext = -3
            elif _ext_dir < -0.5:   _s_ext = 5    # pullback ve EMA = tot
            else:                    _s_ext = 0

            # S8: Fresh move penalty (khong chase pump/dump vua xay ra)
            _s_fresh = 0
            if _atrm > 0:
                _mv10_dir = _mv10_abs * _T   # duong = move cung chieu _T (chase)
                if _mv10_dir > 2.0:     _s_fresh = -15
                elif _mv10_dir > 1.2:   _s_fresh = -8
                # 5-candle
                if len(df_micro) >= 6:
                    _ref5 = float(df_micro["close"].iloc[-6])
                    _mv5_dir = ((_live_p - _ref5) / _atrm) * _T
                    if _mv5_dir > 1.8:   _s_fresh = min(_s_fresh, -15)
                    elif _mv5_dir > 1.0: _s_fresh = min(_s_fresh, -8)

            # S9: Large candle exhaustion — check BOTH current candle AND recent window
            # LRCUSDT-type: pump spike 5 candles ago, current candle small -> old check misses it
            _s_candle_body = 0
            if len(df_micro) >= 2 and _atrm > 0:
                # Check candle cuoi (original)
                _lc_o = float(df_micro["open"].iloc[-1])
                _lc_c = float(df_micro["close"].iloc[-1])
                _lc_body_dir = ((_lc_c - _lc_o) / _atrm) * _T
                if _lc_body_dir > 2.0:   _s_candle_body = -10
                elif _lc_body_dir > 1.5: _s_candle_body = -5
                # NEW: check recent spike AGAINST trade direction (last 10 candles)
                # Neu co spike lon NGUOC chieu trong 10 nen gan nhat -> penalty
                if len(df_micro) >= 10:
                    _recent10 = df_micro.iloc[-10:]
                    _max_counter_body = 0.0
                    for _ri in range(len(_recent10)):
                        _rb = (float(_recent10["close"].iloc[_ri]) - float(_recent10["open"].iloc[_ri])) / _atrm
                        if _rb * (-_T) > _max_counter_body:
                            _max_counter_body = _rb * (-_T)
                    if _max_counter_body > 2.5:
                        _s_candle_body = min(_s_candle_body, -12)  # spike lon nguoc chieu
                    elif _max_counter_body > 1.8:
                        _s_candle_body = min(_s_candle_body, -6)

            # S10: IMM momentum
            if _imm == _T:       _s_imm = 8
            elif _imm == 0:      _s_imm = -4
            else:                _s_imm = -10

            # S11: HH/HL candle structure
            _hh_ll = 0
            _hh_ll = _hh_ll_early  # reuse from ranging block above
            if _hh_ll == _T:     _s_struct = 6
            elif _hh_ll == 0:    _s_struct = 0
            else:                _s_struct = -6

            # S12: Anti-breakout (SHORT khi break dinh, LONG khi break day = nguoc)
            _s_breakout = 0
            if len(df_micro) > 11:
                _prior_high = float(df_micro["high"].iloc[-11:-1].max())
                _prior_low  = float(df_micro["low"].iloc[-11:-1].min())
                _cur_close  = float(df_micro["close"].iloc[-1])
                if _T == -1 and _cur_close > _prior_high: _s_breakout = -8
                if _T == 1 and _cur_close < _prior_low:   _s_breakout = -8

            # --- PRE-SCENARIO UNIFIED SCORE ---
            _pre_score = (_s_fast + _s_m1 + _s_m2 + _s_adx +
                          _s_vol_dir + _s_rsi_zone + _s_ext + _s_fresh +
                          _s_candle_body + _s_imm + _s_struct + _s_breakout)

            # Minimum pre-score: -30 (qua nhieu tin hieu xau, khong co scenario nao cover)
            if _pre_score < -30:
                logger.info(
                    f"{symbol}: GATE skip - pre_score={_pre_score} < -30 (qua nhieu tin hieu xau) | "
                    f"F={_fast_tr}({_s_fast}) M1={macro_trend}({_s_m1}) M2={macro_4h}({_s_m2}) "
                    f"ADX={adx:.0f}({_s_adx}) voldir={_s_vol_dir} rsi={_s_rsi_zone} "
                    f"ext={_ext:.1f}({_s_ext}) fresh={_s_fresh} imm={_imm}({_s_imm})")
                return _block(f"skip - pre_score={_pre_score} (too many opposing signals)")

            # ================================================================
            # SCENARIO GATE v6 - 10,000 HQ SCENARIO MATCHER + PRE-SCORE
            # pre_score (da tinh tren) la bonus/penalty context (EMA/macro/ADX/ext/fresh/imm).
            # Scenario signals (MACD/RSI/vol/BOS/candle) la core classification.
            # TONG: pre_score (context) + scenario_score (signal) -> final score.
            # Threshold: 50. Rat linh hoat - lenh tot (nhieu tin hieu) se qua, lenh xau bi chan.
            # ================================================================
            _macd_line, _macd_sig, _macd_hist = compute_macd(df_micro["close"])
            _prev_hist = float("nan")
            if len(df_micro) >= 28:
                _, _, _prev_hist = compute_macd(df_micro["close"].iloc[:-1])
            _cpat = _candle_pattern_v2(df_micro)
            _vrat = _vrat_early  # reuse from absolute block check above

            # --- SIGNAL 1: EMA STATE (25 pts max) — expanded with 50K EMA50/200 signals ---
            _ema_cat, _ema_dir_v = _ema_state_v2(df_micro)
            _EMA_MAP = {
                # Original 10K signals
                "bullish_stack":          (1,  25),  # 436/3650 = 12% in 95-98%
                "bearish_stack":          (-1, 25),  # 453/3650 = 12%
                "golden_cross":           (1,  20),  # EMA20/50 cross
                "death_cross":            (-1, 20),
                "price_cross_up":         (1,  15),
                "price_cross_down":       (-1, 15),
                # NEW from 50K: EMA50/200 signals
                "ema50_200_golden":       (1,  23),  # Classic Golden Cross: 469/3650=13% #1 EMA in 95-98%
                "ema50_200_death":        (-1, 23),  # Classic Death Cross: 409/3650=11%
                "price_cross_ema200_up":  (1,  20),  # Major signal: 3393 in 50K
                "price_cross_ema200_down":(-1, 20),
                "pullback_ema20_bull":    (1,  18),  # Pullback to EMA20 in uptrend: 3513 in 50K
                "pullback_ema20_bear":    (-1, 18),
                "price_bounce_ema200":    (1,  16),  # EMA200 bounce: 3444 in 50K
                "price_reject_ema200":    (-1, 16),
                "price_bounce_ema50":     (1,  14),  # EMA50 bounce in uptrend: 3387 in 50K
                "price_reject_ema50":     (-1, 14),
                "neutral":                (0,   3),
            }
            _ema_base = _EMA_MAP.get(_ema_cat, (0, 0))[1]
            if _ema_dir_v == _T:    _ema_pts = _ema_base
            elif _ema_dir_v == -_T: _ema_pts = -12
            else:                   _ema_pts = _ema_base

            # --- SIGNAL 2: MACD STATE (22 pts max) + Hidden Divergence bonus ---
            _macd_cat = _macd_state(_macd_hist, _prev_hist)
            _MACD_MAP = {
                "bull_crossover":   (1,  22),   # zero-line cross = strongest
                "bear_crossover":   (-1, 22),
                "hist_up":          (1,  18),   # above 0, improving momentum
                "hist_down":        (-1, 18),
                "bull_divergence":  (1,  14),   # below 0 but improving — weaker than above-zero hist_up
                "bear_divergence":  (-1, 14),
                "above0_bull":      (1,  12),
                "below0_bear":      (-1, 12),
                "neutral":          (0,  0),
            }
            _macd_dir_v, _macd_base = _MACD_MAP.get(_macd_cat, (0, 0))
            if _macd_dir_v == _T:    _macd_pts = _macd_base
            elif _macd_dir_v == -_T: _macd_pts = -8
            else:                    _macd_pts = 0

            # NEW: MACD Hidden Divergence bonus — #1 signal in 95-98% tier (881+784=1665)
            _macd_hidden_div = False
            _macd_hidden_cat = "none"
            try:
                if len(df_micro) >= 50:
                    _ml_s, _, _mh_s = compute_macd_series(df_micro["close"])
                    _macd_div = _detect_divergence_50k(df_micro["close"], _ml_s, lookback=30)
                    if _T == 1 and _macd_div["hidden_bull"]:
                        _macd_hidden_div = True; _macd_hidden_cat = "hidden_bull"
                        _macd_pts = max(_macd_pts, 22)  # at least regular div score
                    elif _T == -1 and _macd_div["hidden_bear"]:
                        _macd_hidden_div = True; _macd_hidden_cat = "hidden_bear"
                        _macd_pts = max(_macd_pts, 22)
            except Exception:
                pass

            # --- SIGNAL 3: RSI ZONE (18 pts max) + Hidden Divergence bonus ---
            _rsi_cat = _rsi_zone(rsi_now, df_micro)
            _RSI_MAP = {
                "bull_divergence":  (1,  18),
                "bear_divergence":  (-1, 18),
                "oversold":         (1,  12),
                "overbought":       (-1, 12),
                "near_oversold":    (1,   8),
                "near_overbought":  (-1,  8),
                "neutral":          (0,   0),
            }
            _rsi_dir_v, _rsi_base = _RSI_MAP.get(_rsi_cat, (0, 0))
            if _rsi_dir_v == _T:    _rsi_pts = _rsi_base
            elif _rsi_dir_v == -_T: _rsi_pts = -5
            else:                   _rsi_pts = 0

            # NEW: RSI Hidden Divergence — 597+567=1164 in 95-98% tier
            _rsi_hidden_div = False
            try:
                if len(df_micro) >= 30:
                    _rsi_s = compute_rsi(df_micro["close"], 14)
                    _rsi_div = _detect_divergence_50k(df_micro["close"], _rsi_s, lookback=25)
                    if _T == 1 and _rsi_div["hidden_bull"]:
                        _rsi_hidden_div = True
                        _rsi_pts = max(_rsi_pts, 18)
                    elif _T == -1 and _rsi_div["hidden_bear"]:
                        _rsi_hidden_div = True
                        _rsi_pts = max(_rsi_pts, 18)
            except Exception:
                pass

            # --- SIGNAL 4: VOLUME (18 pts max) — adds volume climax reversal ---
            if _vrat >= 2.0:
                _vol_cat = "spike"
                _vol_pts = 18 if _vwt_dir == _T else 5
            elif _vrat >= 1.5:
                _vol_cat = "high"
                _vol_pts = 12 if _vwt_dir == _T else 4
            elif _vrat >= 0.7 and _vwt_dir == _T:
                # NEW: "Above-average volume at support/resistance test" — 7850 in 50K
                _vol_cat = "above_avg"
                _vol_pts = 7
            else:
                _vol_cat = "normal"
                _vol_pts = 4

            # --- SIGNAL 5: BOS / CHoCH (12 pts max) — directional scoring ---
            # BOS confirms trade direction (+12). CHoCH = structure flipping AGAINST direction = penalty.
            # 50K: CHoCH 757/3650=21% in 95-98% WHEN it matches direction; opposing CHoCH = stop signal.
            if _hh_ll == _T:    _bos_cat = "bos";     _bos_pts = 12
            elif _hh_ll == -_T: _bos_cat = "choch";   _bos_pts = -8  # structure flipping against us
            else:               _bos_cat = "neutral";  _bos_pts = 3

            # --- SIGNAL 6: CANDLE PATTERN (8 pts max) — expanded with 50K patterns ---
            if _cpat == _T:    _candle_pts = 8
            elif _cpat == -_T: _candle_pts = -5
            else:              _candle_pts = 0

            # --- SIGNAL 7 (NEW): STOCHASTIC (10 pts max) — 50K: oversold/overbought cross ---
            _stoch_pts = 0; _stoch_cat = "neutral"
            try:
                if len(df_micro) >= 17:
                    _roll_hi = df_micro["high"].rolling(14).max()
                    _roll_lo = df_micro["low"].rolling(14).min()
                    _sk_s = 100 * (df_micro["close"] - _roll_lo) / (_roll_hi - _roll_lo + 1e-12)
                    _sd_s = _sk_s.rolling(3).mean()
                    _sk = float(_sk_s.iloc[-1]); _sd = float(_sd_s.iloc[-1])
                    _sk_p = float(_sk_s.iloc[-2]) if len(_sk_s) >= 2 else float("nan")
                    _stoch_cat = _stoch_state_50k(_sk, _sd, _sk_p)
                    _STOCH_MAP = {
                        "oversold_cross_up":        (1,  10),  # 449/3650=12% in 95-98%
                        "overbought_cross_down":     (-1, 10),  # 465/3650=13%
                        "embedded_oversold_cross":   (1,   8),  # 469/3650=13%
                        "embedded_overbought_cross": (-1,  8),  # 429/3650=12%
                        "oversold":                  (1,   5),
                        "overbought":                (-1,  5),
                        "rising_midzone":            (1,   3),   # K=20-50: building momentum
                        "upper_midzone":             (0,   0),   # K=50-80: embedded strength, neutral (not bearish)
                        "neutral":                   (0,   0),
                    }
                    _stoch_dir_v, _stoch_base = _STOCH_MAP.get(_stoch_cat, (0, 0))
                    if _stoch_dir_v == _T:    _stoch_pts = _stoch_base
                    elif _stoch_dir_v == -_T: _stoch_pts = -4
                    else:                     _stoch_pts = 0
                    # Stoch Divergence bonus — 479+453=932 in 95-98% tier
                    if len(df_micro) >= 30:
                        _stoch_div = _detect_divergence_50k(df_micro["close"], _sk_s.dropna(), lookback=20)
                        if _T == 1 and (_stoch_div["regular_bull"] or _stoch_div["hidden_bull"]):
                            _stoch_pts = max(_stoch_pts, 8)
                        elif _T == -1 and (_stoch_div["regular_bear"] or _stoch_div["hidden_bear"]):
                            _stoch_pts = max(_stoch_pts, 8)
            except Exception:
                pass

            # --- SIGNAL 8 (NEW): SESSION TIMING (12 pts max) — London-NY=31% of 95-98% tier ---
            _sess_cat, _sess_pts = _session_score_50k()

            # --- TONG DIEM: scenario signals + pre_score context ---
            _pre_clamped  = max(-30, min(30, _pre_score))
            _scenario_pts = (_ema_pts + _macd_pts + _rsi_pts + _vol_pts +
                             _bos_pts + _candle_pts + _stoch_pts + _sess_pts)
            _score = max(0, min(100, _scenario_pts + _pre_clamped))

            # --- XAC DINH TIER: 95-98% vs 90-95% vs 85-90% ---
            # 95-98%: MACD hidden div + RSI hidden div + vol spike (top combo from 50K)
            # 90-95%: regular MACD div OR RSI div + vol spike
            # 85-90%: single strong signal (EMA50/200 cross, full stack + momentum)
            _macd_any_div = _macd_hidden_div or _macd_cat in ("bull_divergence", "bear_divergence") and _macd_dir_v == _T
            _rsi_any_div  = _rsi_hidden_div  or _rsi_cat  in ("bull_divergence", "bear_divergence") and _rsi_dir_v == _T
            _vol_spike    = _vrat >= 2.0 and _vwt_dir == _T
            _is_9598 = _macd_hidden_div and _rsi_hidden_div and _vol_spike
            _is_9095 = not _is_9598 and (_macd_any_div or _rsi_any_div) and _vol_spike
            _is_8590 = not _is_9598 and not _is_9095 and (
                _ema_cat in ("bullish_stack", "bearish_stack", "ema50_200_golden", "ema50_200_death") and _ema_dir_v == _T
            )
            _is_90plus = _is_9598 or _is_9095
            _div_combo = _macd_any_div and _vol_spike
            _rsi_div_aligned = _rsi_any_div
            _stack_momentum = (
                _ema_cat in ("bullish_stack", "bearish_stack") and _ema_dir_v == _T and
                _vol_spike and _macd_cat in ("bull_crossover","bear_crossover","hist_up","hist_down") and _macd_dir_v == _T
            )

            _threshold = 50
            # HIGH-CONVICTION FAST PATH: F+M1+M2 all aligned + ADX>=25 + vol>=1.5x
            # Lower threshold to 35 ("90%+ chac chan co loi") - bot self-trade outside scenarios
            _high_conviction = (
                _fast_tr == _T and macro_trend == _T and macro_4h == _T
                and adx >= 25
                and _vrat_early >= 1.5
            )
            _effective_threshold = 35 if _high_conviction else _threshold
            if _score < _effective_threshold:
                logger.info(
                    f"{symbol}: SCENARIO v6+ skip - score={_score}/100 < {_effective_threshold}"
                    f"{'(high-conv 35)' if _high_conviction else '(std 50)'} | "
                    f"scenario={_scenario_pts} pre={_pre_clamped} | "
                    f"EMA={_ema_cat}({_ema_pts}) MACD={_macd_cat}({_macd_pts})[hid={_macd_hidden_cat}] "
                    f"RSI={_rsi_cat}({_rsi_pts})[hid={_rsi_hidden_div}] vol={_vol_cat}({_vol_pts}) "
                    f"BOS={_bos_cat}({_bos_pts}) candle={_cpat}({_candle_pts}) "
                    f"stoch={_stoch_cat}({_stoch_pts}) sess={_sess_cat}({_sess_pts}) | "
                    f"F={_fast_tr}({_s_fast}) M1={macro_trend}({_s_m1}) ADX={adx:.0f}({_s_adx}) "
                    f"imm={_imm}({_s_imm}) ext={_ext:.1f}({_s_ext}) fresh={_s_fresh}")
                return _block(f"skip - score={_score} < {_effective_threshold} (not in 50K HQ scenarios)")

            # --- TP VA CONVICTION THEO TIER (3 tiers tu 50K scenarios) ---
            # TP = TARGET ban dau. risk_manager se clamp TP vao: [fee_floor, 2xATR].
            # Fee floor = TOTAL_ROUND_TRIP_COST * leverage * 1.5 (luon co loi sau phi).
            # 95-98%: TP 9.8% ROI, avg R:R 5.84, leverage 10.3x
            # 90-95%: TP 9.1% ROI, avg R:R 5.25
            # 85-90%: TP 7.9% ROI, avg R:R 4.61
            if _is_9598:
                _tp_by_score = 0.098; _conv = 0.97; _tier = "95-98%"
            elif _is_9095:
                _tp_by_score = 0.091; _conv = 0.92; _tier = "90-95%"
            elif _is_8590:
                _tp_by_score = 0.079; _conv = 0.87; _tier = "85-90%"
            elif _is_90plus:
                _tp_by_score = config.TP_ROI_MAX; _conv = 0.90; _tier = "90%+"
            else:
                # Standard tier: TP scale theo score, san 3% (risk_manager tinh fee floor that su)
                _tp_by_score = config.TP_ROI_MIN + (
                    (_score - _threshold) / (100.0 - _threshold)
                ) * (config.TP_ROI_MAX - config.TP_ROI_MIN)
                _tp_by_score = max(config.TP_ROI_MIN, min(config.TP_ROI_MAX, _tp_by_score))
                _conv = _score / 100.0; _tier = "standard"

            if best.direction != _T:
                logger.info(f"{symbol}: TREND override -> {'LONG' if _T==1 else 'SHORT'} (signal={best.direction})")
            best.direction       = _T
            best.tp_roi_override = round(_tp_by_score, 4)
            best.strength        = max(best.strength, min(1.0, _conv))
            best.gate_score      = float(_score)   # truyen score cho risk_manager tinh von

            logger.info(
                f"{symbol}: SCENARIO v6 PASS | {'L' if _T==1 else 'S'} | "
                f"tier={_tier} score={_score}/100 (scenario={_scenario_pts}+pre={_pre_clamped}) | "
                f"EMA={_ema_cat}({_ema_pts}) MACD={_macd_cat}({_macd_pts}) "
                f"RSI={_rsi_cat}({_rsi_pts}) vol={_vol_cat}[{_vrat:.1f}x]({_vol_pts}) "
                f"BOS={_bos_cat}({_bos_pts}) candle={_cpat}({_candle_pts}) | "
                f"F={_fast_tr}({_s_fast}) M1={macro_trend}({_s_m1}) ADX={adx:.0f}({_s_adx}) "
                f"imm={_imm}({_s_imm}) | "
                f"90%+={_is_90plus} TP={_tp_by_score*100:.1f}% conv={_conv:.2f}"
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

        # Flip-guard: block neu chieu moi NGUOC voi lenh truoc trong FLIP_COOLDOWN_SEC
        # NGOAI LE: _direction_flipped=True -> AEQ/VOL-TREND da phan tich va quyet dinh
        # dao chieu -> khong block (flip la ket qua phan tich, khong phai random)
        _last_dir_info = self.executor._last_direction.get(symbol)
        if _last_dir_info is not None and not _direction_flipped:
            _last_dir, _last_close_ts = _last_dir_info
            import time as _time_mod2
            if _time_mod2.time() - _last_close_ts < _flip_cooldown_sec and _last_dir != best.direction:
                _rem_flip = int(_last_close_ts + _flip_cooldown_sec - _time_mod2.time())
                logger.info(f"{symbol}: FLIP-GUARD block {'LONG' if best.direction==1 else 'SHORT'} (last={'LONG' if _last_dir==1 else 'SHORT'}, {_rem_flip}s remaining)")
                return False

        # UNIVERSAL QUALITY GATE - kiem tra lan cuoi truoc khi bat lenh
        _mo_qg_ok, _mo_qg_msg = self._quality_gate(
            best.direction, df_micro, rsi_now, _sc_pos, _sp,
            is_reversal=False, is_breakout=False,
            pos_24h=_24h_pos, hi_24h=_24h_hi, lo_24h=_24h_lo,
            strong_trend=(_strong_bull_trend and best.direction == 1) or (_strong_bear_trend and best.direction == -1))
        if not _mo_qg_ok:
            logger.info(f"{symbol}: [QUALITY-GATE] MOMENTUM blocked - {_mo_qg_msg}")
            return False

        # FINAL TREND CONSISTENCY CHECK - chay SAU TAT CA flip logic.
        # Bat ky flip nao bien LONG thanh SHORT trong uptrend, hoac SHORT thanh LONG trong downtrend
        # deu bi block tai day. Day la tuong chua cuoi cung chong sai chieu.
        #
        # Level 1 (manh): _all_tfs_bull/bear (ca 2 TF xac nhan) -> block toan phan
        # Level 2 (TB):   macro_trend + scalp_trend cung chieu nguoc signal -> block
        #   (bat duoc ZAMAUSDT/AVAAIUSDT: scalp=-1 macro=-1 nhung flip thanh LONG)
        if _hf_mode:
            if _all_tfs_bull and best.direction == -1:
                return _block(
                    f"FINAL-CHECK L1: SHORT trong confirmed uptrend "
                    f"(macro={macro_trend} macro4h={macro_4h}) -> sai chieu"
                )
            if _all_tfs_bear and best.direction == 1:
                return _block(
                    f"FINAL-CHECK L1: LONG trong confirmed downtrend "
                    f"(macro={macro_trend} macro4h={macro_4h}) -> sai chieu"
                )
            # Level 2: CA 3 TF cung chieu nguoc entry -> sai chieu ro rang
            # Yeu cau macro_4h de tranh block top coin khi scalp/macro flip tam thoi (consolidation)
            # scalp oscillates trong trending market -> chi block khi macro_4h CUNG xac nhan
            if macro_trend == -1 and scalp_trend == -1 and macro_4h <= -1 and best.direction == 1:
                return _block(
                    f"FINAL-CHECK L2: LONG nhung ca 3 TF bear "
                    f"(macro={macro_trend} scalp={scalp_trend} macro4h={macro_4h}) -> sai chieu"
                )
            if macro_trend == 1 and scalp_trend == 1 and macro_4h >= 1 and best.direction == -1:
                return _block(
                    f"FINAL-CHECK L2: SHORT nhung ca 3 TF bull "
                    f"(macro={macro_trend} scalp={scalp_trend} macro4h={macro_4h}) -> sai chieu"
                )

        # HIGH-VOL CAPITAL BOOST: coin >= 10M USDT volume + confirmed trend -> cap them von
        # gate_score mac dinh = 0 cho non-scenario -> risk_manager dung potential fallback (~3-5%).
        # Boost gate_score len 70-85 cho coin lon/priority trong trend manh -> cap 4-7% equity.
        # Khong boost scenario entries (da co gate_score rieng tu scenario engine).
        _cur_gs = getattr(best, 'gate_score', 0.0)
        if _cur_gs < 50:   # chua co gate_score tu scenario
            # _is_high_vol da tinh o tren tu volume_map (>= 10M)
            _trend_aligned = (best.direction == 1 and _strong_trend_up) or (best.direction == -1 and _strong_trend_dn)
            if _is_high_vol and _trend_aligned:
                best.gate_score = 85.0   # coin lon + all-3-TF trend -> 6.6% equity
            elif _is_high_vol and is_priority:
                best.gate_score = 70.0   # coin lon + priority queue -> 4.5% equity
            elif is_priority and _trend_aligned:
                best.gate_score = 75.0   # priority coin + trend manh -> 5.5% equity
            elif is_priority:
                best.gate_score = 60.0   # priority coin (high-vol nhung trend chua ro) -> 2.4% equity

        self.executor.execute_signal(symbol, best, equity, open_positions, is_priority=is_priority)
        return True


# -- Entry point ---------------------------------------------------------------

if __name__ == "__main__":
    bot = TradingBot()
    bot.run()
