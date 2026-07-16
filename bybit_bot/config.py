"""
Bybit Futures Auto Trading Bot - Configuration
"""

import os

# --- API Credentials ----------------------------------------------------------
API_KEY    = os.getenv("BYBIT_API_KEY", "YOUR_API_KEY_HERE")
API_SECRET = os.getenv("BYBIT_API_SECRET", "YOUR_API_SECRET_HERE")
TESTNET    = os.getenv("BYBIT_TESTNET", "false").lower() == "true"

# --- Market Scanner -----------------------------------------------------------
TOP_N_SYMBOLS        = 200
MIN_VOLUME_USDT_24H      = 1_000_000   # filter dau vao khi scan 200 coins
SCAN_INTERVAL_SEC    = 3600

# --- Multi-Timeframe Analysis -------------------------------------------------
# 5m  x 1000 = 3.5 ngay — bat scalp signal ngan han
# 15m x 1000 = 10.4 ngay — signal chinh
# 1h  x 500  = 20 ngay — xu huong
# 4h  x 300  = 50 ngay — xu huong lon
TIMEFRAMES = {
    "micro":  "1",     # 1m: xac nhan entry (3 nen gan nhat)
    "scalp":  "5",     # 5m: scalp / signal ngan han
    "signal": "15",    # 15m: signal chinh
    "trend":  "60",    # 1h: xu huong
    "macro":  "240",   # 4h: xu huong lon
}
CANDLE_LIMIT_MICRO       = 300   # 300 nen 1m — du EMA on dinh
CANDLE_LIMIT_SCALP  = 1000
CANDLE_LIMIT_SIGNAL = 1000
CANDLE_LIMIT_TREND  = 500
CANDLE_LIMIT_MACRO  = 300

# --- Phi giao dich Bybit ------------------------------------------------------
TAKER_FEE      = 0.00055
MAKER_FEE      = 0.00020
ROUND_TRIP_FEE = TAKER_FEE * 2   # 0.11% tong phi ca 2 chieu

# --- Risk Management ----------------------------------------------------------
USE_MAX_LEVERAGE       = True
MAX_LEVERAGE           = 20    # Cap 20x — cao hon lam SL hit mat >100% margin
DEFAULT_LEVERAGE       = 10
# Risk-based sizing: RISK_PER_TRADE_PCT% equity mat neu SL hit (scale theo consensus)
# RISK_PER_TRADE_PCT=0.01 → max 1% equity mat moi lenh → 100 lenh SL lien tiep het account
RISK_PER_TRADE_PCT = 0.01   # Max 1% equity mat khi SL hit (base, scale voi consensus)
MAX_CAPITAL_PCT    = 0.10   # Max 10% equity dung lam margin moi lenh
MAX_OPEN_POSITIONS = 3      # Max 3/5 positions mo cung luc (top5 focus mode)
# SL/TP theo ATR — RR >= 1.3 sau phi (truoc: 1.5/1.5 = 1:1, sau phi am)
ATR_PERIOD        = 14
SL_ATR_MULT       = 1.5    # SL  = 1.5x ATR
TP1_ATR_MULT      = 1.5    # TP1 = 1.5x ATR — thuc te hon, exchange auto-close khi price reach
TP2_ATR_MULT      = 3.0    # TP2 = 3.0x ATR — cho 50% con lai sau partial close
BREAKEVEN_TRIGGER = 0.20   # Doi SL ve breakeven tai 20% den TP1 — du de tranh noise, set som de khong miss wick
PARTIAL_CLOSE_TRIGGER = 0.75  # Dong 50% position tai 75% den TP1, de 50% con lai chay den TP2

# --- Signal sensitivity -------------------------------------------------------
MIN_SIGNAL_STRENGTH = 0.60   # Tang tu 0.55 — chi lay signal chat luong cao
MIN_ADX             = 22     # Top5 mode: 22 (largecap BTC/ETH ADX thuong 20-35 trong trend)
MIN_CONSENSUS          = 5   # Top5 mode: can it nhat 5/7 strategies dong thuan (chat luong cao)
MIN_CONSENSUS_TRENDING = 5   # Khong con dung (khong co trending list), giu cho tuong thich
MIN_ATR_PCT         = 0.004  # Tang tu 0.003 — ATR >= 0.4% moi trade (bu phi + spread)
TRADE_SIZE_MULT         = 1    # Khong dung nua — risk-based sizing thay the

# --- Risk Guards --------------------------------------------------------------
# Daily max loss: neu tong PnL trong ngay < -(equity * MAX_DAILY_LOSS_PCT) → dung mo lenh moi
MAX_DAILY_LOSS_PCT   = 0.05   # 5% equity — dung ngay khi mat > 5% trong 1 ngay

# Max spread: neu bid-ask spread > nguong nay → khong entry (thanh khoan kem)
# Largecap (BTC/ETH): 0.05%, Altcoin: 0.15%
MAX_SPREAD_PCT_LARGE = 0.0005   # 0.05% cho BTC/ETH
MAX_SPREAD_PCT_ALT   = 0.0015   # 0.15% cho altcoin

# --- Execution ----------------------------------------------------------------
LOOP_INTERVAL_SEC    = 15   # check moi 15 giay

# --- Logging ------------------------------------------------------------------
LOG_FILE        = "trading_bot.log"
LOG_TRADES_FILE = "trades.json"
LOG_LEVEL       = "INFO"

# --- Telegram (tuy chon) ------------------------------------------------------
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
ENABLE_TELEGRAM  = bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID)
