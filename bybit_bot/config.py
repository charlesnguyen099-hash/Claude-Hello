"""
Bybit Futures Auto Trading Bot - Configuration
"""

import os

# --- API Credentials ----------------------------------------------------------
API_KEY    = os.getenv("BYBIT_API_KEY", "YOUR_API_KEY_HERE")
API_SECRET = os.getenv("BYBIT_API_SECRET", "YOUR_API_SECRET_HERE")
TESTNET    = os.getenv("BYBIT_TESTNET", "false").lower() == "true"

# --- Market Scanner -----------------------------------------------------------
TOP_N_SYMBOLS        = 20             # luon focus top 20 coin trending manh nhat
MIN_VOLUME_USDT_24H  = 10_000_000    # min 10M USDT/24h de dam bao thanh khoan
SCAN_INTERVAL_SEC    = 15            # cap nhat trending list moi 15 giay

# --- Multi-Timeframe Analysis -------------------------------------------------
# 1m  x 2000 = ~33h  — MAIN signal (scalp entry, nhanh, nhieu lenh hon)
# 5m  x 500  = ~42h  — 5m alignment check (tranh vao giua bounce)
# 15m x 500  = 5 ngay — trend confirmation (thay 1h)
# 1h  x 300  = 12 ngay — macro trend (thay 4h)
#
# Chuyen tu 15m/1h/4h sang 1m/5m/15m/1h: bat lenh nhanh hon, nhieu co hoi hon
# ATR cho SL/TP tinh tu 15m (khong phai 1m) de tranh SL qua chat bi noise hit
TIMEFRAMES = {
    "signal": "1",     # 1m: MAIN signal (was 15m)
    "scalp":  "5",     # 5m: alignment check
    "trend":  "15",    # 15m: trend confirmation (was 1h)
    "macro":  "60",    # 1h: macro trend (was 4h)
}
CANDLE_LIMIT_SIGNAL = 2000   # 2 API calls x 1000 = ~33h 1m data
CANDLE_LIMIT_SCALP  = 500    # 500 x 5m = ~42h
CANDLE_LIMIT_TREND  = 500    # 500 x 15m = ~5 ngay
CANDLE_LIMIT_MACRO  = 300    # 300 x 1h  = ~12 ngay

# --- Phi giao dich Bybit ------------------------------------------------------
TAKER_FEE      = 0.00055
MAKER_FEE      = 0.00020
ROUND_TRIP_FEE = TAKER_FEE * 2   # 0.11% tong phi ca 2 chieu

# --- Risk Management ----------------------------------------------------------
USE_MAX_LEVERAGE       = True
MAX_LEVERAGE           = 100   # Dung leverage cao nhat exchange cho phep moi coin
DEFAULT_LEVERAGE       = 10
# Risk-based sizing: RISK_PER_TRADE_PCT% equity mat neu SL hit (scale theo consensus)
# RISK_PER_TRADE_PCT=0.01 -> max 1% equity mat moi lenh -> 100 lenh SL lien tiep het account
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
MIN_SIGNAL_STRENGTH = 0.60   # Chi lay signal chat luong cao
MIN_ADX             = 18     # ADX >= 18 cho 1m scalp (xu huong nho hon nhung co that)
MIN_CONSENSUS          = 3   # 3/7 strategies dong thuan (4 qua cao cho 1m data)
MIN_CONSENSUS_TRENDING = 3   # Dong bo voi MIN_CONSENSUS
MIN_ATR_PCT         = 0.0005 # 0.05% cho 1m (ATR 1m nho hon 15m, largecap BTC ~0.03-0.08%)
TRADE_SIZE_MULT         = 1    # Khong dung nua — risk-based sizing thay the

# --- Risk Guards --------------------------------------------------------------
# Daily max loss: neu tong PnL trong ngay < -(equity * MAX_DAILY_LOSS_PCT) -> dung mo lenh moi
MAX_DAILY_LOSS_PCT   = 0.05   # 5% equity — dung ngay khi mat > 5% trong 1 ngay

# Max spread: neu bid-ask spread > nguong nay -> khong entry (thanh khoan kem)
# Largecap (BTC/ETH): 0.05%, Altcoin: 0.15%
MAX_SPREAD_PCT_LARGE = 0.0005   # 0.05% cho BTC/ETH
MAX_SPREAD_PCT_ALT   = 0.0015   # 0.15% cho altcoin

# --- Execution ----------------------------------------------------------------
LOOP_INTERVAL_SEC    = 10   # check moi 10 giay (nhanh hon de bat signal top20)

# --- Logging ------------------------------------------------------------------
LOG_FILE        = "trading_bot.log"
LOG_TRADES_FILE = "trades.json"
LOG_LEVEL       = "INFO"

# --- Telegram (tuy chon) ------------------------------------------------------
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
ENABLE_TELEGRAM  = bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID)
