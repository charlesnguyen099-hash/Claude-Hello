"""
Bybit Futures Auto Trading Bot - Configuration
"""

import os

# --- API Credentials ----------------------------------------------------------
API_KEY    = os.getenv("BYBIT_API_KEY", "YOUR_API_KEY_HERE")
API_SECRET = os.getenv("BYBIT_API_SECRET", "YOUR_API_SECRET_HERE")
TESTNET    = os.getenv("BYBIT_TESTNET", "false").lower() == "true"

# --- Market Scanner -----------------------------------------------------------
MIN_VOLUME_USDT_24H  = 100_000       # min 100K USDT/24h — bat ca coin nho co trend dep
SCAN_INTERVAL_SEC    = 30            # cap nhat danh sach trending moi 30s (giam API call)
SCAN_BUDGET_SEC      = 8.0           # xu ly coin trong toi da 8s moi tick

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
CANDLE_LIMIT_SIGNAL = 500    # 500 x 1m = ~8h — 1 API call, du cho moi indicator (EMA50 can 50)
CANDLE_LIMIT_SCALP  = 200    # 200 x 5m = ~17h
CANDLE_LIMIT_TREND  = 200    # 200 x 15m = ~2 ngay
CANDLE_LIMIT_MACRO  = 150    # 150 x 1h  = ~6 ngay

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
MAX_OPEN_POSITIONS = 10     # Max 10 positions — trade tat ca co hoi tot
# SL/TP theo ATR — RR >= 1.3 sau phi (truoc: 1.5/1.5 = 1:1, sau phi am)
ATR_PERIOD        = 14
SL_ATR_MULT       = 1.5    # SL  = 1.5x ATR (swing-based se dung high/low 5 nen 15m, clamp 1.5-3x ATR)
TP1_ATR_MULT      = 2.0    # TP1 = 2.0x ATR — dam bao loi sau phi 0.11%
TP2_ATR_MULT      = 3.5    # TP2 = 3.5x ATR — cho 50% con lai sau partial close
BREAKEVEN_TRIGGER = 0.60   # Doi SL ve breakeven tai 60% den TP1
PARTIAL_CLOSE_TRIGGER = 0.75  # Dong 50% position tai 75% den TP1, de 50% con lai chay den TP2

# Hard cap SL/TP theo % gia — tranh SL/TP phi ly khi ATR qua lon (coin pump/dip)
# SL toi da 4%: du rong de vuot qua spike tam thoi ma gia van co the quay dau
# TP1 toi da 6%, TP2 toi da 10%: TP phai co the dat duoc trong dieu kien binh thuong
SL_MAX_PCT   = 0.040   # 4%  — SL khong duoc rong hon 4%
TP1_MAX_PCT  = 0.50    # 50% — TP toi da 50% tu entry
TP2_MAX_PCT  = 0.50    # 50% — TP2 toi da 50% tu entry
SL_MIN_PCT   = 0.60    # 60% — SL toi thieu 60% tu entry

# --- Signal sensitivity -------------------------------------------------------
MIN_SIGNAL_STRENGTH = 0.60   # Chi lay signal chat luong cao
MIN_ADX             = 12     # ADX >= 12 cho 1m scalp — 18 qua cao, block het trong sideway/Asian session
MIN_CONSENSUS          = 2   # 2/7 strategies dong thuan — 3 qua cao, hiem co 3 strategy 15m dong thuan
MIN_CONSENSUS_TRENDING = 2   # Dong bo voi MIN_CONSENSUS
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
LOOP_INTERVAL_SEC    = 1    # minimum pause giua cac tick (rate limit only)
SYMBOL_COOLDOWN_SEC  = 60   # khong re-analyze cung coin trong 60s (tranh spam)

# --- Logging ------------------------------------------------------------------
LOG_FILE        = "trading_bot.log"
LOG_TRADES_FILE = "trades.json"
LOG_LEVEL       = "INFO"

# --- Telegram (tuy chon) ------------------------------------------------------
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
ENABLE_TELEGRAM  = bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID)
