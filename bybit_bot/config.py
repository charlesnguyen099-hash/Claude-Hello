"""
Bybit Futures Auto Trading Bot - Configuration
"""

import os

# --- API Credentials ----------------------------------------------------------
API_KEY    = os.getenv("BYBIT_API_KEY", "YOUR_API_KEY_HERE")
API_SECRET = os.getenv("BYBIT_API_SECRET", "YOUR_API_SECRET_HERE")
TESTNET    = os.getenv("BYBIT_TESTNET", "false").lower() == "true"

# --- Market Scanner -----------------------------------------------------------
TOP_N_SYMBOLS        = 200         # Top 200 cap giao dich
MIN_VOLUME_USDT_24H  = 1_000_000   # Volume 24h >= 1M USDT (mo rong de bat them tin hieu)
SCAN_INTERVAL_SEC    = 3600        # Quet lai top symbols moi 1 gio

# --- Multi-Timeframe Analysis -------------------------------------------------
# Bybit API tra ve toi da 1000 nen moi lan goi — lay du de phan tich xa nhat co the
# 15m x 1000 = 10.4 ngay du lieu signal
# 1h  x 500  = 20 ngay xu huong
# 4h  x 300  = 50 ngay xu huong lon
TIMEFRAMES = {
    "signal": "15",    # 15m: entry chinh
    "trend":  "60",    # 1h:  xu huong
    "macro":  "240",   # 4h:  xu huong lon
}
CANDLE_LIMIT_SIGNAL = 1000   # Toi da Bybit ho tro, ~10 ngay du lieu 15m
CANDLE_LIMIT_TREND  = 500    # ~20 ngay du lieu 1h
CANDLE_LIMIT_MACRO  = 300    # ~50 ngay du lieu 4h

# --- Strategy Selector --------------------------------------------------------
STRATEGY_EVAL_CANDLES  = 200   # Dung 200 nen gan nhat de backtest chon strategy
MIN_WIN_RATE           = 0.40  # Ha nguong de khong bo lo signal nho
STRATEGY_RESCAN_BARS   = 30    # Chay lai selector sau 30 nen

# --- Phi giao dich Bybit ------------------------------------------------------
TAKER_FEE      = 0.00055        # 0.055% moi lan vao/ra (market order)
MAKER_FEE      = 0.00020        # 0.020% (limit order)
ROUND_TRIP_FEE = TAKER_FEE * 2  # 0.11% tong phi ca 2 chieu

# --- Risk Management ----------------------------------------------------------
# SL co dinh: toi da mat 30% von moi lenh
SL_MAX_LOSS_PCT       = 0.30    # SL dat o muc mat toi da 30% capital bo vao lenh

# Von moi lenh: chia deu, khong all-in
CAPITAL_PER_TRADE_PCT = 0.10    # 10% von thuc moi lenh (10 lenh = 100% von)

USE_MAX_LEVERAGE       = True   # Tu dong lay leverage toi da cua tung cap tren Bybit
MAX_LEVERAGE           = 100    # Cap tren leverage
DEFAULT_LEVERAGE       = 20     # Dung khi khong lay duoc tu API
MAX_OPEN_POSITIONS     = 10     # Toi da 10 vi the dong thoi
MAX_POSITIONS_PER_SIDE = 5      # Toi da 5 long hoac 5 short

# ATR de tinh TP dong (SL khong dung ATR nua — dung 30% von thay the)
ATR_PERIOD         = 14
TP1_RR             = 1.5    # TP1 = Risk:Reward 1.5 (TP1 = 1.5 x sl_dist)
TP2_RR             = 3.0    # TP2 = Risk:Reward 3.0
TRAILING_STOP_ATR  = 1.0    # Trailing stop = 1.0 x ATR

# --- Signal sensitivity -------------------------------------------------------
# Ha nguong de khong bo qua bat ky signal nao
MIN_SIGNAL_STRENGTH = 0.40   # Chap nhan signal yeu hon (mac dinh 0.6, giam xuong 0.4)

# --- Execution ----------------------------------------------------------------
ORDER_TYPE        = "Market"
LOOP_INTERVAL_SEC = 60
RETRY_ATTEMPTS    = 3
RETRY_DELAY_SEC   = 2

# --- Logging ------------------------------------------------------------------
LOG_FILE        = "trading_bot.log"
LOG_TRADES_FILE = "trades.json"
LOG_LEVEL       = "INFO"

# --- Telegram Notifications (tuy chon) ----------------------------------------
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
ENABLE_TELEGRAM  = bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID)
