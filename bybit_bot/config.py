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
MIN_VOLUME_USDT_24H  = 1_000_000
SCAN_INTERVAL_SEC    = 3600

# --- Multi-Timeframe Analysis -------------------------------------------------
# 5m  x 1000 = 3.5 ngay — bat scalp signal ngan han
# 15m x 1000 = 10.4 ngay — signal chinh
# 1h  x 500  = 20 ngay — xu huong
# 4h  x 300  = 50 ngay — xu huong lon
TIMEFRAMES = {
    "scalp":  "5",     # 5m: scalp / signal ngan han
    "signal": "15",    # 15m: signal chinh
    "trend":  "60",    # 1h: xu huong
    "macro":  "240",   # 4h: xu huong lon
}
CANDLE_LIMIT_SCALP  = 1000
CANDLE_LIMIT_SIGNAL = 1000
CANDLE_LIMIT_TREND  = 500
CANDLE_LIMIT_MACRO  = 300

# --- Strategy Selector --------------------------------------------------------
STRATEGY_EVAL_CANDLES = 200
MIN_WIN_RATE          = 0.40
STRATEGY_RESCAN_BARS  = 30

# --- Phi giao dich Bybit ------------------------------------------------------
TAKER_FEE      = 0.00055
MAKER_FEE      = 0.00020
ROUND_TRIP_FEE = TAKER_FEE * 2   # 0.11% tong phi ca 2 chieu

# --- Risk Management ----------------------------------------------------------
CAPITAL_PER_TRADE_PCT = 0.10    # khong dung, giu lai de khong loi import cu

USE_MAX_LEVERAGE       = True
MAX_LEVERAGE           = 100
DEFAULT_LEVERAGE       = 20
MAX_OPEN_POSITIONS     = 9999
MAX_POSITIONS_PER_SIDE = 9999

# SL/TP theo ATR — dam bao RR >= 1 sau phi
# SL = 1.5x ATR, TP1 = 1.5x ATR (RR~1), TP2 = 3x ATR (RR~2)
ATR_PERIOD        = 14
SL_ATR_MULT       = 1.5    # SL  = 1.5x ATR
TP1_ATR_MULT      = 1.5    # TP1 = 1.5x ATR → RR ~1 sau phi
TP2_ATR_MULT      = 3.0    # TP2 = 3.0x ATR → RR ~2
TRAILING_STOP_ATR = 0.5    # Trailing stop = 0.5x ATR

# --- Signal sensitivity -------------------------------------------------------
MIN_SIGNAL_STRENGTH = 0.60   # Chi lay signal manh
MIN_CONSENSUS       = 2      # Can it nhat 2 strategies dong thuan cung chieu
MIN_ATR_PCT         = 0.003  # ATR phai >= 0.3% gia de bu phi
QTY_SCALE_CAP       = 5      # Nhan toi da 5x min_qty

# --- Execution ----------------------------------------------------------------
ORDER_TYPE        = "Market"
LOOP_INTERVAL_SEC = 60
RETRY_ATTEMPTS    = 3
RETRY_DELAY_SEC   = 2

# --- Logging ------------------------------------------------------------------
LOG_FILE        = "trading_bot.log"
LOG_TRADES_FILE = "trades.json"
LOG_LEVEL       = "INFO"

# --- Telegram (tuy chon) ------------------------------------------------------
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
ENABLE_TELEGRAM  = bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID)
