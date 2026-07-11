"""
Bybit Futures Auto Trading Bot - Configuration
"""

import os

# ─── API Credentials ──────────────────────────────────────────────────────────
API_KEY    = os.getenv("BYBIT_API_KEY", "YOUR_API_KEY_HERE")
API_SECRET = os.getenv("BYBIT_API_SECRET", "YOUR_API_SECRET_HERE")
TESTNET    = os.getenv("BYBIT_TESTNET", "false").lower() == "true"

# ─── Market Scanner ───────────────────────────────────────────────────────────
TOP_N_SYMBOLS        = 200         # Top 200 cặp giao dịch
MIN_VOLUME_USDT_24H  = 5_000_000   # Lọc cặp có volume 24h >= 5M USDT (hạ ngưỡng cho top 200)
SCAN_INTERVAL_SEC    = 3600        # Quét lại top symbols mỗi 1 giờ

# ─── Multi-Timeframe Analysis ─────────────────────────────────────────────────
# 15m: tín hiệu entry chính — đủ nhanh bắt trend sớm, đủ ít nhiễu
# 1h : xác nhận xu hướng
# 4h : xu hướng lớn
TIMEFRAMES = {
    "signal":  "15",   # phút
    "trend":   "60",
    "macro":   "240",
}
CANDLE_LIMIT = 200  # Số nến lấy về từ Bybit API (không lưu local)

# ─── Strategy Selector ────────────────────────────────────────────────────────
STRATEGY_EVAL_CANDLES  = 100   # Số nến dùng để backtest chọn strategy
MIN_WIN_RATE           = 0.45  # Win rate tối thiểu để chọn strategy
STRATEGY_RESCAN_BARS   = 20    # Chạy lại selector sau N nến

# ─── Risk Management ──────────────────────────────────────────────────────────
ACCOUNT_RISK_PCT       = 0.02   # Rủi ro tối đa mỗi lệnh: 2% vốn
USE_MAX_LEVERAGE       = True   # Tự động dùng leverage tối đa của từng cặp trên Bybit
MAX_LEVERAGE           = 100    # Cap trên (Bybit cho phép tối đa 100x một số cặp)
DEFAULT_LEVERAGE       = 20     # Dùng khi không lấy được max leverage từ API
MAX_OPEN_POSITIONS     = 10     # Tối đa 10 vị thế đồng thời
MAX_POSITIONS_PER_SIDE = 5      # Tối đa 5 long hoặc 5 short

# Stop Loss / Take Profit theo ATR
ATR_PERIOD         = 14
SL_ATR_MULTIPLIER  = 1.5   # SL = 1.5 × ATR
TP1_ATR_MULTIPLIER = 2.0   # TP1 = 2.0 × ATR (đóng 50%)
TP2_ATR_MULTIPLIER = 3.5   # TP2 = 3.5 × ATR (đóng 50% còn lại)
TRAILING_STOP_ATR  = 1.0   # Trailing stop = 1.0 × ATR

# ─── Execution ────────────────────────────────────────────────────────────────
ORDER_TYPE          = "Market"   # Market order để đảm bảo khớp lệnh
LOOP_INTERVAL_SEC   = 60         # Vòng lặp chính mỗi 60 giây
RETRY_ATTEMPTS      = 3
RETRY_DELAY_SEC     = 2

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_FILE        = "trading_bot.log"
LOG_TRADES_FILE = "trades.json"   # Lưu lịch sử giao dịch nhẹ (JSON append)
LOG_LEVEL       = "INFO"

# ─── Telegram Notifications (tuỳ chọn) ───────────────────────────────────────
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
ENABLE_TELEGRAM  = bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID)
