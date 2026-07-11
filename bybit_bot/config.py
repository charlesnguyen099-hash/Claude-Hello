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

# ─── Phí giao dịch Bybit (tính vào cost) ────────────────────────────────────
TAKER_FEE = 0.00055   # 0.055% mỗi lần vào/ra (market order)
MAKER_FEE = 0.00020   # 0.020% (limit order — không dùng hiện tại)
# Tổng phí 1 vòng lệnh (vào + ra) = 2 × TAKER_FEE = 0.11%
ROUND_TRIP_FEE = TAKER_FEE * 2  # 0.00110 = 0.11%

# ─── Risk Management ──────────────────────────────────────────────────────────
ACCOUNT_RISK_PCT       = 0.30   # Chấp nhận thua tối đa 30% vốn mỗi lệnh (dùng margin)
CAPITAL_PER_TRADE_PCT  = 0.10   # Dùng tối đa 10% vốn thực cho mỗi lệnh (không all-in)
                                 # → với 10 lệnh tối đa = 100% vốn phân bổ đều
USE_MAX_LEVERAGE       = True   # Tự động dùng leverage tối đa của từng cặp trên Bybit
MAX_LEVERAGE           = 100    # Cap trên (Bybit cho phép tối đa 100x một số cặp)
DEFAULT_LEVERAGE       = 20     # Dùng khi không lấy được max leverage từ API
MAX_OPEN_POSITIONS     = 10     # Tối đa 10 vị thế đồng thời
MAX_POSITIONS_PER_SIDE = 5      # Tối đa 5 long hoặc 5 short

# Stop Loss / Take Profit theo ATR + phí
# SL/TP tính SAU KHI đã cộng phí vào — để lệnh thực sự có lãi/lỗ đúng như mong muốn
ATR_PERIOD         = 14
SL_ATR_MULTIPLIER  = 1.5   # SL = 1.5 × ATR + phí (tính net)
TP1_ATR_MULTIPLIER = 2.0   # TP1 = 2.0 × ATR + phí
TP2_ATR_MULTIPLIER = 3.5   # TP2 = 3.5 × ATR + phí
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
