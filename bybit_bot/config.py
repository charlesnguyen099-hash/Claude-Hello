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

# --- Multi-Timeframe Analysis -------------------------------------------------
# 1m  x 500  = ~8h   — MAIN signal + entry timing
# 5m  x 200  = ~17h  — scalp alignment (tranh vao giua bounce)
# 5m  x 200  = ~17h  — trend confirmation (reuse df_scalp khi cung TF)
# 15m x 150  = ~37h  — macro trend (was 1h — qua cham, bo lo move 20-30 phut)
#
# Doi tu 15m/1h sang 5m/15m: bat lenh trong 15-30 phut thay vi can 1-3 gio cho EMA 1h flip
# ATR cho swing SL tinh tu 5m trend TF (on dinh hon 1m, nhanh hon 15m)
TIMEFRAMES = {
    "signal": "1",   # 1m duy nhat — 2000 nen 1m chua day du thong tin cua 15m/1h nhung chi tiet hon
    "scalp":  "1",   # reuse df_signal
    "trend":  "1",   # reuse df_signal (EMA dai hon de tinh medium trend)
    "macro":  "1",   # reuse df_signal (EMA rat dai de tinh macro trend)
}
# 2000 x 1m = ~33h — tat ca timeframe logic tinh tu cung bo data nay
# EMA(100/250) tren 1m = tuong duong EMA(20/50) tren 5m
# EMA(300/600) tren 1m = tuong duong EMA(20/40) tren 15m
# EMA(500/1000) tren 1m = tuong duong EMA(8/17) tren 1h — nhung chinh xac hon vi du lieu 1m
CANDLE_LIMIT_SIGNAL = 2000
CANDLE_LIMIT_SCALP  = 2000   # reuse
CANDLE_LIMIT_TREND  = 2000   # reuse
CANDLE_LIMIT_MACRO  = 2000   # reuse

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
# So lenh mo cung luc: khong gioi han cung, phu thuoc do tiem nang thi truong va equity con lai
# ATR period (dung cho compute_atr trong signal analysis, KHONG dung cho SL/TP sizing)
# SL/TP sizing hien tai dung ROI-based (xem TP_ROI_MIN/MAX, SL_TP_RATIO ben duoi)
ATR_PERIOD        = 14
# SL_ATR_MULT / TP1_ATR_MULT / TP2_ATR_MULT: KHONG DUNG — da thay bang ROI-based sizing
# Partial close va breakeven trail da bi xoa — 1 TP duy nhat, hit la dong toan bo

# Hard cap SL/TP theo % gia — tranh SL/TP phi ly khi ATR qua lon (coin pump/dip)
# SL toi da 4%: du rong de vuot qua spike tam thoi ma gia van co the quay dau
# TP1 toi da 6%, TP2 toi da 10%: TP phai co the dat duoc trong dieu kien binh thuong
# SL/TP TINH THEO ROI% (% tren margin = loi/lo / von bo vao)
# ROI = (price_dist / entry) * leverage
# TP ROI: scale theo potential [12%, 50%] — lenh manh TP cao hon
# SL ROI = SL_TP_RATIO x TP ROI (hien tai 5x)
#   -> SL range [60%, 250%] ROI — toi thieu -60% ROI
TP_ROI_MIN  = 0.12   # TP toi thieu 12% ROI (khi lenh yeu) -> SL toi thieu 60% ROI (5x)
TP_ROI_MAX  = 0.50   # TP toi da 50% ROI (khi lenh manh)
SL_TP_RATIO = 5.0    # SL luon gap 5 lan TP (SL ROI = 5 x TP ROI)

# --- Signal sensitivity -------------------------------------------------------
MIN_SIGNAL_STRENGTH = 0.60   # Chi lay signal chat luong cao
MIN_ADX             = 12     # ADX >= 12 cho 1m scalp — 18 qua cao, block het trong sideway/Asian session
MIN_CONSENSUS          = 3   # 3/7 strategies dong thuan — 2 qua thap, de bi noise
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
LOOP_INTERVAL_SEC      = 1    # minimum pause giua cac tick (rate limit only)
SYMBOL_COOLDOWN_SEC    = 10   # cooldown cho coin thuong (giam tu 30s -> 10s)
TOP20_COOLDOWN_SEC     = 5    # top 20 trending: re-analyze moi 5s (gan nhu moi tick)
TOP20_COUNT            = 20   # so coin top duoc uu tien cao
SCAN_BUDGET_TOP20_SEC  = 120.0 # budget rieng cho top 20 (tang de khong bi cat som khi nhieu trade)
SCAN_BUDGET_REST_SEC   = 120.0 # budget cho phan con lai (tang de scan duoc nhieu coin hon)

# --- Logging ------------------------------------------------------------------
LOG_FILE        = "trading_bot.log"
LOG_TRADES_FILE = "trades.json"
LOG_LEVEL       = "INFO"

# --- Telegram (tuy chon) ------------------------------------------------------
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
ENABLE_TELEGRAM  = bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID)
