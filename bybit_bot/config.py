"""
Bybit Futures Auto Trading Bot - Configuration
"""

import os

# --- API Credentials ----------------------------------------------------------
API_KEY    = os.getenv("BYBIT_API_KEY", "YOUR_API_KEY_HERE")
API_SECRET = os.getenv("BYBIT_API_SECRET", "YOUR_API_SECRET_HERE")
TESTNET    = os.getenv("BYBIT_TESTNET", "false").lower() == "true"

# --- Market Scanner -----------------------------------------------------------
MIN_VOLUME_USDT_24H  = 100_000       # min 100K USDT/24h - bat ca coin nho co trend dep
SCAN_INTERVAL_SEC    = 30            # cap nhat danh sach trending moi 30s (giam API call)

# --- Multi-Timeframe Analysis -------------------------------------------------
# 1m  x 500  = ~8h   - MAIN signal + entry timing
# 5m  x 200  = ~17h  - scalp alignment (tranh vao giua bounce)
# 5m  x 200  = ~17h  - trend confirmation (reuse df_scalp khi cung TF)
# 15m x 150  = ~37h  - macro trend (was 1h - qua cham, bo lo move 20-30 phut)
#
# Doi tu 15m/1h sang 5m/15m: bat lenh trong 15-30 phut thay vi can 1-3 gio cho EMA 1h flip
# ATR cho swing SL tinh tu 5m trend TF (on dinh hon 1m, nhanh hon 15m)
TIMEFRAMES = {
    "signal": "1",   # 1m duy nhat - 2000 nen 1m chua day du thong tin cua 15m/1h nhung chi tiet hon
    "scalp":  "1",   # reuse df_signal
    "trend":  "1",   # reuse df_signal (EMA dai hon de tinh medium trend)
    "macro":  "1",   # reuse df_signal (EMA rat dai de tinh macro trend)
}
# 2000 x 1m = ~33h - tat ca timeframe logic tinh tu cung bo data nay
# EMA(100/250) tren 1m = tuong duong EMA(20/50) tren 5m
# EMA(300/600) tren 1m = tuong duong EMA(20/40) tren 15m
# EMA(500/1000) tren 1m = tuong duong EMA(8/17) tren 1h - nhung chinh xac hon vi du lieu 1m
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
# VON MOI LENH THEO TIEM NANG (potential-scaled capital) - TREN EQUITY THAT:
#   potential = f(consensus, strength) trong [0,1]
#   capital = equity_that * (CAPITAL_PCT_MIN + potential * (CAPITAL_PCT_MAX - CAPITAL_PCT_MIN))
#   -> lenh yeu: 4% equity (rui ro nho), lenh manh nhat: 15% equity (von lon hon, khong all-in)
# KHONG all-in 1 lenh: moi lenh toi da MAX_CAPITAL_PCT equity + phai chua margin cho lenh khac
# VON LON HON - vi gio chi trade lenh CHAT LUONG CAO (high-conviction), it lenh hon:
# 'tha it ma chat con hon nhieu ma lo' -> moi lenh dung von lon de an dam.
CAPITAL_PCT_MIN = 0.10   # 10% equity - lenh conviction thap nhat (van du dieu kien chat)
CAPITAL_PCT_MAX = 0.30   # 30% equity - lenh trend cuc ro, xac nhan manh
MAX_CAPITAL_PCT = 0.30   # HARD CAP: 1 lenh KHONG vuot 30% equity that
# Chua margin cho lenh khac: 1 lenh dung toi da 60% margin CON TRONG (free)
# -> van con cho cho 1-2 lenh chat luong tiep theo, khong all-in
MAX_FREE_MARGIN_FRAC = 0.60
# San tinh size CHI de dam bao min-notional Bybit ($5), KHONG dung de phong % equity
EQUITY_FLOOR    = 0.0    # bo hieu ung bom phong % tren tai khoan nho (nguyen nhan all-in)
# RISK_PER_TRADE_PCT: KHONG DUNG
RISK_PER_TRADE_PCT = 0.01
# So lenh mo cung luc: khong gioi han cung, phu thuoc do tiem nang thi truong va equity con lai
# ATR period (dung cho compute_atr trong signal analysis, KHONG dung cho SL/TP sizing)
# SL/TP sizing hien tai dung ROI-based (xem TP_ROI_MIN/MAX, SL_TP_RATIO ben duoi)
ATR_PERIOD        = 14
# SL_ATR_MULT / TP1_ATR_MULT / TP2_ATR_MULT: KHONG DUNG - da thay bang ROI-based sizing
# Partial close va breakeven trail da bi xoa - 1 TP duy nhat, hit la dong toan bo

# Hard cap SL/TP theo % gia - tranh SL/TP phi ly khi ATR qua lon (coin pump/dip)
# SL toi da 4%: du rong de vuot qua spike tam thoi ma gia van co the quay dau
# TP1 toi da 6%, TP2 toi da 10%: TP phai co the dat duoc trong dieu kien binh thuong
# SL/TP TINH THEO ROI% (% tren margin = loi/lo / von bo vao)
# ROI = (price_dist / entry) * leverage
# TP ROI: scale theo potential [10%, 22%] - CHOT LOI AN TOAN, gan, de dat.
# Triet ly (user): "th" lenh nay co 1 range an toan cho TP -> lay muc AN TOAN NHAT,
# tha loi it con hon bi lo. TP nho = gia chi can di 1 chut = HIT nhanh, chot loi truoc
# khi dao chieu. Vi du 50x: TP 20% ROI = gia di 0.4% = rat de dat trong 1 lenh dung trend.
# TP cao (60% cu) can gia di 1.2% -> thuong dao chieu truoc khi toi -> mat lenh loi.
# SL ROI = min(SL_TP_RATIO x TP, tran an toan thanh ly) - SL khong vuot gia thanh ly.
TP_ROI_MIN  = 0.12   # TP toi thieu 12% ROI (lenh yeu) - chot nhanh, an toan
TP_ROI_MAX  = 0.25   # TP toi da 25% ROI (lenh manh nhat) - van de dat truoc khi dao chieu
SL_TP_RATIO = 5.0    # SL muc tieu = 5 x TP (truoc khi clamp thanh ly)

# --- Signal sensitivity -------------------------------------------------------
MIN_SIGNAL_STRENGTH = 0.50   # Giam tu 0.55: bat them lenh - signal yeu gio duoc size nho (5% equity)
                             # nen rui ro da duoc kiem soat bang capital scaling, khong can chan som
MIN_ADX             = 12     # ADX >= 12 cho 1m scalp - 18 qua cao, block het trong sideway/Asian session
MIN_CONSENSUS          = 2   # 2/7 strategies dong thuan - du voi 10+ AEQ gate downstream
MIN_CONSENSUS_TRENDING = 2   # Dong bo voi MIN_CONSENSUS
MIN_ATR_PCT         = 0.0005 # 0.05% cho 1m (ATR 1m nho hon 15m, largecap BTC ~0.03-0.08%)
TRADE_SIZE_MULT         = 1    # Khong dung nua - risk-based sizing thay the

# --- Risk Guards --------------------------------------------------------------
# Daily max loss: neu tong PnL trong ngay < -(equity * MAX_DAILY_LOSS_PCT) -> dung mo lenh moi
MAX_DAILY_LOSS_PCT   = 0.05   # 5% equity - dung ngay khi mat > 5% trong 1 ngay

# Max spread: neu bid-ask spread > nguong nay -> khong entry (thanh khoan kem)
# Largecap (BTC/ETH): 0.05%, Altcoin: 0.15%
MAX_SPREAD_PCT_LARGE = 0.0005   # 0.05% cho BTC/ETH
MAX_SPREAD_PCT_ALT   = 0.0015   # 0.15% cho altcoin

# --- Execution ----------------------------------------------------------------
LOOP_INTERVAL_SEC      = 1    # minimum pause giua cac tick (rate limit only)
SYMBOL_COOLDOWN_SEC    = 10   # cooldown cho coin thuong (giam tu 30s -> 10s)
TOP20_COOLDOWN_SEC     = 5    # top 20 trending: re-analyze moi 5s (gan nhu moi tick)
TOP20_COUNT            = 30   # so coin trending cao nhat duoc uu tien (mo rong 20->30:
                              # tap trung nhom trending lon - nhieu lenh tiem nang nhat)
SCAN_BUDGET_TOP20_SEC  = 150.0 # budget cho nhom trending top (tang 120->150: uu tien scan sau)
SCAN_BUDGET_REST_SEC   = 90.0  # budget phan con lai (giam 120->90: don thoi gian cho nhom trending)

# --- Logging ------------------------------------------------------------------
LOG_FILE        = "trading_bot.log"
LOG_TRADES_FILE = "trades.json"
LOG_LEVEL       = "INFO"

# --- Telegram (tuy chon) ------------------------------------------------------
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
ENABLE_TELEGRAM  = bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID)
