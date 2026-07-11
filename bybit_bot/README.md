# Bybit Futures Auto Trading Bot

Bot giao dịch tự động 24/7 trên Bybit Linear Futures với 6 chiến lược, tự động chọn chiến lược tốt nhất theo thời gian thực.

## Cài đặt

```bash
cd bybit_bot
pip install -r requirements.txt
```

## Cấu hình API

```bash
export BYBIT_API_KEY="your_api_key"
export BYBIT_API_SECRET="your_api_secret"
export BYBIT_TESTNET="true"   # Bỏ dòng này khi trade thật
```

**Tùy chọn — Telegram:**
```bash
export TELEGRAM_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"
```

## Chạy bot

```bash
# Test net (khuyến nghị chạy thử ít nhất 3-7 ngày trước)
BYBIT_TESTNET=true python main.py

# Mainnet
python main.py
```

## Chạy 24/7 (Linux server)

```bash
# Dùng screen
screen -S trading_bot
python main.py

# Hoặc systemd service / Docker
```

## Kiến trúc

| File | Chức năng |
|------|-----------|
| `config.py` | Tất cả tham số cấu hình |
| `client.py` | Bybit API wrapper |
| `scanner.py` | Auto chọn top 50 symbols tốt nhất |
| `selector.py` | Backtest nhanh + chọn strategy tốt nhất |
| `strategies/` | 6 chiến lược: EMA, RSI+MACD, Bollinger, Supertrend, VWAP, Ichimoku |
| `risk_manager.py` | Sizing lệnh, SL/TP dynamic theo ATR |
| `executor.py` | Thực thi lệnh, quản lý trailing stop |
| `bot_logger.py` | Log nhẹ (JSON), Telegram notification |
| `main.py` | Entry point, vòng lặp 60s |

## Chiến lược (tự động chọn theo Expectancy)

| # | Tên | Kiểu | Phù hợp khi |
|---|-----|------|------------|
| 1 | EMA Crossover | Trend following | Trending market |
| 2 | RSI + MACD | Momentum | Breakout phase |
| 3 | Bollinger Bands | Mean reversion | Ranging market |
| 4 | Supertrend + ADX | Trend | Strong trend |
| 5 | VWAP + Volume | Breakout | High volume sessions |
| 6 | Ichimoku Cloud | Comprehensive | All conditions |

## Tham số quan trọng (config.py)

```python
ACCOUNT_RISK_PCT   = 0.01   # 1% vốn mỗi lệnh
MAX_OPEN_POSITIONS = 5      # Tối đa 5 vị thế
MAX_LEVERAGE       = 10     # Đòn bẩy tối đa
SL_ATR_MULTIPLIER  = 1.5   # Stop loss = 1.5× ATR
TP1_ATR_MULTIPLIER = 2.0   # Take profit 1 = 2× ATR
TP2_ATR_MULTIPLIER = 3.5   # Take profit 2 = 3.5× ATR
```

## Lưu ý quan trọng

- **TESTNET TRƯỚC**: Chạy testnet tối thiểu 1 tuần để xác nhận bot hoạt động đúng
- Bot **không lưu dữ liệu nến local** — lấy trực tiếp từ Bybit API mỗi lần cần
- Lịch sử lệnh lưu tối thiểu trong `trades.json` (append, rất nhẹ)
- Bybit giữ lịch sử giao dịch — dùng dashboard Bybit để phân tích thêm
