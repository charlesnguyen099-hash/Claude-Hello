# Bybit Futures Auto Trading Bot

Bot giao dịch tự động 24/7 trên Bybit Linear Futures với 7 chiến lược + BREAKOUT riêng biệt, multi-timeframe analysis, quản lý vốn động theo ATR.

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
# Testnet (khuyến nghị chạy thử ít nhất 3-7 ngày trước)
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
| `client.py` | Bybit API wrapper (pybit v5) |
| `scanner.py` | Scan top N symbols theo volume + top 20 trending |
| `strategies/` | 7 chiến lược + BREAKOUT (xem bảng bên dưới) |
| `risk_manager.py` | Sizing lệnh, SL/TP dynamic theo ATR, consensus scale |
| `executor.py` | Thực thi lệnh, quản lý vị thế 3 cấp (breakeven + partial close + TP2) |
| `bot_logger.py` | Log nhẹ (JSON append), Telegram notification |
| `main.py` | Entry point, vòng lặp 15s, bộ lọc đa tầng |

## Chiến lược

| # | Tên | Kiểu | Ghi chú |
|---|-----|------|---------|
| 1 | EMA Crossover | Trend following | EMA9/21/50 cross + volume xác nhận |
| 2 | RSI + MACD | Momentum | RSI 35-55 + MACD histogram tăng/giảm 2 nến |
| 3 | Bollinger Bands | Mean reversion | Chạm band + RSI extreme + nến đảo chiều |
| 4 | Supertrend + ADX | Trend | Supertrend flip + ADX > 25 |
| 5 | VWAP + Volume | Breakout | Breakout VWAP + volume spike |
| 6 | Ichimoku Cloud | Comprehensive | TK cross + above/below cloud + chikou |
| 7 | Sustained Trend | Trend + Reversal | EMA stack dốc đều 20 nến + reversal RSI |
| B | Breakout | Momentum spike | Volume > 3x MA10 + phá vỡ high/low 20 nến |

## Tham số quan trọng (config.py)

```python
# Risk
SL_ATR_MULT       = 1.5    # Stop loss = 1.5× ATR
TP1_ATR_MULT      = 1.5    # Take profit 1 = 1.5× ATR (safety net trên sàn)
TP2_ATR_MULT      = 3.0    # Take profit 2 = 3.0× ATR (target sau partial close)
BREAKEVEN_TRIGGER = 0.30   # Chuyển SL về entry khi đi được 30% đến TP1
PARTIAL_CLOSE_TRIGGER = 0.75  # Đóng 50% vị thế tại 75% TP1, cập nhật TP lên TP2
MAX_LEVERAGE      = 100    # Dùng leverage tối đa Bybit cho phép

# Signal
MIN_CONSENSUS          = 4   # Priority (top10): cần ít nhất 4/7 strategies
MIN_CONSENSUS_TRENDING = 5   # Trending non-priority: cần ít nhất 5/7 strategies
MIN_SIGNAL_STRENGTH    = 0.55
MIN_ADX                = 15  # ADX < 15 = sideways, không trade
TRADE_SIZE_MULT        = 4   # qty = min_qty × 4 × consensus_scale

# Timeframes
# 1m (300 nến), 5m (1000), 15m (1000), 1h (500), 4h (300)
LOOP_INTERVAL_SEC = 15  # Check mỗi 15 giây
SCAN_INTERVAL_SEC = 3600  # Refresh symbol list mỗi 1 giờ
```

## Cơ chế quản lý vị thế (3 cấp)

1. **Level 1 — Break-even (30% TP1)**: Dời SL về entry + phí → bảo vệ vốn sớm
2. **Level 2 — Partial Close (75% TP1)**: Đóng 50% vị thế, cập nhật TP từ TP1 → TP2 trên sàn, xác nhận SL = break-even
3. **Level 3 — TP2**: 50% còn lại chạy đến TP2 (3× ATR) với zero downside risk

## Lưu ý quan trọng

- **TESTNET TRƯỚC**: Chạy testnet tối thiểu 1 tuần để xác nhận bot hoạt động đúng
- Bot **không lưu dữ liệu nến local** — lấy trực tiếp từ Bybit API mỗi lần cần
- Lịch sử lệnh lưu trong `trades.json` (append, rất nhẹ)
- Bybit giữ lịch sử giao dịch — dùng dashboard Bybit để phân tích thêm
