"""
Lightweight Logger
- Log giao dịch vào file JSON (append) — chiếm rất ít storage
- Dùng Bybit API để tra cứu lịch sử thay vì lưu local
- Tùy chọn: gửi Telegram notification
"""

import json
import logging
import logging.handlers
import os
import time
from datetime import datetime, timezone
from typing import Optional

import requests

import config


def setup_logging():
    """Cấu hình logging chuẩn cho toàn bộ bot."""
    import sys
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    level = getattr(logging, config.LOG_LEVEL, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # Console — force UTF-8 tren Windows CMD (tranh UnicodeEncodeError voi ky tu dac biet)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File (rotating, tối đa 5MB × 3 files)
    fh = logging.handlers.RotatingFileHandler(
        config.LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)


class BotLogger:
    def __init__(self):
        self._logger = logging.getLogger("trade")

    def log_trade(self, data: dict):
        """Ghi giao dịch vào JSON file (append — cực nhẹ)."""
        data["ts"] = datetime.now(timezone.utc).isoformat()
        line = json.dumps(data, ensure_ascii=False)
        self._logger.info(f"TRADE: {line}")

        try:
            with open(config.LOG_TRADES_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:
            self._logger.warning(f"Could not write trade log: {e}")

        if config.ENABLE_TELEGRAM:
            self._send_telegram(data)

    def _send_telegram(self, data: dict):
        event  = data.get("event", "?").upper()
        symbol = data.get("symbol", "?")
        side   = data.get("side", "?")

        if event == "OPEN":
            msg = (
                f"🟢 *OPEN* {symbol} {side}\n"
                f"Strategy: `{data.get('strategy','?')}`\n"
                f"Entry: `{data.get('entry','?')}`\n"
                f"SL: `{data.get('sl','?')}` | TP1: `{data.get('tp1','?')}`\n"
                f"Notional: `{data.get('notional','?')} USDT`\n"
                f"_{data.get('reason','')}_"
            )
        elif event == "CLOSE":
            msg = f"🔴 *CLOSE* {symbol} {side} | {data.get('reason','')}"
        else:
            msg = f"ℹ️ {json.dumps(data)}"

        try:
            requests.post(
                f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": config.TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"},
                timeout=5,
            )
        except Exception as e:
            self._logger.debug(f"Telegram send failed: {e}")
