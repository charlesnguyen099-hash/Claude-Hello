#!/usr/bin/env python3
"""
fetch_klines.py — Tải nến 1m Bybit Futures Linear về máy local.

Sử dụng:
  # Tất cả coin, 1 ngày hôm nay
  python fetch_klines.py --all --days 1

  # Tất cả coin, 1 tháng
  python fetch_klines.py --all --days 30

  # Một số coin cụ thể, 7 ngày
  python fetch_klines.py --coins BTCUSDT ETHUSDT SOLUSDT --days 7

  # Coin cụ thể, 1 tháng, lưu vào thư mục khác
  python fetch_klines.py --coins BTCUSDT --days 30 --out ./data

Kết quả: file CSV mỗi coin, ví dụ: klines/BTCUSDT_1m_20260727.csv
"""

import argparse
import os
import time
import csv
from datetime import datetime, timezone, timedelta

import requests

BYBIT_BASE = "https://api.bybit.com"
KLINE_ENDPOINT = "/v5/market/kline"
INSTRUMENTS_ENDPOINT = "/v5/market/instruments-info"
MAX_LIMIT = 1000          # Bybit tối đa 1000 nến/request
RATE_DELAY = 0.12         # giây giữa các request (tránh rate limit)


def get_all_usdt_perp_symbols() -> list[str]:
    """Lấy toàn bộ symbol Linear USDT Perpetual đang active."""
    symbols = []
    cursor = ""
    print("Đang lấy danh sách symbol...")
    while True:
        params = {
            "category": "linear",
            "status": "Trading",
            "limit": 1000,
        }
        if cursor:
            params["cursor"] = cursor
        r = requests.get(BYBIT_BASE + INSTRUMENTS_ENDPOINT, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        items = data.get("result", {}).get("list", [])
        for item in items:
            sym = item.get("symbol", "")
            ct  = item.get("contractType", "")
            if sym.endswith("USDT") and ct == "LinearPerpetual":
                symbols.append(sym)
        cursor = data.get("result", {}).get("nextPageCursor", "")
        if not cursor or not items:
            break
    print(f"  -> {len(symbols)} symbol tìm thấy")
    return sorted(symbols)


def fetch_klines_for_symbol(symbol: str, start_ms: int, end_ms: int) -> list[dict]:
    """Tải toàn bộ nến 1m từ start_ms đến end_ms cho một symbol."""
    all_rows = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            "category": "linear",
            "symbol":   symbol,
            "interval": "1",
            "start":    current_start,
            "end":      end_ms,
            "limit":    MAX_LIMIT,
        }
        try:
            r = requests.get(BYBIT_BASE + KLINE_ENDPOINT, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"    [!] {symbol}: lỗi request: {e}")
            time.sleep(1)
            break

        rows = data.get("result", {}).get("list", [])
        if not rows:
            break

        # Bybit trả về mới nhất trước, cần đảo ngược
        rows = list(reversed(rows))
        for row in rows:
            ts_ms = int(row[0])
            if ts_ms < start_ms or ts_ms >= end_ms:
                continue
            all_rows.append({
                "timestamp_ms": ts_ms,
                "open":   row[1],
                "high":   row[2],
                "low":    row[3],
                "close":  row[4],
                "volume": row[5],
            })

        # Nến cuối cùng xác định điểm bắt đầu tiếp theo
        last_ts = int(rows[-1][0])
        if last_ts <= current_start:
            break
        current_start = last_ts + 60_000  # +1 phút
        time.sleep(RATE_DELAY)

    # Xóa duplicate và sắp xếp
    seen = {}
    for row in all_rows:
        seen[row["timestamp_ms"]] = row
    return sorted(seen.values(), key=lambda x: x["timestamp_ms"])


def save_csv(symbol: str, rows: list[dict], out_dir: str, start_dt: datetime):
    """Lưu list nến ra file CSV."""
    os.makedirs(out_dir, exist_ok=True)
    date_str = start_dt.strftime("%Y%m%d")
    filename = os.path.join(out_dir, f"{symbol}_1m_{date_str}.csv")
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp_ms", "open", "high", "low", "close", "volume"])
        writer.writeheader()
        writer.writerows(rows)
    return filename


def main():
    parser = argparse.ArgumentParser(description="Tải nến 1m Bybit Futures về local")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all",   action="store_true", help="Tải tất cả coin USDT Perpetual")
    group.add_argument("--coins", nargs="+", metavar="SYMBOL", help="Tải các coin cụ thể, vd: BTCUSDT ETHUSDT")
    parser.add_argument("--days", type=int, default=1,
                        help="Số ngày cần lấy (1=hôm nay, 30=1 tháng). Mặc định: 1")
    parser.add_argument("--out",  default="klines",
                        help="Thư mục lưu file CSV. Mặc định: ./klines")
    args = parser.parse_args()

    # Tính khoảng thời gian UTC
    now_utc  = datetime.now(timezone.utc)
    end_dt   = now_utc.replace(second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=args.days)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms   = int(end_dt.timestamp() * 1000)

    print(f"Khoảng thời gian: {start_dt.strftime('%Y-%m-%d %H:%M')} -> {end_dt.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"Lưu vào: {os.path.abspath(args.out)}/")
    print()

    symbols = get_all_usdt_perp_symbols() if args.all else [s.upper() for s in args.coins]

    total   = len(symbols)
    success = 0
    failed  = []

    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{total}] {symbol} ...", end=" ", flush=True)
        rows = fetch_klines_for_symbol(symbol, start_ms, end_ms)
        if rows:
            path = save_csv(symbol, rows, args.out, start_dt)
            print(f"{len(rows)} nến -> {path}")
            success += 1
        else:
            print("không có data, bỏ qua")
            failed.append(symbol)

    print()
    print(f"Hoàn tất: {success}/{total} symbol")
    if failed:
        print(f"Thất bại ({len(failed)}): {', '.join(failed)}")


if __name__ == "__main__":
    main()
