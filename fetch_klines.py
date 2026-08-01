#!/usr/bin/env python3
"""
fetch_klines.py — Tải nến 1m Bybit Futures Linear về máy local.

Ví dụ sử dụng:
  python fetch_klines.py --date 2026-07-01 --all
  python fetch_klines.py --date 2026-07-01 --coins BTCUSDT ETHUSDT SOLUSDT
  python fetch_klines.py --month 2026-07 --all
  python fetch_klines.py --month 2026-07 --coins BTCUSDT ETHUSDT
  python fetch_klines.py --from 2026-07-01 --to 2026-07-15 --all
  python fetch_klines.py --date 2026-07-01 --coins BTCUSDT --out C:/MyData

Kết quả: D:\\BYBIT_MARKET\\BTCUSDT_1m_20260701.csv  (mỗi ngày 1 file)
"""

import argparse
import os
import time
import csv
from datetime import datetime, timezone, timedelta
from calendar import monthrange

import requests

DEFAULT_OUT_DIR      = r"D:\BYBIT_MARKET"
BYBIT_BASE           = "https://api.bybit.com"
KLINE_ENDPOINT       = "/v5/market/kline"
INSTRUMENTS_ENDPOINT = "/v5/market/instruments-info"
MAX_LIMIT  = 1000
RATE_DELAY = 0.12


def get_all_usdt_perp_symbols():
    symbols = []
    cursor  = ""
    print("Đang lấy danh sách symbol từ Bybit...")
    while True:
        params = {"category": "linear", "status": "Trading", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        r = requests.get(BYBIT_BASE + INSTRUMENTS_ENDPOINT, params=params, timeout=10)
        r.raise_for_status()
        data  = r.json()
        items = data.get("result", {}).get("list", [])
        for item in items:
            sym = item.get("symbol", "")
            ct  = item.get("contractType", "")
            if sym.endswith("USDT") and ct == "LinearPerpetual":
                symbols.append(sym)
        cursor = data.get("result", {}).get("nextPageCursor", "")
        if not cursor or not items:
            break
    print("  -> {} symbol".format(len(symbols)))
    return sorted(symbols)


def fetch_klines(symbol, start_ms, end_ms):
    all_rows = {}
    cur = start_ms
    while cur < end_ms:
        params = {
            "category": "linear",
            "symbol":   symbol,
            "interval": "1",
            "start":    cur,
            "end":      end_ms,
            "limit":    MAX_LIMIT,
        }
        try:
            r    = requests.get(BYBIT_BASE + KLINE_ENDPOINT, params=params, timeout=15)
            r.raise_for_status()
            rows = r.json().get("result", {}).get("list", [])
        except Exception as e:
            print("\n    [!] {}: lỗi request: {}".format(symbol, e))
            time.sleep(2)
            break

        if not rows:
            break
        rows = list(reversed(rows))
        for row in rows:
            ts = int(row[0])
            if start_ms <= ts < end_ms:
                all_rows[ts] = {
                    "timestamp_ms": ts,
                    "open":   row[1],
                    "high":   row[2],
                    "low":    row[3],
                    "close":  row[4],
                    "volume": row[5],
                }
        last_ts = int(rows[-1][0])
        if last_ts <= cur:
            break
        cur = last_ts + 60000
        time.sleep(RATE_DELAY)

    return sorted(all_rows.values(), key=lambda x: x["timestamp_ms"])


def save_csv(symbol, rows, date, out_dir, filename_override=None):
    os.makedirs(out_dir, exist_ok=True)
    if filename_override:
        fname = os.path.join(out_dir, filename_override)
    else:
        fname = os.path.join(out_dir, "{}_1m_{}.csv".format(symbol, date.strftime("%Y%m%d")))
    with open(fname, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp_ms", "open", "high", "low", "close", "volume"])
        w.writeheader()
        w.writerows(rows)
    return fname


def date_range(start, end):
    cur = start
    while cur < end:
        yield cur
        cur += timedelta(days=1)


def process(symbols, start_dt, end_dt, out_dir):
    days    = list(date_range(start_dt, end_dt))
    total_s = len(symbols)
    total_d = len(days)
    print("\nSẽ tải: {} coin x {} ngày = {} file".format(total_s, total_d, total_s * total_d))
    print("Lưu vào: {}\n".format(out_dir))

    success = 0
    failed  = []

    for i, sym in enumerate(symbols, 1):
        for day in days:
            day_start_ms = int(day.replace(tzinfo=timezone.utc).timestamp() * 1000)
            day_end_ms   = day_start_ms + 86400000

            print("[{}/{}] {} {} ...".format(i, total_s, sym, day.strftime("%Y-%m-%d")), end=" ", flush=True)

            rows = fetch_klines(sym, day_start_ms, day_end_ms)
            if rows:
                path = save_csv(sym, rows, day, out_dir)
                print("{} nến -> {}".format(len(rows), path))
                success += 1
            else:
                print("không có data, bỏ qua")
                failed.append("{} {}".format(sym, day.strftime("%Y-%m-%d")))

    print("\nHoàn tất: {}/{} file".format(success, total_s * total_d))
    if failed:
        print("Thất bại ({}):" .format(len(failed)))
        for item in failed:
            print("  - {}".format(item))


def main():
    parser = argparse.ArgumentParser(description="Tải nến 1m Bybit Futures về local")

    time_group = parser.add_mutually_exclusive_group(required=True)
    time_group.add_argument("--date",  metavar="YYYY-MM-DD", help="Tải 1 ngày. Vd: --date 2026-07-01")
    time_group.add_argument("--month", metavar="YYYY-MM",    help="Tải cả tháng. Vd: --month 2026-07")
    time_group.add_argument("--from",  dest="date_from", metavar="YYYY-MM-DD",
                            help="Tải từ ngày (dùng kèm --to). Vd: --from 2026-07-01 --to 2026-07-15")

    parser.add_argument("--to", dest="date_to", metavar="YYYY-MM-DD",
                        help="Tải đến ngày (dùng kèm --from, không bao gồm ngày này).")

    coin_group = parser.add_mutually_exclusive_group(required=True)
    coin_group.add_argument("--all",   action="store_true", help="Tải tất cả coin USDT Perpetual")
    coin_group.add_argument("--coins", nargs="+", metavar="SYMBOL",
                            help="Coin cụ thể. Vd: --coins BTCUSDT ETHUSDT SOLUSDT")

    parser.add_argument("--out", default=DEFAULT_OUT_DIR,
                        help="Thư mục lưu file (mặc định: {})".format(DEFAULT_OUT_DIR))

    args = parser.parse_args()
    out_dir = args.out

    # Tên file = y chang lệnh đã gõ (bỏ "python fetch_klines.py")
    import sys
    _cmd_args = " ".join(sys.argv[1:])
    _file_label = _cmd_args.replace("/", "-").replace("\\", "-")  # tránh ký tự không hợp lệ trên Windows

    # Xác định khoảng thời gian
    if args.date:
        start_dt = datetime.strptime(args.date, "%Y-%m-%d")
        end_dt   = start_dt + timedelta(days=1)

    elif args.month:
        y, m     = map(int, args.month.split("-"))
        start_dt = datetime(y, m, 1)
        end_dt   = datetime(y, m, monthrange(y, m)[1]) + timedelta(days=1)

    elif args.date_from:
        if not args.date_to:
            parser.error("--from cần kèm --to")
        start_dt = datetime.strptime(args.date_from, "%Y-%m-%d")
        end_dt   = datetime.strptime(args.date_to,   "%Y-%m-%d")
        if end_dt <= start_dt:
            parser.error("--to phải sau --from")
    else:
        parser.error("Phải chỉ định --date, --month, hoặc --from/--to")

    last_day_str = (end_dt - timedelta(days=1)).strftime("%Y-%m-%d")
    print("Khoảng thời gian: {} -> {} (UTC)".format(start_dt.strftime("%Y-%m-%d"), last_day_str))

    symbols = get_all_usdt_perp_symbols() if args.all else [s.upper() for s in args.coins]

    process(symbols, start_dt, end_dt, out_dir)


if __name__ == "__main__":
    main()
