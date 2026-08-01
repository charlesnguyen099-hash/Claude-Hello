#!/usr/bin/env python3
"""
fetch_klines.py — Tải nến 1m Bybit Futures Linear về máy local.

Ví dụ sử dụng:
  # Tải 1 ngày cụ thể, tất cả coin
  python fetch_klines.py --date 2026-07-01 --all

  # Tải 1 ngày cụ thể, một số coin
  python fetch_klines.py --date 2026-07-01 --coins BTCUSDT ETHUSDT SOLUSDT

  # Tải cả tháng 07/2026, tất cả coin
  python fetch_klines.py --month 2026-07 --all

  # Tải cả tháng 07/2026, coin cụ thể
  python fetch_klines.py --month 2026-07 --coins BTCUSDT ETHUSDT

  # Tải từ ngày đến ngày
  python fetch_klines.py --from 2026-07-01 --to 2026-07-15 --all

Kết quả lưu vào: D:\\BYBIT_MARKET\\<SYMBOL>_1m_<YYYYMMDD>.csv
  Mỗi ngày = 1 file riêng, ví dụ BTCUSDT_1m_20260701.csv
"""

import argparse
import os
import time
import csv
from datetime import datetime, timezone, timedelta
from calendar import monthrange

import requests

OUT_DIR   = r"D:\BYBIT_MARKET"
BYBIT_BASE          = "https://api.bybit.com"
KLINE_ENDPOINT      = "/v5/market/kline"
INSTRUMENTS_ENDPOINT = "/v5/market/instruments-info"
MAX_LIMIT  = 1000    # Bybit tối đa 1000 nến/request
RATE_DELAY = 0.12    # giây giữa các request (tránh rate limit)


# ---------------------------------------------------------------------------

def get_all_usdt_perp_symbols() -> list[str]:
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
    print(f"  -> {len(symbols)} symbol")
    return sorted(symbols)


def fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list[dict]:
    """Tải nến 1m từ start_ms đến end_ms, tự phân trang."""
    all_rows: dict[int, dict] = {}
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
            print(f"\n    [!] {symbol}: lỗi request: {e}")
            time.sleep(2)
            break

        if not rows:
            break
        rows = list(reversed(rows))          # Bybit trả về mới -> cũ, đảo lại
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
        cur = last_ts + 60_000               # bước sang nến kế tiếp
        time.sleep(RATE_DELAY)

    return sorted(all_rows.values(), key=lambda x: x["timestamp_ms"])


def save_csv(symbol: str, rows: list[dict], date: datetime) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    fname = os.path.join(OUT_DIR, f"{symbol}_1m_{date.strftime('%Y%m%d')}.csv")
    with open(fname, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp_ms","open","high","low","close","volume"])
        w.writeheader()
        w.writerows(rows)
    return fname


def date_range(start: datetime, end: datetime):
    """Yield từng ngày từ start đến end (không bao gồm end)."""
    cur = start
    while cur < end:
        yield cur
        cur += timedelta(days=1)


def process(symbols: list[str], start_dt: datetime, end_dt: datetime):
    """Tải data cho danh sách symbol trong khoảng [start_dt, end_dt)."""
    days    = list(date_range(start_dt, end_dt))
    total_s = len(symbols)
    total_d = len(days)
    print(f"\nSẽ tải: {total_s} coin x {total_d} ngày = {total_s * total_d} file")
    print(f"Lưu vào: {OUT_DIR}\n")

    success = 0
    failed  = []

    for i, sym in enumerate(symbols, 1):
        for day in days:
            day_start_ms = int(day.replace(tzinfo=timezone.utc).timestamp() * 1000)
            day_end_ms   = day_start_ms + 86_400_000   # +24h

            label = f"[{i}/{total_s}] {sym} {day.strftime('%Y-%m-%d')}"
            print(f"{label} ...", end=" ", flush=True)

            rows = fetch_klines(sym, day_start_ms, day_end_ms)
            if rows:
                path = save_csv(sym, rows, day)
                print(f"{len(rows)} nến -> {path}")
                success += 1
            else:
                print("không có data, bỏ qua")
                failed.append(f"{sym} {day.strftime('%Y-%m-%d')}")

    print(f"\nHoàn tất: {success}/{total_s * total_d} file")
    if failed:
        print(f"Thất bại ({len(failed)}):")
        for f in failed:
            print(f"  - {f}")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Tải nến 1m Bybit Futures về D:\\BYBIT_MARKET",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # -- Chọn khoảng thời gian --
    time_group = parser.add_mutually_exclusive_group(required=True)
    time_group.add_argument(
        "--date", metavar="YYYY-MM-DD",
        help="Tải 1 ngày cụ thể. Ví dụ: --date 2026-07-01"
    )
    time_group.add_argument(
        "--month", metavar="YYYY-MM",
        help="Tải cả 1 tháng. Ví dụ: --month 2026-07"
    )
    time_group.add_argument(
        "--from", dest="date_from", metavar="YYYY-MM-DD",
        help="Tải từ ngày (dùng kèm --to). Ví dụ: --from 2026-07-01 --to 2026-07-15"
    )

    parser.add_argument(
        "--to", dest="date_to", metavar="YYYY-MM-DD",
        help="Tải đến ngày (dùng kèm --from, không bao gồm ngày này)."
    )

    # -- Chọn coin --
    coin_group = parser.add_mutually_exclusive_group(required=True)
    coin_group.add_argument("--all",   action="store_true", help="Tải tất cả coin USDT Perpetual")
    coin_group.add_argument("--coins", nargs="+", metavar="SYMBOL",
                            help="Coin cụ thể. Ví dụ: --coins BTCUSDT ETHUSDT SOLUSDT")

    # -- Tuỳ chọn --
    parser.add_argument("--out", default=None,
                        help=f"Thư mục lưu (mặc định: {OUT_DIR})")

    args = parser.parse_args()

    global OUT_DIR
    if args.out:
        OUT_DIR = args.out

    # -- Xác định khoảng thời gian --
    if args.date:
        start_dt = datetime.strptime(args.date, "%Y-%m-%d")
        end_dt   = start_dt + timedelta(days=1)

    elif args.month:
        y, m     = map(int, args.month.split("-"))
        start_dt = datetime(y, m, 1)
        last_day = monthrange(y, m)[1]
        end_dt   = datetime(y, m, last_day) + timedelta(days=1)

    elif args.date_from:
        if not args.date_to:
            parser.error("--from cần kèm --to")
        start_dt = datetime.strptime(args.date_from, "%Y-%m-%d")
        end_dt   = datetime.strptime(args.date_to,   "%Y-%m-%d")
        if end_dt <= start_dt:
            parser.error("--to phải sau --from")
    else:
        parser.error("Phải chỉ định --date, --month, hoặc --from/--to")

    print(f"Khoảng thời gian: {start_dt.strftime('%Y-%m-%d')} -> {(end_dt - timedelta(days=1)).strftime('%Y-%m-%d')} (UTC)")

    # -- Xác định danh sách coin --
    symbols = get_all_usdt_perp_symbols() if args.all else [s.upper() for s in args.coins]

    process(symbols, start_dt, end_dt)


if __name__ == "__main__":
    main()
