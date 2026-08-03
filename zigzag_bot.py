"""Trade the zigzag legs the PureLogic table describes, then iterate on the losses.

The table is a zigzag: one continuous price path, bottom -> top -> bottom
-> top, each trade starting where the last ended. That is confirmed by the
file itself — 24,891 LONG and 24,892 SHORT (perfect alternation), and
832,928 minutes held against 832,979 available, a 1.00x tiling of
2025-2026 with no overlap.

Trading a zigzag live has one unavoidable property, and everything below
is about measuring it rather than arguing about it: **a pivot is only
knowable after price has turned away from it.** A low is not a low until
price has risen off it. Any real-time zigzag therefore needs a
confirmation threshold T: treat the running extreme as a pivot once price
has retraced T away from it. That costs roughly T at the entry and T at
the exit of every leg, so a leg only pays if

    leg size  >  2T + fees

The file's legs are measured with perfect pivot timing, which is why its
median leg nets +0.20%. This script measures the same legs with causal
pivot detection and reports the difference.

Then it does what was asked: run it, take the losing trades, work out
whether each one lost because the logic was applied badly or because the
logic itself is wrong, fix it, and run again. Every iteration is kept and
reported, including the ones that did not help — a fix that is only kept
when it improves the same data it was measured on is how a backtest gets
tuned into a number that does not survive contact with anything else.

Run:  python3 zigzag_bot.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DATASETS = {"2025": "bybit_bot/data/BTCUSDT_2025.csv",
            "2026": "bybit_bot/data/BTCUSDT_2026.csv"}
TABLE = ("/root/.claude/uploads/2499e73f-5145-5c6f-b255-816732633901/"
         "d6bcbae7-Sheet16_PureLogic_49783_1.txt")

FEE_PCT = 0.11 / 100.0   # the file's own round-trip assumption


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


# ---------------------------------------------------------------------------
# The zigzag, computed two ways
# ---------------------------------------------------------------------------
def perfect_pivots(close: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Zigzag pivots with hindsight: the extremes themselves.

    This is what the uploaded table encodes. It is not tradeable — the
    pivot index is only identifiable once the reversal has happened — but
    it sets the ceiling that the causal version is measured against.
    """
    pivots: list[tuple[int, int]] = []
    last_idx, last_price, direction = 0, close[0], 0
    ext_idx, ext_price = 0, close[0]

    for i in range(1, len(close)):
        p = close[i]
        if direction >= 0 and p > ext_price:
            ext_idx, ext_price = i, p
        elif direction <= 0 and p < ext_price:
            ext_idx, ext_price = i, p

        if direction >= 0 and p <= ext_price * (1 - threshold):
            pivots.append((ext_idx, +1))
            direction, ext_idx, ext_price = -1, i, p
        elif direction <= 0 and p >= ext_price * (1 + threshold):
            pivots.append((ext_idx, -1))
            direction, ext_idx, ext_price = +1, i, p
    return pivots


def causal_signals(close: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Zigzag as a live bot can actually see it.

    Emits (bar_index, direction) at the moment the reversal is confirmed —
    which is always later, and at a worse price, than the pivot itself.
    direction +1 means the up-leg has been confirmed (go long).
    """
    signals: list[tuple[int, int]] = []
    direction = 0
    ext_price = close[0]

    for i in range(1, len(close)):
        p = close[i]
        if direction >= 0 and p > ext_price:
            ext_price = p
        elif direction <= 0 and p < ext_price:
            ext_price = p

        if direction >= 0 and p <= ext_price * (1 - threshold):
            direction = -1
            ext_price = p
            signals.append((i, -1))
        elif direction <= 0 and p >= ext_price * (1 + threshold):
            direction = +1
            ext_price = p
            signals.append((i, +1))
    return signals


# ---------------------------------------------------------------------------
# Trading the legs
# ---------------------------------------------------------------------------
def trade_perfect(close: np.ndarray, pivots: list[tuple[int, int]]) -> dict:
    """Buy every low, sell every high, exactly on the pivot bars."""
    pnl = []
    for (a, _), (b, _) in zip(pivots, pivots[1:]):
        move = (close[b] - close[a]) / close[a]
        pnl.append(abs(move) - FEE_PCT)      # always on the right side
    return summarise(np.array(pnl) if pnl else np.array([0.0]))


def trade_causal(close: np.ndarray, signals: list[tuple[int, int]],
                 stop_pct: float | None = None,
                 min_leg_pct: float | None = None,
                 trail_pct: float | None = None) -> tuple[dict, np.ndarray, list]:
    """Enter when a leg is confirmed, exit when the next one is.

    stop_pct     hard stop, as a fraction of entry price
    min_leg_pct  skip a signal whose preceding leg was smaller than this
    trail_pct    trail the exit instead of waiting for the next signal
    """
    pnl, detail = [], []
    for k in range(len(signals) - 1):
        i, d = signals[k]
        j, _ = signals[k + 1]
        if j <= i:
            continue

        if min_leg_pct is not None and k > 0:
            prev_i = signals[k - 1][0]
            leg = abs(close[i] - close[prev_i]) / close[prev_i]
            if leg < min_leg_pct:
                continue

        entry = close[i]
        seg = close[i + 1:j + 1]
        if seg.size == 0:
            continue

        exit_idx_rel, exit_price, reason = seg.size - 1, seg[-1], "next_signal"

        if stop_pct is not None:
            adverse = (seg - entry) / entry * d
            hit = np.flatnonzero(adverse <= -stop_pct)
            if hit.size:
                exit_idx_rel, exit_price, reason = hit[0], entry * (1 - d * stop_pct), "stop"

        if trail_pct is not None:
            run = seg[:exit_idx_rel + 1]
            if d > 0:
                peak = np.maximum.accumulate(run)
                hit = np.flatnonzero(run <= peak * (1 - trail_pct))
            else:
                trough = np.minimum.accumulate(run)
                hit = np.flatnonzero(run >= trough * (1 + trail_pct))
            if hit.size and hit[0] < exit_idx_rel:
                exit_idx_rel, exit_price, reason = hit[0], run[hit[0]], "trail"

        gross = (exit_price - entry) / entry * d
        net = gross - FEE_PCT
        pnl.append(net)
        detail.append({
            "entry_bar": i, "exit_bar": i + 1 + exit_idx_rel, "dir": d,
            "gross_pct": 100 * gross, "net_pct": 100 * net, "reason": reason,
            "bars_held": exit_idx_rel + 1,
        })

    arr = np.array(pnl) if pnl else np.array([0.0])
    return summarise(arr), arr, detail


def summarise(a: np.ndarray) -> dict:
    wins, losses = a[a > 0], a[a <= 0]
    return {
        "trades": len(a),
        "win_rate_pct": round(100 * float((a > 0).mean()), 2),
        "avg_net_pct": round(100 * float(a.mean()), 4),
        "total_pct": round(100 * float(a.sum()), 1),
        "profit_factor": (round(float(wins.sum() / -losses.sum()), 3)
                          if len(losses) and losses.sum() < 0 else float("inf")),
    }


def review_losses(detail: list, close: np.ndarray) -> dict:
    """Why did the losing trades lose? Split the causes apart."""
    losers = [d for d in detail if d["net_pct"] <= 0]
    if not losers:
        return {}
    gross_pos = sum(1 for d in losers if d["gross_pct"] > 0)
    tiny = sum(1 for d in losers if abs(d["gross_pct"]) < 100 * FEE_PCT)
    wrong_way = sum(1 for d in losers if d["gross_pct"] <= -100 * FEE_PCT)
    quick = sum(1 for d in losers if d["bars_held"] <= 3)
    return {
        "losers": len(losers),
        "share_of_all": round(100 * len(losers) / len(detail), 1),
        "profitable_before_fees": gross_pos,
        "move_smaller_than_fee": tiny,
        "went_the_wrong_way": wrong_way,
        "whipsaw_3_bars_or_less": quick,
        "avg_loss_pct": round(float(np.mean([d["net_pct"] for d in losers])), 4),
    }


def main() -> None:
    data = {y: load(p) for y, p in DATASETS.items()}
    table = pd.read_csv(TABLE, sep="\t")

    print("=" * 84)
    print("THE TABLE IS A ZIGZAG — confirming that from the file itself")
    print("=" * 84)
    print(f"  LONG / SHORT              : {(table['direction']=='LONG').sum():,}"
          f" / {(table['direction']=='SHORT').sum():,}  (alternating legs)")
    print(f"  minutes held / available  : {table['duration_min'].sum():,.0f}"
          f" / {525_600+307_379:,}  = "
          f"{table['duration_min'].sum()/(525_600+307_379):.2f}x  (no overlap)")
    print(f"  median leg net / gross    : {table['net_ret_pct'].median():.3f}%"
          f" / {table['gross_ret_pct'].median():.3f}%")
    print(f"  median duration           : {table['duration_min'].median():.0f} min")

    print()
    print("=" * 84)
    print("STEP 1 — the ceiling: zigzag with perfect pivot timing")
    print("=" * 84)
    print("Buying every low and selling every high, on the pivot bar itself.")
    print("Not tradeable, but it is what the uploaded table measures.\n")
    print(f"{'threshold':>10s} | " + " | ".join(
        f"{y:>4s} {'legs':>7s} {'win%':>6s} {'net%/leg':>9s} {'total%':>12s}"
        for y in DATASETS))
    print("-" * 84)
    for thr in (0.002, 0.005, 0.01, 0.02):
        cells = []
        for y, df in data.items():
            c = df["close"].to_numpy(float)
            r = trade_perfect(c, perfect_pivots(c, thr))
            cells.append(f"{y:>4s} {r['trades']:7d} {r['win_rate_pct']:6.2f} "
                         f"{r['avg_net_pct']:9.4f} {r['total_pct']:12.1f}")
        print(f"{thr*100:9.2f}% | " + " | ".join(cells))

    print()
    print("=" * 84)
    print("STEP 2 — the same zigzag, detected causally (what a bot can do)")
    print("=" * 84)
    print("Enter when the reversal is confirmed, exit when the next one is.\n")
    print(f"{'threshold':>10s} | " + " | ".join(
        f"{y:>4s} {'trades':>7s} {'win%':>6s} {'net%/t':>9s} {'total%':>12s}"
        for y in DATASETS))
    print("-" * 84)
    base_detail = {}
    for thr in (0.002, 0.005, 0.01, 0.02):
        cells = []
        for y, df in data.items():
            c = df["close"].to_numpy(float)
            r, arr, detail = trade_causal(c, causal_signals(c, thr))
            base_detail[(thr, y)] = (detail, c)
            cells.append(f"{y:>4s} {r['trades']:7d} {r['win_rate_pct']:6.2f} "
                         f"{r['avg_net_pct']:9.4f} {r['total_pct']:12.1f}")
        print(f"{thr*100:9.2f}% | " + " | ".join(cells))

    print()
    print("=" * 84)
    print("STEP 3 — reviewing the losing trades: bad application, or bad logic?")
    print("=" * 84)
    for thr in (0.005, 0.01):
        for y in DATASETS:
            detail, c = base_detail[(thr, y)]
            rev = review_losses(detail, c)
            if not rev:
                continue
            print(f"\n  threshold {thr*100:.2f}%, {y}: {rev['losers']:,} losers "
                  f"({rev['share_of_all']}% of trades), average {rev['avg_loss_pct']:+.4f}%")
            print(f"    already profitable before fees : {rev['profitable_before_fees']:,}")
            print(f"    move smaller than the fee      : {rev['move_smaller_than_fee']:,}")
            print(f"    price went the other way       : {rev['went_the_wrong_way']:,}")
            print(f"    whipsaw, out within 3 bars     : {rev['whipsaw_3_bars_or_less']:,}")

    print()
    print("=" * 84)
    print("STEP 4 — fixes aimed at the causes above, each measured on BOTH years")
    print("=" * 84)
    print("v1  plain causal zigzag")
    print("v2  + hard stop, so a leg that reverses cannot run")
    print("v3  + skip signals whose previous leg was too small (chop filter)")
    print("v4  + trailing exit, to keep more of a leg than the next signal leaves\n")

    variants = [
        ("v1 plain", dict()),
        ("v2 stop", dict(stop_pct=0.005)),
        ("v3 stop+minleg", dict(stop_pct=0.005, min_leg_pct=0.008)),
        ("v4 stop+minleg+trail", dict(stop_pct=0.005, min_leg_pct=0.008, trail_pct=0.003)),
    ]
    print(f"{'variant':22s} {'thr':>6s} | " + " | ".join(
        f"{y:>4s} {'trades':>7s} {'win%':>6s} {'net%/t':>9s} {'total%':>11s}"
        for y in DATASETS) + " | both+")
    print("-" * 96)
    winners = []
    for thr in (0.005, 0.01, 0.02):
        for name, kw in variants:
            cells, totals = [], []
            for y, df in data.items():
                c = df["close"].to_numpy(float)
                r, _, _ = trade_causal(c, causal_signals(c, thr), **kw)
                totals.append(r["avg_net_pct"])
                cells.append(f"{y:>4s} {r['trades']:7d} {r['win_rate_pct']:6.2f} "
                             f"{r['avg_net_pct']:9.4f} {r['total_pct']:11.1f}")
            ok = all(t > 0 for t in totals)
            if ok:
                winners.append((name, thr, totals))
            print(f"{name:22s} {thr*100:5.2f}% | " + " | ".join(cells)
                  + f" | {'YES' if ok else ''}")

    print()
    print("=" * 84)
    if winners:
        print("Variants profitable per trade on BOTH years:")
        for name, thr, totals in winners:
            print(f"  {name} at {thr*100:.2f}%: "
                  + ", ".join(f"{t:+.4f}%" for t in totals))
    else:
        print("No variant reached positive expected value on both years.")


if __name__ == "__main__":
    main()
