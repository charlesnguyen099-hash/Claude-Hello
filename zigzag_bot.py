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
def _walk(close: np.ndarray, threshold: float, report: str
          ) -> list[tuple[int, int]]:
    """Shared zigzag walk. Tracks the running high and the running low
    separately — the two cannot share one variable, or the "extreme"
    just follows the latest price and no reversal is ever detected.

    report="pivot"  -> the bar where the extreme actually occurred
                       (hindsight; only identifiable afterwards)
    report="signal" -> the bar where the reversal became confirmable
                       (causal; this is what a live bot can act on)
    """
    out: list[tuple[int, int]] = []
    direction = 0
    hi = lo = close[0]
    hi_i = lo_i = 0

    for i in range(1, len(close)):
        p = close[i]
        if direction > 0:
            if p > hi:
                hi, hi_i = p, i
            elif p <= hi * (1 - threshold):
                out.append((hi_i if report == "pivot" else i, +1))
                direction, lo, lo_i = -1, p, i
        elif direction < 0:
            if p < lo:
                lo, lo_i = p, i
            elif p >= lo * (1 + threshold):
                out.append((lo_i if report == "pivot" else i, -1))
                direction, hi, hi_i = +1, p, i
        else:
            if p > hi:
                hi, hi_i = p, i
            if p < lo:
                lo, lo_i = p, i
            if p <= hi * (1 - threshold):
                out.append((hi_i if report == "pivot" else i, +1))
                direction, lo, lo_i = -1, p, i
            elif p >= lo * (1 + threshold):
                out.append((lo_i if report == "pivot" else i, -1))
                direction, hi, hi_i = +1, p, i
    return out


def perfect_pivots(close: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Zigzag pivots with hindsight: the extreme bars themselves.

    This is what the uploaded table encodes. It is not tradeable — the
    pivot bar is only identifiable once the reversal has happened — but
    it sets the ceiling the causal version is measured against.
    Marker +1 = a top, -1 = a bottom.
    """
    return _walk(close, threshold, "pivot")


def causal_signals(close: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Zigzag as a live bot can actually see it.

    Emits (bar_index, direction) at the moment the reversal is confirmed,
    which is always later, and at a worse price, than the pivot itself.
    A +1 marker means a top was just confirmed, so the new leg is DOWN.
    """
    raw = _walk(close, threshold, "signal")
    # Convert "which pivot was confirmed" into "which way to trade now".
    return [(i, -1 if marker > 0 else +1) for i, marker in raw]


# ---------------------------------------------------------------------------
# Trading the legs
# ---------------------------------------------------------------------------
def trade_perfect(close: np.ndarray, pivots: list[tuple[int, int]]) -> dict:
    """Buy every low, sell every high, exactly on the pivot bars."""
    pnl = []
    for (a, _), (b, _) in zip(pivots, pivots[1:]):
        move = (close[b] - close[a]) / close[a]
        pnl.append(abs(move) - FEE_PCT)      # always on the right side
    return summarise(np.array(pnl))


def trade_causal(close: np.ndarray, signals: list[tuple[int, int]],
                 stop_pct: float | None = None,
                 min_leg_pct: float | None = None,
                 trail_pct: float | None = None,
                 invert: bool = False) -> tuple[dict, np.ndarray, list]:
    """Enter when a leg is confirmed, exit when the next one is.

    stop_pct     hard stop, as a fraction of entry price
    min_leg_pct  skip a signal whose preceding leg was smaller than this
    trail_pct    trail the exit instead of waiting for the next signal
    invert       trade AGAINST the confirmed reversal instead of with it
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

        gross = (exit_price - entry) / entry * d * (-1 if invert else 1)
        net = gross - FEE_PCT
        pnl.append(net)
        detail.append({
            "entry_bar": i, "exit_bar": i + 1 + exit_idx_rel, "dir": d,
            "gross_pct": 100 * gross, "net_pct": 100 * net, "reason": reason,
            "bars_held": exit_idx_rel + 1,
        })

    arr = np.array(pnl)
    return summarise(arr), arr, detail


def summarise(a: np.ndarray) -> dict:
    """Net is what lands in the account; gross adds the fee back, which is
    the number that says whether the rule has any edge at all."""
    if a.size == 0:
        return {"trades": 0, "win_rate_pct": 0.0, "avg_net_pct": 0.0,
                "avg_gross_pct": 0.0, "total_pct": 0.0, "profit_factor": 0.0}
    wins, losses = a[a > 0], a[a <= 0]
    return {
        "trades": len(a),
        "win_rate_pct": round(100 * float((a > 0).mean()), 2),
        "avg_net_pct": round(100 * float(a.mean()), 4),
        "avg_gross_pct": round(100 * float(a.mean() + FEE_PCT), 4),
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
    print("The review says most losers are 'price went the other way' — the")
    print("confirmed reversal simply did not hold. So the fixes target that:")
    print("  v1 plain      trade the confirmed reversal")
    print("  v2 stop       cap how far a failed reversal can run")
    print("  v3 +minleg    skip signals following a leg too small to trust")
    print("  v5 INVERTED   if reversals fail this often, trade the continuation")
    print()
    print("The column that decides everything is GROSS — net with the fee added")
    print("back. Gross is the edge itself. If gross is not above the 0.110% fee,")
    print("no amount of tuning the exits can make the rule pay.\n")

    variants = [
        ("v1 plain", dict()),
        ("v2 stop", dict(stop_pct=0.005)),
        ("v3 stop+minleg", dict(stop_pct=0.005, min_leg_pct=0.008)),
        ("v5 INVERTED", dict(invert=True)),
    ]
    print(f"{'variant':16s} {'thr':>6s} | " + " | ".join(
        f"{y:>4s} {'trades':>7s} {'win%':>6s} {'gross%':>8s} {'net%':>8s} {'total%':>9s}"
        for y in DATASETS) + " | both+")
    print("-" * 104)
    winners = []
    for thr in (0.005, 0.01, 0.02, 0.03, 0.05, 0.08):
        for name, kw in variants:
            cells, nets, grosses = [], [], []
            for y, df in data.items():
                c = df["close"].to_numpy(float)
                r, _, _ = trade_causal(c, causal_signals(c, thr), **kw)
                if r["trades"] == 0:
                    cells.append(f"{y:>4s} {0:7d} {'-':>6s} {'-':>8s} {'-':>8s} {'-':>9s}")
                    nets.append(-1.0); grosses.append(-1.0)
                    continue
                nets.append(r["avg_net_pct"]); grosses.append(r["avg_gross_pct"])
                cells.append(f"{y:>4s} {r['trades']:7d} {r['win_rate_pct']:6.2f} "
                             f"{r['avg_gross_pct']:8.4f} {r['avg_net_pct']:8.4f} "
                             f"{r['total_pct']:9.1f}")
            ok = all(n > 0 for n in nets)
            if ok:
                winners.append((name, thr, nets))
            print(f"{name:16s} {thr*100:5.2f}% | " + " | ".join(cells)
                  + f" | {'YES' if ok else ''}")
        print()

    print("=" * 84)
    if winners:
        print("Variants with positive expected value on BOTH years:")
        for name, thr, nets in winners:
            print(f"  {name} at {thr*100:.2f}%: " + ", ".join(f"{n:+.4f}%" for n in nets))
    else:
        print("No variant reached positive expected value on both years.")
        print()
        print("Read the gross column down the page. That is the edge before")
        print("costs, and it is what would have to clear 0.110% for any of this")
        print("to work. Tuning stops and exits moves net around; it does not")
        print("create gross.")


if __name__ == "__main__":
    main()
