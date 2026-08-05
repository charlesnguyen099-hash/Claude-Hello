"""Keep only the rules that pay, then check whether that actually works.

    python -m fp.walkforward                    # both numbers, side by side
    python -m fp.walkforward --fee maker

TWO BACKTESTS, AND ONLY ONE OF THEM MEANS ANYTHING

The first is the one that gets asked for: build the table from 2025, 2026
and August, throw away every rule that loses, then trade those years. It
reports a large profit. It always will, on any data, from any rule set,
because "the rules that won" is what was selected and "did the rules that
won, win?" is not a question. It is printed here only so the number has a
label attached the first time it is seen.

The second walks forward. Time is cut into blocks. To trade block k, the
table may only use blocks 0..k-1 -- rules are learned, filtered for
support and profitability, and then applied to a block that had no say in
choosing them. Every trade is placed with information that existed before
it. That number is what "would this have made money" means.

Both are computed from identical data, identical fees and identical
leverage. The only difference is whether a rule was allowed to see the
bar it is being scored on.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fp import logic as L
from fp import patterns as P

DATA = Path(__file__).resolve().parent.parent / "bybit_bot" / "data"
CACHE = Path(__file__).resolve().parent.parent / ".wf_cache.pkl"

# A 15m trade that has not resolved in 200 bars (just over two days) is
# closed at market. The median target resolves in ~24.
MAX_LOOKFORWARD = 200
# What it takes for a rule to be trusted, inside whatever window is
# allowed to see. These are the "only trade what is certainly profitable"
# knobs, and they are applied identically in both backtests.
MIN_SUPPORT = 20
MIN_WIN_RATE = 0.50
MAJORITY = 0.70


def load_series() -> pd.DataFrame:
    """2025, 2026 and August as one continuous 15m series."""
    frames = []
    for name in ("BTCUSDT_2025.csv", "BTCUSDT_2026.csv", "BTCUSDT_202608.csv"):
        d = pd.read_csv(DATA / name, sep=None, engine="python")
        d.columns = [c.strip().lower() for c in d.columns]
        dt = next(c for c in d.columns if "time" in c or "date" in c)
        d["datetime"] = pd.to_datetime(d[dt])
        frames.append(d[["datetime", "open", "high", "low", "close", "volume"]])
    raw = (pd.concat(frames).drop_duplicates("datetime")
           .sort_values("datetime").reset_index(drop=True))
    return L.to_bars(raw, P.BAR_MINUTES)


def outcomes(bars: pd.DataFrame, exit_name: str) -> dict:
    """Per bar: the signature, and the gross return of a long and a short.

    Gross, so a fee can be applied afterwards without recomputing. Cached,
    because this is the only slow part and the folds reuse it.
    """
    h, lo, c = bars["high"].values, bars["low"].values, bars["close"].values
    v = bars["volume"].values
    atr = P.atr_series(bars)
    n = len(c)
    tp_mult = L.TP_MULTIPLES.get(exit_name, L.TRAIL_MULTIPLE)
    keys = np.full(n, None, dtype=object)
    gl = np.full(n, np.nan)
    gs = np.full(n, np.nan)

    for i in range(max(P.LOOKBACK, 20), n - 1):
        a = atr[i]
        if a <= 0 or not np.isfinite(a):
            continue
        keys[i] = P.signature(c, h, lo, v, i, a, P.LOOKBACK)
        entry = c[i]
        end = min(i + 1 + MAX_LOOKFORWARD, n)
        win_h, win_l = h[i + 1:end], lo[i + 1:end]
        if len(win_h) == 0:
            continue
        for d, out in ((1, "l"), (-1, "s")):
            tp = entry + d * tp_mult * a
            sl = entry - d * L.SL_MULTIPLE * a
            if d > 0:
                hit_sl = np.flatnonzero(win_l <= sl)
                hit_tp = np.flatnonzero(win_h >= tp)
            else:
                hit_sl = np.flatnonzero(win_h >= sl)
                hit_tp = np.flatnonzero(win_l <= tp)
            j_sl = hit_sl[0] if len(hit_sl) else np.inf
            j_tp = hit_tp[0] if len(hit_tp) else np.inf
            if j_sl <= j_tp:                      # ambiguous bar goes to the stop
                r = -L.SL_MULTIPLE * a / entry
            elif j_tp < np.inf:
                r = tp_mult * a / entry
            else:
                r = (c[end - 1] - entry) / entry * d
            if d > 0:
                gl[i] = r
            else:
                gs[i] = r
    return {"keys": keys, "long": gl, "short": gs,
            "atr_pct": 100.0 * atr / c, "when": bars["datetime"].values}


def cached_outcomes(bars: pd.DataFrame, exit_name: str) -> dict:
    sig = (len(bars), exit_name, P.BAR_MINUTES, P.LOOKBACK, MAX_LOOKFORWARD)
    if CACHE.exists():
        try:
            blob = pickle.loads(CACHE.read_bytes())
            if blob.get("sig") == sig:
                return blob["data"]
        except Exception:
            pass
    data = outcomes(bars, exit_name)
    CACHE.write_bytes(pickle.dumps({"sig": sig, "data": data}))
    return data


def build_table(o: dict, idx: np.ndarray, fee: float) -> dict:
    """Learn rules from the bars in `idx` only, and keep the confident ones.

    A rule survives when, inside this window: it fired at least
    MIN_SUPPORT times, its side owns at least MAJORITY of the winning
    labels, it won at least MIN_WIN_RATE of the time, AND its mean return
    after fees is positive. That last condition is the "only what is
    certainly profitable" filter -- and it is applied to the learn window
    only, never to the window being traded.
    """
    stats: dict[str, dict] = {}
    for i in idx:
        k = o["keys"][i]
        if k is None or not np.isfinite(o["long"][i]):
            continue
        rl, rs = o["long"][i] - fee, o["short"][i] - fee
        d = 1 if rl >= rs else -1
        best = max(rl, rs)
        s = stats.setdefault(k, {"nl": 0, "ns": 0, "sum": {1: 0.0, -1: 0.0},
                                 "n": {1: 0, -1: 0}, "w": {1: 0, -1: 0}})
        if best > 0:
            if d > 0:
                s["nl"] += 1
            else:
                s["ns"] += 1
        for dd, r in ((1, rl), (-1, rs)):
            s["n"][dd] += 1
            s["sum"][dd] += r
            s["w"][dd] += r > 0

    table = {}
    for k, s in stats.items():
        tot = s["nl"] + s["ns"]
        if tot == 0:
            continue
        d = 1 if s["nl"] >= s["ns"] else -1
        if max(s["nl"], s["ns"]) / tot < MAJORITY:
            continue
        n = s["n"][d]
        if n < MIN_SUPPORT:
            continue
        win = s["w"][d] / n
        mean = s["sum"][d] / n
        if win < MIN_WIN_RATE or mean <= 0:
            continue
        table[k] = {"dir": d, "n": n, "win": win, "mean": mean}
    return table


def trade(o: dict, idx: np.ndarray, table: dict, fee: float,
          use_leverage: bool) -> dict:
    """Apply a table to a set of bars and report what it made."""
    rets, levs = [], []
    for i in idx:
        k = o["keys"][i]
        rule = table.get(k) if k is not None else None
        if rule is None or not np.isfinite(o["long"][i]):
            continue
        d = rule["dir"]
        r = (o["long"][i] if d > 0 else o["short"][i]) - fee
        lev = 1.0
        if use_leverage:
            chain = L.leverage_potential(o["atr_pct"][i], None,
                                         conviction=rule["win"])
            if not chain["tradeable"]:
                continue
            lev = chain["leverage"]
        rets.append(r)
        levs.append(lev)
    if not rets:
        return {"n": 0, "win_rate": 0.0, "gross_pct": 0.0, "lev_pct": 0.0,
                "total_pct": 0.0, "avg_lev": 0.0}
    r = np.array(rets)
    lv = np.array(levs)
    return {"n": len(r), "win_rate": float((r > 0).mean()),
            "net_per_trade_pct": 100 * float(r.mean()),
            "lev_per_trade_pct": 100 * float((r * lv).mean()),
            "total_pct": 100 * float((r * lv).sum()),
            "avg_lev": float(lv.mean())}


def compound(o: dict, idx: np.ndarray, table: dict, fee: float,
             margin_pct: float = 0.05) -> float:
    """Equity multiple, sizing each trade at margin_pct of current equity."""
    eq = 1.0
    for i in idx:
        k = o["keys"][i]
        rule = table.get(k) if k is not None else None
        if rule is None or not np.isfinite(o["long"][i]):
            continue
        d = rule["dir"]
        r = (o["long"][i] if d > 0 else o["short"][i]) - fee
        chain = L.leverage_potential(o["atr_pct"][i], None,
                                     conviction=rule["win"])
        if not chain["tradeable"]:
            continue
        pnl = r * chain["leverage"] * margin_pct
        eq *= max(0.0, 1.0 + pnl)
        if eq <= 1e-6:
            return 0.0
    return eq


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fee", default="taker", choices=["maker", "taker", "taker-slip"])
    p.add_argument("--exit", default=L.DEFAULT_EXIT, choices=L.EXIT_STRATEGIES)
    p.add_argument("--folds", type=int, default=12)
    a = p.parse_args(argv)
    fee = {"maker": L.MAKER_ROUND_TRIP, "taker": L.TAKER_ROUND_TRIP,
           "taker-slip": L.TAKER_WITH_SLIPPAGE}[a.fee]

    bars = load_series()
    print(f"{len(bars):,} {P.BAR_MINUTES}m bars, "
          f"{bars.datetime.min().date()} .. {bars.datetime.max().date()}")
    print(f"fee {a.fee} {fee*100:.3f}% round trip, exit {a.exit}")
    o = cached_outcomes(bars, a.exit)
    usable = np.array([i for i in range(len(bars))
                       if o["keys"][i] is not None and np.isfinite(o["long"][i])])
    print(f"{len(usable):,} usable bars\n")

    print("=" * 78)
    print("BACKTEST 1 — the table sees every bar it is scored on")
    print("=" * 78)
    t_all = build_table(o, usable, fee)
    r_all = trade(o, usable, t_all, fee, use_leverage=True)
    eq_all = compound(o, usable, t_all, fee)
    print(f"  rules kept        : {len(t_all):,}")
    print(f"  trades            : {r_all['n']:,}")
    print(f"  win rate          : {100*r_all['win_rate']:.1f}%")
    print(f"  net per trade     : {r_all['net_per_trade_pct']:+.4f}% "
          f"unleveraged, {r_all['lev_per_trade_pct']:+.4f}% at "
          f"{r_all['avg_lev']:.0f}x average")
    print(f"  $10 would become  : ${10*eq_all:,.2f}")
    print("\n  This number is circular. The rules were chosen for winning on")
    print("  these bars, so measuring them on these bars asks nothing. Any")
    print("  rule set built this way scores like this, including one built")
    print("  from coin flips.")

    print("\n" + "=" * 78)
    print(f"BACKTEST 2 — walk forward, {a.folds} blocks, each traded with rules")
    print("             learned only from the blocks before it")
    print("=" * 78)
    edges = np.linspace(0, len(usable), a.folds + 1).astype(int)
    print(f"{'block':>6} {'dates':>25} {'rules':>7} {'trades':>7} "
          f"{'win%':>7} {'net/trade':>11} {'total%':>10}")
    eq, all_r, all_lev = 1.0, [], []
    for f in range(1, a.folds):
        learn_idx = usable[:edges[f]]
        test_idx = usable[edges[f]:edges[f + 1]]
        if len(test_idx) == 0:
            continue
        table = build_table(o, learn_idx, fee)
        res = trade(o, test_idx, table, fee, use_leverage=True)
        d0 = pd.Timestamp(o["when"][test_idx[0]]).date()
        d1 = pd.Timestamp(o["when"][test_idx[-1]]).date()
        eq *= compound(o, test_idx, table, fee)
        if res["n"]:
            for i in test_idx:
                k = o["keys"][i]
                rule = table.get(k) if k is not None else None
                if rule is None or not np.isfinite(o["long"][i]):
                    continue
                rr = (o["long"][i] if rule["dir"] > 0 else o["short"][i]) - fee
                ch = L.leverage_potential(o["atr_pct"][i], None,
                                          conviction=rule["win"])
                if ch["tradeable"]:
                    all_r.append(rr)
                    all_lev.append(ch["leverage"])
        print(f"{f:>6} {str(d0)+' '+str(d1):>25} {len(table):>7,} {res['n']:>7,} "
              f"{100*res['win_rate']:>6.1f}% "
              f"{res.get('net_per_trade_pct', 0):>10.4f}% "
              f"{res.get('total_pct', 0):>9.2f}%")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if all_r:
        r = np.array(all_r)
        lv = np.array(all_lev)
        t = r.mean() / (r.std() / np.sqrt(len(r))) if r.std() > 0 else 0.0
        print(f"  out-of-sample trades : {len(r):,}")
        print(f"  win rate             : {100*(r>0).mean():.1f}%")
        print(f"  net per trade        : {100*r.mean():+.4f}% "
              f"(t = {t:.2f}, needs |t| > 2 to be distinguishable from zero)")
        print(f"  with leverage        : {100*(r*lv).mean():+.4f}% per trade, "
              f"{lv.mean():.0f}x average")
        print(f"  $10 would become     : ${10*eq:,.2f}")
        print()
        if r.mean() > 0 and t > 2:
            print("  Positive and significant out of sample.")
        elif r.mean() > 0:
            print("  Positive out of sample but NOT significant -- this much")
            print("  edge is what a coin flip produces about as often as not.")
        else:
            print("  NEGATIVE out of sample. Selecting the profitable rules")
            print("  did not survive contact with bars that had no say in the")
            print("  selection. Compare the two numbers above: that gap is")
            print("  the entire story.")
    else:
        print("  The walk-forward produced no trades at all: no rule cleared")
        print("  the support and profitability filters on data it was then")
        print("  tested against.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
