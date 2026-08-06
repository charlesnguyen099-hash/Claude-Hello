"""Thousands of logics, scored live, only the ones currently working traded.

    python -m fp.ensemble                  # build, select, walk forward
    python -m fp.ensemble --top 10 --review 90

THE IDEA, WHICH IS YOURS

Roughly a hundred factors crossed with roughly fifty ways of turning a
factor into a position gives several thousand logics. At every moment,
look at what each has been doing lately, and trade only the ones that are
currently working. When one stops working it drops out; when another
starts, it comes in.

WHY THIS IS NOT THE SEARCH THAT ALREADY FAILED

fp/search.py tested 67,070 rules and lost 0.27% per trade forward. Every
one of those rules lived inside the old structure: a target and a stop
roughly seven hours apart, at 17-100x leverage. That structure loses on
this data even holding a position that was correct for nineteen months,
because volatility drag is quadratic in leverage and paid per turnover.
So that search was asking which forecast could survive a structure that
nothing survives.

Every logic here is built inside the corrected structure instead:

    daily bars                   a signal that can change every 30m forces
                                 48 turnovers a day and pays drag on each
    held until the logic flips   no target, no stop, no re-entry
    return measured per trade    entry to exit, never rebalanced inside
    leverage solved with drag    move*L - cost*L - drag*L^2 has a maximum

That makes the selection question the only question left, which is what
it should have been all along.

HOW SELECTION WORKS, AND WHY IT IS HONEST BY CONSTRUCTION

Every `--review` days, each logic is scored on the trailing window only.
The top `--top` by that score are held for the next window. A logic never
sees the days it is judged on. There is no fitting step, no threshold
chosen with hindsight, and no way for a future bar to reach backwards --
the whole record is out of sample by construction.

What it reports is therefore the real thing: what this procedure would
have returned, run live, from day one.

WHAT IT REPORTED, on 100 factors x 38 methods = 2,602 usable logics over
583 daily bars of 2025-2026:

    top 10, re-scored every 60d on a trailing 180d window
    windows profitable   0 of 6
    total at 1x          -22.4%
    Sharpe               -1.80

And swept across the parameters, with the inverse tested alongside --
because if picking the winners loses, picking the losers is the next
thing to check:

     top  review  window   pick best   pick worst
       5      30     120      -36.0%        +6.9%
       5      90     360       +4.1%        -2.7%
      25      30     120      -22.8%        +8.6%
     100      90     360       -8.7%        -7.3%
     400      90     360      -10.6%        -6.5%
                     mean      -14.1%        -1.0%
                 positive      1 of 16      5 of 16

Picking the currently-working logics loses in 15 of 16 settings. Picking
the currently-failing ones averages roughly zero. Neither direction pays,
which is a stronger statement than either losing alone: the trailing
ranking carries no information about the next window at all.

The one positive cell (+4.1%) is one of sixteen. That is what chance
produces.

WHY THIS ONE IS WORTH HAVING RUN ANYWAY

The earlier 67,070-rule search failed inside the broken structure, so it
could never separate "the forecast is no good" from "the structure eats
everything". This one runs inside the corrected structure -- daily bars,
held to the flip, drag-aware sizing -- and still finds nothing. That
narrows the answer: the leak was structural AND the forecast is absent.
Fixing the structure was necessary and is not sufficient.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "bybit_bot" / "data"
FEE = 0.0011                # taker in + taker out, per round trip
FUND_DAY = 0.0003           # 0.010% per 8h, charged as a cost either side


# ------------------------------------------------------------------ data

def load_daily() -> pd.DataFrame:
    def rd(p):
        d = pd.read_csv(p, sep=None, engine="python")
        d.columns = [c.strip().lower() for c in d.columns]
        d["datetime"] = pd.to_datetime(
            d[[c for c in d.columns if "time" in c or "date" in c][0]])
        return d[["datetime", "open", "high", "low", "close", "volume"]]

    files = sorted(DATA.glob("BTCUSDT_*.csv"))
    raw = (pd.concat([rd(f) for f in files]).drop_duplicates("datetime")
           .sort_values("datetime").set_index("datetime"))
    d = pd.DataFrame({
        "open": raw["open"].resample("1D").first(),
        "high": raw["high"].resample("1D").max(),
        "low": raw["low"].resample("1D").min(),
        "close": raw["close"].resample("1D").last(),
        "volume": raw["volume"].resample("1D").sum(),
    }).dropna()
    return d


# --------------------------------------------------------------- factors

def factors(d: pd.DataFrame) -> dict[str, pd.Series]:
    """About a hundred, all past-only, all on daily bars."""
    c, h, l, v = d["close"], d["high"], d["low"], d["volume"]
    r = c.pct_change()
    f: dict[str, pd.Series] = {}
    for n in (3, 5, 8, 13, 21, 34, 55, 89, 120, 160, 200):
        f[f"mom{n}"] = c.pct_change(n)
        f[f"ma_dist{n}"] = c / c.rolling(n).mean() - 1
        f[f"vol{n}"] = r.rolling(n).std()
        rng_h, rng_l = h.rolling(n).max(), l.rolling(n).min()
        f[f"pos{n}"] = ((c - rng_l) / (rng_h - rng_l)).where(rng_h > rng_l, 0.5)
        f[f"volr{n}"] = v / v.rolling(n).mean()
        up = (r > 0).rolling(n).mean()
        f[f"upshare{n}"] = up
        f[f"skew{n}"] = r.rolling(n).skew()
        f[f"maxdd{n}"] = c / c.rolling(n).max() - 1
    for a, b in ((5, 21), (8, 34), (13, 55), (21, 89), (34, 120), (55, 200)):
        f[f"macross{a}_{b}"] = c.rolling(a).mean() / c.rolling(b).mean() - 1
        f[f"volratio{a}_{b}"] = (r.rolling(a).std() / r.rolling(b).std()) - 1
    return {k: s for k, s in f.items() if s.notna().sum() > 200}


# --------------------------------------------------------------- methods

def methods() -> list[tuple[str, callable]]:
    """Ways of turning one factor into a position in {-1, 0, +1}.

    Each is a pure function of the factor's own past, so crossing them
    with the factor library gives the full logic set.
    """
    def sign_follow(x, **kw):
        return np.sign(x)

    def sign_fade(x, **kw):
        return -np.sign(x)

    def z_follow(x, w=90, k=1.0):
        z = (x - x.rolling(w).mean()) / x.rolling(w).std()
        return np.sign(z) * (z.abs() > k)

    def z_fade(x, w=90, k=1.0):
        z = (x - x.rolling(w).mean()) / x.rolling(w).std()
        return -np.sign(z) * (z.abs() > k)

    def rank_follow(x, w=180, hi=0.8):
        q = x.rolling(w).rank(pct=True)
        return (q > hi).astype(float) - (q < 1 - hi).astype(float)

    def rank_fade(x, w=180, hi=0.8):
        q = x.rolling(w).rank(pct=True)
        return (q < 1 - hi).astype(float) - (q > hi).astype(float)

    def breakout(x, w=60):
        return ((x >= x.rolling(w).max()).astype(float)
                - (x <= x.rolling(w).min()).astype(float))

    def breakout_fade(x, w=60):
        return ((x <= x.rolling(w).min()).astype(float)
                - (x >= x.rolling(w).max()).astype(float))

    out = []
    for name, fn in (("follow", sign_follow), ("fade", sign_fade)):
        out.append((name, fn))
    for w in (45, 90, 180):
        for k in (0.5, 1.0, 1.5):
            out.append((f"zfollow{w}_{k}", lambda x, w=w, k=k: z_follow(x, w, k)))
            out.append((f"zfade{w}_{k}", lambda x, w=w, k=k: z_fade(x, w, k)))
    for w in (90, 180, 360):
        for hi in (0.7, 0.85):
            out.append((f"rankfollow{w}_{hi}", lambda x, w=w, h=hi: rank_follow(x, w, h)))
            out.append((f"rankfade{w}_{hi}", lambda x, w=w, h=hi: rank_fade(x, w, h)))
    for w in (30, 60, 120):
        out.append((f"break{w}", lambda x, w=w: breakout(x, w)))
        out.append((f"breakfade{w}", lambda x, w=w: breakout_fade(x, w)))
    return out


# ------------------------------------------------------------------ P&L

def trade_returns(close: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Net return per trade, entry to exit, never rebalanced inside.

    This is the whole point. A position that survives twenty days is one
    trade paying one round trip and twenty days of funding -- not twenty
    daily positions each paying drag.

    The exit is the bar AFTER the run ends. The signal still says `d` on
    the run's last bar and only changes on the next one, so the next one
    is when the change can be acted on. Closing a bar earlier books the
    price from before the bar that caused the flip -- and that bar is
    usually the one that went against the position, so skipping it
    flatters every logic measured this way.
    """
    out = []
    n = len(pos)
    i = 0
    while i < n:
        d = pos[i]
        if d == 0 or not np.isfinite(d):
            i += 1
            continue
        j = i
        while j + 1 < n and pos[j + 1] == d:
            j += 1
        if j + 1 < n:                     # a trade still open at the end
            p0, p1 = close[i], close[j + 1]
            if p0 > 0 and np.isfinite(p1):
                held = j + 1 - i
                out.append(d * (p1 - p0) / p0 - FEE - held * FUND_DAY)
        i = j + 1
    return np.array(out)


def daily_net(close: pd.Series, pos: pd.Series) -> pd.Series:
    """Per-day net so trades can be aggregated on a calendar.

    Inside a held position only funding accrues; the round-trip fee lands
    on the day the position opens.
    """
    p = pos.shift(1).fillna(0.0)
    r = close.pct_change().fillna(0.0)
    opened = (p != p.shift(1).fillna(0.0)) & (p != 0)
    return p * r - opened.astype(float) * FEE - p.abs() * FUND_DAY


# --------------------------------------------------------------- the run

def build_logics(d: pd.DataFrame, fast: bool = False,
                 only: set[str] | None = None) -> dict[str, pd.Series]:
    """`fast` drops the rolling-rank methods, which are O(n*window) and
    dominate the cost. Live, on 690 symbols, the full set takes 11 seconds
    per symbol; the fast set takes a fraction of that and keeps the
    families the study actually selected.

    `only` restricts the build to a named set of "factor|method" keys and
    is what makes a book tradeable live. A book names thirty rules; the
    full library is 2,602 and the fast one 1,150, so building everything
    to read thirty of them is roughly forty times the work -- the
    difference between a scan that finishes inside its bar and one that
    does not. Both the factor and the method loops are pruned, not just
    the output, so nothing unused is ever computed.

    The filters below still apply: a named logic that is too static or too
    churny on THIS symbol is dropped exactly as it would be in a full
    build, so a book rule is never traded on a symbol where it degenerates.
    """
    want_f = want_m = None
    if only:
        want_f = {k.split("|", 1)[0] for k in only}
        want_m = {k.split("|", 1)[1] for k in only if "|" in k}
    F = factors(d)
    M = methods()
    if want_f is not None:
        F = {k: v for k, v in F.items() if k in want_f}
        M = [(n, f) for n, f in M if n in want_m]
    if fast:
        F = {k: v for k, v in F.items()
             if k.startswith(("mom", "ma_dist", "macross", "pos", "maxdd"))}
        M = [(n, f) for n, f in M
             if not n.startswith(("rankfollow", "rankfade"))]
    logics = {}
    for (fname, fs), (mname, fn) in itertools.product(F.items(), M):
        try:
            p = fn(fs)
        except Exception:
            continue
        p = pd.Series(np.asarray(p, dtype=float), index=fs.index)
        p = p.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1, 1)
        if (p != 0).sum() < 60:
            continue
        turns = int((p.diff().abs() > 0).sum())
        if turns < 3 or turns > len(p) / 4:      # too static, or too churny
            continue
        key = f"{fname}|{mname}"
        if only is not None and key not in only:
            continue
        logics[key] = p
    return logics


def _sweep(d, a) -> int:
    """The same procedure across its parameters, and inverted.

    One setting losing proves little. A grid losing everywhere, with the
    inverse winning nowhere either, says the ranking carries no
    information in either direction -- which is a different and much
    stronger statement.
    """
    close = d["close"]
    logics = build_logics(d)
    nets = {k: daily_net(close, v) for k, v in logics.items()}
    idx = close.index
    print(f"{len(logics):,} logics\n")
    print(f"{'top':>5} {'review':>7} {'window':>7} {'pick best':>12} "
          f"{'pick worst':>12}")
    for top in (5, 10, 25, 50):
        for review in (30, 60, 120):
            for window in (90, 180, 360):
                row = []
                for invert in (False, True):
                    eq, start = 1.0, a.warmup
                    while start < len(idx) - 1:
                        end = min(start + review, len(idx) - 1)
                        w0 = max(0, start - window)
                        sc = []
                        for k, s2 in nets.items():
                            past = s2.iloc[w0:start]
                            if (past != 0).sum() < 20:
                                continue
                            tot = float((1 + past).prod() - 1)
                            sd = float(past.std())
                            sc.append((tot / sd if sd > 0 else 0.0, k))
                        sc.sort(reverse=not invert)
                        ch = [k for _, k in sc[:top]]
                        if ch:
                            seg = sum(nets[k].iloc[start:end] for k in ch) / len(ch)
                            eq *= float((1 + seg).prod())
                        start = end
                    row.append(eq - 1)
                print(f"{top:>5} {review:>7} {window:>7} "
                      f"{100*row[0]:>11.1f}% {100*row[1]:>11.1f}%")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--top", type=int, default=10,
                   help="how many currently-working logics to hold")
    p.add_argument("--review", type=int, default=60,
                   help="days between re-scoring")
    p.add_argument("--window", type=int, default=180,
                   help="trailing days a logic is judged on")
    p.add_argument("--warmup", type=int, default=260)
    p.add_argument("--invert", action="store_true",
                   help="hold the WORST trailing performers instead of the "
                        "best -- if picking winners loses, picking losers is "
                        "the thing to check next")
    p.add_argument("--sweep", action="store_true",
                   help="run the grid of top/review/window, both directions")
    a = p.parse_args(argv)

    d = load_daily()
    close = d["close"]
    if a.sweep:
        return _sweep(d, a)
    print(f"{len(d):,} daily bars, {d.index[0].date()} .. {d.index[-1].date()}")
    print(f"buy and hold {100*(close.iloc[-1]/close.iloc[0]-1):+.1f}%")

    logics = build_logics(d)
    nets = {k: daily_net(close, v) for k, v in logics.items()}
    print(f"{len(factors(d))} factors x {len(methods())} methods "
          f"-> {len(logics):,} usable logics\n")

    idx = close.index
    equity, held_log, picks = 1.0, [], []
    curve = pd.Series(0.0, index=idx)
    start = a.warmup
    while start < len(idx) - 1:
        end = min(start + a.review, len(idx) - 1)
        w0 = max(0, start - a.window)
        scored = []
        for k, s in nets.items():
            past = s.iloc[w0:start]
            if (past != 0).sum() < 20:
                continue
            tot = float((1 + past).prod() - 1)
            sd = float(past.std())
            scored.append((tot / sd if sd > 0 else 0.0, tot, k))
        scored.sort(reverse=not a.invert)
        chosen = [k for _, _, k in scored[:a.top]]
        picks.append((idx[start].date(), chosen[:3]))
        if chosen:
            seg = sum(nets[k].iloc[start:end] for k in chosen) / len(chosen)
            curve.iloc[start:end] = seg.values
            equity *= float((1 + seg).prod())
            held_log.append((idx[start].date(), len(chosen),
                             float((1 + seg).prod() - 1)))
        start = end

    live = curve.iloc[a.warmup:]
    print("=" * 74)
    print(f"LIVE SELECTION: top {a.top} by trailing {a.window}d, "
          f"re-scored every {a.review}d")
    print("=" * 74)
    print(f"{'from':>12} {'logics':>7} {'window net':>12}")
    for dt, n, ret in held_log:
        print(f"{str(dt):>12} {n:>7} {100*ret:>11.2f}%")
    tot = float((1 + live).prod() - 1)
    sd = float(live.std())
    sharpe = float(live.mean() / sd * np.sqrt(365)) if sd > 0 else 0.0
    wins = sum(1 for _, _, r in held_log if r > 0)
    print(f"\n  windows profitable : {wins}/{len(held_log)}")
    print(f"  total (1x)         : {100*tot:+.1f}%")
    print(f"  Sharpe             : {sharpe:.2f}")
    print(f"  max drawdown       : "
          f"{100*float(((1+live).cumprod()/(1+live).cumprod().cummax()-1).min()):+.1f}%")

    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    if tot > 0 and sharpe > 0.5 and wins > len(held_log) / 2:
        print(f"  Positive out of sample: {100*tot:+.1f}% at Sharpe {sharpe:.2f},")
        print(f"  profitable in {wins} of {len(held_log)} windows. Worth running.")
        print(f"  Wire it in with:  python run_bot.py --signals ensemble")
    else:
        print(f"  {100*tot:+.1f}% at Sharpe {sharpe:.2f}, {wins}/{len(held_log)}")
        print("  windows profitable. Selecting the currently-working logics")
        print("  does not carry forward on this sample -- the logic that led")
        print("  the trailing window is not the one that leads the next.")
        print("\n  Most-picked logics, in order of appearance:")
        seen = []
        for _, ch in picks:
            for k in ch[:1]:
                if k not in seen:
                    seen.append(k)
        for k in seen[:8]:
            print(f"    {k}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
