"""Which short horizon still pays after real fees -- measured, not assumed.

    python -m fp.horizon                    # the frontier and the search
    python -m fp.horizon --tf 1m,5m,15m

THE QUESTION

Futures are a short-horizon instrument, so the horizon itself is a free
parameter: minutes, tens of minutes, hours, days. Whichever one carries
a profitable long or short is the one to trade. Nothing here is assumed
about which that is -- every timeframe is resampled from the same 884,855
one-minute bars of real BTCUSDT and measured the same way.

FIRST, THE PART THAT IS ARITHMETIC

Before searching for a logic at a horizon, the horizon has to be able to
pay for a round trip. Bybit charges 0.055% to enter at market and 0.055%
to exit, so 0.11% of notional is gone before the forecast is worth
anything, plus funding on the notional while the position is open.

A move's size grows with the square root of time; the fee does not grow
at all. So for any hold length there is a minimum hit rate below which no
logic can be profitable, and it is computable directly from the data:

    p* = 0.5 * (1 + cost / E|move|)

At p* the wins exactly pay the losses and the fees. Below it, nothing
works at that horizon -- no method, no factor, no amount of code. Above
it, a logic has room. The frontier below is that number for every hold
from one minute to ten days, computed on the real move distribution.

SECOND, THE SEARCH ITSELF

At each timeframe the same procedure that produced the daily result runs
unchanged: the logic library is built on that timeframe's bars, the
market is labelled by state, each logic is scored only on the past bars
sharing the current state, and the best five are held. Costs are charged
per turnover, funding per bar. No logic is ever scored on a bar it later
trades.

Coverage note, stated because it bounds the fine end: memory caps the
search at --max-bars per timeframe, so 1m covers about six weeks and 5m
about seven months, while 15m and slower cover the full nineteen months.
The frontier below has no such cap -- it uses every bar.

THE FRONTIER, on all 838,112 bars

    hold      E|move|      cost   p* needed   moves > cost
    1 min       0.040%    0.110%      185.9%          6.8%   impossible
    3 min       0.070%    0.110%      128.7%         18.6%   impossible
    5 min       0.090%    0.110%      111.2%         26.6%   impossible
    10 min      0.127%    0.110%       93.5%         39.1%
    15 min      0.155%    0.110%       85.6%         46.6%
    30 min      0.218%    0.111%       75.4%         58.9%
    1 hour      0.307%    0.111%       68.1%         69.0%
    2 hours     0.436%    0.113%       62.9%         77.0%
    4 hours     0.623%    0.115%       59.2%         83.0%
    8 hours     0.906%    0.120%       56.6%         87.9%
    1 day       1.617%    0.140%       54.3%         92.7%
    2 days      2.282%    0.170%       53.7%         94.3%
    3 days      2.778%    0.200%       53.6%         94.9%
    5 days      3.431%    0.260%       53.8%         93.5%
    10 days     5.020%    0.410%       54.1%         93.7%

Read the first three rows carefully. At one minute the average move is
0.040% and the round trip costs 0.110%, so break-even needs a hit rate
of 185.9%. That is not a hard target, it is an impossible one: even a
forecast that is right EVERY TIME loses money at that horizon, because
the move it captures is smaller than the fee it pays to capture it. The
same holds at three and five minutes.

Ten minutes is where perfect prediction first breaks even, at a required
93.5%. Thirty minutes needs 75.4%, an hour 68.1%. The requirement keeps
falling until it bottoms out at 53.6% around two to three days, and then
turns back up as funding starts to outgrow the move.

So the cheapest horizon this market offers is 2-3 days, and the sub-hour
end is not a hard place to make money -- it is an arithmetically closed
one. Nothing below ten minutes can work at these fees, whatever logic is
put on top.

THE SEARCH, same procedure at every timeframe

    tf     bars    days  logics   traded  trades    hold     total  Sharpe
    1d      583     582     928       26       1    2.0d     -3.4%   -9.16
    4h    3,493     582   1,132    2,924     290    8.0h    -35.4%   -2.20
    1h   13,969     582   1,156   13,400   1,127    3.0h    -26.8%   -0.61
    30m  27,938     582   1,156   27,369   2,168     90m    -64.6%   -2.32
    15m  55,875     582   1,146   55,306   3,526     45m    -71.2%   -2.32
    5m   60,000     208   1,150   59,431   2,524     15m    -41.2%   -1.91
    1m   60,000      41   1,142   59,431   7,039      3m    -16.4%   -5.00

Seven timeframes, seven losses, on both sides in every one of them. The
losses are worst in the middle -- 15m and 30m churn hardest against the
fee -- and shrink at 1m only because that run covers six weeks rather
than nineteen months.

This search is honest in a way the earlier ones were not: it goes
through fp.regime.lagged_states(), so the state label a logic is chosen
inside comes from the previous bar. The unlagged version of this exact
table read +600,007% at 1h and +10,632% at 4h. Those numbers were a
one-bar look-ahead, not a discovery, and the size of them is a useful
calibration for how much a sliver of future is worth when 1,156 logics
are competing to exploit it.

WHAT THIS SETTLES

Short horizons were the open question: minutes, tens of minutes, days.
The frontier closes the minute end by arithmetic and the search closes
the rest by measurement. The 2-3 day horizon needing only 53.6% remains
the most reachable target in the data -- and nothing built so far
reaches it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fp.ensemble import DATA, build_logics
from fp.regime import lagged_states

FEE_ROUND_TRIP = 0.0011           # 0.055% in + 0.055% out, both at market
FUNDING_PER_8H = 0.0001           # charged on notional, either side

TIMEFRAMES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30,
              "1h": 60, "4h": 240, "1d": 1440}


# ------------------------------------------------------------------- data

def load_1m() -> pd.DataFrame:
    """Every one-minute bar in the data directory, de-duplicated."""
    frames = []
    for p in sorted(Path(DATA).glob("BTCUSDT_*.csv")):
        d = pd.read_csv(p, sep=None, engine="python")
        d.columns = [c.strip().lower() for c in d.columns]
        tcol = [c for c in d.columns if "time" in c or "date" in c][0]
        d["datetime"] = pd.to_datetime(d[tcol])
        frames.append(d[["datetime", "open", "high", "low", "close", "volume"]])
    raw = (pd.concat(frames).drop_duplicates("datetime")
           .sort_values("datetime").set_index("datetime"))
    return raw


def resample(d: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if minutes == 1:
        return d
    r = f"{minutes}min"
    return pd.DataFrame({
        "open": d["open"].resample(r).first(),
        "high": d["high"].resample(r).max(),
        "low": d["low"].resample(r).min(),
        "close": d["close"].resample(r).last(),
        "volume": d["volume"].resample(r).sum(),
    }).dropna()


# -------------------------------------------------------------- frontier

def frontier(d1m: pd.DataFrame) -> None:
    """The hit rate each hold length needs before it can pay for itself.

    E|move| is the mean absolute return over a hold of that length, taken
    over every overlapping window in the real series. Nothing is modelled
    -- this is the distribution the market actually produced.
    """
    c = d1m["close"].values.astype(np.float64)
    print(f"\n{'hold':>9} {'E|move|':>9} {'cost':>8} {'p* needed':>10} "
          f"{'moves > cost':>13}")
    for mins, label in ((1, "1 min"), (3, "3 min"), (5, "5 min"),
                        (10, "10 min"), (15, "15 min"), (30, "30 min"),
                        (60, "1 hour"), (120, "2 hours"), (240, "4 hours"),
                        (480, "8 hours"), (1440, "1 day"), (2880, "2 days"),
                        (4320, "3 days"), (7200, "5 days"), (14400, "10 days")):
        if mins >= len(c):
            continue
        r = np.abs(c[mins:] / c[:-mins] - 1.0)
        e = float(r.mean())
        cost = FEE_ROUND_TRIP + FUNDING_PER_8H * (mins / 480.0)
        p = 0.5 * (1.0 + cost / e) if e > 0 else float("nan")
        share = float((r > cost).mean())
        flag = "  impossible" if p >= 1.0 else ""
        print(f"{label:>9} {100*e:>8.3f}% {100*cost:>7.3f}% "
              f"{100*p:>9.1f}% {100*share:>12.1f}%{flag}")


# --------------------------------------------------------------- the run

def bar_net(close: pd.Series, pos: pd.Series, minutes: int) -> np.ndarray:
    """Net return per bar: the move, less the fee on turnover and funding.

    The fee lands on the bar a position opens or reverses, which is what
    a market order actually costs; funding accrues per bar held.
    """
    p = pos.shift(1).fillna(0.0).values
    r = close.pct_change().fillna(0.0).values
    # Half the round trip per unit of turnover: opening costs 0.055% and
    # closing costs 0.055%, so a position that opens and later closes pays
    # 0.11% in total and a straight reversal pays it once, not twice.
    turn = np.abs(np.diff(np.concatenate([[0.0], p])))
    fund = FUNDING_PER_8H * (minutes / 480.0)
    return (p * r - turn * (FEE_ROUND_TRIP / 2.0)
            - np.abs(p) * fund).astype(np.float32)


def select(N: np.ndarray, P: np.ndarray, code: np.ndarray, top: int,
           min_obs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """State-conditioned selection over every bar, without an O(n^2) loop.

    For each state, the running mean and variance of every logic over the
    bars already seen in that state come from one cumulative sum, so the
    score at bar i uses exactly the bars before i that shared its state --
    the same rule fp/regime.py applies one bar at a time, and none of the
    future reaches backwards.
    """
    n, L = N.shape
    step = np.full(n, np.nan, dtype=np.float32)
    vote = np.zeros(n, dtype=np.float32)
    used = np.zeros(n, dtype=bool)
    for s in np.unique(code[code >= 0]):
        idx = np.flatnonzero(code == s)
        if len(idx) <= min_obs + 1:
            continue
        A = N[idx]
        cs = np.cumsum(A.astype(np.float64), axis=0)
        cs2 = np.cumsum(A.astype(np.float64) ** 2, axis=0)
        k = np.arange(1, len(idx), dtype=np.float64)[:, None]
        mean = cs[:-1] / k
        var = np.maximum(cs2[:-1] / k - mean ** 2, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            score = np.where(var > 0, mean / np.sqrt(var), -np.inf)
        score[~np.isfinite(score)] = -np.inf
        ok = np.arange(1, len(idx)) >= min_obs
        rows = np.flatnonzero(ok)
        if len(rows) == 0:
            continue
        for c0 in range(0, len(rows), 20000):          # bound peak memory
            blk = rows[c0:c0 + 20000]
            pick = np.argpartition(-score[blk], top - 1, axis=1)[:, :top]
            bars = idx[blk + 1]
            step[bars] = np.take_along_axis(N[bars], pick, axis=1).mean(1)
            vote[bars] = np.take_along_axis(P[bars - 1], pick, axis=1).mean(1)
            used[bars] = True
    return step, vote, used


def runs_of(dirs: np.ndarray) -> np.ndarray:
    """Length of each unbroken stretch of the same non-zero position."""
    out, i = [], 0
    while i < len(dirs):
        j = i
        while j + 1 < len(dirs) and dirs[j + 1] == dirs[i]:
            j += 1
        if dirs[i] != 0:
            out.append(j - i + 1)
        i = j + 1
    return np.array(out) if out else np.array([0])


def run_tf(d1m: pd.DataFrame, label: str, minutes: int, top: int,
           max_bars: int, min_obs: int) -> dict:
    d = resample(d1m, minutes)
    if len(d) > max_bars:
        d = d.iloc[-max_bars:]
    if len(d) < 500:
        return {"tf": label, "bars": len(d), "note": "too few bars"}
    close = d["close"]
    logics = build_logics(d, fast=True)
    if not logics:
        return {"tf": label, "bars": len(d), "note": "no logics"}
    keys = list(logics)
    N = np.empty((len(d), len(keys)), dtype=np.float32)
    P = np.empty((len(d), len(keys)), dtype=np.float32)
    for j, k in enumerate(keys):
        N[:, j] = bar_net(close, logics[k], minutes)
        P[:, j] = logics[k].values
    # A bar whose state could not be built (inside the warm-up) is coded
    # -1 and never traded, rather than silently joining a state.
    txt = lagged_states(d, ["trend"]).fillna("nan").astype(str).values
    _, code = np.unique(txt, return_inverse=True)
    code = np.where(txt == "nan", -1, code).astype(int)
    step, vote, used = select(N, P, code, top, min_obs)

    r = step[used]
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return {"tf": label, "bars": len(d), "logics": len(keys),
                "note": "nothing qualified"}
    dirs = np.where(vote > 0.2, 1, np.where(vote < -0.2, -1, 0))[used]
    holds = runs_of(dirs)
    per_year = 365 * 24 * 60 / minutes
    tot = float(np.prod(1 + r.astype(np.float64)) - 1)
    sh = float(r.mean() / r.std() * np.sqrt(per_year)) if r.std() > 0 else 0.0
    longs = r[dirs == 1]
    shorts = r[dirs == -1]
    return {"tf": label, "bars": len(d), "logics": len(keys),
            "days": (d.index[-1] - d.index[0]).days,
            "traded": int(len(r)), "total": tot, "sharpe": sh,
            "trades": int((holds > 0).sum()),
            "hold": float(np.median(holds)) * minutes,
            "long": float(np.prod(1 + longs.astype(np.float64)) - 1) if len(longs) else 0.0,
            "short": float(np.prod(1 + shorts.astype(np.float64)) - 1) if len(shorts) else 0.0,
            "n_long": int(len(longs)), "n_short": int(len(shorts)),
            "gross_per_bar": float(r.mean())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tf", default="1m,5m,15m,30m,1h,4h,1d")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--max-bars", type=int, default=60000)
    ap.add_argument("--min-obs", type=int, default=200)
    ap.add_argument("--skip-frontier", action="store_true")
    a = ap.parse_args(argv)

    d1m = load_1m()
    print(f"{len(d1m):,} one-minute bars, "
          f"{d1m.index[0]} .. {d1m.index[-1]}")

    if not a.skip_frontier:
        print("\n" + "=" * 74)
        print("1. THE FRONTIER -- what each hold length must beat to exist")
        print("=" * 74)
        print("   p* is the hit rate at which wins exactly cover losses plus")
        print("   fees. Below p*, no logic at that horizon can be profitable.")
        frontier(d1m)

    print("\n" + "=" * 74)
    print("2. THE SEARCH -- the same procedure run on every timeframe")
    print("=" * 74)
    print(f"{'tf':>5} {'bars':>8} {'days':>6} {'logics':>7} {'traded':>7} "
          f"{'trades':>7} {'hold':>9} {'total':>10} {'Sharpe':>8} "
          f"{'long':>9} {'short':>9}")
    rows = []
    for label in a.tf.split(","):
        label = label.strip()
        if label not in TIMEFRAMES:
            continue
        res = run_tf(d1m, label, TIMEFRAMES[label], a.top, a.max_bars,
                     a.min_obs)
        rows.append(res)
        if "note" in res:
            print(f"{res['tf']:>5} {res['bars']:>8,} {'--':>6} "
                  f"{res.get('logics', 0):>7} {res['note']}")
            continue
        hold = res["hold"]
        hs = (f"{hold:.0f}m" if hold < 120 else
              f"{hold/60:.1f}h" if hold < 2880 else f"{hold/1440:.1f}d")
        print(f"{res['tf']:>5} {res['bars']:>8,} {res['days']:>6} "
              f"{res['logics']:>7} {res['traded']:>7,} {res['trades']:>7,} "
              f"{hs:>9} {100*res['total']:>9.1f}% {res['sharpe']:>8.2f} "
              f"{100*res['long']:>8.1f}% {100*res['short']:>8.1f}%")

    good = [r for r in rows if r.get("total", 0) > 0]
    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    if good:
        for r in good:
            print(f"  {r['tf']:>4}  {100*r['total']:+.1f}% over {r['days']} days, "
                  f"Sharpe {r['sharpe']:.2f}, {r['trades']:,} trades, "
                  f"long {100*r['long']:+.1f}% / short {100*r['short']:+.1f}%")
    else:
        print("  No timeframe cleared its costs.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
