"""Which factors actually move a trade's outcome, and does the exhaustion
setup work?

Two questions, both raised directly:

  1. "Look at which factors affect getting the trade direction right and
     how much profit to take -- price differential, buying pressure,
     direction, candle count, and so on. My example is one of a hundred
     thousand such factors."

  2. The example itself: "volume rising ~10% per candle while price falls
     ~11%, seven candles of that, then candle eight has volume collapse to
     nothing, then it jerks up with price and volume +30%. Short the first
     part, long the second."

PART 1 ranks a library of ~100 factors by how well each one predicts the
REALISED net return of a real trade -- stop, target, time exit, fees --
and, critically, whether that relationship keeps the same sign on a year
the factor was not measured on. A factor that predicts in 2025 and
predicts with the opposite sign in 2026 is worse than useless; it is a
trap. Sign stability is the column that matters, not raw strength.

PART 2 encodes the exhaustion sequence as an explicit state machine
rather than as continuous features, because that is what the description
actually is: a RUN of declining candles, with volume BUILDING through
it, then a volume COLLAPSE, then a reversal bar on heavy volume. None of
the earlier work here tested sequential structure like that -- the
gradient-boosted model saw volume rhythm as twenty independent numbers,
which cannot express "seven in a row, then a stop, then a snap back".
The parameters are swept rather than fixed at the example's exact
numbers, and the short leg and the long leg are measured separately, so
each is judged on its own.

Run:  python3 -m research.factor_study
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bot import risk

MAX_LEVERAGE = risk.ABSOLUTE_MAX_LEVERAGE
FEE_ROUNDTRIP = 2 * (risk.TAKER_FEE_PCT + risk.ASSUMED_SLIPPAGE_PCT)
DATASETS = {"2025": "data/BTCUSDT_2025.csv", "2026": "data/BTCUSDT_2026.csv"}

HORIZON = 120
STOP_ATR = 2.0
TARGET_ATR = 4.0


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def block_atr(df: pd.DataFrame, block: int = 15, span: int = 14) -> np.ndarray:
    high = df["high"].rolling(block).max()
    low = df["low"].rolling(block).min()
    prev = df["close"].shift(block)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(span=span, adjust=False).mean().to_numpy()


def realised_net(df: pd.DataFrame, side: str, atr: np.ndarray,
                 horizon: int = HORIZON) -> np.ndarray:
    """Net return of a real trade opened at each bar. Stop-first on a bar
    whose range covers both levels; fees charged either way."""
    c = df["close"].to_numpy(float)
    long = side == "long"
    stop_d, targ_d = STOP_ATR * atr, TARGET_ATR * atr
    fut_low = df["low"].rolling(horizon).min().shift(-horizon).to_numpy()
    fut_high = df["high"].rolling(horizon).max().shift(-horizon).to_numpy()
    fut_close = df["close"].shift(-horizon).to_numpy()

    if long:
        hit_sl, hit_tp = fut_low <= c - stop_d, fut_high >= c + targ_d
        timeout = (fut_close - c) / c
    else:
        hit_sl, hit_tp = fut_high >= c + stop_d, fut_low <= c - targ_d
        timeout = (c - fut_close) / c

    gross = np.where(hit_sl, -stop_d / c, np.where(hit_tp, targ_d / c, timeout))
    net = gross - FEE_ROUNDTRIP
    net[~np.isfinite(fut_close) | ~np.isfinite(atr) | (atr <= 0)] = np.nan
    return net


# ---------------------------------------------------------------------------
# PART 1 -- the factor library
# ---------------------------------------------------------------------------
def factor_library(df: pd.DataFrame) -> dict[str, np.ndarray]:
    o, h, l, c, v = (df[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    n, eps = len(c), 1e-12
    ret = np.zeros(n)
    ret[1:] = c[1:] / c[:-1] - 1.0
    sret, sv = pd.Series(ret), pd.Series(v)
    up, down = (ret > 0).astype(float), (ret < 0).astype(float)
    f: dict[str, np.ndarray] = {}

    # Price differential over many spans -- "độ chênh lệch giá"
    for k in (1, 3, 5, 10, 15, 30, 60, 120, 240, 480):
        f[f"pricediff_{k}"] = pd.Series(c).pct_change(k).to_numpy()
        f[f"range_{k}"] = ((pd.Series(h).rolling(k).max() - pd.Series(l).rolling(k).min())
                           / np.maximum(c, eps)).to_numpy()

    # Buying pressure -- "sức mua"
    for k in (5, 15, 30, 60, 120):
        f[f"buypressure_{k}"] = (pd.Series(v * up).rolling(k).sum()
                                 / np.maximum(sv.rolling(k).sum(), eps)).to_numpy()
        f[f"closepos_{k}"] = ((c - pd.Series(l).rolling(k).min())
                              / np.maximum(pd.Series(h).rolling(k).max()
                                           - pd.Series(l).rolling(k).min(), eps)).to_numpy()
        f[f"upcandle_share_{k}"] = pd.Series(up).rolling(k).mean().to_numpy()

    # Volume level and volume rhythm
    for k in (5, 15, 30, 60, 240):
        f[f"vol_rel_{k}"] = (v / np.maximum(sv.rolling(k).mean(), eps))
        f[f"vol_trend_{k}"] = (sv.rolling(k).mean()
                               / np.maximum(sv.rolling(k * 4).mean(), eps)).to_numpy()
    vr = np.ones(n)
    vr[1:] = v[1:] / np.maximum(v[:-1], eps)
    for k in (3, 5, 10, 20):
        f[f"vol_growth_{k}"] = pd.Series(np.log(np.maximum(vr, eps))).rolling(k).mean().to_numpy()
        f[f"vol_accel_{k}"] = (sv.rolling(k).mean()
                               / np.maximum(sv.shift(k).rolling(k).mean(), eps)).to_numpy()

    # Direction persistence and candle counts -- "chiều", "số nến"
    sign = np.sign(ret)
    same = pd.Series(sign != pd.Series(sign).shift()).cumsum()
    f["run_length"] = pd.Series(np.ones(n)).groupby(same).cumsum().to_numpy()
    f["run_direction"] = sign
    f["run_signed"] = f["run_length"] * sign
    for k in (5, 10, 20, 60):
        f[f"down_count_{k}"] = pd.Series(down).rolling(k).sum().to_numpy()
        f[f"dir_consistency_{k}"] = pd.Series(sign).rolling(k).mean().to_numpy()

    # Candle anatomy
    body = (c - o) / np.maximum(h - l, eps)
    upper = (h - np.maximum(c, o)) / np.maximum(h - l, eps)
    lower = (np.minimum(c, o) - l) / np.maximum(h - l, eps)
    f["body"], f["upper_wick"], f["lower_wick"] = body, upper, lower
    for k in (5, 15, 30):
        f[f"body_mean_{k}"] = pd.Series(body).rolling(k).mean().to_numpy()
        f[f"upper_wick_mean_{k}"] = pd.Series(upper).rolling(k).mean().to_numpy()
        f[f"lower_wick_mean_{k}"] = pd.Series(lower).rolling(k).mean().to_numpy()

    # Volatility regime
    for k in (15, 60, 240):
        f[f"vola_{k}"] = sret.rolling(k).std().to_numpy()
        f[f"vola_ratio_{k}"] = (sret.rolling(k).std()
                                / np.maximum(sret.rolling(k * 4).std(), eps)).to_numpy()

    # Trend context
    for span in (60, 240, 1440):
        ema = pd.Series(c).ewm(span=span, adjust=False).mean().to_numpy()
        f[f"dist_ema_{span}"] = c / np.maximum(ema, eps) - 1.0

    # Price/volume interaction
    for k in (15, 60):
        f[f"corr_pv_{k}"] = sret.rolling(k).corr(sv).to_numpy()
        f[f"vwret_{k}"] = (pd.Series(ret * (v / np.maximum(sv.rolling(k).mean(), eps)))
                           .rolling(k).sum().to_numpy())
    return f


def rank_factors(data: dict[str, pd.DataFrame], side: str) -> None:
    atr = {y: block_atr(df) for y, df in data.items()}
    y = {yr: realised_net(df, side, atr[yr]) for yr, df in data.items()}
    facs = {yr: factor_library(df) for yr, df in data.items()}
    names = sorted(facs["2025"].keys())

    rows = []
    for name in names:
        ics = {}
        for yr in DATASETS:
            a, b = facs[yr][name], y[yr]
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 10_000:
                ics[yr] = np.nan
                continue
            ics[yr] = float(np.corrcoef(a[m], b[m])[0, 1])
        if not all(np.isfinite(v) for v in ics.values()):
            continue
        i25, i26 = ics["2025"], ics["2026"]
        stable = (i25 > 0) == (i26 > 0)
        rows.append((name, i25, i26, stable, min(abs(i25), abs(i26))))

    rows.sort(key=lambda r: -r[4])
    print(f"\n  Top factors for {side.upper()} by weakest-year strength")
    print(f"  (IC = correlation with realised net trade return)")
    print(f"    {'factor':24s} {'IC 2025':>9s} {'IC 2026':>9s}  {'same sign?':>10s}")
    print("    " + "-" * 56)
    for name, i25, i26, stable, _ in rows[:12]:
        print(f"    {name:24s} {i25:+9.4f} {i26:+9.4f}  "
              f"{'yes' if stable else 'NO -- flips':>10s}")

    n_stable = sum(1 for r in rows if r[3])
    strong = [r for r in rows if r[3] and r[4] >= 0.02]
    print(f"\n    {n_stable}/{len(rows)} factors keep the same sign on both years")
    print(f"    {len(strong)} of those reach |IC| >= 0.02 on BOTH years")
    if strong:
        for name, i25, i26, _, _ in strong[:8]:
            print(f"      {name:24s} {i25:+.4f} / {i26:+.4f}")


# ---------------------------------------------------------------------------
# PART 2 -- the exhaustion sequence, as an explicit state machine
# ---------------------------------------------------------------------------
def find_exhaustion(df: pd.DataFrame, run_len: int, vol_build: float,
                    collapse_ratio: float, snap_vol: float) -> tuple[np.ndarray, np.ndarray]:
    """Detect: `run_len` declining candles with volume building, then a
    volume collapse, then a reversal bar on heavy volume.

    Returns (short_entry_idx, long_entry_idx). The short leg enters at the
    end of the confirmed decline run; the long leg enters on the snap-back
    bar. Both use only bars that have already closed.
    """
    c = df["close"].to_numpy(float)
    v = df["volume"].to_numpy(float)
    n = len(c)
    ret = np.zeros(n)
    ret[1:] = c[1:] / c[:-1] - 1.0
    vol_ma = pd.Series(v).rolling(240, min_periods=60).mean().to_numpy()

    shorts, longs = [], []
    i = run_len + 1
    while i < n - 2:
        window = slice(i - run_len, i)
        if not np.all(ret[window] < 0):            # a genuine run of red candles
            i += 1
            continue
        vols = v[window]
        # Volume building through the decline.
        if vols[-1] < vols[0] * (1 + vol_build):
            i += 1
            continue
        shorts.append(i - 1)                        # short at the end of the run

        # Now look for the collapse then the snap-back, within a few bars.
        for j in range(i, min(i + 5, n - 1)):
            if not np.isfinite(vol_ma[j]) or vol_ma[j] <= 0:
                break
            collapsed = v[j] < collapse_ratio * vol_ma[j]
            if not collapsed:
                continue
            for k in range(j + 1, min(j + 4, n - 1)):
                if ret[k] > 0 and v[k] > snap_vol * vol_ma[k]:
                    longs.append(k)
                    break
            break
        i += run_len
    return np.array(shorts, dtype=int), np.array(longs, dtype=int)


def score(df: pd.DataFrame, idx: np.ndarray, side: str, atr: np.ndarray) -> dict:
    if len(idx) == 0:
        return {"trades": 0}
    y = realised_net(df, side, atr)
    vals = y[idx]
    vals = vals[np.isfinite(vals)]
    if len(vals) < 15:
        return {"trades": len(vals)}
    se = vals.std(ddof=1) / np.sqrt(len(vals))
    return {
        "trades": len(vals),
        "avg_net_pct": round(100 * float(vals.mean()), 4),
        "win_rate_pct": round(100 * float((vals > 0).mean()), 2),
        "t_stat": round(float(vals.mean() / se) if se > 0 else 0.0, 2),
    }


def exhaustion_study(data: dict[str, pd.DataFrame]) -> None:
    print("\n" + "=" * 88)
    print("PART 2 -- the exhaustion sequence: red run with volume building,")
    print("          volume collapse, then a snap-back on heavy volume")
    print("=" * 88)
    print("Short leg enters at the end of the run; long leg enters on the snap-back.")
    print("Parameters swept around the example rather than fixed at its exact numbers.\n")

    atr = {y: block_atr(df) for y, df in data.items()}
    print(f"{'run':>4s} {'volbuild':>9s} {'collapse':>9s} {'snap':>5s} {'leg':>6s} | "
          + " | ".join(f"{y+' trades':>10s} {'win%':>6s} {'net%':>8s} {'t':>6s}"
                       for y in DATASETS))
    print("-" * 88)

    both_positive = []
    for run_len in (5, 7, 9):
        for vol_build in (0.0, 0.10, 0.30):
            for collapse in (0.5, 0.8):
                for snap in (1.3, 2.0):
                    res = {}
                    for yr, df in data.items():
                        s_idx, l_idx = find_exhaustion(df, run_len, vol_build, collapse, snap)
                        res[yr] = {"short": score(df, s_idx, "short", atr[yr]),
                                   "long": score(df, l_idx, "long", atr[yr])}
                    for leg in ("short", "long"):
                        cells = [res[y][leg] for y in DATASETS]
                        if any(c.get("trades", 0) < 15 for c in cells):
                            continue
                        line = (f"{run_len:4d} {vol_build:9.2f} {collapse:9.2f} "
                                f"{snap:5.1f} {leg:>6s} | ")
                        line += " | ".join(
                            f"{c['trades']:10d} {c['win_rate_pct']:6.2f} "
                            f"{c['avg_net_pct']:8.4f} {c['t_stat']:6.2f}" for c in cells)
                        print(line)
                        if all(c["avg_net_pct"] > 0 for c in cells):
                            both_positive.append(
                                (run_len, vol_build, collapse, snap, leg, cells))

    print("\n" + "=" * 88)
    if both_positive:
        print("Configurations profitable after fees on BOTH years:")
        for run_len, vb, col, snap, leg, cells in both_positive:
            print(f"  run={run_len} volbuild={vb:.2f} collapse={col:.2f} snap={snap:.1f} "
                  f"{leg}: " + ", ".join(
                      f"{c['avg_net_pct']:+.4f}% (t={c['t_stat']:+.2f}, {c['trades']} trades)"
                      for c in cells))
        print("\nNext step for any of these: wire into bot/strategy.py and re-run the")
        print("full engine with sizing and leverage before believing it.")
    else:
        print("No configuration of the exhaustion setup was profitable after fees")
        print("on both years.")


def main() -> None:
    print(f"Realised-net labels: stop {STOP_ATR} ATR, target {TARGET_ATR} ATR, "
          f"{HORIZON}m max hold, {FEE_ROUNDTRIP*100:.3f}% round trip")
    data = {y: load(p) for y, p in DATASETS.items()}

    print("\n" + "=" * 88)
    print("PART 1 -- which factors actually predict a trade's realised outcome")
    print("=" * 88)
    print("A factor is only usable if its sign holds on a year it was not measured")
    print("on. One that predicts in 2025 and predicts the OPPOSITE in 2026 is worse")
    print("than no factor at all.")
    for side in ("short", "long"):
        rank_factors(data, side)

    exhaustion_study(data)


if __name__ == "__main__":
    main()
