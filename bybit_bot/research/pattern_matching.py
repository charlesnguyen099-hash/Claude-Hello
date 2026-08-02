"""The requested procedure, implemented in general form.

  "Observe some candles. Find where a trade would have profited at max
   leverage after fees. Analyse WHY it was potential, derive the logic,
   code it into the bot, and from then on trade every similar situation.
   Do it for every profit size, miss nothing."

The specific numbers in that request (20 candles, 20 minutes, 10-20%)
were examples, so nothing here is hard-coded to them. The script sweeps:

  pattern length : 10, 20, 40 candles of history
  holding time   : 5, 15, 30, 60, 120, 240 minutes
  profit target  : 10%, 20% at max leverage, after fees
  direction      : short and long

For every combination it does exactly what was asked:

  1. Find every bar where the trade would have hit the profit target.
  2. Take the shape of the candles immediately before it as the pattern
     ("why it was potential"), normalised so that only the shape counts
     -- not the price level, not the volatility regime.
  3. Find the most similar situations elsewhere and check whether they
     were profitable too.

Step 3 is the test the whole idea depends on, and it is a measurement,
not an opinion. Every result is printed next to its BASE RATE -- the
share of all bars that were profitable anyway. The ratio between them is
the LIFT:

  lift  = 1.0  -> the pattern told us nothing; matching it is the same
                  as trading at random
  lift >> 1.0  -> the pattern genuinely predicts the move, and a bot
                  built on it would work

Library and query always come from DIFFERENT years for the out-of-sample
figure, because that is the only situation a live bot is ever in: the
pattern was learned in the past, the trade happens in a future the
pattern has never seen.

Run:  python3 -m research.pattern_matching
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bot import risk

MAX_LEVERAGE = risk.ABSOLUTE_MAX_LEVERAGE
FEE_ROUNDTRIP = 2 * (risk.TAKER_FEE_PCT + risk.ASSUMED_SLIPPAGE_PCT)

WINDOWS = [10, 20, 40]
HORIZONS = [5, 15, 30, 60, 120, 240]
LEVERAGED_TARGETS = [10.0, 20.0]
K_NEIGHBOURS = 50
MAX_LIBRARY = 800
MAX_QUERIES = 40_000
RNG = np.random.default_rng(0)

DATASETS = {"2026": "data/BTCUSDT_2026.csv", "2025": "data/BTCUSDT_2025.csv"}


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def build_patterns(df: pd.DataFrame, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Normalised shape of the `window` candles ending at each bar."""
    close = df["close"].to_numpy(float)
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    vol = df["volume"].to_numpy(float)
    n = len(close)

    rets = np.zeros(n)
    rets[1:] = close[1:] / close[:-1] - 1.0
    rng_pct = (high - low) / np.maximum(close, 1e-9)
    vol_ma = pd.Series(vol).rolling(window * 5, min_periods=window).mean().to_numpy()
    vol_rel = np.where(vol_ma > 0, vol / np.maximum(vol_ma, 1e-9), 1.0)

    idx = np.arange(window, n)
    win_idx = idx[:, None] - np.arange(window - 1, -1, -1)[None, :]

    r = rets[win_idx]
    r = r / np.maximum(r.std(axis=1, keepdims=True), 1e-9)
    g = rng_pct[win_idx]
    g = g / np.maximum(g.mean(axis=1, keepdims=True), 1e-9)
    v = np.clip(vol_rel[win_idx], 0, 5.0)

    feats = np.hstack([r, g, v]).astype(np.float32)
    ok = np.isfinite(feats).all(axis=1)
    return feats[ok], idx[ok]


def label(df: pd.DataFrame, bars: np.ndarray, horizon: int,
          leveraged_pct: float) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised: would a short / long opened at each bar and exited at
    the best price within `horizon` minutes have cleared the target?
    """
    close = df["close"].to_numpy(float)
    need_spot = leveraged_pct / 100.0 / MAX_LEVERAGE + FEE_ROUNDTRIP

    fut_low = (df["low"].rolling(horizon).min().shift(-horizon)).to_numpy()
    fut_high = (df["high"].rolling(horizon).max().shift(-horizon)).to_numpy()

    entry = close[bars]
    lo, hi = fut_low[bars], fut_high[bars]
    valid = np.isfinite(lo) & np.isfinite(hi)

    short_gain = np.where(valid, (entry - lo) / entry, np.nan)
    long_gain = np.where(valid, (hi - entry) / entry, np.nan)
    return (short_gain >= need_spot) & valid, (long_gain >= need_spot) & valid


def matched_rate(library: np.ndarray, queries: np.ndarray,
                 query_labels: np.ndarray, k: int) -> float:
    """Share of each library pattern's k most-similar situations that
    were also profitable. Chunked so memory stays bounded.
    """
    lib_sq = (library ** 2).sum(axis=1)[:, None]
    dists = np.full((len(library), k), np.inf, dtype=np.float32)
    labels = np.zeros((len(library), k), dtype=bool)
    rows = np.arange(len(library))[:, None]

    CH = 10_000
    for s in range(0, len(queries), CH):
        chunk = queries[s:s + CH]
        clab = query_labels[s:s + CH]
        d = (lib_sq + (chunk ** 2).sum(axis=1)[None, :]
             - 2.0 * library @ chunk.T).astype(np.float32)
        md = np.hstack([dists, d])
        ml = np.hstack([labels, np.broadcast_to(clab, d.shape)])
        order = np.argpartition(md, k - 1, axis=1)[:, :k]
        dists, labels = md[rows, order], ml[rows, order]
    return float(labels.mean())


def main() -> None:
    print(f"Max leverage {MAX_LEVERAGE}x, round-trip cost {FEE_ROUNDTRIP*100:.3f}%")
    print("Every number below is paired with the BASE RATE (how often that")
    print("trade won anyway). LIFT = matched / base. Lift ~1.00 means the")
    print("pattern carried no information.\n")

    data = {y: load(p) for y, p in DATASETS.items()}
    for y, df in data.items():
        print(f"  {y}: {len(df):,} candles")
    print()

    header = (f"{'win':>4s} {'horiz':>6s} {'targ':>5s} {'dir':>5s} | "
              f"{'lib':>5s}->{'qry':<5s} {'base%':>7s} {'matched%':>9s} {'lift':>6s}  note")
    print(header)
    print("-" * len(header))

    best_lifts = []
    for window in WINDOWS:
        pats = {y: build_patterns(df, window) for y, df in data.items()}
        for horizon in HORIZONS:
            for target in LEVERAGED_TARGETS:
                lab = {}
                for y, df in data.items():
                    f, bars = pats[y]
                    s, l = label(df, bars, horizon, target)
                    lab[y] = {"feats": f, "short": s, "long": l}

                for direction in ("short", "long"):
                    for lib_y in DATASETS:
                        opp = np.flatnonzero(lab[lib_y][direction])
                        if len(opp) < 50:
                            continue
                        if len(opp) > MAX_LIBRARY:
                            opp = RNG.choice(opp, MAX_LIBRARY, replace=False)
                        library = lab[lib_y]["feats"][opp]

                        q_y = next(y for y in DATASETS if y != lib_y)
                        qf = lab[q_y]["feats"]
                        ql = lab[q_y][direction]
                        if len(qf) > MAX_QUERIES:
                            sel = RNG.choice(len(qf), MAX_QUERIES, replace=False)
                            qf, ql = qf[sel], ql[sel]
                        if ql.mean() == 0:
                            continue

                        rate = matched_rate(library, qf, ql, K_NEIGHBOURS)
                        base = float(ql.mean())
                        lift = rate / base
                        best_lifts.append((lift, window, horizon, target, direction,
                                           lib_y, q_y, base, rate))
                        note = "<-- would be tradable" if lift >= 1.5 else ""
                        print(f"{window:4d} {horizon:6d} {target:5.0f} {direction:>5s} | "
                              f"{lib_y:>5s}->{q_y:<5s} {100*base:7.2f} {100*rate:9.2f} "
                              f"{lift:6.2f}  {note}")

    print("\n" + "=" * 78)
    if not best_lifts:
        print("No combination produced enough opportunities to test.")
        return
    best_lifts.sort(reverse=True)
    print("Strongest out-of-sample lifts found across the whole sweep:")
    for lift, w, h, t, d, ly, qy, base, rate in best_lifts[:8]:
        print(f"  lift {lift:5.2f}  window={w:>2d} horizon={h:>3d}m target={t:.0f}% "
              f"{d:5s} {ly}->{qy}  base {100*base:5.2f}% -> matched {100*rate:5.2f}%")
    top = best_lifts[0][0]
    print()
    if top < 1.2:
        print(f"Best lift in the entire sweep is {top:.2f}x. A pattern that actually")
        print("predicted the move would show a lift far above 1. At this level the")
        print("most-similar past situations are no more likely to be profitable than")
        print("a bar picked at random, so 'trade every similar situation' selects")
        print("trades at the base rate -- and the base rate loses money after fees.")
    else:
        print(f"Best lift is {top:.2f}x -- worth building into the strategy and")
        print("re-testing inside the full engine (sizing, stops, fees) before use.")


if __name__ == "__main__":
    main()
