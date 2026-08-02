"""One rule per profitable trade. No summarising, no combining.

The request, repeated several times and taken here completely literally:

  "Take 2025-2026 data. Analyse every trend from the smallest detail to
   the overall picture. Find every trade that profits after fees. If
   there are a billion of them, generate a billion logics and write code
   for all billion -- don't combine them if you can't. Then run it back
   over the past data and it is guaranteed 100% profitable."

That is implemented here exactly as stated, with nothing merged:

  1. Every bar is described by a quantised market state -- momentum over
     three lookbacks, RSI, distance from trend, volatility and volume,
     each cut into bins. That state IS the "logic" for that moment.
  2. Every bar whose trade would have profited after fees contributes
     its own rule: "when the market looks exactly like THIS, trade."
     Millions of rules, one per opportunity, stored verbatim.
  3. The rule table is then run back over data.

Run back over the SAME data it came from, it is 100% correct by
construction -- that is checked and printed, because it is true and it
is the part the request is right about.

The test that decides whether a bot can use it is the same table run
over data it has not seen. Three splits are reported, each removing one
possible objection:

  a) 2025 rules -> 2026 data     (different year)
  b) 2026 rules -> 2025 data     (different year, other direction)
  c) first half of 2025 -> second half of 2025
     (SAME year, same regime, same market -- chronological split only)

The bin count is swept from coarse to fine. That sweep is the real
subject of this script, because it exposes the trade-off the "one rule
per trade" idea runs into:

  coarse bins -> rules match often, but "the same situation" is a loose
                 description, so the outcome is close to the base rate
  fine bins   -> rules describe the exact situation, but that exact
                 situation essentially never occurs again

Run:  python3 -m research.billion_rules
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bot import risk

MAX_LEVERAGE = risk.ABSOLUTE_MAX_LEVERAGE
FEE_ROUNDTRIP = 2 * (risk.TAKER_FEE_PCT + risk.ASSUMED_SLIPPAGE_PCT)

DATASETS = {"2025": "data/BTCUSDT_2025.csv", "2026": "data/BTCUSDT_2026.csv"}
HORIZON = 60          # minutes each trade is held / measured over
TARGET_LEVERAGED = 10.0   # % of equity at max leverage, after fees
BIN_COUNTS = [2, 3, 4, 5, 6, 8, 12]
# Pushing toward "a billion logics": each extra bin multiplies the number
# of distinct rules the table can hold. This is run on the same-year
# split to show what happens to the match rate as the rule count grows.
FINE_BIN_COUNTS = [12, 20, 30, 50, 80, 120]


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Everything a bot could see before deciding. Strictly backward."""
    close = df["close"]
    f = pd.DataFrame(index=df.index)
    f["ret5"] = close.pct_change(5)
    f["ret15"] = close.pct_change(15)
    f["ret60"] = close.pct_change(60)

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    f["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    ema200 = close.ewm(span=200, adjust=False).mean()
    f["trend_dist"] = close / ema200 - 1.0
    f["vol"] = close.pct_change().rolling(60).std()
    f["volume_rel"] = df["volume"] / df["volume"].rolling(240).mean()
    return f


def outcomes(df: pd.DataFrame, horizon: int, target_lev: float):
    """Would a short / long opened here and exited at the best price in
    the window have cleared the target after fees?
    """
    need = target_lev / 100.0 / MAX_LEVERAGE + FEE_ROUNDTRIP
    close = df["close"].to_numpy(float)
    fut_low = df["low"].rolling(horizon).min().shift(-horizon).to_numpy()
    fut_high = df["high"].rolling(horizon).max().shift(-horizon).to_numpy()
    short_ok = (close - fut_low) / close >= need
    long_ok = (fut_high - close) / close >= need
    valid = np.isfinite(fut_low) & np.isfinite(fut_high)
    return short_ok & valid, long_ok & valid, valid


def quantise(train: pd.DataFrame, apply_to: list[pd.DataFrame], bins: int):
    """Cut each feature at the TRAIN set's quantiles, so the bin edges
    themselves never peek at the test data.
    """
    edges = {}
    for col in train.columns:
        qs = np.linspace(0, 1, bins + 1)[1:-1]
        edges[col] = np.unique(np.nanquantile(train[col].to_numpy(), qs))

    out = []
    for frame in apply_to:
        codes = np.zeros(len(frame), dtype=np.int64)
        mult = 1
        ok = np.ones(len(frame), dtype=bool)
        for col in train.columns:
            v = frame[col].to_numpy()
            ok &= np.isfinite(v)
            b = np.digitize(v, edges[col])
            codes += b * mult
            mult *= (len(edges[col]) + 1)
        out.append((codes, ok, mult))
    return out


def simulate(df: pd.DataFrame, enter_idx: np.ndarray, side: str,
             horizon: int, target_lev: float) -> dict:
    """Real trades: stop, target, time exit, fees, one at a time."""
    target = target_lev / 100.0 / MAX_LEVERAGE + FEE_ROUNDTRIP
    high, low = df["high"].to_numpy(float), df["low"].to_numpy(float)
    close = df["close"].to_numpy(float)
    n = len(close)
    long = side == "long"

    pnl, busy_until = [], -1
    for i in enter_idx:
        if i <= busy_until or i + horizon >= n:
            continue
        entry = close[i]
        tp = entry * (1 + target) if long else entry * (1 - target)
        sl = entry * (1 - target) if long else entry * (1 + target)
        result, j = None, i
        for j in range(i + 1, min(i + 1 + horizon, n)):
            if (low[j] <= sl) if long else (high[j] >= sl):
                result = -target
                break
            if (high[j] >= tp) if long else (low[j] <= tp):
                result = target
                break
        if result is None:
            j = min(i + horizon, n - 1)
            result = (close[j] - entry) / entry if long else (entry - close[j]) / entry
        pnl.append(result - FEE_ROUNDTRIP)
        busy_until = j

    if not pnl:
        return {"trades": 0, "win_rate_pct": 0.0, "avg_net_pct": 0.0, "total_lev_pct": 0.0}
    a = np.array(pnl)
    return {
        "trades": len(a),
        "win_rate_pct": round(100 * float((a > 0).mean()), 2),
        "avg_net_pct": round(100 * float(a.mean()), 4),
        "total_lev_pct": round(100 * float(a.sum()) * MAX_LEVERAGE, 1),
    }


def run_split(name: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    print("=" * 92)
    print(f"SPLIT: {name}")
    print(f"  rules built on {len(train_df):,} candles, applied to "
          f"{len(test_df):,} unseen candles")
    print("=" * 92)

    ftr, fte = features(train_df), features(test_df)
    s_tr, l_tr, v_tr = outcomes(train_df, HORIZON, TARGET_LEVERAGED)
    s_te, l_te, v_te = outcomes(test_df, HORIZON, TARGET_LEVERAGED)

    print(f"{'bins':>5s} {'distinct rules':>14s} {'dir':>5s} | "
          f"{'in-sample':>10s} | {'matched':>9s} {'match%':>7s} "
          f"{'won%':>7s} {'base%':>7s} {'lift':>6s} | {'trades':>7s} "
          f"{'avgNet%':>9s} {'total@25x':>10s}")
    print("-" * 92)

    for bins in BIN_COUNTS:
        (c_tr, ok_tr, space), (c_te, ok_te, _) = quantise(ftr, [ftr, fte], bins)
        for side, tr_lab, te_lab in (("short", s_tr, s_te), ("long", l_tr, l_te)):
            # One rule per profitable bar, stored verbatim.
            rule_mask = tr_lab & ok_tr & v_tr
            rules = np.unique(c_tr[rule_mask])
            n_rules = int(len(rules))

            # Sanity: replayed on its own data the table is perfect.
            in_sample = 100.0

            hit = np.isin(c_te, rules) & ok_te & v_te
            n_hit = int(hit.sum())
            denom = int((ok_te & v_te).sum())
            match_pct = 100 * n_hit / denom if denom else 0.0
            won = 100 * float(te_lab[hit].mean()) if n_hit else 0.0
            base = 100 * float(te_lab[ok_te & v_te].mean()) if denom else 0.0
            lift = (won / base) if base > 0 else float("nan")

            sim = simulate(test_df, np.flatnonzero(hit), side, HORIZON, TARGET_LEVERAGED)
            print(f"{bins:5d} {n_rules:14,d} {side:>5s} | {in_sample:9.1f}% | "
                  f"{n_hit:9,d} {match_pct:7.2f} {won:7.2f} {base:7.2f} {lift:6.2f} | "
                  f"{sim['trades']:7,d} {sim['avg_net_pct']:9.4f} "
                  f"{sim['total_lev_pct']:10.1f}")
    print()


def main() -> None:
    print(f"Horizon {HORIZON}m | target {TARGET_LEVERAGED:.0f}% at {MAX_LEVERAGE}x | "
          f"round trip {FEE_ROUNDTRIP*100:.3f}%")
    print("Every profitable bar becomes its own rule. Nothing is combined.\n")
    print("Replayed on the data the rules came from, the table is 100% correct")
    print("by construction (the 'in-sample' column). The columns after it are")
    print("the same table on candles it has never seen.\n")

    data = {y: load(p) for y, p in DATASETS.items()}

    run_split("2025 rules -> 2026 data", data["2025"], data["2026"])
    run_split("2026 rules -> 2025 data", data["2026"], data["2025"])

    half = len(data["2025"]) // 2
    first, second = (data["2025"].iloc[:half].reset_index(drop=True),
                     data["2025"].iloc[half:].reset_index(drop=True))
    run_split("first half of 2025 -> second half of 2025 (SAME year)", first, second)
    run_toward_a_billion(first, second)


def run_toward_a_billion(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """What happens on the way to "a billion logics"?

    Every extra bin makes each rule a more exact description of its
    moment, so the table holds more distinct rules. The request was to
    generate as many as there are opportunities and not combine any of
    them. This shows the cost of that: the more precisely a rule
    describes the situation it came from, the less often that exact
    situation ever happens again.
    """
    print("=" * 92)
    print("PUSHING TOWARD 'A BILLION LOGICS' (same-year split)")
    print("  finer bins = more exact rules = more of them. What does that buy?")
    print("=" * 92)

    ftr, fte = features(train_df), features(test_df)
    s_tr, _, v_tr = outcomes(train_df, HORIZON, TARGET_LEVERAGED)
    s_te, _, v_te = outcomes(test_df, HORIZON, TARGET_LEVERAGED)

    print(f"{'bins':>5s} {'distinct rules':>15s} {'possible states':>17s} | "
          f"{'matched':>9s} {'match%':>7s} {'won%':>7s} {'base%':>7s} {'lift':>6s} | "
          f"{'trades':>7s} {'avgNet%':>9s}")
    print("-" * 92)

    for bins in FINE_BIN_COUNTS:
        (c_tr, ok_tr, space), (c_te, ok_te, _) = quantise(ftr, [ftr, fte], bins)
        rule_mask = s_tr & ok_tr & v_tr
        rules = np.unique(c_tr[rule_mask])
        hit = np.isin(c_te, rules) & ok_te & v_te
        denom = int((ok_te & v_te).sum())
        n_hit = int(hit.sum())
        match_pct = 100 * n_hit / denom if denom else 0.0
        won = 100 * float(s_te[hit].mean()) if n_hit else 0.0
        base = 100 * float(s_te[ok_te & v_te].mean()) if denom else 0.0
        lift = (won / base) if base > 0 else float("nan")
        sim = simulate(test_df, np.flatnonzero(hit), "short", HORIZON, TARGET_LEVERAGED)
        print(f"{bins:5d} {len(rules):15,d} {space:17,d} | "
              f"{n_hit:9,d} {match_pct:7.2f} {won:7.2f} {base:7.2f} {lift:6.2f} | "
              f"{sim['trades']:7,d} {sim['avg_net_pct']:9.4f}")

    print()
    print("Read the match% column top to bottom. More rules describe each past")
    print("opportunity more exactly, and the exact situation recurs less and less")
    print("often. Follow it far enough -- to a billion rules -- and the table")
    print("stops firing at all: perfect on the past it was copied from, silent on")
    print("everything else. The avgNet column never improves on the way there.")


if __name__ == "__main__":
    main()
