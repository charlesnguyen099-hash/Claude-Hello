"""Does "look at past data, mark every profitable move, then trade it" work?

This script implements EXACTLY the procedure requested:

  "In each period of 2025/2026, look at the trend, trade to the point
   where profit beats fees -- that's a potential trade. Aggregate all of
   them, then trade that on the same data and it wins. It CANNOT be
   negative."

Part A does precisely that and confirms the claim is true: replaying
trades that were selected because we already knew how they turned out
produces a ~100% win rate. That part is not in dispute.

Part B asks the only question that decides whether a bot can be built
from it: can those same trades be *identified in advance*, using only
information that existed before the entry? Part A's selector is allowed
to read future candles. A live bot cannot. Part B therefore tries hard
to reproduce Part A's picks from past-only features:

  B1. Exhaustive search over thousands of simple rule combinations,
      taking the single best performer in-sample and running it on the
      other year (2025 -> 2026 and 2026 -> 2025).
  B2. Logistic regression trained directly on Part A's own labels using
      27 backward-looking features, again cross-year.

Run:  python3 -m research.hindsight_proof
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bot import risk

FEE_ROUNDTRIP = 2 * risk.TAKER_FEE_PCT + 2 * risk.ASSUMED_SLIPPAGE_PCT

DATASETS = {
    "2026": "data/BTCUSDT_2026.csv",
    "2025": "data/BTCUSDT_2025.csv",
}


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    ts = "datetime" if "datetime" in df.columns else df.columns[0]
    df[ts] = pd.to_datetime(df[ts], utc=True)
    return df.rename(columns={ts: "timestamp"}).sort_values("timestamp").reset_index(drop=True)


# --------------------------------------------------------------------------
# PART A -- the requested procedure, with full knowledge of the future
# --------------------------------------------------------------------------
def perfect_hindsight(df: pd.DataFrame, horizon: int, equity: float = 10_000.0,
                      risk_pct: float = 0.01) -> dict:
    """Mark every bar whose *known* forward move beats fees, then trade it.

    Direction is chosen by looking `horizon` bars ahead -- i.e. exactly
    the "aggregate the profitable moves, then trade them" recipe. Every
    trade is exited at the known-best price, so the only way to lose is
    if the move failed to cover fees, and those bars are filtered out.
    """
    close = df["close"].to_numpy(float)
    n = len(close)
    fwd = np.full(n, np.nan)
    fwd[: n - horizon] = close[horizon:] / close[: n - horizon] - 1.0

    tradable = np.abs(fwd) > FEE_ROUNDTRIP
    direction = np.sign(fwd)

    eq = equity
    wins = losses = 0
    pnl_total = 0.0
    i = 0
    # Non-overlapping trades: enter, hold `horizon` bars, exit, repeat.
    while i < n - horizon:
        if not tradable[i]:
            i += 1
            continue
        gross = direction[i] * fwd[i]
        net = gross - FEE_ROUNDTRIP
        stake = eq * risk_pct / FEE_ROUNDTRIP  # size so 1 fee-unit == risk_pct of equity
        pnl = stake * net
        eq += pnl
        pnl_total += pnl
        wins += net > 0
        losses += net <= 0
        i += horizon

    total = wins + losses
    return {
        "horizon_bars": horizon,
        "trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(100 * wins / total, 4) if total else 0.0,
        "ending_equity": round(eq, 2),
        "return_pct": round(100 * (eq / equity - 1), 2),
    }


# --------------------------------------------------------------------------
# Backward-looking features -- everything a live bot could actually see
# --------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    high, low, vol = df["high"], df["low"], df["volume"]
    f = pd.DataFrame(index=df.index)

    for k in (1, 3, 5, 15, 30, 60, 240):
        f[f"ret{k}"] = close.pct_change(k)
    for k in (10, 20, 50, 100, 200):
        ema = close.ewm(span=k, adjust=False).mean()
        f[f"emadist{k}"] = close / ema - 1.0
    f["ema_fast_slow"] = (close.ewm(span=9, adjust=False).mean()
                          / close.ewm(span=21, adjust=False).mean() - 1.0)

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    f["rsi14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    tr = pd.concat([high - low, (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    f["atr_pct"] = atr / close
    f["atr_ratio"] = atr / atr.rolling(200).mean()

    for k in (30, 120):
        f[f"pos_in_range{k}"] = ((close - low.rolling(k).min())
                                 / (high.rolling(k).max() - low.rolling(k).min()))
    f["vol_ratio"] = vol / vol.rolling(60).mean()
    f["vol_ratio_long"] = vol / vol.rolling(480).mean()
    f["realized_vol"] = close.pct_change().rolling(60).std()
    f["realized_vol_ratio"] = f["realized_vol"] / close.pct_change().rolling(480).std()

    hour = df["timestamp"].dt.hour
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    f["dow"] = df["timestamp"].dt.dayofweek / 6.0

    f["streak"] = np.sign(close.diff()).groupby(
        (np.sign(close.diff()) != np.sign(close.diff()).shift()).cumsum()).cumcount() + 1
    return f


# --------------------------------------------------------------------------
# PART B1 -- exhaustive rule search, best in-sample rule tested out-of-sample
# --------------------------------------------------------------------------
def net_return_of_signal(df: pd.DataFrame, sig: np.ndarray, horizon: int) -> dict:
    """PnL of a +1/-1/0 signal array, entered at bar i, exited horizon bars later."""
    close = df["close"].to_numpy(float)
    n = len(close)
    fwd = np.full(n, np.nan)
    fwd[: n - horizon] = close[horizon:] / close[: n - horizon] - 1.0

    idx = np.flatnonzero((sig != 0) & ~np.isnan(fwd))
    if idx.size == 0:
        return {"trades": 0, "win_rate_pct": 0.0, "avg_net_pct": 0.0, "total_net_pct": 0.0}
    # Enforce non-overlap so results aren't inflated by 1000s of stacked bets.
    picked, last = [], -10**9
    for i in idx:
        if i >= last + horizon:
            picked.append(i)
            last = i
    picked = np.array(picked)
    nets = sig[picked] * fwd[picked] - FEE_ROUNDTRIP
    return {
        "trades": int(picked.size),
        "win_rate_pct": round(100 * float((nets > 0).mean()), 2),
        "avg_net_pct": round(100 * float(nets.mean()), 5),
        "total_net_pct": round(100 * float(nets.sum()), 2),
    }


def rule_search(train: pd.DataFrame, test: pd.DataFrame, horizon: int) -> dict:
    """Grid-search simple EMA-cross + RSI + volume rules on `train`, take the
    best, then run that exact rule on `test`."""
    ftr, fte = build_features(train), build_features(test)
    best = None

    for fast in (5, 9, 12, 21, 34):
        for slow in (21, 34, 55, 100, 200):
            if fast >= slow:
                continue
            cross_tr = np.sign(train["close"].ewm(span=fast, adjust=False).mean()
                               - train["close"].ewm(span=slow, adjust=False).mean())
            cross_te = np.sign(test["close"].ewm(span=fast, adjust=False).mean()
                               - test["close"].ewm(span=slow, adjust=False).mean())
            for rsi_lo, rsi_hi in ((0, 100), (30, 70), (40, 60), (20, 80)):
                for vmin in (0.0, 1.0, 1.5):
                    for flip in (1, -1):
                        ok_tr = ((ftr["rsi14"] > rsi_lo) & (ftr["rsi14"] < rsi_hi)
                                 & (ftr["vol_ratio"] > vmin)).to_numpy()
                        s_tr = flip * cross_tr.to_numpy() * ok_tr
                        r_tr = net_return_of_signal(train, s_tr, horizon)
                        if r_tr["trades"] < 30:
                            continue
                        if best is None or r_tr["total_net_pct"] > best["train"]["total_net_pct"]:
                            ok_te = ((fte["rsi14"] > rsi_lo) & (fte["rsi14"] < rsi_hi)
                                     & (fte["vol_ratio"] > vmin)).to_numpy()
                            s_te = flip * cross_te.to_numpy() * ok_te
                            best = {
                                "rule": (f"ema{fast}/{slow} rsi({rsi_lo},{rsi_hi}) "
                                         f"vol>{vmin} dir={'with' if flip == 1 else 'against'}"),
                                "train": r_tr,
                                "test": net_return_of_signal(test, s_te, horizon),
                            }
    return best


# --------------------------------------------------------------------------
# PART B2 -- logistic regression on Part A's own labels (numpy, no sklearn)
# --------------------------------------------------------------------------
def fit_logistic(X: np.ndarray, y: np.ndarray, epochs: int = 400, lr: float = 0.5,
                 l2: float = 1e-4) -> np.ndarray:
    X = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(X.shape[1])
    for _ in range(epochs):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        grad = X.T @ (p - y) / len(X) + l2 * np.r_[0, w[1:]]
        w -= lr * grad
    return w


def predict_logistic(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(np.column_stack([np.ones(len(X)), X]) @ w, -30, 30)))


def ml_test(train: pd.DataFrame, test: pd.DataFrame, horizon: int) -> dict:
    """Train on the hindsight labels of `train`, predict them on `test`."""
    def prep(df):
        f = build_features(df)
        close = df["close"].to_numpy(float)
        n = len(close)
        fwd = np.full(n, np.nan)
        fwd[: n - horizon] = close[horizon:] / close[: n - horizon] - 1.0
        y = (fwd > 0).astype(float)
        mask = ~f.isna().any(axis=1).to_numpy() & ~np.isnan(fwd) & (np.abs(fwd) > FEE_ROUNDTRIP)
        return f[mask].to_numpy(float), y[mask], fwd[mask]

    Xtr, ytr, _ = prep(train)
    Xte, yte, fte = prep(test)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

    w = fit_logistic(Xtr, ytr)
    ptr, pte = predict_logistic(w, Xtr), predict_logistic(w, Xte)

    sig = np.where(pte > 0.5, 1.0, -1.0)
    nets = sig * fte - FEE_ROUNDTRIP
    return {
        "features": Xtr.shape[1],
        "train_samples": len(ytr),
        "test_samples": len(yte),
        "train_accuracy_pct": round(100 * float(((ptr > 0.5) == ytr).mean()), 2),
        "test_accuracy_pct": round(100 * float(((pte > 0.5) == yte).mean()), 2),
        "test_trade_win_rate_pct": round(100 * float((nets > 0).mean()), 2),
        "test_avg_net_pct_per_trade": round(100 * float(nets.mean()), 5),
    }


def main() -> None:
    data = {k: load(v) for k, v in DATASETS.items()}
    horizons = [15, 60, 240]

    print("=" * 74)
    print("PART A -- the requested procedure (selector may read future candles)")
    print("=" * 74)
    print(f"round-trip cost assumed: {FEE_ROUNDTRIP*100:.4f}% per trade\n")
    for year, df in data.items():
        print(f"--- {year} ({len(df):,} 1m candles) ---")
        for h in horizons:
            r = perfect_hindsight(df, h)
            print(f"  hold {h:>3}m | trades {r['trades']:>6,} | win rate {r['win_rate_pct']:>7.3f}% "
                  f"| return {r['return_pct']:>+14,.2f}%")
        print()

    print("=" * 74)
    print("PART B1 -- best rule found in-sample, then run on the OTHER year")
    print("=" * 74)
    for tr_year, te_year in (("2025", "2026"), ("2026", "2025")):
        print(f"\n--- trained on {tr_year}, tested on {te_year} ---")
        for h in (60, 240):
            b = rule_search(data[tr_year], data[te_year], h)
            if b is None:
                continue
            print(f"  hold {h}m | best in-sample rule: {b['rule']}")
            print(f"      in-sample  ({tr_year}): trades {b['train']['trades']:>4} "
                  f"win {b['train']['win_rate_pct']:>5.2f}% total {b['train']['total_net_pct']:>+8.2f}%")
            print(f"      OUT-SAMPLE ({te_year}): trades {b['test']['trades']:>4} "
                  f"win {b['test']['win_rate_pct']:>5.2f}% total {b['test']['total_net_pct']:>+8.2f}%")

    print()
    print("=" * 74)
    print("PART B2 -- logistic regression trained on Part A's own labels")
    print("=" * 74)
    for tr_year, te_year in (("2025", "2026"), ("2026", "2025")):
        for h in (60, 240):
            r = ml_test(data[tr_year], data[te_year], h)
            print(f"\n  train {tr_year} -> test {te_year}, hold {h}m "
                  f"({r['features']} features, {r['train_samples']:,} train rows)")
            print(f"      in-sample accuracy : {r['train_accuracy_pct']:>6.2f}%")
            print(f"      OUT-SAMPLE accuracy: {r['test_accuracy_pct']:>6.2f}%   "
                  f"(50% = coin flip)")
            print(f"      OUT-SAMPLE trade win rate: {r['test_trade_win_rate_pct']:>6.2f}%  "
                  f"avg net/trade {r['test_avg_net_pct_per_trade']:>+.4f}%")


if __name__ == "__main__":
    main()
