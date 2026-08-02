"""Should we sit through the dip, dodge it, or trade it?

Request behind this: "don't tie capital up too long -- if a LONG signal
fires but the market first drops (say 15-20% measured at max leverage)
before the long works out, then instead of longing from the start,
SHORT the drop for a small profit first, then LONG for the bigger one."

The idea is sound as trade management. The catch is *when you know*.
Knowing at signal time that price will dip first and rally second means
knowing the future -- the same lookahead that research/hindsight_proof.py
shows cannot be reproduced from past-only data. So this script does not
implement the clairvoyant version. It implements the three things a live
bot can actually do at the moment a signal fires, and measures them:

  A. IMMEDIATE   -- current behaviour: enter in the signal direction now.
  B. WAIT        -- if very-short-term momentum is running *against* the
                    signal, don't enter yet; wait (up to a deadline) for
                    it to stop, then enter. Dodges the dip, gets a
                    better price, and frees the capital in the meantime.
  C. CAPTURE     -- the literal request: if short-term momentum is
                    against the signal, first take a counter-direction
                    trade with a tight target sized to the expected
                    adverse move, then flip into the main direction.

Adverse excursion is measured in ATR (the market's own units), and the
report converts it to a leveraged-equity percentage so "15-20% at max
leverage" is directly checkable.

Run:  python3 -m research.two_stage_entry
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bot import risk, strategy

DATASETS = {"2026": "data/BTCUSDT_2026.csv", "2025": "data/BTCUSDT_2025.csv"}

# How far ahead a trade is followed when measuring what actually happened.
LOOKAHEAD_BARS = 360           # 6h
# "Short-term momentum against the signal" test, in 1m bars.
MOMENTUM_LOOKBACK = 30
# Mode B: how long we're willing to wait for the counter-move to stop.
MAX_WAIT_BARS = 120
# Mode C: counter-trade target and stop, in ATR units.
CAPTURE_TARGET_ATR = 0.5
CAPTURE_STOP_ATR = 0.5


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def signal_rows(df: pd.DataFrame) -> pd.DataFrame:
    prep = strategy.prepare(df)
    sig = prep[(prep["long_setup"]) | (prep["short_setup"])].copy()
    sig["side"] = np.where(sig["long_setup"], "long", "short")
    return sig


def measure_adverse_excursion(prep: pd.DataFrame, sig: pd.DataFrame) -> pd.DataFrame:
    """For each signal, how far did price move AGAINST it before it moved
    for it? Reported in ATR and in leveraged-equity percent.
    """
    high = prep["high"].to_numpy(float)
    low = prep["low"].to_numpy(float)
    close = prep["close"].to_numpy(float)
    n = len(close)

    out = []
    for idx, side, entry, atr in zip(sig.index, sig["side"], sig["close"], sig["atr14"]):
        end = min(idx + LOOKAHEAD_BARS, n)
        if end <= idx + 1 or not np.isfinite(atr) or atr <= 0:
            continue
        seg_high, seg_low = high[idx + 1:end], low[idx + 1:end]
        if side == "long":
            worst = entry - seg_low.min()          # drawdown before any rally
            best = seg_high.max() - entry
        else:
            worst = seg_high.max() - entry
            best = entry - seg_low.min()
        out.append({
            "idx": idx, "side": side, "entry": entry, "atr": atr,
            "adverse_atr": worst / atr,
            "favorable_atr": best / atr,
            "adverse_pct": 100 * worst / entry,
            "favorable_pct": 100 * best / entry,
        })
    return pd.DataFrame(out)


def simulate(prep: pd.DataFrame, sig: pd.DataFrame, mode: str) -> dict:
    """Run one entry mode over every signal and total the net R."""
    high = prep["high"].to_numpy(float)
    low = prep["low"].to_numpy(float)
    close = prep["close"].to_numpy(float)
    ema_f = prep["ema9"].to_numpy(float)
    ema_m = prep["ema21"].to_numpy(float)
    n = len(close)
    cost = risk.ROUND_TRIP_COST_PCT

    results = []
    capture_leg = {"taken": 0, "won": 0, "pnl_pct": 0.0}

    for idx, side, atr in zip(sig.index, sig["side"], sig["atr14"]):
        if not np.isfinite(atr) or atr <= 0 or idx + 2 >= n:
            continue
        long = side == "long"

        # Is very-short-term momentum running against the signal right now?
        past = max(0, idx - MOMENTUM_LOOKBACK)
        recent_move = close[idx] - close[past]
        against = (recent_move < 0) if long else (recent_move > 0)

        entry_idx = idx
        extra_pct = 0.0

        if mode.startswith("limit"):
            # Don't chase: rest a limit entry BELOW price for a long
            # (above for a short) and only take the trade if the dip
            # actually comes to us within the window. This is the
            # no-lookahead way to "not buy the top".
            depth = float(mode.split("_")[1]) * atr
            target_entry = close[idx] - depth if long else close[idx] + depth
            entry_idx = None
            for j in range(idx + 1, min(idx + MAX_WAIT_BARS, n - 1)):
                filled = (low[j] <= target_entry) if long else (high[j] >= target_entry)
                if filled:
                    entry_idx = j
                    break
            if entry_idx is None:
                continue  # dip never came; no trade, capital stayed free

        elif mode == "always_capture":
            # The literal request, with no momentum precondition: the
            # adverse move is measured above as real and ~median 2 ATR,
            # so always trade it first, then flip into the signal.
            c_entry = close[idx]
            if long:
                target, stop = (c_entry - CAPTURE_TARGET_ATR * atr,
                                c_entry + CAPTURE_STOP_ATR * atr)
            else:
                target, stop = (c_entry + CAPTURE_TARGET_ATR * atr,
                                c_entry - CAPTURE_STOP_ATR * atr)
            hit_idx, won = None, False
            for j in range(idx + 1, min(idx + MAX_WAIT_BARS, n - 1)):
                if long:
                    if high[j] >= stop:
                        hit_idx, won = j, False
                        break
                    if low[j] <= target:
                        hit_idx, won = j, True
                        break
                else:
                    if low[j] <= stop:
                        hit_idx, won = j, False
                        break
                    if high[j] >= target:
                        hit_idx, won = j, True
                        break
            if hit_idx is None:
                hit_idx, won = min(idx + MAX_WAIT_BARS, n - 2), False
            leg_pct = (CAPTURE_TARGET_ATR * atr / c_entry if won
                       else -CAPTURE_STOP_ATR * atr / c_entry)
            leg_pct = 100 * (leg_pct - cost)
            capture_leg["taken"] += 1
            capture_leg["won"] += int(won)
            capture_leg["pnl_pct"] += leg_pct
            extra_pct = leg_pct
            entry_idx = hit_idx

        elif mode == "wait" and against:
            # Wait for the counter-move to stop: fast EMA turning back in
            # our favour, or the deadline.
            entry_idx = None
            for j in range(idx + 1, min(idx + MAX_WAIT_BARS, n - 1)):
                turned = (ema_f[j] > ema_f[j - 1]) if long else (ema_f[j] < ema_f[j - 1])
                if turned:
                    entry_idx = j
                    break
            if entry_idx is None:
                continue  # never stopped falling within the window: skip

        elif mode == "capture" and against:
            # Trade the counter-move first, tight target and stop.
            c_entry = close[idx]
            if long:  # counter-trade is a SHORT
                target, stop = c_entry - CAPTURE_TARGET_ATR * atr, c_entry + CAPTURE_STOP_ATR * atr
            else:
                target, stop = c_entry + CAPTURE_TARGET_ATR * atr, c_entry - CAPTURE_STOP_ATR * atr
            hit_idx = None
            won = False
            for j in range(idx + 1, min(idx + MAX_WAIT_BARS, n - 1)):
                if long:
                    if high[j] >= stop:
                        hit_idx, won = j, False
                        break
                    if low[j] <= target:
                        hit_idx, won = j, True
                        break
                else:
                    if low[j] <= stop:
                        hit_idx, won = j, False
                        break
                    if high[j] >= target:
                        hit_idx, won = j, True
                        break
            if hit_idx is None:
                hit_idx = min(idx + MAX_WAIT_BARS, n - 2)
                won = False
            leg_pct = (CAPTURE_TARGET_ATR * atr / c_entry if won
                       else -CAPTURE_STOP_ATR * atr / c_entry)
            leg_pct = 100 * (leg_pct - cost)
            capture_leg["taken"] += 1
            capture_leg["won"] += int(won)
            capture_leg["pnl_pct"] += leg_pct
            extra_pct = leg_pct
            entry_idx = hit_idx

        # Main-direction trade from entry_idx, standard ATR stop / 2R TP.
        entry = close[entry_idx]
        stop_dist = strategy.ATR_INIT_MULT * atr
        if long:
            stop, tp = entry - stop_dist, entry + strategy.TP1_R_MULT * stop_dist
        else:
            stop, tp = entry + stop_dist, entry - strategy.TP1_R_MULT * stop_dist

        outcome = 0.0
        for j in range(entry_idx + 1, min(entry_idx + LOOKAHEAD_BARS, n)):
            if long:
                if low[j] <= stop:
                    outcome = -1.0
                    break
                if high[j] >= tp:
                    outcome = strategy.TP1_R_MULT
                    break
            else:
                if high[j] >= stop:
                    outcome = -1.0
                    break
                if low[j] <= tp:
                    outcome = strategy.TP1_R_MULT
                    break
        main_pct = 100 * (outcome * stop_dist / entry - cost)
        results.append(main_pct + extra_pct)

    arr = np.array(results) if results else np.array([0.0])
    return {
        "mode": mode,
        "trades": len(results),
        "win_rate_pct": round(100 * float((arr > 0).mean()), 2),
        "total_pct": round(float(arr.sum()), 2),
        "avg_pct": round(float(arr.mean()), 4),
        "capture_legs": capture_leg["taken"],
        "capture_win_pct": (round(100 * capture_leg["won"] / capture_leg["taken"], 2)
                            if capture_leg["taken"] else 0.0),
        "capture_pnl_pct": round(capture_leg["pnl_pct"], 2),
    }


def main() -> None:
    max_lev = risk.ABSOLUTE_MAX_LEVERAGE
    print(f"Max leverage in risk.py: {max_lev}x  ->  a 15-20% leveraged drawdown "
          f"is a {15/max_lev:.2f}-{20/max_lev:.2f}% spot move\n")

    for year, path in DATASETS.items():
        df = load(path)
        prep = strategy.prepare(df)
        sig = prep[(prep["long_setup"]) | (prep["short_setup"])].copy()
        sig["side"] = np.where(sig["long_setup"], "long", "short")
        print("=" * 74)
        print(f"{year}: {len(sig)} raw signals")
        print("=" * 74)

        exc = measure_adverse_excursion(prep, sig)
        if not exc.empty:
            lev_adverse = exc["adverse_pct"] * max_lev
            print(f"  Adverse excursion before the move goes our way (6h window):")
            print(f"    median {exc['adverse_atr'].median():.2f} ATR "
                  f"= {exc['adverse_pct'].median():.2f}% spot "
                  f"= {lev_adverse.median():.1f}% at {max_lev}x")
            print(f"    75th   {exc['adverse_atr'].quantile(.75):.2f} ATR "
                  f"= {exc['adverse_pct'].quantile(.75):.2f}% spot "
                  f"= {lev_adverse.quantile(.75):.1f}% at {max_lev}x")
            share = 100 * float((lev_adverse >= 15).mean())
            print(f"    signals whose adverse move reaches >=15% at {max_lev}x: {share:.1f}%")

        print("\n  Entry mode comparison (net of fees, same signals):")
        for mode in ("immediate", "wait", "capture", "always_capture",
                     "limit_0.5", "limit_1.0", "limit_2.0"):
            r = simulate(prep, sig, mode)
            line = (f"    {r['mode']:10s} trades {r['trades']:>4d}  "
                    f"win {r['win_rate_pct']:>5.2f}%  total {r['total_pct']:>+9.2f}%  "
                    f"avg {r['avg_pct']:>+7.4f}%")
            if r["capture_legs"]:
                line += (f"   [counter-legs {r['capture_legs']}, "
                         f"win {r['capture_win_pct']:.1f}%, "
                         f"pnl {r['capture_pnl_pct']:+.2f}%]")
            print(line)
        print()


if __name__ == "__main__":
    main()
