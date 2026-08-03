"""Run the PureLogic rule over 2025-2026 and account for every losing trade.

Direction, holding time and leverage come from the table. Profit comes
from what BTCUSDT actually did, with liquidation checked bar by bar
against the position's real adverse excursion — the thing that decides
whether 100x is survivable.

Losing trades are then split by cause, so "which part of the logic is
broken" is answered with counts rather than opinion:

  liquidated        adverse move exceeded the margin; leverage too high
  wrong_direction   price moved against the position by more than the fee
  fee_ate_it        direction was right, the move was smaller than the cost
  flat              essentially no movement either way
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pl import strategy as S


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def run(df: pd.DataFrame, logic: S.PureLogic, cells: np.ndarray,
        leverage: int, fee_round_trip: float = S.ROUND_TRIP_PCT) -> dict:
    close = df["close"].to_numpy(float)
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    n = len(close)

    # Liquidation when the adverse move eats the margin. 0.9 keeps a
    # maintenance-margin allowance rather than assuming the full 1/lev.
    liq_move = 0.9 / leverage

    trades, busy = [], -1
    for i in range(n - 1):
        c = cells[i]
        if c < 0 or i <= busy:
            continue
        sig = logic.signal(int(c))
        if sig is None:
            continue

        d = sig["direction"]
        j = min(i + sig["hold_minutes"], n - 1)
        entry = close[i]
        seg_hi, seg_lo = high[i + 1:j + 1], low[i + 1:j + 1]
        if seg_hi.size == 0:
            continue

        mae = ((entry - seg_lo.min()) / entry if d > 0
               else (seg_hi.max() - entry) / entry)
        mae = max(mae, 0.0)

        if mae >= liq_move:
            gross, reason = -liq_move, "liquidated"
        else:
            gross = (close[j] - entry) / entry * d
            reason = "closed"

        net_unlev = gross - fee_round_trip
        trades.append({
            "gross": gross, "net_unlev": net_unlev,
            "net_lev": net_unlev * leverage,
            "mae": mae, "reason": reason, "conviction": sig["conviction"],
        })
        busy = j

    if not trades:
        return {"trades": 0}
    t = pd.DataFrame(trades)
    net = t["net_lev"].to_numpy()
    wins = net > 0

    losers = t[~wins]
    causes = {
        "liquidated": int((losers["reason"] == "liquidated").sum()),
        "wrong_direction": int((losers["gross"] <= -fee_round_trip).sum()),
        "fee_ate_it": int(((losers["gross"] > 0) &
                           (losers["gross"] < fee_round_trip)).sum()),
        "flat": int((losers["gross"].abs() <= fee_round_trip / 2).sum()),
    }
    return {
        "trades": len(t),
        "wins": int(wins.sum()),
        "losses": int((~wins).sum()),
        "win_rate_pct": round(100 * float(wins.mean()), 2),
        "avg_gross_pct": round(100 * float(t["gross"].mean()), 4),
        "avg_net_unlev_pct": round(100 * float(t["net_unlev"].mean()), 4),
        "avg_net_lev_pct": round(100 * float(t["net_lev"].mean()), 3),
        "median_mae_pct": round(100 * float(t["mae"].median()), 4),
        "liquidations": int((t["reason"] == "liquidated").sum()),
        "loss_causes": causes,
        "equity_x": float(np.expm1(np.log1p(np.clip(net, -0.99, None)).sum())),
    }
