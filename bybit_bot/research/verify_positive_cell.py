"""Does the one positive result survive an independent third test?

research/deep_market_model.py produced fifteen negative cells and one
positive one: LONG, 2026 model applied to 2025, top 0.1% of predictions,
+0.2563% per trade over 521 trades. Taken on its own that looks strong,
so it deserves a real check rather than being waved away.

Three things are asked of it here:

  1. Is it significant on its own terms? t-statistic of the per-trade
     returns, and the same for its mirror (2025 model on 2026 data).
  2. Does it hold on a split neither model has seen -- first half of 2025
     to second half of 2025? Same year, same regime, so a genuine effect
     should appear there too.
  3. How large is the multiple-comparison problem? Sixteen cells were
     examined; the best of sixteen noisy draws looks good by definition.

A real edge clears all three. A lucky cell clears the first and fails the
rest.

Run:  python3 -m research.verify_positive_cell
"""
from __future__ import annotations

import numpy as np

from research.deep_market_model import (
    bin_features, block_atr, build_features, fit_gbdt, load, predict_gbdt,
    realised_net, worth_trading,
)

TOP_PCT = 0.1


def evaluate(train_df, test_df, side: str):
    Xtr, _ = build_features(train_df)
    Xte, _ = build_features(test_df)
    atr_tr, atr_te = block_atr(train_df), block_atr(test_df)
    ytr, yte = realised_net(train_df, side, atr_tr), realised_net(test_df, side, atr_te)

    ok_tr = np.isfinite(Xtr).all(1) & np.isfinite(ytr) & worth_trading(train_df, atr_tr)
    ok_te = np.isfinite(Xte).all(1) & np.isfinite(yte) & worth_trading(test_df, atr_te)

    Btr, edges = bin_features(Xtr[ok_tr])
    Bte, _ = bin_features(Xte[ok_te], edges)
    model = fit_gbdt(Btr, ytr[ok_tr])
    pred = predict_gbdt(model, Bte)
    y = yte[ok_te]

    thr = np.percentile(pred, 100 - TOP_PCT)
    sel = y[pred >= thr]
    if len(sel) < 20:
        return None
    mean = float(sel.mean())
    se = float(sel.std(ddof=1) / np.sqrt(len(sel)))
    return {
        "trades": len(sel),
        "mean_pct": 100 * mean,
        "t_stat": mean / se if se > 0 else 0.0,
        "win_rate_pct": 100 * float((sel > 0).mean()),
    }


def show(label: str, r: dict | None) -> None:
    if r is None:
        print(f"  {label:<34s} too few trades to judge")
        return
    verdict = "significant" if abs(r["t_stat"]) >= 2 else "not significant"
    print(f"  {label:<34s} {r['mean_pct']:+.4f}% over {r['trades']:>5,d} trades  "
          f"win {r['win_rate_pct']:5.2f}%  t={r['t_stat']:+.2f}  ({verdict})")


def main() -> None:
    print(f"Re-testing the single positive cell: LONG, top {TOP_PCT}% of predictions\n")
    d2025, d2026 = load("data/BTCUSDT_2025.csv"), load("data/BTCUSDT_2026.csv")

    print("1) The cell itself, and its mirror:")
    show("2026 model -> 2025 data (the cell)", evaluate(d2026, d2025, "long"))
    show("2025 model -> 2026 data (mirror)", evaluate(d2025, d2026, "long"))

    print("\n2) An independent third split, same year:")
    half = len(d2025) // 2
    show("2025 H1 model -> 2025 H2 data",
         evaluate(d2025.iloc[:half].reset_index(drop=True),
                  d2025.iloc[half:].reset_index(drop=True), "long"))
    half26 = len(d2026) // 2
    show("2026 H1 model -> 2026 H2 data",
         evaluate(d2026.iloc[:half26].reset_index(drop=True),
                  d2026.iloc[half26:].reset_index(drop=True), "long"))

    print("\n3) Multiple comparisons:")
    print("   16 cells were examined in deep_market_model.py (2 sides x 2")
    print("   directions x 4 selectivity levels). With 16 independent draws,")
    print("   the chance that at least one clears t=2 purely by luck is about")
    print(f"   {100 * (1 - 0.95 ** 16):.0f}%, so a lone t=2 result proves nothing on its own.")
    print("   It has to repeat on data chosen before the result was seen --")
    print("   which is what the splits above are.")


if __name__ == "__main__":
    main()
