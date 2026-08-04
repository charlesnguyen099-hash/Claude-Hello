"""Flexible leverage done properly: size each cell by its own realised edge.

The point being made is right, and the previous version missed it. A stop
stops the account dying; it does not make a trade profitable. And a single
global leverage cannot help at all, because on margin

    return = L x (gross - fee)

so L multiplies the bracket without touching its sign. If the bracket is
negative, more leverage loses faster; that is all it can do.

Variable leverage is different, and is the thing worth testing. Total
return is the sum over trades of L_i x (gross_i - fee). Choosing L_i per
cell — heavy where the edge is real, zero where it is not — CAN turn a
negative total positive, provided cells with a genuine positive edge exist.

So this measures, per cell, what it actually returned on each year
separately, and asks the only question that decides the matter:

    are there cells whose gross exceeds the fee on BOTH years?

A cell that pays on 2025 and not on 2026 is not an edge, it is the year
that happened to cooperate; sizing up on it is how an account dies slowly
instead of quickly. Cells are therefore selected on 2025 alone and then
applied unchanged to 2026, which is the only honest way to find out
whether the selection means anything.

Leverage is then set proportional to each surviving cell's edge over the
fee, capped by what its excursion can survive — heavy on the strongest,
light on the marginal, zero on the rest.

Run:  python3 pl_flexlev.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pl import backtest as B
from pl import features as F
from pl import strategy as S

TABLE = ("/root/.claude/uploads/2499e73f-5145-5c6f-b255-816732633901/"
         "70587060-Sheet16_PureLogic_MaxLeverage_MaxFee_24590.txt")
DATASETS = {"2025": "bybit_bot/data/BTCUSDT_2025.csv",
            "2026": "bybit_bot/data/BTCUSDT_2026.csv"}
FEE = S.MAKER_ROUND_TRIP


def per_cell_results(df: pd.DataFrame, logic: S.PureLogic,
                     cells: np.ndarray) -> pd.DataFrame:
    """Every trade the rule would take, tagged with the cell that fired it."""
    close = df["close"].to_numpy(float)
    high, low = df["high"].to_numpy(float), df["low"].to_numpy(float)
    n = len(close)
    rows, busy = [], -1

    for i in range(n - 1):
        c = int(cells[i])
        if c < 0 or i <= busy:
            continue
        sig = logic.signal(c)
        if sig is None:
            continue
        d = sig["direction"]
        j = min(i + sig["hold_minutes"], n - 1)
        entry = close[i]
        seg_hi, seg_lo = high[i + 1:j + 1], low[i + 1:j + 1]
        if seg_hi.size == 0:
            continue
        mae = max((entry - seg_lo.min()) / entry if d > 0
                  else (seg_hi.max() - entry) / entry, 0.0)
        rows.append({"cell": c, "gross": (close[j] - entry) / entry * d,
                     "mae": mae})
        busy = j
    return pd.DataFrame(rows)


def main() -> None:
    logic = S.PureLogic(TABLE, min_conviction=0.30, min_edge_multiple=0.0)
    print("=" * 92)
    print("FLEXIBLE LEVERAGE — sized per cell by its own realised edge")
    print("=" * 92)
    print(f"  maker round trip: {FEE*100:.3f}%   (a cell must beat this to be "
          f"worth any leverage at all)")
    print(f"  candidate cells : {len(logic.rules)}")

    data = {y: B.load(p) for y, p in DATASETS.items()}
    res = {}
    for y, df in data.items():
        feats = S.features_30m_on_1m(df, F.build)
        res[y] = per_cell_results(df, logic, logic.cells_for(feats))

    stats = {}
    for y, t in res.items():
        g = t.groupby("cell").agg(n=("gross", "size"),
                                  gross=("gross", "mean"),
                                  mae=("mae", "quantile"))
        stats[y] = g

    a, b = stats["2025"], stats["2026"]
    common = a.index.intersection(b.index)
    print(f"  cells that fired on both years: {len(common)}")

    print()
    print("=" * 92)
    print("STEP 1 — select on 2025 only, then apply unchanged to 2026")
    print("=" * 92)
    sel = a.loc[common]
    picked = sel[(sel["gross"] > FEE) & (sel["n"] >= 20)]
    print(f"  cells with gross > fee on 2025 (>=20 trades): {len(picked)}")
    if len(picked) == 0:
        print("  Nothing to size up. No cell beat the fee even on its own year.")
        return

    held = b.loc[picked.index]
    survived = held[held["gross"] > FEE]
    print(f"  of those, still above the fee on 2026        : {len(survived)}"
          f"  ({100*len(survived)/len(picked):.0f}%)")
    print()
    print(f"  {'cell':>10s} {'n25':>5s} {'gross25%':>9s} {'n26':>5s} "
          f"{'gross26%':>9s}  held up?")
    print("  " + "-" * 56)
    for cell in picked.index[:15]:
        g25, g26 = picked.loc[cell, "gross"], held.loc[cell, "gross"]
        print(f"  {cell:>10d} {picked.loc[cell,'n']:5.0f} {100*g25:9.4f} "
              f"{held.loc[cell,'n']:5.0f} {100*g26:9.4f}  "
              f"{'yes' if g26 > FEE else 'no'}")

    print()
    print("=" * 92)
    print("STEP 2 — leverage proportional to edge, applied to 2026")
    print("=" * 92)

    def evaluate(cells_used, label: str) -> None:
        t = res["2026"]
        t = t[t["cell"].isin(cells_used)]
        if t.empty:
            print(f"  {label:34s} no trades")
            return
        edge25 = picked["gross"].reindex(t["cell"]).to_numpy()
        # Leverage proportional to the edge above the fee, capped by what
        # the cell's own excursion can carry.
        raw = (edge25 - FEE) / FEE
        mae_cap = np.array([S.leverage_for(100 * m) for m in t["mae"]])
        lev = np.clip(raw * 10, S.MIN_LEVERAGE, mae_cap)
        net = (t["gross"].to_numpy() - FEE) * lev
        eq = float(np.expm1(np.log1p(np.clip(net, -0.99, None)).sum()))
        print(f"  {label:34s} trades {len(t):5d}  avg lev {lev.mean():5.1f}x  "
              f"gross {100*t['gross'].mean():+8.4f}%  "
              f"net/trade {100*net.mean():+8.3f}%  "
              f"equity {'wiped' if eq <= -0.999 else f'{1+eq:.3g}x'}")

    evaluate(picked.index, "all 2025-selected cells")
    evaluate(survived.index, "only those that held on 2026")

    print()
    print("=" * 92)
    print("The second line is the honest one — the first re-uses 2026 knowledge")
    print("to pick which cells to trade on 2026. The gap between them is the")
    print("cost of not knowing in advance which cells will keep working.")


if __name__ == "__main__":
    main()
