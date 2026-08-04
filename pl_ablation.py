"""Measure each of the three fixes separately, and then together.

Three problems were raised: liquidation, losses to fees, and losses to
wrong direction. Each has a fix, and each fix is measured on its own so
it is visible which one actually does the work rather than assuming all
three helped.

  BASE       what the previous version did: fixed leverage, taker fees,
             conviction >= 0.30, no stop
  +STOP      a stop placed strictly inside the liquidation distance, so
             liquidation cannot happen
  +MAKER     resting limit orders instead of crossing the spread:
             0.210% round trip -> 0.040%
  +DIRECTION conviction >= 0.60 and >= 40 trades behind each cell, so
             cells where the table had no real opinion are skipped
  ALL        all three

Run:  python3 pl_ablation.py
"""
from __future__ import annotations

from pl import backtest as B
from pl import features as F
from pl import strategy as S

TABLE = ("/root/.claude/uploads/2499e73f-5145-5c6f-b255-816732633901/"
         "70587060-Sheet16_PureLogic_MaxLeverage_MaxFee_24590.txt")
DATASETS = {"2025": "bybit_bot/data/BTCUSDT_2025.csv",
            "2026": "bybit_bot/data/BTCUSDT_2026.csv"}


def main() -> None:
    data = {y: B.load(p) for y, p in DATASETS.items()}
    feats = {y: S.features_30m_on_1m(df, F.build) for y, df in data.items()}

    loose = dict(min_conviction=0.30, min_edge_multiple=2.0)
    tight = dict(min_conviction=0.60, min_edge_multiple=2.0)
    logic_loose = S.PureLogic(TABLE, **loose)
    logic_tight = S.PureLogic(TABLE, **tight)
    cells = {"loose": {y: logic_loose.cells_for(f) for y, f in feats.items()},
             "tight": {y: logic_tight.cells_for(f) for y, f in feats.items()}}

    print(f"cells usable: conviction>=0.30 -> {len(logic_loose.rules)}, "
          f"conviction>=0.60 -> {len(logic_tight.rules)}")
    print(f"fees: taker round trip {S.TAKER_ROUND_TRIP*100:.3f}%, "
          f"maker round trip {S.MAKER_ROUND_TRIP*100:.3f}%")
    print()

    variants = [
        ("BASE",       logic_loose, "loose", S.TAKER_ROUND_TRIP, False, 57),
        ("+STOP",      logic_loose, "loose", S.TAKER_ROUND_TRIP, True,  None),
        ("+MAKER",     logic_loose, "loose", S.MAKER_ROUND_TRIP, False, 57),
        ("+DIRECTION", logic_tight, "tight", S.TAKER_ROUND_TRIP, False, 57),
        ("ALL",        logic_tight, "tight", S.MAKER_ROUND_TRIP, True,  None),
    ]

    hdr = (f"{'variant':11s} | " + " | ".join(
        f"{y} {'trd':>5s} {'win%':>6s} {'lev':>5s} {'gross%':>8s} {'net%':>8s} "
        f"{'lev.net%':>9s} {'liq':>4s} {'stop':>5s} {'equity':>9s}"
        for y in DATASETS))
    print(hdr)
    print("-" * len(hdr))

    for name, logic, ckey, fee, stop, lev in variants:
        out = []
        for y in DATASETS:
            r = B.run(data[y], logic, cells[ckey][y], leverage=lev,
                      fee_round_trip=fee, use_stop=stop)
            if r.get("trades", 0) == 0:
                out.append(f"{y} {0:5d} {'-':>6s} {'-':>5s} {'-':>8s} {'-':>8s} "
                           f"{'-':>9s} {'-':>4s} {'-':>5s} {'-':>9s}")
                continue
            eq = r["equity_x"]
            eqs = "wiped" if eq <= -0.999 else f"{1+eq:8.3g}x"
            out.append(
                f"{y} {r['trades']:5d} {r['win_rate_pct']:6.2f} "
                f"{r['avg_leverage']:5.0f} {r['avg_gross_pct']:8.4f} "
                f"{r['avg_net_unlev_pct']:8.4f} {r['avg_net_lev_pct']:9.3f} "
                f"{r['liquidations']:4d} {r['stops']:5d} {eqs:>9s}")
        print(f"{name:11s} | " + " | ".join(out))

    print()
    print("=" * 100)
    print("net% is per trade before leverage; lev.net% is what actually lands in")
    print("the account. gross% is the edge itself — the number the fee has to be")
    print("smaller than. Leverage multiplies whatever sign gross has.")


if __name__ == "__main__":
    main()
