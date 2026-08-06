"""The measured edge table, and the command that rebuilds it.

    python -m fp.calibrate            # rebuild from everything in data/
    python -m fp.calibrate --show     # print the current table

WHY THIS REPLACED THE FORMULA

The gate used to compute expectancy as p*TP*atr - (1-p)*SL*atr - cost with
p fixed at the 35.1% win rate measured across all trades. Since cost is
fixed and the payoff scales with ATR, that formula says: the higher the
ATR, the better the trade. It set a floor of atr >= 1.384% and let
everything above it through.

The data says the opposite. Measured on the region the gate was actually
selecting:

    atr >= 1.384%   n=85   win 30.6%   gross -0.1837%

and a 9h17m live session on 690 symbols returned 113 closed trades at
30.1% -- an independent sample agreeing almost exactly. The gate was
filtering INTO the losing region, and it was doing so because its formula
assumed a constant win rate that the market does not supply.

So the gate no longer assumes anything. It looks up what each ATR band
has actually returned, takes the WORST of the periods measured, and
subtracts the full cost. A band is tradeable only if that is positive.
This cannot select into a losing region, because a losing region reports
its own loss.

WHAT THE TABLE CURRENTLY SAYS

Nothing is tradeable. Every ATR band of every exit is negative after
taker-in, taker-out and funding. The closest is TP2.0 at 0.80-1.10% ATR,
which still loses 0.047% per trade in its worst period.

That is the honest answer to "only trade what makes money". It is not a
refusal to trade; it is the table reporting that no cell clears the cost.
Send more data and rerun this -- if a cell turns positive it will be
traded, and the bot needs no other change for that to happen.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fp import logic as L

TABLE_PATH = Path(__file__).resolve().parent / "edge_table.json"
DATA = Path(__file__).resolve().parent.parent / "bybit_bot" / "data"

# ATR band edges, in percent. Fine enough to separate regimes, coarse
# enough that every band keeps a usable sample.
BANDS = [0.0, 0.25, 0.35, 0.45, 0.60, 0.80, 1.10, 1.50, 99.0]
MIN_SAMPLES = 30


def band_index(atr14_pct: float) -> int:
    return int(np.clip(np.searchsorted(BANDS, atr14_pct, side="right") - 1,
                       0, len(BANDS) - 2))


def load_table() -> dict:
    if not TABLE_PATH.exists():
        return {}
    return json.loads(TABLE_PATH.read_text())


def measured_edge(atr14_pct: float, exit_name: str = L.DEFAULT_EXIT,
                  table: dict | None = None) -> float | None:
    """Gross return this ATR band has actually produced, worst period.

    None means the band was never measured with enough samples -- treated
    as untradeable, because an unmeasured band is not a profitable one.
    """
    tbl = load_table() if table is None else table
    row = tbl.get("table", {}).get(exit_name)
    if not row:
        return None
    v = row[band_index(atr14_pct)]
    return None if v is None else float(v)


def build(sources: list[Path]) -> dict:
    from fp import features as F
    from fp import methods as M

    def read(p: Path) -> pd.DataFrame:
        d = pd.read_csv(p, sep=None, engine="python")
        d.columns = [c.strip().lower() for c in d.columns]
        dt = next(c for c in d.columns if "time" in c or "date" in c)
        d["datetime"] = pd.to_datetime(d[dt])
        return d[["datetime", "open", "high", "low", "close", "volume"]]

    frames = {p.stem: read(p) for p in sources}
    # A file whose range sits inside another is a later slice of it; score
    # it only over its own window so periods stay independent.
    starts = {}
    for name, d in frames.items():
        for other, o in frames.items():
            if other != name and o["datetime"].min() < d["datetime"].min() \
                    and o["datetime"].max() >= d["datetime"].min():
                starts[name] = d["datetime"].min()
                frames[name] = pd.concat([o[o["datetime"] < d["datetime"].min()]
                                          .tail(30_000), d])
                break

    rows = []
    for name, raw in frames.items():
        bars = L.to_bars(raw)
        feats = F.build(bars)
        votes = M.evaluate_all(feats)
        hi, lo, cl = bars["high"].values, bars["low"].values, bars["close"].values
        atr_pct = feats["atr14_pct"].values
        when = bars["datetime"].values
        start = starts.get(name)
        for i in range(250, len(cl) - 1):
            if start is not None and when[i] < np.datetime64(start):
                continue
            if votes["consensus_dir"].iloc[i] == M.TIE:
                continue
            if int(votes["n_methods_fired"].iloc[i]) < 1:
                continue
            a = atr_pct[i]
            if not np.isfinite(a) or a <= 0:
                continue
            d = 1 if votes["consensus_dir"].iloc[i] == M.LONG else -1
            for ex in L.EXIT_STRATEGIES:
                g, _, _ = L.simulate_exit(hi, lo, cl, i, d, (a / 100) * cl[i],
                                          ex, fee=0.0)
                rows.append((name, ex, band_index(a), g))
        print(f"  scored {name}: {len(bars):,} bars")

    t = pd.DataFrame(rows, columns=["set", "exit", "band", "gross"])
    table, counts = {}, {}
    for ex in L.EXIT_STRATEGIES:
        worst, n_tot = [], []
        for b in range(len(BANDS) - 1):
            per = []
            total = 0
            for s in t["set"].unique():
                g = t[(t.set == s) & (t.exit == ex) & (t.band == b)].gross
                total += len(g)
                if len(g) >= MIN_SAMPLES:
                    per.append(float(g.mean()))
            worst.append(min(per) if per else None)
            n_tot.append(total)
        table[ex] = worst
        counts[ex] = n_tot
    return {"bands": BANDS, "table": table, "samples": counts,
            "sources": sorted(p.name for p in sources),
            "periods": sorted(t["set"].unique().tolist())}


def show(tbl: dict) -> None:
    bands = tbl["bands"]
    hdr = "".join(f"{f'{bands[i]:.2f}-{bands[i+1]:.2f}':>11}"
                  for i in range(len(bands) - 1))
    print(f"periods: {', '.join(tbl['periods'])}")
    print(f"sources: {', '.join(tbl['sources'])}\n")
    print("GROSS, worst period of each band")
    print(f"{'exit':>14} {hdr}")
    for ex in L.EXIT_STRATEGIES:
        row = "".join(f"{100*v:>10.4f} " if v is not None else f"{'--':>10} "
                      for v in tbl["table"][ex])
        print(f"{ex.replace('net_',''):>14} {row}")
    print("\nNET after taker in + taker out + funding -- a cell is traded"
          " only if positive")
    print(f"{'exit':>14} {hdr}")
    tradeable = 0
    for ex in L.EXIT_STRATEGIES:
        cost = L.round_trip_cost(ex, False, L.FUNDING_RATE_TYPICAL)["total"]
        cells = []
        for v in tbl["table"][ex]:
            if v is None:
                cells.append(f"{'--':>10} ")
            else:
                net = v - cost
                tradeable += net > 0
                cells.append((f"{'+'+format(100*net,'.4f'):>10} " if net > 0
                              else f"{100*net:>10.4f} "))
        print(f"{ex.replace('net_',''):>14} {''.join(cells)}")
    print(f"\n  tradeable cells: {tradeable}")
    if not tradeable:
        print("  Nothing clears the cost. The bot will not open a position,")
        print("  which is what 'only trade what makes money' means here.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--show", action="store_true")
    p.add_argument("--data", action="append", default=[])
    a = p.parse_args(argv)

    if a.show:
        tbl = load_table()
        if not tbl:
            print("no table yet -- run `python -m fp.calibrate` first")
            return 1
        show(tbl)
        return 0

    srcs = ([Path(x) for x in a.data] if a.data
            else sorted(DATA.glob("BTCUSDT_*.csv")))
    if not srcs:
        print(f"no data found in {DATA}", file=sys.stderr)
        return 1
    print(f"building the edge table from {len(srcs)} file(s)")
    tbl = build(srcs)
    TABLE_PATH.write_text(json.dumps(tbl, indent=1))
    print(f"\nwrote {TABLE_PATH}\n")
    show(tbl)
    return 0


if __name__ == "__main__":
    sys.exit(main())
