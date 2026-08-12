"""Train the shipped multi-timeframe model and write it to disk.

Trains on every bar available -- June plus the August window -- because
the walk-forward showed the model getting better with more data, not
worse (fold 1 at 38k rows read -0.10%/trade at the 0.002 gate, fold 3 at
129k rows read +1.01%). Holding data back for a fourth validation would
buy a number already measured three times and cost the live model the
thing that helps it most.

What is written:

  mtf_model.pkl    six fitted regressors: three barrier shapes x two
                   sides. Each predicts the NET return of a trade opened
                   at this bar with that shape on that side.
  mtf_model.json   the metadata the bot needs to reproduce the feature
                   matrix exactly -- column order, views, entry
                   timeframe, barrier shapes, and the gate table the
                   walk-forward measured.

The gate table is not a tuned parameter. It is the measured relationship
between predicted net and realized net, and the bot reads it to decide
both WHETHER a setup is worth trading and HOW MUCH to commit.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from fp import mtf_run as R
from fp import wf

HERE = Path(__file__).resolve().parent
PKL = HERE / "mtf_model.pkl"
META = HERE / "mtf_model.json"

FULL = ("2026-05-31", "2026-08-08")

# The gate lives in fp/mtf_bands.json, produced by run_bands.py. It is a
# MARGINAL table -- what a trade landing IN a band earned -- in units of
# net/b, the share of its own winning payout.
#
# The first version of this was CUMULATIVE and used as if it were
# per-band, which had the bot trading two bands the study had measured as
# losing. The live session that followed spent its money in exactly those
# bands: -0.101%/trade in the bottom one.
BANDS_FILE = "mtf_bands.json"


def main() -> None:
    print("building the full panel...", flush=True)
    models, cols, rows = {}, None, 0
    for shape in R.SHAPES:
        p = R.panel(FULL, shape)
        if p is None:
            print(f"  {shape}: no data")
            continue
        X, yl, ys = p[0], p[1], p[2]
        cols = list(X.columns)
        rows += len(X)
        models[f"{shape[0]}_{shape[1]}_{shape[2]}_long"] = wf.fit(X, yl)
        models[f"{shape[0]}_{shape[1]}_{shape[2]}_short"] = wf.fit(X, ys)
        print(f"  {shape}: {len(X):,} rows -> 2 models", flush=True)

    if not models:
        raise SystemExit("no data to train on")

    PKL.write_bytes(pickle.dumps(models))
    META.write_text(json.dumps({
        "entry_minutes": R.ENTRY_MIN,
        "views": __import__("fp.mtf", fromlist=["x"]).VIEWS,
        "shapes": [list(s) for s in R.SHAPES],
        "columns": cols,
        "train_window": list(FULL),
        "train_rows": rows,
        "model": wf.MODEL,
        "bands_file": BANDS_FILE,
        "note": "Trained on 2026-05-31..08-08. The tradeable band is read "
                "from mtf_bands.json: one band of six was measured "
                "profitable, 0.006-0.009, earning 0.406 of a win over 34 "
                "independent trades at t=2.76. One band out of six with "
                "negative neighbours on both sides -- promising, not "
                "proven.",
    }, indent=1))
    print(f"\nwrote {PKL.name} ({PKL.stat().st_size:,} bytes) and {META.name}")
    print(f"{len(models)} models, {len(cols)} features, {rows:,} training rows")


if __name__ == "__main__":
    main()
