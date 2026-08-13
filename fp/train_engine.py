"""Fit the shipped logic on every bar available and write it to disk.

Two artefacts:

  engine_model.pkl   one classifier + isotonic calibration per
                     (target, stop, hold, side). The calibration is
                     fitted on the last 20% of the training window, which
                     the trees never see, so the probability it produces
                     is a frequency rather than a ranking.
  engine_meta.json   everything the live path needs to rebuild the same
                     feature matrix: column order, shape grid, the gate,
                     and how much history the deepest factor requires.

WHAT GOES IN THE FILE AND WHAT DOES NOT. Only shapes that cleared the
walk-forward are written. A shape that lost out of sample is not shipped
with a smaller stake -- it is not shipped. fp/run_engine.py decides which
those are; this file only records the decision so the bot cannot quietly
trade something the study rejected.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from fp import engine as E
from fp import factors as F
from fp import labels as LB

HERE = Path(__file__).resolve().parent
PKL = HERE / "engine_model.pkl"
META = HERE / "engine_meta.json"
KEEP = HERE / "engine_shapes.json"

FULL = ("2026-05-31", "2026-08-14")


def survivors() -> list[list] | None:
    """The shapes the walk-forward passed, or None if it has not run."""
    try:
        return json.loads(KEEP.read_text())["shapes"]
    except Exception:
        return None


def main() -> None:
    keep = survivors()
    if keep is None:
        raise SystemExit(
            "fp/engine_shapes.json is missing -- run python -m fp.run_engine\n"
            "first. Nothing is trained until the walk-forward says what\n"
            "survived; training every shape and sorting it out later is how\n"
            "a study becomes a backtest.")
    wanted = {tuple(k) for k in keep}
    if not wanted:
        PKL.write_bytes(pickle.dumps({}))
        META.write_text(json.dumps({"shapes": [], "columns": [],
                                    "note": "no shape survived the "
                                            "walk-forward"}, indent=1))
        print("no shape survived -- wrote an EMPTY model. The bot will not "
              "open anything, which is the honest consequence.")
        return

    print(f"training {len(wanted)} surviving shape(s) on {FULL[0]}..{FULL[1]}",
          flush=True)
    panel = E.load_panel(FULL)
    cols = list(next(iter(panel.values()))["X"].columns)
    models, rows = {}, 0
    for key in sorted(wanted):
        a = E.stack(panel, key, E.TRAIN_STRIDE)
        if a is None:
            print(f"  {key}: no rows")
            continue
        fit = E.fit_one(a[0], a[1])
        if fit is None:
            print(f"  {key}: could not fit")
            continue
        models[str(key)] = fit
        rows += len(a[0])
        print(f"  {key}: {len(a[0]):,} rows, base win rate "
              f"{100*a[1].mean():.1f}%, calib on {fit['n_calib']:,}",
              flush=True)

    PKL.write_bytes(pickle.dumps(models))
    META.write_text(json.dumps({
        "columns": cols,
        "shapes": [list(k) for k in sorted(wanted)],
        "lookbacks": list(F.LOOKBACKS),
        "state_windows": list(F.STATE_WINDOWS),
        "sigma_window": LB.SIGMA_WINDOW,
        "need_1m_bars": NEED_1M_BARS,
        "train_window": list(FULL),
        "train_rows": rows,
        "gate": json.loads(KEEP.read_text()).get("gate"),
        "fee_round_trip": LB.FEE,
    }, indent=1))
    print(f"\nwrote {PKL.name} ({PKL.stat().st_size:,} bytes) and {META.name}")
    print(f"{len(models)} models, {len(cols)} factors, {rows:,} rows")


# The deepest factor is the 240-bar state window, and the cross-section
# needs the same history on every OTHER coin at the same timestamps. 1500
# bars covers 240 + the 120-bar sigma warm-up with room to spare -- and
# unlike the previous design this is cheap: two kline pages, not eight.
NEED_1M_BARS = 1500


if __name__ == "__main__":
    main()
