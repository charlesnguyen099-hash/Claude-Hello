"""Give every coin on Bybit its own fitted logic, not just a handful.

WHY THIS IS SEPARATE FROM fp/retrain.py's normal cycle. That cycle
refreshes and refits whatever is ALREADY in the cache -- something has
to pick which of the 700+ turnover-ranked contenders on the whole
board get added in the first place, and that picking has to happen
before fp.retrain's machinery applies to a coin at all. This script is
that picking step: it reads the ranked board from Bybit
(fp.universe.ranked), skips whatever already has a model file, and
runs fp.retrain.cycle() over the rest in small batches.

    python -m fp.backfill_all                # whole board, batches of 15
    python -m fp.backfill_all --batch 10
    python -m fp.backfill_all --limit 100     # stop after 100 NEW symbols

RESUMABLE BY DESIGN. "Already fitted" is read fresh from fp/models/ at
startup, not tracked across runs -- Ctrl+C and rerun picks up exactly
where it left off, and a symbol fp.full skips for too few opportunities
(under 500, see fp/full.py's main()) is simply retried on the next run
at whatever cost that check itself has, which is cheap once its bars
are already cached.

DO NOT RUN THIS AT THE SAME TIME AS run_full_bot.py's OWN
--retrain-continuous THREAD. Both read-modify-write data/all_1m.csv.gz
independently; the write itself is atomic (temp file + replace) so it
can never corrupt, but two independent read-merge-write cycles racing
is a LOST UPDATE -- whichever finishes second overwrites the first's
freshly-fetched rows with a stale snapshot that never saw them. Either
stop the live bot for the duration of a pass, or start it with
--retrain-hours 0 (no retrain thread at all) and let this script be
the only thing that touches the cache until a pass finishes -- then
restart the live bot normally so it picks up every newly fitted model.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fp import full as FU
from fp import retrain as RT
from fp import universe as U

HERE = Path(__file__).resolve().parent


def already_fitted() -> set[str]:
    return {f.name.split(".")[0] for f in FU.MODELS.glob("*.pkl.gz")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=15,
                    help="symbols per retrain cycle (default 15 -- a "
                         "crash mid-batch only costs that batch, not "
                         "the whole run)")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after this many NEW symbols (0 = whole "
                         "board, however many runs that takes)")
    a = ap.parse_args()

    client = RT._make_client()
    print("  fetching the full USDT perpetual list from Bybit...",
          flush=True)
    syms, caps, tvr = U.ranked(client)
    print(f"  {len(syms)} symbols on the board, ranked by 24h turnover",
          flush=True)

    done = already_fitted()
    todo = [s for s in syms if s not in done]
    if a.limit > 0:
        todo = todo[:a.limit]
    print(f"  {len(done)} already fitted, {len(todo)} queued this run",
          flush=True)
    if not todo:
        print("  nothing to do -- every ranked symbol already has a "
              "model file", flush=True)
        return 0

    n_batches = -(-len(todo) // a.batch)
    ok_count, fail_count = 0, 0
    for i in range(0, len(todo), a.batch):
        batch = todo[i:i + a.batch]
        print(f"\n  batch {i // a.batch + 1}/{n_batches}: "
              f"{', '.join(batch)}", flush=True)
        ok = RT.cycle(batch, log=print)
        if ok:
            ok_count += len(batch)
        else:
            fail_count += len(batch)
            print("  batch failed -- not retried automatically this "
                  "run; rerun the script later to pick it back up",
                  flush=True)
        remaining = len(todo) - ok_count - fail_count
        print(f"\n  progress: {ok_count} fitted OK, {fail_count} in "
              f"failed batches, {remaining} still queued", flush=True)

    print("\n  backfill pass complete.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
