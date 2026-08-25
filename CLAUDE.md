# fp/ -- Bybit futures logic bot

This repo fits a trading logic per coin from 1-minute Bybit history
(`fp/full.py`), trades it live (`run_full_bot.py`, paper by default,
`--real-trade --i-understand-real-money` for real orders), and keeps
it current with fresh bars (`fp/retrain.py`). Read the docstrings at
the top of those three files before changing anything -- they explain
the design decisions (why models are fitted-not-refit at trade time,
why the cost model includes spread and leverage, why retrain writes
files atomically) in more depth than is worth repeating here.

## Adding new coins -- fully automatic, no manual data upload needed

`python -m fp.retrain --symbols SYM1,SYM2,...` does both steps in one
command:

1. **Fetches full history from Bybit's public API** for any symbol
   not yet in `data/all_1m.csv.gz` -- pages forward from 2025-01-01 to
   now automatically (`fp/retrain.py`'s `refresh_cache`/`fetch_since`).
   No manual JSON upload is needed for a brand-new coin; this only
   works from a machine with real network access to api.bybit.com.
2. **Fits the logic** for those symbols (`python -m fp.full --fresh`
   under the hood).

Once a coin is in the cache, the running bot's own
`--retrain-continuous` retrain thread keeps it current forever --
this command is only needed ONCE per new coin, to seed it.

### Before running this on a machine where `run_full_bot.py` is live

- Check its dashboard shows `open 0` (no real position currently
  open), then stop it (Ctrl+C) before running a manual
  `python -m fp.retrain`. Two processes writing `data/all_1m.csv.gz`
  around the same time can silently lose each other's fetched rows --
  the write itself is atomic (temp file + replace) so it never
  corrupts, but a stale read-before-write can still drop data.
- Check free disk space first. `fp/retrain.py`'s `cycle()` already
  refuses to write below `FP_MIN_FREE_MB` (default 1GB) rather than
  risk a half-written file, but fitting many coins at once still adds
  up -- prefer batches of ~10-15 symbols over one huge run so a crash
  partway through doesn't cost hours of unsaved progress. (Models and
  the cache are both saved incrementally, per coin, as they finish --
  see `fp/full.py`'s `save_models` -- so a batch that gets through 8
  of 15 coins before failing has NOT lost those 8.)
- Restart `run_full_bot.py --retrain-continuous` afterward.

### Where things live

- `data/all_1m.csv.gz` -- the bar cache, one row per symbol per
  minute. `fp/models/*.pkl.gz` -- fitted logic per coin, gitignored
  (rebuilt locally, not shipped via git -- see git history around
  2026-08-25 for why: it used to be tracked and bloated `.git` to
  2GB+ from repeatedly committing multi-megabyte boosters).
- `data/instruments.json` -- Bybit's real per-symbol leverage/qty
  limits, refreshed by `fp.costs.refresh()` at real-trade startup.
  Real orders on a symbol missing usable data here are refused, not
  guessed at (`fp.costs.round_qty` returns 0.0) -- see the SNDKUSDT
  history in git log if this comes up again.

## A cloud/remote Claude session cannot do the fetch step

Any session running in a sandboxed cloud container (proxy-restricted
outbound network) cannot reach api.bybit.com at all -- confirmed by a
blocked `curl` to `api.bybit.com` during this repo's development.
That is the entire reason new-coin data used to require manual JSON
uploads from the user instead of an automatic fetch. A Claude Code
session running directly on the machine that also runs
`run_full_bot.py` (real Bybit network access, proven by real orders
succeeding there) can run the `python -m fp.retrain --symbols ...`
command above directly, with no upload step at all.
