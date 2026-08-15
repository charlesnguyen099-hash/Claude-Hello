"""The live side of fp/full.py: same logic, one bar at a time.

fp/full.py fits per coin and replays the past. This loads exactly those
fitted objects and answers the only question a bot asks:

    given the last N bars, is there a trade here, and if so --
    which way, how big a stake, how much leverage, exit where?

Every answer is a FRACTION. Nothing here knows a price, a date or a coin
name beyond which model file to open, so the same code trades a $0.002
coin and a $100,000 one without a constant changing.

WHY IT LOADS RATHER THAN REFITS. The 100%-on-past-data result belongs to
specific fitted boosters. A bot that refits at startup is trading a
different model with different mistakes, and the measured result would
say nothing about what is actually running. So `python -m fp.full` is
the build step and this is the consumer; if a model file is missing the
coin is simply not traded, rather than silently downgraded.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fp import costs as C
from fp import direction as DIR
from fp import full as FU


@dataclass(frozen=True)
class Decision:
    """One trade, entirely in fractions of price and of equity."""
    symbol: str
    side: int             # +1 long, -1 short
    potential: float      # 0..100, drives both stake and leverage
    stake: float          # fraction of current equity
    leverage: float       # multiple, already capped for this coin
    target: float         # net gain to aim for, fraction of entry
    stop: float           # adverse fraction that ends it
    trail: float          # giveback from peak that books the profit
    cost: float           # round-trip cost, fraction of notional

    @property
    def hold_limit(self) -> int:
        return FU.HORIZON


class FullLogic:
    """Loads the fitted logic per coin and scores live bars with it."""

    def __init__(self, symbols):
        self.models, self.meta, self.missing = {}, {}, []
        for s in symbols:
            blob = FU.load_models(s)
            if blob is None:
                self.missing.append(s)
                continue
            self.models[s] = blob["models"]
            self.meta[s] = blob["meta"]

    @property
    def ready(self) -> list[str]:
        return sorted(self.models)

    def decide(self, d: pd.DataFrame, panel: dict, sym: str) -> Decision | None:
        """The verdict for the LAST bar of `d`. None means no trade.

        `d` must be the recent bars for `sym` and `panel` the same for
        every coin, because a third of the 200 columns are cross-section
        -- what the other coins are doing this minute.
        """
        m = self.models.get(sym)
        if m is None or len(d) < 2:
            return None
        X = DIR.features(d, panel, sym).values.astype("float32")
        row = X[-1:]
        if not np.isfinite(row).all():
            return None                      # still warming up

        meta = self.meta[sym]
        side, conf, profit, mae = FU.call(m, row)
        s = int(side[0])
        if s == 0 or float(conf[0]) <= meta["gate"]:
            return None

        # The live cost is estimated from the bars in hand, not from the
        # figure that happened to hold during training -- spreads widen.
        cv = C.round_trip(d)
        cost = float(np.nanmedian(cv[-240:])) if len(cv) else meta["cost"]
        if not np.isfinite(cost):
            cost = meta["cost"]

        target = float(profit[0])
        if target < meta["floor"]:
            return None                      # below the floor, not a trade

        sc = float(FU.potential(conf, profit, mae, np.array([cost]))[0])
        if sc <= 0:
            return None
        stake = meta["min_stake"] + (meta["max_stake"] - meta["min_stake"]) * sc / 100.0
        stop = max(float(mae[0]) * 1.5, meta["floor"] + cost)
        lev = meta["lev_floor"] + (meta["lev_cap"] - meta["lev_floor"]) * sc / 100.0
        lev = float(np.clip(min(lev, meta["liq_safety"] / stop),
                            meta["lev_floor"], meta["lev_cap"]))
        trail = max(min(float(mae[0]), 0.5 * target), 2.0 * cost)
        return Decision(symbol=sym, side=s, potential=sc, stake=stake,
                        leverage=lev, target=target, stop=stop,
                        trail=trail, cost=cost)

    @staticmethod
    def exit_now(dec: Decision, entry: float, high: float, low: float,
                 last: float, peak: float, bars_held: int):
        """Should the open position close on this bar, and at what move?

        The same ladder fp/full.py replays, in the same order: target,
        then trail, then the stop -- which is UNCONDITIONAL, because a
        stop that switches itself off once a trade is in profit is how
        one BLESS trade reached -203.94%.

        Returns (reason, realised move as a fraction) or (None, peak).
        """
        s = dec.side
        fav = (high / entry - 1.0) if s > 0 else (1.0 - low / entry)
        adv = (1.0 - low / entry) if s > 0 else (high / entry - 1.0)
        cur = (last / entry - 1.0) * s
        if fav >= dec.target + dec.cost:
            return "target", dec.target + dec.cost
        peak = max(peak, fav)
        if peak - dec.trail > dec.cost and peak - cur >= dec.trail:
            return "trail", peak - dec.trail
        if adv >= dec.stop:
            return "stop", -dec.stop
        if bars_held >= dec.hold_limit:
            return "time", cur
        return None, peak
