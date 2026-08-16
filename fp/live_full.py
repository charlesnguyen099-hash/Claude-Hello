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

    def ensemble_call(self, row, sym: str):
        """Score a coin that has no model of its own, using every model.

        All 200 columns are scale-free, so a booster fitted on SOLUSDT
        produces a number on a coin it has never seen. What it does NOT
        produce is a reason to trust that number, so the panel votes:

          side    only if EVERY model agrees. One dissent and the coin
                  is left alone -- across six hundred coins the cost of
                  skipping is one missed trade, and the cost of being
                  wrong is a levered loss.
          profit  the MINIMUM predicted, not the mean. The target sets
                  the exit, and the optimistic member of a disagreeing
                  panel is the one that leaves a trade hanging.
          mae     the MAXIMUM predicted, for the mirror reason: the stop
                  should respect the most pessimistic member.
          conf    the minimum, so the gate is met by the weakest vote.

        Measured honestly, cross-coin transfer showed no edge out of
        sample -- see fp/transfer.py, 1,423 independent trades, 0/10
        coins clearing their own break-even at p<0.05. This path exists
        because the operator asked for the whole board; unanimity and
        worst-case sizing are what keep that from being reckless.
        """
        sides, confs, profits, maes, gates = [], [], [], [], []
        for other, m in self.models.items():
            s, c, p, a = FU.call(m, row)
            sides.append(int(s[0]))
            confs.append(float(c[0]))
            profits.append(float(p[0]))
            maes.append(float(a[0]))
            gates.append(self.meta[other]["gate"])
        if not sides or 0 in sides or len(set(sides)) != 1:
            return 0, 0.0, 0.0, 0.0, 1.0
        return (sides[0], min(confs), min(profits), max(maes), max(gates))

    def decide(self, d: pd.DataFrame, panel: dict, sym: str) -> Decision | None:
        """The verdict for the LAST bar of `d`. None means no trade.

        `d` must be the recent bars for `sym` and `panel` the same for
        every coin, because a third of the 200 columns are cross-section
        -- what the other coins are doing this minute.
        """
        if not self.models or len(d) < 2:
            return None
        X = DIR.features(d, panel, sym).values.astype("float32")
        row = X[-1:]
        if not np.isfinite(row).all():
            return None                      # still warming up

        own = self.models.get(sym)
        if own is not None:
            meta = dict(self.meta[sym])
            sd, cf, pf, ma = FU.call(own, row)
            s, conf0 = int(sd[0]), float(cf[0])
            profit0, mae0 = float(pf[0]), float(ma[0])
        else:
            # No model for this coin: the panel votes, and the leverage
            # ceiling comes from Bybit for THIS coin, not from whichever
            # coin the models happen to have been fitted on.
            meta = dict(next(iter(self.meta.values())))
            s, conf0, profit0, mae0, gate = self.ensemble_call(row, sym)
            meta["gate"] = gate
            meta["lev_cap"] = C.max_leverage(sym)
        if s == 0 or conf0 <= meta["gate"]:
            return None
        conf = np.array([conf0])
        profit = np.array([profit0])
        mae = np.array([mae0])

        # The live cost is estimated from the bars in hand, not from the
        # figure that happened to hold during training -- spreads widen.
        cv = C.round_trip(d)
        cost = float(np.nanmedian(cv[-240:])) if len(cv) else meta["cost"]
        if not np.isfinite(cost):
            cost = meta["cost"]

        target = float(profit0)
        if target < meta["floor"]:
            return None                      # below the floor, not a trade

        sc = float(FU.potential(conf, profit, mae, np.array([cost]))[0])
        if sc <= 0:
            return None
        stake = meta["min_stake"] + (meta["max_stake"] - meta["min_stake"]) * sc / 100.0
        stop = max(float(mae0) * 1.5, meta["floor"] + cost)
        lev = meta["lev_floor"] + (meta["lev_cap"] - meta["lev_floor"]) * sc / 100.0
        lev = float(np.clip(min(lev, meta["liq_safety"] / stop),
                            meta["lev_floor"], meta["lev_cap"]))
        trail = max(min(float(mae0), 0.5 * target), 2.0 * cost)
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
