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
    # WHICH LOGIC FIRED. "own" when the coin's own fitted model called
    # it, otherwise the coin whose model recognised the setup here.
    # Without this, a losing session cannot be split into "the logic is
    # wrong" and "the logic was applied where it does not belong", and
    # that is the only split that matters on a 713-coin board.
    source: str = "own"

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
        """Whichever piece of logic RECOGNISES this bar gets to trade it.

        Each fitted model is a piece of logic, not a coin. All 200
        columns are scale-free, so a booster fitted on SOLUSDT reads a
        bar of any coin and says whether the shape it learned is present
        here. When one of them recognises its own setup on a coin it has
        never seen, that is the same signal on a different coin, and the
        signal is what the logic was built from.

        So the rule is recognition, not agreement:

          * every model scores the bar;
          * a model COUNTS only if its confidence clears the gate that
            was fitted to its own mistakes, and its predicted move
            clears the floor -- the same two tests it has to pass on the
            coin it was fitted on;
          * among the models that count, the most confident one names
            the side, the target and the stop.

        The earlier version required all nine to agree. That is a
        different rule than the one asked for and a much narrower one:
        it silences a logic that recognises its setup precisely because
        eight others, built from different coins' shapes, do not see
        theirs. Absence of another logic's signal is not evidence
        against this one.

        The one case still refused is a genuine contradiction: two
        models equally confident and pointing opposite ways. There is no
        signal to follow there, only a coin flip.

        Returns (side, conf, profit, mae, gate, which model) -- gate
        1.0 means no trade, since nothing can clear it. The gate
        compared against is `live_gate`: a confidence floor measured on
        a fold that model never trained on, not `gate`, which is
        measured on the same rows the model was fit to reproduce and is
        therefore ~0.0000 by construction -- see fp/full.py's
        oof_gate() for why the same-rows number admits everything.
        """
        best = None
        for other, m in self.models.items():
            s, c, p, a = FU.call(m, row)
            side, conf = int(s[0]), float(c[0])
            profit, mae = float(p[0]), float(a[0])
            meta = self.meta[other]
            live_gate = meta.get("live_gate", meta["gate"])
            if side == 0 or conf <= live_gate or profit < meta["floor"]:
                continue
            cand = (conf, side, profit, mae, live_gate, other)
            if best is None or cand[0] > best[0]:
                best = cand
        if best is None:
            return 0, 0.0, 0.0, 0.0, 1.0, ""
        # A dead heat pointing both ways is not a signal.
        for other, m in self.models.items():
            if other == best[5]:
                continue
            s, c, _, _ = FU.call(m, row)
            if int(s[0]) == -best[1] and abs(float(c[0]) - best[0]) < 1e-12:
                return 0, 0.0, 0.0, 0.0, 1.0, ""
        conf, side, profit, mae, gate, who = best
        return side, conf, profit, mae, gate, who

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

        # A coin with its own fitted logic is judged by it. That logic
        # was measured at 100% on this coin's own history; letting a
        # logic built from a different coin overrule it is a trade made
        # on weaker evidence than the one available. Briefly it did:
        # foreign logics firing on BLESSUSDT took the same replay from
        # 15/15 to 11/13 with two stop-outs.
        #
        # Everything else on the board -- the ~590 coins with no logic
        # of their own -- goes to recognition across all of them.
        own = self.models.get(sym)
        if own is not None:
            meta = dict(self.meta[sym])
            sd, cf, pf, ma = FU.call(own, row)
            s, conf0 = int(sd[0]), float(cf[0])
            profit0, mae0 = float(pf[0]), float(ma[0])
            source = "own"
        else:
            s, conf0, profit0, mae0, gate, who = self.ensemble_call(row, sym)
            meta = dict(self.meta.get(who) or next(iter(self.meta.values())))
            meta["live_gate"] = gate
            source = who or "none"
        # The leverage ceiling is always THIS coin's, read from Bybit,
        # never inherited from whichever coin the winning logic was
        # fitted on.
        meta["lev_cap"] = C.max_leverage(sym)
        # live_gate, not gate: see the docstring above ensemble_call.
        # Falls back to gate only for a model file built before this was
        # added.
        live_gate = meta.get("live_gate", meta["gate"])
        if s == 0 or conf0 <= live_gate:
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

        # The reference comes from the fit, never from this one row.
        sc = float(FU.potential(conf, profit, mae, np.array([cost]),
                                ref=meta.get("edge_ref"))[0])
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
                        trail=trail, cost=cost, source=source)

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
        # The stop ratchets to break-even once the trade has been up by
        # more than the round trip: a winner is not allowed to become a
        # loser. See fp/full.py -- BTCUSDT's only two losses in 2,665
        # trades were exactly this, and both were true opportunities.
        guard = dec.cost if peak >= FU.FLOOR + dec.cost else -dec.stop
        if cur <= guard:
            return ("breakeven" if guard >= 0 else "stop"), guard
        if bars_held >= dec.hold_limit:
            return "time", cur
        return None, peak
