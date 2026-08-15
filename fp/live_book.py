"""Run the shipped logic book against a live exchange.

fp/logic_book.json holds (coin, strategy) pairs that were profitable
FORWARD in every walk-forward fold they traded in. This turns each of
them into a live position.

The contract is the same one the study measured, and that matters more
than anything else here: a strategy is a signed STATE, and the bot holds
exactly that state.

    state +1  -> be long        state -1  -> be short       0 -> be flat

  * ENTRY happens when the state turns on, at whatever the market is.
  * EXIT happens when the state turns off or flips, at whatever the
    market is then.
  * There is no target, no stop and no time limit, because the study
    that measured these pairs did not use one. Adding one live would be
    trading a different rule from the one that was validated.

THE WAIT. A state must persist MIN_RUN bars before the bot acts on it,
exactly as in the study -- entering the moment a state flickers on, and
measuring as though you had waited, is the look-ahead that once made
solo:boll_break read +0.47%/trade. The live path waits for the same
number of closed bars.

THE BOARD. Five of the methods are cross-sectional: they rank this coin
against the others. They cannot be computed one symbol at a time, so the
live path fetches every scanned coin and builds the panel together.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from fp import strategies as SG

logger = logging.getLogger("fp.live_book")

HERE = Path(__file__).resolve().parent
BOOK = HERE / "logic_book.json"

# The deepest method lookback is the 200-bar EMA/Donchian and the 480-bar
# volatility regime; 1,500 one-minute bars covers both with room. Two
# kline pages per coin.
NEED_1M_BARS = 1500


class LogicBook:
    """The shipped pairs, and the live states they resolve to."""

    def __init__(self, path: Path = BOOK):
        self.pairs: list[dict] = []
        self.meta: dict = {}
        self.ok = path.exists()
        if not self.ok:
            logger.warning("%s missing -- run python -m fp.run_percoin",
                           path.name)
            return
        self.meta = json.loads(path.read_text())
        # New format: per_coin -> [{strategy, expect, stake}], ranked.
        # The bot holds the best signal that is ON, one per coin.
        per = self.meta.get("per_coin")
        if per is not None:
            self.by_symbol = {s_: sorted(v, key=lambda r: -r["expect"])
                              for s_, v in per.items() if v}
            self.pairs = [{"symbol": s_, **r}
                          for s_, v in self.by_symbol.items() for r in v]
        else:
            self.pairs = list(self.meta.get("pairs", []))
            self.by_symbol = {}
            for p_ in self.pairs:
                self.by_symbol.setdefault(p_["symbol"], []).append(p_)
        self.walk = self.meta.get("walk_forward", {})
        self.min_stake = float(self.meta.get("min_stake", 0.05))
        if not self.pairs:
            logger.warning("%s holds no tradeable strategy -- nothing will "
                           "be opened", path.name)

    @property
    def symbols(self) -> list[str]:
        return sorted(self.by_symbol)

    def states(self, bars: dict[str, pd.DataFrame]) -> dict[str, dict]:
        """The ONE signal each coin should be holding, or nothing.

        A coin holds at most one position, and it is the highest-
        expectation strategy currently on. That is the operator's rule and
        it is also what keeps the account solvent: 623 strategies riding
        the same move asked for exposure the account does not have, and
        switching between them every few minutes cost 0.11% a time --
        measured at -540% in fees against +193% of gross signal.
        """
        good = {s: d for s, d in bars.items()
                if d is not None and len(d) >= 400}
        if len(good) < 2:
            return {}
        out: dict[str, dict] = {}
        for sym, want in self.by_symbol.items():
            d = good.get(sym)
            if d is None:
                continue
            try:
                S = SG.all_strategies(d, good, sym)
            except Exception:
                logger.debug("strategy build failed %s", sym, exc_info=True)
                continue
            # `want` is already ranked by expectation, so the first one
            # found ON is the best one available.
            chosen = {}
            for p in want:
                v = S.get(p["strategy"])
                if v is None or len(v) < SG.MIN_RUN + 1:
                    continue
                tail = v[-SG.MIN_RUN:]
                if tail[-1] != 0 and np.all(tail == tail[-1]):
                    chosen[p["strategy"]] = int(tail[-1])
                    break
            out[sym] = chosen
        return out

    def stake_for(self, symbol: str, strategy: str) -> float:
        for p in self.by_symbol.get(symbol, ()):
            if p["strategy"] == strategy:
                return float(p.get("stake", 0.0))
        return 0.0

    def edge_for(self, symbol: str, strategy: str) -> float:
        for p in self.by_symbol.get(symbol, ()):
            if p["strategy"] == strategy:
                return float(p.get("mean", p.get("expect", 0.0)))
        return 0.0
