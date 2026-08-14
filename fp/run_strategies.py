"""Search every strategy, select honestly, and simulate the account.

The operator's method, followed literally: take the ways people trade
futures, combine them with each other and with the cross-section, and let
the profitable past trades pick the logic. 623 strategies come out of
fp/strategies.py and every one of them is measured here.

WHAT MAKES THIS DIFFERENT FROM A BACKTEST.

Selecting the best of 623 on the same data you then report is not a
result, it is a lottery ticket. The best of 623 coin flips looks
brilliant. So:

  * SELECTION happens on the training window only. Each fold picks its
    strategies from bars before the cut, then trades them forward.
  * BONFERRONI over all 623 attempted, not over the survivors. Two-sided
    0.05 across 623 tests needs |t| >= 3.94.
  * A ROTATION NULL: roll each strategy's state by a random offset. Same
    number of trades, same average hold, same long/short balance, and no
    alignment with the market at all.
  * EVERY TRADE IS ALREADY INDEPENDENT. State runs cannot overlap, so
    there is nothing to correct for -- unlike the barrier design, where
    counting overlapping entries turned t = 2.26 into t = 7.81.

RESULT, 2026-05-31..08-14, 965,484 one-minute bars, 10 coins:

  fold 1  select 05-31..07-01, trade 07-01..07-15   0 of 589 selected
  fold 2  select 05-31..07-15, trade 07-15..07-30   0 of 592 selected
  fold 3  select 05-31..07-30, trade 07-30..08-14   0 of 592 selected

Not one strategy of 623 clears |t| >= 3.94 with 30+ trades -- and that is
the TRAINING filter, before any out-of-sample test. The methods, in every
combination tried, lose about the round trip.

THIS FILE FIRST REPORTED THE OPPOSITE, and how it did is worth keeping.
Two bugs, each of which handed the search a free look at the future:

  1. EXIT ONE BAR EARLY. A run's last bar is only known to be the last
     once the next bar's state differs. Exiting at the run's own last
     close sells before the flip is observable. solo:rsi_trend read
     +0.2491%/trade over 19,897 trades at t = +40.85; corrected,
     -0.0817% at t = -21.03.

  2. FILTERING RUNS BY THEIR OWN LENGTH. Dropping runs shorter than
     MIN_RUN selects on the outcome -- short runs are the breakouts that
     failed. solo:boll_break read +0.47%/trade and all three folds "beat
     their rotation null" at p = 0.000. Corrected to WAIT for the
     persistence instead of filtering on it: -0.16%/trade.

Both bugs produced results that passed a rotation null, Bonferroni over
623, walk-forward selection and out-of-sample testing. No amount of
statistical hygiene catches a look-ahead; only reading the accounting
does.

SIZING. A selected strategy's stake is its own measured potential:
half-Kelly on the edge and dispersion it showed in training, floored at
--min-stake of live equity and capped at 100%. The floor is the
operator's requirement -- no trade smaller than 5% -- and the cap means a
strategy that is certain by its own numbers can take the account.
"""
from __future__ import annotations

import sys

import numpy as np

from fp import data as D
from fp import strategies as SG
from fp.data import FEE
from fp.data import FUNDING_PER_8H as FUND

# Two-sided 0.05 spread over 623 attempted strategies.
N_ATTEMPTED = 623
BONF_T = 3.94

# A strategy needs this many independent trades in a window before its
# mean means anything.
MIN_TRADES = 30

# The operator's floor and ceiling on one trade's margin.
MIN_STAKE = 0.05
MAX_STAKE = 1.00

FOLDS = (
    (("2026-05-31", "2026-07-01"), ("2026-07-01", "2026-07-15")),
    (("2026-05-31", "2026-07-15"), ("2026-07-15", "2026-07-30")),
    (("2026-05-31", "2026-07-30"), ("2026-07-30", "2026-08-14")),
)


def measure(window):
    """Every strategy's trades across the whole board, in one window.

    Returns {name: (net array, mean hold minutes)} pooled over coins.
    """
    P = D.load(*window)
    P = {s: d for s, d in P.items() if len(d) > 3000}
    pooled: dict[str, list] = {}
    holds: dict[str, list] = {}
    for sym, d in sorted(P.items()):
        S = SG.all_strategies(d, P, sym)
        c = d["close"].values.astype("float64")
        for name, state in S.items():
            st, en, sd, net = SG.trades(c, state)
            if len(net) == 0:
                continue
            pooled.setdefault(name, []).append(net)
            holds.setdefault(name, []).append(en - st)
    out = {}
    for name, chunks in pooled.items():
        v = np.concatenate(chunks)
        h = np.concatenate(holds[name])
        out[name] = (v, float(h.mean()))
    return out


def stats(v):
    n = len(v)
    if n < 2:
        return dict(n=n, mean=np.nan, t=np.nan, win=np.nan, sd=np.nan)
    sd = v.std(ddof=1)
    se = sd / np.sqrt(n)
    return dict(n=n, mean=float(v.mean()),
                t=float(v.mean() / se) if se > 0 else 0.0,
                win=float((v > 0).mean()), sd=float(sd))


def kelly_stake(mean, sd):
    """Half-Kelly on a measured edge, floored and capped.

    For a bet whose per-trade return has mean m and dispersion s, the
    growth-optimal fraction of notional is m / s^2. Half of it, because
    the edge is an ESTIMATE and full Kelly is optimal only when it is
    known exactly.
    """
    if not np.isfinite(mean) or not np.isfinite(sd) or sd <= 0 or mean <= 0:
        return 0.0
    f = 0.5 * mean / (sd * sd)
    return float(np.clip(f, MIN_STAKE, MAX_STAKE))


def rotation_null(window, names, rounds=40, seed=0):
    """Roll each state; keep the trade count, destroy the timing."""
    rng = np.random.default_rng(seed)
    P = D.load(*window)
    P = {s: d for s, d in P.items() if len(d) > 3000}
    per_round = np.zeros(rounds)
    counts = np.zeros(rounds)
    for sym, d in sorted(P.items()):
        S = SG.all_strategies(d, P, sym)
        c = d["close"].values.astype("float64")
        for name in names:
            state = S.get(name)
            if state is None:
                continue
            for r in range(rounds):
                off = int(rng.integers(1, max(len(state) - 1, 2)))
                _, _, _, net = SG.trades(c, np.roll(state, off))
                if len(net):
                    per_round[r] += net.sum()
                    counts[r] += len(net)
    return per_round / np.maximum(counts, 1)



def portfolio(window, picked, min_stake=MIN_STAKE, max_gross=1.0):
    """One account, positions netted, fees on ACTUAL turnover.

    Summing per-trade returns across selected strategies is not a
    portfolio. Ninety-one strategies built on the same handful of methods
    fire together on the same bar, so the same market exposure gets
    counted dozens of times: the first version of this reported
    "+7219% on equity over 193,745 trades" in a two-week window, which
    would need ninety-one simultaneous positions at 5% or more each.

    So here every selected strategy votes its stake, the votes are NETTED
    per coin, gross exposure is capped at the account, and the fee is
    charged on the CHANGE in position -- which is what an exchange
    actually bills. A hundred strategies all holding the same long pay
    for one long, not a hundred.
    """
    P = D.load(*window)
    P = {s_: d for s_, d in P.items() if len(d) > 3000}
    names = [n for n, _, _ in picked]
    w = {n: kelly_stake(st["mean"], st["sd"]) for n, st, _ in picked}

    per_coin = {}
    for sym, d in sorted(P.items()):
        S = SG.all_strategies(d, P, sym)
        n = len(d)
        want = np.zeros(n)
        for nm in names:
            v = S.get(nm)
            if v is None or w[nm] <= 0:
                continue
            want += w[nm] * v.astype("float64")
        per_coin[sym] = (want, d["close"].values.astype("float64"))

    syms = sorted(per_coin)
    n = min(len(per_coin[s_][0]) for s_ in syms)
    W = np.vstack([per_coin[s_][0][-n:] for s_ in syms])
    C = np.vstack([per_coin[s_][1][-n:] for s_ in syms])

    # Cap gross exposure at the account, and honour the floor: a coin the
    # book wants at all is held at no less than min_stake.
    small = (np.abs(W) > 0) & (np.abs(W) < min_stake)
    W = np.where(small, np.sign(W) * min_stake, W)
    gross = np.abs(W).sum(axis=0)
    scale = np.where(gross > max_gross, max_gross / np.maximum(gross, 1e-9), 1.0)
    W = W * scale

    ret = np.zeros_like(C)
    ret[:, 1:] = C[:, 1:] / C[:, :-1] - 1.0
    pnl = (W[:, :-1] * ret[:, 1:]).sum(axis=0)
    turn = np.abs(np.diff(W, axis=1)).sum(axis=0)
    cost = turn * (FEE / 2.0)          # one side per unit of turnover
    fund = np.abs(W[:, :-1]).sum(axis=0) * (1.0 / 60.0 / 8.0) * FUND
    step = pnl - cost - fund
    eq = float(np.cumprod(1.0 + step)[-1] - 1.0) if len(step) else 0.0
    return {"return_pct": 100 * eq,
            "gross_mean": float(np.abs(W).sum(axis=0).mean()),
            "turnover": float(turn.sum()),
            "fees_pct": 100 * float(cost.sum()),
            "minutes": int(n)}


def main():
    print("=" * 78)
    print("  STRATEGY SEARCH: every method, every combination, priced in full")
    print(f"  Bonferroni over {N_ATTEMPTED} attempted -> |t| >= {BONF_T}")
    print(f"  stake: half-Kelly on the measured edge, floor {100*MIN_STAKE:.0f}%"
          f", cap {100*MAX_STAKE:.0f}%")
    print("=" * 78, flush=True)

    fold_rows = []
    for i, (trw, tew) in enumerate(FOLDS, 1):
        print(f"\n--- FOLD {i}   select on {trw[0]}..{trw[1]}   "
              f"trade {tew[0]}..{tew[1]}", flush=True)
        tr = measure(trw)
        te = measure(tew)

        P = D.load(*tew)
        mv = [100 * (d["close"].iloc[-1] / d["close"].iloc[0] - 1)
              for d in P.values()]
        print(f"  MARKET this window : median coin {np.median(mv):+.2f}%, "
              f"{sum(1 for v in mv if v < 0)}/{len(mv)} down")

        # SELECT on training only.
        picked = []
        for name, (v, hold) in tr.items():
            s = stats(v)
            if s["n"] >= MIN_TRADES and s["mean"] > 0 and s["t"] >= BONF_T:
                picked.append((name, s, hold))
        picked.sort(key=lambda r: -r[1]["t"])
        print(f"  selected on train  : {len(picked)} of {len(tr)} strategies "
              f"clear |t| >= {BONF_T} with {MIN_TRADES}+ trades")

        if not picked:
            print("  nothing to trade forward.")
            fold_rows.append((0, np.nan, np.nan, np.nan))
            continue

        for name, s, hold in picked[:8]:
            print(f"      {name:<34} train n={s['n']:>5} "
                  f"{100*s['mean']:+.4f}%  t={s['t']:+.2f}  "
                  f"hold {hold:.0f}m")

        # TRADE FORWARD, weighted by each strategy's own measured stake.
        raw, staked = [], []
        for name, s, hold in picked:
            got = te.get(name)
            if got is None:
                continue
            v = got[0]
            w = kelly_stake(s["mean"], s["sd"])
            if w <= 0:
                continue
            raw.append(v)              # per-trade return on NOTIONAL
            staked.append(w * v)       # the same trade's return on EQUITY
        if not raw:
            print("  none of them traded in the test window.")
            fold_rows.append((len(picked), np.nan, np.nan, np.nan))
            continue

        st = stats(np.concatenate(raw))
        eq = np.concatenate(staked)
        print(f"  TRADED FORWARD     : {st['n']:>6} trades  "
              f"{100*st['mean']:+.4f}%/trade of notional  t={st['t']:+.2f}  "
              f"win {100*st['win']:.1f}%")
        print(f"  ON EQUITY          : {100*eq.sum():+.2f}% summed over "
              f"{len(eq)} staked trades")

        pf = portfolio(tew, picked)
        print(f"  AS A PORTFOLIO     : {pf['return_pct']:+.2f}% on the account "
              f"over {pf['minutes']:,} minutes")
        print(f"                       avg gross exposure "
              f"{pf['gross_mean']:.2f}x equity, fees {pf['fees_pct']:.2f}%")

        null = rotation_null(tew, [p[0] for p in picked[:12]])
        pv = float((null >= st["mean"]).mean())
        print(f"  NULL (rolled, 40x) : {100*null.mean():+.4f}%/trade   "
              f"p(null >= real) = {pv:.3f}")
        fold_rows.append((len(picked), st["mean"], st["t"], pv))

    print("\n" + "=" * 78)
    ok = [r for r in fold_rows if np.isfinite(r[1])]
    if not ok:
        print("  No strategy survived selection in any fold.")
    else:
        pos = sum(1 for r in ok if r[1] > 0)
        beat = sum(1 for r in ok if r[3] < 0.05)
        print(f"  folds with a selection    : {len(ok)}/{len(FOLDS)}")
        print(f"  folds profitable forward  : {pos}/{len(ok)}")
        print(f"  folds beating their null  : {beat}/{len(ok)}")
        if pos == len(ok) and beat == len(ok):
            print("  EVERY fold paid and beat its null. That is a result.")
        else:
            print("  Not every fold paid. A strategy set that works in some")
            print("  windows and not others is a bet on the window, not an")
            print("  edge, and it is not shipped.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
