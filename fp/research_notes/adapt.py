"""Learn per-cell: trade forward, trade reversed, or do not trade.

Exactly the proposal. Split trades into cells by what is visible at
entry. In each cell measure what the forward trade returned and what the
reverse would have returned, both after full cost. Then:

    forward net > 0  ->  trade it the old way
    reverse net > 0  ->  trade it the other way
    both <= 0        ->  drop the cell

Fit on 2025+2026, apply unchanged to August. The fit number will look
good -- it is chosen to. Only the August number is evidence.
"""
import sys, math
sys.path.insert(0, "/home/user/Claude-Hello")
import numpy as np, pandas as pd
from fp import logic as L, features as F, methods as M

D = "/home/user/Claude-Hello/bybit_bot/data/"
COST = L.round_trip_cost(L.DEFAULT_EXIT, False)["total"]
ATR_EDGES = [0.25, 0.35, 0.45, 0.60, 0.80, 1.10, 1.50]


def rd(p):
    d = pd.read_csv(p, sep=None, engine="python")
    d.columns = [c.strip().lower() for c in d.columns]
    d["datetime"] = pd.to_datetime(
        d[[c for c in d.columns if "time" in c or "date" in c][0]])
    return d[["datetime", "open", "high", "low", "close", "volume"]]


def collect(raw, start):
    b = L.to_bars(raw); f = F.build(b); v = M.evaluate_all(f)
    hi, lo, cl = b["high"].values, b["low"].values, b["close"].values
    ap = f["atr14_pct"].values; when = b["datetime"].values
    rsi = f["rsi14"].values if "rsi14" in f else np.full(len(cl), 50.0)
    rows, hist = [], []
    for i in range(250, len(cl) - 1):
        if v["consensus_dir"].iloc[i] == M.TIE: continue
        nf = int(v["n_methods_fired"].iloc[i])
        if nf < 1: continue
        a = ap[i]
        if not np.isfinite(a) or a <= 0: continue
        d = 1 if v["consensus_dir"].iloc[i] == M.LONG else -1
        atr = (a / 100) * cl[i]
        g, bars, _ = L.simulate_exit(hi, lo, cl, i, d, atr, L.DEFAULT_EXIT, 0.0)
        gr, _, _ = L.simulate_exit(hi, lo, cl, i, -d, atr, L.DEFAULT_EXIT, 0.0)
        done = [w for (cb, w) in hist if cb <= i]
        prev = done[-1] if done else -1
        if start is None or when[i] >= np.datetime64(start):
            rows.append((
                int(np.searchsorted(ATR_EDGES, a)),      # atr band
                int(prev),                                # previous outcome
                1 if d > 0 else 0,                        # direction
                min(nf, 3),                               # votes, capped
                int(rsi[i] > 50),                         # rsi side
                g, gr))
        hist.append((i + bars, 1 if g > 0 else 0))
    return pd.DataFrame(rows, columns=["atr", "prev", "dir", "votes",
                                       "rsi", "fwd", "rev"])


def learn(t, min_n):
    """Per cell: +1 trade forward, -1 trade reversed, 0 drop."""
    rule = {}
    for key, s in t.groupby(["atr", "prev", "dir", "votes", "rsi"]):
        if len(s) < min_n:
            continue
        f_net = s.fwd.mean() - COST
        r_net = s.rev.mean() - COST
        if max(f_net, r_net) <= 0:
            rule[key] = 0
        else:
            rule[key] = 1 if f_net >= r_net else -1
    return rule


def apply(t, rule):
    took, net = 0, []
    for key, s in t.groupby(["atr", "prev", "dir", "votes", "rsi"]):
        r = rule.get(key, 0)
        if r == 0:
            continue
        col = s.fwd if r > 0 else s.rev
        net.extend((col - COST).values)
        took += len(s)
    if not net:
        return 0, 0.0, 0.0, 0.0
    n = np.array(net)
    return took, 100 * n.mean(), 100 * n.sum(), 100 * (n > 0).mean()


def main():
    h26, aug = rd(D + "BTCUSDT_2026.csv"), rd(D + "BTCUSDT_202608.csv")
    a0 = aug.datetime.min()
    fit = collect(pd.concat([rd(D + "BTCUSDT_2025.csv"),
                             h26[h26.datetime < a0]]), None)
    held = collect(pd.concat([h26[h26.datetime < a0].tail(30000), aug]), a0)
    print(f"fit {len(fit):,} trades, held-out {len(held):,}")
    print(f"full cost {100*COST:.4f}%, baseline net "
          f"{100*(fit.fwd.mean()-COST):+.4f}% fit / "
          f"{100*(held.fwd.mean()-COST):+.4f}% held out\n")

    print(f"{'min n':>6} {'cells':>6} {'fwd':>5} {'rev':>5} {'drop':>5} "
          f"| {'FIT taken':>10} {'net/trade':>10} {'total':>9} "
          f"| {'HELD taken':>11} {'net/trade':>10} {'total':>9}")
    for min_n in (20, 50, 100, 200, 400):
        rule = learn(fit, min_n)
        nf = sum(1 for v in rule.values() if v > 0)
        nr = sum(1 for v in rule.values() if v < 0)
        nd = sum(1 for v in rule.values() if v == 0)
        tk, mu, tot, _ = apply(fit, rule)
        htk, hmu, htot, _ = apply(held, rule)
        print(f"{min_n:>6} {len(rule):>6} {nf:>5} {nr:>5} {nd:>5} "
              f"| {tk:>10,} {mu:>9.4f}% {tot:>8.1f}% "
              f"| {htk:>11,} {hmu:>9.4f}% {htot:>8.1f}%")

    print("\nA fit column that is positive while the held-out column is not")
    print("means the cells were fitted to noise. That is the whole test.")
    print("\nAugust alone is only 150 signals, so the same rule is run")
    print("walk-forward over the fit years: learn on everything before a")
    print("block, trade that block, never look ahead.\n")

    all_t = pd.concat([fit, held], ignore_index=True)
    for min_n in (20, 50, 100, 200):
        oos = []
        folds = 12
        edges = np.linspace(0, len(all_t), folds + 1).astype(int)
        for k in range(1, folds):
            tr = all_t.iloc[:edges[k]]
            te = all_t.iloc[edges[k]:edges[k + 1]]
            rule = learn(tr, min_n)
            for key, s2 in te.groupby(["atr", "prev", "dir", "votes", "rsi"]):
                r = rule.get(key, 0)
                if r == 0:
                    continue
                col = s2.fwd if r > 0 else s2.rev
                oos.extend((col - COST).values)
        if oos:
            o = np.array(oos)
            t_stat = o.mean() / (o.std() / np.sqrt(len(o))) if o.std() > 0 else 0
            print(f"  min n {min_n:>3}: {len(o):>6,} out-of-sample trades   "
                  f"net {100*o.mean():>+8.4f}%/trade   total {100*o.sum():>+8.1f}%"
                  f"   t={t_stat:>5.2f}")
        else:
            print(f"  min n {min_n:>3}: no trades taken out of sample")


if __name__ == "__main__":
    main()
