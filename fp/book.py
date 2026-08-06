"""Every rule from every study, merged into one book the bot just runs.

    python -m fp.book            # rebuild fp/book.json from the parts
    python -m fp.book --list     # show what is in it

WHY THIS EXISTS

The studies each wrote their own file -- btc_book.json, coin_book.json,
coin_tiers.json -- and running the bot meant choosing one, which is the
wrong question to put to anybody. There is no reason a BTC rule and a
cross-symbol rule cannot run side by side: they are separate rules with
separate barriers, and a book is a set of rules.

So they are merged here, once, and `python run_bot.py` runs all of them
on every symbol it scans. No flag, no choice, nothing to remember.

WHAT MERGING MEANS

    provenance   every rule keeps the study and the data it came from,
                 so a live trade can always be traced to the measurement
                 that justified it
    dedupe       the same (timeframe, entry, side, target, stop, limit)
                 from two studies is ONE rule, and it keeps the wider
                 evidence -- a rule confirmed on nine symbols outranks
                 the same rule confirmed on one
    scope        rules are applied to every symbol. A BTC-fitted rule
                 running on ETHUSDT is not a bug, it is the cross-symbol
                 test the project has been waiting to run, and its
                 provenance says plainly that BTC is where it came from

WHAT IS NOT MERGED AWAY

Every rule in here is FITTED to the data it was found on. Merging does
not launder that and the merged file carries the same in_sample flag the
parts did. What the merge changes is convenience, not evidence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "book.json"

# source file -> what that study established, for provenance
PARTS = {
    # Best evidence first: this is the only study with BOTH a time split
    # and cross-symbol agreement, on a balanced window.
    "tier_book.json": "10 symbols 2026-06-30..08-06, paid in BOTH halves on "
                      "3+ correlated-block members, 2-21x the rotation null",
    "btc_book.json": "BTCUSDT 2025-2026, profitable in both years separately",
    "coin_tiers.json": "9 symbols 2026-07-31..08-06, paid on 7+ of 9",
    "coin_book.json": "9 symbols 2026-07-31..08-06, survived leave-one-out",
}


def key(r: dict) -> tuple:
    return (r["tf"], r.get("name") or r.get("logic"), r["side"],
            float(r["tp"]), float(r["sl"]), int(r["hmax"]))


def evidence(r: dict) -> int:
    """How many independent symbols back this rule. Wider wins a tie."""
    return int(r.get("n_coins") or 1)


def merge() -> dict:
    rules: dict[tuple, dict] = {}
    seen_parts = []
    for fname, what in PARTS.items():
        p = HERE / fname
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        seen_parts.append(f"{fname} ({len(data.get('logics', []))} rules)")
        for r in data.get("logics", []):
            k = key(r)
            merged = {
                "tf": r["tf"],
                "name": r.get("name") or r.get("logic"),
                "side": r["side"],
                "tp": float(r["tp"]),
                "sl": float(r["sl"]),
                "hmax": int(r["hmax"]),
                "hold_min": float(r.get("hold_min") or 0.0),
                "mean": float(r.get("mean") or 0.0),
                "from": fname,
                "evidence": what,
                "n_coins": evidence(r),
            }
            old = rules.get(k)
            if old is None or merged["n_coins"] > old["n_coins"]:
                if old is not None:
                    merged["also_from"] = old["from"]
                rules[k] = merged
            elif old is not None:
                old["also_from"] = fname
    out = sorted(rules.values(),
                 key=lambda r: (-r["n_coins"], -r["mean"]))
    return {
        "built_from": seen_parts,
        "in_sample": True,
        "note": "Every rule is fitted to the data it was found on. Merging "
                "changes convenience, not evidence. Rules run on every "
                "symbol scanned; 'evidence' says where each was measured.",
        "logics": out,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)

    if a.list and OUT.exists():
        book = json.loads(OUT.read_text())
    else:
        book = merge()
        OUT.write_text(json.dumps(book, indent=2))

    rules = book["logics"]
    print(f"{len(rules)} rules merged from: {', '.join(book['built_from'])}")
    tf = {}
    for r in rules:
        tf.setdefault(r["tf"], []).append(r)
    print(f"\n{'tf':>5} {'rules':>6} {'long':>5} {'short':>6} "
          f"{'from':>34}")
    for t in sorted(tf, key=lambda x: len(tf[x]), reverse=True):
        g = tf[t]
        srcs = {}
        for r in g:
            srcs[r["from"]] = srcs.get(r["from"], 0) + 1
        src = ", ".join(f"{k.replace('.json', '')}:{v}"
                        for k, v in sorted(srcs.items()))
        print(f"{t:>5} {len(g):>6} "
              f"{sum(1 for r in g if r['side'] == 'long'):>5} "
              f"{sum(1 for r in g if r['side'] == 'short'):>6} "
              f"{src:>34}")

    if a.list:
        print(f"\n{'tf':>5} {'entry':<28} {'side':>5} {'exit':>16} "
              f"{'coins':>6} {'mean':>8}  source")
        for r in rules:
            exitspec = f"tp{r['tp']}/sl{r['sl']}/{r['hmax']}b"
            print(f"{r['tf']:>5} {r['name']:<28} {r['side']:>5} "
                  f"{exitspec:>16} {r['n_coins']:>6} "
                  f"{100*r['mean']:>7.3f}%  {r['from']}")
    print(f"\nwritten to {OUT.name} -- run_bot.py loads this by default")
    return 0


if __name__ == "__main__":
    sys.exit(main())
