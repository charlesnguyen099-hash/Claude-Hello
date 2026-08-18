import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_full_bot as B
from fp.live_full import Decision

PASS = 0
def chk(c, m):
    global PASS
    assert c, m
    PASS += 1

# --- every granted trade is >= 5% of EQUITY, not of a shrinking free pool ---
acct = B.Account(equity=10.0, start=10.0)
low_pot = Decision(symbol="AUSDT", side=1, potential=1.0, stake=0.05,
                   leverage=2.0, target=0.02, stop=0.011, trail=0.005, cost=0.0011)
p1 = acct.open_position(low_pot, 100.0)
chk(p1 is not None, "a valid low-potential signal must still open")
chk(abs(p1.margin - 0.5) < 1e-9, f"5% of $10 equity is $0.50, got {p1.margin}")

# Open a second, larger position first so free equity shrinks --
# the NEXT low-potential trade must still be sized off total equity.
big = Decision(symbol="BUSDT", side=1, potential=90.0, stake=0.9,
              leverage=5.0, target=0.05, stop=0.011, trail=0.005, cost=0.0011)
acct2 = B.Account(equity=10.0, start=10.0)
acct2.open_position(big, 100.0)          # commits $9.00, leaves $1.00 free
chk(abs(acct2.free - 1.0) < 1e-9, f"free should be $1.00, got {acct2.free}")
p2 = acct2.open_position(low_pot, 100.0)  # 5% of $10 = $0.50, well within the $1 free
chk(p2 is not None, "must still open when the 5% floor fits in what's free")
chk(abs(p2.margin - 0.5) < 1e-9,
    f"must be 5% of EQUITY ($0.50), not 5% of free ($0.05), got {p2.margin}")

# When free cannot cover the 5% floor, the trade is REJECTED, not shrunk.
acct3 = B.Account(equity=10.0, start=10.0)
huge = Decision(symbol="CUSDT", side=1, potential=97.0, stake=0.97,
               leverage=5.0, target=0.05, stop=0.011, trail=0.005, cost=0.0011)
acct3.open_position(huge, 100.0)          # leaves $0.30 free, under the $0.50 floor
before = len(acct3.open)
p3 = acct3.open_position(low_pot, 100.0)
chk(p3 is None, "a trade that cannot meet the 5% floor must be skipped, not shrunk")
chk(acct3.rejected == 1, "it must count as rejected")
chk(len(acct3.open) == before, "the account must not gain a phantom position")

# --- the account can still never be committed past 100% of equity ---
acct4 = B.Account(equity=10.0, start=10.0)
opened = 0
for i in range(30):
    d = Decision(symbol=f"D{i}USDT", side=1, potential=100.0, stake=1.0,
                leverage=10.0, target=0.05, stop=0.011, trail=0.005, cost=0.0011)
    if acct4.open_position(d, 100.0):
        opened += 1
chk(acct4.committed <= acct4.equity + 1e-9,
    f"committed {acct4.committed} must never exceed equity {acct4.equity}")
chk(opened == 1, f"100%-stake signals leave no free room for a second, got {opened}")

print(f"all {PASS} checks passed")


def main():
    return 0


if __name__ == "__main__":
    sys.exit(main())
