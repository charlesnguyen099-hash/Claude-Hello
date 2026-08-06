"""Per method: does the after-a-loss effect differ, and does any method
combined with it clear the cost?"""
import sys, math; sys.path.insert(0,"/home/user/Claude-Hello")
import numpy as np, pandas as pd
from fp import logic as L, features as F, methods as M
D="/home/user/Claude-Hello/bybit_bot/data/"
def rd(p):
    d=pd.read_csv(p,sep=None,engine="python"); d.columns=[c.strip().lower() for c in d.columns]
    d["datetime"]=pd.to_datetime(d[[c for c in d.columns if "time" in c or "date" in c][0]])
    return d[["datetime","open","high","low","close","volume"]]
h26,aug=rd(D+"BTCUSDT_2026.csv"),rd(D+"BTCUSDT_202608.csv")
A0=aug.datetime.min()
sets={"2025":(rd(D+"BTCUSDT_2025.csv"),None),"2026":(h26[h26.datetime<A0],None),
      "AUG":(pd.concat([h26[h26.datetime<A0].tail(30000),aug]),A0)}
COST=L.round_trip_cost(L.DEFAULT_EXIT,False)["total"]

rows=[]
for name,(raw,start) in sets.items():
    b=L.to_bars(raw); f=F.build(b); v=M.evaluate_all(f)
    hi,lo,cl=b["high"].values,b["low"].values,b["close"].values
    ap=f["atr14_pct"].values; when=b["datetime"].values
    # per-method trade history so "previous outcome" is per method
    hist={m:[] for m in M.METHOD_NAMES}   # (close_bar, win)
    for i in range(250,len(cl)-1):
        a=ap[i]
        if not np.isfinite(a) or a<=0: continue
        for m in M.METHOD_NAMES:
            s=v[m].iloc[i]
            if s==M.NONE: continue
            d=1 if s==M.LONG else -1
            g,bars,_=L.simulate_exit(hi,lo,cl,i,d,(a/100)*cl[i],L.DEFAULT_EXIT,fee=0.0)
            done=[w for (cb,w) in hist[m] if cb<=i]
            prev = done[-1] if done else -1
            if start is None or when[i]>=np.datetime64(start):
                rows.append((name,m,g,1 if g>0 else 0,prev))
            hist[m].append((i+bars,1 if g>0 else 0))
t=pd.DataFrame(rows,columns=["set","method","g","y","prev"])

print(f"Chi phi {100*COST:.4f}%.  NET = gross - chi phi.  Duong = co loi.\n")
print(f"{'method':>22} " + "".join(f"{k:>26}" for k in ("2025","2026","AUG")))
print(f"{'':>22} " + "".join(f"{'tat ca  sau-thua  chenh':>26}" for _ in range(3)))
best=[]
for m in M.METHOD_NAMES:
    line=f"{m.split('_',1)[1]:>22} "
    for k in ("2025","2026","AUG"):
        s=t[(t.set==k)&(t.method==m)]
        sl=s[s.prev==0]
        if len(s)<50: line+=f"{'--':>26}"; continue
        na=100*(s.g.mean()-COST); nl=100*(sl.g.mean()-COST) if len(sl)>=30 else float('nan')
        line+=f"{na:>8.3f}{nl:>10.3f}{nl-na:>8.3f}"
        if k!="AUG" and np.isfinite(nl): best.append((m,k,nl))
    print(line)

print("\nTOT NHAT: method + chi trade sau lenh thua cua chinh no")
tab={}
for m in M.METHOD_NAMES:
    w=[]
    for k in ("2025","2026","AUG"):
        s=t[(t.set==k)&(t.method==m)]; sl=s[s.prev==0]
        if len(sl)>=30: w.append(100*(sl.g.mean()-COST))
    if w: tab[m]=min(w)
for m,x in sorted(tab.items(),key=lambda r:-r[1])[:5]:
    print(f"  {m.split('_',1)[1]:>22}: ky te nhat {x:>+8.4f}%  {'CO LOI' if x>0 else ''}")
print(f"\n  So method co loi o MOI ky: {sum(1 for x in tab.values() if x>0)}/{len(tab)}")
