"""Does the previous trade's outcome predict the next one? Per period."""
import sys; sys.path.insert(0,"/home/user/Claude-Hello")
import numpy as np, pandas as pd
from fp import logic as L, features as F, methods as M

def rd(p):
    d=pd.read_csv(p,sep=None,engine="python"); d.columns=[c.strip().lower() for c in d.columns]
    d["datetime"]=pd.to_datetime(d[[c for c in d.columns if "time" in c or "date" in c][0]])
    return d[["datetime","open","high","low","close","volume"]]
h26,aug=rd("/home/user/Claude-Hello/bybit_bot/data/BTCUSDT_2026.csv"),rd("/home/user/Claude-Hello/bybit_bot/data/BTCUSDT_202608.csv")
A0=aug.datetime.min()
sets={"2025":(rd("/home/user/Claude-Hello/bybit_bot/data/BTCUSDT_2025.csv"),None),"2026":(h26[h26.datetime<A0],None),
      "AUG":(pd.concat([h26[h26.datetime<A0].tail(30000),aug]),A0)}
COST=L.round_trip_cost(L.DEFAULT_EXIT,False)["total"]

def build(raw,start):
    b=L.to_bars(raw); f=F.build(b); v=M.evaluate_all(f)
    hi,lo,cl=b["high"].values,b["low"].values,b["close"].values
    ap=f["atr14_pct"].values; when=b["datetime"].values
    bar=[];win=[];close=[];g_=[]
    for i in range(250,len(cl)-1):
        if v["consensus_dir"].iloc[i]==M.TIE or int(v["n_methods_fired"].iloc[i])<1: continue
        a=ap[i]
        if not np.isfinite(a) or a<=0: continue
        d=1 if v["consensus_dir"].iloc[i]==M.LONG else -1
        g,bars,_=L.simulate_exit(hi,lo,cl,i,d,(a/100)*cl[i],L.DEFAULT_EXIT,fee=0.0)
        bar.append(i);win.append(1 if g>0 else 0);close.append(i+bars);g_.append(g)
    bar=np.array(bar);win=np.array(win);close=np.array(close);g_=np.array(g_)
    rows=[]
    for k in range(len(bar)):
        i=bar[k]
        if start is not None and when[i]<np.datetime64(start): continue
        done=np.flatnonzero(close[:k]<=i)
        if len(done)==0: continue
        nloss=0
        for j in done[::-1]:
            if win[j]==0: nloss+=1
            else: break
        rows.append((win[k],g_[k],win[done[-1]],min(nloss,4)))
    return pd.DataFrame(rows,columns=["y","g","prev","loss_run"])

D={k:build(r,s) for k,(r,s) in sets.items()}
print(f"Chi phi day du {100*COST:.4f}%,  hoa von gop can win >= 33.3%\n")
print("SAU LENH THUA vs SAU LENH THANG")
print(f"{'ky':>6} {'n sau thua':>11} {'win%':>7} {'net':>9} | {'n sau thang':>12} {'win%':>7} {'net':>9} | {'p-value':>8}")
for k,d in D.items():
    a=d[d.prev==0]; b=d[d.prev==1]
    if len(a)<30 or len(b)<30: continue
    # two-proportion z test, no scipy
    p1,p2=a.y.mean(),b.y.mean(); n1,n2=len(a),len(b)
    pp=(a.y.sum()+b.y.sum())/(n1+n2)
    se=np.sqrt(pp*(1-pp)*(1/n1+1/n2))
    z=(p1-p2)/se if se>0 else 0.0
    p=2*(1-0.5*(1+__import__("math").erf(abs(z)/np.sqrt(2)))) if se>0 else 1.0
    print(f"{k:>6} {len(a):>11,} {100*a.y.mean():>6.1f}% {100*(a.g.mean()-COST):>8.4f}% | "
          f"{len(b):>12,} {100*b.y.mean():>6.1f}% {100*(b.g.mean()-COST):>8.4f}% | {p:>8.4f}")

print("\nTHEO CHUOI THUA LIEN TIEP (0 = lenh truoc thang)")
print(f"{'chuoi thua':>11} " + "".join(f"{k:>22}" for k in D))
print(f"{'':>11} " + "".join(f"{'n     win%     net':>22}" for _ in D))
for r in range(5):
    line=f"{r:>11} "
    for k,d in D.items():
        s=d[d.loss_run==r]
        if len(s)<25: line+=f"{'--':>22}"; continue
        line+=f"{len(s):>7,}{100*s.y.mean():>8.1f}%{100*(s.g.mean()-COST):>7.3f}%"
    print(line)
print("\n  Neu chi trade sau chuoi thua >= 2:")
for k,d in D.items():
    s=d[d.loss_run>=2]
    if len(s)<25: continue
    print(f"    {k:>5}: n={len(s):>6,}  win {100*s.y.mean():>5.1f}%  "
          f"net {100*(s.g.mean()-COST):>+8.4f}%  tong {100*(s.g-COST).sum():>+8.1f}%")
