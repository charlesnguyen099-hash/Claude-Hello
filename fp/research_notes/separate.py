"""Can ANYTHING visible at entry tell the winners from the losers?

That is the whole question behind "just trade the winning ones". You can
only do it if winners look different from losers BEFORE the outcome.
106 features, ranked by how well each separates them.
"""
import sys; sys.path.insert(0,"/home/user/Claude-Hello")
import numpy as np, pandas as pd
from fp import logic as L, features as F, methods as M

def rd(p):
    d=pd.read_csv(p,sep=None,engine="python"); d.columns=[c.strip().lower() for c in d.columns]
    d["datetime"]=pd.to_datetime(d[[c for c in d.columns if "time" in c or "date" in c][0]])
    return d[["datetime","open","high","low","close","volume"]]
h26,aug=rd("bybit_bot/data/BTCUSDT_2026.csv"),rd("bybit_bot/data/BTCUSDT_202608.csv")
A0=aug.datetime.min()
sets={"FIT":(pd.concat([rd("bybit_bot/data/BTCUSDT_2025.csv"),h26[h26.datetime<A0]]),None),
      "AUG":(pd.concat([h26[h26.datetime<A0].tail(30000),aug]),A0)}
out={}
for name,(raw,start) in sets.items():
    b=L.to_bars(raw); f=F.build(b); v=M.evaluate_all(f)
    hi,lo,cl=b["high"].values,b["low"].values,b["close"].values
    ap=f["atr14_pct"].values; when=b["datetime"].values
    idx,y=[],[]
    for i in range(250,len(cl)-1):
        if start is not None and when[i]<np.datetime64(start): continue
        if v["consensus_dir"].iloc[i]==M.TIE or int(v["n_methods_fired"].iloc[i])<1: continue
        a=ap[i]
        if not np.isfinite(a) or a<=0: continue
        d=1 if v["consensus_dir"].iloc[i]==M.LONG else -1
        g,_,_=L.simulate_exit(hi,lo,cl,i,d,(a/100)*cl[i],L.DEFAULT_EXIT,fee=0.0)
        idx.append(i); y.append(1 if g>0 else 0)
    out[name]=(f.iloc[idx].reset_index(drop=True), np.array(y))
    print(f"{name}: {len(y):,} trades, {100*np.mean(y):.1f}% winners")

def auc(x,y):
    m=np.isfinite(x)
    if m.sum()<100 or len(np.unique(y[m]))<2: return 0.5
    r=pd.Series(x[m]).rank().values; yy=y[m]
    n1,n0=yy.sum(),(1-yy).sum()
    if n1==0 or n0==0: return 0.5
    return (r[yy==1].sum()-n1*(n1+1)/2)/(n1*n0)

Xf,yf=out["FIT"]; Xa,ya=out["AUG"]
cols=[c for c in Xf.columns if Xf[c].dtype.kind in "fi"]
res=[]
for c in cols:
    a_fit=auc(Xf[c].values.astype(float),yf)
    a_aug=auc(Xa[c].values.astype(float),ya) if c in Xa.columns else 0.5
    res.append((c,a_fit,a_aug,abs(a_fit-0.5)))
res.sort(key=lambda r:-r[3])
n=len(res)
# 95% CI half-width for AUC with this sample size
se=0.5/np.sqrt(min((yf==1).sum(),(yf==0).sum()))
thr=0.5+1.96*se*np.sqrt(2)*np.sqrt(np.log(n))/np.sqrt(2)  # Bonferroni-ish
print(f"\n{n} dac trung. AUC 0.50 = khong phan biet duoc gi.")
print(f"Nguong y nghia sau hieu chinh {n} phep thu: {thr:.4f}\n")
print(f"{'dac trung':>28} {'AUC fit':>9} {'AUC AUG':>9} {'y nghia?':>10} {'giu dau?':>10}")
for c,af,aa,_ in res[:15]:
    sig = abs(af-0.5) > (thr-0.5)
    same = (af-0.5)*(aa-0.5) > 0
    print(f"{c[:28]:>28} {af:>9.4f} {aa:>9.4f} {'CO' if sig else '':>10} {'CO' if same else 'DAO':>10}")
best=max(abs(r[1]-0.5) for r in res)
print(f"\n  AUC lech nhieu nhat: {0.5+best:.4f}  (nguong can: {thr:.4f})")
nsig=sum(1 for c,af,aa,_ in res if abs(af-0.5)>(thr-0.5))
nsame=sum(1 for c,af,aa,_ in res if abs(af-0.5)>(thr-0.5) and (af-0.5)*(aa-0.5)>0)
print(f"  Vuot nguong: {nsig}/{n}    trong do giu dau tren AUG: {nsame}")
