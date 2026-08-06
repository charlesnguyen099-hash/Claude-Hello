"""Do PRIOR signals carry information the current bar does not?

Everything tested so far reads the bar in front of the entry. This reads
the run-up: how long since the last signal, how the last few resolved,
whether the direction has been repeating. Strictly past-only -- a prior
trade counts only if it CLOSED at or before the entry bar.
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

def build(raw,start):
    b=L.to_bars(raw); f=F.build(b); v=M.evaluate_all(f)
    hi,lo,cl=b["high"].values,b["low"].values,b["close"].values
    ap=f["atr14_pct"].values; when=b["datetime"].values; n=len(cl)
    # every consensus bar, with outcome and the bar it closed on
    sig_bar=[];sig_dir=[];sig_win=[];sig_close=[]
    for i in range(250,n-1):
        if v["consensus_dir"].iloc[i]==M.TIE or int(v["n_methods_fired"].iloc[i])<1: continue
        a=ap[i]
        if not np.isfinite(a) or a<=0: continue
        d=1 if v["consensus_dir"].iloc[i]==M.LONG else -1
        g,bars,_=L.simulate_exit(hi,lo,cl,i,d,(a/100)*cl[i],L.DEFAULT_EXIT,fee=0.0)
        sig_bar.append(i);sig_dir.append(d);sig_win.append(1 if g>0 else 0);sig_close.append(i+bars)
    sig_bar=np.array(sig_bar);sig_dir=np.array(sig_dir)
    sig_win=np.array(sig_win);sig_close=np.array(sig_close)
    rows=[]
    for k in range(len(sig_bar)):
        i=sig_bar[k]
        if start is not None and when[i]<np.datetime64(start): continue
        # prior signals whose trade had CLOSED by bar i -- no lookahead
        done=np.flatnonzero((sig_close[:k]<=i))
        prev_win = sig_win[done[-1]] if len(done) else -1
        prev3 = sig_win[done[-3:]].mean() if len(done)>=3 else -1
        prev10 = sig_win[done[-10:]].mean() if len(done)>=10 else -1
        gap = i-sig_bar[k-1] if k>0 else -1
        # direction persistence among the last 5 signals (known at entry)
        last5 = sig_dir[max(0,k-5):k]
        same5 = float((last5==sig_dir[k]).mean()) if len(last5) else -1
        streak=0
        for j in range(k-1,-1,-1):
            if sig_dir[j]==sig_dir[k]: streak+=1
            else: break
        dens20=int(((sig_bar[:k]>=i-20)&(sig_bar[:k]<i)).sum())
        dens50=int(((sig_bar[:k]>=i-50)&(sig_bar[:k]<i)).sum())
        rows.append(dict(y=sig_win[k],prev_win=prev_win,prev3=prev3,prev10=prev10,
                         gap=gap,same_dir_5=same5,dir_streak=streak,
                         density_20=dens20,density_50=dens50,
                         n_closed=len(done)))
    return pd.DataFrame(rows)

def auc(x,y):
    m=np.isfinite(x)&(x>=0) if x.min()<0 else np.isfinite(x)
    if m.sum()<200 or len(np.unique(y[m]))<2: return np.nan,0
    r=pd.Series(x[m]).rank().values; yy=y[m]
    n1,n0=yy.sum(),(1-yy).sum()
    if n1==0 or n0==0: return np.nan,0
    return (r[yy==1].sum()-n1*(n1+1)/2)/(n1*n0), int(m.sum())

D={k:build(r,s) for k,(r,s) in sets.items()}
for k,d in D.items(): print(f"{k}: {len(d):,} tin hieu, {100*d.y.mean():.1f}% thang")
feats=[c for c in D["FIT"].columns if c!="y"]
print(f"\nBOI CANH TIN HIEU TRUOC DO ({len(feats)} dac trung moi)")
print(f"{'dac trung':>14} {'n':>7} {'AUC fit':>9} {'AUC AUG':>9} {'giu dau':>9}")
best=0
for c in feats:
    af,nf=auc(D["FIT"][c].values.astype(float),D["FIT"].y.values)
    aa,_=auc(D["AUG"][c].values.astype(float),D["AUG"].y.values)
    if not np.isfinite(af): continue
    best=max(best,abs(af-0.5))
    same = np.isfinite(aa) and (af-0.5)*(aa-0.5)>0
    print(f"{c:>14} {nf:>7,} {af:>9.4f} {aa:>9.4f} {'CO' if same else 'DAO':>9}")
se=0.5/np.sqrt(min((D['FIT'].y==1).sum(),(D['FIT'].y==0).sum()))
print(f"\n  nguong y nghia (Bonferroni {len(feats)} phep thu): {0.5+2.6*se:.4f}")
print(f"  AUC lech nhat: {0.5+best:.4f}")

print("\nWIN RATE THEO KET QUA LENH TRUOC (co cum thang khong?)")
for k,d in D.items():
    for col,lab in (("prev_win","lenh truoc"),):
        s=d[d[col]>=0]
        if len(s)<100: continue
        a=s[s[col]==1].y.mean(); b=s[s[col]==0].y.mean()
        print(f"  {k}: sau lenh THANG {100*a:.1f}%   sau lenh THUA {100*b:.1f}%   "
              f"chenh {100*(a-b):+.2f}pp   (n={len(s):,})")
