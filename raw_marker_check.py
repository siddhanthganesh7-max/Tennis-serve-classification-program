"""Raw marker (.trc) check used in the revision.
Reapplies the laboratory's serve-detection rule (racket-tip 'top' marker speed peaks > 15 m/s,
>= 2.5 s apart; window impact-1.3 s .. impact+0.9 s), compares racket speed with the laboratory's
values, computes a simple marker-based elbow flexion ROM (shoulder-top, elbow-centre, wrist-centre),
and reports right upper-arm marker (Q_RArm) loss by serve type.
Usage: python raw_marker_check.py <folder with .trc files> serves_all70_flagged.csv"""
import sys, os, numpy as np, pandas as pd
def read_trc(p):
    L=open(p,errors="ignore").read().splitlines()
    names=[h.strip() for h in L[3].split("\t")[2:] if h.strip()]
    ncol=2+3*len(names); rows=[]
    for l in L[5:]:
        if not l.strip(): continue
        r=l.split("\t"); r+=[""]*(ncol-len(r))
        rows.append([float(x) if x.strip() else np.nan for x in r[:ncol]])
    a=np.array(rows); return a[:,1], {n:a[:,2+3*i:5+3*i]/1000 for i,n in enumerate(names)}
def peaks(t,s,thr=15,sep=2.5):
    s2=np.where(np.isnan(s),-1,s); idx=[]
    for i in sorted([i for i in range(1,len(s2)-1) if s2[i]>thr and s2[i]>=s2[i-1] and s2[i]>=s2[i+1]],key=lambda i:-s2[i]):
        if all(abs(t[i]-t[j])>=sep for j in idx): idx.append(i)
    return sorted(idx)
def ang(a,b,c):
    u,v=a-b,c-b; return np.degrees(np.arccos(np.clip(np.sum(u*v,1)/(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)),-1,1)))
raw,labf=sys.argv[1],sys.argv[2]; out=[]
for f in sorted(x for x in os.listdir(raw) if x.endswith(".trc")):
    t,M=read_trc(os.path.join(raw,f)); s=np.linalg.norm(np.gradient(M["top"],t,axis=0),axis=1)
    el=180-ang(M["Q_RShoulderTop"],(M["Q_RElbowOut"]+M["Q_RElbowIn"])/2,(M["Q_RWristIn"]+M["Q_RWristOut"])/2)
    for n,i in enumerate(peaks(t,s),1):
        w=(t>=t[i]-1.3)&(t<=t[i]+0.9)
        out.append(dict(trc=f[:-4],serve=n,impact_t=float(t[i]),speed=float(s[i]),
            m_EL=float(np.nanmax(el[w])-np.nanmin(el[w])) if np.isfinite(el[w]).any() else np.nan,
            RArm_missing=float(np.isnan(M["Q_RArm"][w][:,0]).mean())))
    print(f, "serves detected:", sum(o["trc"]==f[:-4] for o in out))
mk=pd.DataFrame(out); lab=pd.read_csv(labf)
kmap={"Measurement_1":"kick_1","Measurement_3":"kick_2","Measurement_4":"kick_3","Measurement_5":"kick_4","Measurement_6":"kick_5"}
lab["trc"]=[kmap.get(r,r) if t=="kick" else r for r,t in zip(lab.rec,lab.serve_type)]
m=lab.merge(mk,on=["trc","serve"])
print("matched serves:",len(m)); m.to_csv("marker_check_serves.csv",index=False)
print("racket speed r = %.3f, max |difference| = %.2f m/s"%(np.corrcoef(m.racket_speed,m.speed)[0,1],(m.racket_speed-m.speed).abs().max()))
e,r=m[m.bad_capture],m[~m.bad_capture]
print("excluded: marker elbow ROM median %.0f (range %.0f-%.0f, n=%d); laboratory median %.0f"%(e.m_EL.median(),e.m_EL.min(),e.m_EL.max(),e.m_EL.notna().sum(),e.EL_flex.median()))
print("retained: marker elbow ROM median %.0f"%r.m_EL.median())
for t in ["kick","slice","flat"]:
    d=r[r.serve_type==t][["EL_flex","m_EL"]].dropna(); print(t,"lab vs marker elbow r = %.2f"%np.corrcoef(d.EL_flex,d.m_EL)[0,1])
print("serves with right upper-arm marker missing for >99% of window, by type:")
print(m.assign(lost=m.RArm_missing>0.99).groupby(["serve_type","trc"]).lost.mean().round(2).to_string())
