"""Shared helpers: read .trc marker files and OpenSim .mot files, detect serves, compute ROM."""
import numpy as np, pandas as pd
VAR={'SH_flex':'arm_flex_r','SH_abd':'arm_add_r','SH_rot':'arm_rot_r','EL_flex':'elbow_flex_r','EL_prosup':'pro_sup_r',
 'LB_ext':'lumbar_extension','LB_bend':'lumbar_bending','LB_rot':'lumbar_rotation',
 'R_HI_flex':'hip_flexion_r','R_HI_abd':'hip_adduction_r','R_HI_rot':'hip_rotation_r','R_KN_flex':'knee_angle_r',
 'R_AN_flex':'ankle_angle_r','R_AN_inv':'subtalar_angle_r','L_HI_flex':'hip_flexion_l','L_HI_abd':'hip_adduction_l',
 'L_HI_rot':'hip_rotation_l','L_KN_flex':'knee_angle_l','L_AN_flex':'ankle_angle_l','L_AN_inv':'subtalar_angle_l'}
def read_trc(p):
    L=open(p,errors="ignore").read().splitlines()
    names=[h.strip() for h in L[3].split("\t")[2:] if h.strip()]; ncol=2+3*len(names); rows=[]
    for l in L[5:]:
        if not l.strip(): continue
        r=l.split("\t"); r+=[""]*(ncol-len(r)); rows.append([float(x) if x.strip() else np.nan for x in r[:ncol]])
    a=np.array(rows); return a[:,1], {n:a[:,2+3*i:5+3*i]/1000 for i,n in enumerate(names)}
def read_mot(p):
    L=open(p).read().splitlines(); i=L.index("endheader")
    return pd.DataFrame(np.loadtxt(L[i+2:]),columns=L[i+1].split("\t"))
def peaks(t,s,thr=15,sep=2.5):
    s2=np.where(np.isnan(s),-1,s); idx=[]
    for i in sorted([i for i in range(1,len(s2)-1) if s2[i]>thr and s2[i]>=s2[i-1] and s2[i]>=s2[i+1]],key=lambda i:-s2[i]):
        if all(abs(t[i]-t[j])>=sep for j in idx): idx.append(i)
    return sorted(idx)
def impacts(trc_path):
    t,M=read_trc(trc_path); s=np.linalg.norm(np.gradient(M["top"],t,axis=0),axis=1)
    return [(float(t[i]),float(s[i])) for i in peaks(t,s)], len(t)
def rom(mot,ti,pre=1.3,post=0.9,medfilt_k=None):
    t=mot.time.values; w=(t>=ti-pre)&(t<=ti+post); out={}
    for k,c in VAR.items():
        x=mot[c].values
        if medfilt_k:
            from scipy.signal import medfilt; x=medfilt(x,medfilt_k)
        x=x[w]; out[k]=float(x.max()-x.min())
    return out
