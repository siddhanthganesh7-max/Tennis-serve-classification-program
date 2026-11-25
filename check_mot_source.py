"""Checks each OpenSim .mot file against the raw .trc recordings and identifies which file the laboratory's
reported ROM values came from.
Usage: python check_mot_source.py <trc folder> <mot folder> <lab report csv>
The lab csv has columns serve_id, trc (recording name), serve (number within recording), serve_type and the ROM
columns named as in mocap_io.VAR. Recording and file names are e.g. kick_1, slice_3, flat_5."""
import sys, os, glob, hashlib, numpy as np, pandas as pd
from mocap_io import read_trc, read_mot, impacts, rom, VAR
trc_dir,mot_dir,lab_csv=sys.argv[1:4]
trc={os.path.basename(p)[:-4]:p for p in glob.glob(os.path.join(trc_dir,"*.trc"))}
mot={os.path.basename(p)[:-4]:p for p in glob.glob(os.path.join(mot_dir,"*.mot"))}
print("1. Length check (a .mot file has one row per frame of its .trc file)")
for k in sorted(mot):
    n=len(read_mot(mot[k])); m=len(read_trc(trc[k])[0]) if k in trc else None
    print(f"   {k}: .mot rows {n}, .trc frames {m}", "" if n==m else "  <-- MISMATCH")
print("2. Identical data (ignoring the first header line)")
h={k:hashlib.md5("\n".join(open(p).read().splitlines()[1:]).encode()).hexdigest() for k,p in mot.items()}
for v in set(h.values()):
    same=[k for k in h if h[k]==v]
    if len(same)>1: print("   identical:",same)
print("3. Which recording's joint angles reproduce each reported serve")
lab=pd.read_csv(lab_csv); ks=[k for k in VAR if not k.startswith("SH")]
# only files whose length matches their own recording are used as candidates, so each candidate stands for that recording
M={k:read_mot(p) for k,p in mot.items() if k in trc and len(read_mot(p))==len(read_trc(trc[k])[0])}
print("   candidate files (length matches own recording):",sorted(M))
imp={k:impacts(p)[0] for k,p in trc.items()}; rows=[]
for _,r in lab.iterrows():
    ti=imp[r.trc][r.serve-1][0]; best=[]
    for k,m in M.items():
        tt=m.time.values
        if ((tt>=ti-1.3)&(tt<=ti+0.9)).sum()<50: continue  # window may be cut short at the end of a file
        v=rom(m,ti); best.append((float(np.median([abs(r[c]-v[c]) for c in ks])),k))
    best.sort(); rows.append(dict(serve_id=r.serve_id,recording=r.trc,values_match=best[0][1],median_abs_diff=round(best[0][0],2),next_best=round(best[1][0],2) if len(best)>1 else None))
out=pd.DataFrame(rows); out["own_recording"]=out.values_match==out.recording; out.to_csv("mot_source_check.csv",index=False)
print(out.groupby(out.recording.str.split("_").str[0]).own_recording.agg(["sum","count"]))
