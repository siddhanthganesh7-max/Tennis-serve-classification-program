"""Recomputes joint ROM for every detected serve from the corrected .mot files, using the laboratory's definition
(maximum minus minimum within 1.3 s before to 0.9 s after peak racket-tip speed).
Usage: python recompute_rom.py <trc folder> <mot folder> <lab csv with serve_id, trc, serve, serve_type, rec, file, racket_speed> [--medfilt]
Writes serves70_corrected.csv (or serves70_medfilt.csv with --medfilt, a five-frame median filter)."""
import sys, os, numpy as np, pandas as pd
from mocap_io import read_mot, impacts, rom
trc_dir,mot_dir,lab_csv=sys.argv[1:4]; mf=5 if "--medfilt" in sys.argv else None
lab=pd.read_csv(lab_csv); rows=[]
for rec in lab.trc.unique():
    m=read_mot(os.path.join(mot_dir,rec+".mot")); imp,nframes=impacts(os.path.join(trc_dir,rec+".trc"))
    assert len(m)==nframes, f"{rec}: .mot rows {len(m)} != .trc frames {nframes}"
    for n,(ti,_) in enumerate(imp,1): rows.append(dict(trc=rec,serve=n,**rom(m,ti,medfilt_k=mf)))
cols=["serve_id","trc","serve","serve_type","rec","file","racket_speed"]
out=lab[cols].merge(pd.DataFrame(rows),on=["trc","serve"])
fn="serves70_medfilt.csv" if mf else "serves70_corrected.csv"; out.to_csv(fn,index=False); print(fn,len(out),out.serve_type.value_counts().to_dict())
