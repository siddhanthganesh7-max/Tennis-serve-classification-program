"""Supplementary Figure S1: laboratory elbow flexion ROM vs marker-based elbow flexion ROM.
Run raw_marker_check.py first; it writes marker_check_serves.csv."""
import pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
m=pd.read_csv("marker_check_serves.csv")
fig,ax=plt.subplots(figsize=(7.5,6)); mark={"kick":"o","slice":"s","flat":"^"}
for t in ["kick","slice","flat"]:
    for bad,col,lbl in [(False,"#2E5A88","retained"),(True,"#C0392B","excluded")]:
        d=m[(m.serve_type==t)&(m.bad_capture==bad)]
        if len(d): ax.scatter(d.m_EL,d.EL_flex,c=col,marker=mark[t],s=55,edgecolor="white",lw=0.6,label=f"{t.capitalize()}, {lbl}")
ax.set_xlabel("Elbow flexion ROM from raw markers (°)",fontsize=12); ax.set_ylabel("Elbow flexion ROM in laboratory report (°)",fontsize=12)
ax.set_xlim(35,110); ax.legend(fontsize=9.5,frameon=False,loc="lower right"); ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(); plt.savefig("FigureS1.png",dpi=300,bbox_inches="tight")
