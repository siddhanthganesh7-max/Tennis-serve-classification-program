"""Sensitivity analyses (Supplementary Table S1): without forearm pronation/supination, ROM after a five-frame
median filter, and trunk and leg ROM only. Usage: python sensitivity.py serves70_corrected.csv serves70_medfilt.csv"""
import sys; files=sys.argv[1:3]; sys.argv=["x"]
from classification_analysis import *
def acc(d,fs,m): return round(accuracy_score(d.serve_type,oof(d,fs,m))*100,1)
NP=[f for f in ROM if f!="EL_prosup"]; TL=[f for f in ROM if not f.startswith("EL")]
for name,fn,allf,romf in [("Primary",files[0],FEATURES,ROM),("Without forearm pronation/supination",files[0],["racket_speed"]+NP,NP),
                          ("Five-frame median filter",files[1],FEATURES,ROM),("Trunk and leg ROM only",files[0],["racket_speed"]+TL,TL)]:
    d=pd.read_csv(fn); sf=d[d.serve_type.isin(["slice","flat"])]
    print(name,{n:(acc(d,allf,m),acc(d,romf,m),acc(sf,romf,m)) for n,m in MODELS.items()})
