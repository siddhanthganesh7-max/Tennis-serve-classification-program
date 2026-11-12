"""Sensitivity analysis added in revision: models without forearm pronation/supination (EL_prosup),
the variable most exposed to the missing right upper-arm marker. Uses revised_analysis.py."""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from revised_analysis import *
df=pd.read_csv("serves_all70_flagged.csv"); d=df[~df.bad_capture & ~df.serve_id.isin(DUP_IDS)].copy(); y=d.serve_type.values
F2=[f for f in FEATURES if f!="EL_prosup"]; R2=[f for f in F2 if f!="racket_speed"]
for lab,fs in [("all variables without forearm",F2),("joint ROM only without forearm",R2)]:
    for n,m in MODELS.items():
        p=oof(d,fs,m); print(lab,n,"accuracy %.3f"%accuracy_score(y,p))
