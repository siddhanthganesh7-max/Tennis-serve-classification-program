import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from revised_analysis import *
df=pd.read_csv("serves_all70_flagged.csv"); d51=df[~df.bad_capture]; d=d51[~d51.serve_id.isin(DUP_IDS)].copy()
y=d.serve_type.values; g=d.file.values
for name,m in MODELS.items():
    pa=oof(d,FEATURES,m); ps=oof(d,["racket_speed"],m)
    rng=np.random.default_rng(0); G=np.unique(g); idx={k:np.where(g==k)[0] for k in G}; v=[]
    for _ in range(2000):
        ii=np.concatenate([idx[k] for k in rng.choice(G,len(G),replace=True)])
        v.append((pa[ii]==y[ii]).mean()-(ps[ii]==y[ii]).mean())
    print(name,"all-speed diff",round((pa==y).mean()-(ps==y).mean(),3),np.percentile(v,[2.5,97.5]).round(3))
    print(" speed-only cm",confusion_matrix(y,ps,labels=TYPES).tolist())
