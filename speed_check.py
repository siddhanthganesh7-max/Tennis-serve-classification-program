import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from revised_analysis import *
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
df=pd.read_csv("serves_all70_flagged.csv"); d51=df[~df.bad_capture]; d=d51[~d51.serve_id.isin(DUP_IDS)].copy()
print(d.groupby("serve_type").racket_speed.describe().round(2))
for a,b in [("kick","slice"),("slice","flat")]:
    lo=d[d.serve_type==b].racket_speed.min(); hi=d[d.serve_type==a].racket_speed.max()
    band=d[(d.racket_speed>=lo)&(d.racket_speed<=hi)&d.serve_type.isin([a,b])]
    print(a,b,"overlap",lo,hi,band.serve_type.value_counts().to_dict(),"recordings",band.file.nunique())
class Resid(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None): self.m=LinearRegression().fit(X[:,:1],X[:,1:]); return self
    def transform(self,X): return X[:,1:]-self.m.predict(X[:,:1])
y=d.serve_type.values
for name,est in [("RF",lambda: make_pipeline(Resid(),RandomForestClassifier(n_estimators=500,random_state=1,min_samples_leaf=2,n_jobs=-1))),
                 ("LR",lambda: make_pipeline(Resid(),StandardScaler(),LogisticRegression(max_iter=2000)))]:
    p=oof(d,FEATURES,est)
    print("speed-residualised ROM",name,round(accuracy_score(y,p),3),round(balanced_accuracy_score(y,p),3),round(f1_score(y,p,average="macro"),3))
# pairwise: kick vs slice only, ROM only, among serves in overlap band -> too small; report counts above
# duplicate pairs: speed vs label
for i,j in [(6,58),(13,32),(14,33),(19,40),(26,61)]:
    a=df[df.serve_id==i].iloc[0]; b=df[df.serve_id==j].iloc[0]
    print(i,a.file,a.serve_type,a.racket_speed,"|",j,b.file,b.serve_type,b.racket_speed)
