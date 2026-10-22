"""Revised analysis for the shadow-serve classification paper (revised version).
Primary dataset: 43 serves (51 after occlusion exclusion, minus both members of the
four duplicated pairs). Sensitivity: the original 51.
All randomness is seeded; no hyperparameter tuning; no class weighting;
StandardScaler sits inside the pipeline so it is refit on each training fold only."""
import pandas as pd, numpy as np, warnings, json; warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (LeaveOneGroupOut, cross_val_predict, StratifiedGroupKFold,
                                     RepeatedStratifiedKFold, cross_val_score)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support, confusion_matrix)
TYPES=["kick","slice","flat"]
FEATURES=["racket_speed","EL_prosup","LB_ext","LB_bend","LB_rot","R_HI_flex","R_HI_abd","R_HI_rot",
          "R_AN_flex","R_AN_inv","L_HI_flex","L_HI_abd","L_HI_rot","L_AN_flex","L_AN_inv"]
ROM=FEATURES[1:]
DUP_IDS=[6,58,13,32,14,33,19,40]          # four duplicated pairs among retained serves
def rf(): return RandomForestClassifier(n_estimators=500,random_state=1,min_samples_leaf=2,n_jobs=-1)
def lr(): return make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000))
MODELS={"RF":rf,"LR":lr}
def oof(d,feats,m): return cross_val_predict(m(),d[feats].values,d.serve_type.values,cv=LeaveOneGroupOut(),groups=d.file.values)
def boot(y,p,g,metric,B=2000,seed=0):
    rng=np.random.default_rng(seed); G=np.unique(g); idx={k:np.where(g==k)[0] for k in G}; v=[]
    for _ in range(B):
        ii=np.concatenate([idx[k] for k in rng.choice(G,len(G),replace=True)])
        if len(np.unique(y[ii]))<3: continue
        v.append(metric(y[ii],p[ii]))
    return [float(x) for x in np.percentile(v,[2.5,97.5])]
def full(d,feats=FEATURES):
    y=d.serve_type.values; g=d.file.values; R={}
    for n,m in MODELS.items():
        p=oof(d,feats,m)
        pr,rc,f,s=precision_recall_fscore_support(y,p,labels=TYPES,zero_division=0)
        R[n]=dict(acc=accuracy_score(y,p),acc_ci=boot(y,p,g,accuracy_score),
                  bal=balanced_accuracy_score(y,p),bal_ci=boot(y,p,g,balanced_accuracy_score),
                  macro=f1_score(y,p,average="macro"),weighted=f1_score(y,p,average="weighted"),
                  per={t:dict(P=a,R=b,F=c,n=int(k)) for t,a,b,c,k in zip(TYPES,pr,rc,f,s)},
                  cm=confusion_matrix(y,p,labels=TYPES).tolist())
    return R
def ablation(d):
    y=d.serve_type.values; out={}
    for lab,fs in [("all",FEATURES),("speed",["racket_speed"]),("rom",ROM)]:
        out[lab]={}
        for n,m in MODELS.items():
            p=oof(d,fs,m); out[lab][n]=dict(acc=accuracy_score(y,p),bal=balanced_accuracy_score(y,p),macro=f1_score(y,p,average="macro"))
    return out
def perm(d,model,reps=4,seed=0):
    X=d[FEATURES].values; y=d.serve_type.values; g=d.file.values; drops=[]
    for r in range(reps):
        for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=seed+r).split(X,y,g):
            m=model().fit(X[tr],y[tr])
            pi=permutation_importance(m,X[te],y[te],n_repeats=10,random_state=seed+r,scoring="balanced_accuracy",n_jobs=-1)
            drops.append(pi.importances_mean)
    D=np.array(drops); return pd.DataFrame({"mean":D.mean(0),"sd":D.std(0)},index=FEATURES).sort_values("mean",ascending=False)
if __name__=="__main__":
    df=pd.read_csv("serves_all70_flagged.csv")
    d51=df[~df.bad_capture].copy(); d43=d51[~d51.serve_id.isin(DUP_IDS)].copy()
    res={}
    for lab,d in [("n43",d43),("n51",d51)]:
        vc=d.serve_type.value_counts()
        res[lab]=dict(n=len(d),counts=vc.to_dict(),recordings=int(d.file.nunique()),nir=vc.max()/len(d),
                      full=full(d),ablation=ablation(d),
                      random_cv=float(cross_val_score(rf(),d[FEATURES].values,d.serve_type.values,
                                   cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=20,random_state=1)).mean()),
                      speed_range={t:[float(d[d.serve_type==t].racket_speed.min()),float(d[d.serve_type==t].racket_speed.max())] for t in TYPES})
        print(lab,json.dumps(res[lab],indent=1,default=float))
    # sensitivity to thresholds on the duplicate-free data
    sens={}
    for kind,ths in [("elbow",[40,50,55,60,70,85]),("knee",[20,30,40,50,60])]:
        rows=[]
        for t in ths:
            e,k=(t,40) if kind=="elbow" else (55,t)
            d=df[(df.EL_flex>=e)&(df.R_KN_flex>=k)&(df.L_KN_flex>=k)&~df.serve_id.isin(DUP_IDS)]
            y=d.serve_type.values
            pa=oof(d,FEATURES,rf); ps=oof(d,["racket_speed"],rf); pr_=oof(d,ROM,rf)
            rows.append(dict(t=t,n=len(d),counts=d.serve_type.value_counts().to_dict(),nir=d.serve_type.value_counts().max()/len(d),
                             all=accuracy_score(y,pa),all_bal=balanced_accuracy_score(y,pa),
                             speed=accuracy_score(y,ps),rom=accuracy_score(y,pr_),rom_bal=balanced_accuracy_score(y,pr_)))
            print(kind,rows[-1])
        sens[kind]=rows
    res["sens"]=sens
    # out-of-fold permutation importance, primary data
    for n,m in MODELS.items():
        pi=perm(d43,m); pi.to_csv(f"perm_{n}_n43.csv"); print(n,pi.round(3).to_string())
    json.dump(res,open("revised_results.json","w"),indent=1,default=float)
