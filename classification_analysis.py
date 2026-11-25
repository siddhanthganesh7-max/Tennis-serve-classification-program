"""Corrected analysis (revision 2). 70 serves (20 kick, 25 slice, 25 flat) from 15 recordings.
Joint ROM recomputed from the corrected OpenSim .mot files (max - min within impact-1.3 s .. impact+0.9 s).
No exclusions; shoulder variables omitted (frame-to-frame discontinuities). No tuning, no class weighting,
StandardScaler inside the pipeline. All randomness seeded."""
import pandas as pd, numpy as np, warnings, json, sys; warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict, StratifiedGroupKFold, RepeatedStratifiedKFold, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
TYPES=["kick","slice","flat"]
ROM=["EL_flex","EL_prosup","LB_ext","LB_bend","LB_rot","R_HI_flex","R_HI_abd","R_HI_rot","R_KN_flex","R_AN_flex","R_AN_inv",
     "L_HI_flex","L_HI_abd","L_HI_rot","L_KN_flex","L_AN_flex","L_AN_inv"]
FEATURES=["racket_speed"]+ROM
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
def evaluate(d,feats):
    y=d.serve_type.values; g=d.file.values; R={}
    for n,m in MODELS.items():
        p=oof(d,feats,m); pr,rc,f,s=precision_recall_fscore_support(y,p,labels=TYPES,zero_division=0)
        R[n]=dict(acc=accuracy_score(y,p),acc_ci=boot(y,p,g,accuracy_score),bal=balanced_accuracy_score(y,p),
                  bal_ci=boot(y,p,g,balanced_accuracy_score),macro=f1_score(y,p,average="macro"),
                  per={t:dict(P=a,R=b,F=c,n=int(k)) for t,a,b,c,k in zip(TYPES,pr,rc,f,s)},
                  cm=confusion_matrix(y,p,labels=TYPES).tolist())
    return R
def perm(d,model,feats=FEATURES,reps=4,seed=0):
    X=d[feats].values; y=d.serve_type.values; g=d.file.values; drops=[]
    for r in range(reps):
        for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=seed+r).split(X,y,g):
            m=model().fit(X[tr],y[tr])
            drops.append(permutation_importance(m,X[te],y[te],n_repeats=10,random_state=seed+r,scoring="balanced_accuracy",n_jobs=-1).importances_mean)
    D=np.array(drops); return pd.DataFrame({"mean":D.mean(0),"sd":D.std(0)},index=feats).sort_values("mean",ascending=False)
if __name__=="__main__":
    d=pd.read_csv(sys.argv[1] if len(sys.argv)>1 else "serves70_corrected.csv")
    vc=d.serve_type.value_counts(); res=dict(n=len(d),counts=vc.to_dict(),recordings=int(d.file.nunique()),nir=vc.max()/len(d))
    res["sets"]={lab:evaluate(d,fs) for lab,fs in [("all",FEATURES),("speed",["racket_speed"]),("rom",ROM),
              ("all_no_prosup",[f for f in FEATURES if f!="EL_prosup"]),("rom_no_prosup",[f for f in ROM if f!="EL_prosup"])]}
    res["random_cv_RF"]=float(cross_val_score(rf(),d[FEATURES].values,d.serve_type.values,cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=20,random_state=1)).mean())
    res["speed_range"]={t:[float(d[d.serve_type==t].racket_speed.min()),float(d[d.serve_type==t].racket_speed.max())] for t in TYPES}
    res["rom_median"]={t:d[d.serve_type==t][ROM].median().round(1).to_dict() for t in TYPES}
    for n,m in MODELS.items():
        pi=perm(d,m); pi.to_csv(f"perm_{n}.csv"); res["perm_"+n]=pi.round(4).to_dict()
    json.dump(res,open("results_v2.json","w"),indent=1,default=float)
    for lab,R in res["sets"].items():
        print(lab, {n:(round(r["acc"],3),[round(x,3) for x in r["acc_ci"]],round(r["bal"],3),round(r["macro"],3)) for n,r in R.items()})
    print(json.dumps({k:v for k,v in res.items() if k not in ("sets",)},indent=1,default=float)[:3000])
