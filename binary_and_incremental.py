import sys; sys.argv=['x']
from classification_analysis import *
import json
d=pd.read_csv('serves70_corrected.csv'); y=d.serve_type.values; g=d.file.values
def diffboot(y,p1,p2,g,B=2000,seed=0):
    rng=np.random.default_rng(seed); G=np.unique(g); idx={k:np.where(g==k)[0] for k in G}; v=[]
    for _ in range(B):
        ii=np.concatenate([idx[k] for k in rng.choice(G,len(G),replace=True)])
        if len(np.unique(y[ii]))<3: continue
        v.append(accuracy_score(y[ii],p1[ii])-accuracy_score(y[ii],p2[ii]))
    return [float(x) for x in np.percentile(v,[2.5,97.5])]
out={}
for n,m in MODELS.items():
    pa=oof(d,FEATURES,m); ps=oof(d,["racket_speed"],m)
    out['inc_'+n]=dict(diff=accuracy_score(y,pa)-accuracy_score(y,ps),ci=diffboot(y,pa,ps,g))
sf=d[d.serve_type.isin(['slice','flat'])].reset_index(drop=True); ys=sf.serve_type.values; gs=sf.file.values
def bboot(y,p,g,B=2000,seed=0):
    rng=np.random.default_rng(seed); G=np.unique(g); idx={k:np.where(g==k)[0] for k in G}; v=[]
    for _ in range(B):
        ii=np.concatenate([idx[k] for k in rng.choice(G,len(G),replace=True)])
        if len(np.unique(y[ii]))<2: continue
        v.append(accuracy_score(y[ii],p[ii]))
    return [float(x) for x in np.percentile(v,[2.5,97.5])]
for lab,fs in [("all",FEATURES),("speed",["racket_speed"]),("rom",ROM)]:
    for n,m in MODELS.items():
        p=oof(sf,fs,m); out[f'sf_{lab}_{n}']=dict(acc=accuracy_score(ys,p),ci=bboot(ys,p,gs))
# kick vs not-kick ROM only
kb=d.copy(); kb['serve_type']=np.where(kb.serve_type=='kick','kick','other')
for n,m in MODELS.items():
    p=cross_val_predict(m(),kb[ROM].values,kb.serve_type.values,cv=LeaveOneGroupOut(),groups=kb.file.values)
    out['kick_rom_'+n]=accuracy_score(kb.serve_type,p)
# speed overlaps
r={t:(d[d.serve_type==t].racket_speed.min(),d[d.serve_type==t].racket_speed.max()) for t in TYPES}
ks=d[(d.racket_speed>=r['slice'][0])&(d.racket_speed<=r['kick'][1])&d.serve_type.isin(['kick','slice'])]
sfo=d[(d.racket_speed>=r['flat'][0])&(d.racket_speed<=r['slice'][1])&d.serve_type.isin(['slice','flat'])]
out['overlap_ks']=dict(n=len(ks),counts=ks.serve_type.value_counts().to_dict(),recs=int(ks.file.nunique()))
out['overlap_sf']=dict(n=len(sfo),counts=sfo.serve_type.value_counts().to_dict(),recs=int(sfo.file.nunique()))
json.dump(out,open('extra2.json','w'),indent=1,default=float); print(json.dumps(out,indent=1,default=float))
