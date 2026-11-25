import pandas as pd, numpy as np, json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, os
os.makedirs("figs",exist_ok=True)
R=json.load(open("results_v2.json")); TYPES=["kick","slice","flat"]
COL={"kick":"#2E5A88","slice":"#E08E0B","flat":"#C0392B"}
LABEL={"racket_speed":"Racket-tip speed","EL_flex":"Right elbow flexion","EL_prosup":"Right forearm pronation/supination","LB_ext":"Lumbar extension",
 "LB_bend":"Lumbar lateral bending","LB_rot":"Lumbar rotation","R_HI_flex":"Right hip flexion","R_HI_abd":"Right hip abduction/adduction",
 "R_HI_rot":"Right hip rotation","R_KN_flex":"Right knee flexion","R_AN_flex":"Right ankle dorsi/plantarflexion","R_AN_inv":"Right ankle inversion/eversion",
 "L_HI_flex":"Left hip flexion","L_HI_abd":"Left hip abduction/adduction","L_HI_rot":"Left hip rotation","L_KN_flex":"Left knee flexion",
 "L_AN_flex":"Left ankle dorsi/plantarflexion","L_AN_inv":"Left ankle inversion/eversion"}
d=pd.read_csv("serves70_corrected.csv"); c=pd.read_csv("compare_lab_vs_corrected.csv")
# Figure 1: originally reported vs corrected ROM
fig,axs=plt.subplots(1,2,figsize=(11,5))
for ax,(v,t) in zip(axs,[("EL_flex","Right elbow flexion ROM (°)"),("R_KN_flex","Right knee flexion ROM (°)")]):
    own=c[c.joint_data_is_own]; wrong=c[~c.joint_data_is_own]
    ax.scatter(own[v],own[v+"_lab"],s=38,color="#2E5A88",edgecolor="white",lw=0.5,label="Values computed from the serve's own recording (30)")
    w1=wrong[~wrong.bad_capture]; w2=wrong[wrong.bad_capture]
    ax.scatter(w1[v],w1[v+"_lab"],s=38,color="#C0392B",edgecolor="white",lw=0.5,label="Values computed from a different recording (21)")
    ax.scatter(w2[v],w2[v+"_lab"],s=44,facecolor="white",edgecolor="#C0392B",lw=1.4,label="As above, and excluded as implausible in the previous version (19)")
    lo=0; hi=max(c[v].max(),c[v+"_lab"].max())+5; ax.plot([lo,hi],[lo,hi],color="#888",lw=0.8,ls="--")
    ax.set_xlim(lo,hi); ax.set_ylim(lo,hi); ax.set_aspect("equal")
    ax.set_xlabel("Corrected value, from the serve's own recording",fontsize=10.5); ax.set_ylabel("Value in the laboratory's original report",fontsize=10.5)
    ax.set_title(t,fontsize=12); ax.spines[["top","right"]].set_visible(False)
h,l=axs[0].get_legend_handles_labels(); fig.legend(h,l,loc="lower center",ncol=1,fontsize=9.5,frameon=False,bbox_to_anchor=(0.5,-0.13))
plt.tight_layout(); plt.savefig("figs/Figure1.png",dpi=300,bbox_inches="tight"); plt.close()
# Figure 2: confusion matrices
fig,axs=plt.subplots(1,2,figsize=(11,5))
for ax,(m,title) in zip(axs,[("RF","Random forest"),("LR","Logistic regression")]):
    x=R["sets"]["all"][m]; cm=np.array(x["cm"]); ax.imshow(cm,cmap="Blues",vmin=0,vmax=25)
    for i in range(3):
        for j in range(3): ax.text(j,i,cm[i,j],ha="center",va="center",fontsize=17,fontweight="bold",color="white" if cm[i,j]>12 else "#222")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels([t.capitalize() for t in TYPES],fontsize=11); ax.set_yticklabels([t.capitalize() for t in TYPES],fontsize=11)
    ax.set_xlabel("Serve type predicted by the model",fontsize=11.5); ax.set_ylabel("Serve type intended by the player",fontsize=11.5)
    ax.set_title(f"{title} (accuracy {x['acc']*100:.1f}%)",fontsize=12)
plt.tight_layout(); plt.savefig("figs/Figure2.png",dpi=300,bbox_inches="tight"); plt.close()
# Figure 3: permutation importance
fig,axs=plt.subplots(1,2,figsize=(13,6.8))
for ax,(m,title) in zip(axs,[("RF","Random forest"),("LR","Logistic regression")]):
    p=pd.read_csv(f"perm_{m}.csv",index_col=0).sort_values("mean")
    ax.barh([LABEL[k] for k in p.index],p["mean"],xerr=p["sd"],color=["#C0392B" if k=="racket_speed" else "#2E5A88" for k in p.index],ecolor="#555",capsize=2)
    ax.axvline(0,color="k",lw=0.8); ax.set_title(title,fontsize=12)
    ax.set_xlabel("Fall in balanced accuracy when the variable is shuffled",fontsize=10.5)
    ax.spines[["top","right"]].set_visible(False); ax.tick_params(labelsize=9.5)
plt.tight_layout(); plt.savefig("figs/Figure3.png",dpi=300,bbox_inches="tight"); plt.close()
# Figure 5: racket speed
rg={t:(d[d.serve_type==t].racket_speed.min(),d[d.serve_type==t].racket_speed.max()) for t in TYPES}
fig,ax=plt.subplots(figsize=(9,3.6)); rng=np.random.default_rng(0)
ax.axvspan(rg["slice"][0],rg["kick"][1],color="#bbbbbb",alpha=0.35,lw=0); ax.axvspan(rg["flat"][0],rg["slice"][1],color="#bbbbbb",alpha=0.35,lw=0)
for i,t in enumerate(TYPES):
    s=d[d.serve_type==t].racket_speed; ax.scatter(s,np.full(len(s),i)+rng.uniform(-0.15,0.15,len(s)),color=COL[t],s=40,edgecolor="white",lw=0.5)
ax.set_yticks(range(3)); ax.set_yticklabels([f"{t.capitalize()} (n = {(d.serve_type==t).sum()})" for t in TYPES],fontsize=11)
ax.set_xlabel("Peak racket-tip speed (m/s)",fontsize=11); ax.invert_yaxis(); ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(); plt.savefig("figs/Figure5.png",dpi=300,bbox_inches="tight"); plt.close()
# Figure 4: four ROM variables by type
V=["EL_flex","EL_prosup","LB_rot","R_HI_rot"]
fig,axs=plt.subplots(1,4,figsize=(13,3.8))
for ax,v in zip(axs,V):
    for i,t in enumerate(TYPES):
        s=d[d.serve_type==t][v]; ax.scatter(np.full(len(s),i)+rng.uniform(-0.15,0.15,len(s)),s,color=COL[t],s=30,edgecolor="white",lw=0.5)
        ax.plot([i-0.25,i+0.25],[s.median()]*2,color="k",lw=1.6)
    ax.set_xticks(range(3)); ax.set_xticklabels([t.capitalize() for t in TYPES],fontsize=10.5)
    ax.set_title(LABEL[v],fontsize=10.5); ax.set_ylabel("ROM (°)",fontsize=10); ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(); plt.savefig("figs/Figure4.png",dpi=300,bbox_inches="tight"); plt.close()
# Figure S1: all 17 ROM variables
ROM=[k for k in LABEL if k!="racket_speed"]
fig,axs=plt.subplots(3,6,figsize=(16,8.5)); axs=axs.ravel()
for ax,v in zip(axs,ROM):
    for i,t in enumerate(TYPES):
        s=d[d.serve_type==t][v]; ax.scatter(np.full(len(s),i)+rng.uniform(-0.15,0.15,len(s)),s,color=COL[t],s=14,edgecolor="none")
        ax.plot([i-0.25,i+0.25],[s.median()]*2,color="k",lw=1.3)
    ax.set_xticks(range(3)); ax.set_xticklabels(["Kick","Slice","Flat"],fontsize=8); ax.set_title(LABEL[v],fontsize=8.5); ax.tick_params(labelsize=8)
    ax.spines[["top","right"]].set_visible(False)
axs[-1].axis("off"); fig.supylabel("ROM (°)",fontsize=10)
plt.tight_layout(); plt.savefig("figs/FigureS1.png",dpi=300,bbox_inches="tight"); plt.close()
print("ok")
