import pandas as pd, numpy as np, json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R=json.load(open("revised_results.json")); TYPES=["kick","slice","flat"]
LABEL={"racket_speed":"Racket-tip speed","EL_prosup":"Right elbow pronation/supination","LB_ext":"Lumbar extension",
 "LB_bend":"Lumbar lateral bending","LB_rot":"Lumbar rotation","R_HI_flex":"Right hip flexion","R_HI_abd":"Right hip abduction/adduction",
 "R_HI_rot":"Right hip rotation","R_AN_flex":"Right ankle dorsi/plantarflexion","R_AN_inv":"Right ankle inversion/eversion",
 "L_HI_flex":"Left hip flexion","L_HI_abd":"Left hip abduction/adduction","L_HI_rot":"Left hip rotation",
 "L_AN_flex":"Left ankle dorsi/plantarflexion","L_AN_inv":"Left ankle inversion/eversion"}
# Figure 4
fig,axs=plt.subplots(1,2,figsize=(11,5))
for ax,(m,title) in zip(axs,[("RF","Random forest"),("LR","Logistic regression")]):
    cm=np.array(R["n43"]["full"][m]["cm"]); ax.imshow(cm,cmap="Blues",vmin=0,vmax=16)
    for i in range(3):
        for j in range(3):
            ax.text(j,i,cm[i,j],ha="center",va="center",fontsize=17,fontweight="bold",color="white" if cm[i,j]>8 else "#222")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels([t.capitalize() for t in TYPES],fontsize=11); ax.set_yticklabels([t.capitalize() for t in TYPES],fontsize=11)
    ax.set_xlabel("Serve type predicted by the model",fontsize=11.5); ax.set_ylabel("Serve type intended by the player",fontsize=11.5)
    ax.set_title(f"{title} (accuracy {R['n43']['full'][m]['acc']*100:.1f}%)",fontsize=12)
plt.tight_layout(); plt.savefig("figs/Figure4.png",dpi=300,bbox_inches="tight"); plt.close()
# Figure 5
fig,axs=plt.subplots(1,2,figsize=(13,6.4),sharey=False)
for ax,(m,title) in zip(axs,[("RF","Random forest"),("LR","Logistic regression")]):
    p=pd.read_csv(f"perm_{m}_n43.csv",index_col=0).sort_values("mean")
    col=["#C0392B" if k=="racket_speed" else "#2E5A88" for k in p.index]
    ax.barh([LABEL[k] for k in p.index],p["mean"],xerr=p["sd"],color=col,ecolor="#555",capsize=2)
    ax.axvline(0,color="k",lw=0.8); ax.set_title(title,fontsize=12)
    ax.set_xlabel("Fall in balanced accuracy when the variable is shuffled",fontsize=10.5)
    ax.spines[["top","right"]].set_visible(False); ax.tick_params(labelsize=9.5)
plt.tight_layout(); plt.savefig("figs/Figure5.png",dpi=300,bbox_inches="tight"); plt.close()
# Figure 6
df=pd.read_csv("serves_all70_flagged.csv"); d=df[~df.bad_capture & ~df.serve_id.isin([6,58,13,32,14,33,19,40])]
fig,ax=plt.subplots(figsize=(9,3.6)); rng=np.random.default_rng(0)
colors={"kick":"#2E5A88","slice":"#E08E0B","flat":"#C0392B"}
ax.axvspan(28.1,29.9,color="#bbbbbb",alpha=0.35,lw=0); ax.axvspan(32.7,33.2,color="#bbbbbb",alpha=0.35,lw=0)
for i,t in enumerate(TYPES):
    s=d[d.serve_type==t].racket_speed
    ax.scatter(s,np.full(len(s),i)+rng.uniform(-0.15,0.15,len(s)),color=colors[t],s=40,edgecolor="white",lw=0.5)
ax.set_yticks(range(3)); ax.set_yticklabels([f"{t.capitalize()} (n = {(d.serve_type==t).sum()})" for t in TYPES],fontsize=11)
ax.set_xlabel("Peak racket-tip speed (m/s)",fontsize=11); ax.invert_yaxis(); ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(); plt.savefig("figs/Figure6.png",dpi=300,bbox_inches="tight"); plt.close()
print("ok")
