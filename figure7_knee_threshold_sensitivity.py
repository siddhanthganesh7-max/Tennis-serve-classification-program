"""
=============================================================================
FIGURE 7 — Variable importance across knee exclusion thresholds
=============================================================================
Recomputes variable importance at five different knee ROM thresholds
(20-60 degrees, applied to both legs), with the right-elbow threshold held
fixed at 55 degrees. Shows that racket-tip speed remains the dominant
variable no matter where the knee threshold is placed.

Above 60 degrees too many recordings are removed entirely for
leave-one-recording-out cross-validation to be possible elsewhere in the
analysis (at least two recordings of each serve type must remain), so this
sensitivity check stops at 60.

Input : serves_all70_flagged.csv  (produced by model_pipeline.py)
Output: Figure7.png
=============================================================================
"""

import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

ELBOW_ROM_THRESHOLD = 55
KNEE_THRESHOLDS_TO_TEST = [20, 30, 40, 50, 60]

TRUNK = ["LB_ext", "LB_bend", "LB_rot"]
LEG10 = ["R_HI_flex", "R_HI_abd", "R_HI_rot", "R_AN_flex", "R_AN_inv",
         "L_HI_flex", "L_HI_abd", "L_HI_rot", "L_AN_flex", "L_AN_inv"]
FEATURES = ["racket_speed", "EL_prosup"] + TRUNK + LEG10

LABEL = {
    "racket_speed": "Racket-tip speed", "EL_prosup": "Right elbow pronation/supination",
    "LB_ext": "Lumbar extension", "LB_bend": "Lumbar lateral bending", "LB_rot": "Lumbar rotation",
    "R_HI_flex": "Right hip flexion", "R_HI_abd": "Right hip abduction/adduction", "R_HI_rot": "Right hip rotation",
    "R_AN_flex": "Right ankle dorsi/plantarflexion", "R_AN_inv": "Right ankle inversion/eversion",
    "L_HI_flex": "Left hip flexion", "L_HI_abd": "Left hip abduction/adduction", "L_HI_rot": "Left hip rotation",
    "L_AN_flex": "Left ankle dorsi/plantarflexion", "L_AN_inv": "Left ankle inversion/eversion",
}

df = pd.read_csv("serves_all70_flagged.csv")

importances = {}
sample_sizes = {}
for t in KNEE_THRESHOLDS_TO_TEST:
    d = df[(df.EL_flex >= ELBOW_ROM_THRESHOLD) & (df.R_KN_flex >= t) & (df.L_KN_flex >= t)]
    rf = RandomForestClassifier(n_estimators=500, random_state=1, min_samples_leaf=2)
    rf.fit(d[FEATURES].values, d.serve_type.values)
    importances[t] = pd.Series(rf.feature_importances_, index=FEATURES)
    sample_sizes[t] = len(d)

fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
for ax, t in zip(axes.ravel(), KNEE_THRESHOLDS_TO_TEST):
    imp = importances[t].sort_values()
    colors = ["#C0392B" if k == "racket_speed" else "#2E5A88" for k in imp.index]
    ax.barh([LABEL[k] for k in imp.index], imp.values, color=colors)
    ax.set_title(f"Knee threshold {t}\u00b0  (n = {sample_sizes[t]})", fontsize=10, fontweight="bold")
    ax.set_xlabel("Mean decrease in impurity", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, 0.45)
axes.ravel()[-1].axis("off")   # only 5 thresholds tested, last panel unused
plt.tight_layout()
plt.savefig("Figure7.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved Figure7.png")
for t in KNEE_THRESHOLDS_TO_TEST:
    print(f"  threshold={t}  n={sample_sizes[t]}  racket_speed importance={importances[t]['racket_speed']:.3f}")
