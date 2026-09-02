"""
=============================================================================
FIGURE 6 — Variable importance across right-elbow exclusion thresholds
=============================================================================
Recomputes variable importance at six different right-elbow ROM thresholds
(40-85 degrees), with the knee threshold held fixed at 40 degrees for both
legs. Shows that racket-tip speed remains the dominant variable no matter
where the elbow threshold is placed, so the paper's central finding does
not depend on this choice.

Input : serves_all70_flagged.csv  (produced by model_pipeline.py)
Output: Figure6.png
=============================================================================
"""

import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

KNEE_ROM_THRESHOLD = 40
ELBOW_THRESHOLDS_TO_TEST = [40, 50, 55, 60, 70, 85]

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
for t in ELBOW_THRESHOLDS_TO_TEST:
    d = df[(df.EL_flex >= t) & (df.R_KN_flex >= KNEE_ROM_THRESHOLD) & (df.L_KN_flex >= KNEE_ROM_THRESHOLD)]
    rf = RandomForestClassifier(n_estimators=500, random_state=1, min_samples_leaf=2)
    rf.fit(d[FEATURES].values, d.serve_type.values)
    importances[t] = pd.Series(rf.feature_importances_, index=FEATURES)
    sample_sizes[t] = len(d)

fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
for ax, t in zip(axes.ravel(), ELBOW_THRESHOLDS_TO_TEST):
    imp = importances[t].sort_values()
    colors = ["#C0392B" if k == "racket_speed" else "#2E5A88" for k in imp.index]
    ax.barh([LABEL[k] for k in imp.index], imp.values, color=colors)
    ax.set_title(f"Right-elbow threshold {t}\u00b0  (n = {sample_sizes[t]})", fontsize=10, fontweight="bold")
    ax.set_xlabel("Mean decrease in impurity", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, 0.45)
plt.tight_layout()
plt.savefig("Figure6.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved Figure6.png")
for t in ELBOW_THRESHOLDS_TO_TEST:
    print(f"  threshold={t}  n={sample_sizes[t]}  racket_speed importance={importances[t]['racket_speed']:.3f}")
