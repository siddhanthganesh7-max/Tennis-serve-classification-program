"""
=============================================================================
FIGURE 5 — Random forest variable importance
=============================================================================
Shows how much each of the 15 model variables contributed to the random
forest's decisions (mean decrease in impurity), fit on all 51 retained
serves. Racket-tip speed dominates by a wide margin.

Input : serves_all70_flagged.csv  (produced by model_pipeline.py)
Output: Figure5.png
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
KNEE_ROM_THRESHOLD = 40

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
df["bad_capture"] = ((df.EL_flex < ELBOW_ROM_THRESHOLD) |
                      (df.R_KN_flex < KNEE_ROM_THRESHOLD) |
                      (df.L_KN_flex < KNEE_ROM_THRESHOLD))
data = df[~df.bad_capture]

X = data[FEATURES].values
y = data.serve_type.values

rf = RandomForestClassifier(n_estimators=500, random_state=1, min_samples_leaf=2)
rf.fit(X, y)
importance = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()

fig, ax = plt.subplots(figsize=(9, 6.6))
colors = ["#C0392B" if k == "racket_speed" else "#2E5A88" for k in importance.index]
ax.barh([LABEL[k] for k in importance.index], importance.values, color=colors)
ax.set_xlabel("Mean decrease in impurity (random forest importance)", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=10)
plt.tight_layout()
plt.savefig("Figure5.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved Figure5.png")
print(importance.sort_values(ascending=False).round(4))
