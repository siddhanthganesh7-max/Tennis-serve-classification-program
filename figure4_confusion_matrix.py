"""
=============================================================================
FIGURE 4 — Confusion matrix (random forest, leave-one-recording-out CV)
=============================================================================
Shows which serve types the random forest confused with which, under
leave-one-recording-out cross-validation on the 51 retained serves.

Input : serves_all70_flagged.csv  (produced by model_pipeline.py)
Output: Figure4.png
=============================================================================
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score

ELBOW_ROM_THRESHOLD = 55
KNEE_ROM_THRESHOLD = 40
TYPES = ["kick", "slice", "flat"]

# 15-variable feature set: excludes shoulder (tracking faults) and
# EL_flex / R_KN_flex / L_KN_flex (used to define the exclusion, so
# circular if also used as predictors)
TRUNK = ["LB_ext", "LB_bend", "LB_rot"]
LEG10 = ["R_HI_flex", "R_HI_abd", "R_HI_rot", "R_AN_flex", "R_AN_inv",
         "L_HI_flex", "L_HI_abd", "L_HI_rot", "L_AN_flex", "L_AN_inv"]
FEATURES = ["racket_speed", "EL_prosup"] + TRUNK + LEG10

df = pd.read_csv("serves_all70_flagged.csv")
df["bad_capture"] = ((df.EL_flex < ELBOW_ROM_THRESHOLD) |
                      (df.R_KN_flex < KNEE_ROM_THRESHOLD) |
                      (df.L_KN_flex < KNEE_ROM_THRESHOLD))
data = df[~df.bad_capture]

X = data[FEATURES].values
y = data.serve_type.values
groups = data.file.values

rf = RandomForestClassifier(n_estimators=500, random_state=1, min_samples_leaf=2)
logo = LeaveOneGroupOut()
pred = cross_val_predict(rf, X, y, cv=logo, groups=groups)
acc = accuracy_score(y, pred)
cm = confusion_matrix(y, pred, labels=TYPES)

fig, ax = plt.subplots(figsize=(6, 5.2))
ax.imshow(cm, cmap="Blues")
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=17, fontweight="bold",
                color="white" if cm[i, j] > cm.max() / 2 else "#222")
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels([t.capitalize() for t in TYPES], fontsize=11)
ax.set_yticklabels([t.capitalize() for t in TYPES], fontsize=11)
ax.set_xlabel("Serve type predicted by the model", fontsize=11.5)
ax.set_ylabel("Serve type actually performed", fontsize=11.5)
plt.tight_layout()
plt.savefig("Figure4.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved Figure4.png")
print(f"Accuracy = {acc:.3f}")
print(cm)
