"""
=============================================================================
FIGURE 1 — Whole-body compression among excluded vs. retained serves
=============================================================================
For each of the 70 serves, counts how many of 14 joint range-of-motion (ROM)
variables fall below half that variable's "normal" value (the median among
the 51 retained serves). Plots a grouped histogram comparing this count for
retained serves vs. serves excluded for marker occlusion.

This shows that excluded serves are compressed across MANY joints at once,
not only at the joint(s) used to define the exclusion thresholds.

Input : serves_all70_flagged.csv  (produced by model_pipeline.py)
Output: Figure1.png
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- exclusion thresholds, must match model_pipeline.py ----
ELBOW_ROM_THRESHOLD = 55
KNEE_ROM_THRESHOLD = 40

df = pd.read_csv("serves_all70_flagged.csv")
df["bad_capture"] = ((df.EL_flex < ELBOW_ROM_THRESHOLD) |
                      (df.R_KN_flex < KNEE_ROM_THRESHOLD) |
                      (df.L_KN_flex < KNEE_ROM_THRESHOLD))

# The 14 diagnostic joint variables: excludes the 3 shoulder variables
# (separate tracking fault) and the 3 variables used to DEFINE the
# exclusion (EL_flex, R_KN_flex, L_KN_flex), since counting those would be
# circular with the exclusion rule itself.
DIAGNOSTIC_VARS = ["EL_prosup", "LB_ext", "LB_bend", "LB_rot",
                    "R_HI_flex", "R_HI_abd", "R_HI_rot", "R_AN_flex", "R_AN_inv",
                    "L_HI_flex", "L_HI_abd", "L_HI_rot", "L_AN_flex", "L_AN_inv"]

retained = df[~df.bad_capture]
normal_value = retained[DIAGNOSTIC_VARS].median()

df["n_low"] = (df[DIAGNOSTIC_VARS] < 0.5 * normal_value).sum(axis=1)

counts_retained = df[~df.bad_capture].n_low.value_counts().reindex(range(15), fill_value=0)
counts_excluded = df[df.bad_capture].n_low.value_counts().reindex(range(15), fill_value=0)

fig, ax = plt.subplots(figsize=(9, 6))
x = np.arange(15)
width = 0.4
ax.bar(x - width / 2, counts_retained.values, width, color="#2E5A88", label="Retained")
ax.bar(x + width / 2, counts_excluded.values, width, color="#C0392B", label="Excluded (marker occlusion)")
ax.set_xlabel("Number of the 14 joint ROM variables below half their normal value", fontsize=12)
ax.set_ylabel("Number of serves", fontsize=12)
ax.set_xticks(x)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=11)
ax.legend(fontsize=12, frameon=False)
plt.tight_layout()
plt.savefig("Figure1.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved Figure1.png")
print(f"Retained serves: {(~df.bad_capture).sum()}, median count = {df[~df.bad_capture].n_low.median():.0f}")
print(f"Excluded serves: {df.bad_capture.sum()}, median count = {df[df.bad_capture].n_low.median():.0f}")
