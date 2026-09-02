"""
=============================================================================
FIGURE 3 — Serves excluded by type
=============================================================================
Bar chart showing, for each serve type, how many serves were retained vs.
excluded for marker occlusion. Marker occlusion affected flat and slice
serves far more than kick serves, matching the laboratory's own report of
when markers were lost.

Input : serves_all70_flagged.csv  (produced by model_pipeline.py)
Output: Figure3.png
=============================================================================
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ELBOW_ROM_THRESHOLD = 55
KNEE_ROM_THRESHOLD = 40

df = pd.read_csv("serves_all70_flagged.csv")
df["bad_capture"] = ((df.EL_flex < ELBOW_ROM_THRESHOLD) |
                      (df.R_KN_flex < KNEE_ROM_THRESHOLD) |
                      (df.L_KN_flex < KNEE_ROM_THRESHOLD))

TYPES = ["kick", "slice", "flat"]
COLORS = {"kick": "#2E5A88", "slice": "#E07A2F", "flat": "#4C9F70"}

counts = df.groupby(["serve_type", "bad_capture"]).size().unstack(fill_value=0).reindex(TYPES)

fig, ax = plt.subplots(figsize=(7.5, 6))
ax.bar([t.capitalize() for t in TYPES], counts[False], color=[COLORS[t] for t in TYPES],
       label="Retained", width=0.6)
ax.bar([t.capitalize() for t in TYPES], counts[True], bottom=counts[False],
       color="#C0392B", alpha=0.8, label="Excluded", width=0.6)

for i, t in enumerate(TYPES):
    ax.text(i, counts.loc[t, False] / 2, f"{counts.loc[t, False]}",
            ha="center", va="center", fontsize=14, color="white", fontweight="bold")
    if counts.loc[t, True] > 0:
        ax.text(i, counts.loc[t, False] + counts.loc[t, True] / 2, f"{counts.loc[t, True]}",
                ha="center", va="center", fontsize=13, color="white", fontweight="bold")

ax.set_ylabel("Number of serves", fontsize=12)
ax.legend(fontsize=11, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=11)
plt.tight_layout()
plt.savefig("Figure3.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved Figure3.png")
print(counts)
