"""
=============================================================================
FIGURE 2 — The two exclusion thresholds
=============================================================================
Plots each serve's right-elbow flexion ROM (x-axis) against the lower of its
two knee flexion ROMs (y-axis), with the two exclusion thresholds marked.
A serve was excluded if it fell into either shaded band.

The y-axis uses the LOWER of the two knees because the exclusion rule tests
each knee independently: a serve is excluded if EITHER knee falls below the
threshold, which is mathematically the same as testing whether the smaller
(lower) of the two knee values falls below it.

Input : serves_all70_flagged.csv  (produced by model_pipeline.py)
Output: Figure2.png
=============================================================================
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ELBOW_ROM_THRESHOLD = 55
KNEE_ROM_THRESHOLD = 40

df = pd.read_csv("serves_all70_flagged.csv")
TYPES = ["kick", "slice", "flat"]
COLORS = {"kick": "#2E5A88", "slice": "#E07A2F", "flat": "#4C9F70"}

df["lower_knee"] = df[["R_KN_flex", "L_KN_flex"]].min(axis=1)

fig, ax = plt.subplots(figsize=(9, 6.4))
for t in TYPES:
    s = df[df.serve_type == t]
    ax.scatter(s.EL_flex, s.lower_knee, s=90, color=COLORS[t], label=t.capitalize(),
               edgecolor="w", linewidth=1.2, alpha=0.88)

ax.axvline(ELBOW_ROM_THRESHOLD, color="#C0392B", ls="--", lw=2)
ax.axhline(KNEE_ROM_THRESHOLD, color="#C0392B", ls="--", lw=2)
ax.axvspan(0, ELBOW_ROM_THRESHOLD, color="#C0392B", alpha=0.07)
ax.axhspan(0, KNEE_ROM_THRESHOLD, color="#C0392B", alpha=0.07)
ax.text(ELBOW_ROM_THRESHOLD + 2, 7, "Right-elbow threshold, 55\u00b0", fontsize=10.5,
        color="#C0392B", fontweight="bold")
ax.text(88, KNEE_ROM_THRESHOLD + 2, "Knee threshold, 40\u00b0\n(applied to each knee separately)",
        fontsize=10.5, color="#C0392B", fontweight="bold")

ax.set_xlabel("Right (racket-arm) elbow flexion range of motion (degrees)", fontsize=11.5)
ax.set_ylabel("Lower of the two knee flexion ranges of motion (degrees)", fontsize=11.5)
ax.legend(fontsize=11, title="Serve type", title_fontsize=11, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=11)
plt.tight_layout()
plt.savefig("Figure2.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved Figure2.png")
