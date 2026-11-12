# Tennis serve classification (revised analysis)

Code for the paper *What Distinguishes a Kick, Slice and Flat Serve? Joint Range of Motion Versus Racket Speed in a Single-Player Pilot Study of Shadow Serves* (Siddhanth Ganesh).

## Revised analysis (used in the current paper)

| File | What it produces |
|---|---|
| `revised_analysis.py` | All main results: leave-one-recording-out accuracy, balanced accuracy, precision/recall/F1, macro- and weighted-F1, cluster-bootstrap confidence intervals, the racket-speed-only and joint-ROM-only comparisons, the exclusion-threshold sensitivity tables, and out-of-fold permutation importance. Primary dataset: 43 serves (duplicate pairs removed); sensitivity: 51 serves. |
| `diff_check.py` | Bootstrap intervals for the change in accuracy when joint ROM variables are added to racket speed. |
| `speed_check.py` | Racket-speed ranges by serve type and the number of serves in each speed-overlap band. |
| `make_figures.py` | Figures 4–6 (confusion matrices, permutation importance, racket speed by serve type). |
| `figure1_whole_body_compression.py`, `figure2_exclusion_thresholds.py`, `figure3_exclusions_by_type.py` | Figures 1–3 (screening of all 70 detected serves for implausible ROM values). The exclusion step is the same as in the earlier version; `figure2_exclusion_thresholds.py` is unchanged and the other two had their labels updated in the revision. |

All settings are fixed in the code (random forest: 500 trees, `min_samples_leaf=2`, seed 1; logistic regression: L2, C = 1, lbfgs, inside a pipeline with standardisation so scaling is fitted on training folds only). No hyperparameter tuning and no class weighting. Written for Python 3 with scikit-learn 1.9.1, pandas, NumPy and matplotlib.

The scripts read `serves_all70_flagged.csv`, the processed serve-level dataset, which is available from the author on request.

## Earlier version

`model_pipeline.py` and `figure4_confusion_matrix.py` to `figure7_knee_threshold_sensitivity.py` are from the earlier version of the analysis (51 serves, impurity-based importance) and are kept for reference only. They do not produce the numbers or figures in the current paper; Figures 4–6 now come from `make_figures.py`.

## Added in the revision (response to reviewers)

| File | What it produces |
|---|---|
| `raw_marker_check.py` | Checks against the raw Qualisys marker files (.trc): reapplies the laboratory's serve-detection rule, compares racket-tip speed with the laboratory's values (r = 0.999), computes a marker-based elbow flexion ROM for excluded and retained serves, and reports right upper-arm marker loss by recording. Run as `python raw_marker_check.py <folder of .trc files> serves_all70_flagged.csv`. |
| `sensitivity_without_forearm.py` | Sensitivity analysis without forearm pronation/supination. |
| `figureS1_marker_vs_lab_elbow.py` | Supplementary Figure S1. |
| `figure1_whole_body_compression.py`, `figure3_exclusions_by_type.py` | Updated labels: excluded serves are described as having implausible ROM values rather than marker occlusion. |

The raw .trc files are available from the author on request.
