# Tennis serve classification (revised analysis)

Code for the paper *What Distinguishes a Kick, Slice and Flat Serve? Joint Range of Motion Versus Racket Speed in a Single-Player Pilot Study of Shadow Serves* (Siddhanth Ganesh).

## Revised analysis (used in the current paper)

| File | What it produces |
|---|---|
| `revised_analysis.py` | All main results: leave-one-recording-out accuracy, balanced accuracy, precision/recall/F1, macro- and weighted-F1, cluster-bootstrap confidence intervals, the racket-speed-only and joint-ROM-only comparisons, the exclusion-threshold sensitivity tables, and out-of-fold permutation importance. Primary dataset: 43 serves (duplicate pairs removed); sensitivity: 51 serves. |
| `diff_check.py` | Bootstrap intervals for the change in accuracy when joint ROM variables are added to racket speed. |
| `speed_check.py` | Racket-speed ranges by serve type and the number of serves in each speed-overlap band. |
| `make_figures.py` | Figures 4–6 (confusion matrices, permutation importance, racket speed by serve type). |
| `figure1_whole_body_compression.py`, `figure2_exclusion_thresholds.py`, `figure3_exclusions_by_type.py` | Figures 1–3 (the occlusion screening of all 70 detected serves). These scripts are from the earlier version but are unchanged, because the exclusion step is the same in the revised paper. |

All settings are fixed in the code (random forest: 500 trees, `min_samples_leaf=2`, seed 1; logistic regression: L2, C = 1, lbfgs, inside a pipeline with standardisation so scaling is fitted on training folds only). No hyperparameter tuning and no class weighting. Written for Python 3 with scikit-learn 1.9.1, pandas, NumPy and matplotlib.

The scripts read `serves_all70_flagged.csv`, the processed serve-level dataset, which is available from the author on request.

## Earlier version

`model_pipeline.py` and `figure4_confusion_matrix.py` to `figure7_knee_threshold_sensitivity.py` are from the earlier version of the analysis (51 serves, impurity-based importance) and are kept for reference only. They do not produce the numbers or figures in the current paper; Figures 4–6 now come from `make_figures.py`.
