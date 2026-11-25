# Tennis serve classification (kick, slice, flat) from joint range of motion and racket speed

Code for the paper *What Distinguishes a Kick, Slice and Flat Serve? Joint Range of Motion Versus Racket Speed in a Single-Player Pilot Study of Shadow Serves* (revised version).

## What changed in this revision

The laboratory's joint-angle files (OpenSim `.mot`) were checked against the raw marker files (`.trc`). Eight of the 15 files used for the original reports were copies of other recordings: the five flat files and slice files 2–4 contained kick-serve data. As a result, the joint values for 40 of the 70 serves came from the wrong recordings. That error caused all 19 "implausible" serves and all four duplicated pairs in the first version. The laboratory reran inverse kinematics on the eight recordings. Every joint value was then recomputed from the corrected files, and the analysis now uses all 70 serves without exclusions.

The exclusion-threshold scripts and figures from the earlier version (`figure1_whole_body_compression.py`, `figure3_exclusions_by_type.py`, `figureS1_marker_vs_lab_elbow.py`, `raw_marker_check.py`, `sensitivity_without_forearm.py`) no longer apply and have been removed.

## Files

| Script | What it does |
|---|---|
| `mocap_io.py` | Shared helpers: reads `.trc` and `.mot` files, detects serves (racket-tip speed peaks > 15 m/s, ≥ 2.5 s apart), computes ROM (max − min from 1.3 s before to 0.9 s after impact) |
| `check_mot_source.py` | Checks each `.mot` file against its `.trc` recording (length, identical copies) and finds which recording's joint angles reproduce each value in the laboratory's report |
| `recompute_rom.py` | Recomputes all ROM values from the corrected `.mot` files (`--medfilt` for the five-frame median-filter version) |
| `compare_original_corrected.py` | Compares the originally reported ROM values with the corrected ones (Figure 1, Supplementary Table S2) |
| `classification_analysis.py` | Main analysis: leave-one-recording-out cross-validation, random forest and logistic regression, cluster-bootstrap confidence intervals, variable subsets, out-of-fold permutation importance |
| `binary_and_incremental.py` | Kick against the other two types, slice against flat, the gain from adding ROM to racket speed, and the speed-overlap counts |
| `sensitivity.py` | Supplementary Table S1: without forearm pronation/supination, median-filtered angles, and trunk and leg ROM only |
| `make_figures.py` | Figures 1–5 and Supplementary Figure S1 |

## Running

```
python check_mot_source.py trc/ mot_original/ lab_report.csv
python recompute_rom.py trc/ mot_corrected/ lab_report.csv
python recompute_rom.py trc/ mot_corrected/ lab_report.csv --medfilt
python compare_original_corrected.py serves70_corrected.csv lab_report.csv mot_source_check.csv
python classification_analysis.py serves70_corrected.csv
python binary_and_incremental.py
python sensitivity.py serves70_corrected.csv serves70_medfilt.csv
python make_figures.py
```

Python 3 with numpy, pandas, scipy, scikit-learn 1.9.1 and matplotlib. All random seeds are fixed. Recordings and files are named `kick_1`–`kick_5`, `slice_1`–`slice_5` and `flat_1`–`flat_5`.

## Data

The raw marker files, joint-angle files and processed dataset are available from the author on reasonable request. They contain no identifying information.
