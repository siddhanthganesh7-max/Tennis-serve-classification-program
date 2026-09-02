# Serve-Type Classification from Motion Capture

Code accompanying the paper "What Distinguishes a Kick, Slice and Flat Serve? Comparing Joint Range of Motion Against Racket Speed in a Single-Player Motion Capture Study."

This project asks whether a machine learning model can tell a kick, slice, and flat tennis serve apart using motion-capture measurements of the server's body, and, more importantly, whether the answer comes from how far the joints move (range of motion) or from how fast the racket is swinging.

## Data

The raw data are three spreadsheets exported from the motion-capture lab's OpenSim processing pipeline, one per serve type:

- kick_serve_data__1_.xlsx
- slice_serve_IK_report__1_.xlsx
- Flat_Serve_IK_Report__1___1_.xlsx

These must be placed in the same folder as the scripts before running anything.

## Repository structure

All files sit in one folder:

model_pipeline.py (builds the dataset and runs the classifier), figure1_whole_body_compression.py, figure2_exclusion_thresholds.py, figure3_exclusions_by_type.py, figure4_confusion_matrix.py, figure5_variable_importance.py, figure6_elbow_threshold_sensitivity.py, figure7_knee_threshold_sensitivity.py, README.md

## How to run

Step 1, run this first: model_pipeline.py reads the three .xlsx files, consolidates them into one table of 70 serves, applies the two exclusion thresholds described below, fits the models, and prints all the headline results (accuracy, confusion matrix, variable importance, sensitivity tables). It also writes out serves_all70_flagged.csv in the same folder, which every figure script needs.

python model_pipeline.py

Step 2, figures: Each figureN_*.py script reads serves_all70_flagged.csv from the same folder and produces one figure as a PNG. Once step 1 has run, each figure script can be run independently, for example:

python figure5_variable_importance.py

Figures 6 and 7 each retrain several random forests to test different threshold values, so they take noticeably longer to run than the others (roughly a minute); this is expected, not a bug.

## Requirements

pandas, numpy, scikit-learn, matplotlib, openpyxl

Install with:

pip install pandas numpy scikit-learn matplotlib openpyxl

## Method summary

- 90 serves (30 kick, 30 slice, 30 flat) were recorded; automatic event detection identified 70.
- Serves affected by marker occlusion were excluded using two thresholds set from published serve kinematics, not from this dataset: right-elbow flexion ROM below 55 degrees, or either knee's flexion ROM below 40 degrees. This leaves 51 serves (20 kick, 19 slice, 12 flat).
- The model uses 15 variables: racket-tip speed, elbow pronation/supination, 3 lumbar variables, and 10 leg variables. Elbow flexion and both knee flexion variables are deliberately excluded as predictors, since they were used to decide which serves to keep, and using them again to predict serve type would be circular.
- Two classifiers are compared: a random forest and logistic regression, evaluated with leave-one-recording-out cross-validation (never random splits, since serves from the same recording share too much in common, which inflates accuracy if mixed across train/test).

## Headline results

Random forest: 70.6%
Logistic regression: 68.6%
Always guess the largest class (no-information rate): 39.2%
Racket-tip speed alone: 72.5%
14 joint ROM variables, without racket speed: 33 to 37%

Racket-tip speed alone matches or exceeds the full 15-variable model. Joint range of motion adds essentially nothing once racket speed is known, which is the central finding of the paper.

## AI use disclosure

Claude (Anthropic) was used extensively in this project for code generation and data analysis, and gave assistance in drafting and editing the paper manuscript.
