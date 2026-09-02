"""
=============================================================================
SERVE-TYPE CLASSIFICATION MODEL — Random Forest + Logistic Regression
=============================================================================
Reproduces the classification analysis reported in the paper:
"What Distinguishes a Kick, Slice and Flat Serve? Comparing Joint Range of
Motion Against Racket Speed in a Single-Player Motion Capture Study"

This script:
  1. Consolidates the three lab spreadsheets (kick / slice / flat) into one
     70-serve table.
  2. Applies the two exclusion thresholds (right-elbow flexion ROM < 55 deg,
     or either knee flexion ROM < 40 deg) to remove serves affected by
     marker occlusion, leaving 51 serves.
  3. Fits a random forest and a logistic regression to predict serve type
     from 15 variables (racket speed, elbow pronation, 3 lumbar, 10 leg
     ROM variables), evaluated with leave-one-recording-out cross-validation.
  4. Reports accuracy, confidence intervals, the no-information rate,
     variable importance, and the racket-speed-isolation comparison.
  5. Runs the threshold sensitivity analysis (Tables 4 and 5 in the paper).

Expected headline results on this data:
  Random forest      : 70.6% accuracy (95% CI approx 58.1-83.1%)
  Logistic regression : 68.6% accuracy (95% CI approx 55.9-81.4%)
  No-information rate : 39.2%
  Racket speed alone   : ~72.5%
  14 ROM vars, no speed: ~37.3% (below the no-information rate)

Requires: pandas, numpy, scikit-learn, openpyxl
=============================================================================
"""

import openpyxl
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (LeaveOneGroupOut, RepeatedStratifiedKFold,
                                      cross_val_predict, cross_val_score)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =============================================================================
# CONFIGURATION — the two exclusion thresholds used in the paper
# =============================================================================

KICK_FILE  = "kick_serve_data__1_.xlsx"
SLICE_FILE = "slice_serve_IK_report__1_.xlsx"
FLAT_FILE  = "Flat_Serve_IK_Report__1___1_.xlsx"

# Right-elbow flexion floor. Derived from published serve kinematics: elbow
# flexion is reported at 85-107 deg at the trophy position and ~30 deg at
# ball impact, implying an expected ROM of 55-77 deg; the floor is set at
# the bottom of that range.
ELBOW_ROM_THRESHOLD = 55

# Knee flexion floor, applied to EACH leg separately. Published values place
# peak knee flexion at ~65 deg at the trophy position and 6-20 deg at
# impact, implying an expected ROM of 44-58 deg; the floor is set below the
# bottom of that range.
KNEE_ROM_THRESHOLD = 40

TYPES = ["kick", "slice", "flat"]


# =============================================================================
# PART 1 — CONSOLIDATION (kick / slice / flat have different sheet layouts)
# =============================================================================

def sheet_to_df(worksheet, header_row=1):
    rows = list(worksheet.iter_rows(values_only=True))
    header = [str(h).strip() if h is not None else f"col{i}"
              for i, h in enumerate(rows[header_row - 1])]
    data = [r for r in rows[header_row:] if r and r[0] is not None]
    return pd.DataFrame(data, columns=header[:len(data[0])] if data else header)


UPPER_COLS = ["rec", "serve", "impact_t", "SH_flex", "SH_abd", "SH_rot",
              "EL_flex", "EL_prosup", "LB_ext", "LB_bend", "LB_rot"]
LOWER_COLS = ["rec", "serve", "impact_t",
              "R_HI_flex", "R_HI_abd", "R_HI_rot", "R_KN_flex", "R_AN_flex", "R_AN_inv",
              "L_HI_flex", "L_HI_abd", "L_HI_rot", "L_KN_flex", "L_AN_flex", "L_AN_inv"]


def load_kick(path):
    wb = openpyxl.load_workbook(path, data_only=True)
    imp = sheet_to_df(wb["Detected Repetitions"]).iloc[:, :5]
    imp.columns = ["rec", "src", "serve", "impact_t", "racket_speed"]
    up = sheet_to_df(wb["ROM - Upper Body & Trunk"]).iloc[:, :11]; up.columns = UPPER_COLS
    lo = sheet_to_df(wb["ROM - Lower Body"]).iloc[:, :15];         lo.columns = LOWER_COLS
    out = (imp.drop(columns=["src", "impact_t"])
              .merge(up.drop(columns=["impact_t"]), on=["rec", "serve"])
              .merge(lo.drop(columns=["impact_t"]), on=["rec", "serve"]))
    out["serve_type"] = "kick"
    return out


def load_slice(path):
    wb = openpyxl.load_workbook(path, data_only=True)
    imp = sheet_to_df(wb["Impacts & Peak Speed"]).iloc[:, :4]
    imp.columns = ["rec", "serve", "impact_t", "racket_speed"]
    up = sheet_to_df(wb["ROM - Right Arm & Trunk"]).iloc[:, :11]; up.columns = UPPER_COLS
    lo = sheet_to_df(wb["ROM - Lower Body"]).iloc[:, :15];        lo.columns = LOWER_COLS
    up = up[up["serve"].apply(lambda x: isinstance(x, (int, float)))]
    out = (imp.drop(columns=["impact_t"])
              .merge(up.drop(columns=["impact_t"]), on=["rec", "serve"])
              .merge(lo.drop(columns=["impact_t"]), on=["rec", "serve"]))
    out["serve_type"] = "slice"
    return out


def load_flat(path):
    wb = openpyxl.load_workbook(path, data_only=True)
    rows = []
    for sheet in ["flat_1", "flat_2", "flat_3", "flat_4", "flat_5"]:
        r = list(wb[sheet].iter_rows(values_only=True))
        upper, lower, speed = r[4:9], r[13:18], r[22:27]
        for u, l, s in zip(upper, lower, speed):
            rows.append(dict(
                rec=sheet, serve=u[0], racket_speed=s[2],
                SH_flex=u[2], SH_abd=u[3], SH_rot=u[4], EL_flex=u[5], EL_prosup=u[6],
                LB_ext=u[7], LB_bend=u[8], LB_rot=u[9],
                R_HI_flex=l[2], R_HI_abd=l[3], R_HI_rot=l[4],
                R_KN_flex=l[5], R_AN_flex=l[6], R_AN_inv=l[7],
                L_HI_flex=l[8], L_HI_abd=l[9], L_HI_rot=l[10],
                L_KN_flex=l[11], L_AN_flex=l[12], L_AN_inv=l[13]))
    out = pd.DataFrame(rows)
    out["serve_type"] = "flat"
    return out


def build_dataset():
    df = pd.concat([load_kick(KICK_FILE), load_slice(SLICE_FILE), load_flat(FLAT_FILE)],
                    ignore_index=True)
    numeric = [c for c in df.columns if c not in ["rec", "serve", "serve_type"]]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")

    # Normalise kick's recording names (Measurement_1,3,4,5,6) to kick_1..kick_5
    kick_map = {"Measurement_1": "kick_1", "Measurement_3": "kick_2",
                "Measurement_4": "kick_3", "Measurement_5": "kick_4",
                "Measurement_6": "kick_5"}
    df["file"] = [kick_map.get(r, r) for r in df["rec"]]

    # Shoulder tracking faults: coordinate wrap shows up as ROM of ~180 deg
    df["shoulder_flag"] = df["SH_flex"] >= 179

    # Marker occlusion: excluded if right-elbow flexion ROM is too low,
    # OR if either knee's flexion ROM is too low (tested independently per leg)
    df["bad_capture"] = ((df["EL_flex"] < ELBOW_ROM_THRESHOLD) |
                          (df["R_KN_flex"] < KNEE_ROM_THRESHOLD) |
                          (df["L_KN_flex"] < KNEE_ROM_THRESHOLD))

    df.insert(0, "serve_id", range(1, len(df) + 1))
    return df


# =============================================================================
# PART 2 — FEATURE SET (15 variables)
# =============================================================================
# Excluded from the feature set:
#   - the 3 shoulder variables (tracking faults)
#   - EL_flex, R_KN_flex, L_KN_flex (all three were used to decide which
#     serves to keep; using them again as predictors would be circular,
#     since the exclusion truncates their distributions in a way tied to
#     serve type)
# =============================================================================

TRUNK = ["LB_ext", "LB_bend", "LB_rot"]
LEG10 = ["R_HI_flex", "R_HI_abd", "R_HI_rot", "R_AN_flex", "R_AN_inv",
         "L_HI_flex", "L_HI_abd", "L_HI_rot", "L_AN_flex", "L_AN_inv"]
FEATURES = ["racket_speed", "EL_prosup"] + TRUNK + LEG10   # 15 variables


# =============================================================================
# PART 3 — MODELLING
# =============================================================================

def run_models(df):
    data = df[~df.bad_capture].copy()
    X = data[FEATURES].values
    y = data.serve_type.values
    groups = data.file.values   # recording ID, for grouped cross-validation

    counts = data.serve_type.value_counts()
    nir = counts.max() / len(data)   # no-information rate

    print(f"{'='*70}\nDATASET\n{'='*70}")
    print(f"n = {len(data)}  ({dict(counts)})")
    print(f"No-information rate (always predict '{counts.idxmax()}'): {nir:.3f}\n")

    rf = RandomForestClassifier(n_estimators=500, random_state=1, min_samples_leaf=2)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))

    # ---- Random (leaky) cross-validation, for contrast only ----
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=1)
    rf_random = cross_val_score(rf, X, y, cv=cv).mean()

    # ---- Leave-one-recording-out cross-validation (the honest number) ----
    logo = LeaveOneGroupOut()
    rf_pred = cross_val_predict(rf, X, y, cv=logo, groups=groups)
    lr_pred = cross_val_predict(lr, X, y, cv=logo, groups=groups)
    rf_acc = accuracy_score(y, rf_pred)
    lr_acc = accuracy_score(y, lr_pred)

    def ci(acc, n):
        se = np.sqrt(acc * (1 - acc) / n)
        return acc - 1.96 * se, acc + 1.96 * se

    print(f"{'='*70}\nCLASSIFICATION RESULTS (leave-one-recording-out CV)\n{'='*70}")
    for name, acc in [("Random forest", rf_acc), ("Logistic regression", lr_acc)]:
        lo, hi = ci(acc, len(y))
        print(f"  {name:<22} {acc:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  {'No-information rate':<22} {nir:.3f}")
    print(f"\n  (Random, non-grouped CV for random forest, shown only as a caution "
          f"about\n   validation leakage: {rf_random:.3f} — do not report this as the result)")

    print(f"\nLogistic regression classification report:")
    print(classification_report(y, lr_pred, digits=3))
    print("Confusion matrix (random forest), rows = actual, columns = predicted:")
    print(f"  order: {TYPES}")
    print(confusion_matrix(y, rf_pred, labels=TYPES))

    # ---- Where does the signal live? ----
    print(f"\n{'='*70}\nISOLATING RACKET-TIP SPEED\n{'='*70}")
    for label, feats in [("All 15 variables", FEATURES),
                          ("Racket-tip speed only", ["racket_speed"]),
                          ("14 joint ROM variables, no racket speed",
                           [f for f in FEATURES if f != "racket_speed"])]:
        Xs = data[feats].values
        a = accuracy_score(y, cross_val_predict(rf, Xs, y, cv=logo, groups=groups))
        b = accuracy_score(y, cross_val_predict(lr, Xs, y, cv=logo, groups=groups))
        print(f"  {label:<42} RF={a:.3f}   LR={b:.3f}")

    # ---- Variable importance ----
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print(f"\n{'='*70}\nVARIABLE IMPORTANCE (random forest)\n{'='*70}")
    print(imp.round(4).to_string())

    return data, rf_acc, lr_acc


# =============================================================================
# PART 4 — SENSITIVITY ANALYSIS (Tables 4 and 5 in the paper)
# =============================================================================

def run_sensitivity(df):
    # Fewer trees here purely for speed; the paper's reported sensitivity
    # figures used n_estimators=500 as in run_models() above. Results are
    # very similar either way, since the point of this section is the
    # overall pattern across thresholds, not a precise headline number.
    logo = LeaveOneGroupOut()
    rf = RandomForestClassifier(n_estimators=200, random_state=1, min_samples_leaf=2)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))

    print(f"\n{'='*70}\nSENSITIVITY — VARYING THE ELBOW THRESHOLD (knee fixed at 40)\n{'='*70}")
    print(f"{'Elbow thresh':>13}{'n':>5}{'RF':>8}{'Speed only':>12}{'ROM only':>10}")
    for t in [40, 50, 55, 60, 70, 85]:
        d = df[(df.EL_flex >= t) & (df.R_KN_flex >= KNEE_ROM_THRESHOLD) &
               (df.L_KN_flex >= KNEE_ROM_THRESHOLD)]
        if d.serve_type.nunique() < 3 or d.file.nunique() < 2:
            continue
        X, y, g = d[FEATURES].values, d.serve_type.values, d.file.values
        a = accuracy_score(y, cross_val_predict(rf, X, y, cv=logo, groups=g))
        sp = accuracy_score(y, cross_val_predict(rf, d[["racket_speed"]].values, y, cv=logo, groups=g))
        ns = accuracy_score(y, cross_val_predict(rf, d[[f for f in FEATURES if f != "racket_speed"]].values,
                                                   y, cv=logo, groups=g))
        print(f"{t:>13}{len(d):>5}{a:>8.3f}{sp:>12.3f}{ns:>10.3f}")

    print(f"\n{'='*70}\nSENSITIVITY — VARYING THE KNEE THRESHOLD (elbow fixed at 55)\n{'='*70}")
    print(f"{'Knee thresh':>13}{'n':>5}{'RF':>8}{'Speed only':>12}{'ROM only':>10}")
    for t in [20, 30, 40, 50, 60]:
        d = df[(df.EL_flex >= ELBOW_ROM_THRESHOLD) & (df.R_KN_flex >= t) & (df.L_KN_flex >= t)]
        if d.serve_type.nunique() < 3 or d.file.nunique() < 2:
            continue
        X, y, g = d[FEATURES].values, d.serve_type.values, d.file.values
        a = accuracy_score(y, cross_val_predict(rf, X, y, cv=logo, groups=g))
        sp = accuracy_score(y, cross_val_predict(rf, d[["racket_speed"]].values, y, cv=logo, groups=g))
        ns = accuracy_score(y, cross_val_predict(rf, d[[f for f in FEATURES if f != "racket_speed"]].values,
                                                   y, cv=logo, groups=g))
        print(f"{t:>13}{len(d):>5}{a:>8.3f}{sp:>12.3f}{ns:>10.3f}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    df = build_dataset()
    df.to_csv("serves_all70_flagged.csv", index=False)
    print(f"Consolidated {len(df)} serves -> serves_all70_flagged.csv")
    print(f"Excluded (marker occlusion): {df.bad_capture.sum()}   "
          f"Retained: {(~df.bad_capture).sum()}\n")

    data, rf_acc, lr_acc = run_models(df)
    run_sensitivity(df)
