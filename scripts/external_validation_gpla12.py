"""
================================================================================
 EXTERNAL GENERALIZATION CHECK — GPLA-12 ACOUSTIC LEAK DATASET
 Companion validation script for Paper 1
 "Anatomy of Evaluation Bias in Machine Learning-Based Leak Detection for
  Two-Phase Flow Pipelines"

 PURPOSE
 A reviewer of Paper 1 asked directly: is the LOCO evaluation principle
 specific to this 6 m two-phase flow rig, or does it generalize to other
 experimental setups? This script answers that question using a fully
 independent, publicly available dataset (GPLA-12), rather than more
 analysis of the original rig's data.

 CITATION (include in Paper 1's bibliography if this section is added):
   J. Li and L. Yao, "GPLA-12: An Acoustic Signal Dataset of Gas Pipeline
   Leakage," arXiv preprint arXiv:2106.10277, Jun. 2021.

 WHAT THIS DOES AND DOES NOT TEST
 GPLA-12 has no trial/session ID within each of its 12 (microphone x
 pressure x noise) groups (57 samples each), so this CANNOT replicate
 Paper 1's specific "partition after windowing" temporal-adjacency
 mechanism (Issue 1's exact form). It CAN test the more general principle
 Paper 1 argues for: that the correct cross-validation unit is the
 experimental grouping structure, not a random sample index — and it CAN
 replicate the feature-encoding-bias mechanism (Issue 2) by testing whether
 including the group-identifying variables (microphone, noise) as features
 inflates apparent performance versus excluding them.

 GPLA-12 has no multi-leak structure (single artificial leak only, varying
 pressure/mic/noise) — it is NOT relevant to Paper 2 (leak counting).

 DESIGN
 Target (y):        pressure level, 3 classes {0.2, 0.4, 0.5 MPa} — this
                     dataset's severity label, playing the role Paper 1's
                     leak-class target played.
 Group variable:     (microphone, noise) tuple, 4 groups — playing the role
                     Paper 1's (VSG, VSL) operating condition played.
 Condition A:        random stratified 80/20 split, WITH microphone/noise
                     one-hot features included (replicates the original
                     rig-paper's protocol issue).
 Condition B:        Leave-One-Group-Out (4 folds) over the 4 (mic, noise)
                     groups, WITH microphone/noise features included
                     (isolates temporal/grouping-leakage impact only).
 Condition C:        Leave-One-Group-Out, WITHOUT microphone/noise features
                     (primary corrected external-validation result).

 USAGE:
   1. Edit DATA_PATH below to point at gpla12_acoustic_signals.csv
   2. pip install catboost scikit-learn pandas numpy
   3. python external_validation_gpla12.py
================================================================================
"""

import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

# ── EDIT THIS PATH ────────────────────────────────────────────────────────────
_ROOTS = [
    os.environ.get("GPLA12_CSV", ""),            # full path to gpla12_acoustic_signals.csv
    "data/GPLA-12/gpla12_acoustic_signals.csv",
    "./gpla12_acoustic_signals.csv",
]
DATA_PATH = next((Path(p) for p in _ROOTS if p and Path(p).exists()), Path(_ROOTS[1]))
OUTPUT_DIR = DATA_PATH.parent / "External_Validation_Results"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

N_SEEDS = 10
SIGNAL_COLS = [f"signal_{i}" for i in range(1024)]
PRESSURE_MAP = {"0.2MP": 0, "0.4MP": 1, "0.5MP": 2}

print("=" * 80)
print("  EXTERNAL GENERALIZATION CHECK — GPLA-12 ACOUSTIC LEAK DATASET")
print("=" * 80)
print(f"  Data path : {DATA_PATH}  (exists={DATA_PATH.exists()})")
print(f"  Output    : {OUTPUT_DIR}\n")

if not DATA_PATH.exists():
    raise SystemExit(
        "Data file not found — edit DATA_PATH at the top of this script to point "
        "at gpla12_acoustic_signals.csv. This script cannot produce results without it."
    )

# ══════════════════════════════════════════════════════════════════════════
# 1. LOAD AND PARSE LABELS
# ══════════════════════════════════════════════════════════════════════════
print("[1] Loading data")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")

# label format: Data{1,2}_{0.2,0.4,0.5}MP_{none,noisy}
parts = df['label'].str.extract(r'Data(?P<mic>\d)_(?P<pressure>[\d.]+MP)_(?P<noise>none|noisy)')
df['mic'] = parts['mic'].astype(int) - 1                      # 0 or 1
df['pressure_class'] = parts['pressure'].map(PRESSURE_MAP)    # 0/1/2
df['noisy'] = (parts['noise'] == 'noisy').astype(int)          # 0 or 1
df['group'] = df['mic'].astype(str) + "_" + df['noisy'].astype(str)  # 4 groups

print("  Parsed label distribution:")
print(df.groupby(['mic', 'pressure_class', 'noisy']).size().to_string())
groups = sorted(df['group'].unique())
print(f"\n  Groups (mic_noisy): {groups}  (n={len(groups)})")


# ══════════════════════════════════════════════════════════════════════════
# 2. FEATURE EXTRACTION — same 8 generic statistics used throughout Paper 1,
#    applied to the raw acoustic waveform (signal_0..signal_1023), so the
#    external check uses a methodologically identical feature family.
# ══════════════════════════════════════════════════════════════════════════
def extract_features(row_signal):
    x = row_signal.values.astype(float)
    n = len(x); t = np.arange(n, dtype=float); eps = 1e-12
    slope = (np.dot(t - t.mean(), x - x.mean()) /
             (np.dot(t - t.mean(), t - t.mean()) + eps)) if n > 1 else 0.0
    return {
        "mean": float(np.mean(x)), "min": float(np.min(x)), "max": float(np.max(x)),
        "std": float(np.std(x)), "rms": float(np.sqrt(np.mean(x**2))),
        "range": float(np.ptp(x)), "slope": float(slope),
        "meandiff": float(np.mean(np.abs(np.diff(x))) if n > 1 else 0.0),
    }

print("\n[2] Extracting features (8 generic statistics per sample, from raw waveform)")
feat_rows = [extract_features(df.loc[i, SIGNAL_COLS]) for i in df.index]
feat_df = pd.DataFrame(feat_rows)
feat_df['mic'] = df['mic'].values
feat_df['noisy'] = df['noisy'].values
feat_df['group'] = df['group'].values
feat_df['y'] = df['pressure_class'].values
print(f"  Feature table: {feat_df.shape}")

STAT_COLS = ["mean", "min", "max", "std", "rms", "range", "slope", "meandiff"]


def make_X(sub_df, include_group_features):
    cols = STAT_COLS + (["mic", "noisy"] if include_group_features else [])
    return sub_df[cols].values.astype(float)


# ══════════════════════════════════════════════════════════════════════════
# 3. CONDITION A — random stratified split, WITH group features
# ══════════════════════════════════════════════════════════════════════════
print("\n[3] Condition A — random stratified split, WITH mic/noise features")
X_all = make_X(feat_df, include_group_features=True)
y_all = feat_df['y'].values
exp_a_accs = []
for seed in range(N_SEEDS):
    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2,
                                               random_state=seed, stratify=y_all)
    sc = StandardScaler()
    m = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05,
                            random_seed=seed, verbose=0)
    m.fit(sc.fit_transform(X_tr), y_tr)
    exp_a_accs.append(accuracy_score(y_te, m.predict(sc.transform(X_te)).ravel()))
exp_a_mean, exp_a_std = np.mean(exp_a_accs), np.std(exp_a_accs)
print(f"    Accuracy: {exp_a_mean*100:.1f}% ± {exp_a_std*100:.1f}%  ({N_SEEDS} seeds)")


# ══════════════════════════════════════════════════════════════════════════
# 4. CONDITIONS B & C — Leave-One-Group-Out (4 folds), with / without
#    the microphone+noise group-identifying features
# ══════════════════════════════════════════════════════════════════════════
def logo_run(include_group_features, seed):
    all_true, all_pred = [], []
    per_fold = []
    for held_out in groups:
        train_df = feat_df[feat_df['group'] != held_out]
        test_df = feat_df[feat_df['group'] == held_out]
        X_tr = make_X(train_df, include_group_features)
        X_te = make_X(test_df, include_group_features)
        y_tr, y_te = train_df['y'].values, test_df['y'].values
        sc = StandardScaler()
        m = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05,
                                random_seed=seed, verbose=0)
        m.fit(sc.fit_transform(X_tr), y_tr)
        pred = m.predict(sc.transform(X_te)).ravel()
        all_true.extend(y_te.tolist()); all_pred.extend(pred.tolist())
        per_fold.append({'held_out_group': held_out, 'n_test': len(y_te),
                          'fold_acc': accuracy_score(y_te, pred)})
    acc = accuracy_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred, average='macro')
    return acc, f1, per_fold


print("\n[4] Condition B — Leave-One-Group-Out, WITH mic/noise features")
exp_b_runs = [logo_run(True, seed) for seed in range(N_SEEDS)]
exp_b_accs = [r[0] for r in exp_b_runs]
exp_b_f1s = [r[1] for r in exp_b_runs]
print(f"    Accuracy: {np.mean(exp_b_accs)*100:.1f}% ± {np.std(exp_b_accs)*100:.1f}%  "
      f"Macro F1: {np.mean(exp_b_f1s):.3f} ± {np.std(exp_b_f1s):.3f}")

print("\n[5] Condition C — Leave-One-Group-Out, WITHOUT mic/noise features (PRIMARY)")
exp_c_runs = [logo_run(False, seed) for seed in range(N_SEEDS)]
exp_c_accs = [r[0] for r in exp_c_runs]
exp_c_f1s = [r[1] for r in exp_c_runs]
print(f"    Accuracy: {np.mean(exp_c_accs)*100:.1f}% ± {np.std(exp_c_accs)*100:.1f}%  "
      f"Macro F1: {np.mean(exp_c_f1s):.3f} ± {np.std(exp_c_f1s):.3f}")

# Per-fold breakdown from the last seed's run, for reporting fold range
# (the statistically appropriate uncertainty measure, per Paper 1's own
# revised statistical-reporting practice)
_, _, per_fold_c = exp_c_runs[0]
fold_accs_by_seed = {g: [] for g in groups}
for seed_run in exp_c_runs:
    for f in seed_run[2]:
        fold_accs_by_seed[f['held_out_group']].append(f['fold_acc'])
fold_means = {g: np.mean(v) for g, v in fold_accs_by_seed.items()}
fold_range = (max(fold_means.values()) - min(fold_means.values())) / 2
print(f"\n    Per-group breakdown (Condition C, mean over {N_SEEDS} seeds):")
for g, v in fold_means.items():
    print(f"      group={g}: acc={v*100:.1f}%  (n_test={dict((f['held_out_group'], f['n_test']) for f in per_fold_c)[g]})")
print(f"    Fold range (max-min)/2 across the 4 groups: ±{fold_range*100:.1f}pp")


# ══════════════════════════════════════════════════════════════════════════
# 5. SUMMARY TABLE — same A/B/C decomposition format as Paper 1's Table I
# ══════════════════════════════════════════════════════════════════════════
leakage_impact = exp_a_mean - np.mean(exp_b_accs)
encoding_impact = np.mean(exp_b_accs) - np.mean(exp_c_accs)

summary = pd.DataFrame([
    {"Condition": "A: random split, WITH mic/noise features",
     "Accuracy": f"{exp_a_mean*100:.1f}% ± {exp_a_std*100:.1f}%"},
    {"Condition": "B: LOGO (4 groups), WITH mic/noise features",
     "Accuracy": f"{np.mean(exp_b_accs)*100:.1f}% ± {np.std(exp_b_accs)*100:.1f}%"},
    {"Condition": "C: LOGO (4 groups), WITHOUT mic/noise features (PRIMARY)",
     "Accuracy": f"{np.mean(exp_c_accs)*100:.1f}% ± {np.std(exp_c_accs)*100:.1f}%  "
                  f"(fold range ±{fold_range*100:.1f}pp)"},
    {"Condition": "Grouping-leakage impact (A - B)",
     "Accuracy": f"-{leakage_impact*100:.1f}%"},
    {"Condition": "Feature-encoding-bias impact (B - C)",
     "Accuracy": f"-{encoding_impact*100:.1f}%"},
])
summary.to_csv(OUTPUT_DIR / "External_Validation_GPLA12_Summary.csv", index=False)
print("\n" + "=" * 80)
print("SUMMARY TABLE (paste into Paper 1's External Generalization Check section):")
print(summary.to_string(index=False))
print("=" * 80)

print(f"""
INTERPRETATION GUIDE (fill in once real numbers are in hand — do not assume
the direction of the effect before running this):
  - If Condition A >> Condition C: the same qualitative pattern documented
    on the two-phase flow rig (random-split evaluation overstates real
    performance) replicates on a fully independent acoustic dataset,
    supporting Paper 1's claim that the LOCO principle is general, not
    rig-specific.
  - If Condition A ≈ Condition C: either this task/feature-set combination
    is less prone to grouping leakage than the rig dataset, or 57
    samples/group with only 8 generic statistical features do not carry
    enough group-identifying signal to leak — report honestly either way,
    do not force a narrative.
  - The 4-group LOGO fold sizes here (~171 test samples/fold) are
    substantially better powered than the rig dataset's 3-condition LOCO
    (which has as few as ~180 samples per fold total, not per class) —
    note this contrast explicitly if adding this section, since it means
    this external check is not just "another dataset" but a
    higher-powered confirmation opportunity.

All outputs written to: {OUTPUT_DIR}
""")
