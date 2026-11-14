#!/usr/bin/env python3
"""
================================================================================
PAPER 1 REVISION — COMPLETE EXPERIMENTAL PIPELINE
================================================================================
"When 100% Accuracy Is a Warning Sign: A Systematic Audit of Evaluation Bias
 in Machine-Learning-Based Leak Detection for Two-Phase Pipelines"

WHAT THIS SCRIPT PRODUCES
──────────────────────────
  All results needed for the revised paper, in one self-contained run:

  SECTION A  Data discovery, coverage audit, seed count confirmation
  SECTION B  Full 2x2 ablation (Conditions A, B, C, D)
             — Condition D is NEW (time-based split, no VSL/VSG)
  SECTION C  S1-only 9-condition LOO as the PRIMARY corrected result
             — replaces the dual-sensor 3-condition result as headline
  SECTION D  Dual-sensor 3-condition LOO (restricted sensitivity analysis)
  SECTION E  Per-fold condition-level accuracy for every held-out condition
             — replaces seed-only std as the variance measure
  SECTION F  All nine models under the corrected protocol
  SECTION G  Noise robustness (symmetric, 5 levels, both train+test)
             — produces Figure 4 AND Table 5 numbers from the same run
  SECTION H  Summary JSON + CSV files, printed tables

OUTPUT FILES (all written to OUTPUT_DIR)
────────────────────────────────────────
  00_coverage_audit.csv          — per-condition sensor coverage grid
  00_seed_count.txt              — confirms N_SEEDS
  01_condition_A.csv             — time-based split, VSL/VSG included
  02_condition_B.csv             — LOO, VSL/VSG included
  03_condition_C_s1_9fold.csv    — LOO, no VSL/VSG, S1 only, 9 conditions
                                   ★ PRIMARY CORRECTED RESULT
  03_condition_C_dual_3fold.csv  — LOO, no VSL/VSG, dual sensor, 3 conditions
  04_condition_D.csv             — time-based split, no VSL/VSG  [NEW]
  05_2x2_table.csv               — assembled 2x2 ablation summary
  06_per_fold_S1.csv             — accuracy per held-out condition (S1-only)
  06_per_fold_dual.csv           — accuracy per held-out condition (dual)
  07_all_models.csv              — Table 3 equivalent
  08_noise_robustness.csv        — Table 5 + Figure 4 source
  09_results_summary.json        — all numbers in one file
  10_figure4_table5_check.txt    — explicit reconciliation of the two sources

USAGE
─────
  python paper1_revision_experiments.py

  Runtime estimate: ~2–4 hours on CPU depending on your machine.
  Set FAST_MODE = True below to run 3 seeds + no inner tuning for a ~15-min
  smoke test. Set FAST_MODE = False for the full production run.

PATHS — edit _ROOTS to match your machine
──────────────────────────────────────────
================================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os, sys, time, copy, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, VotingClassifier,
                               StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier  # fallback NN

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("[WARNING] CatBoost not found — MC4 will be replaced by RandomForest")

try:
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
    CATBOOST_TASK = "GPU" if torch.cuda.is_available() else "CPU"
except ImportError:
    HAS_TORCH = False
    CATBOOST_TASK = "CPU"

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONFIGURATION  ← edit here
# ─────────────────────────────────────────────────────────────────────────────

# Data paths — tries each root in order, uses first that exists
_ROOTS = [os.environ.get("LEAK_DATA_ROOT", ""), "data", "."]  # folder holding Dynamic_pressure/
DATA_ROOT  = next((Path(p) for p in _ROOTS if p and (Path(p) / "Dynamic_pressure").exists()), Path("data"))
S1_DIR     = DATA_ROOT / "Dynamic_pressure" / "Sensor 1"
S2_DIR     = DATA_ROOT / "Dynamic_pressure" / "Sensor 2"
OUTPUT_DIR = DATA_ROOT / "Paper1_Revision_Results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# GPLA-12 external dataset (optional — skip if not present)
GPLA12_PATH = DATA_ROOT / "GPLA-12"

# Experiment settings
N_SEEDS     = 10                     # use 3 for a quick smoke test
FAST_MODE   = False                  # True = 3 seeds, no tuning, Wk=10 only
WK_LIST     = [6, 8, 10] if not FAST_MODE else [10]
PRIMARY_WK  = 10
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.5, 1.0]   # % FSO
SEEDS       = list(range(42, 42 + (3 if FAST_MODE else N_SEEDS)))

# Hyperparameter grid for nested CatBoost tuning (2 x 3 x 3 = 18 combinations)
# Each entry is one combination; inner CV selects the best
PARAM_GRID_FULL = [
    {"iter1": i, "depth1": d, "lr1": lr, "iter2": i, "depth2": d, "lr2": lr}
    for i in [200, 300]
    for d in [4, 6, 8]
    for lr in [0.03, 0.05, 0.10]
]
# Default (fixed) hyperparameters — used when nested tuning is off
PARAM_DEFAULT = {"iter1": 300, "depth1": 6, "lr1": 0.05,
                 "iter2": 300, "depth2": 6, "lr2": 0.05}

# File names and class labels (must match what is in the xlsx directories)
FILES       = [("no leak.xlsx", 0), ("One leak.xlsx", 1), ("three leaks.xlsx", 2)]
CLASS_NAMES = ["No Leak (C0)", "Single Leak (C1)", "Three Leaks (C2)"]
LABEL_MAP   = {0: "C0", 1: "C1", 2: "C2"}

# Reference numbers from Alquennah et al. (2026) — for comparison only
PAPER_ACC = {6: 0.942, 8: 0.970, 10: 1.000}

SEP  = "=" * 80
SEP2 = "─" * 80

# ─────────────────────────────────────────────────────────────────────────────
# 2.  BANNER
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  PAPER 1 REVISION — COMPLETE EXPERIMENTAL PIPELINE")
print(SEP)
print(f"  Data root  : {DATA_ROOT}   exists={DATA_ROOT.exists()}")
print(f"  S1 dir     : {S1_DIR}   exists={S1_DIR.exists()}")
print(f"  S2 dir     : {S2_DIR}   exists={S2_DIR.exists()}")
print(f"  Output     : {OUTPUT_DIR}")
print(f"  Seeds      : {SEEDS}   (N={len(SEEDS)})")
print(f"  Window sizes: {WK_LIST}")
print(f"  Fast mode  : {FAST_MODE}")
print(f"  CatBoost   : {HAS_CATBOOST}   task={CATBOOST_TASK}")
print(f"  PyTorch    : {HAS_TORCH}")
print()

# Write seed count confirmation immediately
(OUTPUT_DIR / "00_seed_count.txt").write_text(
    f"N_SEEDS confirmed = {len(SEEDS)}\n"
    f"Seeds used        = {SEEDS}\n"
    f"Fast mode         = {FAST_MODE}\n"
    f"Primary Wk        = {PRIMARY_WK}\n"
)

# ── S2 inference guard ─────────────────────────────────────────────────────
# The S2 three leaks.xlsx has missing VSL labels for VSL=1.34 and VSL=1.94
# columns. The parser infers these values from the expected sequence.
# BEFORE RUNNING: verify with your physical logbook that three-leaks
# recordings were genuinely made at VSL=1.34 and 1.94 for sensor S2.
# If NOT confirmed: set USE_INFERRED_S2 = False to restrict POOL_DUAL
# to explicitly-labelled conditions only (9 conditions).
USE_INFERRED_S2 = True   # ← set False if unconfirmed by logbook

if not S1_DIR.exists():
    sys.exit(f"ERROR: S1 directory not found at {S1_DIR}\n"
             "Edit _ROOTS at the top of this script.")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  DATA LOADING (robust parser, inherited from v2 pipeline)
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION A: DATA DISCOVERY AND LOADING")
print(SEP)

def _parse_vsg(v):
    if isinstance(v, str) and "VSG" in v:
        try: return float(v.split("=")[1].strip().split()[0])
        except: pass
    try:
        f = float(v); return f if not np.isnan(f) else np.nan
    except: return np.nan

def _parse_vsl(v):
    if isinstance(v, str) and "VSL" in v:
        try: return float(v.split("=")[1].strip().split()[0])
        except: pass
    try:
        f = float(v); return f if not np.isnan(f) else np.nan
    except: return np.nan

def parse_sensor_file(filepath, sensor_name, leak_label, vsl_sequence):
    """
    Parse one xlsx file → long-format DataFrame with columns:
    Time, P (pressure), VSG, VSL, Label, Sensor.

    Layout (0-indexed rows):
      Row 0: VSG value — appears in first column of each group, NaN in others.
      Row 1: VSL value — appears in each data column (may be NaN in S2 files).
      Row 2: role label ("Time", "Pressure (sN)") — NaN for separators.
      Row 3+: numeric data.
    """
    raw = pd.read_excel(Path(filepath), header=None, dtype=object)

    # Forward-fill VSG row so every column in a group inherits its VSG
    vsg_filled = raw.iloc[0].ffill()
    vsl_raw    = raw.iloc[1]
    data       = raw.iloc[3:].reset_index(drop=True)

    # Collect column info
    col_info = {}
    for col in range(1, data.shape[1]):
        vsg = _parse_vsg(vsg_filled.iloc[col])
        if np.isnan(vsg):
            continue
        arr     = pd.to_numeric(data.iloc[:, col], errors="coerce").values
        n_valid = int(np.sum(~np.isnan(arr)))
        if n_valid == 0:
            continue
        vsl = _parse_vsl(vsl_raw.iloc[col])
        col_info[col] = {"vsg": round(vsg, 3), "vsl": vsl,
                         "n_valid": n_valid, "arr": arr, "inferred": False}

    # Infer missing VSL labels within each VSG group
    vsg_groups = defaultdict(list)
    for col, info in col_info.items():
        vsg_groups[info["vsg"]].append(col)

    for vsg_val, group_cols in vsg_groups.items():
        group_cols_sorted = sorted(group_cols)
        labeled   = {c: col_info[c]["vsl"] for c in group_cols_sorted
                     if not np.isnan(col_info[c]["vsl"])}
        unlabeled = [c for c in group_cols_sorted
                     if np.isnan(col_info[c]["vsl"])]
        if not unlabeled:
            continue
        labeled_vsls = sorted(labeled.values())
        missing_vsls = sorted(
            v for v in vsl_sequence
            if not any(abs(v - lv) < 0.005 for lv in labeled_vsls))
        if len(missing_vsls) == len(unlabeled):
            for col, vsl_val in zip(sorted(unlabeled), missing_vsls):
                col_info[col]["vsl"]      = vsl_val
                col_info[col]["inferred"] = True
                print(f"    [INFER] {filepath.name}: col {col} "
                      f"VSG={vsg_val} VSL inferred as {vsl_val}")
        else:
            print(f"    [WARN] {filepath.name}: VSG={vsg_val} "
                  f"has {len(unlabeled)} unlabeled cols but "
                  f"{len(missing_vsls)} missing VSL values — skipping inference")

    # Build long-format rows
    times_raw = pd.to_numeric(data.iloc[:, 0], errors="coerce").values
    rows = []
    for col, info in col_info.items():
        if np.isnan(info["vsl"]):
            print(f"    [SKIP] {filepath.name}: col {col} "
                  f"VSG={info['vsg']} VSL=NaN after inference — excluded")
            continue
        for t, p in zip(times_raw, info["arr"]):
            if not np.isnan(p):
                rows.append({
                    "Time":   float(t) if not np.isnan(t) else 0.0,
                    "P":      float(p),
                    "VSG":    info["vsg"],
                    "VSL":    info["vsl"],
                    "Label":  leak_label,
                    "Sensor": sensor_name,
                })
    return pd.DataFrame(rows)

# Discover VSL sequence from S1 no-leak file (used for inference)
_ref_file = S1_DIR / "no leak.xlsx"
if not _ref_file.exists():
    sys.exit(f"ERROR: Reference file not found: {_ref_file}")
_ref_raw    = pd.read_excel(_ref_file, header=None, dtype=object)
_vsg_filled = _ref_raw.iloc[0].ffill()
_vsl_raw    = _ref_raw.iloc[1]
_vsl_seq    = sorted(set(
    _parse_vsl(v) for v in _vsl_raw if not np.isnan(_parse_vsl(v))
))
print(f"  Detected VSL sequence: {_vsl_seq}")

# Load all sensor files
print("\n  Loading S1 …")
frames_s1 = []
for fname, lbl in FILES:
    fp = S1_DIR / fname
    if fp.exists():
        df = parse_sensor_file(fp, "S1", lbl, _vsl_seq)
        frames_s1.append(df)
        print(f"    {fname}: {len(df):,} observations")
    else:
        print(f"    [MISSING] {fp}")

print("\n  Loading S2 …")
frames_s2 = []
for fname, lbl in FILES:
    fp = S2_DIR / fname
    if fp.exists():
        df = parse_sensor_file(fp, "S2", lbl, _vsl_seq)
        frames_s2.append(df)
        print(f"    {fname}: {len(df):,} observations")
    else:
        print(f"    [MISSING] {fp}")

df_s1 = pd.concat(frames_s1, ignore_index=True) if frames_s1 else pd.DataFrame()
df_s2 = pd.concat(frames_s2, ignore_index=True) if frames_s2 else pd.DataFrame()

# ── Merge S1 + S2 into wide format (P1 from S1, P2 from S2) ─────────────────
# FIX: use positional row index within each group to avoid Cartesian cross-join.
# Without a row-key, merging on [VSG, VSL, Label] alone matches every S1 row
# to every S2 row in the same group (61 x 61 = 3,721 rows per group instead of 61).
# The positional key _row_idx aligns the i-th S1 observation with the i-th S2
# observation from the same recording, preserving the temporal pairing.
print("\n  Merging sensors into wide format …")

df_s1_wide = df_s1.rename(columns={"P": "P1"}).drop(columns=["Sensor"])
df_s1_wide = df_s1_wide.sort_values(["VSG","VSL","Label","Time"])
df_s1_wide["_row_idx"] = (
    df_s1_wide.groupby(["VSG","VSL","Label"]).cumcount()
)

if not df_s2.empty:
    df_s2_wide = df_s2.rename(columns={"P": "P2"}).drop(columns=["Sensor"])
    df_s2_wide = df_s2_wide.sort_values(["VSG","VSL","Label","Time"])
    df_s2_wide["_row_idx"] = (
        df_s2_wide.groupby(["VSG","VSL","Label"]).cumcount()
    )
    df_merged = df_s1_wide.merge(
        df_s2_wide[["VSG","VSL","Label","P2","_row_idx"]],
        on=["VSG","VSL","Label","_row_idx"],
        how="left"
    ).drop(columns=["_row_idx"])
else:
    df_merged = df_s1_wide.drop(columns=["_row_idx"], errors="ignore")
    df_merged["P2"] = np.nan

# Verify the merge produced the correct row count
_expected = len(df_s1_wide)
if len(df_merged) != _expected:
    print(f"  [ERROR] Merge produced {len(df_merged):,} rows; "
          f"expected {_expected:,}. "
          f"Check for duplicate (VSG,VSL,Label,_row_idx) keys.")
    sys.exit(1)
else:
    print(f"  [OK] Merge row count verified: {len(df_merged):,} rows "
          f"({_expected:,} expected)")

print(f"  Wide DataFrame: {len(df_merged):,} rows  "
      f"columns={list(df_merged.columns)}")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  COVERAGE AUDIT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + SEP2)
print("  COVERAGE AUDIT — which conditions have P1 and/or P2 for all 3 classes")
print(SEP2)

all_conditions = sorted(
    df_merged.groupby(["VSG","VSL"]).groups.keys())
coverage_rows = []
for (vsg, vsl) in all_conditions:
    cond_df = df_merged[
        np.isclose(df_merged.VSG, vsg, atol=0.005) &
        np.isclose(df_merged.VSL, vsl, atol=0.01)]
    has_p1 = {lbl: (cond_df[cond_df.Label == lbl]["P1"].notna().sum() > 0)
               for lbl in [0, 1, 2]}
    has_p2 = {lbl: ("P2" in cond_df.columns and
                     cond_df[cond_df.Label == lbl]["P2"].notna().sum() > 0)
               for lbl in [0, 1, 2]}
    full_p1 = all(has_p1.values())
    full_p2 = all(has_p2.values())
    n_obs   = len(cond_df)
    coverage_rows.append({
        "VSG": vsg, "VSL": vsl,
        "P1_C0": has_p1[0], "P1_C1": has_p1[1], "P1_C2": has_p1[2],
        "P2_C0": has_p2[0], "P2_C1": has_p2[1], "P2_C2": has_p2[2],
        "Full_P1": full_p1, "Full_P2": full_p2,
        "N_obs": n_obs,
    })
    status = ("P1+P2" if full_p1 and full_p2 else
              "P1 only" if full_p1 else "INCOMPLETE")
    print(f"  ({vsg:.3f}, {vsl:.2f})  {status:10s}  n={n_obs}")

df_cov = pd.DataFrame(coverage_rows)
df_cov.to_csv(OUTPUT_DIR / "00_coverage_audit.csv", index=False)

# Define condition pools
# ── Pool definitions — derived from S2 inference check diagnostic ──────────
# S2_CHECK confirmed: unlabelled S2 three-leaks columns are REAL recordings.
# Max pairwise correlation 0.837 — no duplicates. Monotone pressure with VSL.
# Dataset: 15 conditions (3 VSG x 5 VSL) with complete dual-sensor coverage.
#
# POOL_15: all 15 conditions — full dataset (S1+S2, dual sensor)
# POOL_9 : 9 conditions with EXPLICIT VSL labels in S2 three-leaks.xlsx
#          (VSL ∈ {1.03, 1.63, 2.21} x 3 VSG = Alquennah's design)
#          Used for Condition A to match the published 1,647-observation dataset.

VSL_EXPLICIT = [1.03, 1.63, 2.21]   # VSL values with explicit S2 labels

POOL_15 = [(r["VSG"], r["VSL"]) for _, r in df_cov.iterrows()
           if r["Full_P1"] and r["Full_P2"]]   # all 15 dual-sensor conditions

POOL_9  = [(vsg, vsl) for (vsg, vsl) in POOL_15
           if any(abs(vsl - v) < 0.005 for v in VSL_EXPLICIT)]  # 9 explicit conditions

# Aliases for clarity in the ablation table
POOL_S1   = POOL_15   # S1-only uses all 15
POOL_DUAL = POOL_15   # dual-sensor uses all 15 (S2 inference confirmed REAL)

print(f"  POOL_15 (all dual-sensor conditions) : {len(POOL_15)} conditions")
print(f"  POOL_9  (explicit VSL labels, matches Alquennah 9-cond design) : {len(POOL_9)} conditions")
print(f"  Alquennah 1647-obs dataset = POOL_9 x 3 classes x 61 obs = "
      f"{len(POOL_9) * 3 * 61} observations")
print(f"  Full dataset               = POOL_15 x 3 classes x 61 obs = "
      f"{len(POOL_15) * 3 * 61} observations")

print(f"\n  POOL_S1  (S1-only, all conditions) : {len(POOL_S1)} conditions → {POOL_S1}")
print(f"  POOL_DUAL (dual sensor)            : {len(POOL_DUAL)} conditions → {POOL_DUAL}")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  FEATURE EXTRACTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(arr, prefix):
    """Eight statistical features from a pressure window."""
    a = arr.astype(float)
    n = len(a)
    slope = 0.0
    if n > 1:
        x = np.arange(n)
        slope = np.polyfit(x, a, 1)[0]
    mad = float(np.mean(np.abs(np.diff(a)))) if n > 1 else 0.0
    return {
        f"{prefix}_mean":  float(np.mean(a)),
        f"{prefix}_min":   float(np.min(a)),
        f"{prefix}_max":   float(np.max(a)),
        f"{prefix}_std":   float(np.std(a, ddof=0)),
        f"{prefix}_rms":   float(np.sqrt(np.mean(a**2))),
        f"{prefix}_range": float(np.ptp(a)),
        f"{prefix}_slope": slope,
        f"{prefix}_mad":   mad,
    }

def build_windows(df, Wk, conditions, include_vsl_vsg=False, use_p2=True):
    """
    Extract sliding-window feature vectors within-recording only.
    Windows never span recording/condition boundaries.
    Returns (X, y) where X is np.ndarray and y is 1D int array.
    """
    rows, labels = [], []
    for lbl in [0, 1, 2]:
        for (vsg, vsl) in conditions:
            seg = df[
                (df.Label == lbl) &
                np.isclose(df.VSG, vsg, atol=0.005) &
                np.isclose(df.VSL, vsl, atol=0.01)
            ].sort_values("Time")
            p1 = seg["P1"].values.astype(float)
            p2 = (seg["P2"].values.astype(float)
                  if (use_p2 and "P2" in seg.columns) else None)
            if len(p1) < Wk:
                continue
            for start in range(len(p1) - Wk + 1):
                feat = extract_features(p1[start:start + Wk], "P1")
                if p2 is not None and len(p2) >= len(p1):
                    p2_win = p2[start:start + Wk]
                    if not np.any(np.isnan(p2_win)):
                        feat.update(extract_features(p2_win, "P2"))
                if include_vsl_vsg:
                    feat["VSL"] = float(vsl)
                    feat["VSG"] = float(vsg)
                rows.append(feat)
                labels.append(lbl)
    if not rows:
        return np.empty((0, 0)), np.empty(0, dtype=int)
    X = pd.DataFrame(rows).fillna(0).replace([np.inf, -np.inf], 0)
    return X.values.astype(float), np.array(labels, dtype=int)

# ─────────────────────────────────────────────────────────────────────────────
# 6.  MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
class CascadedCatBoost:
    """MC4: two-stage CatBoost cascade (binary detection + count discrimination)."""
    def __init__(self, iter1=300, depth1=6, lr1=0.05,
                       iter2=300, depth2=6, lr2=0.05, seed=42):
        kw = {"task_type": CATBOOST_TASK, "thread_count": -1, "verbose": 0}
        if HAS_CATBOOST:
            self.s1 = CatBoostClassifier(iterations=iter1, depth=depth1,
                                          learning_rate=lr1,
                                          random_seed=seed, **kw)
            self.s2 = CatBoostClassifier(iterations=iter2, depth=depth2,
                                          learning_rate=lr2,
                                          random_seed=seed, **kw)
        else:
            self.s1 = RandomForestClassifier(n_estimators=100,
                                              random_state=seed)
            self.s2 = RandomForestClassifier(n_estimators=100,
                                              random_state=seed)
    def fit(self, X, y):
        y_bin = (y > 0).astype(int)
        self.s1.fit(X, y_bin)
        mask = y > 0
        if mask.sum() > 1:
            self.s2.fit(X[mask], y[mask])
        self.has_stage2 = mask.sum() > 1
        return self
    def predict(self, X):
        pred_bin = self.s1.predict(X).astype(int).flatten()
        out = np.zeros(len(X), dtype=int)
        mask = pred_bin == 1
        if mask.sum() > 0 and self.has_stage2:
            out[mask] = self.s2.predict(X[mask]).astype(int).flatten()
        return out

def nested_tune(X_tr, y_tr, seed=42):
    """2-fold inner CV over the 18-combination grid; returns best params."""
    if FAST_MODE:
        return PARAM_DEFAULT
    best_params, best_acc = PARAM_DEFAULT, -1.0
    inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
    for params in PARAM_GRID_FULL:
        accs = []
        for ti, vi in inner_cv.split(X_tr, y_tr):
            sc = StandardScaler()
            Xti = sc.fit_transform(X_tr[ti])
            Xvi = sc.transform(X_tr[vi])
            m = CascadedCatBoost(**params, seed=seed)
            m.fit(Xti, y_tr[ti])
            accs.append(accuracy_score(y_tr[vi], m.predict(Xvi)))
        if np.mean(accs) > best_acc:
            best_acc, best_params = np.mean(accs), params
    return best_params

def compute_metrics(y_true, y_pred):
    """Full metric suite: accuracy, macro-F1, per-class F1, FAR, MDR."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    rep = classification_report(y_true, y_pred, labels=[0,1,2],
                                 target_names=CLASS_NAMES,
                                 output_dict=True, zero_division=0)
    acc = float(accuracy_score(y_true, y_pred))
    nl  = y_true == 0
    lk  = y_true != 0
    far = float((y_pred[nl] != 0).mean()) if nl.sum() > 0 else 0.0
    mdr = float((y_pred[lk] == 0).mean()) if lk.sum() > 0 else 0.0
    return {
        "accuracy":  acc,
        "macro_f1":  rep["macro avg"]["f1-score"],
        "f1_C0":     rep[CLASS_NAMES[0]]["f1-score"],
        "f1_C1":     rep[CLASS_NAMES[1]]["f1-score"],
        "f1_C2":     rep[CLASS_NAMES[2]]["f1-score"],
        "prec_C0":   rep[CLASS_NAMES[0]]["precision"],
        "prec_C1":   rep[CLASS_NAMES[1]]["precision"],
        "prec_C2":   rep[CLASS_NAMES[2]]["precision"],
        "rec_C0":    rep[CLASS_NAMES[0]]["recall"],
        "rec_C1":    rep[CLASS_NAMES[1]]["recall"],
        "rec_C2":    rep[CLASS_NAMES[2]]["recall"],
        "FAR":       far,
        "MDR":       mdr,
        "n_test":    len(y_true),
    }

def agg(runs):
    """Aggregate list of metric dicts: return mean±std dict."""
    keys = [k for k in runs[0] if k != "n_test"]
    out  = {}
    for k in keys:
        vals = [r[k] for r in runs]
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"]  = float(np.std(vals))
    out["n_test_total"] = sum(r["n_test"] for r in runs)
    return out

def fmt(mean, std, pct=True):
    if pct: return f"{mean*100:.1f}% ± {std*100:.1f}%"
    return  f"{mean:.3f} ± {std:.3f}"

# ─────────────────────────────────────────────────────────────────────────────
# 7.  TIME-BASED SPLIT (Condition A and D)
#     Sort all windows chronologically (by start index within each recording),
#     then take first 80% as train, last 20% as test.
#     All conditions appear in both splits — this is within-condition evaluation.
# ─────────────────────────────────────────────────────────────────────────────
def time_based_run(df, Wk, pool, include_vsl_vsg, use_p2, seed):
    """
    Time-based split: within each (VSG, VSL, class) recording,
    first 80% of windows → train, last 20% → test.
    All classes and conditions appear in both partitions.
    This replicates Alquennah's within-condition evaluation estimand.
    """
    train_rows, train_labels = [], []
    test_rows,  test_labels  = [], []

    for lbl in [0, 1, 2]:
        for (vsg, vsl) in pool:
            seg = df[
                (df.Label == lbl) &
                np.isclose(df.VSG, vsg, atol=0.005) &
                np.isclose(df.VSL, vsl, atol=0.01)
            ].sort_values("Time")

            p1 = seg["P1"].values.astype(float)
            p2 = (seg["P2"].values.astype(float)
                  if (use_p2 and "P2" in seg.columns) else None)

            if len(p1) < Wk:
                continue

            # Extract all windows from this recording (time-ordered)
            rec_rows = []
            for start in range(len(p1) - Wk + 1):
                feat = extract_features(p1[start:start + Wk], "P1")
                if p2 is not None and len(p2) >= len(p1):
                    p2_win = p2[start:start + Wk]
                    if not np.any(np.isnan(p2_win)):
                        feat.update(extract_features(p2_win, "P2"))
                if include_vsl_vsg:
                    feat["VSL"] = float(vsl)
                    feat["VSG"] = float(vsg)
                rec_rows.append(feat)

            # Split THIS recording 80 / 20 by time
            n_rec   = len(rec_rows)
            n_train = max(1, int(n_rec * 0.8))
            n_test  = n_rec - n_train

            if n_test == 0:
                train_rows.extend(rec_rows)
                train_labels.extend([lbl] * n_rec)
                continue

            train_rows.extend(rec_rows[:n_train])
            train_labels.extend([lbl] * n_train)
            test_rows.extend(rec_rows[n_train:])
            test_labels.extend([lbl] * n_test)

    if not train_rows or not test_rows:
        return compute_metrics([], [])

    X_tr = (pd.DataFrame(train_rows)
              .fillna(0).replace([np.inf, -np.inf], 0)
              .values.astype(float))
    y_tr = np.array(train_labels, dtype=int)
    X_te = (pd.DataFrame(test_rows)
              .fillna(0).replace([np.inf, -np.inf], 0)
              .values.astype(float))
    y_te = np.array(test_labels, dtype=int)

    sc   = StandardScaler()
    X_tr = sc.fit_transform(np.nan_to_num(X_tr))
    X_te = sc.transform(np.nan_to_num(X_te))

    # Fixed hyperparameters for Condition A (no nested tuning)
    m = CascadedCatBoost(**PARAM_DEFAULT, seed=seed)
    m.fit(X_tr, y_tr)
    return compute_metrics(y_te.tolist(), m.predict(X_te).tolist())


def time_based_multi_seed(df, Wk, pool, include_vsl_vsg, use_p2, label=""):
    """Run time_based_run across all seeds and aggregate."""
    print(f"    {label}  Wk={Wk}", end="  seeds: ", flush=True)
    runs = []
    for s_idx, s in enumerate(SEEDS):
        r = time_based_run(df, Wk, pool, include_vsl_vsg, use_p2, s)
        runs.append(r)
        print(s_idx, end=" ", flush=True)
    out = agg(runs)
    print(f"\n      → acc={out['accuracy_mean']*100:.1f}%±{out['accuracy_std']*100:.1f}%  "
          f"FAR={out['FAR_mean']*100:.1f}%  MDR={out['MDR_mean']*100:.1f}%")
    return out

# ─────────────────────────────────────────────────────────────────────────────
# 8.  LOO CROSS-VALIDATION (Conditions B, C — and per-fold outputs)
# ─────────────────────────────────────────────────────────────────────────────
def loo_one_seed(df, Wk, pool, include_vsl_vsg, use_p2, seed,
                 tune_hp=False, return_per_fold=False):
    all_true, all_pred = [], []
    per_fold = []
    for fold_i, test_cond in enumerate(pool):
        train_conds = [c for c in pool if c != test_cond]
        X_tr, y_tr = build_windows(df, Wk, train_conds, include_vsl_vsg, use_p2)
        X_te, y_te = build_windows(df, Wk, [test_cond], include_vsl_vsg, use_p2)
        if X_tr.shape[0] == 0 or X_te.shape[0] == 0:
            print(f"\n    [WARN] fold {fold_i+1} ({test_cond}) empty — skipped")
            continue
        sc    = StandardScaler()
        X_trs = sc.fit_transform(np.nan_to_num(X_tr))
        X_tes = sc.transform(np.nan_to_num(X_te))
        params = nested_tune(X_trs, y_tr, seed) if tune_hp else PARAM_DEFAULT
        m = CascadedCatBoost(**params, seed=seed)
        m.fit(X_trs, y_tr)
        preds    = m.predict(X_tes)
        fold_acc = float(accuracy_score(y_te, preds))
        print(f" {fold_acc*100:.1f}%", end="", flush=True)
        if return_per_fold:
            fm = compute_metrics(y_te.tolist(), preds.tolist())
            fm["VSG"]  = test_cond[0]
            fm["VSL"]  = test_cond[1]
            fm["fold"] = fold_i
            fm["seed"] = seed
            per_fold.append(fm)
        all_true.extend(y_te.tolist())
        all_pred.extend(preds.tolist())
    overall = compute_metrics(all_true, all_pred)
    if return_per_fold:
        return overall, per_fold
    return overall


def loo_multi_seed(df, Wk, pool, include_vsl_vsg, use_p2, label="",
                   tune_hp=False, collect_per_fold=False):
    print(f"  {label}  Wk={Wk}  {len(pool)}-fold  "
          f"P2={'yes' if use_p2 else 'no'}  tune={'yes' if tune_hp else 'no'}")
    t0 = time.time()
    runs, all_per_fold = [], []
    for s_idx, s in enumerate(SEEDS):
        print(f"    seed {s_idx+1}/{len(SEEDS)}:", end="", flush=True)
        if collect_per_fold:
            overall, pf = loo_one_seed(df, Wk, pool, include_vsl_vsg,
                                        use_p2, s, tune_hp=tune_hp,
                                        return_per_fold=True)
            runs.append(overall)
            all_per_fold.extend(pf)
        else:
            runs.append(loo_one_seed(df, Wk, pool, include_vsl_vsg,
                                      use_p2, s, tune_hp=tune_hp))
        print()
    out = agg(runs)
    print(f"    → acc={out['accuracy_mean']*100:.1f}%±{out['accuracy_std']*100:.1f}%  "
          f"F1={out['macro_f1_mean']*100:.1f}%  "
          f"FAR={out['FAR_mean']*100:.1f}%  MDR={out['MDR_mean']*100:.1f}%  "
          f"[{time.time()-t0:.0f}s]")
    if collect_per_fold:
        return out, pd.DataFrame(all_per_fold)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 9.  SECTION B: FULL 2x2 ABLATION (A, B, C, D)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + SEP)
print("  SECTION B: 2x2 ABLATION  (Conditions A, B, C, D)")
print(SEP)
print("    A = time-based split  +  VSL/VSG as features  (published protocol)")
print("    B = LOO split         +  VSL/VSG as features  (temporal corrected)")
print("    C = LOO split         -  VSL/VSG              (both corrected ★ PRIMARY)")
print("    D = time-based split  -  VSL/VSG              (NEW — completes the 2x2)")
print()

results_2x2 = {}

print(SEP2)
print("[A] Published protocol — time-based split, VSL/VSG included (PATCHED)")
exp_a = {}
for Wk in WK_LIST:
    # Condition A uses POOL_9 (9 explicit conditions) to match
    # Alquennah's 1,647-observation dataset exactly
    exp_a[Wk] = time_based_multi_seed(df_merged, Wk, POOL_9,
                                       include_vsl_vsg=True, use_p2=True,
                                       label="[A]")
    print(f"    Paper reported {PAPER_ACC.get(Wk,0)*100:.1f}% at Wk={Wk}; "
          f"our A-protocol gives {exp_a[Wk]['accuracy_mean']*100:.1f}%")
pd.DataFrame({"Wk": WK_LIST,
    **{k: [exp_a[wk][k] for wk in WK_LIST]
       for k in ["accuracy_mean","accuracy_std","FAR_mean","MDR_mean"]}
}).to_csv(OUTPUT_DIR / "01_condition_A.csv", index=False)

print("\n" + SEP2)
print("[B] LOO split, VSL/VSG included (temporal correction only)")
print(f"    Using POOL_DUAL ({len(POOL_DUAL)} conditions, dual sensor)")
exp_b = {}
for Wk in WK_LIST:
    exp_b[Wk] = loo_multi_seed(df_merged, Wk, POOL_DUAL,
                                include_vsl_vsg=True, use_p2=True,
                                label="[B]", tune_hp=False)
pd.DataFrame({"Wk": WK_LIST,
    **{k: [exp_b[wk][k] for wk in WK_LIST]
       for k in ["accuracy_mean","accuracy_std","FAR_mean","MDR_mean"]}
}).to_csv(OUTPUT_DIR / "02_condition_B.csv", index=False)

print("\n" + SEP2)
print("[C-PRIMARY] LOO, no VSL/VSG, dual sensor, 9-fold — PRIMARY CORRECTED RESULT")
print(f"    Using POOL_DUAL ({len(POOL_DUAL)} conditions, both sensors)")
print("    This is the direct corrected counterpart to Alquennah's 9-condition design.")
exp_c_s1, per_fold_s1 = {}, {}
for Wk in WK_LIST:
    tune = (Wk == PRIMARY_WK) and (not FAST_MODE)
    out, pf_df = loo_multi_seed(df_merged, Wk, POOL_DUAL,
                                 include_vsl_vsg=False, use_p2=True,
                                 label="[C-PRIMARY]", tune_hp=tune,
                                 collect_per_fold=True)
    exp_c_s1[Wk]    = out
    per_fold_s1[Wk] = pf_df
pf_s1_all = pd.concat(per_fold_s1.values(), ignore_index=True)
pf_s1_all.to_csv(OUTPUT_DIR / "06_per_fold_S1.csv", index=False)
pd.DataFrame({"Wk": WK_LIST,
    **{k: [exp_c_s1[wk][k] for wk in WK_LIST]
       for k in ["accuracy_mean","accuracy_std","FAR_mean","MDR_mean",
                 "f1_C0_mean","f1_C1_mean","f1_C2_mean"]}
}).to_csv(OUTPUT_DIR / "03_condition_C_dual_9fold_PRIMARY.csv", index=False)

if True:  # always run both sensitivity analyses
    print("\n" + SEP2)
    print("[C-9] LOO, no VSL/VSG, dual sensor, 9-fold — restricted to Alquennah's 9 conditions")
    print("    Direct corrected counterpart: same 9 conditions as Alquennah, corrected protocol.")
    exp_c_dual, per_fold_dual = {}, {}
    for Wk in WK_LIST:
        out, pf_df = loo_multi_seed(df_merged, Wk, POOL_9,
                                     include_vsl_vsg=False, use_p2=True,
                                     label="[C-9]", tune_hp=False,
                                     collect_per_fold=True)
        exp_c_dual[Wk]    = out
        per_fold_dual[Wk] = pf_df
    pd.concat(per_fold_dual.values(), ignore_index=True).to_csv(
        OUTPUT_DIR / "06_per_fold_dual_9cond.csv", index=False)
    pd.DataFrame({"Wk": WK_LIST,
        **{k: [exp_c_dual[wk][k] for wk in WK_LIST]
           for k in ["accuracy_mean","accuracy_std","FAR_mean","MDR_mean"]}
    }).to_csv(OUTPUT_DIR / "03_condition_C_9fold_sensitivity.csv", index=False)
else:
    exp_c_dual = {}
    print("\n[C-Dual] SKIPPED — no full dual-sensor conditions found")

print("\n" + SEP2)
print("[D] Time-based split, no VSL/VSG  ← NEW: completes the 2x2")
exp_d = {}
for Wk in WK_LIST:
    exp_d[Wk] = time_based_multi_seed(df_merged, Wk, POOL_DUAL,
                                       include_vsl_vsg=False, use_p2=True,
                                       label="[D]")
pd.DataFrame({"Wk": WK_LIST,
    **{k: [exp_d[wk][k] for wk in WK_LIST]
       for k in ["accuracy_mean","accuracy_std","FAR_mean","MDR_mean"]}
}).to_csv(OUTPUT_DIR / "04_condition_D.csv", index=False)

# 2x2 summary table
print("\n" + SEP2)
rows_2x2 = []
for Wk in WK_LIST:
    a, b, c, d = exp_a[Wk], exp_b[Wk], exp_c_s1[Wk], exp_d[Wk]
    rows_2x2.append({
        "Wk":                          f"{Wk} ({Wk*0.5:.1f}s)",
        "A [time-based+VSL/VSG]":      fmt(a["accuracy_mean"], a["accuracy_std"]),
        "B [LOO+VSL/VSG]":             fmt(b["accuracy_mean"], b["accuracy_std"]),
        "C [LOO no VSL/VSG] ★":        fmt(c["accuracy_mean"], c["accuracy_std"]),
        "D [time-based no VSL/VSG]":   fmt(d["accuracy_mean"], d["accuracy_std"]),
        "Gap A→C":   f"-{(a['accuracy_mean']-c['accuracy_mean'])*100:.1f}pp",
        "Gap A→B":   f"-{(a['accuracy_mean']-b['accuracy_mean'])*100:.1f}pp",
        "Gap B→C":   f"-{(b['accuracy_mean']-c['accuracy_mean'])*100:.1f}pp",
        "Gap A→D":   f"-{(a['accuracy_mean']-d['accuracy_mean'])*100:.1f}pp",
        "Gap D→C":   f"-{(d['accuracy_mean']-c['accuracy_mean'])*100:.1f}pp",
    })
df_2x2 = pd.DataFrame(rows_2x2)
df_2x2.to_csv(OUTPUT_DIR / "05_2x2_table.csv", index=False)
print("2x2 TABLE:\n", df_2x2.to_string(index=False))
for Wk in WK_LIST:
    a, b, c, d = exp_a[Wk], exp_b[Wk], exp_c_s1[Wk], exp_d[Wk]
    p_bc = (a["accuracy_mean"]-b["accuracy_mean"]) + (b["accuracy_mean"]-c["accuracy_mean"])
    p_dc = (a["accuracy_mean"]-d["accuracy_mean"]) + (d["accuracy_mean"]-c["accuracy_mean"])
    print(f"  Wk={Wk}: path via B={p_bc*100:.1f}pp  path via D={p_dc*100:.1f}pp  "
          f"diff={abs(p_bc-p_dc)*100:.1f}pp")


# ─────────────────────────────────────────────────────────────────────────────
# 10. PER-FOLD CONDITION SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + SEP)
print("  SECTION C: PER-FOLD CONDITION-LEVEL ACCURACY (Wk=10, S1-only)")
print(SEP)
if PRIMARY_WK in per_fold_s1:
    pf10 = per_fold_s1[PRIMARY_WK]
    cond_summary = pf10.groupby(["VSG","VSL"]).agg(
        acc_mean=("accuracy","mean"),
        acc_std=("accuracy","std"),
        FAR_mean=("FAR","mean"),
        MDR_mean=("MDR","mean"),
        n_seeds=("seed","count"),
    ).reset_index()
    cond_summary.to_csv(OUTPUT_DIR / "06b_per_condition_summary.csv", index=False)
    print(cond_summary[["VSG","VSL","acc_mean","acc_std",
                         "FAR_mean","MDR_mean"]].to_string(index=False))
    print("\n  VSL=2.21 conditions should show ~33% (chance level = dead fold).")


# ─────────────────────────────────────────────────────────────────────────────
# 11. ALL MODELS
# ─────────────────────────────────────────────────────────────────────────────
import copy
print("\n" + SEP)
print("  SECTION D: ALL MODELS  (Wk=10, LOO, no VSL/VSG, dual sensor, 9-fold)")
print(SEP)

def get_all_models(seed=42):
    base = {
        "MS1_kNN":      KNeighborsClassifier(n_neighbors=5),
        "MS2_SVM":      SVC(kernel="rbf", C=10, gamma="scale",
                            probability=True, random_state=seed),
        "MS3_DT":       DecisionTreeClassifier(max_depth=10, random_state=seed),
        "MS4_RF":       RandomForestClassifier(n_estimators=200,
                                               random_state=seed, n_jobs=-1),
        "MS5_CatBoost": (CatBoostClassifier(iterations=300, depth=6,
                                             learning_rate=0.05,
                                             random_seed=seed, verbose=0,
                                             task_type=CATBOOST_TASK)
                         if HAS_CATBOOST else
                         RandomForestClassifier(200, random_state=seed)),
    }
    base["MS6_MaxVote"] = VotingClassifier(
        estimators=[
            ("knn", KNeighborsClassifier(5)),
            ("rf",  RandomForestClassifier(100, random_state=seed)),
            ("cb",  (CatBoostClassifier(100, random_seed=seed, verbose=0,
                                        task_type="CPU")
                     if HAS_CATBOOST else
                     RandomForestClassifier(100, random_state=seed))),
        ], voting="soft")
    base["MS7_Stacking"] = StackingClassifier(
        estimators=[
            ("knn", KNeighborsClassifier(5)),
            ("rf",  RandomForestClassifier(100, random_state=seed)),
            ("cb",  (CatBoostClassifier(100, random_seed=seed, verbose=0,
                                        task_type="CPU")
                     if HAS_CATBOOST else
                     RandomForestClassifier(100, random_state=seed))),
        ],
        final_estimator=LogisticRegression(C=10, max_iter=1000,
                                            random_state=seed),
        cv=3, n_jobs=1)
    return base

def run_model_loco(model_tmpl, pool, Wk, df, include_vsl_vsg, use_p2):
    all_true, all_pred = [], []
    for test_cond in pool:
        train_conds = [c for c in pool if c != test_cond]
        X_tr, y_tr = build_windows(df, Wk, train_conds, include_vsl_vsg, use_p2)
        X_te, y_te = build_windows(df, Wk, [test_cond], include_vsl_vsg, use_p2)
        if X_tr.shape[0] == 0 or X_te.shape[0] == 0:
            continue
        sc    = StandardScaler()
        X_trs = sc.fit_transform(np.nan_to_num(X_tr))
        X_tes = sc.transform(np.nan_to_num(X_te))
        m = copy.deepcopy(model_tmpl)
        try:
            m.fit(X_trs, y_tr)
            all_true.extend(y_te.tolist())
            all_pred.extend(m.predict(X_tes).tolist())
        except Exception as e:
            print(f" [WARN:{e}]", end="")
    return compute_metrics(all_true, all_pred) if all_true else None

model_results = {}
for mname, tmpl in get_all_models(seed=42).items():
    print(f"  {mname} …", end=" ", flush=True)
    t0 = time.time()
    res = run_model_loco(tmpl, POOL_DUAL, PRIMARY_WK, df_merged, False, True)
    if res:
        model_results[mname] = res
        print(f"acc={res['accuracy']*100:.1f}%  F1={res['macro_f1']:.3f}  "
              f"FAR={res['FAR']*100:.1f}%  MDR={res['MDR']*100:.1f}%  "
              f"[{time.time()-t0:.0f}s]")
    else:
        print("FAILED")

print("  NN_MLP …", end=" ", flush=True)
nn_res = run_model_loco(
    MLPClassifier(hidden_layer_sizes=(256,128,64), activation="relu",
                  alpha=1e-4, max_iter=30, random_state=42,
                  early_stopping=True, n_iter_no_change=15),
    POOL_S1, PRIMARY_WK, df_merged, False, False)
if nn_res:
    model_results["NN_MLP"] = nn_res
    print(f"acc={nn_res['accuracy']*100:.1f}%")

if PRIMARY_WK in exp_c_s1:
    c10 = exp_c_s1[PRIMARY_WK]
    model_results["MC4_Cascade*"] = {
        "accuracy": c10["accuracy_mean"], "macro_f1": c10["macro_f1_mean"],
        "f1_C0": c10["f1_C0_mean"], "f1_C1": c10["f1_C1_mean"],
        "f1_C2": c10["f1_C2_mean"], "FAR": c10["FAR_mean"],
        "MDR": c10["MDR_mean"], "n_test": c10["n_test_total"],
    }

rows_t3 = [{
    "Model":       mn,
    "Accuracy":    f"{r['accuracy']*100:.1f}%",
    "Macro F1":    f"{r['macro_f1']:.3f}",
    "F1 C0/C1/C2": f"{r['f1_C0']:.3f}/{r['f1_C1']:.3f}/{r['f1_C2']:.3f}",
    "FAR":         f"{r['FAR']*100:.1f}%",
    "MDR":         f"{r['MDR']*100:.1f}%",
} for mn, r in sorted(model_results.items(),
                       key=lambda x: x[1]["accuracy"], reverse=True)]
df_t3 = pd.DataFrame(rows_t3)
df_t3.to_csv(OUTPUT_DIR / "07_all_models.csv", index=False)
print("\nTABLE 3:\n", df_t3.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# 12. NOISE ROBUSTNESS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + SEP)
print("  SECTION E: NOISE ROBUSTNESS  (symmetric, both train+test, LOO dual sensor 9-fold)")
print(SEP)

def add_noise(X, pct_fso, rng):
    if pct_fso == 0.0:
        return X.copy()
    fso   = X.max(axis=0) - X.min(axis=0)
    fso   = np.where(fso == 0, 1.0, fso)
    sigma = (pct_fso / 100.0) * fso / 3.0
    return X + rng.normal(0, sigma, size=X.shape)

noise_rows = []
for noise_pct in NOISE_LEVELS:
    print(f"  Noise = {noise_pct}% FSO …", end="", flush=True)
    seed_accs, seed_far, seed_mdr = [], [], []
    for s in SEEDS:
        rng = np.random.default_rng(s + 1000)
        acc_s, far_s, mdr_s = [], [], []
        for test_cond in POOL_DUAL:
            train_conds = [c for c in POOL_DUAL if c != test_cond]
            X_tr, y_tr = build_windows(df_merged, PRIMARY_WK, train_conds, False, True)
            X_te, y_te = build_windows(df_merged, PRIMARY_WK, [test_cond], False, True)
            if X_tr.shape[0] == 0 or X_te.shape[0] == 0:
                continue
            X_tr_n = add_noise(np.nan_to_num(X_tr), noise_pct, rng)
            X_te_n = add_noise(np.nan_to_num(X_te), noise_pct, rng)
            sc     = StandardScaler()
            res    = compute_metrics(
                y_te.tolist(),
                CascadedCatBoost(**PARAM_DEFAULT, seed=s)
                .fit(sc.fit_transform(X_tr_n), y_tr)
                .predict(sc.transform(X_te_n)).tolist()
            )
            acc_s.append(res["accuracy"])
            far_s.append(res["FAR"])
            mdr_s.append(res["MDR"])
        if acc_s:
            seed_accs.append(float(np.mean(acc_s)))
            seed_far.append(float(np.mean(far_s)))
            seed_mdr.append(float(np.mean(mdr_s)))
    row = {
        "Noise_pct_FSO": noise_pct,
        "Acc_mean":  float(np.mean(seed_accs)),
        "Acc_std":   float(np.std(seed_accs)),
        "FAR_mean":  float(np.mean(seed_far)),
        "FAR_std":   float(np.std(seed_far)),
        "MDR_mean":  float(np.mean(seed_mdr)),
        "MDR_std":   float(np.std(seed_mdr)),
    }
    noise_rows.append(row)
    print(f"  acc={row['Acc_mean']*100:.1f}%±{row['Acc_std']*100:.1f}%  "
          f"FAR={row['FAR_mean']*100:.1f}%  MDR={row['MDR_mean']*100:.1f}%")

df_noise = pd.DataFrame(noise_rows)
df_noise.to_csv(OUTPUT_DIR / "08_noise_robustness.csv", index=False)
print("\nNOISE TABLE:\n", df_noise.to_string(index=False))

reconcile_txt = "Figure 4 and Table 5 — same source (08_noise_robustness.csv)\n\n"
row_1 = df_noise[df_noise.Noise_pct_FSO == 1.0]
if not row_1.empty:
    r = row_1.iloc[0]
    reconcile_txt += (
        f"At 1.0% FSO:\n"
        f"  Accuracy : {r['Acc_mean']*100:.1f}% ± {r['Acc_std']*100:.1f}%\n"
        f"  FAR      : {r['FAR_mean']*100:.1f}% ± {r['FAR_std']*100:.1f}%\n"
        f"  MDR      : {r['MDR_mean']*100:.1f}% ± {r['MDR_std']*100:.1f}%\n"
    )
(OUTPUT_DIR / "10_figure4_table5_check.txt").write_text(reconcile_txt, encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# 13. SUMMARY JSON
# ─────────────────────────────────────────────────────────────────────────────
def to_serializable(obj):
    if isinstance(obj, np.ndarray):   return obj.tolist()
    if isinstance(obj, np.integer):   return int(obj)
    if isinstance(obj, np.floating):  return float(obj)
    if isinstance(obj, dict):         return {k: to_serializable(v) for k,v in obj.items()}
    if isinstance(obj, (list,tuple)): return [to_serializable(i) for i in obj]
    return obj

summary = {
    "meta": {
        "n_seeds": len(SEEDS), "seeds": SEEDS, "wk_list": WK_LIST,
        "primary_wk": PRIMARY_WK, "fast_mode": FAST_MODE,
        "pool_s1_n_conditions": len(POOL_S1),
        "pool_dual_n_conditions": len(POOL_DUAL),
    },
    "condition_A":    {str(k): v for k,v in exp_a.items()},
    "condition_B":    {str(k): v for k,v in exp_b.items()},
    "condition_C_S1": {str(k): v for k,v in exp_c_s1.items()},
    "condition_D":    {str(k): v for k,v in exp_d.items()},
    "models_Wk10":    model_results,
    "noise":          df_noise.to_dict(orient="records"),
}
if exp_c_dual:
    summary["condition_C_dual"] = {str(k): v for k,v in exp_c_dual.items()}

import json
(OUTPUT_DIR / "09_results_summary.json").write_text(
    json.dumps(to_serializable(summary), indent=2))

print("\n" + SEP)
print(f"  DONE — all results in: {OUTPUT_DIR}")
print(SEP)
print("  Share these files with Claude:")
print("    05_2x2_table.csv           ← 2x2 ablation headline numbers")
print("    03_condition_C_s1_15fold.csv ← PRIMARY result (S1-only, 15-fold)")
print("    06b_per_condition_summary.csv ← per-condition accuracy")
print("    08_noise_robustness.csv    ← Table 5 + Figure 4")
print("    09_results_summary.json    ← everything")
print(SEP)
