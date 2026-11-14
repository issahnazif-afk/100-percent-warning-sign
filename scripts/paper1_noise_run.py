#!/usr/bin/env python3
"""
================================================================================
NOISE ROBUSTNESS RUN — Paper 1 Revision
================================================================================
Produces Table V and Figure 4 source data.

Protocol:
  Model    : MC4 cascaded CatBoost (pre-specified primary)
  Wk       : 10  (primary window size)
  Pool     : POOL_15  (15 dual-sensor conditions)
  Features : P1 + P2, no VSL/VSG
  Seeds    : 10 independent seeds per noise level
  Noise    : symmetric Gaussian contamination applied to BOTH training
             and test feature vectors at identical magnitude
  Levels   : 0.0, 0.1, 0.2, 0.5, 1.0 % FSO

FSO definition:
  Per-feature full-scale output = (training-fold max − min) × 1.10
  Noise sigma = (pct_FSO / 100) × FSO / 3   (pct treated as 3σ interval)
  FSO parameters are computed from the training fold ONLY — no test
  information leaks into the noise calibration.

Outputs (written to OUTPUT_DIR):
  08_noise_robustness.csv     — Table V source (all 5 noise levels × all metrics)
  08_figure4_data.csv         — same data, long format for plotting
  08_figure4_table5_check.txt — reconciliation note confirming both outputs
                                derive from a single source array

Runtime estimate: ~15–30 minutes on GPU.

Run:  python paper1_noise_run.py
================================================================================
"""
import sys, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

try:
    from catboost import CatBoostClassifier
    HAS_CB = True
except ImportError:
    HAS_CB = False
    print("[WARNING] CatBoost not found — install catboost and rerun")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
import os
_ROOTS = [os.environ.get("LEAK_DATA_ROOT", ""), "data", "."]  # folder holding Dynamic_pressure/
DATA_ROOT  = next((Path(p) for p in _ROOTS if p and (Path(p) / "Dynamic_pressure").exists()), Path("data"))
S1_DIR     = DATA_ROOT / "Dynamic_pressure" / "Sensor 1"
S2_DIR     = DATA_ROOT / "Dynamic_pressure" / "Sensor 2"
OUTPUT_DIR = DATA_ROOT / "Paper1_Revision_Results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS         = list(range(42, 52))     # 10 seeds
PRIMARY_WK    = 10
NOISE_LEVELS  = [0.0, 0.1, 0.2, 0.5, 1.0]   # % FSO
CATBOOST_TASK = "GPU"
PARAM_DEFAULT = {"iter1": 300, "depth1": 6, "lr1": 0.05,
                 "iter2": 300, "depth2": 6, "lr2": 0.05}
FILES         = [("no leak.xlsx", 0), ("One leak.xlsx", 1), ("three leaks.xlsx", 2)]
CLASS_NAMES   = ["No Leak (C0)", "Single Leak (C1)", "Two-point (C2)"]
VSL_EXPLICIT  = [1.03, 1.63, 2.21]

SEP  = "=" * 76
SEP2 = "─" * 76

print(SEP)
print("  NOISE ROBUSTNESS RUN — Table V + Figure 4 source")
print(SEP)
print(f"  Data root  : {DATA_ROOT}")
print(f"  Output     : {OUTPUT_DIR}")
print(f"  Noise levels: {NOISE_LEVELS} % FSO")
print(f"  Seeds      : {len(SEEDS)}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING (same robust parser as all previous scripts)
# ─────────────────────────────────────────────────────────────────────────────
def _try_float(v):
    if isinstance(v, str):
        for kw in ["VSG","VSL","="]:
            if kw in v:
                for p in v.replace("="," ").split():
                    try:
                        f = float(p)
                        if 0 < f < 100: return f
                    except: pass
                return np.nan
    try:
        f = float(v); return f if not np.isnan(f) else np.nan
    except: return np.nan

def parse_sensor_file(filepath, sensor_name, leak_label, vsl_seq):
    raw  = pd.read_excel(Path(filepath), header=None, dtype=object)
    data = raw.iloc[3:].reset_index(drop=True)
    last_vsg = np.nan
    vsg_filled = []
    for v in raw.iloc[0]:
        f = _try_float(v)
        if not np.isnan(f): last_vsg = f
        vsg_filled.append(last_vsg)
    vsl_row   = [_try_float(v) for v in raw.iloc[1]]
    col_info  = {}
    for col in range(1, data.shape[1]):
        vsg = vsg_filled[col] if col < len(vsg_filled) else np.nan
        if np.isnan(vsg): continue
        arr = pd.to_numeric(data.iloc[:, col], errors="coerce").values
        if int(np.sum(~np.isnan(arr))) == 0: continue
        vsl = vsl_row[col] if col < len(vsl_row) else np.nan
        col_info[col] = {"vsg": round(vsg,3), "vsl": vsl, "arr": arr}
    vsg_grps = defaultdict(list)
    for c, info in col_info.items(): vsg_grps[info["vsg"]].append(c)
    for vsg_val, grp in vsg_grps.items():
        labeled   = {c: col_info[c]["vsl"] for c in grp
                     if not np.isnan(col_info[c]["vsl"])}
        unlabeled = [c for c in grp if np.isnan(col_info[c]["vsl"])]
        if not unlabeled: continue
        missing = sorted(v for v in vsl_seq
                         if not any(abs(v-lv)<0.005 for lv in labeled.values()))
        if len(missing) == len(unlabeled):
            for c, vsl_val in zip(sorted(unlabeled), missing):
                col_info[c]["vsl"] = vsl_val
    times = pd.to_numeric(data.iloc[:,0], errors="coerce").values
    rows  = []
    for col, info in col_info.items():
        if np.isnan(info["vsl"]): continue
        for t, p in zip(times, info["arr"]):
            if not np.isnan(p):
                rows.append({"Time": float(t) if not np.isnan(t) else 0.0,
                             "P": float(p), "VSG": info["vsg"],
                             "VSL": info["vsl"], "Label": leak_label,
                             "Sensor": sensor_name})
    return pd.DataFrame(rows)

print("Loading data...")
vsl_seq = sorted({_try_float(v)
                  for v in pd.read_excel(S1_DIR/"no leak.xlsx",
                                         header=None, dtype=object).iloc[1]
                  if not np.isnan(_try_float(v))})
print(f"  VSL sequence: {vsl_seq}")

frames_s1 = [parse_sensor_file(S1_DIR/f,"S1",l,vsl_seq)
             for f,l in FILES if (S1_DIR/f).exists()]
frames_s2 = [parse_sensor_file(S2_DIR/f,"S2",l,vsl_seq)
             for f,l in FILES if (S2_DIR/f).exists()]

df_s1 = pd.concat(frames_s1, ignore_index=True)
df_s2 = pd.concat(frames_s2, ignore_index=True) if frames_s2 else pd.DataFrame()

df_s1w = df_s1.rename(columns={"P":"P1"}).drop(columns=["Sensor"])
df_s1w = df_s1w.sort_values(["VSG","VSL","Label","Time"])
df_s1w["_idx"] = df_s1w.groupby(["VSG","VSL","Label"]).cumcount()

if not df_s2.empty:
    df_s2w = df_s2.rename(columns={"P":"P2"}).drop(columns=["Sensor"])
    df_s2w = df_s2w.sort_values(["VSG","VSL","Label","Time"])
    df_s2w["_idx"] = df_s2w.groupby(["VSG","VSL","Label"]).cumcount()
    df = df_s1w.merge(df_s2w[["VSG","VSL","Label","P2","_idx"]],
                      on=["VSG","VSL","Label","_idx"], how="left").drop(columns=["_idx"])
else:
    df = df_s1w.drop(columns=["_idx"])
    df["P2"] = np.nan

assert len(df) == len(df_s1w), f"Merge error: {len(df)} rows"
print(f"  [OK] {len(df):,} rows")

# Build POOL_15
all_conds = sorted(df.groupby(["VSG","VSL"]).groups.keys())
POOL_15   = all_conds
print(f"  POOL_15: {len(POOL_15)} conditions\n")

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(arr, prefix):
    a = arr.astype(float); n = len(a)
    slope = np.polyfit(np.arange(n,dtype=float), a, 1)[0] if n>1 else 0.0
    mad   = float(np.mean(np.abs(np.diff(a)))) if n>1 else 0.0
    return {f"{prefix}_mean": float(np.mean(a)),
            f"{prefix}_min":  float(np.min(a)),
            f"{prefix}_max":  float(np.max(a)),
            f"{prefix}_std":  float(np.std(a)),
            f"{prefix}_rms":  float(np.sqrt(np.mean(a**2))),
            f"{prefix}_range":float(np.ptp(a)),
            f"{prefix}_slope":slope,
            f"{prefix}_mad":  mad}

def build_windows(df, Wk, pool, use_p2=True):
    rows, labels = [], []
    for lbl in [0,1,2]:
        for (vsg,vsl) in pool:
            seg = df[(df.Label==lbl) &
                     np.isclose(df.VSG,vsg,atol=0.005) &
                     np.isclose(df.VSL,vsl,atol=0.01)].sort_values("Time")
            p1 = seg["P1"].values.astype(float)
            p2 = (seg["P2"].values.astype(float)
                  if use_p2 and "P2" in seg.columns else None)
            if len(p1) < Wk: continue
            for start in range(len(p1)-Wk+1):
                feat = extract_features(p1[start:start+Wk], "P1")
                if p2 is not None and len(p2)>=len(p1):
                    w2 = p2[start:start+Wk]
                    if not np.any(np.isnan(w2)):
                        feat.update(extract_features(w2,"P2"))
                rows.append(feat); labels.append(lbl)
    if not rows: return np.empty((0,0)), np.empty(0,dtype=int), []
    X   = pd.DataFrame(rows).fillna(0).replace([np.inf,-np.inf],0)
    return X.values.astype(float), np.array(labels,dtype=int), list(X.columns)

# ─────────────────────────────────────────────────────────────────────────────
# MC4 CASCADE
# ─────────────────────────────────────────────────────────────────────────────
class CascadedCatBoost:
    def __init__(self, seed=42):
        kw = {"task_type":CATBOOST_TASK,"thread_count":-1,"verbose":0}
        self.s1 = CatBoostClassifier(iterations=300,depth=6,
                                      learning_rate=0.05,random_seed=seed,**kw)
        self.s2 = CatBoostClassifier(iterations=300,depth=6,
                                      learning_rate=0.05,random_seed=seed,**kw)
    def fit(self, X, y):
        self.s1.fit(X,(y>0).astype(int))
        mask = y>0
        if mask.sum()>1: self.s2.fit(X[mask],y[mask])
        self.has_s2 = mask.sum()>1
        return self
    def predict(self, X):
        out  = np.zeros(len(X),dtype=int)
        mask = self.s1.predict(X).astype(int).flatten()==1
        if mask.sum()>0 and self.has_s2:
            out[mask] = self.s2.predict(X[mask]).astype(int).flatten()
        return out

# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(yt, yp):
    yt, yp = np.array(yt), np.array(yp)
    rep = classification_report(yt,yp,labels=[0,1,2],
          target_names=CLASS_NAMES,output_dict=True,zero_division=0)
    nl = yt==0; lk = yt!=0
    return {"accuracy":  float(accuracy_score(yt,yp)),
            "macro_f1":  rep["macro avg"]["f1-score"],
            "f1_C0":     rep[CLASS_NAMES[0]]["f1-score"],
            "f1_C1":     rep[CLASS_NAMES[1]]["f1-score"],
            "f1_C2":     rep[CLASS_NAMES[2]]["f1-score"],
            "FAR":  float((yp[nl]!=0).mean()) if nl.sum()>0 else 0.0,
            "MDR":  float((yp[lk]==0).mean()) if lk.sum()>0 else 0.0}

# ─────────────────────────────────────────────────────────────────────────────
# NOISE INJECTION (FSO calibrated on training fold only)
# ─────────────────────────────────────────────────────────────────────────────
def add_noise(X, pct_fso, fso_per_col, rng):
    """Add symmetric Gaussian noise at pct_fso % of each feature's FSO."""
    if pct_fso == 0.0:
        return X.copy()
    Xn = X.copy().astype(float)
    for i, fso in enumerate(fso_per_col):
        if fso > 0:
            sigma = (pct_fso / 100.0) * fso / 3.0   # pct as 3σ interval
            Xn[:, i] += rng.normal(0.0, sigma, Xn.shape[0])
    return Xn

# ─────────────────────────────────────────────────────────────────────────────
# MAIN NOISE LOOP
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  NOISE ROBUSTNESS EXPERIMENT")
print(SEP)
print(f"  Protocol: LOCO, POOL_15 ({len(POOL_15)} conds), Wk={PRIMARY_WK}, "
      f"MC4, {len(SEEDS)} seeds × {len(NOISE_LEVELS)} noise levels")
print()

noise_rows = []

for noise_pct in NOISE_LEVELS:
    t0 = time.time()
    print(f"  Noise = {noise_pct:.1f}% FSO ...", end="  ", flush=True)

    seed_accs, seed_f1s  = [], []
    seed_f1C0, seed_f1C1, seed_f1C2 = [], [], []
    seed_far,  seed_mdr  = [], []

    for seed in SEEDS:
        rng = np.random.default_rng(seed + 2000)  # offset to differ from main seeds
        fold_true, fold_pred = [], []

        for test_cond in POOL_15:
            train_conds = [c for c in POOL_15 if c != test_cond]

            X_tr, y_tr, feat_cols = build_windows(df, PRIMARY_WK,
                                                   train_conds, use_p2=True)
            X_te, y_te, _         = build_windows(df, PRIMARY_WK,
                                                   [test_cond], use_p2=True)
            if X_tr.shape[0] == 0 or X_te.shape[0] == 0:
                continue

            X_tr = np.nan_to_num(X_tr)
            X_te = np.nan_to_num(X_te)

            # Compute FSO from training fold only (no test leakage)
            fso_per_col = [
                max((X_tr[:, i].max() - X_tr[:, i].min()) * 1.10, 1e-12)
                for i in range(X_tr.shape[1])
            ]

            # Apply noise symmetrically to both train and test
            X_tr_n = add_noise(X_tr, noise_pct, fso_per_col, rng)
            X_te_n = add_noise(X_te, noise_pct, fso_per_col, rng)

            # Fit scaler on noisy training data
            sc     = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr_n)
            X_te_s = sc.transform(X_te_n)

            m = CascadedCatBoost(seed=seed)
            m.fit(X_tr_s, y_tr)
            preds = m.predict(X_te_s)

            fold_true.extend(y_te.tolist())
            fold_pred.extend(preds.tolist())

        if fold_true:
            res = compute_metrics(fold_true, fold_pred)
            seed_accs.append(res["accuracy"])
            seed_f1s.append(res["macro_f1"])
            seed_f1C0.append(res["f1_C0"])
            seed_f1C1.append(res["f1_C1"])
            seed_f1C2.append(res["f1_C2"])
            seed_far.append(res["FAR"])
            seed_mdr.append(res["MDR"])
        print(".", end="", flush=True)

    row = {
        "Noise_pct_FSO":  noise_pct,
        "Acc_mean":  float(np.mean(seed_accs)),
        "Acc_std":   float(np.std(seed_accs)),
        "F1_mean":   float(np.mean(seed_f1s)),
        "F1_std":    float(np.std(seed_f1s)),
        "F1_C0_mean":float(np.mean(seed_f1C0)),
        "F1_C1_mean":float(np.mean(seed_f1C1)),
        "F1_C2_mean":float(np.mean(seed_f1C2)),
        "FAR_mean":  float(np.mean(seed_far)),
        "FAR_std":   float(np.std(seed_far)),
        "MDR_mean":  float(np.mean(seed_mdr)),
        "MDR_std":   float(np.std(seed_mdr)),
    }
    noise_rows.append(row)
    print(f"  [{time.time()-t0:.0f}s]"
          f"  acc={row['Acc_mean']*100:.1f}%±{row['Acc_std']*100:.1f}%"
          f"  FAR={row['FAR_mean']*100:.1f}%"
          f"  MDR={row['MDR_mean']*100:.1f}%")

df_noise = pd.DataFrame(noise_rows)
df_noise.to_csv(OUTPUT_DIR/"08_noise_robustness.csv", index=False)

# Long format for figure plotting
long_rows = []
for _, r in df_noise.iterrows():
    for metric, col_m, col_s in [
        ("Accuracy", "Acc_mean", "Acc_std"),
        ("FAR",      "FAR_mean", "FAR_std"),
        ("MDR",      "MDR_mean", "MDR_std"),
    ]:
        long_rows.append({
            "Noise_pct_FSO": r["Noise_pct_FSO"],
            "Metric":        metric,
            "Mean":          r[col_m],
            "Std":           r[col_s],
        })
pd.DataFrame(long_rows).to_csv(OUTPUT_DIR/"08_figure4_data.csv", index=False)

# Reconciliation note
r1 = df_noise[df_noise.Noise_pct_FSO == 1.0].iloc[0]
r0 = df_noise[df_noise.Noise_pct_FSO == 0.0].iloc[0]

reconcile = (
    "Figure 4 and Table V — single source reconciliation\n"
    "=" * 60 + "\n\n"
    "Both Table V and Figure 4 derive from 08_noise_robustness.csv.\n"
    "There is no separate noise-figure run — one array, two outputs.\n\n"
    f"At 0.0% FSO (noise-free baseline):\n"
    f"  Accuracy: {r0['Acc_mean']*100:.1f}% ± {r0['Acc_std']*100:.1f}%\n"
    f"  FAR:      {r0['FAR_mean']*100:.1f}% ± {r0['FAR_std']*100:.1f}%\n"
    f"  MDR:      {r0['MDR_mean']*100:.1f}% ± {r0['MDR_std']*100:.1f}%\n\n"
    f"  → Should match Condition C in the 2×2 table (59.1%±0.6%)\n\n"
    f"At 1.0% FSO:\n"
    f"  Accuracy: {r1['Acc_mean']*100:.1f}% ± {r1['Acc_std']*100:.1f}%\n"
    f"  FAR:      {r1['FAR_mean']*100:.1f}% ± {r1['FAR_std']*100:.1f}%\n"
    f"  MDR:      {r1['MDR_mean']*100:.1f}% ± {r1['MDR_std']*100:.1f}%\n"
)
(OUTPUT_DIR/"08_figure4_table5_check.txt").write_text(reconcile, encoding="utf-8")

# ─────────────────────────────────────────────────────────────────────────────
# PRINT TABLE V
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  TABLE V — Noise robustness (symmetric contamination, MC4, Wk=10, LOCO)")
print(SEP)
print(f"  {'Noise':>10}  {'Paper (asymm.)':>16}  {'Corrected Acc.':>16}  "
      f"{'FAR':>14}  {'MDR':>14}")
print(f"  {'-'*10}  {'-'*16}  {'-'*16}  {'-'*14}  {'-'*14}")

paper_vals = {0.0: "100%", 0.1: "100%", 0.2: "99%", 0.5: "N/A", 1.0: "N/A"}
for _, r in df_noise.iterrows():
    pct  = r["Noise_pct_FSO"]
    pval = paper_vals.get(pct, "N/A")
    print(f"  {pct:>9.1f}%  {pval:>16s}  "
          f"{r['Acc_mean']*100:.1f}%±{r['Acc_std']*100:.1f}%  "
          f"{r['FAR_mean']*100:.1f}%±{r['FAR_std']*100:.1f}%  "
          f"{r['MDR_mean']*100:.1f}%±{r['MDR_std']*100:.1f}%")

print(f"\n{SEP}")
print("  Files saved:")
print(f"    {OUTPUT_DIR / '08_noise_robustness.csv'}")
print(f"    {OUTPUT_DIR / '08_figure4_data.csv'}")
print(f"    {OUTPUT_DIR / '08_figure4_table5_check.txt'}")
print(f"\n  Share 08_noise_robustness.csv to fill in Table V in both papers.")
print(f"  Share 08_figure4_data.csv to generate Figure 4.")
print(SEP)
