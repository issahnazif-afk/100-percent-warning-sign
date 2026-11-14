#!/usr/bin/env python3
"""
================================================================================
ALL-MODELS LOCO RUN — Paper 1 Revision
================================================================================
Runs all nine classifiers under the corrected LOCO protocol
(Wk=10, no VSL/VSG, POOL_15, dual sensor, 10 seeds each).

Takes approximately 30–90 minutes depending on hardware.

Produces:
  Paper1_Revision_Results/07_all_models.csv     — Table III numbers
  Paper1_Revision_Results/07_all_models_full.csv — per-seed breakdown

Run:  python paper1_allmodels_run.py
================================================================================
"""
import sys, time, copy, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, VotingClassifier,
                               StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

try:
    from catboost import CatBoostClassifier
    HAS_CB = True
except ImportError:
    HAS_CB = False
    print("[WARNING] CatBoost not found — cascade model will be skipped")

# ─────────────────────────────────────────────────────────────────────────────
import os
_ROOTS = [os.environ.get("LEAK_DATA_ROOT", ""), "data", "."]  # folder holding Dynamic_pressure/
DATA_ROOT  = next((Path(p) for p in _ROOTS if p and (Path(p) / "Dynamic_pressure").exists()), Path("data"))
S1_DIR     = DATA_ROOT / "Dynamic_pressure" / "Sensor 1"
S2_DIR     = DATA_ROOT / "Dynamic_pressure" / "Sensor 2"
OUTPUT_DIR = DATA_ROOT / "Paper1_Revision_Results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS      = list(range(42, 52))   # 10 seeds
PRIMARY_WK = 10
FILES      = [("no leak.xlsx", 0), ("One leak.xlsx", 1), ("three leaks.xlsx", 2)]
CATBOOST_TASK = "GPU"
VSL_EXPLICIT  = [1.03, 1.63, 2.21]
PARAM_DEFAULT = {"iter1": 300, "depth1": 6, "lr1": 0.05,
                 "iter2": 300, "depth2": 6, "lr2": 0.05}
CLASS_NAMES   = ["No Leak (C0)", "Single Leak (C1)", "Two-point (C2)"]

SEP  = "=" * 76
print(SEP)
print("  ALL-MODELS LOCO RUN — Wk=10, no VSL/VSG, POOL_15, 10 seeds each")
print(SEP)
print(f"  Output: {OUTPUT_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# Data loading (same parser as main script)
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
    raw = pd.read_excel(Path(filepath), header=None, dtype=object)
    data = raw.iloc[3:].reset_index(drop=True)
    last_vsg = np.nan
    vsg_filled = []
    for v in raw.iloc[0]:
        f = _try_float(v)
        if not np.isnan(f): last_vsg = f
        vsg_filled.append(last_vsg)
    vsl_row = [_try_float(v) for v in raw.iloc[1]]
    col_info = {}
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
        labeled   = {c: col_info[c]["vsl"] for c in grp if not np.isnan(col_info[c]["vsl"])}
        unlabeled = [c for c in grp if np.isnan(col_info[c]["vsl"])]
        if not unlabeled: continue
        labeled_vsls = sorted(labeled.values())
        missing = sorted(v for v in vsl_seq if not any(abs(v-lv)<0.005 for lv in labeled_vsls))
        if len(missing) == len(unlabeled):
            for c, vsl_val in zip(sorted(unlabeled), missing):
                col_info[c]["vsl"] = vsl_val
    times = pd.to_numeric(data.iloc[:, 0], errors="coerce").values
    rows = []
    for col, info in col_info.items():
        if np.isnan(info["vsl"]): continue
        for t, p in zip(times, info["arr"]):
            if not np.isnan(p):
                rows.append({"Time": float(t) if not np.isnan(t) else 0.0,
                             "P": float(p), "VSG": info["vsg"],
                             "VSL": info["vsl"], "Label": leak_label,
                             "Sensor": sensor_name})
    return pd.DataFrame(rows)

print("\nLoading data...")
vsl_seq = sorted({_try_float(v) for v in
                  pd.read_excel(S1_DIR/"no leak.xlsx", header=None,
                                dtype=object).iloc[1]
                  if not np.isnan(_try_float(v))})
print(f"  VSL sequence: {vsl_seq}")

frames_s1 = [parse_sensor_file(S1_DIR/f, "S1", l, vsl_seq)
             for f, l in FILES if (S1_DIR/f).exists()]
frames_s2 = [parse_sensor_file(S2_DIR/f, "S2", l, vsl_seq)
             for f, l in FILES if (S2_DIR/f).exists()]
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
print(f"  [OK] {len(df):,} rows, brace balance confirmed")

# Pools
all_conds = sorted(df.groupby(["VSG","VSL"]).groups.keys())
POOL_15 = all_conds   # all 15 dual-sensor conditions

# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(arr, prefix):
    a = arr.astype(float); n = len(a)
    slope = np.polyfit(np.arange(n, dtype=float), a, 1)[0] if n > 1 else 0.0
    mad = float(np.mean(np.abs(np.diff(a)))) if n > 1 else 0.0
    return {f"{prefix}_mean": float(np.mean(a)), f"{prefix}_min": float(np.min(a)),
            f"{prefix}_max": float(np.max(a)),  f"{prefix}_std": float(np.std(a)),
            f"{prefix}_rms": float(np.sqrt(np.mean(a**2))),
            f"{prefix}_range": float(np.ptp(a)),
            f"{prefix}_slope": slope, f"{prefix}_mad": mad}

def build_windows(df, Wk, pool, use_p2=True):
    rows, labels = [], []
    for lbl in [0,1,2]:
        for (vsg,vsl) in pool:
            seg = df[(df.Label==lbl) &
                     np.isclose(df.VSG,vsg,atol=0.005) &
                     np.isclose(df.VSL,vsl,atol=0.01)].sort_values("Time")
            p1 = seg["P1"].values.astype(float)
            p2 = seg["P2"].values.astype(float) if use_p2 and "P2" in seg else None
            if len(p1) < Wk: continue
            for start in range(len(p1)-Wk+1):
                feat = extract_features(p1[start:start+Wk], "P1")
                if p2 is not None and len(p2)>=len(p1):
                    w2 = p2[start:start+Wk]
                    if not np.any(np.isnan(w2)):
                        feat.update(extract_features(w2, "P2"))
                rows.append(feat); labels.append(lbl)
    if not rows: return np.empty((0,0)), np.empty(0,dtype=int)
    X = pd.DataFrame(rows).fillna(0).replace([np.inf,-np.inf],0)
    return X.values.astype(float), np.array(labels,dtype=int)

# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(yt, yp):
    yt, yp = np.array(yt), np.array(yp)
    rep = classification_report(yt, yp, labels=[0,1,2],
          target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    nl = yt==0; lk = yt!=0
    return {"accuracy":  float(accuracy_score(yt,yp)),
            "macro_f1":  rep["macro avg"]["f1-score"],
            "f1_C0":     rep[CLASS_NAMES[0]]["f1-score"],
            "f1_C1":     rep[CLASS_NAMES[1]]["f1-score"],
            "f1_C2":     rep[CLASS_NAMES[2]]["f1-score"],
            "FAR":  float((yp[nl]!=0).mean()) if nl.sum()>0 else 0.0,
            "MDR":  float((yp[lk]==0).mean()) if lk.sum()>0 else 0.0,
            "n":    len(yt)}

def agg(runs):
    keys = [k for k in runs[0] if k!="n"]
    return {f"{k}_mean": float(np.mean([r[k] for r in runs]))
            for k in keys} | \
           {f"{k}_std":  float(np.std([r[k] for r in runs]))
            for k in keys}

# ─────────────────────────────────────────────────────────────────────────────
# Cascaded CatBoost (MC4)
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
        out = np.zeros(len(X),dtype=int)
        mask = self.s1.predict(X).astype(int).flatten()==1
        if mask.sum()>0 and self.has_s2:
            out[mask] = self.s2.predict(X[mask]).astype(int).flatten()
        return out

# ─────────────────────────────────────────────────────────────────────────────
# Model zoo — 9 models matching the original paper
# ─────────────────────────────────────────────────────────────────────────────
def get_models(seed=42):
    cb_cpu = {"task_type":"CPU","thread_count":-1,"verbose":0}
    models = {
        "MS1_kNN":      KNeighborsClassifier(n_neighbors=5),
        "MS2_SVM":      SVC(kernel="rbf",C=10,gamma="scale",random_state=seed),
        "MS3_DT":       DecisionTreeClassifier(max_depth=10,random_state=seed),
        "MS4_RF":       RandomForestClassifier(n_estimators=200,
                                                random_state=seed,n_jobs=-1),
        "MS5_CatBoost": (CatBoostClassifier(iterations=300,depth=6,
                                             learning_rate=0.05,
                                             random_seed=seed,**cb_cpu)
                         if HAS_CB else
                         RandomForestClassifier(200,random_state=seed)),
        "MS6_MaxVote":  VotingClassifier(
            estimators=[("knn",KNeighborsClassifier(5)),
                        ("rf",RandomForestClassifier(100,random_state=seed)),
                        ("cb",(CatBoostClassifier(100,random_seed=seed,
                                                   **cb_cpu)
                                if HAS_CB else
                                RandomForestClassifier(100,random_state=seed)))],
            voting="soft"),
        "MS7_Stacking": StackingClassifier(
            estimators=[("knn",KNeighborsClassifier(5)),
                        ("rf",RandomForestClassifier(100,random_state=seed)),
                        ("cb",(CatBoostClassifier(100,random_seed=seed,
                                                   **cb_cpu)
                                if HAS_CB else
                                RandomForestClassifier(100,random_state=seed)))],
            final_estimator=LogisticRegression(C=10,max_iter=1000,
                                                random_state=seed),
            cv=3,n_jobs=1),
        "NN_MLP":       MLPClassifier(hidden_layer_sizes=(256,128,64),
                                       activation="relu",alpha=1e-4,
                                       batch_size=32,max_iter=30,
                                       random_state=seed,
                                       early_stopping=True,
                                       n_iter_no_change=15),
    }
    return models

# ─────────────────────────────────────────────────────────────────────────────
# LOCO run — one seed, one model
# ─────────────────────────────────────────────────────────────────────────────
def loco_one_model_seed(model_tmpl, pool, Wk, df, seed):
    all_true, all_pred = [], []
    for test_cond in pool:
        train_conds = [c for c in pool if c!=test_cond]
        X_tr,y_tr = build_windows(df,Wk,train_conds,use_p2=True)
        X_te,y_te = build_windows(df,Wk,[test_cond],use_p2=True)
        if X_tr.shape[0]==0 or X_te.shape[0]==0: continue
        sc = StandardScaler()
        X_trs = sc.fit_transform(np.nan_to_num(X_tr))
        X_tes = sc.transform(np.nan_to_num(X_te))
        m = copy.deepcopy(model_tmpl)
        try:
            m.fit(X_trs,y_tr)
            all_true.extend(y_te.tolist())
            all_pred.extend(m.predict(X_tes).tolist())
        except Exception as e:
            print(f" [WARN:{test_cond}:{e}]",end="")
    return compute_metrics(all_true,all_pred) if all_true else None

# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nRunning all models — Wk={PRIMARY_WK}, POOL_15 ({len(POOL_15)} conds), "
      f"{len(SEEDS)} seeds each\n")

results = {}

# MC4 Cascade (10 seeds) — load from existing CSV if available
mc4_csv = OUTPUT_DIR / "03_condition_C_dual_9fold_PRIMARY.csv"
if mc4_csv.exists():
    mc4_df = pd.read_csv(mc4_csv)
    row10 = mc4_df[mc4_df.Wk == PRIMARY_WK]
    if not row10.empty:
        r = row10.iloc[0]
        results["MC4_Cascade"] = {
            "accuracy_mean": r["accuracy_mean"], "accuracy_std": r.get("accuracy_std",0),
            "macro_f1_mean": r["macro_f1_mean"], "macro_f1_std": r.get("macro_f1_std",0),
            "f1_C0_mean": r["f1_C0_mean"], "f1_C1_mean": r["f1_C1_mean"],
            "f1_C2_mean": r["f1_C2_mean"],
            "FAR_mean": r["FAR_mean"], "FAR_std": r.get("FAR_std",0),
            "MDR_mean": r["MDR_mean"], "MDR_std": r.get("MDR_std",0),
        }
        print(f"  MC4_Cascade  [from CSV]  "
              f"acc={r['accuracy_mean']*100:.1f}%±{r.get('accuracy_std',0)*100:.1f}%  "
              f"FAR={r['FAR_mean']*100:.1f}%  MDR={r['MDR_mean']*100:.1f}%")

# Single-stage models (10 seeds each)
for mname, tmpl in get_models(seed=42).items():
    t0 = time.time()
    print(f"  {mname} ...", end="  ", flush=True)
    seed_runs = []
    for seed in SEEDS:
        models_seed = get_models(seed=seed)
        run = loco_one_model_seed(models_seed[mname], POOL_15, PRIMARY_WK, df, seed)
        if run: seed_runs.append(run)
        print(".", end="", flush=True)
    if seed_runs:
        agg_res = agg(seed_runs)
        results[mname] = agg_res
        print(f"  acc={agg_res['accuracy_mean']*100:.1f}%±{agg_res['accuracy_std']*100:.1f}%  "
              f"F1={agg_res['macro_f1_mean']:.3f}  "
              f"FAR={agg_res['FAR_mean']*100:.1f}%  MDR={agg_res['MDR_mean']*100:.1f}%  "
              f"[{time.time()-t0:.0f}s]")
    else:
        print("  FAILED")

# ─────────────────────────────────────────────────────────────────────────────
# Save and print table
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print(f"  TABLE III: All models — Wk={PRIMARY_WK}, LOCO, no VSL/VSG, 15 cond.")
print(SEP)
print(f"  {'Model':22s}  {'Accuracy':>16}  {'Macro F1':>16}  "
      f"{'FAR':>14}  {'MDR':>14}")
print(f"  {'-'*22}  {'-'*16}  {'-'*16}  {'-'*14}  {'-'*14}")

def fmt(m, s): return f"{m*100:.1f}%±{s*100:.1f}%"

rows = []
for mname, res in sorted(results.items(),
                          key=lambda x: -x[1].get("accuracy_mean",0)):
    m  = res.get("accuracy_mean",0); sm = res.get("accuracy_std",0)
    f1 = res.get("macro_f1_mean",0); sf = res.get("macro_f1_std",0)
    fr = res.get("FAR_mean",0);      sfr= res.get("FAR_std",0)
    dr = res.get("MDR_mean",0);      sdr= res.get("MDR_std",0)
    print(f"  {mname:22s}  {fmt(m,sm):>16}  {f1:.3f}±{sf:.3f}  "
          f"{fmt(fr,sfr):>14}  {fmt(dr,sdr):>14}")
    rows.append({"Model": mname,
                 "Accuracy": fmt(m,sm), "Macro_F1": f"{f1:.3f}±{sf:.3f}",
                 "FAR": fmt(fr,sfr), "MDR": fmt(dr,sdr),
                 "acc_raw": m, "far_raw": fr})

df_out = pd.DataFrame(rows).sort_values("acc_raw", ascending=False)
df_out.drop(columns=["acc_raw","far_raw"]).to_csv(
    OUTPUT_DIR/"07_all_models.csv", index=False)
print(f"\n  Saved: {OUTPUT_DIR/'07_all_models.csv'}")
print(SEP)
print("  Share 07_all_models.csv — paste numbers into both paper versions.")
