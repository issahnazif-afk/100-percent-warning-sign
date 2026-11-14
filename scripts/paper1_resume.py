#!/usr/bin/env python3
"""
================================================================================
PAPER 1 — RESUME SCRIPT
================================================================================
Picks up exactly where the main run was killed.

Reads already-completed results from disk (Conditions A, B Wk=6/8, C Wk=6/8).
Runs only the missing pieces:
  1. Condition B  Wk=10  (if not already on disk)
  2. Condition C  Wk=10  (tune=False — fast)
  3. Condition D  Wk=6, 8, 10  (all new)
  4. Condition C-9  Wk=10  (9-condition direct comparison with Alquennah)

Then assembles and prints the complete 2×2 table.
================================================================================
"""

import sys, time, warnings, json
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

try:
    from catboost import CatBoostClassifier
    HAS_CB = True
except ImportError:
    HAS_CB = False
    print("WARNING: CatBoost not found")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS — same as main script
# ─────────────────────────────────────────────────────────────────────────────
import os
_ROOTS = [os.environ.get("LEAK_DATA_ROOT", ""), "data", "."]  # folder holding Dynamic_pressure/
DATA_ROOT  = next((Path(p) for p in _ROOTS if p and (Path(p) / "Dynamic_pressure").exists()), Path("data"))
S1_DIR     = DATA_ROOT / "Dynamic_pressure" / "Sensor 1"
S2_DIR     = DATA_ROOT / "Dynamic_pressure" / "Sensor 2"
OUTPUT_DIR = DATA_ROOT / "Paper1_Revision_Results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS      = list(range(42, 52))   # same 10 seeds as main script
WK_LIST    = [6, 8, 10]
PRIMARY_WK = 10
FILES      = [("no leak.xlsx", 0), ("One leak.xlsx", 1), ("three leaks.xlsx", 2)]
CATBOOST_TASK = "GPU"
PARAM_DEFAULT = {"iter1": 300, "depth1": 6, "lr1": 0.05,
                 "iter2": 300, "depth2": 6, "lr2": 0.05}
VSL_EXPLICIT  = [1.03, 1.63, 2.21]   # explicit S2 labels → POOL_9

SEP  = "=" * 78
SEP2 = "─" * 78

print(SEP)
print("  PAPER 1 RESUME — running only missing conditions")
print(SEP)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load data (unavoidable, ~1 min)
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading data …")

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

def _parse_vsl_sequence(s1_dir):
    raw = pd.read_excel(s1_dir / "no leak.xlsx", header=None, dtype=object)
    return sorted({_try_float(v) for v in raw.iloc[1] if not np.isnan(_try_float(v))})

def parse_sensor_file(filepath, sensor_name, leak_label, vsl_seq):
    raw       = pd.read_excel(Path(filepath), header=None, dtype=object)
    vsg_row   = raw.iloc[0].copy()
    vsl_row   = raw.iloc[1]
    data      = raw.iloc[3:].reset_index(drop=True)
    last_vsg  = np.nan
    vsg_filled = []
    for v in vsg_row:
        f = _try_float(v)
        if not np.isnan(f): last_vsg = f
        vsg_filled.append(last_vsg)
    col_info = {}
    for col in range(1, data.shape[1]):
        vsg = vsg_filled[col] if col < len(vsg_filled) else np.nan
        if np.isnan(vsg): continue
        arr = pd.to_numeric(data.iloc[:, col], errors="coerce").values
        n_valid = int(np.sum(~np.isnan(arr)))
        if n_valid == 0: continue
        vsl = _try_float(vsl_row.iloc[col]) if col < len(vsl_row) else np.nan
        col_info[col] = {"vsg": round(vsg,3), "vsl": vsl,
                         "n_valid": n_valid, "arr": arr}
    # Infer missing VSL
    from collections import defaultdict
    vsg_grps = defaultdict(list)
    for c, info in col_info.items(): vsg_grps[info["vsg"]].append(c)
    for vsg_val, grp_cols in vsg_grps.items():
        labeled   = {c: col_info[c]["vsl"] for c in grp_cols
                     if not np.isnan(col_info[c]["vsl"])}
        unlabeled = [c for c in grp_cols if np.isnan(col_info[c]["vsl"])]
        if not unlabeled: continue
        missing = sorted(v for v in vsl_seq
                         if not any(abs(v-lv)<0.005 for lv in labeled.values()))
        if len(missing) == len(unlabeled):
            for c, vsl_val in zip(sorted(unlabeled), missing):
                col_info[c]["vsl"] = vsl_val
    times_raw = pd.to_numeric(data.iloc[:, 0], errors="coerce").values
    rows = []
    for col, info in col_info.items():
        if np.isnan(info["vsl"]): continue
        for t, p in zip(times_raw, info["arr"]):
            if not np.isnan(p):
                rows.append({"Time": float(t) if not np.isnan(t) else 0.0,
                             "P": float(p), "VSG": info["vsg"],
                             "VSL": info["vsl"], "Label": leak_label,
                             "Sensor": sensor_name})
    return pd.DataFrame(rows)

vsl_seq = _parse_vsl_sequence(S1_DIR)
print(f"  VSL sequence: {vsl_seq}")

frames_s1, frames_s2 = [], []
for fname, lbl in FILES:
    fp = S1_DIR / fname
    if fp.exists(): frames_s1.append(parse_sensor_file(fp,"S1",lbl,vsl_seq))
    fp = S2_DIR / fname
    if fp.exists(): frames_s2.append(parse_sensor_file(fp,"S2",lbl,vsl_seq))

df_s1 = pd.concat(frames_s1, ignore_index=True)
df_s2 = pd.concat(frames_s2, ignore_index=True)

df_s1_wide = df_s1.rename(columns={"P":"P1"}).drop(columns=["Sensor"])
df_s1_wide = df_s1_wide.sort_values(["VSG","VSL","Label","Time"])
df_s1_wide["_idx"] = df_s1_wide.groupby(["VSG","VSL","Label"]).cumcount()

df_s2_wide = df_s2.rename(columns={"P":"P2"}).drop(columns=["Sensor"])
df_s2_wide = df_s2_wide.sort_values(["VSG","VSL","Label","Time"])
df_s2_wide["_idx"] = df_s2_wide.groupby(["VSG","VSL","Label"]).cumcount()

df_merged = df_s1_wide.merge(
    df_s2_wide[["VSG","VSL","Label","P2","_idx"]],
    on=["VSG","VSL","Label","_idx"], how="left"
).drop(columns=["_idx"])

assert len(df_merged) == len(df_s1_wide), \
    f"Merge produced {len(df_merged)} rows, expected {len(df_s1_wide)}"
print(f"  [OK] {len(df_merged):,} rows merged correctly")

# Build pools
all_conditions = sorted(df_merged.groupby(["VSG","VSL"]).groups.keys())
cov = {}
for (vsg,vsl) in all_conditions:
    seg = df_merged[np.isclose(df_merged.VSG,vsg,atol=0.005) &
                    np.isclose(df_merged.VSL,vsl,atol=0.01)]
    s1_ok = all(seg[seg.Label==l]["P1"].notna().sum()>0 for l in [0,1,2])
    s2_ok = all(seg[seg.Label==l]["P2"].notna().sum()>0 for l in [0,1,2])
    cov[(vsg,vsl)] = {"s1": s1_ok, "s2": s2_ok}

POOL_15 = [(vsg,vsl) for (vsg,vsl),v in cov.items() if v["s1"] and v["s2"]]
POOL_9  = [(vsg,vsl) for (vsg,vsl) in POOL_15
           if any(abs(vsl-e)<0.005 for e in VSL_EXPLICIT)]
print(f"  POOL_15: {len(POOL_15)} conditions  |  POOL_9: {len(POOL_9)} conditions")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Feature extraction helpers
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(arr, prefix):
    a = arr.astype(float); n = len(a)
    slope = 0.0
    if n > 1:
        x = np.arange(n, dtype=float)
        slope = np.polyfit(x, a, 1)[0]
    mad = float(np.mean(np.abs(np.diff(a)))) if n > 1 else 0.0
    return {f"{prefix}_mean": float(np.mean(a)),
            f"{prefix}_min":  float(np.min(a)),
            f"{prefix}_max":  float(np.max(a)),
            f"{prefix}_std":  float(np.std(a)),
            f"{prefix}_rms":  float(np.sqrt(np.mean(a**2))),
            f"{prefix}_range":float(np.ptp(a)),
            f"{prefix}_slope":slope,
            f"{prefix}_mad":  mad}

def build_windows(df, Wk, pool, include_vsl_vsg=False, use_p2=True):
    rows, labels = [], []
    for lbl in [0,1,2]:
        for (vsg,vsl) in pool:
            seg = df[
                (df.Label==lbl) &
                np.isclose(df.VSG,vsg,atol=0.005) &
                np.isclose(df.VSL,vsl,atol=0.01)
            ].sort_values("Time")
            p1 = seg["P1"].values.astype(float)
            p2 = seg["P2"].values.astype(float) if use_p2 and "P2" in seg else None
            if len(p1) < Wk: continue
            for start in range(len(p1)-Wk+1):
                feat = extract_features(p1[start:start+Wk], "P1")
                if p2 is not None and len(p2)>=len(p1):
                    w2 = p2[start:start+Wk]
                    if not np.any(np.isnan(w2)):
                        feat.update(extract_features(w2, "P2"))
                if include_vsl_vsg:
                    feat["VSL"]=float(vsl); feat["VSG"]=float(vsg)
                rows.append(feat); labels.append(lbl)
    if not rows: return np.empty((0,0)), np.empty(0,dtype=int)
    X = pd.DataFrame(rows).fillna(0).replace([np.inf,-np.inf],0)
    return X.values.astype(float), np.array(labels,dtype=int)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Model helpers
# ─────────────────────────────────────────────────────────────────────────────
class CascadedCatBoost:
    def __init__(self, iter1=300, depth1=6, lr1=0.05,
                       iter2=300, depth2=6, lr2=0.05, seed=42):
        kw = {"task_type":CATBOOST_TASK,"thread_count":-1,"verbose":0}
        self.s1 = CatBoostClassifier(iterations=iter1,depth=depth1,
                                      learning_rate=lr1,random_seed=seed,**kw)
        self.s2 = CatBoostClassifier(iterations=iter2,depth=depth2,
                                      learning_rate=lr2,random_seed=seed,**kw)
    def fit(self, X, y):
        self.s1.fit(X, (y>0).astype(int))
        mask = y>0
        if mask.sum()>1: self.s2.fit(X[mask], y[mask])
        self.has_s2 = mask.sum()>1
        return self
    def predict(self, X):
        out = np.zeros(len(X),dtype=int)
        mask = self.s1.predict(X).astype(int).flatten()==1
        if mask.sum()>0 and self.has_s2:
            out[mask] = self.s2.predict(X[mask]).astype(int).flatten()
        return out

def compute_metrics(y_true, y_pred):
    yt, yp = np.array(y_true), np.array(y_pred)
    rep = classification_report(yt,yp,labels=[0,1,2],
          target_names=["C0","C1","C2"],output_dict=True,zero_division=0)
    nl = yt==0; lk = yt!=0
    return {"accuracy":  float(accuracy_score(yt,yp)),
            "macro_f1":  rep["macro avg"]["f1-score"],
            "f1_C0":     rep["C0"]["f1-score"],
            "f1_C1":     rep["C1"]["f1-score"],
            "f1_C2":     rep["C2"]["f1-score"],
            "FAR":       float((yp[nl]!=0).mean()) if nl.sum()>0 else 0.0,
            "MDR":       float((yp[lk]==0).mean()) if lk.sum()>0 else 0.0,
            "n_test":    len(yt)}

def agg(runs):
    keys = [k for k in runs[0] if k!="n_test"]
    out  = {f"{k}_mean": float(np.mean([r[k] for r in runs]))
            for k in keys}
    out.update({f"{k}_std": float(np.std([r[k] for r in runs]))
                for k in keys})
    return out

def fmt(m, s): return f"{m*100:.1f}%±{s*100:.1f}%"

def loo_run(df, Wk, pool, include_vsl_vsg, use_p2, seed):
    all_true, all_pred = [], []
    for test_cond in pool:
        train_conds = [c for c in pool if c!=test_cond]
        X_tr,y_tr = build_windows(df,Wk,train_conds,include_vsl_vsg,use_p2)
        X_te,y_te = build_windows(df,Wk,[test_cond],include_vsl_vsg,use_p2)
        if X_tr.shape[0]==0 or X_te.shape[0]==0: continue
        sc = StandardScaler()
        X_trs = sc.fit_transform(np.nan_to_num(X_tr))
        X_tes = sc.transform(np.nan_to_num(X_te))
        m = CascadedCatBoost(**PARAM_DEFAULT, seed=seed)
        m.fit(X_trs, y_tr)
        preds = m.predict(X_tes)
        acc = accuracy_score(y_te, preds)
        print(f" {acc*100:.1f}%", end="", flush=True)
        all_true.extend(y_te.tolist())
        all_pred.extend(preds.tolist())
    return compute_metrics(all_true, all_pred)

def time_based_run(df, Wk, pool, include_vsl_vsg, use_p2, seed):
    train_rows, train_lbl, test_rows, test_lbl = [], [], [], []
    for lbl in [0,1,2]:
        for (vsg,vsl) in pool:
            seg = df[
                (df.Label==lbl) &
                np.isclose(df.VSG,vsg,atol=0.005) &
                np.isclose(df.VSL,vsl,atol=0.01)
            ].sort_values("Time")
            p1 = seg["P1"].values.astype(float)
            p2 = seg["P2"].values.astype(float) if use_p2 and "P2" in seg else None
            if len(p1)<Wk: continue
            rec = []
            for start in range(len(p1)-Wk+1):
                feat = extract_features(p1[start:start+Wk],"P1")
                if p2 is not None and len(p2)>=len(p1):
                    w2=p2[start:start+Wk]
                    if not np.any(np.isnan(w2)):
                        feat.update(extract_features(w2,"P2"))
                if include_vsl_vsg:
                    feat["VSL"]=float(vsl); feat["VSG"]=float(vsg)
                rec.append(feat)
            n_tr = max(1,int(len(rec)*0.8))
            train_rows.extend(rec[:n_tr]); train_lbl.extend([lbl]*n_tr)
            test_rows.extend(rec[n_tr:]);  test_lbl.extend([lbl]*(len(rec)-n_tr))
    if not train_rows or not test_rows:
        return compute_metrics([],[])
    X_tr = pd.DataFrame(train_rows).fillna(0).values.astype(float)
    y_tr = np.array(train_lbl,dtype=int)
    X_te = pd.DataFrame(test_rows).fillna(0).values.astype(float)
    y_te = np.array(test_lbl,dtype=int)
    sc = StandardScaler()
    X_trs = sc.fit_transform(np.nan_to_num(X_tr))
    X_tes = sc.transform(np.nan_to_num(X_te))
    m = CascadedCatBoost(**PARAM_DEFAULT, seed=seed)
    m.fit(X_trs, y_tr)
    return compute_metrics(y_te.tolist(), m.predict(X_tes).tolist())

def multi_seed_loo(label, pool, Wk, include_vsl_vsg, use_p2, fn=loo_run):
    print(f"\n  [{label}]  Wk={Wk}  {len(pool)}-fold  P2={'yes' if use_p2 else 'no'}")
    t0 = time.time()
    runs = []
    for si, s in enumerate(SEEDS):
        print(f"    seed {si+1}/{len(SEEDS)}:", end="", flush=True)
        runs.append(fn(df_merged, Wk, pool, include_vsl_vsg, use_p2, s))
        print()
    out = agg(runs)
    print(f"    → acc={fmt(out['accuracy_mean'],out['accuracy_std'])}  "
          f"F1={out['macro_f1_mean']*100:.1f}%  "
          f"FAR={out['FAR_mean']*100:.1f}%  "
          f"MDR={out['MDR_mean']*100:.1f}%  [{time.time()-t0:.0f}s]")
    return out

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Load already-completed results from disk
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + SEP)
print("  Loading already-completed results from disk")
print(SEP)

def load_csv(fname):
    p = OUTPUT_DIR / fname
    if p.exists():
        df = pd.read_csv(p)
        print(f"  [OK] {fname}  ({len(df)} rows)")
        return df
    print(f"  [MISSING] {fname} — will recompute")
    return None

df_A = load_csv("01_condition_A.csv")
df_B = load_csv("02_condition_B.csv")
df_C = load_csv("03_condition_C_dual_9fold_PRIMARY.csv")

# Reconstruct dicts from CSVs
def csv_to_dict(df, wk_list=WK_LIST):
    if df is None: return {}
    out = {}
    for Wk in wk_list:
        row = df[df["Wk"]==Wk] if "Wk" in df.columns else df.iloc[[Wk//2-3]]
        if row.empty: continue
        r = row.iloc[0]
        out[Wk] = {c: float(r[c]) for c in r.index if c != "Wk"}
    return out

exp_a = csv_to_dict(df_A)
exp_b = csv_to_dict(df_B)
exp_c = csv_to_dict(df_C)
print(f"\n  Condition A complete: Wk in {sorted(exp_a.keys())}")
print(f"  Condition B complete: Wk in {sorted(exp_b.keys())}")
print(f"  Condition C complete: Wk in {sorted(exp_c.keys())}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Run only what is missing
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + SEP)
print("  Running missing conditions")
print(SEP)

missing_b = [Wk for Wk in WK_LIST if Wk not in exp_b]
missing_c = [Wk for Wk in WK_LIST if Wk not in exp_c]

# ── Condition B Wk=10 if missing ─────────────────────────────────────────────
if missing_b:
    print(f"\n  Condition B missing: Wk={missing_b}")
    for Wk in missing_b:
        exp_b[Wk] = multi_seed_loo("B", POOL_15, Wk,
                                    include_vsl_vsg=True, use_p2=True)
    pd.DataFrame([{"Wk":k,**v} for k,v in exp_b.items()]).to_csv(
        OUTPUT_DIR/"02_condition_B.csv", index=False)
else:
    print("  Condition B: all window sizes on disk ✓")

# ── Condition C Wk=10 — tune=False (fast) ────────────────────────────────────
if missing_c:
    print(f"\n  Condition C missing: Wk={missing_c} — running WITHOUT tuning (fast)")
    for Wk in missing_c:
        exp_c[Wk] = multi_seed_loo("C-PRIMARY", POOL_15, Wk,
                                    include_vsl_vsg=False, use_p2=True)
    pd.DataFrame([{"Wk":k,**v} for k,v in exp_c.items()]).to_csv(
        OUTPUT_DIR/"03_condition_C_dual_9fold_PRIMARY.csv", index=False)
else:
    print("  Condition C: all window sizes on disk ✓")

# ── Condition D — all window sizes (always new) ───────────────────────────────
print(f"\n  Condition D — time-based split, no VSL/VSG (all Wk — new)")
exp_d = {}
for Wk in WK_LIST:
    exp_d[Wk] = multi_seed_loo("D", POOL_15, Wk,
                                include_vsl_vsg=False, use_p2=True,
                                fn=time_based_run)
pd.DataFrame([{"Wk":k,**v} for k,v in exp_d.items()]).to_csv(
    OUTPUT_DIR/"04_condition_D.csv", index=False)

# ── Condition C-9 — 9-condition direct comparison with Alquennah ─────────────
print(f"\n  Condition C-9 — LOO, no VSL/VSG, POOL_9 (9 conditions, Alquennah's design)")
exp_c9 = {}
for Wk in WK_LIST:
    exp_c9[Wk] = multi_seed_loo("C-9", POOL_9, Wk,
                                 include_vsl_vsg=False, use_p2=True)
pd.DataFrame([{"Wk":k,**v} for k,v in exp_c9.items()]).to_csv(
    OUTPUT_DIR/"03_condition_C_9fold_sensitivity.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Print the complete 2×2 table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + SEP)
print("  COMPLETE 2×2 ABLATION TABLE")
print(SEP)
print(f"\n  {'Wk':>6}  {'A [time+VSL/VSG]':>22}  {'B [LOO+VSL/VSG]':>22}  "
      f"{'C [LOO no VSL/VSG] ★':>22}  {'D [time no VSL/VSG]':>22}")
print(f"  {'-'*6}  {'-'*22}  {'-'*22}  {'-'*22}  {'-'*22}")

rows_2x2 = []
for Wk in WK_LIST:
    a = exp_a.get(Wk,{}); b = exp_b.get(Wk,{})
    c = exp_c.get(Wk,{}); d = exp_d.get(Wk,{})
    def g(d,k): return d.get(k,float("nan"))
    a_acc = g(a,"accuracy_mean"); a_std = g(a,"accuracy_std")
    b_acc = g(b,"accuracy_mean"); b_std = g(b,"accuracy_std")
    c_acc = g(c,"accuracy_mean"); c_std = g(c,"accuracy_std")
    d_acc = g(d,"accuracy_mean"); d_std = g(d,"accuracy_std")
    gap_ab = (a_acc-b_acc)*100
    gap_bc = (b_acc-c_acc)*100
    gap_ac = (a_acc-c_acc)*100
    gap_ad = (a_acc-d_acc)*100
    gap_dc = (d_acc-c_acc)*100
    print(f"  {Wk:>6}  {fmt(a_acc,a_std):>22}  {fmt(b_acc,b_std):>22}  "
          f"{fmt(c_acc,c_std):>22}  {fmt(d_acc,d_std):>22}")
    rows_2x2.append({
        "Wk": Wk,
        "A_acc": f"{fmt(a_acc,a_std)}",
        "B_acc": f"{fmt(b_acc,b_std)}",
        "C_acc (PRIMARY)": f"{fmt(c_acc,c_std)}",
        "D_acc": f"{fmt(d_acc,d_std)}",
        "Gap A→C (total)": f"−{gap_ac:.1f}pp",
        "Gap A→B (split)": f"−{gap_ab:.1f}pp",
        "Gap B→C (encoding)": f"−{gap_bc:.1f}pp",
        "Gap A→D (encoding alone)": f"−{gap_ad:.1f}pp",
        "Gap D→C (split alone)": f"−{gap_dc:.1f}pp",
    })

print()
print("  DECOMPOSITION:")
print(f"  {'Wk':>4}  {'A→C total':>12}  {'A→B split':>12}  "
      f"{'B→C encoding':>14}  {'A→D encoding':>14}  {'D→C split':>12}  "
      f"{'Additive?':>10}")
print(f"  {'-'*4}  {'-'*12}  {'-'*12}  {'-'*14}  {'-'*14}  {'-'*12}  {'-'*10}")
for Wk in WK_LIST:
    a = exp_a.get(Wk,{}); b = exp_b.get(Wk,{})
    c = exp_c.get(Wk,{}); d = exp_d.get(Wk,{})
    def g(d,k): return d.get(k,float("nan"))
    a_acc=g(a,"accuracy_mean"); b_acc=g(b,"accuracy_mean")
    c_acc=g(c,"accuracy_mean"); d_acc=g(d,"accuracy_mean")
    path_bc = (a_acc-b_acc)+(b_acc-c_acc)
    path_dc = (a_acc-d_acc)+(d_acc-c_acc)
    diff = abs(path_bc-path_dc)*100
    additive = "YES ✓" if diff < 2.0 else f"diff={diff:.1f}pp"
    print(f"  {Wk:>4}  {(a_acc-c_acc)*100:>11.1f}pp  "
          f"{(a_acc-b_acc)*100:>11.1f}pp  "
          f"{(b_acc-c_acc)*100:>13.1f}pp  "
          f"{(a_acc-d_acc)*100:>13.1f}pp  "
          f"{(d_acc-c_acc)*100:>11.1f}pp  "
          f"{additive:>10}")

pd.DataFrame(rows_2x2).to_csv(OUTPUT_DIR/"05_2x2_table.csv", index=False)

# C-9 sensitivity
print(f"\n  C-9 SENSITIVITY (9 conditions, direct comparison with Alquennah):")
for Wk in WK_LIST:
    c9 = exp_c9.get(Wk,{})
    if not c9: continue
    print(f"  Wk={Wk}: {fmt(c9['accuracy_mean'],c9['accuracy_std'])}  "
          f"FAR={c9['FAR_mean']*100:.1f}%  MDR={c9['MDR_mean']*100:.1f}%")

print(f"\n  All results saved to: {OUTPUT_DIR}")
print("  Files written:")
for f in sorted(OUTPUT_DIR.glob("0*.csv")):
    print(f"    {f.name}")
print(SEP)
