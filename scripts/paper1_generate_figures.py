#!/usr/bin/env python3
"""
================================================================================
FIGURE GENERATION SCRIPT — Paper 1 (PSEP Submission)
================================================================================
Generates all four figures flagged by the reviewer as having incorrect values.
All figures are saved as 300 DPI PNG ready for LaTeX inclusion.

Figures produced:
  Fig1_Protocol_Comparison.png   — accuracy vs window size, all conditions
  Fig2_Confusion_Matrix.png      — normalised confusion matrix (Wk=10, Cond. C)
  Fig3_AllModels_Comparison.png  — horizontal bar chart, all nine models
  Fig4_Noise_Robustness.png      — dual panel: accuracy and FAR/MDR vs noise

Data sources:
  Fig 1 : reads 01-05 CSV files from Paper1_Revision_Results/
  Fig 2 : runs ONE seed of LOCO (Wk=10, Cond. C) to get actual predictions
  Fig 3 : reads 07_all_models.csv
  Fig 4 : reads 08_noise_robustness.csv

Runtime: Fig 2 requires ~20 min (single LOO pass). Figs 1, 3, 4 are instant.

Run:  python paper1_generate_figures.py
================================================================================
"""

import sys, time, warnings, os
warnings.filterwarnings("ignore")
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import confusion_matrix, accuracy_score, \
                                classification_report
    HAS_SK = True
except ImportError:
    HAS_SK = False
    print("[WARNING] scikit-learn not found — Fig 2 (confusion matrix) cannot be drawn")

try:
    from catboost import CatBoostClassifier
    HAS_CB = True
except ImportError:
    HAS_CB = False
    print("[WARNING] CatBoost not found — Fig 2 (confusion matrix) cannot be drawn")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
_ROOTS = [os.environ.get("LEAK_DATA_ROOT", ""), "data", "."]  # folder holding Dynamic_pressure/
DATA_ROOT  = next((Path(p) for p in _ROOTS if p and (Path(p) / "Dynamic_pressure").exists()), Path("data"))
S1_DIR     = DATA_ROOT / "Dynamic_pressure" / "Sensor 1"
S2_DIR     = DATA_ROOT / "Dynamic_pressure" / "Sensor 2"
RESULTS    = DATA_ROOT / "Paper1_Revision_Results"
FIG_DIR    = DATA_ROOT / "Paper1_Figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 72)
print("  FIGURE GENERATION — Paper 1 (PSEP Submission)")
print("=" * 72)
print(f"  Results  : {RESULTS}")
print(f"  Output   : {FIG_DIR}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# STYLE — consistent across all four figures
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "legend.fontsize":  9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "figure.dpi":       300,
})

# Paper colour palette
COL = {
    "A":   "#888780",   # gray  — published/within-condition
    "B":   "#BA7517",   # amber — LOO + VSL/VSG
    "C":   "#1D9E75",   # teal  — primary corrected (LOO, no VSL/VSG)
    "D":   "#534AB7",   # purple— time-based, no VSL/VSG
    "C9":  "#993C1D",   # coral — 9-condition sensitivity
}

WK   = [6, 8, 10]
WLBL = ["$W_k$=6\n(3.0 s)", "$W_k$=8\n(4.0 s)", "$W_k$=10\n(5.0 s)"]
CLASS_NAMES = ["No Leak\n(C0)", "Single-point\n(C1)", "Two-point/mult.\n(C2)"]

# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — Protocol comparison across window sizes
# ─────────────────────────────────────────────────────────────────────────────
print("  Generating Figure 1 — Protocol comparison ...")

def need(name):
    """Path to a required result file; stop if it is missing."""
    p = RESULTS / name
    if not p.exists():
        sys.exit(f"ERROR: {p} not found. Run the experiment scripts first.")
    return p

def load_csv_condition(name):
    """Return (acc_mean_list, acc_std_list) in % for Wk=6,8,10."""
    df = pd.read_csv(need(name))
    means, stds = [], []
    for wk in WK:
        row = df[df.Wk == wk]
        if row.empty:
            means.append(np.nan); stds.append(np.nan)
        else:
            means.append(float(row.iloc[0]["accuracy_mean"]) * 100)
            stds.append(float(row.iloc[0]["accuracy_std"]) * 100)
    return means, stds

# All values are read from the result files written by the experiment scripts.
_FIG1_SPEC = [
    ("A",  "01_condition_A.csv",                   "A (time-based + $V_{SL}$/$V_{SG}$)"),
    ("B",  "02_condition_B.csv",                   "B (LOCO + $V_{SL}$/$V_{SG}$)"),
    ("C",  "03_condition_C_dual_9fold_PRIMARY.csv", "C ★ (LOCO, no $V_{SL}$/$V_{SG}$, 15 cond.)"),
    ("D",  "04_condition_D.csv",                   "D (time-based, no $V_{SL}$/$V_{SG}$)"),
    ("C9", "03_condition_C_9fold_sensitivity.csv",  "C-9 (LOCO, 9 cond., Alquennah design)"),
]
DATA_FIG1 = {}
for key, fname, lbl in _FIG1_SPEC:
    m, s = load_csv_condition(fname)
    DATA_FIG1[key] = {"means": m, "stds": s, "col": COL[key], "label": lbl}

x = np.arange(len(WK))
fig, ax = plt.subplots(figsize=(6.5, 4.2))

markers = ["o", "s", "^", "D", "v"]
for (key, d), mk in zip(DATA_FIG1.items(), markers):
    means = np.array(d["means"])
    stds  = np.array(d["stds"])
    lbl   = d.get("label", key)
    ax.errorbar(x, means, yerr=stds, label=lbl,
                color=d["col"], marker=mk, markersize=6,
                capsize=4, linewidth=1.8, capthick=1.5)

ax.set_xticks(x)
ax.set_xticklabels(WLBL)
ax.set_xlabel("Window size $W_k$ (duration at 2 Hz)")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(35, 100)
ax.axhline(100, color="#BBBBBB", lw=0.8, ls=":", label="Published claim (100 %)")
ax.legend(loc="upper left", framealpha=0.9, fontsize=8.5)
ax.set_title("Accuracy across evaluation protocols and window sizes\n"
             "(10 seeds; error bars: ±1 std, training instability)")
fig.tight_layout()
p1 = FIG_DIR / "Fig1_Protocol_Comparison.png"
fig.savefig(p1, dpi=300, bbox_inches="tight")
plt.close(fig)
_a10 = DATA_FIG1['A']['means'][2]
print(f"    Saved: {p1.name}  ✓  Condition A Wk=10 = {_a10:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — Confusion matrix (one LOCO pass, seed 42)
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Generating Figure 2 — Confusion matrix ...")

def try_compute_confusion_matrix():
    """Attempt to compute the actual confusion matrix from a single LOO pass."""
    if not HAS_SK or not HAS_CB:
        return None
    if not S1_DIR.exists():
        return None

    print("    Running single-seed LOCO (Wk=10, Cond C, seed 42) ...")

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

    FILES = [("no leak.xlsx",0),("One leak.xlsx",1),("three leaks.xlsx",2)]
    vsl_seq = sorted({_try_float(v)
                      for v in pd.read_excel(S1_DIR/"no leak.xlsx",
                                             header=None,dtype=object).iloc[1]
                      if not np.isnan(_try_float(v))})

    def parse_file(fp, sensor, lbl, vsl_seq):
        raw  = pd.read_excel(Path(fp), header=None, dtype=object)
        data = raw.iloc[3:].reset_index(drop=True)
        last_vsg, vsg_f = np.nan, []
        for v in raw.iloc[0]:
            f = _try_float(v)
            if not np.isnan(f): last_vsg = f
            vsg_f.append(last_vsg)
        vsl_row = [_try_float(v) for v in raw.iloc[1]]
        cols = {}
        for c in range(1, data.shape[1]):
            vsg = vsg_f[c] if c < len(vsg_f) else np.nan
            if np.isnan(vsg): continue
            arr = pd.to_numeric(data.iloc[:,c], errors="coerce").values
            if int(np.sum(~np.isnan(arr))) == 0: continue
            vsl = vsl_row[c] if c < len(vsl_row) else np.nan
            cols[c] = {"vsg": round(vsg,3), "vsl": vsl, "arr": arr}
        vgrp = defaultdict(list)
        for c, info in cols.items(): vgrp[info["vsg"]].append(c)
        for vg, grp in vgrp.items():
            labeled = {c: cols[c]["vsl"] for c in grp
                       if not np.isnan(cols[c]["vsl"])}
            unlabeled = [c for c in grp if np.isnan(cols[c]["vsl"])]
            if not unlabeled: continue
            missing = sorted(v for v in vsl_seq
                             if not any(abs(v-lv)<0.005 for lv in labeled.values()))
            if len(missing)==len(unlabeled):
                for c, vval in zip(sorted(unlabeled), missing):
                    cols[c]["vsl"] = vval
        times = pd.to_numeric(data.iloc[:,0], errors="coerce").values
        rows = []
        for c, info in cols.items():
            if np.isnan(info["vsl"]): continue
            for t, p in zip(times, info["arr"]):
                if not np.isnan(p):
                    rows.append({"Time":float(t) if not np.isnan(t) else 0.,
                                 "P":float(p),"VSG":info["vsg"],
                                 "VSL":info["vsl"],"Label":lbl,"Sensor":sensor})
        return pd.DataFrame(rows)

    f1s=[parse_file(S1_DIR/f,"S1",l,vsl_seq) for f,l in FILES
         if (S1_DIR/f).exists()]
    f2s=[parse_file(S2_DIR/f,"S2",l,vsl_seq) for f,l in FILES
         if (S2_DIR/f).exists()]
    df_s1=pd.concat(f1s,ignore_index=True)
    df_s2=pd.concat(f2s,ignore_index=True) if f2s else pd.DataFrame()

    df_s1w=df_s1.rename(columns={"P":"P1"}).drop(columns=["Sensor"])
    df_s1w=df_s1w.sort_values(["VSG","VSL","Label","Time"])
    df_s1w["_i"]=df_s1w.groupby(["VSG","VSL","Label"]).cumcount()
    if not df_s2.empty:
        df_s2w=df_s2.rename(columns={"P":"P2"}).drop(columns=["Sensor"])
        df_s2w=df_s2w.sort_values(["VSG","VSL","Label","Time"])
        df_s2w["_i"]=df_s2w.groupby(["VSG","VSL","Label"]).cumcount()
        df=df_s1w.merge(df_s2w[["VSG","VSL","Label","P2","_i"]],
                        on=["VSG","VSL","Label","_i"],how="left").drop(columns=["_i"])
    else:
        df=df_s1w.drop(columns=["_i"]); df["P2"]=np.nan

    pool=sorted(df.groupby(["VSG","VSL"]).groups.keys())

    def ext(arr,pfx):
        a=arr.astype(float); n=len(a)
        sl=np.polyfit(np.arange(n,dtype=float),a,1)[0] if n>1 else 0.
        md=float(np.mean(np.abs(np.diff(a)))) if n>1 else 0.
        return {f"{pfx}_mean":float(np.mean(a)),f"{pfx}_min":float(np.min(a)),
                f"{pfx}_max":float(np.max(a)),f"{pfx}_std":float(np.std(a)),
                f"{pfx}_rms":float(np.sqrt(np.mean(a**2))),
                f"{pfx}_range":float(np.ptp(a)),f"{pfx}_slope":sl,f"{pfx}_mad":md}

    def bw(df,Wk,pool):
        rows,labels=[],[]
        for lbl in [0,1,2]:
            for (vsg,vsl) in pool:
                seg=df[(df.Label==lbl)&np.isclose(df.VSG,vsg,atol=0.005)
                       &np.isclose(df.VSL,vsl,atol=0.01)].sort_values("Time")
                p1=seg["P1"].values.astype(float)
                p2=(seg["P2"].values.astype(float)
                    if "P2" in seg.columns else None)
                if len(p1)<Wk: continue
                for s in range(len(p1)-Wk+1):
                    feat=ext(p1[s:s+Wk],"P1")
                    if p2 is not None and len(p2)>=len(p1):
                        w2=p2[s:s+Wk]
                        if not np.any(np.isnan(w2)): feat.update(ext(w2,"P2"))
                    rows.append(feat); labels.append(lbl)
        if not rows: return np.empty((0,0)),np.empty(0,dtype=int)
        X=pd.DataFrame(rows).fillna(0).replace([np.inf,-np.inf],0)
        return X.values.astype(float),np.array(labels,dtype=int)

    class MC4:
        def __init__(self,seed=42):
            kw={"task_type":"CPU","thread_count":-1,"verbose":0}
            self.s1=CatBoostClassifier(300,depth=6,learning_rate=0.05,
                                        random_seed=seed,**kw)
            self.s2=CatBoostClassifier(300,depth=6,learning_rate=0.05,
                                        random_seed=seed,**kw)
        def fit(self,X,y):
            self.s1.fit(X,(y>0).astype(int))
            m=y>0
            if m.sum()>1: self.s2.fit(X[m],y[m])
            self.has2=m.sum()>1; return self
        def predict(self,X):
            out=np.zeros(len(X),dtype=int)
            m=self.s1.predict(X).astype(int).flatten()==1
            if m.sum()>0 and self.has2:
                out[m]=self.s2.predict(X[m]).astype(int).flatten()
            return out

    all_t, all_p = [], []
    for tc in pool:
        tr=[c for c in pool if c!=tc]
        Xtr,ytr=bw(df,10,tr); Xte,yte=bw(df,10,[tc])
        if Xtr.shape[0]==0 or Xte.shape[0]==0: continue
        sc=StandardScaler()
        Xtrs=sc.fit_transform(np.nan_to_num(Xtr))
        Xtes=sc.transform(np.nan_to_num(Xte))
        m=MC4(42); m.fit(Xtrs,ytr)
        all_t.extend(yte.tolist()); all_p.extend(m.predict(Xtes).tolist())
        print(".", end="", flush=True)
    print()

    acc=accuracy_score(all_t,all_p)
    cm=confusion_matrix(all_t,all_p,labels=[0,1,2])
    cm_norm=(cm.astype(float)/cm.sum(axis=1,keepdims=True))
    print(f"    Single-seed (seed 42) acc={acc*100:.1f}%")
    return cm_norm

# The matrix is computed from one LOCO pass (seed 42). There is no fallback:
# without CatBoost, scikit-learn and the raw data the script stops.
cm_norm = try_compute_confusion_matrix()
if cm_norm is None:
    sys.exit("ERROR: confusion matrix needs CatBoost, scikit-learn and the raw data "
             "under LEAK_DATA_ROOT/Dynamic_pressure.")
print(f"    Confusion matrix computed ✓")

# Ten-seed summary for the annotation, read from the primary result file
_c = pd.read_csv(need("03_condition_C_dual_9fold_PRIMARY.csv"))
_c = _c[_c.Wk == 10].iloc[0]

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(5.0, 4.2))

# Custom teal colormap
teal_cmap = LinearSegmentedColormap.from_list(
    "teal_cm", ["#FFFFFF", "#1D9E75"], N=256)
im = ax.imshow(cm_norm, cmap=teal_cmap, vmin=0, vmax=1, aspect="auto")

# Annotate cells
for i in range(3):
    for j in range(3):
        val = cm_norm[i, j]
        col = "white" if val > 0.55 else "#1a1a1a"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=11, color=col, fontweight="bold")

ax.set_xticks([0,1,2]); ax.set_yticks([0,1,2])
ax.set_xticklabels(CLASS_NAMES, fontsize=9)
ax.set_yticklabels(CLASS_NAMES, fontsize=9)
ax.set_xlabel("Predicted class")
ax.set_ylabel("True class")
ax.set_title(f"Normalised confusion matrix — Condition C, $W_k$=10\n"
             f"(LOCO, no $V_{{SL}}$/$V_{{SG}}$, 15 conditions, seed 42)")
plt.colorbar(im, ax=ax, label="Proportion", shrink=0.9)

# Overlay ten-seed accuracy / FAR / MDR
ax.text(1.0, -0.18,
        f"10-seed mean: Accuracy {_c.accuracy_mean*100:.1f} %   "
        f"FAR {_c.FAR_mean*100:.1f} %   MDR {_c.MDR_mean*100:.1f} %\n"
        f"Macro-F1 {_c.macro_f1_mean*100:.1f} %",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8.5, color="#444444",
        bbox=dict(boxstyle="round,pad=0.3", fc="#F5F5F0", ec="#CCCCCC"))
fig.tight_layout(rect=[0, 0.05, 1, 1])
p2 = FIG_DIR / "Fig2_Confusion_Matrix.png"
fig.savefig(p2, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"    Saved: {p2.name}  ✓")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — All-models comparison (horizontal bar chart)
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Generating Figure 3 — All models comparison ...")

MODEL_LABELS = {
    "MS5_CatBoost": "MS5  Single-stage CatBoost",
    "MC4_Cascade":  "MC4  Cascade (primary ★)",
    "NN_MLP":       "NN   MLP neural network",
    "MS2_SVM":      "MS2  SVM",
    "MS6_MaxVote":  "MS6  Max-vote ensemble",
    "MS7_Stacking": "MS7  Stacking ensemble",
    "MS4_RF":       "MS4  Random forest",
    "MS3_DT":       "MS3  Decision tree",
    "MS1_kNN":      "MS1  $k$-NN",
}

def _pm(s):
    """Parse '61.6%±0.7%' into (61.6, 0.7)."""
    a, b = str(s).replace("%", "").split("±")
    return float(a), float(b)

_am = pd.read_csv(need("07_all_models.csv"))
MODELS_DATA = []   # (label, acc_mean, acc_std, highlight), best model first
for _, r in _am.iterrows():
    a, s = _pm(r["Accuracy"])
    MODELS_DATA.append((MODEL_LABELS.get(r["Model"], r["Model"]), a, s,
                        r["Model"].startswith("MC4")))
MODELS_DATA.sort(key=lambda m: -m[1])

labels   = [m[0] for m in MODELS_DATA]
accs     = np.array([m[1] for m in MODELS_DATA])
stds     = np.array([m[2] for m in MODELS_DATA])
highl    = [m[3] for m in MODELS_DATA]
mc4_acc  = next(m[1] for m in MODELS_DATA if m[3])
colors   = [COL["C"] if h else "#AAAAAA" for h in highl]

fig, ax = plt.subplots(figsize=(7.0, 4.5))
y = np.arange(len(labels))

bars = ax.barh(y, accs, xerr=stds, color=colors, edgecolor="white",
               height=0.7, capsize=4, error_kw={"elinewidth": 1.2})

# Value labels
for i, (a, s) in enumerate(zip(accs, stds)):
    ax.text(a + s + 0.8, i, f"{a:.1f}%", va="center", fontsize=8.5)

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Accuracy (%)")
ax.set_xlim(25, 92)
ax.axvline(mc4_acc, color=COL["C"], lw=1.2, ls="--", alpha=0.6)
ax.set_title("All classifiers under condition-blocked LOCO evaluation\n"
             "($W_k$=10, no $V_{SL}$/$V_{SG}$, POOL$_{{15}}$, 10 seeds; "
             "error bars: ±1 std)")

# Range annotation
lo, hi = accs.min(), accs.max()
ytop = len(labels) - 0.55
ax.annotate("", xy=(hi, ytop), xytext=(lo, ytop),
            arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.2))
ax.text((lo + hi) / 2, ytop + 0.1, f"{lo:.1f} – {hi:.1f} %",
        ha="center", fontsize=8, color="#555555")

legend_els = [
    mpatches.Patch(color=COL["C"], label="MC4 cascade (primary)"),
    mpatches.Patch(color="#AAAAAA", label="Comparison models"),
]
ax.legend(handles=legend_els, loc="center right", fontsize=8.5)
fig.tight_layout()
p3 = FIG_DIR / "Fig3_AllModels_Comparison.png"
fig.savefig(p3, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"    Saved: {p3.name}  ✓")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 — Noise robustness (dual panel)
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Generating Figure 4 — Noise robustness ...")

_nz = pd.read_csv(need("08_noise_robustness.csv")).sort_values("Noise_pct_FSO")
NOISE = {
    "pct":  _nz.Noise_pct_FSO.tolist(),
    "acc":  (_nz.Acc_mean * 100).tolist(),
    "astd": (_nz.Acc_std * 100).tolist(),
    "far":  (_nz.FAR_mean * 100).tolist(),
    "fstd": (_nz.FAR_std * 100).tolist(),
    "mdr":  (_nz.MDR_mean * 100).tolist(),
    "mstd": (_nz.MDR_std * 100).tolist(),
}
pct = NOISE["pct"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.8), sharey=False)

# Left panel: accuracy
ax1.errorbar(pct, NOISE["acc"], yerr=NOISE["astd"],
             color=COL["C"], marker="o", capsize=4, linewidth=2,
             label="Accuracy")
ax1.set_xlabel("Noise level (% FSO)")
ax1.set_ylabel("Accuracy (%)")
ax1.set_ylim(50, 70)
ax1.set_title("Accuracy vs. noise level")
ax1.annotate("Performance floor\nset by LOCO, not noise",
             xy=(pct[-2], NOISE["acc"][-2]), xytext=(0.55, 64.5),
             fontsize=8, color="#444444",
             arrowprops=dict(arrowstyle="->", color="#444444", lw=0.9))

# Right panel: FAR and MDR
ax2.errorbar(pct, NOISE["far"], yerr=NOISE["fstd"],
             color="#993C1D", marker="s", capsize=4, linewidth=2,
             label="FAR (false alarm)")
ax2.errorbar(pct, NOISE["mdr"], yerr=NOISE["mstd"],
             color=COL["B"], marker="^", capsize=4, linewidth=2,
             label="MDR (missed detection)")
ax2.set_xlabel("Noise level (% FSO)")
ax2.set_ylabel("Rate (%)")
ax2.set_ylim(0, 45)
ax2.set_title("FAR and MDR vs. noise level")
ax2.legend(fontsize=8.5, loc="center right")
ax2.annotate("FAR decreases\nwith noise", xy=(pct[-1], NOISE["far"][-1]),
             xytext=(0.5, 38), fontsize=8, color="#993C1D",
             arrowprops=dict(arrowstyle="->", color="#993C1D", lw=0.9))

for ax in [ax1, ax2]:
    ax.set_xticks(pct)
    ax.set_xticklabels(["0.0", "0.1", "0.2", "0.5", "1.0"])

fig.suptitle("Noise robustness — symmetric contamination of training and test\n"
             "(MC4 cascade, $W_k$=10, LOCO, POOL$_{15}$, 10 seeds)",
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.92])
p4 = FIG_DIR / "Fig4_Noise_Robustness.png"
fig.savefig(p4, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"    Saved: {p4.name}  ✓  0% FSO: {NOISE['acc'][0]:.1f}% acc, {NOISE['far'][0]:.1f}% FAR")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("  FIGURES COMPLETE — upload these four files to your paper:")
print("=" * 72)
for p in [p1, p2, p3, p4]:
    sz = Path(p).stat().st_size / 1024
    print(f"  {p.name:<45} {sz:>6.0f} KB")
print()
print("  LaTeX inclusion (copy into your .tex file):")
for p, lbl in [(p1,"fig:fig1"), (p2,"fig:fig2"), (p3,"fig:fig3"), (p4,"fig:fig4")]:
    name = Path(p).stem
    print(f'  \\includegraphics[width=\\linewidth]{{figures/{name}}}  % \\label{{{lbl}}}')
print("=" * 72)
