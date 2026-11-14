#!/usr/bin/env python3
"""
================================================================================
S2 INFERENCE CHECK — Paper 1 Revision
================================================================================
Answers exactly one question before any experiment runs:

  Are the unlabelled columns in S2 three leaks.xlsx
  (the ones the parser infers as VSL=1.34 and VSL=1.94)
  genuine independent recordings, or duplicates/noise?

Checks performed:
  1. Print the raw pressure values for EVERY column in S2 three leaks.xlsx
     so you can see whether the unlabelled columns look like distinct data.

  2. Compute the correlation between each unlabelled column and every
     labelled column in the same VSG group.
     Real data: correlation < 0.99 with all neighbours.
     Duplicated data: correlation ≈ 1.00 with one neighbour.

  3. Compare the statistical distribution (mean, std, min, max) of each
     column against its neighbours.
     Real data: different distributions for different VSL.
     Noise/empty: near-zero std or identical distributions.

  4. Print a VERDICT for each inferred column: REAL or DUPLICATE or EMPTY.

  5. Print the final recommended setting for USE_INFERRED_S2 in the
     experiment script.

Output files:
  S2_CHECK_01_raw_values.csv         — first 10 rows of every column
  S2_CHECK_02_statistics.csv         — mean/std/min/max per column
  S2_CHECK_03_correlations.csv       — pairwise correlations within each VSG group
  S2_CHECK_04_verdict.txt            — REAL / DUPLICATE / EMPTY per column + recommendation
================================================================================
"""

import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
import os
_ROOTS = [os.environ.get("LEAK_DATA_ROOT", ""), "data", "."]  # folder holding Dynamic_pressure/
DATA_ROOT  = next((Path(p) for p in _ROOTS if p and (Path(p) / "Dynamic_pressure").exists()), Path("data"))
S2_DIR    = DATA_ROOT / "Dynamic_pressure" / "Sensor 2"
TARGET    = S2_DIR / "three leaks.xlsx"
OUTPUT    = DATA_ROOT / "Paper1_S2_Check"
OUTPUT.mkdir(parents=True, exist_ok=True)

SEP  = "=" * 80
SEP2 = "-" * 80

print(SEP)
print("  S2 INFERENCE CHECK — three leaks.xlsx")
print(SEP)
print(f"  Target file : {TARGET}")
print(f"  Exists      : {TARGET.exists()}")
print(f"  Output      : {OUTPUT}")
print()

if not TARGET.exists():
    sys.exit(f"ERROR: {TARGET} not found. Check _ROOTS at top of script.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Read the raw file exactly as-is
# ─────────────────────────────────────────────────────────────────────────────
print(SEP2)
print("STEP 1: RAW FILE STRUCTURE")
print(SEP2)

raw = pd.read_excel(TARGET, header=None, dtype=object)
print(f"  File shape: {raw.shape[0]} rows × {raw.shape[1]} columns")
print(f"  Row 0 (VSG labels): {list(raw.iloc[0])}")
print(f"  Row 1 (VSL labels): {list(raw.iloc[1])}")
print(f"  Row 2 (role labels): {list(raw.iloc[2])}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Parse every data column with its label status
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP2)
print("STEP 2: COLUMN-BY-COLUMN INVENTORY")
print(SEP2)
print(f"  {'Col':>4}  {'VSG':>7}  {'VSL label':>12}  {'Role':>14}  "
      f"{'N non-NaN':>10}  {'Mean':>10}  {'Std':>10}  {'Min':>10}  {'Max':>10}  Status")
print(f"  {'-'*4}  {'-'*7}  {'-'*12}  {'-'*14}  "
      f"{'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  ------")

# Forward-fill VSG row
vsg_row = list(raw.iloc[0])
last_vsg = None
vsg_filled = []
for v in vsg_row:
    try:
        f = float(str(v).split("=")[-1].strip().split()[0])
        if 0 < f < 100:
            last_vsg = round(f, 3)
    except Exception:
        pass
    vsg_filled.append(last_vsg)

col_records = []
data_rows   = raw.iloc[3:].reset_index(drop=True)

for col in range(raw.shape[1]):
    vsg = vsg_filled[col]
    if vsg is None:
        continue   # no VSG → time column or outer separator

    raw_vsl  = raw.iloc[1, col]
    raw_role = raw.iloc[2, col]

    # Parse VSL
    try:
        vsl_val = float(str(raw_vsl).split("=")[-1].strip().split()[0])
        vsl_label = round(vsl_val, 2)
        vsl_explicit = True
    except Exception:
        vsl_label    = None
        vsl_explicit = False

    # Parse role
    try:
        role = str(raw_role).strip() if pd.notna(raw_role) else "NaN"
    except Exception:
        role = "NaN"

    # Parse data
    arr = pd.to_numeric(data_rows.iloc[:, col], errors="coerce").dropna().values
    n   = len(arr)

    if n == 0:
        status = "EMPTY"
        mean = std = mn = mx = float("nan")
    else:
        mean = float(np.mean(arr))
        std  = float(np.std(arr))
        mn   = float(np.min(arr))
        mx   = float(np.max(arr))
        status = "labelled" if vsl_explicit else "INFERRED"

    print(f"  {col:>4}  {vsg:>7.3f}  {str(vsl_label):>12}  {role[:14]:>14}  "
          f"{n:>10}  {mean:>10.4f}  {std:>10.4f}  {mn:>10.4f}  {mx:>10.4f}  {status}")

    col_records.append({
        "col":          col,
        "VSG":          vsg,
        "VSL_label":    vsl_label,
        "VSL_explicit": vsl_explicit,
        "role":         role,
        "n_valid":      n,
        "mean":         mean,
        "std":          std,
        "min":          mn,
        "max":          mx,
        "status":       status,
    })

df_cols = pd.DataFrame(col_records)
df_cols.to_csv(OUTPUT / "S2_CHECK_02_statistics.csv", index=False)
print(f"\n  Written: S2_CHECK_02_statistics.csv")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — First 10 raw values of every non-empty data column
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP2)
print("STEP 3: FIRST 10 PRESSURE VALUES PER COLUMN (scan for duplicates visually)")
print(SEP2)

raw_rows = []
for _, r in df_cols.iterrows():
    if r["n_valid"] == 0:
        continue
    col  = int(r["col"])
    arr  = pd.to_numeric(data_rows.iloc[:, col], errors="coerce").dropna().values
    tag  = f"VSG={r['VSG']:.3f} VSL={r['VSL_label']} [{r['status']}]"
    vals = arr[:10]
    print(f"  {tag:45s}  {' '.join(f'{v:.6f}' for v in vals)}")
    raw_rows.append({"col": col, "tag": tag,
                     **{f"row{i+1}": float(arr[i]) if i < len(arr) else None
                        for i in range(10)}})

pd.DataFrame(raw_rows).to_csv(OUTPUT / "S2_CHECK_01_raw_values.csv", index=False)
print(f"\n  Written: S2_CHECK_01_raw_values.csv")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Pairwise Pearson correlation within each VSG group
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP2)
print("STEP 4: PAIRWISE CORRELATIONS WITHIN EACH VSG GROUP")
print("  Duplicated data → correlation ≈ 1.00")
print("  Genuine data    → correlation < 0.99 (different VSL should look different)")
print(SEP2)

corr_rows  = []
verdicts   = {}   # col_id → verdict string

for vsg_val in sorted(df_cols["VSG"].dropna().unique()):
    grp = df_cols[df_cols["VSG"] == vsg_val].copy()
    grp_data_cols = grp[grp["n_valid"] > 0]["col"].tolist()

    if len(grp_data_cols) < 2:
        continue

    # Build array: rows=observations, cols=VSL columns
    arrays = {}
    for col in grp_data_cols:
        arr = pd.to_numeric(data_rows.iloc[:, col], errors="coerce").dropna().values
        arrays[col] = arr

    # Correlations
    print(f"\n  VSG = {vsg_val:.3f}:")
    for i, c1 in enumerate(grp_data_cols):
        for c2 in grp_data_cols[i+1:]:
            a1 = arrays[c1]
            a2 = arrays[c2]
            n  = min(len(a1), len(a2))
            if n < 5:
                corr = float("nan")
            else:
                corr = float(np.corrcoef(a1[:n], a2[:n])[0, 1])

            r1 = grp[grp["col"] == c1].iloc[0]
            r2 = grp[grp["col"] == c2].iloc[0]
            tag1 = f"VSL={r1['VSL_label']}[{r1['status'][:3]}]"
            tag2 = f"VSL={r2['VSL_label']}[{r2['status'][:3]}]"
            flag = " ← DUPLICATE?" if corr > 0.999 else ""
            print(f"    {tag1:25s} vs {tag2:25s}  r = {corr:+.6f}{flag}")
            corr_rows.append({
                "VSG": vsg_val,
                "col1": c1, "VSL1": r1["VSL_label"], "status1": r1["status"],
                "col2": c2, "VSL2": r2["VSL_label"], "status2": r2["status"],
                "correlation": corr,
                "flag": "POSSIBLE_DUPLICATE" if corr > 0.999 else "OK",
            })

df_corr = pd.DataFrame(corr_rows)
df_corr.to_csv(OUTPUT / "S2_CHECK_03_correlations.csv", index=False)
print(f"\n  Written: S2_CHECK_03_correlations.csv")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — VERDICT per inferred column
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("  STEP 5: VERDICT PER INFERRED COLUMN")
print(SEP)

inferred = df_cols[df_cols["status"] == "INFERRED"]
verdict_lines = []
any_duplicate = False
any_empty     = False

for _, r in inferred.iterrows():
    col = int(r["col"])
    vsg = r["VSG"]
    vsl = r["VSL_label"]

    if r["n_valid"] == 0:
        verdict = "EMPTY"
        reason  = "No numeric data found in this column."
        any_empty = True
    else:
        # Check if any neighbouring column is nearly identical
        neighbours = df_corr[
            ((df_corr["col1"] == col) | (df_corr["col2"] == col)) &
            (df_corr["VSG"] == vsg) &
            (df_corr["flag"] == "POSSIBLE_DUPLICATE")
        ]
        if not neighbours.empty:
            # Find which neighbour
            dup_vsls = []
            for _, nr in neighbours.iterrows():
                other_vsl = nr["VSL2"] if nr["col1"] == col else nr["VSL1"]
                dup_vsls.append(other_vsl)
            verdict = "DUPLICATE"
            reason  = (f"Correlation > 0.999 with VSL={dup_vsls}. "
                       f"Data appears identical to a labelled column.")
            any_duplicate = True
        else:
            verdict = "REAL"
            reason  = (f"Data is non-empty (n={r['n_valid']}) and not "
                       f"correlated > 0.999 with any labelled neighbour. "
                       f"Likely a genuine independent recording.")

    line = (f"  Col {col:2d}  VSG={vsg:.3f}  VSL={vsl}  → {verdict:10s}  {reason}")
    print(line)
    verdict_lines.append(line)

# Final recommendation
print()
print(SEP2)
if any_duplicate:
    recommendation = (
        "RECOMMENDATION: SET  USE_INFERRED_S2 = False\n"
        "One or more inferred columns are duplicates of labelled columns.\n"
        "Restrict POOL_DUAL to explicitly-labelled conditions only (9 conditions).\n"
        "This matches Alquennah's nine-condition design and avoids data contamination."
    )
elif any_empty:
    recommendation = (
        "RECOMMENDATION: SET  USE_INFERRED_S2 = False\n"
        "One or more inferred columns are empty (no data).\n"
        "Restrict POOL_DUAL to explicitly-labelled conditions only (9 conditions)."
    )
else:
    recommendation = (
        "RECOMMENDATION: USE_INFERRED_S2 = True  (already the default)\n"
        "All inferred columns appear to contain genuine independent recordings.\n"
        "However: CONFIRM WITH YOUR PHYSICAL LOGBOOK that three-leaks experiments\n"
        "were conducted at VSL=1.34 and VSL=1.94 with sensor S2 before proceeding.\n"
        "If confirmed, POOL_DUAL has 15 conditions and the paper description of\n"
        "'nine operating conditions' needs to be updated."
    )

print(recommendation)

# Write verdict file
verdict_txt = (
    "S2 INFERENCE CHECK — VERDICT\n"
    "=" * 60 + "\n\n"
    "File checked: " + str(TARGET) + "\n\n"
    "Per-column verdicts:\n"
    + "\n".join(verdict_lines) + "\n\n"
    + "=" * 60 + "\n\n"
    + recommendation + "\n"
)
(OUTPUT / "S2_CHECK_04_verdict.txt").write_text(
    verdict_txt, encoding="utf-8")
print(f"\n  Written: S2_CHECK_04_verdict.txt")

print()
print(SEP)
print("  CHECK COMPLETE")
print(SEP)
print(f"  Share S2_CHECK_04_verdict.txt and S2_CHECK_03_correlations.csv")
print(f"  Output folder: {OUTPUT}")
print(SEP)
