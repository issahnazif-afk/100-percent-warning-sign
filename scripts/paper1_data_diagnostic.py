#!/usr/bin/env python3
"""
================================================================================
DATA DIAGNOSTIC — Paper 1 Revision
================================================================================
Purpose: Establish EXACTLY what data exists before running any experiments.

This script makes NO assumptions about:
  - How many VSL or VSG values exist
  - Which conditions have S2 coverage
  - What the file names are
  - How many observations per condition

It reads everything from disk, prints it verbatim, and produces a
coverage grid you can verify against the physical experimental logbook.

Run this BEFORE running paper1_revision_experiments.py.
Share the full terminal output and the CSV files it produces.

Output files (written to OUTPUT_DIR):
  DIAG_01_directory_listing.txt     — every file found in S1 and S2 dirs
  DIAG_02_file_headers.txt          — raw first 5 rows of every xlsx file
  DIAG_03_coverage_grid.csv         — (VSG, VSL, Class, Sensor) × n_obs
  DIAG_04_vsg_vsl_found.csv         — unique VSG and VSL values per file
  DIAG_05_summary.txt               — plain-English summary for the paper
================================================================================
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# PATHS — edit _ROOTS to match your machine
# ─────────────────────────────────────────────────────────────────────────────
_ROOTS = [os.environ.get("LEAK_DATA_ROOT", ""), "data", "."]  # folder holding Dynamic_pressure/
DATA_ROOT  = next((Path(p) for p in _ROOTS if p and (Path(p) / "Dynamic_pressure").exists()), Path("data"))
S1_DIR     = DATA_ROOT / "Dynamic_pressure" / "Sensor 1"
S2_DIR     = DATA_ROOT / "Dynamic_pressure" / "Sensor 2"
OUTPUT_DIR = DATA_ROOT / "Paper1_Diagnostic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEP  = "=" * 80
SEP2 = "─" * 80

print(SEP)
print("  DATA DIAGNOSTIC — Paper 1 Revision")
print(SEP)
print(f"  Data root : {DATA_ROOT}   exists={DATA_ROOT.exists()}")
print(f"  S1 dir    : {S1_DIR}   exists={S1_DIR.exists()}")
print(f"  S2 dir    : {S2_DIR}   exists={S2_DIR.exists()}")
print(f"  Output    : {OUTPUT_DIR}")
print()

if not DATA_ROOT.exists():
    sys.exit(f"ERROR: Data root not found. Edit _ROOTS in the script.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Directory listing: every file in S1 and S2
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  STEP 1: DIRECTORY LISTING")
print(SEP)

dir_lines = []

for sensor_name, sensor_dir in [("Sensor 1 (S1)", S1_DIR),
                                  ("Sensor 2 (S2)", S2_DIR)]:
    print(f"\n  {sensor_name}: {sensor_dir}")
    dir_lines.append(f"\n{sensor_name}: {sensor_dir}")

    if not sensor_dir.exists():
        msg = f"    *** DIRECTORY DOES NOT EXIST ***"
        print(msg)
        dir_lines.append(msg)
        continue

    all_files = sorted(sensor_dir.iterdir())
    if not all_files:
        msg = "    *** DIRECTORY IS EMPTY ***"
        print(msg)
        dir_lines.append(msg)
        continue

    for f in all_files:
        size_kb = f.stat().st_size / 1024 if f.is_file() else 0
        line = f"    [{f.suffix.upper():6s}]  {f.name:40s}  {size_kb:8.1f} KB"
        print(line)
        dir_lines.append(line)

(OUTPUT_DIR / "DIAG_01_directory_listing.txt").write_text(
    "\n".join(dir_lines))
print(f"\n  Written: DIAG_01_directory_listing.txt")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Raw file headers: first 5 rows of every xlsx, exactly as they are
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + SEP)
print("  STEP 2: RAW FILE HEADERS (first 5 rows of every xlsx)")
print(SEP)
print("  This shows the ACTUAL structure — VSG row, VSL row, column layout.")
print()

header_lines = []

def read_header(filepath):
    """Return first 5 rows as a DataFrame without any parsing."""
    try:
        raw = pd.read_excel(Path(filepath), header=None, dtype=object,
                            nrows=5)
        return raw, None
    except Exception as e:
        return None, str(e)

xlsx_files = {}  # sensor_name → list of Path
for sensor_name, sensor_dir in [("S1", S1_DIR), ("S2", S2_DIR)]:
    if not sensor_dir.exists():
        xlsx_files[sensor_name] = []
        continue
    xlsx_files[sensor_name] = sorted(sensor_dir.glob("*.xlsx"))

for sensor_name, files in xlsx_files.items():
    header_lines.append(f"\n{'='*60}\n{sensor_name}\n{'='*60}")
    print(f"  {sensor_name}:")
    if not files:
        header_lines.append("  No xlsx files found.")
        print("    No xlsx files found.")
        continue
    for fp in files:
        raw, err = read_header(fp)
        line_head = f"\n  File: {fp.name}"
        print(line_head)
        header_lines.append(line_head)
        if err:
            msg = f"    ERROR reading: {err}"
            print(msg)
            header_lines.append(msg)
            continue
        n_cols = raw.shape[1]
        line_shape = f"    Shape of full file (nrows×ncols): reading below…"
        line_cols  = f"    Columns visible in header: {n_cols}"
        print(line_cols)
        header_lines.append(line_cols)
        # Print row 0 (VSG) and row 1 (VSL) verbatim — these are the critical rows
        for row_idx in range(min(5, raw.shape[0])):
            row_label = {0: "Row 0 (VSG?)",
                         1: "Row 1 (VSL?)",
                         2: "Row 2 (role?)",
                         3: "Row 3 (data?)",
                         4: "Row 4 (data?)"}.get(row_idx, f"Row {row_idx}")
            vals = list(raw.iloc[row_idx])
            # Show first 20 cells to avoid truncation
            shown = vals[:20]
            shown_str = "  |  ".join(
                str(v)[:12] if pd.notna(v) else "NaN"
                for v in shown
            )
            line = f"    {row_label:15s}: {shown_str}"
            if len(vals) > 20:
                line += f"  … ({len(vals)} cols total)"
            print(line)
            header_lines.append(line)

(OUTPUT_DIR / "DIAG_02_file_headers.txt").write_text(
    "\n".join(header_lines))
print(f"\n  Written: DIAG_02_file_headers.txt")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Parse every xlsx file and extract (VSG, VSL) → n_observations
#           Uses the SAME parser as the experiment script (robust, no assumptions)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + SEP)
print("  STEP 3: COVERAGE GRID — observations per (Sensor, File, VSG, VSL)")
print(SEP)
print("  Parsing every xlsx file from scratch, no assumptions about values.")
print()

def _try_float(v):
    """Parse a cell value to float if possible."""
    if isinstance(v, str):
        # Handle "VSG = 0.063 m/s" or "VSL = 1.03 m/s" style labels
        for kw in ["VSG", "VSL", "="]:
            if kw in v:
                parts = v.replace("=", " ").split()
                for p in parts:
                    try:
                        f = float(p)
                        if 0 < f < 100:  # plausible VSL/VSG range
                            return f
                    except ValueError:
                        pass
                return np.nan
        try:
            return float(v)
        except ValueError:
            return np.nan
    try:
        f = float(v)
        return f if not (isinstance(f, float) and np.isnan(f)) else np.nan
    except (TypeError, ValueError):
        return np.nan

def parse_file_coverage(filepath):
    """
    Parse one xlsx file and return a list of dicts:
      {"VSG": float, "VSL": float, "n_obs": int}
    One entry per data column.
    Makes NO assumptions about VSG/VSL values.
    """
    raw = pd.read_excel(Path(filepath), header=None, dtype=object)
    if raw.shape[0] < 4:
        return []

    # Row 0: VSG — forward-fill so every column in a group inherits its VSG
    vsg_row = raw.iloc[0].copy()
    # Forward fill non-NaN numeric values
    last_vsg = np.nan
    vsg_filled = []
    for v in vsg_row:
        fv = _try_float(v)
        if not np.isnan(fv):
            last_vsg = fv
        vsg_filled.append(last_vsg)

    # Row 1: VSL — may have NaN for some columns
    vsl_row = [_try_float(v) for v in raw.iloc[1]]

    # Data starts at row 3
    data = raw.iloc[3:].reset_index(drop=True)

    results = []
    for col in range(1, raw.shape[1]):
        vsg = vsg_filled[col] if col < len(vsg_filled) else np.nan
        vsl = vsl_row[col]    if col < len(vsl_row)    else np.nan

        if np.isnan(vsg):
            continue  # separator column with no VSG group

        # Count numeric (non-NaN) observations in this column
        arr = pd.to_numeric(data.iloc[:, col], errors="coerce")
        n_valid = int(arr.notna().sum())

        if n_valid == 0:
            continue  # empty separator

        results.append({
            "VSG":   round(vsg, 4),
            "VSL":   round(vsl, 4) if not np.isnan(vsl) else np.nan,
            "n_obs": n_valid,
        })

    return results

# Map file names to class labels (0=no leak, 1=one leak, 2=three leaks)
# This mapping is derived from the file names found, not hardcoded
def infer_class_label(filename):
    fn = filename.lower()
    if "no" in fn and "leak" in fn:
        return 0
    if "one" in fn or "1" in fn or "single" in fn:
        return 1
    if "three" in fn or "3" in fn or "multiple" in fn or "two" in fn:
        return 2
    return -1  # unknown

coverage_records = []
vsg_vsl_records  = []

for sensor_name, files in xlsx_files.items():
    print(f"  {sensor_name}:")
    for fp in files:
        class_lbl  = infer_class_label(fp.name)
        class_name = {0: "No Leak (C0)", 1: "Single Leak (C1)",
                      2: "Three Leaks (C2)", -1: "UNKNOWN"}[class_lbl]
        print(f"    {fp.name:40s}  → class={class_lbl} ({class_name})")

        cols = parse_file_coverage(fp)
        if not cols:
            print(f"      *** No data columns parsed ***")
            continue

        # Collect unique VSG and VSL values from this file
        vsgs = sorted({r["VSG"] for r in cols if not np.isnan(r["VSG"])})
        vsls = sorted({r["VSL"] for r in cols if not np.isnan(r["VSL"])})
        print(f"      VSG values found: {vsgs}")
        print(f"      VSL values found: {vsls}")
        print(f"      Data columns    : {len(cols)}")

        for r in cols:
            coverage_records.append({
                "Sensor":     sensor_name,
                "File":       fp.name,
                "Class":      class_lbl,
                "Class_name": class_name,
                "VSG":        r["VSG"],
                "VSL":        r["VSL"],
                "n_obs":      r["n_obs"],
            })
            vsg_vsl_records.append({
                "Sensor": sensor_name,
                "File":   fp.name,
                "VSG":    r["VSG"],
                "VSL":    r["VSL"],
            })

df_cov  = pd.DataFrame(coverage_records)
df_vgsl = pd.DataFrame(vsg_vsl_records).drop_duplicates()

if df_cov.empty:
    print("\n  *** NO DATA PARSED — check that xlsx files exist and are readable ***")
    sys.exit(1)

df_cov.to_csv(OUTPUT_DIR / "DIAG_03_coverage_grid.csv", index=False)
df_vgsl.to_csv(OUTPUT_DIR / "DIAG_04_vsg_vsl_found.csv", index=False)
print(f"\n  Written: DIAG_03_coverage_grid.csv")
print(f"  Written: DIAG_04_vsg_vsl_found.csv")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Pivot table: the coverage matrix every reviewer will ask about
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + SEP)
print("  STEP 4: COVERAGE MATRIX")
print(SEP)

# Unique values found — these drive the paper description
all_vsgs   = sorted(df_cov["VSG"].dropna().unique())
all_vsls   = sorted(df_cov["VSL"].dropna().unique())
all_classes = sorted(df_cov["Class"].dropna().unique())
all_sensors = sorted(df_cov["Sensor"].unique())

print(f"\n  VSG values found  : {all_vsgs}  → {len(all_vsgs)} values")
print(f"  VSL values found  : {all_vsls}  → {len(all_vsls)} values")
print(f"  Leak classes found: {all_classes}  → {len(all_classes)} classes")
print(f"  Sensors found     : {all_sensors}")
print(f"  Nominal conditions: {len(all_vsgs)} × {len(all_vsls)} = "
      f"{len(all_vsgs)*len(all_vsls)} (VSG × VSL combinations)")

# Build a readable pivot: rows=(VSG,VSL), columns=(Sensor,Class), values=n_obs
if not df_cov.empty:
    pivot = df_cov.pivot_table(
        index=["VSG", "VSL"],
        columns=["Sensor", "Class_name"],
        values="n_obs",
        aggfunc="sum",
        fill_value=0,
    )
    print("\n  OBSERVATIONS PER CONDITION × SENSOR × CLASS")
    print("  (0 = no data for that combination)")
    print()
    print(pivot.to_string())
    pivot.to_csv(OUTPUT_DIR / "DIAG_03b_pivot.csv")
    print(f"\n  Written: DIAG_03b_pivot.csv")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Which conditions are usable for which analyses
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + SEP)
print("  STEP 5: USABILITY SUMMARY — what each condition can support")
print(SEP)

usability = []
for vsg in all_vsgs:
    for vsl in all_vsls:
        cond_df = df_cov[
            np.isclose(df_cov.VSG, vsg, atol=0.005) &
            np.isclose(df_cov.VSL, vsl, atol=0.01)
        ]
        if cond_df.empty:
            usability.append({
                "VSG": vsg, "VSL": vsl,
                "S1_C0": False, "S1_C1": False, "S1_C2": False,
                "S2_C0": False, "S2_C1": False, "S2_C2": False,
                "S1_full": False, "S2_full": False, "Dual_full": False,
                "Note": "NO DATA",
            })
            continue

        s1 = cond_df[cond_df.Sensor == "S1"]
        s2 = cond_df[cond_df.Sensor == "S2"]

        def has_class(df_sub, cls):
            return (df_sub[df_sub.Class == cls]["n_obs"].sum() > 0)

        s1_c0 = has_class(s1, 0)
        s1_c1 = has_class(s1, 1)
        s1_c2 = has_class(s1, 2)
        s2_c0 = has_class(s2, 0)
        s2_c1 = has_class(s2, 1)
        s2_c2 = has_class(s2, 2)

        s1_full   = s1_c0 and s1_c1 and s1_c2
        s2_full   = s2_c0 and s2_c1 and s2_c2
        dual_full = s1_full and s2_full

        if dual_full:
            note = "P1 + P2  (dual-sensor analysis eligible)"
        elif s1_full:
            note = "P1 only  (S1-only analysis eligible)"
        elif s1_c0 or s1_c1 or s1_c2:
            missing = [c for c, v in [(0,s1_c0),(1,s1_c1),(2,s1_c2)] if not v]
            note = f"INCOMPLETE S1: missing class {missing}"
        else:
            note = "NO S1 DATA"

        usability.append({
            "VSG": vsg, "VSL": vsl,
            "S1_C0": s1_c0, "S1_C1": s1_c1, "S1_C2": s1_c2,
            "S2_C0": s2_c0, "S2_C1": s2_c1, "S2_C2": s2_c2,
            "S1_full": s1_full, "S2_full": s2_full, "Dual_full": dual_full,
            "Note": note,
        })

df_use = pd.DataFrame(usability)
df_use.to_csv(OUTPUT_DIR / "DIAG_03c_usability.csv", index=False)

print()
print(df_use[["VSG","VSL","S1_full","S2_full","Dual_full","Note"]].to_string(index=False))

pool_s1   = [(r.VSG, r.VSL) for _, r in df_use.iterrows() if r.S1_full]
pool_dual = [(r.VSG, r.VSL) for _, r in df_use.iterrows() if r.Dual_full]
incomplete = [(r.VSG, r.VSL) for _, r in df_use.iterrows()
              if not r.S1_full and r.Note != "NO DATA"]

print(f"\n  POOL_S1  (S1-only, all classes present)  : {len(pool_s1)} conditions")
for c in pool_s1:
    print(f"    VSG={c[0]:.3f}  VSL={c[1]:.2f}")

print(f"\n  POOL_DUAL (S1+S2, all classes for both)  : {len(pool_dual)} conditions")
for c in pool_dual:
    print(f"    VSG={c[0]:.3f}  VSL={c[1]:.2f}")

if incomplete:
    print(f"\n  INCOMPLETE conditions (missing some classes): {len(incomplete)}")
    for r_idx, r in df_use[df_use.Note.str.startswith("INCOMPLETE")].iterrows():
        print(f"    VSG={r.VSG:.3f}  VSL={r.VSL:.2f}  {r.Note}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Plain-English summary for the paper
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + SEP)
print("  STEP 6: PLAIN-ENGLISH SUMMARY")
print(SEP)

summary_lines = [
    "DATA DIAGNOSTIC SUMMARY",
    "=" * 60,
    f"Data root       : {DATA_ROOT}",
    f"S1 directory    : {S1_DIR}  (exists={S1_DIR.exists()})",
    f"S2 directory    : {S2_DIR}  (exists={S2_DIR.exists()})",
    "",
    "FILES FOUND",
    "-" * 40,
]
for sensor_name, files in xlsx_files.items():
    summary_lines.append(f"  {sensor_name}: {len(files)} xlsx file(s)")
    for fp in files:
        lbl = infer_class_label(fp.name)
        lbl_name = {0:"C0 no-leak",1:"C1 single-leak",2:"C2 three-leaks",-1:"UNKNOWN"}[lbl]
        summary_lines.append(f"    {fp.name}  → {lbl_name}")

summary_lines += [
    "",
    "CONDITION SPACE",
    "-" * 40,
    f"  VSG values    : {all_vsgs}",
    f"  VSL values    : {all_vsls}",
    f"  Nominal grid  : {len(all_vsgs)} VSG × {len(all_vsls)} VSL "
    f"= {len(all_vsgs)*len(all_vsls)} conditions",
    "",
    "ANALYSIS POOLS",
    "-" * 40,
    f"  POOL_S1  (S1-only, for primary LOO) : {len(pool_s1)} conditions",
    f"  POOL_DUAL (dual-sensor, S1+S2)      : {len(pool_dual)} conditions",
    "",
    "VERDICT — verify these numbers against your experimental logbook",
    "-" * 40,
]
if len(pool_s1) == 9:
    summary_lines.append("  ✓ 9 conditions with full S1 coverage — matches paper description")
elif len(pool_s1) == 15:
    summary_lines.append("  ⚠  15 conditions with full S1 coverage — paper says 9: UPDATE NEEDED")
else:
    summary_lines.append(f"  ⚠  {len(pool_s1)} conditions with S1 — verify against logbook")

if len(pool_dual) > 0:
    summary_lines.append(f"  ✓ {len(pool_dual)} conditions with dual-sensor coverage")
else:
    summary_lines.append("  ✗ No conditions with full dual-sensor coverage found")

summary_txt = "\n".join(summary_lines)
(OUTPUT_DIR / "DIAG_05_summary.txt").write_text(summary_txt)
print("\n" + summary_txt)
print(f"\n  Written: DIAG_05_summary.txt")

print("\n" + SEP)
print("  DIAGNOSTIC COMPLETE")
print(SEP)
print(f"\n  Share ALL files in: {OUTPUT_DIR}")
print("  Most important to share first:")
print("    DIAG_02_file_headers.txt   — shows raw Excel structure")
print("    DIAG_03b_pivot.csv         — coverage matrix")
print("    DIAG_03c_usability.csv     — which conditions are usable")
print("    DIAG_05_summary.txt        — plain-English summary")
print()
print("  DO NOT run paper1_revision_experiments.py until")
print("  you have verified the diagnostic output matches your logbook.")
print(SEP)
