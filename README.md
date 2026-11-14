# When 100% Accuracy Is a Warning Sign

Analysis code for **"Evaluating Near-Perfect Accuracy Claims in Machine-Learning-Based Leak Detection for Two-Phase Pipelines: A Condition-Blocked Audit"**.

Authors: I. N. Suleiman, S. Serpoosh, N. Elkholy, H. Ferroudji, M. A. Rahman, M. Hamilton.

**Paper:** arXiv link to be added.

The paper is the reference for every reported number. This repository holds only the code used to produce them. Running the scripts on the data writes the results and figures locally; none are stored here.

The study audits the pressure-based leak classifier of Alquennah et al. (2026), *IEEE Open Journal of Instrumentation and Measurement*, doi:[10.1109/OJIM.2026.3706192](https://doi.org/10.1109/OJIM.2026.3706192). It re-evaluates that classifier with condition-blocked leave-one-condition-out (LOCO) validation, in which each (VSG, VSL) flow condition is held out of training in turn. It also runs a 2 × 2 ablation of the split protocol against the VSL/VSG inputs, and an external check on the public GPLA-12 acoustic dataset.

## Contents

| Script | What it does |
|---|---|
| `scripts/paper1_data_diagnostic.py` | Lists the flow conditions and files present. Run it before any experiment. |
| `scripts/paper1_s2_inference_check.py` | Checks the six downstream-sensor (S2) columns that have no VSL label, using correlations and column means. |
| `scripts/paper1_revision_experiments.py` | Main pipeline: coverage audit, then conditions A, B, C, C-9 and D (the 2 × 2 ablation). |
| `scripts/paper1_resume.py` | Resumes an interrupted revision run from the files already on disk. |
| `scripts/paper1_allmodels_run.py` | Runs all nine classifiers under the condition-blocked protocol (10 seeds). |
| `scripts/paper1_noise_run.py` | Noise robustness, with the same noise added to training and test data. |
| `scripts/paper1_generate_figures.py` | Draws the figures from the result files. |
| `scripts/external_validation_gpla12.py` | External evaluation on GPLA-12. |

## Data

The two-phase dynamic-pressure recordings are **not included**. To request access, contact the corresponding author.

The scripts expect this layout:

```
<LEAK_DATA_ROOT>/
  Dynamic_pressure/
    Sensor 1/   no leak.xlsx, One leak.xlsx, three leaks.xlsx   (upstream, S1)
    Sensor 2/   no leak.xlsx, One leak.xlsx, three leaks.xlsx   (downstream, S2)
```

```bash
export LEAK_DATA_ROOT=/path/to/folder_containing_Dynamic_pressure   # PowerShell: $env:LEAK_DATA_ROOT = "..."
```

Results are written to `<LEAK_DATA_ROOT>/Paper1_Revision_Results/` and figures to `<LEAK_DATA_ROOT>/Paper1_Figures/`.

GPLA-12 is public (Li & Yao, 2021, [arXiv:2106.10277](https://arxiv.org/abs/2106.10277)). Set `GPLA12_CSV` to the path of `gpla12_acoustic_signals.csv`.

## Running

```bash
pip install -r requirements.txt
python scripts/paper1_data_diagnostic.py          # check the data first
python scripts/paper1_revision_experiments.py     # conditions A-D, 2x2 table
python scripts/paper1_allmodels_run.py            # nine classifiers
python scripts/paper1_noise_run.py                # noise robustness
python scripts/paper1_generate_figures.py         # figures
python scripts/external_validation_gpla12.py      # GPLA-12
```

On Windows, set `PYTHONIOENCODING=utf-8` so the progress output prints. The full pipeline takes several hours on a CPU.

Run the experiment scripts before `paper1_generate_figures.py`. It reads every plotted value from the result files they write, and stops if a file is missing. The confusion matrix comes from one LOCO pass with seed 42.

`requirements.txt` pins the tested versions (Python 3.11.3). CatBoost runs on the GPU when PyTorch finds CUDA. GPU and CPU runs can differ slightly in the last digit.

## Citation

See `CITATION.cff`, or use "Cite this repository" on GitHub.

## License

MIT, see `LICENSE`.
