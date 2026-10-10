# Layer-wise Incremental Training of Variational Quantum Circuits to Mitigate Barren Plateaus

A college research project investigating whether **growing a variational
quantum circuit (VQC) one layer at a time** (Layer-wise Training) trains
more reliably than building the full-depth circuit and training all
parameters at once (Direct Training), on a real credit-risk
classification task, using the UCI Statlog German Credit dataset.

## 1. What this project does

1. Downloads and validates the real **UCI Statlog German Credit** dataset
   (1000 applicants, 20 features, no synthetic data).
2. Reduces the processed features to 4 / 6 / 8 dimensions with PCA
   (fit on the training split only) and encodes them as rotation angles
   for a VQC built in **PennyLane**.
3. Trains the same circuit architecture two ways:
   - **Direct Training**: all layers trained together from a random
     initialization.
   - **Layer-wise Training**: one layer at a time, newest layer only
     trainable, older layers frozen (verified by tests), under the same
     total optimizer-update budget.
4. Runs a noiseless comparison, a finite-measurement-shots comparison, and
   a simulated depolarizing-noise comparison.
5. Runs a separate circuit-only **gradient-variance (barren plateau)
   diagnostic**, independent of the dataset.
6. Compares both protocols against classical baselines (Logistic
   Regression, a small MLP).
7. Produces CSV tables, an Excel workbook, PNG/PDF figures, real circuit
   diagrams, and a written report — all generated from saved run data,
   never hand-typed.

## 2. Research question and hypotheses

**Primary question**: Does layer-wise incremental training improve
trainability, convergence, predictive performance, or noise robustness
of a VQC on credit-risk classification, compared with direct training?

This project does **not** assume layer-wise training wins. See
`reports/final_report.md` for the honest, evidence-based answer from the
runs actually executed (quick-mode by default — see Section 17).

## 3. Dataset

- Source: UCI Machine Learning Repository, *Statlog (German Credit
  Data)*, id 144.
  https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
- DOI: https://doi.org/10.24432/C5NC77
- 1000 instances, 20 attributes (13 categorical, 7 numerical), binary
  target (1 = good credit, 2 = bad credit; 700 / 300 split).
- Retrieved via the `ucimlrepo` package and cached locally at
  `data/raw/german_credit_raw.csv` so the project can be re-run offline.
- Official cost matrix (from the UCI documentation) is used for a
  cost-sensitive metric: misclassifying a bad-risk applicant as good
  costs 5, misclassifying a good-risk applicant as bad costs 1.
- Label convention used throughout the code: **good credit = -1,
  bad credit = +1** (bad credit is the positive class for
  precision/recall/F1/cost).

## 4. Circuit architecture

- Encoding: `RY(x_i)` on each of the `n_qubits` wires.
- Per layer: trainable `RY` then trainable `RZ` on every qubit, followed
  by a ring of fixed `CZ` gates between neighbouring qubits (for 2 qubits
  the ring collapses to one edge, not a duplicated edge).
- Readout: `expval(PauliZ(wire 0))`, an output in `[-1, 1]`.
- Trainable parameter count: `p = 2 * n_qubits * n_layers`.
- **Important caveat** (documented in `src/circuits.py`): because the CZ
  gates are always active, initializing a new layer's rotations near
  zero does **not** make that layer an identity operation. The project
  measures and reports the loss change each layer insertion actually
  causes rather than assuming it is negligible.

## 5. Direct vs. Layer-wise Training

| | Direct Training | Layer-wise Training |
|---|---|---|
| Circuit | Full `L`-layer circuit built up front | Starts at 1 layer, grows to `L` |
| Trainable params per stage | All `2*n_qubits*L` together | Only the newest layer; older layers frozen |
| Init (new layer) | Uniform `[0, 2*pi)` | Uniform `[0, 2*pi)` for layer 1; near-zero `[-0.01, 0.01]` for later layers |
| Update budget | Fixed total `N` updates | `N` split evenly across `L` stages |

Both protocols use the same data split, batch size, Adam optimizer,
learning rate, and MSE loss, so the comparison is apples-to-apples except
for the trained-parameter schedule.

## 6. Project layout

```
vqc_layerwise_credit_risk/
  README.md
  requirements.txt
  config.json              <- every experiment setting lives here
  run_project.py           <- single entry point (quick or full mode)
  notebooks/               <- interactive walkthrough (same src/ code)
  src/                     <- all tested implementation modules
  tests/                   <- pytest suite
  data/raw/                <- cached real dataset
  data/processed/          <- fitted-pipeline metadata
  results/quick/           <- quick-mode run-level CSVs, histories, manifest
  results/full/            <- full-mode results (separate location)
  tables/                  <- CSV tables 1-9 + Excel workbook
  figures/                 <- PNG/PDF figures + circuit diagrams
  reports/                 <- final_report.md
```

## 7. Supported environment

- **Python 3.13.7** (tested), Windows 11, CPU only. No GPU, no real
  quantum hardware, no cloud account needed.
- Exact package versions that were installed and smoke-tested in this
  project are pinned in `requirements.txt`.

## 8. Windows PowerShell setup (copy-paste)

```powershell
cd "C:\Users\tharu\Desktop\quantum project final\vqc_layerwise_credit_risk"

# Create and activate a virtual environment (Python 3.13.7)
py -3.13 -m venv venv
.\venv\Scripts\Activate.ps1

# Install the exact tested dependency versions
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `py -3.13` is not found, install Python 3.13 from python.org first, or
substitute whichever Python 3.11+ interpreter you have (see
"Troubleshooting" below).

## 9. Launching JupyterLab

```powershell
.\venv\Scripts\Activate.ps1
jupyter lab notebooks\vqc_layerwise_project.ipynb
```

## 10. Running the quick demonstration (recommended first run)

```powershell
.\venv\Scripts\Activate.ps1
python run_project.py --mode quick
```

This downloads/validates the dataset (first run only; cached afterwards),
preprocesses it, trains the classical baselines, runs the reduced
quick-mode grid (see `config.json -> experiment_grid.quick`), runs the
reduced shots/noise experiments, runs the gradient diagnostic, draws
circuit diagrams, and writes all tables/figures. **Measured time for this
project's actual, complete quick-mode run (including dataset download):
under 6 minutes** (355 seconds; `results/quick/run_summary.json` records
`total_wall_clock_s` for the exact run this report's numbers came from).
The noise sub-experiment was narrowed from its originally planned depths
`[2, 6]` to `[2]` only after a live pilot measurement showed depth-6
parameter-shift + density-matrix runs would take several minutes each
(`config.json -> noise_experiment.quick_subset_deviation_note` documents
this; `reports/final_report.md` Section 14 explains the measurement that
drove the decision).

## 11. Running the full experiment

```powershell
.\venv\Scripts\Activate.ps1
python run_project.py --mode full
```

The full grid (`config.json -> experiment_grid.full`: qubit counts
4/6/8, depths 2/6/12/24, both protocols, 10 seeds) is **far** more
expensive than quick mode. Based on this project's own pilot timing
(see `reports/final_report.md`, Section "Pilot run and feasibility"),
the full noiseless grid alone is estimated at several hours of CPU time;
the shots and noise sub-experiments add more. Plan to run it unattended,
ideally overnight.

## 12. Resuming an interrupted run

Just re-run the same command:

```powershell
python run_project.py --mode full
```

Every run is logged to `results/<mode>/*.csv` with a `status` column.
Rows marked `success` are never re-executed; only missing or `failed`
combinations are (re)attempted. Nothing is silently duplicated or
deleted.

## 13. Where things are saved

| What | Where |
|---|---|
| Raw cached dataset | `data/raw/german_credit_raw.csv` + `*_metadata.json` |
| Fitted preprocessing pipelines | `data/processed/pipeline_{n}q.joblib` |
| Run-level results (quick) | `results/quick/*.csv`, `results/quick/histories/*.json` |
| Run-level results (full) | `results/full/*.csv`, `results/full/histories/*.json` |
| Result manifest | `results/<mode>/result_manifest.json` |
| CSV tables (1-9) | `tables/*.csv` |
| Excel workbook | `tables/results_summary.xlsx` |
| Figures (PNG + PDF) | `figures/*.png`, `figures/*.pdf` |
| Circuit diagrams | `figures/circuit_diagram_*.png` |
| Final report | `reports/final_report.md` |

## 14. Troubleshooting

- **`ModuleNotFoundError: No module named 'src'`**: run commands from the
  project root (`vqc_layerwise_credit_risk/`), not from inside `src/`.
- **`ucimlrepo` fails to download**: check your internet connection, then
  retry `python -c "from ucimlrepo import fetch_ucirepo; fetch_ucirepo(id=144)"`.
  If it still fails, follow the recovery instructions printed by
  `src/data_retrieval.py` (manual download + placing the CSV at the
  documented cache path). The project will refuse to substitute another
  dataset.
- **PennyLane `argnum` vs `argnums` errors**: this project targets
  PennyLane 0.45.1, which uses `argnums`. If you installed a different
  PennyLane version with a different API, reinstall the pinned version
  from `requirements.txt`.
- **Garbled circuit-diagram text in the PowerShell console**: this is a
  Windows console code-page (cp1252) limitation when printing Unicode box
  characters, not a bug in the circuit; use the PNG/PDF figures or
  JupyterLab instead of printing `qml.draw(...)` output directly to the
  PowerShell console.
- **Full mode seems to hang**: it is not hanging; depths of 12-24 layers
  with `parameter-shift` (used for the shots/noise sub-experiments) are
  genuinely CPU-intensive. Check `results/<mode>/*.csv` for growing row
  counts, or watch CPU usage in Task Manager.

## 15. Reproducing the reported results

All seeds are fixed and recorded in `config.json`
(`data.split.split_seed`, `experiment_grid.*.seeds`, `random_seed_global`).
Re-running `python run_project.py --mode quick` from a clean clone with
the pinned `requirements.txt` reproduces the same dataset split, the same
PCA/angle-scaling parameters, and (subject to floating-point
determinism of the installed BLAS backend) the same training
trajectories, because every stochastic component (NumPy RNG, data split,
batch order, parameter initialization) is seeded.

## 16. What has and has not been tested in this environment

- Tested and executed in this environment: dataset retrieval and schema
  validation, the full preprocessing pipeline (all leakage checks),
  circuit construction/forward pass/gradients (exact, finite-shot via
  parameter-shift, and depolarizing-noise via `default.mixed`), Direct
  and Layer-wise training end to end, classical baselines, the gradient
  diagnostic, table/figure generation, and the full `pytest` suite.
- **Quick mode** (`--mode quick`) was executed in full for this
  deliverable; its real results are in `results/quick/` and are what
  `reports/final_report.md` discusses.
- **Full mode** (`--mode full`) was *configured* (see
  `config.json -> experiment_grid.full`) but, per the pilot timing
  estimate, was not exhaustively executed as part of this delivery. The
  report labels every number by which mode produced it and does not
  present full-mode conclusions that were not actually run.

## 17. Quick mode vs. full mode

`run_project.py --mode quick` and `--mode full` write to **separate**
directories (`results/quick/` vs `results/full/`) and are never merged.
`config.json -> quick_mode` mirrors whichever `--mode` flag was last used
by `run_project.py`, but the directory separation is what actually keeps
the two result sets apart.
