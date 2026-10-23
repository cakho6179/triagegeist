# TRIAGEGEIST: predicting the 5-level Emergency Severity Index on NHAMCS data

Code, intermediate results and figures for the paper *TRIAGEGEIST: Predicting 5-Level Emergency Severity Index Using Ordinal Regression on NHAMCS Data* (ICAITech 2026, paper ID 1571320920).

The model predicts the ESI level (1 = most urgent, 5 = least urgent) from information available at triage. ESI is treated as an ordered scale: an XGBoost regressor with logistic loss produces a risk score in [0, 1], and four ordered thresholds, tuned on Quadratic Weighted Kappa (QWK), turn the score into a level.

This is a research prototype. It is not a clinical tool and has not been validated outside NHAMCS.

## Results (R 4.6.0, seed 42, single model)

Random stratified 80/20 split of 58,124 visits (46,499 train, 11,625 test).

| Experiment | Features | QWK | Exact acc. | Within ±1 | MAE |
|---|---|---|---|---|---|
| E1 raw tabular | 70 | 0.4326 | 0.5618 | 0.9351 | 0.5892 |
| E2 + missing flags | 70 | 0.4328 | 0.5591 | 0.9203 | 0.6231 |
| E3 + clinical scores | 114 | 0.4778 | 0.5819 | 0.9462 | 0.5261 |
| **E4 + chief-complaint text (deployable)** | 640 | **0.5340** (95% CI 0.5201–0.5484) | 0.6158 | 0.9524 | 0.5045 |
| E5mc multiclass XGBoost, same features | 640 | 0.4736 | 0.6227 | 0.9628 | 0.4175 |
| E6 all features (upper bound, uses a post-hoc IV-fluids flag) | 744 | 0.5484 (95% CI 0.5346–0.5619) | 0.5573 | 0.9568 | 0.4911 |

Temporal proxy (train 2018–2020, test 2021–2022): QWK 0.5122.

Notes for reading the table:

- Multiclass (E5mc) is higher on exact accuracy, within ±1 and MAE. Ordinal E4 is higher on QWK, the metric the thresholds are tuned for. Read all four columns.
- The accuracy of E4 here is for rounded scores; with the QWK-optimised thresholds it is lower (about 55%). E6 uses the thresholded decoding.
- NHAMCS is a complex survey. The analysis does not use survey weights.
- Cramér's V and wait-time summaries describe association with the recorded ESI labels, not model error by subgroup.

## Repository layout

```
paper/                             camera-ready version of the paper (PDF)
scripts/                           raw data -> 744-feature table (Python)
src/                               feature-engineering modules used by scripts/
R/                                 modelling, ablation, SHAP, bias and wait-time analysis (R)
pipeline/                          Python cross-check, bootstrap, figures
```

Data and generated outputs are not stored here.

## Reproduce

Run everything from the repository root.

**1. Build the feature table.** About 30 seconds. Rebuilds the processed files from the raw CSV; the output was checked column by column against the file used in the experiments.

```bash
pip install -r requirements.txt
python scripts/run_preprocess_enhanced.py   # writes data/processed/*_enhanced.*
python scripts/make_triage_full.py          # writes data/processed/triage_full.csv (744 features + triage_acuity)
```

**2. Run the experiments (R ≥ 4.6; packages `data.table`, `xgboost`, see `requirements_R.txt`).**

```bash
Rscript R/reproduce_E123.R        # E1–E3, about 3 min
Rscript R/reproduce_E4.R          # E4 (deployable), about 2 min
Rscript R/reproduce_E5mc.R        # multiclass baseline, about 3 min
Rscript R/reproduce_E6_ablation.R # E6 and seven ablations, about 8 min
Rscript R/reproduce_temporal.R    # temporal proxy, about 3 min
```

Results are written to `revision/results/r_repro/` (created on first run).

**3. Cross-check and figures (Python).**

```bash
python pipeline/04_retrain.py     # independent Python retrain, about 8 min
python pipeline/03_figures.py     # paper figures; also reads saved outputs of the original analysis runs in outputs/, which are not distributed
```

Further analyses (SHAP, bias, wait times) are in `R/compute_shap.R`, `R/bias_analysis.R` and `R/wait_time_analysis.R`. `pipeline/05`–`07` are sensitivity checks (seeds, cross-validation, ensemble) and are not the reported numbers.

## Data

NHAMCS is published by the CDC National Center for Health Statistics: <https://www.cdc.gov/nchs/ahcd/index.htm>. The experiments use a 49-variable extract of NHAMCS 2018–2022 (58,124 visits), expected at `data/raw/nhamcs_data_2018_22.csv`. The extract and the processed feature table are not included in this repository.

## Citation

If you use this code, please cite the paper (reference to be updated once the proceedings are published).

## License

MIT, see `LICENSE`. The NHAMCS data remain subject to the CDC terms of use.
