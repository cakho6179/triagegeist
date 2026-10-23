# TRIAGEGEIST: predicting the 5-level Emergency Severity Index on NHAMCS data

Code for the paper *TRIAGEGEIST: Predicting 5-Level Emergency Severity Index Using Ordinal Regression on NHAMCS Data* (ICAITech 2026, paper ID 1571320920). The camera-ready paper is in `paper/`.

The model predicts the ESI level (1 = most urgent, 5 = least urgent) from information available at triage. ESI is treated as an ordered scale: an XGBoost regressor with logistic loss produces a risk score in [0, 1], and four ordered thresholds, tuned on Quadratic Weighted Kappa (QWK), turn the score into a level.

This is a research prototype. It is not a clinical tool and has not been validated outside NHAMCS.

## Results (R 4.6.0, seed 42, single model)

Random 80/20 split of 58,124 visits (46,499 train, 11,625 test). The split comes from `sample(nrow, 0.8 * nrow)` with seed 42; it is not stratified. QWK thresholds are fitted on the test-set predictions (see the notes). Intervals are 95% bootstrap CIs of the test confusion matrix (2,000 resamples).

| Model | Features | QWK (95% CI) | Exact acc. | Within ±1 | MAE |
|---|---|---|---|---|---|
| E1 raw tabular | 70 | 0.4326 (0.4174–0.4478) | 0.4819 | 0.9351 | 0.5892 |
| E2 + missingness flags | 78 | 0.4325 (0.4169–0.4477) | 0.4967 | 0.9407 | 0.5677 |
| E3 + clinical scores | 114 | 0.4778 (0.4626–0.4928) | 0.5322 | 0.9462 | 0.5261 |
| **E4 + chief-complaint text (deployable)** | 640 | **0.5340** (0.5202–0.5478) | 0.5486 | 0.9524 | 0.5045 |
| Multiclass XGBoost, argmax | 640 | 0.4736 (0.4572–0.4893) | 0.6227 | 0.9628 | 0.4175 |
| Multiclass XGBoost, expected class + thresholds | 640 | 0.5372 (0.5235–0.5501) | 0.5665 | 0.9537 | 0.4849 |
| CORN (MLP) | 640 | 0.4904 (0.4764–0.5052) | 0.6163 | 0.9645 | 0.4216 |
| CORAL (MLP) | 640 | 0.4139 (0.3988–0.4300) | 0.5825 | 0.9608 | 0.4587 |
| MiniLM embeddings instead of TF-IDF | 498 | 0.5450 (0.5306–0.5592) | 0.5585 | 0.9530 | 0.4936 |
| Random Forest (500 trees) | 640 | 0.4250 (0.4094–0.4407) | 0.6096 | 0.9615 | 0.4317 |
| CART (depth 6) | 640 | 0.3418 (0.3257–0.3583) | 0.5702 | 0.9517 | 0.4809 |

Temporal proxy (train 2018–2020, test 2021–2022): QWK 0.5122.

Notes for reading the table:

- The gain of E4 over argmax multiclass comes from the decoding. When the multiclass probabilities are reduced to an expected-class score and cut with the same thresholds, QWK is 0.5372, on par with E4 (paired ΔQWK −0.003, 95% CI −0.011 to 0.004). Argmax multiclass has the higher exact accuracy and the higher share within ±1.
- Thresholds are fitted on the test predictions, as in the original runs. Fitting them on 5-fold out-of-fold training scores gives E4 QWK 0.5333, a difference of 0.0007.
- Missing values are median-imputed and the TF-IDF vocabulary is fitted on all rows, test rows included.
- NHAMCS is a complex survey. The analysis does not use survey weights.
- Cramér's V and wait-time summaries describe association with the recorded ESI labels, not model error by subgroup.
- Leave-one-group-out ablation, TreeSHAP importance and per-class results for E4 are written to `revision/results/final/` by the scripts below.

## Repository layout

```
paper/                             camera-ready version of the paper (PDF)
scripts/                           raw data -> 744-feature table (Python)
src/                               feature-engineering modules used by scripts/
R/                                 modelling, ablation, SHAP, bias and wait-time analysis (R)
R/lib_triage.R                     shared helpers for the R/final_*.R scripts
pipeline/                          neural baselines, metrics and bootstrap, figures, Python cross-check
```

Data and generated outputs are not stored here.

## Reproduce

Run everything from the repository root. The R scripts depend on the order of model fits (xgboost draws its seed from R's random number generator), so run each script as a whole and in the order below.

**1. Build the feature table.** Rebuilds the processed files from the raw CSV; the output was checked column by column against the file used in the experiments.

```bash
pip install -r requirements.txt
python scripts/run_preprocess_enhanced.py   # writes data/processed/*_enhanced.*
python scripts/make_triage_full.py          # writes data/processed/triage_full.csv (744 features + triage_acuity)
```

**2. Run the experiments (R ≥ 4.6; packages `data.table`, `xgboost`, `ranger`, `rpart`, see `requirements_R.txt`).** Every script writes to `revision/results/final/` (created on first run).

```bash
Rscript R/final_E123_predictions.R     # E1 and E3 test predictions (replays R/reproduce_E123.R)
Rscript R/final_E2_threshold_check.R   # E4 and E2 predictions, thresholds fitted on out-of-fold training scores
Rscript R/final_multiclass.R           # multiclass XGBoost, argmax and expected-class decoding
Rscript R/final_baselines_tree.R       # Random Forest and CART
Rscript R/final_ablation_E4.R          # leave-one-group-out ablation of E4
Rscript R/final_importance_E4.R        # gain importance and TreeSHAP for E4
```

The original single-model runs are still available: `R/reproduce_E123.R`, `R/reproduce_E4.R`, `R/reproduce_E5mc.R` and `R/reproduce_temporal.R` (results in `revision/results/r_repro/`).

**3. Neural and embedding baselines (Python, PyTorch and sentence-transformers).** The first run downloads the `all-MiniLM-L6-v2` weights from Hugging Face.

```bash
python pipeline/08_baselines_neural.py   # CORAL, CORN, MiniLM variant; uses the split written by step 2
```

**4. Metrics, intervals and figure.**

```bash
python pipeline/09_final_tables.py       # summary.csv, confusion and per-class tables, paired differences
python pipeline/10_progression_figure.py # QWK progression figure (revision/figures_final/)
python pipeline/03_figures.py            # confusion matrix and fairness/wait-time figures (needs outputs/ of the bias and wait-time scripts)
```

`R/compute_shap.R`, `R/bias_analysis.R` and `R/wait_time_analysis.R` hold the earlier SHAP, bias and wait-time analyses. `R/experiments.R` and `pipeline/04_retrain.py` are the earlier whole-pipeline scripts; they also fit an all-features reference (E6, with a post-hoc IV-fluids flag) that is not part of the reported results.

`pipeline/04_retrain.py`, `05_seed_ensemble.py` and `06_validate.py` are an independent Python retrain and sensitivity checks (seeds, 5-fold cross-validation). They use a stratified split, unlike the R runs, so their numbers differ slightly and are not the reported ones.

Setting `TRIAGE_SPLIT=stratified` makes the R `final_*` scripts and `pipeline/08` use a stratified 80/20 split and write to `revision/results/final_stratified/`. It is a sensitivity option; the reported results use the default random split.

## Data

NHAMCS is published by the CDC National Center for Health Statistics: <https://www.cdc.gov/nchs/ahcd/index.htm>. The experiments use a 49-variable extract of NHAMCS 2018–2022 (58,124 visits), expected at `data/raw/nhamcs_data_2018_22.csv`. The extract and the processed feature table are not included in this repository.

## Citation

If you use this code, please cite the paper (reference to be updated once the proceedings are published).

## License

MIT, see `LICENSE`. The NHAMCS data remain subject to the CDC terms of use.
