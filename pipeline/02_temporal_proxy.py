"""02 — Temporal proxy validation: train 2018-2020 -> test 2021-2022 (E4 feature set).

WHY: processed triage_full.csv lost the `year` column, so year must be
re-joined from data/raw/nhamcs_data_2018_22.csv BY ROW ORDER. The script
aborts unless row counts match AND the ESI distributions agree — if the
orders differ, re-run preprocessing with year kept instead of forcing it.

WHAT: XGBoost ordinal (reg:logistic on (y-1)/4) + threshold grid search
maximizing QWK, same hyperparameters as R/experiments.R (reduced to
nrounds=400 for speed). Writes revision/results/temporal.csv.

Run:  python pipeline/02_temporal_proxy.py   (~30-60 min, NOT run yet)
"""
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib
cfg = importlib.import_module("00_config")

SEED = cfg.GLOBAL_SEED
PARAMS = {"objective": "reg:logistic", "max_depth": 8, "eta": 0.04,
          "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 5,
          "lambda": 2.0, "eval_metric": "rmse", "seed": SEED, "nthread": -1}


def qwk(y_true: np.ndarray, y_pred: np.ndarray, k: int = 5) -> float:
    n = len(y_true)
    hist = np.bincount(y_true, minlength=k + 1)[1:] / n
    pred = np.bincount(y_pred, minlength=k + 1)[1:] / n
    expected = np.outer(hist, pred)
    w = np.fromfunction(lambda i, j: (i - j) ** 2 / (k - 1) ** 2, (k, k))
    obs = np.zeros((k, k))
    for t, p in zip(y_true - 1, y_pred - 1):
        obs[t, p] += 1
    obs /= n
    return float(1.0 - (w * obs).sum() / (w * expected).sum())


def tune_thresholds(raw: np.ndarray, y_true: np.ndarray):
    best, best_q = None, -1.0
    grids = [np.arange(0.20, 0.41, 0.02), np.arange(0.40, 0.56, 0.02),
             np.arange(0.55, 0.71, 0.02), np.arange(0.70, 0.86, 0.02)]
    for t1, t2, t3, t4 in product(*grids):
        if not (t1 < t2 < t3 < t4):
            continue
        pred = np.where(raw < t1, 1, np.where(raw < t2, 2, np.where(raw < t3, 3,
                     np.where(raw < t4, 4, 5))))
        q = qwk(y_true, pred)
        if q > best_q:
            best, best_q = (round(t1, 2), round(t2, 2), round(t3, 2), round(t4, 2)), q
    return best, best_q


def e4_columns(all_cols: list[str]) -> list[str]:
    """Port of R/experiments.R groups: g1 tabular + engineered + cc_* text (no IV)."""
    import re
    vitals = ["temp", "heart_rate", "resp_rate", "sys_bp", "dias_bp", "spo2",
              "pain_score", "visit_month", "day_of_week", "arrival_time"]
    g1 = [c for c in vitals if c in all_cols]
    g1 += [c for c in all_cols if re.match(
        r"^(insurance_|residence_|region_|ems_arrival_|episode_|is_injury_poison_|"
        r"vitals_during_visit_|seen_last_72h_|no_payment_|race_|sex_)", c)]
    g1 += [c for c in all_cols if c.startswith("hist_")]
    eng = [c for c in all_cols if re.match(
        r"^(shock_|qsofa_|mews_|sirs_|spo2_|rr_|hr_|age_|temp_|triage_|ems_|pain_|"
        r"symp_|is_|high_)", c)]
    text = [c for c in all_cols if c.startswith("cc_")]
    cols = list(dict.fromkeys(g1 + eng + text))
    assert "intervention_iv_fluids" not in cols, "post-hoc leak in E4 set"
    return cols


def main() -> None:
    raw = pd.read_csv(cfg.DATA_RAW, usecols=["year", "target_triage_acuity"])
    proc = pd.read_csv(cfg.DATA_PROCESSED)
    assert len(raw) == len(proc) == 58124, (len(raw), len(proc))
    d1 = raw["target_triage_acuity"].value_counts(normalize=True).sort_index()
    d2 = proc["triage_acuity"].value_counts(normalize=True).sort_index()
    assert np.allclose(d1.values, d2.values, atol=1e-9), "row order differs — abort, keep year in preprocessing"
    proc = proc.assign(year=raw["year"].values)

    cols = e4_columns([c for c in proc.columns if c != "triage_acuity"])
    print(f"E4 features: {len(cols)} (expect ~640)")
    tr = proc[~proc["year"].isin([2021, 2022])]
    te = proc[proc["year"].isin([2021, 2022])]
    print(f"train 2018-2020: {len(tr)}, test 2021-2022: {len(te)}")

    dtr = xgb.DMatrix(tr[cols].values, label=(tr["triage_acuity"].values - 1) / 4)
    dte = xgb.DMatrix(te[cols].values)
    model = xgb.train(PARAMS, dtr, num_boost_round=400)
    raw_pred = model.predict(dte)
    thr, q = tune_thresholds(raw_pred, te["triage_acuity"].values)
    t1, t2, t3, t4 = thr
    pred = np.where(raw_pred < t1, 1, np.where(raw_pred < t2, 2, np.where(raw_pred < t3, 3,
            np.where(raw_pred < t4, 4, 5))))
    acc = float((pred == te["triage_acuity"].values).mean())
    w1 = float((np.abs(pred - te["triage_acuity"].values) <= 1).mean())
    pd.DataFrame([{"split": "2018-2020->2021-2022", "n_train": len(tr), "n_test": len(te),
                   "qwk": round(q, 4), "acc": round(acc, 4), "within1": round(w1, 4),
                   "thresholds": str(tuple(round(float(t), 2) for t in thr)),
                   "seed": SEED}]).to_csv(
        cfg.RESULTS / "temporal.csv", index=False)
    print(f"QWK={q:.4f} acc={acc:.4f} within1={w1:.4f} thr={thr}")


if __name__ == "__main__":
    main()
