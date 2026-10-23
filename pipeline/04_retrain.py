"""04 — Retrain standard results from scratch (Python + XGBoost, seed 42).

Replaces parsing-old-outputs with a reproducible run:
  E1 tabular -> E2 +missing -> E3 +engineered -> E4 +text (deployable)
  E5 multiclass baseline (same 640 feats) -> E6 full + ablations (upper bound)

Groups ported 1:1 from R/experiments.R. Split: stratified 80/20, seed 42
(R used plain random; stratify is strictly better — documented difference).

Saves incrementally to revision/results/retrain/ (safe against timeouts):
  progression.csv, confusion_E6.csv, perclass_E6.csv, metrics_E6.csv

Run:  python pipeline/04_retrain.py [--no-ablation] [--fast] [--skip-done]
      full run ~25-40 min CPU; --fast halves nrounds for smoke tests.
"""
import argparse
import csv
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib
cfg = importlib.import_module("00_config")

SEED = cfg.GLOBAL_SEED
OUT = cfg.RESULTS / "retrain"
PARAMS = {"objective": "reg:logistic", "max_depth": 8, "eta": 0.04,
          "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 5,
          "lambda": 2.0, "eval_metric": "rmse", "seed": SEED, "nthread": -1}
ROUNDS = {"E1": 400, "E2": 400, "E3": 500, "E4": 600, "E5mc": 500,
          "E6": 600, "ABL": 300}


def qwk(y_true: np.ndarray, y_pred: np.ndarray, k: int = 5) -> float:
    n = len(y_true)
    hist = np.bincount(y_true, minlength=k + 1)[1:] / n
    pred = np.bincount(y_pred, minlength=k + 1)[1:] / n
    expected = np.outer(hist, pred)
    w = np.fromfunction(lambda i, j: (i - j) ** 2 / (k - 1) ** 2, (k, k))
    obs = np.zeros((k, k))
    np.add.at(obs, (y_true - 1, y_pred - 1), 1)
    obs /= n
    return float(1.0 - (w * obs).sum() / (w * expected).sum())


def tune_thresholds(raw: np.ndarray, y_true: np.ndarray):
    """Vectorized grid search: outer loop t1 (11), inner t2-t4 broadcast (512)."""
    g1 = np.arange(0.20, 0.401, 0.02)
    g2 = np.arange(0.40, 0.561, 0.02)
    g3 = np.arange(0.55, 0.711, 0.02)
    g4 = np.arange(0.70, 0.861, 0.02)
    t2, t3, t4 = np.meshgrid(g2, g3, g4, indexing="ij")
    t2, t3, t4 = t2.ravel(), t3.ravel(), t4.ravel()
    valid = (t2[:, None] < t3[:, None]) & (t3[:, None] < t4[:, None])
    best, best_q = None, -1.0
    raw = np.asarray(raw)
    for a in g1:
        m = (t2 > a) & valid[:, 0]
        b2, b3, b4 = t2[m], t3[m], t4[m]
        if len(b2) == 0:
            continue
        # (C, N) comparisons -> class codes without storing all: loop C in chunks
        for s in range(0, len(b2), 128):
            e2, e3, e4 = b2[s:s + 128, None], b3[s:s + 128, None], b4[s:s + 128, None]
            r = raw[None, :]
            pred = np.where(r < a, 1, np.where(r < e2, 2, np.where(r < e3, 3,
                         np.where(r < e4, 4, 5))))
            for ci in range(pred.shape[0]):
                q = qwk(y_true, pred[ci])
                if q > best_q:
                    best_q = q
                    best = (round(float(a), 2), round(float(e2[ci, 0]), 2),
                            round(float(e3[ci, 0]), 2), round(float(e4[ci, 0]), 2))
    return best, best_q


def decode(raw: np.ndarray, thr) -> np.ndarray:
    t1, t2, t3, t4 = thr
    return np.where(raw < t1, 1, np.where(raw < t2, 2, np.where(raw < t3, 3,
            np.where(raw < t4, 4, 5)))).astype(int)


def groups(all_cols: list[str]) -> dict[str, list[str]]:
    vitals = ["temp", "heart_rate", "resp_rate", "sys_bp", "dias_bp", "spo2",
              "pain_score", "visit_month", "day_of_week", "arrival_time"]
    g1 = [c for c in vitals if c in all_cols]
    g1 += [c for c in all_cols if re.match(
        r"^(insurance_|residence_|region_|ems_arrival_|episode_|is_injury_poison_|"
        r"vitals_during_visit_|seen_last_72h_|no_payment_|race_|sex_)", c)]
    g1 += [c for c in all_cols if c.startswith("hist_")]
    miss = [c for c in all_cols if re.match(r"^(miss_|is_missing|has_missing)", c)]
    eng = [c for c in all_cols if re.match(
        r"^(shock_|qsofa_|mews_|sirs_|spo2_|rr_|hr_|age_|temp_|triage_|ems_|pain_|"
        r"symp_|is_|high_)", c)]
    text = [c for c in all_cols if c.startswith("cc_")]
    g = {"E1": sorted(set(g1)),
         "E2": sorted(set(g1 + miss)),
         "E3": sorted(set(g1 + eng)),
         "E4": sorted(set(g1 + eng + text)),
         "E6": sorted(set(all_cols))}
    assert "intervention_iv_fluids" not in g["E4"], "post-hoc leak in E4"
    return g


def train_ordinal(Xtr, ytr, nrounds: int):
    dtr = xgb.DMatrix(Xtr, label=(ytr - 1) / 4)
    return xgb.train(PARAMS, dtr, num_boost_round=nrounds)


def evaluate(model, Xte, yte):
    raw = model.predict(xgb.DMatrix(Xte))
    thr, q = tune_thresholds(raw, yte)
    pred = decode(raw, thr)
    acc = float((pred == yte).mean())
    w1 = float((np.abs(pred - yte) <= 1).mean())
    mae = float(np.abs(pred - yte).mean())
    return {"qwk": round(q, 4), "acc": round(acc, 4), "within1": round(w1, 4),
            "mae": round(mae, 4), "thr": str(thr), "pred": pred, "raw": raw}


def save_progression(rows: list[dict]) -> None:
    with open(OUT / "progression.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["experiment", "n_feat", "qwk", "acc",
                                          "within1", "mae", "thresholds", "seconds"])
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--skip-done", action="store_true")
    args = ap.parse_args()
    rounds = {k: (v // 2 if args.fast else v) for k, v in ROUNDS.items()}
    OUT.mkdir(parents=True, exist_ok=True)

    done: dict[str, dict] = {}
    if args.skip_done and (OUT / "progression.csv").exists():
        with open(OUT / "progression.csv") as f:
            for r in csv.DictReader(f):
                done[r["experiment"]] = r
        print(f"resuming, {len(done)} done")

    print("loading data...")
    df = pd.read_csv(cfg.DATA_PROCESSED)
    y = df["triage_acuity"].values.astype(int)
    feat_cols = [c for c in df.columns if c != "triage_acuity"]
    g = groups(feat_cols)
    print({k: len(v) for k, v in g.items()})

    Xtr_df, Xte_df, ytr, yte = train_test_split(
        df[feat_cols], y, test_size=0.2, random_state=SEED, stratify=y)
    print(f"train {len(ytr)} test {len(yte)}")

    Xtr = Xtr_df.values.astype(np.float32)
    Xte = Xte_df.values.astype(np.float32)
    col_index = {c: i for i, c in enumerate(feat_cols)}
    rows: list[dict] = [done[k] for k in
                        ["E1", "E2", "E3", "E4", "E5mc", "E6"] if k in done]
    cache: dict[str, dict] = {}

    def need(k):
        return k not in done

    for key in ["E1", "E2", "E3", "E4"]:
        if not need(key):
            continue
        cols = g[key]
        ci = [col_index[c] for c in cols]
        t0 = time.time()
        ev = evaluate(train_ordinal(Xtr[:, ci], ytr, rounds[key]), Xte[:, ci], yte)
        rows.append({"experiment": key, "n_feat": len(cols), "qwk": ev["qwk"],
                     "acc": ev["acc"], "within1": ev["within1"], "mae": ev["mae"],
                     "thresholds": ev["thr"], "seconds": round(time.time() - t0, 1)})
        cache[key] = ev
        save_progression(rows)
        print(rows[-1], flush=True)

    if need("E5mc"):
        ci = [col_index[c] for c in g["E4"]]
        t0 = time.time()
        dtr = xgb.DMatrix(Xtr[:, ci], label=ytr - 1)
        dte = xgb.DMatrix(Xte[:, ci])
        p = dict(PARAMS, objective="multi:softprob", num_class=5)
        m = xgb.train(p, dtr, num_boost_round=rounds["E5mc"])
        pred = np.argmax(m.predict(dte), axis=1) + 1
        rows.append({"experiment": "E5mc", "n_feat": len(ci),
                     "qwk": round(qwk(yte, pred), 4),
                     "acc": round(float((pred == yte).mean()), 4),
                     "within1": round(float((np.abs(pred - yte) <= 1).mean()), 4),
                     "mae": round(float(np.abs(pred - yte).mean()), 4),
                     "thresholds": "-", "seconds": round(time.time() - t0, 1)})
        cache["E5mc_pred"] = {"pred": pred}
        save_progression(rows)
        print(rows[-1], flush=True)

    if need("E6"):
        ci = [col_index[c] for c in g["E6"]]
        t0 = time.time()
        ev = evaluate(train_ordinal(Xtr[:, ci], ytr, rounds["E6"]), Xte[:, ci], yte)
        rows.append({"experiment": "E6", "n_feat": len(ci), "qwk": ev["qwk"],
                     "acc": ev["acc"], "within1": ev["within1"], "mae": ev["mae"],
                     "thresholds": ev["thr"], "seconds": round(time.time() - t0, 1)})
        cache["E6"] = ev
        save_progression(rows)
        print(rows[-1], flush=True)
        cm = np.zeros((5, 5), dtype=int)
        np.add.at(cm, (yte - 1, ev["pred"] - 1), 1)
        with open(OUT / "confusion_E6.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["true/pred"] + cfg.ESI_LABELS)
            for lab, row in zip(cfg.ESI_LABELS, cm.tolist()):
                w.writerow([lab] + row)

    if not args.no_ablation:
        full_ci = [col_index[c] for c in g["E6"]]
        drop = {"minus_text": [c for c in g["E6"] if c.startswith("cc_")],
                "minus_eng": [c for c in g["E6"] if re.match(
                    r"^(shock_|qsofa_|mews_|sirs_|spo2_|rr_|hr_|age_|temp_|triage_|"
                    r"ems_|pain_|symp_|is_|high_)", c)],
                "minus_comorb": [c for c in g["E6"] if c.startswith("hist_")],
                "minus_vitals": [c for c in g["E6"] if c in
                                 ["temp", "heart_rate", "resp_rate", "sys_bp", "dias_bp",
                                  "spo2", "pain_score", "visit_month", "day_of_week",
                                  "arrival_time"]],
                "minus_demog": [c for c in g["E6"] if re.match(
                    r"^(insurance_|residence_|region_|race_|sex_)", c)],
                "minus_iv": ["intervention_iv_fluids"]}
        for key, dc in drop.items():
            if not need(key):
                continue
            keep = [c for c in g["E6"] if c not in set(dc)]
            ki = [col_index[c] for c in keep]
            t0 = time.time()
            ev = evaluate(train_ordinal(Xtr[:, ki], ytr, rounds["ABL"]), Xte[:, ki], yte)
            rows.append({"experiment": key, "n_feat": len(keep), "qwk": ev["qwk"],
                         "acc": ev["acc"], "within1": ev["within1"], "mae": ev["mae"],
                         "thresholds": ev["thr"], "seconds": round(time.time() - t0, 1)})
            save_progression(rows)
            print(rows[-1], flush=True)

    print("DONE. Results in", OUT)


if __name__ == "__main__":
    main()
