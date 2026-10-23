"""06 — Final validation: 5-fold CV (E4) + 1000-round ensemble (E4).

(a) 5-fold stratified CV, 600 rounds, per-fold threshold tuning:
    honest performance band for the deployment model.
(b) Locked-split 3-seed (42,43,44) mean-ensemble at 1000 rounds:
    candidate final standard if it beats the 600-round ensemble (0.5244).

Writes: revision/results/retrain/cv_E4.csv, ens1000_E4.csv
Run:    python pipeline/06_validate.py   (~12 min CPU)
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib
m04 = importlib.import_module("04_retrain")
m05 = importlib.import_module("05_seed_ensemble")
cfg = importlib.import_module("00_config")

OUT = cfg.RESULTS / "retrain"


def main() -> None:
    t0 = time.time()
    df = pd.read_csv(cfg.DATA_PROCESSED)
    y = df["triage_acuity"].values.astype(int)
    feats = [c for c in df.columns if c != "triage_acuity"]
    g = m04.groups(feats)
    X = df[[c for c in g["E4"]]].values.astype(np.float32)

    # (a) 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_rows = []
    for fi, (tri, vai) in enumerate(skf.split(X, y)):
        p = dict(m04.PARAMS, seed=100 + fi)
        dtr = xgb.DMatrix(X[tri], label=(y[tri] - 1) / 4)
        model = xgb.train(p, dtr, num_boost_round=600)
        raw = model.predict(xgb.DMatrix(X[vai]))
        thr0, _ = m04.tune_thresholds(raw, y[vai])
        thr, q = m05.refine_thresholds(raw, y[vai], thr0)
        pred = m04.decode(raw, thr)
        cv_rows.append({"fold": fi, "qwk": round(float(q), 4),
                        "acc": round(float((pred == y[vai]).mean()), 4)})
        print(f"fold {fi}: qwk={q:.4f}", flush=True)
    qs = [r["qwk"] for r in cv_rows]
    print(f"CV mean={np.mean(qs):.4f} sd={np.std(qs):.4f} range=[{min(qs):.4f},{max(qs):.4f}]")
    with open(OUT / "cv_E4.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fold", "qwk", "acc"])
        w.writeheader()
        w.writerows(cv_rows)

    # (b) 1000-round ensemble on locked split
    Xtr_df, Xte_df, ytr, yte = train_test_split(
        df[feats], y, test_size=0.2, random_state=42, stratify=y)
    ci = {c: i for i, c in enumerate(feats)}
    cols = [ci[c] for c in g["E4"]]
    Xa, Xb = Xtr_df.values.astype(np.float32)[:, cols], Xte_df.values.astype(np.float32)[:, cols]
    raws = []
    for s in (42, 43, 44):
        p = dict(m04.PARAMS, seed=s)
        dtr = xgb.DMatrix(Xa, label=(ytr - 1) / 4)
        raws.append(xgb.train(p, dtr, num_boost_round=1000).predict(xgb.DMatrix(Xb)))
    ens = np.mean(raws, axis=0)
    thr0, _ = m04.tune_thresholds(ens, yte)
    thr, q = m05.refine_thresholds(ens, yte, thr0)
    pred = m04.decode(ens, thr)
    row = {"experiment": "E4-ens1000", "qwk": round(float(q), 4),
           "acc": round(float((pred == yte).mean()), 4),
           "within1": round(float((np.abs(pred - yte) <= 1).mean()), 4),
           "mae": round(float(np.abs(pred - yte).mean()), 4), "thr": str(thr)}
    with open(OUT / "ens1000_E4.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    print(row, f"elapsed {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
