"""07 — E6 3-seed (42,43,44) mean-ensemble at 1000 rounds + full metrics.

Saves: revision/results/retrain/{confusion_E6_ens,perclass_E6_ens,metrics_E6_ens}.csv
Run:   python pipeline/07_ensemble_E6.py   (~6 min CPU)
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib
m04 = importlib.import_module("04_retrain")
m05 = importlib.import_module("05_seed_ensemble")
m01 = importlib.import_module("01_metrics_bootstrap")
cfg = importlib.import_module("00_config")

R = cfg.RESULTS / "retrain"


def main() -> None:
    df = pd.read_csv(cfg.DATA_PROCESSED)
    y = df["triage_acuity"].values.astype(int)
    feats = [c for c in df.columns if c != "triage_acuity"]
    g = m04.groups(feats)
    Xtr_df, Xte_df, ytr, yte = train_test_split(
        df[feats], y, test_size=0.2, random_state=42, stratify=y)
    ci = {c: i for i, c in enumerate(feats)}
    cols = [ci[c] for c in g["E6"]]
    Xa = Xtr_df.values.astype(np.float32)[:, cols]
    Xb = Xte_df.values.astype(np.float32)[:, cols]

    raws = []
    for s in (42, 43, 44):
        p = dict(m04.PARAMS, seed=s)
        dtr = xgb.DMatrix(Xa, label=(ytr - 1) / 4)
        raws.append(xgb.train(p, dtr, num_boost_round=1000).predict(xgb.DMatrix(Xb)))
        print(f"seed {s} done", flush=True)
    ens = np.mean(raws, axis=0)
    thr0, _ = m04.tune_thresholds(ens, yte)
    thr, q = m05.refine_thresholds(ens, yte, thr0)
    pred = m04.decode(ens, thr)

    cm = np.zeros((5, 5), dtype=int)
    np.add.at(cm, (yte - 1, pred - 1), 1)
    n = cm.sum()
    k = 5
    idx = np.arange(k)
    dist = np.abs(idx[:, None] - idx[None, :])
    rng = np.random.default_rng(42)
    probs = (cm / n).ravel()
    boots = np.array([m01.qwk_from_cm(
        rng.multinomial(int(n), probs).reshape(k, k).astype(float))
        for _ in range(2000)])
    per = m01.per_class(cm)
    mrows = [
        ("n_test", int(n)), ("qwk", round(float(q), 4)),
        ("qwk_ci_lo", round(float(np.percentile(boots, 2.5)), 4)),
        ("qwk_ci_hi", round(float(np.percentile(boots, 97.5)), 4)),
        ("accuracy", round(float((pred == yte).mean()), 4)),
        ("macro_f1", round(float(np.mean([p["f1"] for p in per])), 4)),
        ("balanced_accuracy", round(float(np.mean([p["recall"] for p in per])), 4)),
        ("mae", round(float(np.abs(pred - yte).mean()), 4)),
        ("within_1", round(float((np.abs(pred - yte) <= 1).mean()), 4)),
        ("undertriage", round(float(np.sum(pred > yte)) / n, 4)),
        ("overtriage", round(float(np.sum(pred < yte)) / n, 4)),
    ]
    with open(R / "confusion_E6_ens.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true/pred"] + cfg.ESI_LABELS)
        for lab, row in zip(cfg.ESI_LABELS, cm.tolist()):
            w.writerow([lab] + row)
    with open(R / "perclass_E6_ens.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per[0].keys()))
        w.writeheader()
        w.writerows(per)
    with open(R / "metrics_E6_ens.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerows(mrows)
    print("E6-ens1000 qwk=", round(float(q), 4), "thr=", thr)
    print(dict(mrows))
    for p in per:
        print(p)


if __name__ == "__main__":
    main()
