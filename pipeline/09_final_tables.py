"""09 — Metrics and bootstrap CIs for every model, built only from the saved test predictions.

Reads:  revision/results/final/pred_*.csv  (written by R/final_*.R and pipeline/08)
Writes: revision/results/final/summary.csv, paired_differences.csv, confusion_<name>.csv, perclass_<name>.csv
Run:    python pipeline/09_final_tables.py
The 95% CI is the multinomial resample of the test confusion distribution used in pipeline/01.
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
cfg = importlib.import_module("00_config")

FINAL = cfg.ROOT / "revision" / "results" / "final"
K = 5


def confusion(y, p):
    cm = np.zeros((K, K))
    np.add.at(cm, (y - 1, p - 1), 1)
    return cm


def qwk_cm(cm):
    n = cm.sum()
    exp = np.outer(cm.sum(1) / n, cm.sum(0) / n)
    w = np.fromfunction(lambda i, j: (i - j) ** 2 / (K - 1) ** 2, (K, K))
    return 1.0 - (w * cm / n).sum() / (w * exp).sum()


def summarize(name, y, p, rng):
    cm = confusion(y, p)
    n = cm.sum()
    d = np.abs(np.arange(K)[:, None] - np.arange(K)[None, :])
    boots = np.array([qwk_cm(rng.multinomial(int(n), (cm / n).ravel()).reshape(K, K).astype(float))
                      for _ in range(cfg.BOOTSTRAP_REPS)])
    prec = np.diag(cm) / np.maximum(cm.sum(0), 1)
    rec = np.diag(cm) / cm.sum(1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    pc = pd.DataFrame({"class": cfg.ESI_LABELS, "support": cm.sum(1).astype(int),
                       "precision": prec.round(4), "recall": rec.round(4), "f1": f1.round(4)})
    pc.to_csv(FINAL / f"perclass_{name}.csv", index=False)
    pd.DataFrame(cm.astype(int), index=cfg.ESI_LABELS, columns=cfg.ESI_LABELS).to_csv(FINAL / f"confusion_{name}.csv")
    return dict(model=name, n=int(n), qwk=round(qwk_cm(cm), 4),
                ci_lo=round(float(np.percentile(boots, 2.5)), 4), ci_hi=round(float(np.percentile(boots, 97.5)), 4),
                acc=round(float(np.trace(cm) / n), 4), within1=round(float(cm[d <= 1].sum() / n), 4),
                mae=round(float((d * cm).sum() / n), 4),
                undertriage=round(float(np.triu(cm, 1).sum() / n), 4),
                overtriage=round(float(np.tril(cm, -1).sum() / n), 4),
                macro_f1=round(float(f1.mean()), 4), macro_recall=round(float(rec.mean()), 4))


PAIRS = [("E4_test_tuned", "multiclass_argmax"), ("E4_test_tuned", "multiclass_expected_test_tuned"),
         ("E4_test_tuned", "MiniLM_test_tuned"), ("E4_test_tuned", "CORN"), ("E4_test_tuned", "CORAL"),
         ("E4_test_tuned", "E4_train_tuned")]


def paired(a, b, rng):
    """Paired bootstrap of QWK(a) - QWK(b): the same resampled test rows are scored for both models."""
    da, db = pd.read_csv(FINAL / f"pred_{a}.csv"), pd.read_csv(FINAL / f"pred_{b}.csv")
    assert (da["y"].to_numpy() == db["y"].to_numpy()).all()
    y, pa, pb = da["y"].to_numpy(), da["pred"].to_numpy(), db["pred"].to_numpy()
    n = len(y)
    diffs = []
    for _ in range(cfg.BOOTSTRAP_REPS):
        i = rng.integers(0, n, n)
        diffs.append(qwk_cm(confusion(y[i], pa[i])) - qwk_cm(confusion(y[i], pb[i])))
    return dict(a=a, b=b, diff=round(qwk_cm(confusion(y, pa)) - qwk_cm(confusion(y, pb)), 4),
                ci_lo=round(float(np.percentile(diffs, 2.5)), 4), ci_hi=round(float(np.percentile(diffs, 97.5)), 4))


def main():
    rng = np.random.default_rng(cfg.GLOBAL_SEED)
    rows = []
    for f in sorted(FINAL.glob("pred_*.csv")):
        d = pd.read_csv(f)
        rows.append(summarize(f.stem[len("pred_"):], d["y"].to_numpy(), d["pred"].to_numpy(), rng))
    out = pd.DataFrame(rows)
    out.to_csv(FINAL / "summary.csv", index=False)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(out.to_string(index=False))
    pr = pd.DataFrame([paired(a, b, rng) for a, b in PAIRS if (FINAL / f"pred_{a}.csv").exists()
                       and (FINAL / f"pred_{b}.csv").exists()])
    pr.to_csv(FINAL / "paired_differences.csv", index=False)
    print(pr.to_string(index=False))


if __name__ == "__main__":
    main()
