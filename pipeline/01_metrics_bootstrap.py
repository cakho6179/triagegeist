"""01 — Metrics + bootstrap CI from saved R/Python outputs. No retraining, no hardcoded numbers.

Reads:  outputs/report.txt (E6 CM), outputs/r_experiments_summary.csv,
        outputs/model_comparison_*.csv, outputs/bias/r_bias_output.txt
Writes: revision/results/*.csv + prints manuscript-ready numbers.
Run:    python pipeline/01_metrics_bootstrap.py   (~10 seconds)
"""
import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from00_config = __import__("00_config")  # noqa: E402  (module name starts with digits)
cfg = from00_config

rng = np.random.default_rng(cfg.GLOBAL_SEED)


def parse_cm(report_path: Path) -> np.ndarray:
    """Parse the 5x5 confusion matrix from report.txt section 3."""
    text = report_path.read_text(encoding="utf-8", errors="replace")
    rows = re.findall(r"^\s+[1-5]\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", text, re.M)
    assert len(rows) == 5, f"expected 5 CM rows, found {len(rows)} in {report_path}"
    return np.array([[int(v) for v in r] for r in rows], dtype=float)


def qwk_from_cm(cm: np.ndarray, k: int = 5) -> float:
    n = cm.sum()
    hist = cm.sum(axis=1) / n
    pred = cm.sum(axis=0) / n
    expected = np.outer(hist, pred)
    w = np.fromfunction(lambda i, j: (i - j) ** 2 / (k - 1) ** 2, (k, k))
    return float(1.0 - (w * cm / n).sum() / (w * expected).sum())


def per_class(cm: np.ndarray):
    out = []
    for i, lab in enumerate(cfg.ESI_LABELS):
        tp = cm[i, i]
        prec = tp / cm[:, i].sum()
        rec = tp / cm[i, :].sum()
        f1 = 2 * prec * rec / (prec + rec)
        out.append({"class": lab, "n_true": int(cm[i, :].sum()),
                    "n_pred": int(cm[:, i].sum()),
                    "precision": round(float(prec), 4), "recall": round(float(rec), 4),
                    "f1": round(float(f1), 4)})
    return out


def main() -> None:
    cfg.RESULTS.mkdir(parents=True, exist_ok=True)
    cm = parse_cm(cfg.REPORT_TXT)
    n = cm.sum()
    k = cm.shape[0]
    idx = np.arange(k)

    acc = float(np.diag(cm).sum() / n)
    dist = np.abs(idx[:, None] - idx[None, :])
    mae = float((dist * cm).sum() / n)
    within1 = float(cm[dist <= 1].sum() / n)
    undertriage = float(cm[np.triu_indices(k, 1)].sum() / n)  # pred less urgent than true
    overtriage = float(cm[np.tril_indices(k, -1)].sum() / n)  # pred more urgent than true
    off_by = {f"off_by_{d}": float(cm[dist == d].sum() / n) for d in range(k)}
    qwk = qwk_from_cm(cm)

    # Bootstrap 95% CI: multinomial resample of the test confusion distribution
    probs = (cm / n).ravel()
    boots = []
    for _ in range(cfg.BOOTSTRAP_REPS):
        sample = rng.multinomial(int(n), probs).reshape(k, k).astype(float)
        boots.append(qwk_from_cm(sample))
    boots = np.array(boots)

    per = per_class(cm)
    macro_f1 = round(float(np.mean([p["f1"] for p in per])), 4)
    bal_acc = round(float(np.mean([p["recall"] for p in per])), 4)

    with open(cfg.RESULTS / "metrics_E6.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerows([
            ("n_test", int(n)), ("qwk", round(qwk, 4)),
            ("qwk_ci_lo", round(float(np.percentile(boots, 2.5)), 4)),
            ("qwk_ci_hi", round(float(np.percentile(boots, 97.5)), 4)),
            ("accuracy", round(acc, 4)), ("macro_f1", macro_f1),
            ("balanced_accuracy", bal_acc), ("mae", round(mae, 4)),
            ("within_1", round(within1, 4)), ("undertriage", round(undertriage, 4)),
            ("overtriage", round(overtriage, 4)),
        ] + [(kk, round(vv, 4)) for kk, vv in off_by.items()])

    with open(cfg.RESULTS / "perclass_E6.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per[0].keys()))
        w.writeheader()
        w.writerows(per)

    with open(cfg.RESULTS / "confusion_E6.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true/pred"] + cfg.ESI_LABELS)
        for lab, row in zip(cfg.ESI_LABELS, cm.astype(int).tolist()):
            w.writerow([lab] + row)

    print(f"QWK={qwk:.4f} 95%CI [{np.percentile(boots, 2.5):.4f}-{np.percentile(boots, 97.5):.4f}]")
    print(f"acc={acc:.4f} macroF1={macro_f1} balAcc={bal_acc} MAE={mae:.4f} within1={within1:.4f}")
    print(f"undertriage={undertriage:.4f} overtriage={overtriage:.4f} " +
          " ".join(f"{kk}={vv:.4f}" for kk, vv in off_by.items()))
    for p in per:
        print(p)


if __name__ == "__main__":
    main()
