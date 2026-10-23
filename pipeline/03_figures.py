"""03 — Paper figures 2 (confusion matrix of E4) and 4 (Cramér's V and wait times).

The QWK progression figure is made by pipeline/10_progression_figure.py.
Reads:  revision/results/final/confusion_E4_test_tuned.csv, outputs/bias/r_bias_output.txt, outputs/r_wait_by_acuity.csv
Writes: revision/figures_final/fig1_confusion_matrix.png, fig4_fairness_wait.png (300 DPI, IEEE single-column)
Run:    python pipeline/03_figures.py
"""
import csv
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib
cfg = importlib.import_module("00_config")

# IEEE standard typography
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 7.5,
    "axes.titlesize": 8,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 300
})


def read_confusion(path: Path) -> np.ndarray:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return np.array([[float(v) for v in r[1:]] for r in rows[1:]])


def read_cramers(path: Path) -> list[tuple[str, float]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    found = re.findall(r"(\w+) vs acuity:.*?V=([\d.]+)", text)
    return [(k, float(v)) for k, v in found]


def read_wait_by_acuity(path: Path) -> dict[int, float]:
    with open(path, encoding="utf-8") as f:
        return {int(float(r["target_triage_acuity"])): float(r["median"])
                for r in csv.DictReader(f)}


def main() -> None:
    cfg.FIGURES.mkdir(parents=True, exist_ok=True)
    
    out = cfg.ROOT / "revision" / "figures_final"
    out.mkdir(parents=True, exist_ok=True)
    cm = read_confusion(cfg.RESULTS / "final" / "confusion_E4_test_tuned.csv")
    labels = cfg.ESI_LABELS

    # ==============================================================================
    # FIG 1: CONFUSION MATRIX (Row %, Primary Deployable Model E4)
    # ==============================================================================
    row_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    im = ax.imshow(row_pct, vmin=0, vmax=100, cmap="Blues")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted Acuity Level (ESI)", fontsize=7.5)
    ax.set_ylabel("Reference True Acuity (ESI)", fontsize=7.5)

    for i in range(5):
        for j in range(5):
            val_str = f"{row_pct[i, j]:.1f}%\n({int(cm[i, j])})"
            ax.text(j, i, val_str, ha="center", va="center",
                    fontsize=5.8, color="white" if row_pct[i, j] > 48 else "#111111")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label("Recall Rate (% of True Class)", fontsize=6.8)
    fig.tight_layout()
    fig.savefig(out / "fig1_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ==============================================================================
    # FIG 4: FAIRNESS & WAIT-TIME GRADIENT
    # ==============================================================================
    cv = read_cramers(cfg.BIAS_TXT)
    # Sort or map standard categories descending
    cat_map = {"age": "Age", "age_group": "Age", "arrival": "EMS", "ems_arrival": "EMS", 
               "payer": "Payer", "insurance": "Payer", "sex": "Sex", "race": "Race"}
    items = []
    for k, v in cv:
        label = cat_map.get(k.lower(), k)
        items.append((label, v))
    assert items, "no Cramér's V values found in " + str(cfg.BIAS_TXT)
    items.sort(key=lambda t: t[1], reverse=True)

    cats_plot = [it[0] for it in items]
    vals_plot = [it[1] for it in items]

    wait = read_wait_by_acuity(cfg.OUTPUTS / "r_wait_by_acuity.csv")
    wait_med = [int(round(wait.get(i, 0))) for i in range(1, 6)]
    assert sum(wait_med) > 0, "no wait-time medians found"

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.4, 2.0), gridspec_kw={"width_ratios": [1, 1]})

    # Subplot (a)
    a1.bar(cats_plot, vals_plot, color="#0288d1", edgecolor="#263238", linewidth=0.5, width=0.58)
    a1.set_ylabel("Cramér's V (Acuity Assoc.)", fontsize=6.8)
    a1.set_title("(a) Demographic Fairness", fontsize=7.2, fontweight="bold")
    a1.tick_params(axis="x", labelsize=6.5)
    a1.tick_params(axis="y", labelsize=6.5)
    a1.set_ylim(0, 0.20)
    a1.grid(axis="y", linestyle="--", alpha=0.3)
    
    # Stagger labels if first two are close
    if len(vals_plot) >= 2 and abs(vals_plot[0] - vals_plot[1]) < 0.01:
        a1.text(0, vals_plot[0] + 0.012, f"{vals_plot[0]:.3f}", ha="center", fontsize=5.5)
        a1.text(1, vals_plot[1] + 0.004, f"{vals_plot[1]:.3f}", ha="center", fontsize=5.5)
        for i in range(2, len(vals_plot)):
            a1.text(i, vals_plot[i] + 0.006, f"{vals_plot[i]:.3f}", ha="center", fontsize=5.5)
    else:
        for i in range(len(vals_plot)):
            a1.text(i, vals_plot[i] + 0.006, f"{vals_plot[i]:.3f}", ha="center", fontsize=5.5)

    # Subplot (b)
    wait_labels = ["ESI-1", "ESI-2", "ESI-3", "ESI-4", "ESI-5"]
    a2.bar(wait_labels, wait_med, color="#26a69a", edgecolor="#263238", linewidth=0.5, width=0.6)
    a2.set_ylabel("Median Wait Time (min)", fontsize=6.8)
    a2.set_title("(b) Wait by ESI Level", fontsize=7.2, fontweight="bold")
    a2.tick_params(axis="x", labelsize=6.2)
    a2.tick_params(axis="y", labelsize=6.5)
    a2.set_ylim(0, 22)
    a2.grid(axis="y", linestyle="--", alpha=0.3)
    for i, v in enumerate(wait_med):
        a2.text(i, v + 0.6, f"{v}m", ha="center", fontsize=5.8)

    fig.tight_layout()
    fig.savefig(out / "fig4_fairness_wait.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Clean 300-DPI IEEE figures successfully saved to:", sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
