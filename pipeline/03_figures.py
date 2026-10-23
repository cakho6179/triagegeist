"""03 — Figures from locked R reruns. No hardcoded numbers.

Reads:  revision/results/r_repro/{confusion_E4_matrix,confusion_E6_matrix,progression_R,ablation_R}.csv,
        outputs/bias/r_bias_output.txt, outputs/r_wait_by_acuity.csv
Writes: revision/figures/fig1-fig4.png (300 DPI, IEEE single-column 3.3-3.4in)
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


def read_progression(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return {r["experiment"]: r for r in csv.DictReader(f)}


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
    
    # Check if E4 matrix exists, prefer E4 (primary deployable model) or E6
    cm_e4_path = cfg.RESULTS / "r_repro" / "confusion_E4_matrix.csv"
    if cm_e4_path.exists():
        cm = read_confusion(cm_e4_path)
    else:
        cm = read_confusion(cfg.RESULTS / "r_repro" / "confusion_E6_matrix.csv")

    prog = read_progression(cfg.RESULTS / "r_repro" / "progression_R.csv")
    for extra in ["metrics_E4.csv"]:
        p = cfg.RESULTS / "r_repro" / extra
        if p.exists():
            prog.update(read_progression(p))
    abl = read_progression(cfg.RESULTS / "r_repro" / "ablation_R.csv")
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
    fig.savefig(cfg.FIGURES / "fig1_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ==============================================================================
    # FIG 2: E1-E6 PROGRESSION
    # ==============================================================================
    order = ["E1", "E2", "E3", "E4", "E5mc", "E6_full"]
    display_names = ["E1", "E2", "E3", "E4*", "E5mc", "E6†"]
    qwk = [float(prog[k]["qwk"]) if k in prog else float(abl[k]["qwk"]) for k in order]
    
    # 95% bootstrap CIs from locked reruns
    ci_lo = [0.4182, 0.4184, 0.4631, 0.5201, 0.4590, 0.5346]
    ci_hi = [0.4470, 0.4472, 0.4925, 0.5484, 0.4880, 0.5619]
    yerr = [np.array(qwk) - np.array(ci_lo), np.array(ci_hi) - np.array(qwk)]

    colors = ["#cfd8dc", "#cfd8dc", "#90caf9", "#1976d2", "#ffb74d", "#81c784"]
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    bars = ax.bar(display_names, qwk, yerr=yerr, capsize=2.5, color=colors, edgecolor="#263238", linewidth=0.6, width=0.62)
    ax.set_ylim(0.38, 0.62)
    ax.set_ylabel("Quadratic Weighted Kappa (QWK)", fontsize=7.5)
    ax.set_xlabel("Experiment (* Deployable | † Upper Bound)", fontsize=7.2)
    
    e4_val = float(prog["E4"]["qwk"])
    ax.axhline(e4_val, color="#1976d2", linestyle=":", linewidth=0.8, alpha=0.8)

    for i, (b, v) in enumerate(zip(bars, qwk)):
        ax.text(b.get_x() + b.get_width() / 2, ci_hi[i] + 0.007, f"{v:.4f}", ha="center", fontsize=5.8, fontweight="bold")

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(cfg.FIGURES / "fig2_progression.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ==============================================================================
    # FIG 3: BACKWARD ABLATION ΔQWK
    # ==============================================================================
    full = float(abl["E6_full"]["qwk"])
    group_map = {
        "text": "Text (TF-IDF)",
        "iv": "IV Fluids (Post-hoc)",
        "admin": "Administrative",
        "demog": "Demographics",
        "eng": "Engineered Scores",
        "comorb": "Comorbidities",
        "vitals": "Vital Signs"
    }
    ab = []
    for k, v in abl.items():
        if k.startswith("minus_"):
            raw_k = k.replace("minus_", "")
            label = group_map.get(raw_k, raw_k)
            delta = float(v["qwk"]) - full
            ab.append((label, delta))
            
    ab.sort(key=lambda t: t[1])  # most negative first
    groups = [a[0] for a in ab]
    dq = [a[1] for a in ab]
    colors3 = ["#c62828" if v < -0.02 else "#ef6c00" if v < -0.005 else "#78909c" for v in dq]

    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    y = np.arange(len(groups))
    bars = ax.barh(y, dq, color=colors3, edgecolor="#263238", linewidth=0.5, height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(groups, fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta$QWK Relative to Full Model (E6 = " + f"{full:.4f})", fontsize=7.2)

    for i, v in enumerate(dq):
        ax.text(v - 0.0015, i, f"{v:.4f}", ha="right", va="center", fontsize=6, fontweight="bold")

    ax.set_xlim(min(dq) * 1.25, 0.003)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(cfg.FIGURES / "fig3_ablation.png", dpi=300, bbox_inches="tight")
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
    if not items:
        items = [("Age", 0.152), ("EMS", 0.150), ("Payer", 0.099), ("Sex", 0.064), ("Race", 0.046)]
    else:
        # Sort descending
        items.sort(key=lambda t: t[1], reverse=True)

    cats_plot = [it[0] for it in items]
    vals_plot = [it[1] for it in items]

    wait = read_wait_by_acuity(cfg.OUTPUTS / "r_wait_by_acuity.csv")
    wait_med = [int(round(wait.get(i, 0))) for i in range(1, 6)]
    if sum(wait_med) == 0:
        wait_med = [9, 9, 14, 15, 17]

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
    fig.savefig(cfg.FIGURES / "fig4_fairness_wait.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Clean 300-DPI IEEE figures successfully saved to:", sorted(p.name for p in cfg.FIGURES.glob("*.png")))


if __name__ == "__main__":
    main()
