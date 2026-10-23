"""10 — Progression figure (paper Fig. 1) from revision/results/final/summary.csv.

Run after pipeline/09_final_tables.py. Writes revision/figures_final/fig2_progression.png (300 dpi).
"""
import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
cfg = importlib.import_module("00_config")

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
                     "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7})

S = pd.read_csv(cfg.ROOT / "revision" / "results" / "final" / "summary.csv").set_index("model")
order = [("E1", "E1"), ("E2", "E2_test_tuned"), ("E3", "E3"), ("E4*", "E4_test_tuned"), ("E5mc", "multiclass_argmax")]
q = np.array([S.loc[k, "qwk"] for _, k in order])
lo = np.array([S.loc[k, "ci_lo"] for _, k in order])
hi = np.array([S.loc[k, "ci_hi"] for _, k in order])
colors = ["#cfd8dc", "#cfd8dc", "#90caf9", "#1976d2", "#ffb74d"]

fig, ax = plt.subplots(figsize=(3.3, 2.2))
bars = ax.bar([n for n, _ in order], q, yerr=[q - lo, hi - q], capsize=2.5, color=colors,
              edgecolor="#263238", linewidth=0.6, width=0.62)
ax.set_ylim(0.38, 0.60)
ax.set_ylabel("Quadratic Weighted Kappa (QWK)")
ax.set_xlabel("Experiment (* deployable model)")
ax.axhline(q[3], color="#1976d2", linestyle=":", linewidth=0.8, alpha=0.8)
for b, v, h in zip(bars, q, hi):
    ax.text(b.get_x() + b.get_width() / 2, h + 0.006, f"{v:.4f}", ha="center", fontsize=5.8, fontweight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.3)
fig.tight_layout()
out = cfg.ROOT / "revision" / "figures_final"
out.mkdir(exist_ok=True)
fig.savefig(out / "fig2_progression.png", dpi=300, bbox_inches="tight")
print("saved", out / "fig2_progression.png")
