"""TRIAGEGEIST clean pipeline — global config. Paths + seed only, no numbers."""
from pathlib import Path

GLOBAL_SEED = 42
BOOTSTRAP_REPS = 2000

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed" / "triage_full.csv"
DATA_RAW = ROOT / "data" / "raw" / "nhamcs_data_2018_22.csv"
OUTPUTS = ROOT / "outputs"
REVISION = ROOT / "revision"
RESULTS = REVISION / "results"
FIGURES = REVISION / "figures"

# Source-of-truth files produced by the original R/Python runs (read, never edited)
REPORT_TXT = OUTPUTS / "report.txt"                       # E6 CM + per-class (analysis.R run)
EXPERIMENTS_CSV = OUTPUTS / "r_experiments_summary.csv"   # E1-E6 progression (experiments.R run)
CMP_TEXT_CSV = OUTPUTS / "model_comparison_with_text.csv"
CMP_TAB_CSV = OUTPUTS / "model_comparison_tabular_only.csv"
BIAS_TXT = OUTPUTS / "bias" / "r_bias_output.txt"
BIAS_CSV = OUTPUTS / "r_bias_summary.csv"

ESI_LABELS = ["ESI-1", "ESI-2", "ESI-3", "ESI-4", "ESI-5"]
