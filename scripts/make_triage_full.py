"""Build data/processed/triage_full.csv (744 features + triage_acuity) from the
enhanced train/validation parquet files written by scripts/run_preprocess_enhanced.py.
Same logic as the export snippet in R/modules/01_data.R. Run from the repository root."""
import numpy as np
import pandas as pd

P = "data/processed/"
X = pd.read_parquet(P + "X_train_enhanced.parquet").drop(columns=["year"], errors="ignore")
Xv = pd.read_parquet(P + "X_val_enhanced.parquet").drop(columns=["year"], errors="ignore")
y = pd.read_csv(P + "y_train_enhanced.csv").squeeze("columns").values
yv = pd.read_csv(P + "y_val_enhanced.csv").squeeze("columns").values

df = pd.concat([X, Xv], axis=0)
df["triage_acuity"] = np.concatenate([y, yv])
df.to_csv(P + "triage_full.csv", index=False)
print("triage_full.csv:", df.shape)
