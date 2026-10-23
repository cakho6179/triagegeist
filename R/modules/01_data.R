# ============================================================
# Module 01: Data Loading and Preparation
# ============================================================
source("R/modules/utils.R")

load_data <- function(sample_size = NULL, csv_path = NULL) {
  log_msg("Loading data...")
  
  if (!is.null(csv_path)) {
    df <- fread(csv_path)
  } else {
    # Try loading from preprocessed parquet via Python export
    # First check if CSV exists
    if (file.exists("data/processed/triage_full.csv")) {
      df <- fread("data/processed/triage_full.csv")
    } else {
      # Need to export from Python first
      log_msg("CSV not found. Exporting from Python...")
      system("python -c \"
import pandas as pd, numpy as np
X=pd.read_parquet('data/processed/X_train_enhanced.parquet').drop(columns=['year'],errors='ignore')
Xv=pd.read_parquet('data/processed/X_val_enhanced.parquet').drop(columns=['year'],errors='ignore')
y=pd.read_csv('data/processed/y_train_enhanced.csv').squeeze('columns').values
yv=pd.read_csv('data/processed/y_val_enhanced.csv').squeeze('columns').values
X_all=pd.concat([X, Xv], axis=0)
y_all=np.concatenate([y, yv])
df=X_all.copy(); df['triage_acuity']=y_all
df.to_csv('data/processed/triage_full.csv', index=False)
print('Exported triage_full.csv')
\"")
      df <- fread("data/processed/triage_full.csv")
    }
  }
  
  if (!is.null(sample_size) && sample_size < nrow(df)) {
    set.seed(42)
    df <- df[sample(nrow(df), sample_size), ]
  }
  
  log_msg(sprintf("Loaded: %d rows, %d columns", nrow(df), ncol(df)))
  log_msg(sprintf("Classes: %s", paste(table(df$triage_acuity), collapse = ", ")))
  df
}

prepare_data <- function(df, test_frac = 0.2, use_temporal = FALSE, temporal_years = NULL) {
  log_msg("Preparing data split...")
  
  y <- df$triage_acuity
  X <- df[, !names(df) %in% "triage_acuity", with = FALSE]
  feature_names <- names(X)
  
  if (use_temporal && !is.null(temporal_years) && "year" %in% names(X)) {
    train_idx <- which(!df$year %in% temporal_years[[2]])
    test_idx  <- which(df$year %in% temporal_years[[2]])
    log_msg(sprintf("Temporal split: train=%d, test=%d", length(train_idx), length(test_idx)))
  } else {
    set.seed(42)
    n <- nrow(df)
    train_idx <- sample(n, floor((1 - test_frac) * n))
    test_idx <- setdiff(1:n, train_idx)
  }
  
  list(
    X_train = X[train_idx, ], X_test = X[test_idx, ],
    y_train = y[train_idx],   y_test = y[test_idx],
    feature_names = feature_names,
    train_idx = train_idx,     test_idx = test_idx
  )
}

summarize_classes <- function(y) {
  tbl <- table(y)
  cat("Class distribution:\n")
  for (c in names(tbl)) cat(sprintf("  ESI-%s: %d (%.1f%%)\n", c, tbl[c], 100*tbl[c]/sum(tbl)))
}
