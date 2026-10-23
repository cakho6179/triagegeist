# Export all RDS files to CSV so the Kaggle notebook can use them
# without requiring R to be installed in the Kaggle environment.

library(data.table)

OUTPUTS <- "outputs"
# Run from project root, not from R/
setwd(normalizePath(file.path(dirname(sys.frame(1)$ofile), "..")))

# 1. Raw predictions and test labels (for confusion matrix)
raw <- readRDS(file.path(OUTPUTS, "r_best_raw_predictions.rds"))
y_test <- readRDS(file.path(OUTPUTS, "r_best_y_test.rds"))
data.frame(index = seq_along(raw), raw_pred = raw, y_test = y_test) |>
  fwrite(file.path(OUTPUTS, "r_best_predictions.csv"))

# 2. Feature importance
imp <- readRDS(file.path(OUTPUTS, "r_feature_importance.rds"))
imp |>
  fwrite(file.path(OUTPUTS, "r_feature_importance.csv"))

# 3. Bias summary
b <- readRDS(file.path(OUTPUTS, "r_bias_summary.rds"))
bias_df <- data.frame(
  Variable = rep(names(b), sapply(b, nrow)),
  Level    = unlist(lapply(b, function(x) x[, 1])),
  stringsAsFactors = FALSE
)
for (col in c("mean_acuity", "median_wait", "N")) {
  bias_df[[col]] <- unlist(lapply(b, function(x) x[[col]]))
}
fwrite(bias_df, file.path(OUTPUTS, "r_bias_summary.csv"))

# 4. Wait time summary
w <- readRDS(file.path(OUTPUTS, "r_wait_time_summary.rds"))
for (nm in names(w)) {
  if (is.data.frame(w[[nm]])) {
    fwrite(w[[nm]], file.path(OUTPUTS, paste0("r_wait_", nm, ".csv")))
  }
}
reg <- w$regression
reg_df <- data.frame(
  metric = names(reg),
  value  = unlist(lapply(reg, as.numeric))
)
fwrite(reg_df, file.path(OUTPUTS, "r_wait_regression.csv"))

# 5. Experiments results
exp_res <- readRDS(file.path(OUTPUTS, "r_experiments_results.rds"))
exp_df <- data.frame(
  Experiment = names(exp_res),
  QWK        = sapply(exp_res, function(x) x$qwk),
  Accuracy   = sapply(exp_res, function(x) x$acc),
  N_Features = sapply(exp_res, function(x) x$n_features)
)
fwrite(exp_df, file.path(OUTPUTS, "r_experiments_full.csv"))

# 6. Thresholds
thr <- readRDS(file.path(OUTPUTS, "r_best_xgboost_thresholds.rds"))
data.frame(index = seq_along(thr), threshold = thr) |>
  fwrite(file.path(OUTPUTS, "r_best_thresholds.csv"))

cat("Exported all RDS to CSV in", OUTPUTS, "\n")
