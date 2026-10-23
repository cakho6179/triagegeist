# Reproduce R E6 + backward ablations (seed 42, plain-random 80/20).
# Same split for all runs (one idx), same protocol as R/experiments.R.
# Incremental CSV saves to revision/results/r_repro/ (timeout-safe).
suppressMessages({ library(data.table); library(xgboost) })

outdir <- "revision/results/r_repro"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

d <- fread("data/processed/triage_full.csv")
y <- d[["triage_acuity"]]
X_full <- d[, setdiff(names(d), "triage_acuity"), with = FALSE]
for (col in names(X_full)) {
  if (is.character(X_full[[col]])) X_full[[col]] <- as.numeric(factor(X_full[[col]]))
}
all_cols <- names(X_full)

qwk <- function(y_true, y_pred, min.rating = 1, max.rating = 5) {
  y_true <- as.integer(y_true); y_pred <- as.integer(y_pred)
  obs <- max.rating - min.rating + 1
  hist.rate <- sapply(min.rating:max.rating, function(i) mean(y_true == i))
  pred.rate <- sapply(min.rating:max.rating, function(i) mean(y_pred == i))
  expected <- outer(hist.rate, pred.rate)
  weights <- matrix(0, obs, obs)
  for (i in 1:obs) for (j in 1:obs) weights[i, j] <- ((i - j)^2) / ((obs - 1)^2)
  O <- table(factor(y_true, levels = min.rating:max.rating),
             factor(y_pred, levels = min.rating:max.rating)) / length(y_true)
  1 - sum(weights * O) / sum(weights * expected)
}

find_thresholds <- function(raw_pred, y_true) {
  best_thr <- c(0.32, 0.45, 0.58, 0.72); best_qwk <- -Inf
  for (t1 in seq(0.20, 0.40, 0.02)) for (t2 in seq(0.40, 0.55, 0.02))
    for (t3 in seq(0.55, 0.70, 0.02)) for (t4 in seq(0.70, 0.85, 0.02)) {
      if (t1 >= t2 || t2 >= t3 || t3 >= t4) next
      pc <- rep(5L, length(raw_pred))
      pc[raw_pred < t1] <- 1L
      pc[raw_pred >= t1 & raw_pred < t2] <- 2L
      pc[raw_pred >= t2 & raw_pred < t3] <- 3L
      pc[raw_pred >= t3 & raw_pred < t4] <- 4L
      q <- qwk(y_true, pc)
      if (q > best_qwk) { best_qwk <- q; best_thr <- c(t1, t2, t3, t4) }
    }
  list(thresholds = best_thr, qwk = best_qwk)
}

run_case <- function(cols, label, nrounds = 600) {
  t0 <- Sys.time()
  dtrain <- xgb.DMatrix(as.matrix(X_full[idx, ..cols]), label = (y[idx] - 1) / 4)
  model <- xgb.train(list(objective = "reg:logistic", max_depth = 8, eta = 0.04,
                           subsample = 0.85, colsample_bytree = 0.85,
                           min_child_weight = 5, reg_lambda = 2.0, eval_metric = "rmse"),
                     dtrain, nrounds = nrounds, verbose = 0)
  raw <- predict(model, xgb.DMatrix(as.matrix(X_full[-idx, ..cols])))
  yt <- y[-idx]
  thr <- find_thresholds(raw, yt)
  t1 <- thr$thresholds
  pc <- rep(5L, length(raw))
  pc[raw < t1[1]] <- 1L
  pc[raw >= t1[1] & raw < t1[2]] <- 2L
  pc[raw >= t1[2] & raw < t1[3]] <- 3L
  pc[raw >= t1[3] & raw < t1[4]] <- 4L
  row <- data.table(experiment = label, n_feat = length(cols), qwk = round(thr$qwk, 4),
                    acc = round(mean(pc == yt), 4),
                    within1 = round(mean(abs(pc - yt) <= 1), 4),
                    mae = round(mean(abs(pc - yt)), 4),
                    t1 = t1[1], t2 = t1[2], t3 = t1[3], t4 = t1[4],
                    seconds = round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1))
  print(row)
  fwrite(row, file.path(outdir, "ablation_R.csv"), append = file.exists(file.path(outdir, "ablation_R.csv")))
  if (label == "E6_full") {
    cm <- table(factor(yt, levels = 1:5), factor(pc, levels = 1:5))
    fwrite(data.table(as.table(cm)), file.path(outdir, "confusion_E6.csv"))
    saveRDS(list(raw = raw, y = yt, thr = t1, pred = pc), file.path(outdir, "pred_E6.rds"))
  }
  invisible(row)
}

set.seed(42)
idx <- sample(nrow(X_full), nrow(X_full) * 0.8)

drop_pat <- list(
  minus_text   = "^cc_",
  minus_eng    = "^(shock_|qsofa_|mews_|sirs_|spo2_|rr_|hr_|age_|temp_|triage_|ems_|pain_|symp_|is_|high_)",
  minus_comorb = "^hist_",
  minus_vitals = "^(temp|heart_rate|resp_rate|sys_bp|dias_bp|spo2|pain_score|visit_month|day_of_week|arrival_time)$",
  minus_demog  = "^(insurance_|residence_|region_|race_|sex_)",
  minus_admin  = "^(ems_arrival_|episode_|vitals_during_visit_|seen_last_72h_|no_payment_|is_injury_poison_)",
  minus_iv     = "^intervention_iv_fluids$"
)

done <- character(0)
if (file.exists(file.path(outdir, "ablation_R.csv")))
  done <- fread(file.path(outdir, "ablation_R.csv"))$experiment

if (!("E6_full" %in% done)) run_case(all_cols, "E6_full")
for (nm in names(drop_pat)) {
  if (nm %in% done) next
  dc <- grep(drop_pat[[nm]], all_cols, value = TRUE)
  cat(sprintf("%s: dropping %d\n", nm, length(dc)))
  run_case(setdiff(all_cols, dc), nm, nrounds = 500)
}
cat("DONE\n")
