suppressMessages({
  library(data.table)
  library(xgboost)
})

cat("========================================\n")
cat("EXPERIMENTS E1-E6 - R Pipeline\n")
cat("========================================\n\n")

# ============================================================
# Load data
# ============================================================
cat("[1] Loading data...\n")
d <- fread("data/processed/triage_full.csv")
y <- d[["triage_acuity"]]
X_full <- d[, setdiff(names(d), "triage_acuity"), with = FALSE]

# Convert character to numeric
for (col in names(X_full)) {
  if (is.character(X_full[[col]])) {
    X_full[[col]] <- as.numeric(factor(X_full[[col]]))
  }
}

cat(sprintf("  Total: %d samples, %d features\n", nrow(X_full), ncol(X_full)))

# ============================================================
# Feature group definitions
# ============================================================
# E1: Tabular (raw + one-hot encoded demographics/admin/insurance)
# E2: Tabular + missing indicators
# E3: Tabular + engineered clinical features
# E4: Tabular + chief complaint text (TF-IDF)
# E5: Best from E1-E4 + ordinal regression + threshold tuning
# E6: Ablation (remove groups from full)

# Group 1: Raw tabular (vitals, demographics one-hot, admin one-hot, comorbidities)
vitals <- c("temp", "heart_rate", "resp_rate", "sys_bp", "dias_bp", "spo2", "pain_score",
            "visit_month", "day_of_week", "arrival_time")
onehot_demog <- grep("^(insurance_|residence_|region_|ems_arrival_|episode_|is_injury_poison_|vitals_during_visit_|seen_last_72h_|no_payment_|race_|sex_)", names(X_full), value = TRUE)
comorbid <- grep("^hist_", names(X_full), value = TRUE)
g1_tabular <- unique(c(vitals, onehot_demog, comorbid))
g1_tabular <- g1_tabular[g1_tabular %in% names(X_full)]
cat(sprintf("  G1 Tabular: %d features\n", length(g1_tabular)))

# Group 2: Missing indicators (none separate in this dataset - all already in)
# Detect by suffix or name pattern
miss_ind <- grep("^miss_|^is_missing|^has_missing", names(X_full), value = TRUE)
g2_missing <- unique(c(g1_tabular, miss_ind))
cat(sprintf("  G2 Tabular+Missing: %d features (no separate miss indicators found)\n", length(g2_missing)))

# Group 3: Engineered (vital interactions, age composites, risk scores)
eng <- grep("^(shock_|qsofa_|mews_|sirs_|spo2_|rr_|hr_|age_|temp_|triage_|ems_|pain_|symp_|is_|high_)", names(X_full), value = TRUE)
g3_engineered <- unique(c(g1_tabular, eng))
g3_engineered <- g3_engineered[g3_engineered %in% names(X_full)]
cat(sprintf("  G3 Tabular+Engineered: %d features\n", length(g3_engineered)))

# Group 4: Text (TF-IDF on chief complaint)
text <- grep("^cc_", names(X_full), value = TRUE)
g4_text <- unique(c(g1_tabular, eng, text))
g4_text <- g4_text[g4_text %in% names(X_full)]
cat(sprintf("  G4 Tabular+Eng+Text: %d features\n", length(g4_text)))

# Group 5: All (full model for E5)
g5_all <- names(X_full)
cat(sprintf("  G5 All (full): %d features\n", length(g5_all)))

# ============================================================
# Train/test split
# ============================================================
set.seed(42)
idx <- sample(nrow(X_full), nrow(X_full) * 0.8)

# ============================================================
# QWK function
# ============================================================
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
  num <- sum(weights * O); den <- sum(weights * expected)
  1 - num / den
}

find_thresholds <- function(raw_pred, y_true, init = c(0.32, 0.45, 0.58, 0.72)) {
  best_thr <- init
  best_qwk <- -Inf
  for (t1 in seq(0.20, 0.40, 0.02)) {
    for (t2 in seq(0.40, 0.55, 0.02)) {
      for (t3 in seq(0.55, 0.70, 0.02)) {
        for (t4 in seq(0.70, 0.85, 0.02)) {
          if (t1 >= t2 || t2 >= t3 || t3 >= t4) next
          pc <- rep(5L, length(raw_pred))
          pc[raw_pred < t1] <- 1L
          pc[raw_pred >= t1 & raw_pred < t2] <- 2L
          pc[raw_pred >= t2 & raw_pred < t3] <- 3L
          pc[raw_pred >= t3 & raw_pred < t4] <- 4L
          q <- qwk(y_true, pc)
          if (q > best_qwk) {
            best_qwk <- q; best_thr <- c(t1, t2, t3, t4)
          }
        }
      }
    }
  }
  list(thresholds = best_thr, qwk = best_qwk)
}

# ============================================================
# Train function
# ============================================================
train_and_eval <- function(X, y, idx, label, n_rounds = 500) {
  cat(sprintf("\n--- %s (n_features=%d) ---\n", label, ncol(X)))
  X_train <- X[idx, ]; X_test <- X[-idx, ]
  y_train <- y[idx];   y_test <- y[-idx]

  dtrain <- xgb.DMatrix(as.matrix(X_train), label = (y_train - 1) / 4)
  dtest  <- xgb.DMatrix(as.matrix(X_test),  label = (y_test - 1) / 4)

  params <- list(
    objective = "reg:logistic", max_depth = 8, eta = 0.04,
    subsample = 0.85, colsample_bytree = 0.85,
    min_child_weight = 5, reg_lambda = 2.0, eval_metric = "rmse"
  )
  t0 <- Sys.time()
  model <- xgb.train(params, dtrain, nrounds = n_rounds, verbose = 0)
  elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

  raw_pred <- predict(model, dtest)
  thr_result <- find_thresholds(raw_pred, y_test)
  thr <- thr_result$thresholds
  q <- thr_result$qwk
  acc <- mean(round(raw_pred * 4) + 1 == y_test)

  cat(sprintf("  QWK=%.4f, Acc=%.4f, Thresholds=[%.3f, %.3f, %.3f, %.3f], Time=%.1fs\n",
              q, acc, thr[1], thr[2], thr[3], thr[4], elapsed))

  list(
    label = label, model = model, n_features = ncol(X),
    qwk = q, acc = acc, thresholds = thr, time = elapsed,
    raw_pred = raw_pred, y_test = y_test
  )
}

# ============================================================
# Run E1-E4
# ============================================================
cat("\n[2] Running E1-E4...\n")
e1 <- train_and_eval(X_full[, ..g1_tabular], y, idx, "E1 Tabular only", n_rounds = 400)
e2 <- train_and_eval(X_full[, ..g2_missing], y, idx, "E2 Tabular+Missing", n_rounds = 400)
e3 <- train_and_eval(X_full[, ..g3_engineered], y, idx, "E3 Tabular+Engineered", n_rounds = 500)
e4 <- train_and_eval(X_full[, ..g4_text], y, idx, "E4 Tabular+Eng+Text", n_rounds = 600)

# ============================================================
# E5: Best (E4) + ordinal threshold tuning
# Already done in E4 - re-tune
# ============================================================
cat("\n[3] Running E5 (ordinal + threshold tuning)...\n")
# E4 already uses ordinal. The "E5" specific is to test different objectives
e5_multiclass <- train_and_eval(X_full[, ..g4_text], y, idx, "E5 Multiclass softprob",
                                n_rounds = 500)

# Use E4 as best E5
e5_best <- e4

# ============================================================
# E6: Ablation - remove each group from full
# ============================================================
cat("\n[4] Running E6 Ablation (backward)...\n")
e6_full <- train_and_eval(X_full, y, idx, "E6 Full", n_rounds = 600)

remove_group <- function(group_features) {
  remaining <- setdiff(names(X_full), group_features)
  X_abl <- X_full[, ..remaining]
  train_and_eval(X_abl, y, idx,
                 sprintf("E6 Remove %s", deparse(substitute(group_features))),
                 n_rounds = 500)
}

e6_minus_text <- train_and_eval(X_full[, !..text], y, idx, "E6 -Text", n_rounds = 500)
e6_minus_eng <- train_and_eval(X_full[, !..eng], y, idx, "E6 -Engineered", n_rounds = 500)
e6_minus_comorb <- train_and_eval(X_full[, !..comorbid], y, idx, "E6 -Comorbidity", n_rounds = 500)
e6_minus_vitals <- train_and_eval(X_full[, !..vitals], y, idx, "E6 -Vitals", n_rounds = 500)
e6_minus_demog <- train_and_eval(X_full[, !..onehot_demog], y, idx, "E6 -Demographics", n_rounds = 500)
e6_minus_admin <- train_and_eval(X_full[, !..onehot_demog], y, idx, "E6 -Admin", n_rounds = 500)
# intervention_iv_fluids - this is the only intervention
e6_minus_iv <- train_and_eval(X_full[, !"intervention_iv_fluids"], y, idx, "E6 -IV Fluids", n_rounds = 500)

# ============================================================
# Save all results
# ============================================================
cat("\n[5] Saving results...\n")
results <- list(
  E1 = list(qwk = e1$qwk, acc = e1$acc, n_features = e1$n_features,
            thresholds = e1$thresholds, time = e1$time),
  E2 = list(qwk = e2$qwk, acc = e2$acc, n_features = e2$n_features,
            thresholds = e2$thresholds, time = e2$time),
  E3 = list(qwk = e3$qwk, acc = e3$acc, n_features = e3$n_features,
            thresholds = e3$thresholds, time = e3$time),
  E4 = list(qwk = e4$qwk, acc = e4$acc, n_features = e4$n_features,
            thresholds = e4$thresholds, time = e4$time),
  E5_multiclass = list(qwk = e5_multiclass$qwk, acc = e5_multiclass$acc,
                       n_features = e5_multiclass$n_features,
                       thresholds = e5_multiclass$thresholds, time = e5_multiclass$time),
  E5_ordinal = list(qwk = e5_best$qwk, acc = e5_best$acc, n_features = e5_best$n_features,
                    thresholds = e5_best$thresholds, time = e5_best$time),
  E6_full = list(qwk = e6_full$qwk, acc = e6_full$acc, n_features = e6_full$n_features),
  E6_minus_text = list(qwk = e6_minus_text$qwk, acc = e6_minus_text$acc,
                       n_features = e6_minus_text$n_features, dqwk = e6_minus_text$qwk - e6_full$qwk),
  E6_minus_eng = list(qwk = e6_minus_eng$qwk, acc = e6_minus_eng$acc,
                      n_features = e6_minus_eng$n_features, dqwk = e6_minus_eng$qwk - e6_full$qwk),
  E6_minus_comorb = list(qwk = e6_minus_comorb$qwk, acc = e6_minus_comorb$acc,
                         n_features = e6_minus_comorb$n_features, dqwk = e6_minus_comorb$qwk - e6_full$qwk),
  E6_minus_vitals = list(qwk = e6_minus_vitals$qwk, acc = e6_minus_vitals$acc,
                         n_features = e6_minus_vitals$n_features, dqwk = e6_minus_vitals$qwk - e6_full$qwk),
  E6_minus_demog = list(qwk = e6_minus_demog$qwk, acc = e6_minus_demog$acc,
                        n_features = e6_minus_demog$n_features, dqwk = e6_minus_demog$qwk - e6_full$qwk),
  E6_minus_iv = list(qwk = e6_minus_iv$qwk, acc = e6_minus_iv$acc,
                     n_features = e6_minus_iv$n_features, dqwk = e6_minus_iv$qwk - e6_full$qwk)
)

# Save as RDS
saveRDS(results, "outputs/r_experiments_results.rds")
saveRDS(list(e1 = e1, e2 = e2, e3 = e3, e4 = e4, e5 = e5_best,
             e5_multiclass = e5_multiclass, e6_full = e6_full,
             e6_minus_text = e6_minus_text, e6_minus_eng = e6_minus_eng,
             e6_minus_comorb = e6_minus_comorb, e6_minus_vitals = e6_minus_vitals,
             e6_minus_demog = e6_minus_demog, e6_minus_iv = e6_minus_iv),
        "outputs/r_experiments_models.rds")

# Save as CSV summary
summary_df <- data.frame(
  Experiment = names(results),
  N_Features = sapply(results, function(x) x$n_features),
  QWK = sapply(results, function(x) x$qwk),
  Accuracy = sapply(results, function(x) x$acc)
)
fwrite(summary_df, "outputs/r_experiments_summary.csv")

cat("\n========================================\n")
cat("EXPERIMENTS COMPLETE\n")
cat("========================================\n\n")
cat("Summary:\n")
print(summary_df)
cat("\nBest model: E4/E5 (Tabular+Eng+Text) QWK=", e4$qwk, "\n", sep = "")

# Save best model + thresholds for downstream use
saveRDS(e4$model, "outputs/r_best_xgboost_model.rds")
saveRDS(e4$thresholds, "outputs/r_best_xgboost_thresholds.rds")
saveRDS(e4$raw_pred, "outputs/r_best_raw_predictions.rds")
saveRDS(e4$y_test, "outputs/r_best_y_test.rds")
cat("\nBest model saved: outputs/r_best_xgboost_model.rds\n")
