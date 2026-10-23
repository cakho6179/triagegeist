# Reproduce original R E4 run (seed 42, plain-random 80/20, 600 rounds).
# Faithful port of R/experiments.R E4 path. Saves to revision/results/r_repro/.
suppressMessages({ library(data.table); library(xgboost) })

RSCRIPT <- "C:/Program Files/R/R-4.6.0/bin/Rscript.exe"  # informational only
outdir <- "revision/results/r_repro"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

cat("[1] Loading data...\n")
d <- fread("data/processed/triage_full.csv")
y <- d[["triage_acuity"]]
X_full <- d[, setdiff(names(d), "triage_acuity"), with = FALSE]
for (col in names(X_full)) {
  if (is.character(X_full[[col]])) X_full[[col]] <- as.numeric(factor(X_full[[col]]))
}

vitals <- c("temp", "heart_rate", "resp_rate", "sys_bp", "dias_bp", "spo2", "pain_score",
            "visit_month", "day_of_week", "arrival_time")
onehot_demog <- grep("^(insurance_|residence_|region_|ems_arrival_|episode_|is_injury_poison_|vitals_during_visit_|seen_last_72h_|no_payment_|race_|sex_)", names(X_full), value = TRUE)
eng <- grep("^(shock_|qsofa_|mews_|sirs_|spo2_|rr_|hr_|age_|temp_|triage_|ems_|pain_|symp_|is_|high_)", names(X_full), value = TRUE)
text <- grep("^cc_", names(X_full), value = TRUE)
comorbid <- grep("^hist_", names(X_full), value = TRUE)
g4 <- unique(c(vitals, onehot_demog, comorbid, eng, text))
g4 <- g4[g4 %in% names(X_full)]
cat(sprintf("E4 features: %d (expect 640)\n", length(g4)))
stopifnot(!("intervention_iv_fluids" %in% g4))

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

set.seed(42)
idx <- sample(nrow(X_full), nrow(X_full) * 0.8)
X <- X_full[, ..g4]
X_train <- X[idx, ]; X_test <- X[-idx, ]
y_train <- y[idx];   y_test <- y[-idx]

dtrain <- xgb.DMatrix(as.matrix(X_train), label = (y_train - 1) / 4)
params <- list(objective = "reg:logistic", max_depth = 8, eta = 0.04,
               subsample = 0.85, colsample_bytree = 0.85,
               min_child_weight = 5, reg_lambda = 2.0, eval_metric = "rmse")
t0 <- Sys.time()
model <- xgb.train(params, dtrain, nrounds = 600, verbose = 0)
cat(sprintf("trained in %.0fs\n", as.numeric(difftime(Sys.time(), t0, units = "secs"))))

raw_pred <- predict(model, xgb.DMatrix(as.matrix(X_test)))
thr <- find_thresholds(raw_pred, y_test)
t1 <- thr$thresholds
pc <- rep(5L, length(raw_pred))
pc[raw_pred < t1[1]] <- 1L
pc[raw_pred >= t1[1] & raw_pred < t1[2]] <- 2L
pc[raw_pred >= t1[2] & raw_pred < t1[3]] <- 3L
pc[raw_pred >= t1[3] & raw_pred < t1[4]] <- 4L
acc_round <- mean(round(raw_pred * 4) + 1 == y_test)  # original R accuracy definition
acc_thr <- mean(pc == y_test)
within1 <- mean(abs(pc - y_test) <= 1)
mae <- mean(abs(pc - y_test))
cat(sprintf("QWK=%.4f acc_round=%.4f acc_thr=%.4f within1=%.4f MAE=%.4f thr=[%.2f %.2f %.2f %.2f]\n",
            thr$qwk, acc_round, acc_thr, within1, mae, t1[1], t1[2], t1[3], t1[4]))

cm <- table(factor(y_test, levels = 1:5), factor(pc, levels = 1:5))
fwrite(data.table(as.table(cm)), file.path(outdir, "confusion_E4.csv"))
saveRDS(list(raw = raw_pred, y = y_test, thr = t1, pred = pc), file.path(outdir, "pred_E4.rds"))
fwrite(data.table(experiment = "E4", n_feat = length(g4), qwk = round(thr$qwk, 4),
                  acc_round = round(acc_round, 4), acc_thr = round(acc_thr, 4),
                  within1 = round(within1, 4), mae = round(mae, 4),
                  t1 = t1[1], t2 = t1[2], t3 = t1[3], t4 = t1[4], seed = 42),
       file.path(outdir, "metrics_E4.csv"))
cat("saved to", outdir, "\n")
