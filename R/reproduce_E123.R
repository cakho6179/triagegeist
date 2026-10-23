# Reproduce R E1-E3 (seed 42, same split as reproduce_E4/E6 for comparability).
# Note: original experiments.R used an independent sample() call per script run;
# here one shared split is used so E1-E6 are directly comparable. Labeled as such.
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

vitals <- c("temp", "heart_rate", "resp_rate", "sys_bp", "dias_bp", "spo2", "pain_score",
            "visit_month", "day_of_week", "arrival_time")
onehot_demog <- grep("^(insurance_|residence_|region_|ems_arrival_|episode_|is_injury_poison_|vitals_during_visit_|seen_last_72h_|no_payment_|race_|sex_)", names(X_full), value = TRUE)
comorbid <- grep("^hist_", names(X_full), value = TRUE)
eng <- grep("^(shock_|qsofa_|mews_|sirs_|spo2_|rr_|hr_|age_|temp_|triage_|ems_|pain_|symp_|is_|high_)", names(X_full), value = TRUE)
text <- grep("^cc_", names(X_full), value = TRUE)
g1 <- unique(c(vitals, onehot_demog, comorbid))

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

run_case <- function(cols, label, nrounds) {
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
                    acc_round = round(mean(round(raw * 4) + 1 == yt), 4),
                    within1 = round(mean(abs(pc - yt) <= 1), 4),
                    mae = round(mean(abs(pc - yt)), 4),
                    t1 = t1[1], t2 = t1[2], t3 = t1[3], t4 = t1[4])
  print(row)
  fwrite(row, file.path(outdir, "progression_R.csv"),
         append = file.exists(file.path(outdir, "progression_R.csv")))
}

set.seed(42)
idx <- sample(nrow(X_full), nrow(X_full) * 0.8)

done <- character(0)
if (file.exists(file.path(outdir, "progression_R.csv")))
  done <- fread(file.path(outdir, "progression_R.csv"))$experiment
if (!("E1" %in% done)) run_case(g1, "E1", 400)
if (!("E2" %in% done)) run_case(g1, "E2", 400)  # no separate miss indicators in this build
if (!("E3" %in% done)) run_case(unique(c(g1, eng)), "E3", 500)
cat("DONE\n")
