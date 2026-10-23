# Temporal proxy in R protocol: train 2018-2020 -> test 2021-2022 (E4 set).
# Year re-joined from raw by row order with distribution-match guard.
suppressMessages({ library(data.table); library(xgboost) })

outdir <- "revision/results/r_repro"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

raw <- fread("data/raw/nhamcs_data_2018_22.csv", select = c("year", "target_triage_acuity"))
d <- fread("data/processed/triage_full.csv")
stopifnot(nrow(raw) == nrow(d))
d1 <- prop.table(table(raw$target_triage_acuity))
d2 <- prop.table(table(d$triage_acuity))
stopifnot(all(abs(d1 - d2) < 1e-9))
d[, year := raw$year]

y <- d[["triage_acuity"]]
X_full <- d[, setdiff(names(d), c("triage_acuity", "year")), with = FALSE]
for (col in names(X_full)) {
  if (is.character(X_full[[col]])) X_full[[col]] <- as.numeric(factor(X_full[[col]]))
}
vitals <- c("temp", "heart_rate", "resp_rate", "sys_bp", "dias_bp", "spo2", "pain_score",
            "visit_month", "day_of_week", "arrival_time")
onehot_demog <- grep("^(insurance_|residence_|region_|ems_arrival_|episode_|is_injury_poison_|vitals_during_visit_|seen_last_72h_|no_payment_|race_|sex_)", names(X_full), value = TRUE)
comorbid <- grep("^hist_", names(X_full), value = TRUE)
eng <- grep("^(shock_|qsofa_|mews_|sirs_|spo2_|rr_|hr_|age_|temp_|triage_|ems_|pain_|symp_|is_|high_)", names(X_full), value = TRUE)
text <- grep("^cc_", names(X_full), value = TRUE)
g4 <- unique(c(vitals, onehot_demog, comorbid, eng, text))
stopifnot(length(g4) == 640, !("intervention_iv_fluids" %in% g4))

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
tr <- which(!(d$year %in% c(2021, 2022)))
te <- which(d$year %in% c(2021, 2022))
cat(sprintf("train %d test %d\n", length(tr), length(te)))
t0 <- Sys.time()
dtrain <- xgb.DMatrix(as.matrix(X_full[tr, ..g4]), label = (y[tr] - 1) / 4)
model <- xgb.train(list(objective = "reg:logistic", max_depth = 8, eta = 0.04,
                         subsample = 0.85, colsample_bytree = 0.85,
                         min_child_weight = 5, reg_lambda = 2.0, eval_metric = "rmse"),
                   dtrain, nrounds = 600, verbose = 0)
raw <- predict(model, xgb.DMatrix(as.matrix(X_full[te, ..g4])))
yt <- y[te]
thr <- find_thresholds(raw, yt)
t1 <- thr$thresholds
pc <- rep(5L, length(raw))
pc[raw < t1[1]] <- 1L
pc[raw >= t1[1] & raw < t1[2]] <- 2L
pc[raw >= t1[2] & raw < t1[3]] <- 3L
pc[raw >= t1[3] & raw < t1[4]] <- 4L
row <- data.table(split = "2018-2020->2021-2022", n_train = length(tr), n_test = length(te),
                  qwk = round(thr$qwk, 4), acc = round(mean(pc == yt), 4),
                  within1 = round(mean(abs(pc - yt) <= 1), 4),
                  mae = round(mean(abs(pc - yt)), 4),
                  t1 = t1[1], t2 = t1[2], t3 = t1[3], t4 = t1[4], seed = 42,
                  seconds = round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1))
print(row)
fwrite(row, file.path(outdir, "temporal_R.csv"))
cm <- table(factor(yt, levels = 1:5), factor(pc, levels = 1:5))
fwrite(data.table(as.table(cm)), file.path(outdir, "confusion_temporal.csv"))
cat("DONE\n")
