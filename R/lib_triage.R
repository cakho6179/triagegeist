# Shared helpers for the supplementary analysis scripts (R/final_*.R).
# Data, feature groups, XGBoost settings and the 80/20 split (set.seed(42); sample) follow R/reproduce_E*.R,
# so the first model fitted in each script reproduces the corresponding locked result.
#
# TRIAGE_SPLIT=random      (default) the split used for the reported results
# TRIAGE_SPLIT=stratified  80% of every ESI level; results go to revision/results/final_stratified
#
# Decision thresholds are reported two ways: fitted on the test predictions ("test_tuned", as in
# R/reproduce_E*.R) and on 5-fold out-of-fold scores of the training rows ("train_tuned").
suppressMessages({ library(data.table); library(xgboost) })

SPLIT <- Sys.getenv("TRIAGE_SPLIT", "random")
stopifnot(SPLIT %in% c("random", "stratified"))
OUTDIR <- if (SPLIT == "random") "revision/results/final" else "revision/results/final_stratified"
dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)

load_triage <- function(path = "data/processed/triage_full.csv") {
  d <- fread(path)
  y <- d[["triage_acuity"]]
  X <- d[, setdiff(names(d), "triage_acuity"), with = FALSE]
  for (col in names(X)) if (is.character(X[[col]])) X[[col]] <- as.numeric(factor(X[[col]]))
  set.seed(42)
  idx <- if (SPLIT == "random") sample(nrow(X), nrow(X) * 0.8) else stratified_train_rows(y)
  list(X = X, y = y, idx = idx)
}

# 80% of every ESI level goes to training; the rest is the test set.
stratified_train_rows <- function(y, frac = 0.8) {
  sort(unlist(lapply(sort(unique(y)), function(k) {
    rows <- which(y == k)
    rows[sample.int(length(rows), round(length(rows) * frac))]
  })))
}

vitals_cols <- c("temp", "heart_rate", "resp_rate", "sys_bp", "dias_bp", "spo2", "pain_score",
                 "visit_month", "day_of_week", "arrival_time")

feature_sets <- function(nm) {
  onehot <- grep("^(insurance_|residence_|region_|ems_arrival_|episode_|is_injury_poison_|vitals_during_visit_|seen_last_72h_|no_payment_|race_|sex_)", nm, value = TRUE)
  comorb <- grep("^hist_", nm, value = TRUE)
  eng <- grep("^(shock_|qsofa_|mews_|sirs_|spo2_|rr_|hr_|age_|temp_|triage_|ems_|pain_|symp_|is_|high_)", nm, value = TRUE)
  text <- grep("^cc_", nm, value = TRUE)
  g1 <- unique(c(vitals_cols, onehot, comorb))
  g3 <- unique(c(g1, eng))
  g4 <- unique(c(g3, text))
  list(g1 = g1, g3 = g3, g4 = g4, onehot = onehot, comorb = comorb, eng = eng, text = text)
}

qwk <- function(y_true, y_pred, k = 5) {
  y_true <- as.integer(y_true); y_pred <- as.integer(y_pred)
  hist.rate <- sapply(1:k, function(i) mean(y_true == i))
  pred.rate <- sapply(1:k, function(i) mean(y_pred == i))
  expected <- outer(hist.rate, pred.rate)
  w <- outer(1:k, 1:k, function(i, j) (i - j)^2 / (k - 1)^2)
  O <- table(factor(y_true, levels = 1:k), factor(y_pred, levels = 1:k)) / length(y_true)
  1 - sum(w * O) / sum(w * expected)
}

decode <- function(raw, t) {
  pc <- rep(5L, length(raw))
  pc[raw < t[1]] <- 1L
  pc[raw >= t[1] & raw < t[2]] <- 2L
  pc[raw >= t[2] & raw < t[3]] <- 3L
  pc[raw >= t[3] & raw < t[4]] <- 4L
  pc
}

# Grid search identical to R/reproduce_E4.R; y_true decides which data the thresholds are fitted on.
find_thresholds <- function(raw_pred, y_true) {
  best_thr <- c(0.32, 0.45, 0.58, 0.72); best_qwk <- -Inf
  for (t1 in seq(0.20, 0.40, 0.02)) for (t2 in seq(0.40, 0.55, 0.02))
    for (t3 in seq(0.55, 0.70, 0.02)) for (t4 in seq(0.70, 0.85, 0.02)) {
      if (t1 >= t2 || t2 >= t3 || t3 >= t4) next
      q <- qwk(y_true, decode(raw_pred, c(t1, t2, t3, t4)))
      if (q > best_qwk) { best_qwk <- q; best_thr <- c(t1, t2, t3, t4) }
    }
  list(thresholds = best_thr, qwk = best_qwk)
}

XGB_ORD <- list(objective = "reg:logistic", max_depth = 8, eta = 0.04, subsample = 0.85,
                colsample_bytree = 0.85, min_child_weight = 5, reg_lambda = 2.0, eval_metric = "rmse")

# xgboost's R interface draws its seed from R's RNG, so a fit depends on what ran before it in the session.
fit_ordinal <- function(X, y, rows, cols, nrounds) {
  dtrain <- xgb.DMatrix(as.matrix(X[rows, ..cols]), label = (y[rows] - 1) / 4)
  xgb.train(XGB_ORD, dtrain, nrounds = nrounds, verbose = 0)
}

predict_ordinal <- function(model, X, rows, cols) {
  predict(model, xgb.DMatrix(as.matrix(X[rows, ..cols])))
}

# Out-of-fold scores on the training rows (5 folds), for thresholds that never see the test set.
oof_scores <- function(X, y, train_rows, cols, nrounds, k = 5, seed = 42) {
  set.seed(seed)
  fold <- sample(rep(1:k, length.out = length(train_rows)))
  oof <- numeric(length(train_rows))
  for (f in 1:k) {
    tr <- train_rows[fold != f]; va <- fold == f
    m <- fit_ordinal(X, y, tr, cols, nrounds)
    oof[va] <- predict_ordinal(m, X, train_rows[va], cols)
  }
  oof
}

metrics_row <- function(label, n_feat, yt, pc, extra = list()) {
  as.data.table(c(list(model = label, n_feat = n_feat, qwk = round(qwk(yt, pc), 4),
                       acc = round(mean(pc == yt), 4), within1 = round(mean(abs(pc - yt) <= 1), 4),
                       mae = round(mean(abs(pc - yt)), 4)), extra))
}

save_pred <- function(name, idx_test, yt, pc, raw = NULL) {
  dt <- data.table(row = idx_test, y = yt, pred = pc)
  if (!is.null(raw)) dt[, raw := raw]
  fwrite(dt, file.path(OUTDIR, paste0("pred_", name, ".csv")))
}

append_csv <- function(row, file) {
  f <- file.path(OUTDIR, file)
  fwrite(row, f, append = file.exists(f))
}

# Score one set of raw ordinal predictions under both threshold protocols and save the predictions.
report_ordinal <- function(label, n_feat, raw, yt, te, thr_train = NULL, file) {
  th_test <- find_thresholds(raw, yt)$thresholds
  protos <- list(test_tuned = th_test)
  if (!is.null(thr_train)) protos$train_tuned <- thr_train
  for (p in names(protos)) {
    th <- protos[[p]]
    pc <- decode(raw, th)
    append_csv(metrics_row(label, n_feat, yt, pc,
                           list(protocol = p, t1 = th[1], t2 = th[2], t3 = th[3], t4 = th[4])), file)
    save_pred(paste0(label, "_", p), te, yt, pc, raw)
  }
}
