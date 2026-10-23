# True multiclass baseline on the IDENTICAL R split (same seed-42 idx family).
# Note: fresh sample() draw with set.seed(42) reproduces the same idx as the
# other reproduce_*.R scripts (same R version, same call order position 1).
suppressMessages({ library(data.table); library(xgboost) })

outdir <- "revision/results/r_repro"
d <- fread("data/processed/triage_full.csv")
y <- d[["triage_acuity"]]
X_full <- d[, setdiff(names(d), "triage_acuity"), with = FALSE]
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

set.seed(42)
idx <- sample(nrow(X_full), nrow(X_full) * 0.8)
dtrain <- xgb.DMatrix(as.matrix(X_full[idx, ..g4]), label = y[idx] - 1)
dtest <- xgb.DMatrix(as.matrix(X_full[-idx, ..g4]))
t0 <- Sys.time()
model <- xgb.train(list(objective = "multi:softprob", num_class = 5, max_depth = 8,
                         eta = 0.04, subsample = 0.85, colsample_bytree = 0.85,
                         min_child_weight = 5, reg_lambda = 2.0, eval_metric = "mlogloss"),
                   dtrain, nrounds = 500, verbose = 0)
prob <- predict(model, dtest)
yt <- y[-idx]
pred <- max.col(matrix(prob, nrow = length(yt), ncol = 5, byrow = FALSE))  # class-major layout
yt <- y[-idx]
row <- data.table(experiment = "E5mc", n_feat = 640, qwk = round(qwk(yt, pred), 4),
                  acc = round(mean(pred == yt), 4),
                  within1 = round(mean(abs(pred - yt) <= 1), 4),
                  mae = round(mean(abs(pred - yt)), 4),
                  seconds = round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1))
print(row)
fwrite(row, file.path(outdir, "progression_R.csv"), append = TRUE)
cm <- table(factor(yt, levels = 1:5), factor(pred, levels = 1:5))
fwrite(data.table(as.table(cm)), file.path(outdir, "confusion_E5mc.csv"))
cat("DONE\n")
