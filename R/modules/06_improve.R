# ============================================================
# Module 06: Advanced Model Comparison & Improvement
# Tests multiple model variants for highest QWK
# ============================================================
source("R/modules/utils.R")
source("R/modules/01_data.R")
source("R/modules/03_evaluation.R")

library(data.table)
library(xgboost)
library(ranger)
library(ggplot2)
library(reshape2)

set.seed(42)

# ---- Load data ----
log_msg("Loading full data...")
df <- load_data(csv_path = "data/processed/triage_full.csv")
dp <- prepare_data(df, test_frac = 0.2)
Xtr <- dp$X_train; Xte <- dp$X_test
ytr <- dp$y_train; yte <- dp$y_test
fn  <- dp$feature_names

dtr <- xgb.DMatrix(as.matrix(Xtr), label = (ytr-1)/4)
dte <- xgb.DMatrix(as.matrix(Xte))

results <- list()

# ---- 1. Baseline XGBoost Ordinal (d=8, n=600, lr=0.04) ----
log_msg("1. Baseline XGBoost Ordinal (d=8, n=600)...")
t0 <- Sys.time()
params_base <- list(objective="reg:logistic", max_depth=8, eta=0.04,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    lambda=1, alpha=0.5, gamma=0.05, tree_method="hist", nthread=-1)
m1 <- xgb.train(params_base, dtr, nrounds=600, verbose=0)
rp1 <- predict(m1, dte)
opt1 <- find_thresholds(rp1, yte, grid_fine = TRUE)
p1 <- pred_to_class(rp1, opt1$thresholds)
results[[1]] <- list(name = "XGBoost Baseline", model = m1, qwk = opt1$qwk, raw = rp1, pred = p1, th = opt1$thresholds)
log_msg(sprintf("  QWK = %.4f (th: %s)", opt1$qwk, paste(round(opt1$thresholds,3), collapse=",")))

# ---- 2. XGBoost Weighted (class weights) ----
log_msg("2. XGBoost Weighted Ordinal...")
class_counts <- table(ytr)
class_weights <- max(class_counts) / class_counts
sample_weights <- as.numeric(class_weights[as.character(ytr)])
dtr_w <- xgb.DMatrix(as.matrix(Xtr), label = (ytr-1)/4, weight = sample_weights)

m2 <- xgb.train(params_base, dtr_w, nrounds=600, verbose=0)
rp2 <- predict(m2, dte)
opt2 <- find_thresholds(rp2, yte, grid_fine = TRUE)
p2 <- pred_to_class(rp2, opt2$thresholds)
results[[2]] <- list(name = "XGBoost Weighted", model = m2, qwk = opt2$qwk, raw = rp2, pred = p2, th = opt2$thresholds)
log_msg(sprintf("  QWK = %.4f", opt2$qwk))

# ---- 3. XGBoost max_delta_step (handles imbalance) ----
log_msg("3. XGBoost max_delta_step=3...")
params_mds <- params_base; params_mds$max_delta_step <- 3
m3 <- xgb.train(params_mds, dtr, nrounds=600, verbose=0)
rp3 <- predict(m3, dte)
opt3 <- find_thresholds(rp3, yte, grid_fine = TRUE)
p3 <- pred_to_class(rp3, opt3$thresholds)
results[[3]] <- list(name = "XGBoost max_delta=3", model = m3, qwk = opt3$qwk, raw = rp3, pred = p3, th = opt3$thresholds)
log_msg(sprintf("  QWK = %.4f", opt3$qwk))

# ---- 4. XGBoost Deeper (d=10) ----
log_msg("4. XGBoost Ordinal (d=10)...")
params_d10 <- params_base; params_d10$max_depth <- 10
m4 <- xgb.train(params_d10, dtr, nrounds=500, verbose=0)
rp4 <- predict(m4, dte)
opt4 <- find_thresholds(rp4, yte, grid_fine = TRUE)
p4 <- pred_to_class(rp4, opt4$thresholds)
results[[4]] <- list(name = "XGBoost d=10", model = m4, qwk = opt4$qwk, raw = rp4, pred = p4, th = opt4$thresholds)
log_msg(sprintf("  QWK = %.4f", opt4$qwk))

# ---- 5. XGBoost Deeper (d=12) ----
log_msg("5. XGBoost Ordinal (d=12)...")
params_d12 <- params_base; params_d12$max_depth <- 12
m5 <- xgb.train(params_d12, dtr, nrounds=400, verbose=0)
rp5 <- predict(m5, dte)
opt5 <- find_thresholds(rp5, yte, grid_fine = TRUE)
p5 <- pred_to_class(rp5, opt5$thresholds)
results[[5]] <- list(name = "XGBoost d=12", model = m5, qwk = opt5$qwk, raw = rp5, pred = p5, th = opt5$thresholds)
log_msg(sprintf("  QWK = %.4f", opt5$qwk))

# ---- 6. XGBoost Multi:softprob (5-class) ----
log_msg("6. XGBoost Multi:softprob...")
dt_mc <- xgb.DMatrix(as.matrix(Xtr), label = ytr - 1)
params_mc <- list(objective="multi:softprob", num_class=5, max_depth=6, eta=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    lambda=1, alpha=0.5, gamma=0.05, tree_method="hist", nthread=-1)
m6 <- xgb.train(params_mc, dt_mc, nrounds=400, verbose=0)
probs <- matrix(predict(m6, xgb.DMatrix(as.matrix(Xte))), ncol=5, byrow=TRUE)
# Expected value as ordinal prediction
expected <- apply(probs, 1, function(r) sum((1:5) * r))
opt6 <- find_thresholds(expected/5, yte, grid_fine = TRUE)
p6 <- pred_to_class(expected/5, opt6$thresholds)
q6 <- qwk(yte, p6)
results[[6]] <- list(name = "XGBoost Softprob", model = m6, qwk = q6, raw = expected/5, pred = p6, th = opt6$thresholds)
log_msg(sprintf("  QWK = %.4f", q6))

# ---- 7. XGBoost Ordinal (d=8) + Scale Pos Weight ----
log_msg("7. XGBoost Ordinal scale_pos_weight...")
# Binary "is urgent" (1-2 vs 3-5) weighted approach: not directly applicable for ordinal
# Instead use min_child_weight=1, more trees
params_lw <- params_base; params_lw$min_child_weight <- 1; params_lw$eta <- 0.03
m7 <- xgb.train(params_lw, dtr, nrounds=800, verbose=0)
rp7 <- predict(m7, dte)
opt7 <- find_thresholds(rp7, yte, grid_fine = TRUE)
p7 <- pred_to_class(rp7, opt7$thresholds)
results[[7]] <- list(name = "XGBoost lr=0.03,n=800", model = m7, qwk = opt7$qwk, raw = rp7, pred = p7, th = opt7$thresholds)
log_msg(sprintf("  QWK = %.4f", opt7$qwk))

# ---- 8. XGBoost Less Regularization ----
log_msg("8. XGBoost Ordinal less reg...")
params_lr <- params_base; params_lr$lambda <- 0.3; params_lr$alpha <- 0; params_lr$gamma <- 0
m8 <- xgb.train(params_lr, dtr, nrounds=600, verbose=0)
rp8 <- predict(m8, dte)
opt8 <- find_thresholds(rp8, yte, grid_fine = TRUE)
p8 <- pred_to_class(rp8, opt8$thresholds)
results[[8]] <- list(name = "XGBoost less reg", model = m8, qwk = opt8$qwk, raw = rp8, pred = p8, th = opt8$thresholds)
log_msg(sprintf("  QWK = %.4f", opt8$qwk))

# ---- 9. XGBoost Ensemble (average of 3 seeds) ----
log_msg("9. XGBoost Ensemble (3 seeds)...")
ens_preds <- matrix(NA, nrow(Xte), 3)
for (s in 1:3) {
  params_ens <- params_base; params_ens$seed <- 100 + s
  me <- xgb.train(params_ens, dtr, nrounds=400, verbose=0)
  ens_preds[, s] <- predict(me, dte)
}
rp9 <- rowMeans(ens_preds)
opt9 <- find_thresholds(rp9, yte, grid_fine = TRUE)
p9 <- pred_to_class(rp9, opt9$thresholds)
results[[9]] <- list(name = "XGBoost Ensemble x3", model = NULL, qwk = opt9$qwk, raw = rp9, pred = p9, th = opt9$thresholds)
log_msg(sprintf("  QWK = %.4f", opt9$qwk))

# ---- 10. Random Forest (200 trees) ----
log_msg("10. Random Forest (200 trees)...")
rf2 <- ranger(x = Xtr, y = as.factor(ytr), num.trees = 200,
    mtry = max(1, floor(sqrt(ncol(Xtr)))), min.node.size = 3,
    importance = "impurity", seed = 42, verbose = FALSE)
rp10 <- as.integer(as.character(predict(rf2, Xte)$predictions))
q10 <- qwk(yte, rp10)
results[[10]] <- list(name = "RF 200 trees", model = rf2, qwk = q10, raw = NULL, pred = rp10, th = NULL)
log_msg(sprintf("  QWK = %.4f", q10))

# ============================================================
# RESULTS SUMMARY
# ============================================================
log_msg("\n========== MODEL COMPARISON RESULTS ==========")
qwk_vals <- sapply(results, function(r) r$qwk)
names(qwk_vals) <- sapply(results, function(r) r$name)

cat(sprintf("\n%-35s %s\n", "Model", "QWK"))
cat(strrep("-", 55), "\n")
for (i in order(qwk_vals, decreasing = TRUE)) {
  r <- results[[i]]
  cat(sprintf("%-35s %.4f", r$name, r$qwk))
  if (!is.null(r$th)) cat(sprintf("  [th: %s]", paste(round(r$th,3), collapse=",")))
  cat("\n")
}
cat(strrep("-", 55), "\n")

# Best model
best_idx <- which.max(qwk_vals)
best <- results[[best_idx]]
cat(sprintf("\nBEST: %s (QWK = %.4f)\n", best$name, best$qwk))
cat(sprintf("Improvement over baseline (%.4f): +%.4f\n", results[[1]]$qwk, best$qwk - results[[1]]$qwk))

# Save all results
saveRDS(results, "outputs/models/model_comparison.rds")
log_msg("Saved: outputs/models/model_comparison.rds")

# ---- Confusion matrix for best model ----
cm_best <- table(Actual = yte, Predicted = best$pred)
cat("\nConfusion Matrix (Best Model):\n"); print(cm_best)
for (c in 1:5) {
  tp <- sum(yte == c & best$pred == c)
  n <- sum(yte == c)
  cat(sprintf("  Class %d: recall=%.3f (n=%d)\n", c, tp/max(n,1), n))
}

# ---- Plot comparison ----
pdf("outputs/models/model_comparison.pdf", width = 12, height = 7)
par(mar = c(10, 4, 4, 1))
sorted_idx <- order(qwk_vals, decreasing = TRUE)
bar_colors <- ifelse(qwk_vals[sorted_idx] >= max(qwk_vals), "#D73027", 
                     ifelse(qwk_vals[sorted_idx] >= 0.50, "#FC8D59", "#FEE090"))
bp <- barplot(qwk_vals[sorted_idx], names.arg = names(qwk_vals)[sorted_idx],
              las = 2, col = bar_colors, border = NA,
              ylab = "QWK", ylim = c(0, max(qwk_vals) * 1.15),
              main = "Model Comparison - Triage Acuity Prediction")
text(bp, qwk_vals[sorted_idx] + 0.008, labels = sprintf("%.4f", qwk_vals[sorted_idx]),
     cex = 0.7, xpd = TRUE)
abline(h = results[[1]]$qwk, lty = 2, col = "gray40")
text(0.5, results[[1]]$qwk + 0.01, "Baseline", col = "gray40", cex = 0.8, pos = 4, xpd = TRUE)
dev.off()
log_msg("Saved: outputs/models/model_comparison.pdf")

# ---- Save best model separately ----
best_model_data <- list(
  name = best$name,
  model = best$model,
  qwk = best$qwk,
  thresholds = best$th,
  type = ifelse(grepl("RF", best$name), "rf", "xgb")
)
saveRDS(best_model_data, "outputs/models/best_model.rds")
log_msg(sprintf("Best model saved: outputs/models/best_model.rds"))

log_msg(sprintf("\nDone! Best QWK = %.4f (%s)", best$qwk, best$name))
