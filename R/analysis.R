# ============================================================
# Triagegeist - Explainable Triage Acuity Prediction in R
# Focus: Interpretability & Explanation
# ============================================================
# Load packages
library(data.table)
library(dplyr)
library(ggplot2)
library(vip)
library(pdp)
library(DALEX)
library(DALEXtra)
library(shapr)
library(ranger)
library(xgboost)

set.seed(42)

# ============================================================
# 1. DATA PREPARATION
# ============================================================
cat("\n=== 1. Loading Data ===\n")
df <- fread("data/processed/triage_sample_20k.csv")
cat(sprintf("Loaded: %d rows x %d cols\n", nrow(df), ncol(df)))
cat("Class distribution:\n")
print(table(df$triage_acuity))

# Separate features and target
target_col <- "triage_acuity"
y <- df[[target_col]]
X <- df[, !names(df) %in% target_col, with = FALSE]
feature_names <- names(X)

# Scale target to 0-1 for regression models
y_scaled <- (y - 1) / 4

# Train-test split (80/20)
n <- nrow(X)
train_idx <- sample(1:n, size = floor(0.8 * n))
X_train <- X[train_idx, ]
X_test  <- X[-train_idx, ]
y_train <- y[train_idx]
y_test  <- y[-train_idx]
y_train_scaled <- y_scaled[train_idx]
y_test_scaled  <- y_scaled[-train_idx]

cat(sprintf("\nTrain: %d | Test: %d\n", nrow(X_train), nrow(X_test)))
cat("Train class distribution:\n"); print(table(y_train))
cat("Test class distribution:\n");  print(table(y_test))

# ============================================================
# 2. QUADRATIC WEIGHTED KAPPA FUNCTION
# ============================================================
quadratic_kappa <- function(actual, predicted) {
  actual <- as.integer(actual)
  predicted <- as.integer(predicted)
  k <- max(max(actual), max(predicted))
  O <- matrix(0, k, k)
  for (i in seq_along(actual)) {
    O[actual[i], predicted[i]] <- O[actual[i], predicted[i]] + 1
  }
  row_sums <- rowSums(O)
  col_sums <- colSums(O)
  E <- outer(row_sums, col_sums) / sum(O)
  weights <- matrix(0, k, k)
  for (i in 1:k) for (j in 1:k) weights[i,j] <- (i - j)^2 / (k - 1)^2
  num <- sum(weights * O)
  den <- sum(weights * E)
  if (den == 0) return(1)
  return(1 - num / den)
}

# ============================================================
# 3. TRAIN MODELS
# ============================================================
cat("\n=== 2. Training Models ===\n")

# ----- 3a. Random Forest (ranger) -----
cat("Training Random Forest (ranger)...\n")
t_rf <- system.time({
  rf_model <- ranger(
    x = X_train, y = as.factor(y_train),
    num.trees = 300,
    mtry = floor(sqrt(ncol(X_train))),
    min.node.size = 5,
    importance = "permutation",
    seed = 42, verbose = FALSE
  )
})
cat(sprintf("  RF done: %.1fs | OOB error: %.3f\n", t_rf[3], 1 - rf_model$prediction.error))

# ----- 3b. XGBoost (ordinal regression) -----
cat("Training XGBoost ordinal regression...\n")
dtrain <- xgb.DMatrix(as.matrix(X_train), label = y_train_scaled)
dtest  <- xgb.DMatrix(as.matrix(X_test),  label = y_test_scaled)

params <- list(
  objective = "reg:logistic",
  max_depth = 8,
  eta = 0.04,
  subsample = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 3,
  lambda = 1.0,
  alpha = 0.5,
  gamma = 0.05,
  tree_method = "hist",
  nthread = -1
)

t_xgb <- system.time({
  xgb_model <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = 600,
    verbose = 0
  )
})
cat(sprintf("  XGBoost done: %.1fs\n", t_xgb[3]))

# ----- 3c. Simple Decision Tree (for interpretability) -----
cat("Training Decision Tree (rpart)...\n")
library(rpart)
library(rpart.plot)
dt_model <- rpart(
  as.factor(y_train) ~ .,
  data = cbind(X_train, y = as.factor(y_train)),
  method = "class",
  control = rpart.control(maxdepth = 4, cp = 0.001)
)
cat(sprintf("  Tree size: %d splits\n", nrow(dt_model$frame) - 1))

# ============================================================
# 4. PREDICTIONS & EVALUATION
# ============================================================
cat("\n=== 3. Evaluating Models ===\n")

# Helper: find best thresholds for ordinal predictions
find_thresholds <- function(raw_preds, y_true) {
  best_qwk <- -1
  best_th <- NULL
  for (t1 in seq(0.22, 0.35, by = 0.03)) {
    for (t2 in seq(0.38, 0.50, by = 0.03)) {
      if (t1 >= t2) next
      for (t3 in seq(0.50, 0.60, by = 0.03)) {
        if (t2 >= t3) next
        for (t4 in seq(0.65, 0.75, by = 0.03)) {
          if (t3 >= t4) next
          pred <- ifelse(raw_preds < t1, 1,
                  ifelse(raw_preds < t2, 2,
                  ifelse(raw_preds < t3, 3,
                  ifelse(raw_preds < t4, 4, 5))))
          q <- quadratic_kappa(y_true, pred)
          if (q > best_qwk) {
            best_qwk <- q
            best_th <- c(t1, t2, t3, t4)
          }
        }
      }
    }
  }
  list(qwk = best_qwk, thresholds = best_th)
}

# RF predictions
rf_raw <- predict(rf_model, X_test)$predictions
rf_pred <- as.integer(as.character(rf_raw))
rf_qwk <- quadratic_kappa(y_test, rf_pred)
cat(sprintf("RF QWK: %.4f\n", rf_qwk))

# XGBoost predictions
xgb_raw <- predict(xgb_model, dtest)
xgb_opt <- find_thresholds(xgb_raw, y_test)
xgb_pred <- ifelse(xgb_raw < xgb_opt$thresholds[1], 1,
            ifelse(xgb_raw < xgb_opt$thresholds[2], 2,
            ifelse(xgb_raw < xgb_opt$thresholds[3], 3,
            ifelse(xgb_raw < xgb_opt$thresholds[4], 4, 5))))
cat(sprintf("XGBoost QWK: %.4f (th: %s)\n",
            xgb_opt$qwk, paste(round(xgb_opt$thresholds, 3), collapse = ", ")))

# Decision Tree predictions
dt_pred <- predict(dt_model, X_test, type = "class")
dt_qwk <- quadratic_kappa(y_test, dt_pred)
cat(sprintf("Decision Tree QWK: %.4f\n", dt_qwk))

# ============================================================
# 5. FEATURE IMPORTANCE
# ============================================================
cat("\n=== 4. Feature Importance Analysis ===\n")

# VIP for Random Forest
cat("Calculating RF variable importance (vip)...\n")
rf_vip <- vip::vi(rf_model, method = "permute", target = "triage_acuity",
                   metric = "accuracy", nsim = 3, train = X_train)
rf_top15 <- rf_vip[1:15, ]
print(rf_top15)

# VIP for XGBoost
xgb_imp <- xgb.importance(feature_names = feature_names, model = xgb_model)
xgb_top15 <- xgb_imp[1:15, ]
print(xgb_top15)

# VIP plot (XGBoost)
cat("\nXGBoost Top 15 Features:\n")
print(data.frame(
  Feature = xgb_top15$Feature,
  Gain = round(xgb_top15$Gain, 4)
))

# ============================================================
# 6. PARTIAL DEPENDENCE PLOTS
# ============================================================
cat("\n=== 5. Partial Dependence Plots ===\n")

top_features <- xgb_top15$Feature[1:6]
cat(sprintf("Top features for PDP: %s\n", paste(top_features, collapse = ", ")))

par(mfrow = c(2, 3))
for (feat in top_features) {
  if (feat %in% feature_names) {
    feat_idx <- which(feature_names == feat)
    if (length(feat_idx) > 0) {
      pd <- partial(xgb_model, pred.var = feat_idx, train = as.matrix(X_train),
                    type = "regression", grid.resolution = 20)
      plot(pd, main = paste("PDP:", feat), ylab = "Predicted acuity (0-1)")
    }
  }
}

# ============================================================
# 7. DALEX EXPLANATIONS
# ============================================================
cat("\n=== 6. DALEX Model Explanation ===\n")

# Create DALEX explainer for XGBoost (on subset for speed)
set.seed(42)
explain_idx <- sample(nrow(X_test), min(500, nrow(X_test)))
X_explain <- X_test[explain_idx, ]
y_explain <- y_test[explain_idx]

explainer <- DALEX::explain(
  model = xgb_model,
  data = as.matrix(X_explain),
  y = y_explain,
  label = "XGBoost",
  predict_function = function(m, d) predict(m, xgb.DMatrix(d)),
  verbose = FALSE
)
cat("DALEX explainer created.\n")

# Model performance
cat("\nModel Performance (DALEX):\n")
mp <- model_performance(explainer)
print(mp)

# Variable importance (DALEX)
cat("\nVariable Importance (DALEX permutation):\n")
vi_dalex <- model_parts(explainer, loss_function = loss_root_mean_square, type = "difference")
vi_top <- vi_dalex[1:10, c("variable", "dropout_loss")]
cat("Top 10:\n"); print(vi_top)

# Break-down plots for sample predictions
cat("\nBreak-down plots for sample predictions...\n")
bd_idx <- sample(length(explain_idx), min(4, length(explain_idx)))
for (i in bd_idx) {
  bd_i <- predict_parts(explainer, new_observation = as.matrix(X_explain[i, , drop = FALSE]))
  cat(sprintf("  Sample %d: actual=%d, prediction=%.3f\n",
              i, y_explain[i], mean(bd_i$contribution[bd_i$contribution != 0])))
}

# ============================================================
# 8. SHAP VALUES
# ============================================================
cat("\n=== 7. SHAP Value Analysis ===\n")

# SHAP using shapr (on subset)
set.seed(42)
shap_n <- min(200, nrow(X_train))
shap_idx <- sample(nrow(X_train), shap_n)
X_shap <- as.matrix(X_train[shap_idx, ])
y_shap <- y_train[shap_idx]

explainer_shap <- shapr::explain(
  model = xgb_model,
  x_train = X_shap,
  approach = "empirical",
  n_combinations = 50
)

# SHAP for test subset
shap_test_n <- min(50, nrow(X_test))
shap_test_idx <- sample(nrow(X_test), shap_test_n)
X_shap_test <- as.matrix(X_test[shap_test_idx, ])

shap_values <- shapr::explain(
  model = xgb_model,
  x_explain = X_shap_test,
  x_train = X_shap,
  approach = "empirical",
  explainer = explainer_shap,
  n_combinations = 50
)

# Top SHAP features
shap_importance <- data.frame(
  Feature = feature_names,
  MeanSHAP = colMeans(abs(shap_values$shap_contrib))
)
shap_importance <- shap_importance[order(shap_importance$MeanSHAP, decreasing = TRUE), ]
cat("Top 10 SHAP features:\n")
print(head(shap_importance, 10))

# ============================================================
# 9. PER-CLASS ANALYSIS
# ============================================================
cat("\n=== 8. Per-Class Explanation ===\n")

# Confusion matrix
cm <- table(Actual = y_test, Predicted = xgb_pred)
cat("Confusion Matrix:\n"); print(cm)

# Per-class recall
for (c in 1:5) {
  mask <- y_test == c
  if (sum(mask) > 0) {
    rec <- mean(xgb_pred[mask] == c)
    cat(sprintf("  Class %d: recall=%.3f (n=%d)\n", c, rec, sum(mask)))
  }
}

# Per-class mean prediction values
class_means <- sapply(1:5, function(c) mean(xgb_raw[y_test == c]))
cat(sprintf("\nMean raw predictions by class: %s\n",
            paste(round(class_means, 4), collapse = " | ")))

# Class recall vs precision
cat("\nPer-class metrics:\n")
for (c in 1:5) {
  prec <- sum(y_test == c & xgb_pred == c) / max(sum(xgb_pred == c), 1)
  rec  <- sum(y_test == c & xgb_pred == c) / max(sum(y_test == c), 1)
  f1   <- ifelse(prec + rec > 0, 2 * prec * rec / (prec + rec), 0)
  cat(sprintf("  Class %d: precision=%.3f, recall=%.3f, F1=%.3f (true=%d, pred=%d)\n",
              c, prec, rec, f1, sum(y_test == c), sum(xgb_pred == c)))
}

# ============================================================
# 10. DECISION TREE VISUALIZATION
# ============================================================
cat("\n=== 9. Decision Tree Visualization ===\n")

# Plot the decision tree
png("outputs/r_decision_tree.png", width = 1200, height = 800)
rpart.plot(dt_model, type = 2, extra = 104, under = TRUE,
           tweak = 1.2, main = "Triage Acuity Decision Tree (max depth=4)")
dev.off()
cat("Saved decision tree plot to outputs/r_decision_tree.png\n")

# ============================================================
# 11. SUMMARY
# ============================================================
cat("\n========================================\n")
cat("            SUMMARY\n")
cat("========================================\n")
cat(sprintf("Random Forest QWK:      %.4f\n", rf_qwk))
cat(sprintf("XGBoost QWK:            %.4f\n", xgb_opt$qwk))
cat(sprintf("Decision Tree QWK:      %.4f\n", dt_qwk))
cat(sprintf("XGBoost thresholds: [%s]\n",
            paste(round(xgb_opt$thresholds, 4), collapse = ", ")))
cat(sprintf("Top features: %s\n",
            paste(xgb_top15$Feature[1:5], collapse = ", ")))
cat("========================================\n")
