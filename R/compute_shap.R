suppressMessages({
  library(data.table)
  library(xgboost)
  library(kernelshap)
  library(ggplot2)
})

cat("[1] Loading data...\n")
d <- fread("data/processed/triage_full.csv")
y <- d[["triage_acuity"]]
X <- d[, setdiff(names(d), "triage_acuity"), with = FALSE]
for (col in names(X)) {
  if (is.character(X[[col]])) X[[col]] <- as.numeric(factor(X[[col]]))
}
cat(sprintf("  Loaded %d rows, %d features\n", nrow(X), ncol(X)))

set.seed(42)
idx <- sample(nrow(X), nrow(X) * 0.8)
X_train <- X[idx, ]; X_test <- X[-idx, ]
y_train <- y[idx]; y_test <- y[-idx]

dtrain <- xgb.DMatrix(as.matrix(X_train), label = (y_train - 1) / 4)
dtest  <- xgb.DMatrix(as.matrix(X_test), label = (y_test - 1) / 4)

params <- list(
  objective = "reg:logistic", max_depth = 8, eta = 0.04,
  subsample = 0.85, colsample_bytree = 0.85,
  min_child_weight = 5, reg_lambda = 2.0, eval_metric = "rmse"
)
cat("[2] Training XGBoost ordinal (600 rounds)...\n")
model <- xgb.train(params, dtrain, nrounds = 600, verbose = 0)
cat("    Done.\n")

cat("[3] Getting top 30 features by gain...\n")
imp <- xgb.importance(feature_names = colnames(X_train), model = model)
top_feats <- imp$Feature[1:30]
cat(sprintf("    Top 5: %s\n", paste(top_feats[1:5], collapse = ", ")))

cat("[4] Computing SHAP on top 30 features (bg=100, test=50)...\n")
X_train_mat <- as.matrix(X_train)[, top_feats]
X_test_mat  <- as.matrix(X_test)[, top_feats]
set.seed(42)
bg_idx <- sample(nrow(X_train_mat), 100)
te_idx <- sample(nrow(X_test_mat), 50)

pred_fun_top <- function(m, X) {
  pred <- predict(m, xgb.DMatrix(X))
  pred
}

ks <- kernelshap(
  model,
  X_train_mat[bg_idx, ],
  X_test_mat[te_idx, ],
  pred_fun = pred_fun_top
)
cat("    SHAP done. Aggregating...\n")

shap_imp <- data.frame(
  Feature = top_feats,
  MeanSHAP = colMeans(abs(ks$S))
)
shap_imp <- shap_imp[order(shap_imp$MeanSHAP, decreasing = TRUE), ]
cat(sprintf("    Top 10 SHAP features:\n"))
print(head(shap_imp, 10))

dir.create("outputs", showWarnings = FALSE)
saveRDS(ks, "outputs/r_shap_values_top30.rds")
fwrite(shap_imp, "outputs/r_shap_importance_top30.csv")

# Plot top 20
top <- head(shap_imp, 20)
top$Feature <- factor(top$Feature, levels = rev(top$Feature))

pdf("outputs/r_shap_summary.pdf", width = 10, height = 8)
p <- ggplot(top, aes(x = Feature, y = MeanSHAP)) +
  geom_bar(stat = "identity", fill = "#D73027", color = "#67001F") +
  coord_flip() +
  labs(
    title = "Top 20 SHAP Feature Importance (XGBoost Ordinal)",
    subtitle = "Mean |SHAP| over 50 test samples, 100 background (top 30 features)",
    x = "Feature", y = "Mean |SHAP value|"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))
print(p)
dev.off()
cat("    Saved: outputs/r_shap_summary.pdf\n")
cat("Done!\n")
