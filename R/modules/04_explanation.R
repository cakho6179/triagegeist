# ============================================================
# Module 04: Model Explanation (Importance, PDP, SHAP)
# ============================================================
source("R/modules/utils.R")

# ---- 4a. Feature Importance ----
compute_feature_importance <- function(model, feature_names, type = "xgb") {
  log_msg("Computing feature importance...")
  
  if (type == "xgb") {
    imp <- xgb.importance(feature_names = feature_names, model = model)
    imp <- as.data.frame(imp)
  } else if (type == "rf") {
    imp <- data.frame(
      Feature = names(model$variable.importance),
      Gain = as.numeric(model$variable.importance)
    )
    imp <- imp[order(imp$Gain, decreasing = TRUE), ]
  }
  
  imp$Rank <- 1:nrow(imp)
  imp$Cumulative <- cumsum(imp$Gain) / sum(imp$Gain)
  log_msg(sprintf("Top 3: %s", paste(imp$Feature[1:3], collapse = ", ")))
  imp
}

plot_feature_importance <- function(imp, top_n = 20, filename = "outputs/importance.pdf") {
  top <- head(imp, top_n)
  dir.create(dirname(filename), showWarnings = FALSE, recursive = TRUE)
  
  pdf(filename, width = 10, height = max(6, top_n * 0.35))
  par(mar = c(4, 12, 3, 1))
  bar_pos <- barplot(top$Gain, names.arg = top$Feature, horiz = TRUE, las = 1,
                     xlab = "Importance (Gain)", main = sprintf("Top %d Features", top_n),
                     col = colorRampPalette(c("#2166AC", "#92C5DE", "#F7F7F7"))(top_n),
                     border = NA)
  text(x = top$Gain, y = bar_pos, labels = sprintf("%.1f%%", 100 * top$Gain / sum(imp$Gain)),
       pos = 4, cex = 0.7, col = "#333333")
  dev.off()
  log_msg(sprintf("Saved: %s", filename))
}

# ---- 4b. Partial Dependence ----
compute_partial_dependence <- function(model, X_train, feature_names, top_n = 6, filename = "outputs/pdp.pdf") {
  log_msg(sprintf("Computing partial dependence for top %d features...", top_n))
  
  # Get top features if model is xgb
  imp <- xgb.importance(feature_names = feature_names, model = model)
  top_features <- imp$Feature[1:min(top_n, nrow(imp))]
  X_mat <- as.matrix(X_train)
  
  pdf(filename, width = 12, height = 8)
  par(mfrow = c(ceiling(sqrt(top_n)), ceiling(top_n / ceiling(sqrt(top_n)))))
  
  for (feat in top_features) {
    fi <- which(feature_names == feat)
    if (length(fi) == 0) next
    tryCatch({
      pd <- partial(model, pred.var = fi, train = X_mat, type = "regression",
                    grid.resolution = 25, progress = "none")
      plot(pd, main = paste("PDP:", feat), ylab = "Predicted acuity", xlab = feat,
           type = "l", lwd = 2, col = "#2166AC")
    }, error = function(e) {
      cat(sprintf("  PDP error for %s: %s\n", feat, e$message))
    })
  }
  dev.off()
  log_msg(sprintf("Saved: %s", filename))
  
  top_features
}

# ---- 4c. Per-class Prediction Distribution ----
plot_prediction_distribution <- function(raw_preds, y_test, thresholds = NULL, filename = "outputs/pred_dist.pdf") {
  log_msg("Plotting prediction distribution by class...")
  
  df_plot <- data.frame(
    Actual = factor(y_test, levels = 1:5, labels = paste("ESI", 1:5)),
    Prediction = raw_preds
  )
  
  p <- ggplot(df_plot, aes(x = Prediction, fill = Actual)) +
    geom_density(alpha = 0.4, adjust = 1.5) +
    scale_fill_brewer(palette = "Set1") +
    labs(title = "Raw Prediction Distribution by Actual ESI Level",
         subtitle = "Lower values = more urgent",
         x = "Model Raw Prediction", y = "Density", fill = "True ESI") +
    theme_minimal(base_size = 12) +
    theme(legend.position = "bottom")
  
  if (!is.null(thresholds)) {
    p <- p + geom_vline(xintercept = thresholds, linetype = "dashed",
                        color = "gray40", alpha = 0.6) +
      annotate("text", x = mean(thresholds[1:2]), y = 0, label = "Th 1-2", size = 3) +
      annotate("text", x = mean(thresholds[2:3]), y = 0, label = "Th 2-3", size = 3) +
      annotate("text", x = mean(thresholds[3:4]), y = 0, label = "Th 3-4", size = 3) +
      annotate("text", x = thresholds[4] + 0.05, y = 0, label = "Th 4-5", size = 3)
  }
  
  dir.create(dirname(filename), showWarnings = FALSE, recursive = TRUE)
  ggsave(filename, p, width = 10, height = 6)
  log_msg(sprintf("Saved: %s", filename))
}

# ---- 4d. SHAP Analysis ----
compute_shap <- function(model, X_train, X_test, feature_names, n_train = 200, n_test = 50) {
  log_msg(sprintf("Computing SHAP (background=%d, explain=%d)...", n_train, n_test))
  
  set.seed(42)
  X_train_mat <- as.matrix(X_train)
  X_test_mat  <- as.matrix(X_test)
  
  shap_n <- min(n_train, nrow(X_train_mat))
  shap_test_n <- min(n_test, nrow(X_test_mat))
  
  bg_idx <- sample(nrow(X_train_mat), shap_n)
  te_idx <- sample(nrow(X_test_mat), shap_test_n)
  
  # Use kernelshap for faster computation
  shap_values <- tryCatch({
    library(kernelshap)
    ks <- kernelshap(model, X_train_mat[bg_idx, ], X_test_mat[te_idx, ],
                     pred_fun = function(m, X) predict(m, xgb.DMatrix(X)))
    ks
  }, error = function(e) {
    log_msg(sprintf("SHAP computation note: %s", e$message))
    NULL
  })
  
  if (!is.null(shap_values)) {
    shap_imp <- data.frame(
      Feature = feature_names,
      MeanSHAP = colMeans(abs(shap_values$S))
    )
    shap_imp <- shap_imp[order(shap_imp$MeanSHAP, decreasing = TRUE), ]
    log_msg(sprintf("SHAP top 3: %s", paste(shap_imp$Feature[1:3], collapse = ", ")))
    list(values = shap_values, importance = shap_imp)
  } else {
    log_msg("SHAP computation skipped (kernelshap not available)")
    NULL
  }
}

plot_shap_summary <- function(shap_result, top_n = 15, filename = "outputs/shap_summary.pdf") {
  if (is.null(shap_result)) {
    log_msg("No SHAP results to plot")
    return()
  }
  
  top <- head(shap_result$importance, top_n)
  pdf(filename, width = 10, height = max(5, top_n * 0.3))
  par(mar = c(4, 10, 3, 1))
  barplot(top$MeanSHAP, names.arg = top$Feature, horiz = TRUE, las = 1,
          xlab = "Mean |SHAP|", main = sprintf("Top %d SHAP Features", top_n),
          col = "#D73027", border = NA)
  dev.off()
  log_msg(sprintf("Saved: %s", filename))
}

# ---- 4e. Decision Tree Visualization ----
plot_decision_tree <- function(dt_result, X_test, y_test, filename = "outputs/decision_tree.pdf") {
  log_msg("Plotting decision tree...")
  
  preds <- as.integer(as.character(predict(dt_result$model, 
    {dt_te <- as.data.frame(X_test); names(dt_te) <- dt_result$clean_names; dt_te},
    type = "class")))
  dt_q <- qwk(y_test, preds)
  
  pdf(filename, width = 14, height = 10)
  rpart.plot(dt_result$model, type = 2, extra = 104, under = TRUE, tweak = 1.3,
             main = paste("Triage Acuity Decision Tree (QWK =", round(dt_q, 3), ")"))
  dev.off()
  log_msg(sprintf("Saved: %s (QWK=%.4f)", filename, dt_q))
  dt_q
}

# ---- 4f. Confusion Matrix Heatmap ----
plot_confusion_matrix <- function(cm, filename = "outputs/confusion_matrix.pdf") {
  log_msg("Plotting confusion matrix heatmap...")
  
  cm_df <- as.data.frame(cm)
  names(cm_df) <- c("Actual", "Predicted", "Freq")
  cm_df$Actual <- factor(cm_df$Actual, levels = 1:5)
  cm_df$Predicted <- factor(cm_df$Predicted, levels = 1:5)
  
  p <- ggplot(cm_df, aes(x = Predicted, y = Actual, fill = Freq)) +
    geom_tile() +
    geom_text(aes(label = Freq), size = 4, fontface = "bold") +
    scale_fill_gradient(low = "white", high = "#2166AC", trans = "log1p") +
    labs(title = "Confusion Matrix", x = "Predicted ESI", y = "Actual ESI") +
    theme_minimal(base_size = 12) +
    coord_fixed()
  
  dir.create(dirname(filename), showWarnings = FALSE, recursive = TRUE)
  ggsave(filename, p, width = 8, height = 6)
  log_msg(sprintf("Saved: %s", filename))
}

# ---- 4g. Metrics Barplot ----
plot_metrics <- function(metrics, filename = "outputs/metrics.pdf") {
  log_msg("Plotting per-class metrics...")
  
  df_m <- reshape2::melt(metrics, id.vars = "Class", measure.vars = c("Precision", "Recall", "F1"))
  
  p <- ggplot(df_m, aes(x = factor(Class), y = value, fill = variable)) +
    geom_bar(stat = "identity", position = "dodge", width = 0.7) +
    scale_fill_brewer(palette = "Set2") +
    labs(title = "Per-class Performance Metrics",
         x = "ESI Class", y = "Score", fill = "Metric") +
    theme_minimal(base_size = 12) +
    ylim(0, 1) +
    geom_hline(yintercept = 0.5, linetype = "dashed", alpha = 0.5)
  
  dir.create(dirname(filename), showWarnings = FALSE, recursive = TRUE)
  ggsave(filename, p, width = 8, height = 5)
  log_msg(sprintf("Saved: %s", filename))
}
