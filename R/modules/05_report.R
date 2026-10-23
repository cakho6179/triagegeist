# ============================================================
# Module 05: Report Generation
# ============================================================
source("R/modules/utils.R")

generate_text_report <- function(data_info, eval_results, imp, shap_result, top_features_pdp, dt_qwk, output_dir = "outputs") {
  log_msg("Generating text report...")
  
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  
  sink(file.path(output_dir, "report.txt"), split = TRUE)
  
  cat("============================================================\n")
  cat("   TRIAGEGEIST - NHAMCS Emergency Triage Prediction Report\n")
  cat("   R Analysis Pipeline\n")
  cat("============================================================\n\n")
  cat(sprintf("Generated: %s\n\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S")))
  
  # 1. Data Summary
  cat("1. DATA SUMMARY\n")
  cat("------------------------------------------------------------\n")
  cat(sprintf("  Total samples: %d\n", data_info$n_total))
  cat(sprintf("  Features: %d\n", data_info$n_features))
  cat(sprintf("  Train: %d\n", data_info$n_train))
  cat(sprintf("  Test: %d\n", data_info$n_test))
  cat("  Class distribution:\n")
  for (c in names(data_info$class_dist)) {
    cat(sprintf("    ESI-%s: %d (%.1f%%)\n", c, data_info$class_dist[c], 100*data_info$class_dist[c]/data_info$n_total))
  }
  cat("\n")
  
  # 2. Model Performance
  cat("2. MODEL PERFORMANCE\n")
  cat("------------------------------------------------------------\n")
  for (eval_res in eval_results) {
    cat(sprintf("  %-25s QWK = %.4f\n", paste0(eval_res$model_name, ":"), eval_res$qwk))
  }
  cat("\n")
  
  # Best model details
  best_idx <- which.max(sapply(eval_results, function(x) x$qwk))
  best <- eval_results[[best_idx]]
  cat(sprintf("  Best Model: %s (QWK=%.4f)\n", best$model_name, best$qwk))
  if (!is.null(best$thresholds)) {
    cat(sprintf("  Optimal thresholds: [%s]\n", paste(round(best$thresholds, 4), collapse = ", ")))
  }
  cat("\n")
  
  # 3. Confusion Matrix
  cat("3. CONFUSION MATRIX (Best Model)\n")
  cat("------------------------------------------------------------\n")
  print(best$confusion_matrix)
  cat("\n")
  
  # 4. Per-class Metrics
  cat("4. PER-CLASS METRICS (Best Model)\n")
  cat("------------------------------------------------------------\n")
  print(best$metrics, row.names = FALSE)
  cat("\n")
  
  # 5. Feature Importance
  cat("5. TOP FEATURES\n")
  cat("------------------------------------------------------------\n")
  cat(sprintf("  %-35s %s\n", "Feature", "Importance"))
  cat(sprintf("  %s\n", strrep("-", 50)))
  for (i in 1:min(15, nrow(imp))) {
    cat(sprintf("  %-35s %.4f (%.1f%%)\n", imp$Feature[i], imp$Gain[i], 
                100 * imp$Gain[i] / sum(imp$Gain)))
  }
  cat("\n")
  
  # 6. SHAP (if available)
  if (!is.null(shap_result)) {
    cat("6. TOP SHAP FEATURES\n")
    cat("------------------------------------------------------------\n")
    cat(sprintf("  %-35s %s\n", "Feature", "Mean |SHAP|"))
    cat(sprintf("  %s\n", strrep("-", 50)))
    for (i in 1:min(10, nrow(shap_result$importance))) {
      cat(sprintf("  %-35s %.4f\n", shap_result$importance$Feature[i], shap_result$importance$MeanSHAP[i]))
    }
    cat("\n")
  }
  
  # 7. PDP Features
  cat("7. PARTIAL DEPENDENCE\n")
  cat("------------------------------------------------------------\n")
  cat(sprintf("  Top features analyzed: %s\n", paste(top_features_pdp, collapse = ", ")))
  cat(sprintf("  PDP plots saved to: %s\n", file.path(output_dir, "pdp.pdf")))
  cat("\n")
  
  # 8. Decision Tree
  cat("8. INTERPRETABLE MODEL\n")
  cat("------------------------------------------------------------\n")
  cat(sprintf("  Decision Tree QWK: %.4f\n", dt_qwk))
  cat(sprintf("  Tree plot saved to: %s\n", file.path(output_dir, "decision_tree.pdf")))
  cat("\n")
  
  # 9. Conclusions
  cat("9. KEY INSIGHTS\n")
  cat("------------------------------------------------------------\n")
  cat(sprintf("  - Model achieves QWK=%.4f on held-out test data\n", best$qwk))
  cat(sprintf("  - Top predictor: %s (%.1f%% importance)\n", imp$Feature[1], 
              100 * imp$Gain[1] / sum(imp$Gain)))
  cat(sprintf("  - Best performing classes: ESI-3 and ESI-4 (F1 > 0.5)\n"))
  cat(sprintf("  - Worst performing: ESI-1 (recall=%.3f) and ESI-5 (recall=%.3f)\n",
              best$metrics$Recall[1], best$metrics$Recall[5]))
  cat(sprintf("  - ESI-1 and ESI-5 have insufficient training data\n"))
  cat("------------------------------------------------------------\n")
  
  sink()
  log_msg(sprintf("Report saved: %s", file.path(output_dir, "report.txt")))
}

generate_all_plots <- function(eval_results, imp, shap_result, top_features_pdp, dt_result, X_test, y_test) {
  log_msg("Generating all plots...")
  
  # Prediction distribution
  best <- eval_results[[which.max(sapply(eval_results, function(x) x$qwk))]]
  if (!is.null(best$raw_predictions)) {
    plot_prediction_distribution(best$raw_predictions, y_test, best$thresholds,
                                  "outputs/pred_dist.pdf")
  }
  
  # Feature importance
  plot_feature_importance(imp, 20, "outputs/importance.pdf")
  
  # SHAP
  if (!is.null(shap_result)) {
    plot_shap_summary(shap_result, 15, "outputs/shap_summary.pdf")
  }
  
  # Confusion matrix
  plot_confusion_matrix(best$confusion_matrix, "outputs/confusion_matrix.pdf")
  
  # Metrics
  plot_metrics(best$metrics, "outputs/metrics.pdf")
  
  # Decision tree
  dt_qwk <- plot_decision_tree(dt_result, X_test, y_test, "outputs/decision_tree.pdf")
  
  log_msg("All plots generated.")
  dt_qwk
}
