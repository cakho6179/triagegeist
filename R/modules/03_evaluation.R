# ============================================================
# Module 03: Evaluation
# ============================================================
source("R/modules/utils.R")

evaluate_model <- function(model, X_test, y_test, model_name = "Model", type = "xgb", thresholds = NULL) {
  log_msg(sprintf("Evaluating %s...", model_name))
  
  if (type == "xgb") {
    raw_preds <- predict(model, xgb.DMatrix(as.matrix(X_test)))
    if (is.null(thresholds)) {
      opt <- find_thresholds(raw_preds, y_test, grid_fine = TRUE)
      thresholds <- opt$thresholds
      qwk_val <- opt$qwk
    } else {
      qwk_val <- qwk(y_test, pred_to_class(raw_preds, thresholds))
    }
    preds <- pred_to_class(raw_preds, thresholds)
  } else if (type == "rf") {
    preds <- as.integer(as.character(predict(model, X_test)$predictions))
    raw_preds <- NULL
    qwk_val <- qwk(y_test, preds)
    thresholds <- NULL
  } else if (type == "dt") {
    clean_names <- make.names(names(X_test), unique = TRUE)
    dt_te <- as.data.frame(X_test)
    names(dt_te) <- clean_names
    preds <- as.integer(as.character(predict(model, dt_te, type = "class")))
    raw_preds <- NULL
    qwk_val <- qwk(y_test, preds)
    thresholds <- NULL
  }
  
  cm <- table(Actual = y_test, Predicted = preds)
  metrics <- class_metrics(y_test, preds)
  
  list(
    model_name = model_name,
    qwk = qwk_val,
    thresholds = thresholds,
    predictions = preds,
    raw_predictions = raw_preds,
    confusion_matrix = cm,
    metrics = metrics
  )
}

print_evaluation <- function(eval_result) {
  cat(sprintf("\n========== %s ==========\n", eval_result$model_name))
  cat(sprintf("QWK: %.4f\n", eval_result$qwk))
  if (!is.null(eval_result$thresholds)) {
    cat(sprintf("Thresholds: [%s]\n", paste(round(eval_result$thresholds, 4), collapse = ", ")))
  }
  cat("\nConfusion Matrix:\n")
  print(eval_result$confusion_matrix)
  cat("\nPer-class Metrics:\n")
  print(eval_result$metrics, row.names = FALSE)
}

compare_models <- function(...) {
  results <- list(...)
  cat("\n========== MODEL COMPARISON ==========\n")
  cat(sprintf("%-25s %s\n", "Model", "QWK"))
  cat(strrep("-", 40), "\n")
  for (r in results) {
    cat(sprintf("%-25s %.4f\n", r$model_name, r$qwk))
  }
}
