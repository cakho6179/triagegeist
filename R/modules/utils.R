# ============================================================
# Triagegeist R Pipeline - Module: Utilities
# Shared helper functions
# ============================================================

#' Quadratic Weighted Kappa
qwk <- function(actual, predicted) {
  actual <- as.integer(actual); predicted <- as.integer(predicted)
  k <- max(max(actual), max(predicted))
  O <- matrix(0, k, k)
  for (i in seq_along(actual)) O[actual[i], predicted[i]] <- O[actual[i], predicted[i]] + 1
  rs <- rowSums(O); cs <- colSums(O)
  E <- outer(rs, cs) / sum(O)
  w <- outer(1:k, 1:k, function(x,y) (x-y)^2/(k-1)^2)
  1 - sum(w*O) / sum(w*E)
}

#' Find optimal thresholds for ordinal regression via grid search
find_thresholds <- function(raw_preds, y_true, grid_fine = FALSE) {
  best <- -1; bt <- NULL
  t1_seq <- if (grid_fine) seq(0.18, 0.38, by = 0.02) else seq(0.22, 0.35, by = 0.03)
  t2_seq <- if (grid_fine) seq(0.35, 0.55, by = 0.02) else seq(0.38, 0.50, by = 0.03)
  t3_seq <- if (grid_fine) seq(0.48, 0.65, by = 0.02) else seq(0.50, 0.60, by = 0.03)
  t4_seq <- if (grid_fine) seq(0.60, 0.80, by = 0.02) else seq(0.65, 0.75, by = 0.03)
  for (t1 in t1_seq) for (t2 in t2_seq) {
    if (t1 >= t2) next
    for (t3 in t3_seq) {
      if (t2 >= t3) next
      for (t4 in t4_seq) {
        if (t3 >= t4) next
        p <- ifelse(raw_preds < t1, 1,
              ifelse(raw_preds < t2, 2,
              ifelse(raw_preds < t3, 3,
              ifelse(raw_preds < t4, 4, 5))))
        q <- qwk(y_true, p)
        if (q > best) { best <- q; bt <- c(t1, t2, t3, t4) }
      }
    }
  }
  list(qwk = best, thresholds = bt)
}

#' Convert raw predictions to class labels using thresholds
pred_to_class <- function(raw_preds, thresholds) {
  ifelse(raw_preds < thresholds[1], 1,
  ifelse(raw_preds < thresholds[2], 2,
  ifelse(raw_preds < thresholds[3], 3,
  ifelse(raw_preds < thresholds[4], 4, 5))))
}

#' Per-class metrics
class_metrics <- function(actual, predicted) {
  classes <- sort(unique(actual))
  result <- data.frame(Class = integer(), N_true = integer(), N_pred = integer(),
                       Precision = numeric(), Recall = numeric(), F1 = numeric())
  for (c in classes) {
    tp <- sum(actual == c & predicted == c)
    pp <- sum(predicted == c)
    pt <- sum(actual == c)
    prec <- if (pp > 0) tp / pp else 0
    rec <- if (pt > 0) tp / pt else 0
    f1 <- if (prec + rec > 0) 2 * prec * rec / (prec + rec) else 0
    result <- rbind(result, data.frame(Class = c, N_true = pt, N_pred = pp,
                                       Precision = prec, Recall = rec, F1 = f1))
  }
  result
}

#' Safe log
log_msg <- function(msg) cat(sprintf("[%s] %s\n", format(Sys.time(), "%H:%M:%S"), msg))
