# Feature importance and SHAP values for the deployable model E4 (no IV-fluids flag).
# Gain importance comes from xgb.importance; SHAP values are the exact TreeSHAP contributions of the
# xgboost model (predict(..., predcontrib = TRUE)) on the whole test set. The model is fitted first, so it is
# the model of R/reproduce_E4.R. Scores are on the logit scale of the [0, 1] acuity score, lower = more urgent.
source("R/lib_triage.R")

D <- load_triage()
g4 <- feature_sets(names(D$X))$g4
tr <- D$idx; te <- setdiff(seq_len(nrow(D$X)), tr)
yt <- D$y[te]

model <- fit_ordinal(D$X, D$y, tr, g4, 600)
raw <- predict_ordinal(model, D$X, te, g4)
cat(sprintf("QWK (test-tuned thresholds) = %.4f\n", qwk(yt, decode(raw, find_thresholds(raw, yt)$thresholds))))

imp <- xgb.importance(feature_names = g4, model = model)
fwrite(imp[1:30], file.path(OUTDIR, "importance_gain_E4.csv"))

Xte <- as.matrix(D$X[te, ..g4])
contrib <- predict(model, xgb.DMatrix(Xte), predcontrib = TRUE)
contrib <- contrib[, seq_along(g4), drop = FALSE]   # the last column is the bias term
colnames(contrib) <- g4
mean_abs <- colMeans(abs(contrib))
top <- names(sort(mean_abs, decreasing = TRUE))[1:20]
direction <- sapply(top, function(f) {
  v <- Xte[, f]
  if (length(unique(v)) < 2) return(NA_real_)
  cor(v, contrib[, f], method = "spearman")
})
out <- data.table(feature = top, mean_abs_shap = round(mean_abs[top], 4), spearman_value_vs_shap = round(direction, 3))
fwrite(out, file.path(OUTDIR, "shap_E4_top20.csv"))
print(imp[1:15, .(Feature, Gain = round(Gain, 4))])
print(out)
cat("DONE\n")
