# True multiclass XGBoost on the E4 features (same settings as R/reproduce_E5mc.R, 500 rounds), decoded two ways:
#   argmax                       the reported E5mc baseline
#   expected class + thresholds  the class probabilities are collapsed to a score in [0, 1] and cut with four
#                                ordinal thresholds, as the ordinal model does; thresholds fitted on the test
#                                predictions ("test_tuned", the protocol of R/reproduce_E4.R) or on 5-fold
#                                out-of-fold training scores ("train_tuned")
# The argmax model is fitted first, so it reproduces R/reproduce_E5mc.R.
source("R/lib_triage.R")

D <- load_triage()
g4 <- feature_sets(names(D$X))$g4
tr <- D$idx; te <- setdiff(seq_len(nrow(D$X)), tr)
yt <- D$y[te]

XGB_MC <- list(objective = "multi:softprob", num_class = 5, max_depth = 8, eta = 0.04,
               subsample = 0.85, colsample_bytree = 0.85, min_child_weight = 5,
               reg_lambda = 2.0, eval_metric = "mlogloss")
N_MC <- 500

fit_mc <- function(rows)
  xgb.train(XGB_MC, xgb.DMatrix(as.matrix(D$X[rows, ..g4]), label = D$y[rows] - 1), nrounds = N_MC, verbose = 0)

probs <- function(m, rows)
  matrix(predict(m, xgb.DMatrix(as.matrix(D$X[rows, ..g4]))), nrow = length(rows), ncol = 5, byrow = FALSE)
exp_score <- function(p) as.numeric((p %*% (1:5) - 1) / 4)

p <- probs(fit_mc(tr), te)
pc <- max.col(p, ties.method = "first")
append_csv(metrics_row("multiclass_argmax", length(g4), yt, pc, list(protocol = "argmax")), "multiclass.csv")
save_pred("multiclass_argmax", te, yt, pc)

set.seed(42)
fold <- sample(rep(1:5, length.out = length(tr)))
oof <- numeric(length(tr))
for (f in 1:5) oof[fold == f] <- exp_score(probs(fit_mc(tr[fold != f]), tr[fold == f]))
thr_train <- find_thresholds(oof, D$y[tr])$thresholds

report_ordinal("multiclass_expected", length(g4), exp_score(p), yt, te, thr_train, "multiclass.csv")
print(fread(file.path(OUTDIR, "multiclass.csv")))
cat("DONE\n")
