# Two additions to the E1-E4 progression, on the split of R/reproduce_E4.R:
#   E2    a real "missingness flags" experiment: E1 features plus the ten *_missing columns (400 rounds, as E1)
#   E4    the same model as R/reproduce_E4.R, scored with thresholds fitted on out-of-fold training scores
#         instead of the test predictions
# E4 is fitted first so that it reproduces the locked result (xgboost's seed comes from R's RNG).
source("R/lib_triage.R")

D <- load_triage()
fs <- feature_sets(names(D$X))
tr <- D$idx; te <- setdiff(seq_len(nrow(D$X)), tr)
yt <- D$y[te]
e2_cols <- unique(c(fs$g1, grep("_missing$", names(D$X), value = TRUE)))

raw4 <- predict_ordinal(fit_ordinal(D$X, D$y, tr, fs$g4, 600), D$X, te, fs$g4)
raw2 <- predict_ordinal(fit_ordinal(D$X, D$y, tr, e2_cols, 400), D$X, te, e2_cols)
thr4 <- find_thresholds(oof_scores(D$X, D$y, tr, fs$g4, 600), D$y[tr])$thresholds

report_ordinal("E4", length(fs$g4), raw4, yt, te, thr4, "progression_extra.csv")
report_ordinal("E2", length(e2_cols), raw2, yt, te, NULL, "progression_extra.csv")
print(fread(file.path(OUTDIR, "progression_extra.csv")))
cat("DONE\n")
