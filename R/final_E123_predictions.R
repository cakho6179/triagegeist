# Saves the test predictions of E1 and E3 so that their confusion matrices and bootstrap intervals can be built
# (pipeline/09). The fits replay R/reproduce_E123.R exactly: split, E1, the second E1-feature fit that script
# labels "E2", then E3, with nothing in between, so QWK equals the locked 0.4326 and 0.4778.
source("R/lib_triage.R")

D <- load_triage()
fs <- feature_sets(names(D$X))
tr <- D$idx; te <- setdiff(seq_len(nrow(D$X)), tr)
yt <- D$y[te]

raw1 <- predict_ordinal(fit_ordinal(D$X, D$y, tr, fs$g1, 400), D$X, te, fs$g1)
invisible(fit_ordinal(D$X, D$y, tr, fs$g1, 400))
raw3 <- predict_ordinal(fit_ordinal(D$X, D$y, tr, fs$g3, 500), D$X, te, fs$g3)

for (cs in list(list("E1", fs$g1, raw1), list("E3", fs$g3, raw3))) {
  th <- find_thresholds(cs[[3]], yt)$thresholds
  pc <- decode(cs[[3]], th)
  append_csv(metrics_row(cs[[1]], length(cs[[2]]), yt, pc), "progression_E123.csv")
  save_pred(cs[[1]], te, yt, pc, cs[[3]])
}
print(fread(file.path(OUTDIR, "progression_E123.csv")))
cat("DONE\n")
