# Random Forest and CART baselines on the E4 features, same split as E1-E6.
# Writes revision/results/final/pred_RF.csv, pred_CART.csv and the split/feature lists used by pipeline/08.
suppressMessages(library(ranger)); suppressMessages(library(rpart))
source("R/lib_triage.R")

D <- load_triage()
fs <- feature_sets(names(D$X))
g4 <- fs$g4
stopifnot(length(g4) == 640)
tr <- D$idx; te <- setdiff(seq_len(nrow(D$X)), tr)
yt <- D$y[te]

fwrite(data.table(row = te), file.path(OUTDIR, "split_test_rows.csv"))
fwrite(data.table(row = tr), file.path(OUTDIR, "split_train_rows.csv"))
writeLines(g4, file.path(OUTDIR, "e4_columns.txt"))
writeLines(fs$g3, file.path(OUTDIR, "e3_columns.txt"))

cat("Random Forest (500 trees)...\n")
xtr <- as.data.frame(D$X[tr, ..g4]); xte <- as.data.frame(D$X[te, ..g4])
rf <- ranger(x = xtr, y = factor(D$y[tr]), num.trees = 500, seed = 42, num.threads = 8)
pr <- as.integer(as.character(predict(rf, xte)$predictions))
append_csv(metrics_row("RF_500", length(g4), yt, pr), "baselines_tree.csv")
save_pred("RF", te, yt, pr)
print(fread(file.path(OUTDIR, "baselines_tree.csv")))

cat("CART (maxdepth 6)...\n")
names(xtr) <- names(xte) <- make.names(g4, unique = TRUE)
cart <- rpart(y ~ ., data = cbind(xtr, y = factor(D$y[tr])), method = "class",
              control = rpart.control(maxdepth = 6, cp = 0.001))
pc <- as.integer(as.character(predict(cart, xte, type = "class")))
append_csv(metrics_row("CART_depth6", length(g4), yt, pc), "baselines_tree.csv")
save_pred("CART", te, yt, pc)
print(fread(file.path(OUTDIR, "baselines_tree.csv")))
cat("DONE\n")
