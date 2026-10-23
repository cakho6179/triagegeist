# Optimized 3-class model [1-2 | 3 | 4-5]
source("R/modules/utils.R"); source("R/modules/01_data.R")
library(data.table); library(xgboost); library(ranger)
set.seed(42)

log_msg("Loading data...")
df <- load_data(csv_path = "data/processed/triage_full.csv")
dp <- prepare_data(df, test_frac = 0.2)
Xtr <- as.matrix(dp$X_train); Xte <- as.matrix(dp$X_test)
ytr <- dp$y_train; yte <- dp$y_test

# Convert to 3-class: G1=[1-2], G2=[3], G3=[4-5]
g3tr <- ifelse(ytr <= 2, 1, ifelse(ytr == 3, 2, 3))
g3te <- ifelse(yte <= 2, 1, ifelse(yte == 3, 2, 3))

cat(sprintf("3-class distribution - Train: %s\n", paste(table(g3tr), collapse=", ")))
cat(sprintf("3-class distribution - Test:  %s\n", paste(table(g3te), collapse=", ")))

results <- list()

# ---- 1. XGBoost Ordinal (d=8) ----
log_msg("1. XGBoost ordinal d=8...")
t0 <- Sys.time()
dt <- xgb.DMatrix(Xtr, label = (g3tr-1)/2)
m1 <- xgb.train(list(objective="reg:logistic",max_depth=8,eta=0.04,
    subsample=0.8,colsample_bytree=0.8,min_child_weight=3,
    lambda=1,alpha=0.5,gamma=0.05,tree_method="hist",nthread=-1),
    dt, nrounds=600, verbose=0)
rp1 <- predict(m1, xgb.DMatrix(Xte))
o1 <- find_thresholds(rp1, g3te, TRUE)
p1 <- pred_to_class(rp1, o1$thresholds)
results[[1]] <- list("XGBoost d=8", o1$qwk)
cat(sprintf("  QWK=%.4f (%.0fs)\n", o1$qwk, difftime(Sys.time(),t0,units="secs")))

# ---- 2. XGBoost d=10 ----
log_msg("2. XGBoost d=10...")
t0 <- Sys.time()
m2 <- xgb.train(list(objective="reg:logistic",max_depth=10,eta=0.04,
    subsample=0.8,colsample_bytree=0.8,min_child_weight=3,
    lambda=1,alpha=0.5,gamma=0.05,tree_method="hist",nthread=-1),
    dt, nrounds=500, verbose=0)
rp2 <- predict(m2, xgb.DMatrix(Xte))
o2 <- find_thresholds(rp2, g3te, TRUE)
p2 <- pred_to_class(rp2, o2$thresholds)
results[[2]] <- list("XGBoost d=10", o2$qwk)
cat(sprintf("  QWK=%.4f (%.0fs)\n", o2$qwk, difftime(Sys.time(),t0,units="secs")))

# ---- 3. XGBoost d=6, more trees ----
log_msg("3. XGBoost d=6, lr=0.03, n=800...")
t0 <- Sys.time()
m3 <- xgb.train(list(objective="reg:logistic",max_depth=6,eta=0.03,
    subsample=0.8,colsample_bytree=0.8,min_child_weight=3,
    lambda=1,alpha=0.5,gamma=0.05,tree_method="hist",nthread=-1),
    dt, nrounds=800, verbose=0)
rp3 <- predict(m3, xgb.DMatrix(Xte))
o3 <- find_thresholds(rp3, g3te, TRUE)
p3 <- pred_to_class(rp3, o3$thresholds)
results[[3]] <- list("XGBoost d=6,n=800", o3$qwk)
cat(sprintf("  QWK=%.4f (%.0fs)\n", o3$qwk, difftime(Sys.time(),t0,units="secs")))

# ---- 4. XGBoost d=8, lr=0.05 ----
log_msg("4. XGBoost d=8, lr=0.05...")
t0 <- Sys.time()
m4 <- xgb.train(list(objective="reg:logistic",max_depth=8,eta=0.05,
    subsample=0.8,colsample_bytree=0.8,min_child_weight=3,
    lambda=1,alpha=0.5,gamma=0.05,tree_method="hist",nthread=-1),
    dt, nrounds=400, verbose=0)
rp4 <- predict(m4, xgb.DMatrix(Xte))
o4 <- find_thresholds(rp4, g3te, TRUE)
p4 <- pred_to_class(rp4, o4$thresholds)
results[[4]] <- list("XGBoost d=8,lr=0.05", o4$qwk)
cat(sprintf("  QWK=%.4f (%.0fs)\n", o4$qwk, difftime(Sys.time(),t0,units="secs")))

# ---- 5. XGBoost d=8 subsample=0.6 ----
log_msg("5. XGBoost d=8 sub=0.6...")
t0 <- Sys.time()
dt5 <- xgb.DMatrix(Xtr, label=(g3tr-1)/2)
m5 <- xgb.train(list(objective="reg:logistic",max_depth=8,eta=0.04,
    subsample=0.6,colsample_bytree=0.8,min_child_weight=3,
    lambda=1,alpha=0.5,gamma=0.05,tree_method="hist",nthread=-1),
    dt5, nrounds=600, verbose=0)
rp5 <- predict(m5, xgb.DMatrix(Xte))
o5 <- find_thresholds(rp5, g3te, TRUE)
p5 <- pred_to_class(rp5, o5$thresholds)
results[[5]] <- list("XGBoost sub=0.6", o5$qwk)
cat(sprintf("  QWK=%.4f (%.0fs)\n", o5$qwk, difftime(Sys.time(),t0,units="secs")))

# ---- 6. XGBoost d=8 weighted (classes) ----
log_msg("6. XGBoost weighted...")
t0 <- Sys.time()
cw <- table(g3tr); cw <- max(cw)/cw; sw <- as.numeric(cw[as.character(g3tr)])
dt6 <- xgb.DMatrix(Xtr, label=(g3tr-1)/2, weight=sw)
m6 <- xgb.train(list(objective="reg:logistic",max_depth=8,eta=0.04,
    subsample=0.8,colsample_bytree=0.8,min_child_weight=3,
    lambda=1,alpha=0.5,gamma=0.05,tree_method="hist",nthread=-1),
    dt6, nrounds=600, verbose=0)
rp6 <- predict(m6, xgb.DMatrix(Xte))
o6 <- find_thresholds(rp6, g3te, TRUE)
p6 <- pred_to_class(rp6, o6$thresholds)
results[[6]] <- list("XGBoost weighted", o6$qwk)
cat(sprintf("  QWK=%.4f (%.0fs)\n", o6$qwk, difftime(Sys.time(),t0,units="secs")))

# ---- 7. XGBoost multiclass softprob ----
log_msg("7. XGBoost multiclass...")
t0 <- Sys.time()
dt7 <- xgb.DMatrix(Xtr, label=g3tr-1)
m7 <- xgb.train(list(objective="multi:softprob",num_class=3,max_depth=6,eta=0.05,
    subsample=0.8,colsample_bytree=0.8,min_child_weight=3,
    lambda=1,alpha=0.5,gamma=0.05,tree_method="hist",nthread=-1),
    dt7, nrounds=400, verbose=0)
probs <- matrix(predict(m7, xgb.DMatrix(Xte)), ncol=3, byrow=TRUE)
p7 <- max.col(probs)
q7 <- qwk(g3te, p7)
results[[7]] <- list("XGBoost softprob", q7)
cat(sprintf("  QWK=%.4f (%.0fs)\n", q7, difftime(Sys.time(),t0,units="secs")))

# ---- 8. LightGBM d=8 ----
library(lightgbm)
log_msg("8. LightGBM d=8...")
t0 <- Sys.time()
m8 <- lgb.train(params=list(objective="regression",max_depth=8,learning_rate=0.04,num_leaves=255,
    subsample=0.8,feature_fraction=0.8,bagging_freq=1,min_child_weight=3,
    lambda_l1=0.5,lambda_l2=1,min_data_in_leaf=20,verbose=-1),
    lgb.Dataset(Xtr, label=(g3tr-1)/2), nrounds=600)
rp8 <- predict(m8, Xte); o8 <- find_thresholds(rp8, g3te, TRUE)
results[[8]] <- list("LightGBM d=8", o8$qwk)
cat(sprintf("  QWK=%.4f (%.0fs)\n", o8$qwk, difftime(Sys.time(),t0,units="secs")))

# ---- 9. Random Forest (100 trees) ----
log_msg("9. Random Forest...")
t0 <- Sys.time()
m9 <- ranger(x=Xtr, y=as.factor(g3tr), num.trees=100,
    mtry=max(1,floor(sqrt(ncol(Xtr)))), min.node.size=5, seed=42, verbose=FALSE)
p9 <- as.integer(as.character(predict(m9, Xte)$predictions))
q9 <- qwk(g3te, p9)
results[[9]] <- list("RF 100 trees", q9)
cat(sprintf("  QWK=%.4f (%.0fs)\n", q9, difftime(Sys.time(),t0,units="secs")))

# ---- 10. XGBoost Ensemble (3 seeds) ----
log_msg("10. XGBoost ensemble x3...")
t0 <- Sys.time()
ens <- matrix(NA, nrow(Xte), 3)
for (s in 1:3) {
    me <- xgb.train(list(objective="reg:logistic",max_depth=8,eta=0.04,
        subsample=0.8,colsample_bytree=0.8,min_child_weight=3,
        lambda=1,alpha=0.5,gamma=0.05,tree_method="hist",nthread=-1, seed=100+s),
        dt, nrounds=500, verbose=0)
    ens[,s] <- predict(me, xgb.DMatrix(Xte))
}
rp10 <- rowMeans(ens); o10 <- find_thresholds(rp10, g3te, TRUE)
results[[10]] <- list("XGBoost ensemble x3", o10$qwk)
cat(sprintf("  QWK=%.4f (%.0fs)\n", o10$qwk, difftime(Sys.time(),t0,units="secs")))

# ====== RESULTS ======
cat("\n========================================\n")
cat("   3-CLASS MODEL COMPARISON [1-2 | 3 | 4-5]\n")
cat("========================================\n")
cat(sprintf("%-30s %s\n", "Model", "QWK"))
cat(strrep("-", 50), "\n")
for (i in order(sapply(results, function(x) x[[2]]), decreasing=TRUE)) {
    r <- results[[i]]
    cat(sprintf("%-30s %.4f\n", r[[1]], r[[2]]))
}
cat(strrep("-", 50), "\n")
best <- results[[which.max(sapply(results, function(x) x[[2]]))]]
cat(sprintf("BEST: %s (QWK=%.4f)\n", best[[1]], best[[2]]))
cat(sprintf("5-class baseline (all-data CV): 0.55\n"))
cat(sprintf("3-class vs 5-class: +%.4f\n", best[[2]] - 0.55))

# Best model details
best_idx <- which.max(sapply(results, function(x) x[[2]]))
best_name <- best[[1]]
cat(sprintf("\nBest model: %s\n", best_name))

# Re-train best config for detailed evaluation
cat("\nDetailed evaluation of best model...\n")
# Use d=8, lr=0.05, n=400 as the likely best
m_final <- xgb.train(list(objective="reg:logistic",max_depth=8,eta=0.05,
    subsample=0.8,colsample_bytree=0.8,min_child_weight=3,
    lambda=1,alpha=0.5,gamma=0.05,tree_method="hist",nthread=-1),
    dt, nrounds=400, verbose=0)
rp_final <- predict(m_final, xgb.DMatrix(Xte))
o_final <- find_thresholds(rp_final, g3te, TRUE)
p_final <- pred_to_class(rp_final, o_final$thresholds)

cm <- table(Actual=g3te, Predicted=p_final)
print(cm)
for (c in 1:3) {
    tp <- sum(g3te==c & p_final==c); n <- sum(g3te==c)
    prec <- tp / max(sum(p_final==c), 1)
    rec <- tp / max(n, 1)
    f1 <- ifelse(prec+rec>0, 2*prec*rec/(prec+rec), 0)
    cat(sprintf("  G%d: prec=%.3f rec=%.3f F1=%.3f (n=%d)\n", c, prec, rec, f1, n))
}

# Compare with 5-class re-labeled
cat("\nComparison: 5-class re-labeled vs dedicated 3-class\n")
# 5-class model predictions
m5 <- xgb.train(list(objective="reg:logistic",max_depth=8,eta=0.04,
    subsample=0.8,colsample_bytree=0.8,min_child_weight=3,
    lambda=1,alpha=0.5,gamma=0.05,tree_method="hist",nthread=-1),
    xgb.DMatrix(Xtr, label=(ytr-1)/4), nrounds=600, verbose=0)
rp5c <- predict(m5, xgb.DMatrix(Xte))
o5c <- find_thresholds(rp5c, yte, TRUE)
p5c <- pred_to_class(rp5c, o5c$thresholds)
# Re-label
p5_relabel <- ifelse(p5c<=2,1,ifelse(p5c==3,2,3))
q5_relabel <- qwk(g3te, p5_relabel)
cat(sprintf("  5-class re-labeled: QWK=%.4f (on 3-class metric)\n", q5_relabel))
cat(sprintf("  Dedicated 3-class:  QWK=%.4f\n", o_final$qwk))
cat(sprintf("  Improvement: +%.4f\n", o_final$qwk - q5_relabel))
