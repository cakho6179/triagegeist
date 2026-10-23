# ============================================================
# Module 02: Model Training
# ============================================================
source("R/modules/utils.R")

train_xgboost_ordinal <- function(X_train, y_train, params = NULL) {
  log_msg("Training XGBoost ordinal regression...")
  
  default_params <- list(
    objective = "reg:logistic",
    max_depth = 8,
    eta = 0.04,
    subsample = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 3,
    lambda = 1.0,
    alpha = 0.5,
    gamma = 0.05,
    tree_method = "hist",
    nthread = -1
  )
  if (!is.null(params)) default_params[names(params)] <- params
  
  y_scaled <- (y_train - 1) / 4
  dt <- xgb.DMatrix(as.matrix(X_train), label = y_scaled)
  
  t0 <- Sys.time()
  model <- xgb.train(params = default_params, data = dt, nrounds = 600, verbose = 0)
  elapsed <- difftime(Sys.time(), t0, units = "secs")
  log_msg(sprintf("XGBoost trained: %.1fs, %d rounds", elapsed, 600))
  
  model
}

train_ranger_rf <- function(X_train, y_train, num_trees = 200) {
  log_msg(sprintf("Training Random Forest (%d trees)...", num_trees))
  
  t0 <- Sys.time()
  model <- ranger(
    x = X_train,
    y = as.factor(y_train),
    num.trees = num_trees,
    mtry = max(1, floor(sqrt(ncol(X_train)))),
    min.node.size = 5,
    importance = "impurity",
    seed = 42,
    verbose = FALSE
  )
  elapsed <- difftime(Sys.time(), t0, units = "secs")
  log_msg(sprintf("RF trained: %.1fs, OOB error=%.3f", elapsed, 1 - model$prediction.error))
  
  model
}

train_decision_tree <- function(X_train, y_train, maxdepth = 3) {
  log_msg(sprintf("Training Decision Tree (maxdepth=%d)...", maxdepth))
  
  # Clean column names for rpart formula
  clean_names <- make.names(names(X_train), unique = TRUE)
  dt_data <- as.data.frame(X_train)
  names(dt_data) <- clean_names
  dt_data$y <- as.factor(y_train)
  
  t0 <- Sys.time()
  model <- rpart(y ~ ., data = dt_data, method = "class",
                 control = rpart.control(maxdepth = maxdepth, cp = 0.001))
  elapsed <- difftime(Sys.time(), t0, units = "secs")
  log_msg(sprintf("Decision tree trained: %.1fs, %d splits", elapsed, nrow(model$frame) - 1))
  
  list(model = model, clean_names = clean_names)
}

load_model <- function(filepath) {
  log_msg(sprintf("Loading model from %s...", filepath))
  readRDS(filepath)
}

save_model <- function(model, filepath) {
  dir.create(dirname(filepath), showWarnings = FALSE, recursive = TRUE)
  saveRDS(model, filepath)
  log_msg(sprintf("Model saved: %s", filepath))
}
