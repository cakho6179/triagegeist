suppressMessages({
  library(data.table)
  library(ggplot2)
  library(xgboost)
})

cat("========================================\n")
cat("WAIT TIME ANALYSIS (Section 18)\n")
cat("========================================\n\n")

# ============================================================
# Load raw data (has wait_time_minutes)
# ============================================================
d <- fread("data/raw/nhamcs_data_2018_22.csv")
cat(sprintf("Loaded %d rows from raw data\n", nrow(d)))

# Drop rows with missing wait time
d <- d[!is.na(wait_time_minutes) & wait_time_minutes >= 0]
cat(sprintf("After dropping NA/invalid: %d rows\n", nrow(d)))

# Cap wait time at 99th percentile (extreme outliers)
cap <- quantile(d$wait_time_minutes, 0.99, na.rm = TRUE)
d[wait_time_minutes > cap, wait_time_minutes := cap]
cat(sprintf("Capped wait_time at %.0f min (99th percentile)\n", cap))

# ============================================================
# Wait time by acuity
# ============================================================
cat("\n[1] Wait time by acuity...\n")
dir.create("outputs/wait", showWarnings = FALSE)
summary_wt <- d[, .(mean = mean(wait_time_minutes),
                    median = median(wait_time_minutes),
                    sd = sd(wait_time_minutes),
                    N = .N), by = target_triage_acuity]
cat("  Wait time by ESI level:\n")
print(summary_wt)
saveRDS(summary_wt, "outputs/wait/summary_by_acuity.rds")

# ============================================================
# Wait time by hour of day
# ============================================================
cat("\n[2] Wait time by hour of day...\n")
d[, hour := as.integer(substr(sprintf("%04d", arrival_time), 1, 2))]
wt_hour <- d[!is.na(hour), .(mean = mean(wait_time_minutes),
                             median = median(wait_time_minutes),
                             N = .N), by = hour][order(hour)]
cat("  Wait time by hour:\n")
print(head(wt_hour, 24))
saveRDS(wt_hour, "outputs/wait/summary_by_hour.rds")

# Plot
p1 <- ggplot(wt_hour, aes(x = hour, y = median)) +
  geom_line(color = "#2166AC", linewidth = 1.2) +
  geom_point(color = "#2166AC", size = 2.5) +
  labs(title = "Median Wait Time by Hour of Day",
       x = "Hour of Day (24h)", y = "Median Wait (minutes)") +
  theme_minimal(base_size = 12) +
  scale_x_continuous(breaks = seq(0, 23, 2))
ggsave("outputs/wait/wait_by_hour.pdf", p1, width = 10, height = 5)
cat("  Saved: outputs/wait/wait_by_hour.pdf\n")

# ============================================================
# Wait time by day of week
# ============================================================
cat("\n[3] Wait time by day of week...\n")
wt_dow <- d[, .(mean = mean(wait_time_minutes),
                median = median(wait_time_minutes),
                N = .N), by = day_of_week][order(day_of_week)]
wt_dow[, day_name := c("Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat")[day_of_week]]
cat("  Wait time by day of week:\n")
print(wt_dow)

# ============================================================
# Wait time by month
# ============================================================
cat("\n[4] Wait time by month...\n")
wt_month <- d[, .(mean = mean(wait_time_minutes),
                  median = median(wait_time_minutes),
                  N = .N), by = visit_month][order(visit_month)]
cat("  Wait time by month:\n")
print(wt_month)

# ============================================================
# Wait time by region
# ============================================================
cat("\n[5] Wait time by region...\n")
wt_region <- d[!is.na(region), .(mean = mean(wait_time_minutes),
                                 median = median(wait_time_minutes),
                                 N = .N), by = region][order(-N)]
cat("  Wait time by region:\n")
print(wt_region)

# ============================================================
# Wait time by EMS arrival
# ============================================================
cat("\n[6] Wait time by EMS arrival...\n")
wt_ems <- d[, .(mean = mean(wait_time_minutes),
                median = median(wait_time_minutes),
                N = .N), by = ems_arrival]
cat("  Wait time by EMS arrival:\n")
print(wt_ems)

# ============================================================
# Wait time by age group
# ============================================================
cat("\n[7] Wait time by age group...\n")
d[, age_group := cut(age, breaks = c(0, 18, 40, 65, 120),
                     labels = c("0-17", "18-39", "40-64", "65+"))]
wt_age <- d[!is.na(age_group), .(mean = mean(wait_time_minutes),
                                 median = median(wait_time_minutes),
                                 N = .N), by = age_group]
cat("  Wait time by age group:\n")
print(wt_age)

# ============================================================
# Wait time regression
# ============================================================
cat("\n[8] Wait time regression with XGBoost...\n")

# Convert categoricals - all character columns
cat_cols <- names(d)[sapply(d, is.character)]
for (col in cat_cols) {
  d[[col]] <- as.numeric(factor(d[[col]]))
}

# Drop text columns and target
text_cols <- c("chief_complaint_text", "injury_cause_text",
               "target_triage_acuity", "wait_time_minutes", "age_group")
features <- setdiff(names(d), text_cols)

X_wt <- as.matrix(d[, ..features])
y_wt <- d$wait_time_minutes

# Remove columns with no variance
nzv <- apply(X_wt, 2, function(x) var(x, na.rm = TRUE) > 0)
X_wt <- X_wt[, nzv]
cat(sprintf("  Features: %d (after removing zero-var)\n", ncol(X_wt)))

# Train/test split
set.seed(42)
idx_wt <- sample(nrow(X_wt), nrow(X_wt) * 0.8)
X_train_wt <- X_wt[idx_wt, ]; X_test_wt <- X_wt[-idx_wt, ]
y_train_wt <- y_wt[idx_wt];   y_test_wt <- y_wt[-idx_wt]

dtrain_wt <- xgb.DMatrix(X_train_wt, label = y_train_wt)
dtest_wt  <- xgb.DMatrix(X_test_wt,  label = y_test_wt)

params_wt <- list(
  objective = "reg:squarederror", max_depth = 6, eta = 0.05,
  subsample = 0.85, colsample_bytree = 0.85,
  min_child_weight = 5, reg_lambda = 2.0, eval_metric = "rmse"
)
model_wt <- xgb.train(params_wt, dtrain_wt, nrounds = 300, verbose = 0)

pred_wt <- predict(model_wt, dtest_wt)

# Metrics
mae_wt <- mean(abs(y_test_wt - pred_wt))
rmse_wt <- sqrt(mean((y_test_wt - pred_wt)^2))
ss_res <- sum((y_test_wt - pred_wt)^2)
ss_tot <- sum((y_test_wt - mean(y_test_wt))^2)
r2_wt <- 1 - ss_res / ss_tot
spearman_wt <- cor(y_test_wt, pred_wt, method = "spearman")

cat(sprintf("  Wait time regression: MAE=%.2f, RMSE=%.2f, R2=%.3f, Spearman=%.3f\n",
            mae_wt, rmse_wt, r2_wt, spearman_wt))

# Importance
imp_wt <- xgb.importance(feature_names = colnames(X_wt), model = model_wt)
fwrite(imp_wt, "outputs/wait/feature_importance.csv")
cat("  Top 10 wait time features:\n")
print(head(imp_wt, 10))

# Save model
saveRDS(model_wt, "outputs/r_xgboost_wait_time.rds")

# ============================================================
# Plots
# ============================================================
cat("\n[9] Generating plots...\n")

# Predicted vs actual
p2 <- ggplot(data.frame(actual = y_test_wt, predicted = pred_wt),
             aes(x = actual, y = predicted)) +
  geom_point(alpha = 0.1, color = "#2166AC") +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  coord_cartesian(xlim = c(0, 100), ylim = c(0, 100)) +
  labs(title = "Wait Time: Predicted vs Actual",
       subtitle = sprintf("R2=%.3f, MAE=%.1f min", r2_wt, mae_wt),
       x = "Actual (minutes)", y = "Predicted (minutes)") +
  theme_minimal(base_size = 12)
ggsave("outputs/wait/predicted_vs_actual.pdf", p2, width = 7, height = 6)
cat("  Saved: outputs/wait/predicted_vs_actual.pdf\n")

# Top 15 importance
top15 <- head(imp_wt, 15)
top15$Feature <- factor(top15$Feature, levels = rev(top15$Feature))
p3 <- ggplot(top15, aes(x = Feature, y = Gain)) +
  geom_bar(stat = "identity", fill = "#B35806") +
  coord_flip() +
  labs(title = "Top 15 Wait Time Predictors",
       subtitle = "XGBoost feature importance (gain)",
       x = "Feature", y = "Gain") +
  theme_minimal(base_size = 12)
ggsave("outputs/wait/importance.pdf", p3, width = 9, height = 6)
cat("  Saved: outputs/wait/importance.pdf\n")

# Save all summaries
saveRDS(list(by_acuity = summary_wt, by_hour = wt_hour,
             by_dow = wt_dow, by_month = wt_month,
             by_region = wt_region, by_ems = wt_ems,
             by_age = wt_age, regression = list(mae = mae_wt, rmse = rmse_wt,
             r2 = r2_wt, spearman = spearman_wt)),
        "outputs/r_wait_time_summary.rds")

cat("\nDONE.\n")
