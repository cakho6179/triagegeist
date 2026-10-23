suppressMessages({
  library(data.table)
  library(ggplot2)
})

cat("========================================\n")
cat("EDA - EXPLORATORY DATA ANALYSIS (Section 10-11)\n")
cat("========================================\n\n")

# Load data
d_raw <- fread("data/raw/nhamcs_data_2018_22.csv")
d_proc <- fread("data/processed/triage_full.csv")

cat(sprintf("Raw: %d rows, %d columns\n", nrow(d_raw), ncol(d_raw)))
cat(sprintf("Processed: %d rows, %d columns\n", nrow(d_proc), ncol(d_proc)))

dir.create("outputs/eda", showWarnings = FALSE)

# ============================================================
# 1. Class distribution
# ============================================================
cat("\n[1] Class distribution...\n")
class_dist <- d_proc[, .N, by = triage_acuity][order(triage_acuity)]
class_dist[, pct := 100 * N / sum(N)]
cat("  ESI distribution:\n")
print(class_dist)
saveRDS(class_dist, "outputs/eda/class_dist.rds")

# Plot
p1 <- ggplot(class_dist, aes(x = factor(triage_acuity), y = N, fill = factor(triage_acuity))) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = sprintf("%.1f%%", pct)), vjust = -0.3, size = 4) +
  scale_fill_brewer(palette = "RdYlGn", guide = "none") +
  labs(title = "ESI Class Distribution (n=58,124)",
       subtitle = "Most urgent (1) -> Least urgent (5)",
       x = "ESI Level", y = "Count") +
  theme_minimal(base_size = 12)
ggsave("outputs/eda/class_dist.pdf", p1, width = 8, height = 5)
cat("  Saved: outputs/eda/class_dist.pdf\n")

# ============================================================
# 2. Missingness
# ============================================================
cat("\n[2] Missingness analysis (raw data)...\n")
miss_summary <- data.table(
  variable = names(d_raw),
  n_missing = sapply(d_raw, function(x) sum(is.na(x))),
  pct_missing = sapply(d_raw, function(x) 100 * mean(is.na(x))),
  type = sapply(d_raw, function(x) class(x)[1])
)
miss_summary <- miss_summary[order(-pct_missing)]
cat("  Top 20 most-missing variables:\n")
print(head(miss_summary, 20))
fwrite(miss_summary, "outputs/eda/missingness.csv")

# Plot
miss_top <- head(miss_summary[!is.na(pct_missing) & pct_missing > 0], 20)
if (nrow(miss_top) > 0) {
  miss_top$variable <- factor(miss_top$variable, levels = rev(miss_top$variable))
  p2 <- ggplot(miss_top, aes(x = variable, y = pct_missing)) +
    geom_bar(stat = "identity", fill = "#D7301F") +
    coord_flip() +
    labs(title = "Top 20 Most-Missing Variables",
         x = "Variable", y = "% Missing") +
    theme_minimal(base_size = 12)
  ggsave("outputs/eda/missingness.pdf", p2, width = 9, height = 7)
  cat("  Saved: outputs/eda/missingness.pdf\n")
}

# ============================================================
# 3. Vital signs summary by class
# ============================================================
cat("\n[3] Vital signs by ESI class...\n")
vitals <- c("temp", "heart_rate", "resp_rate", "sys_bp", "dias_bp", "spo2", "pain_score")
vital_summary <- d_raw[!is.na(target_triage_acuity),
                       lapply(.SD, function(x) c(mean = mean(x, na.rm = TRUE),
                                                  median = median(x, na.rm = TRUE),
                                                  sd = sd(x, na.rm = TRUE))),
                       .SDcols = vitals,
                       by = target_triage_acuity]
cat("  Vital signs by ESI:\n")
print(vital_summary)
saveRDS(vital_summary, "outputs/eda/vital_summary.rds")

# Plot: heart_rate by class
p3 <- ggplot(d_raw[!is.na(target_triage_acuity)],
             aes(x = factor(target_triage_acuity), y = heart_rate,
                 fill = factor(target_triage_acuity))) +
  geom_boxplot(outlier.size = 0.5) +
  scale_fill_brewer(palette = "RdYlGn", guide = "none") +
  labs(title = "Heart Rate by ESI Level",
       x = "ESI Level", y = "Heart Rate (bpm)") +
  theme_minimal(base_size = 12) +
  coord_cartesian(ylim = c(40, 160))
ggsave("outputs/eda/hr_by_class.pdf", p3, width = 8, height = 5)
cat("  Saved: outputs/eda/hr_by_class.pdf\n")

# Plot: pain_score by class
p4 <- ggplot(d_raw[!is.na(target_triage_acuity)],
             aes(x = factor(target_triage_acuity), y = pain_score,
                 fill = factor(target_triage_acuity))) +
  geom_boxplot(outlier.size = 0.5) +
  scale_fill_brewer(palette = "RdYlGn", guide = "none") +
  labs(title = "Pain Score by ESI Level",
       x = "ESI Level", y = "Pain Score (0-10)") +
  theme_minimal(base_size = 12)
ggsave("outputs/eda/pain_by_class.pdf", p4, width = 8, height = 5)
cat("  Saved: outputs/eda/pain_by_class.pdf\n")

# ============================================================
# 4. Demographics summary
# ============================================================
cat("\n[4] Demographics summary...\n")
cat("  Age:\n")
print(summary(d_raw$age))
cat("  Sex:\n")
print(d_raw[, .N, by = sex])
cat("  Race:\n")
print(d_raw[, .N, by = race][order(-N)])
cat("  Insurance:\n")
print(d_raw[, .N, by = insurance][order(-N)])

# Age histogram
p5 <- ggplot(d_raw[!is.na(age)], aes(x = age)) +
  geom_histogram(bins = 30, fill = "#4393C3", color = "white") +
  labs(title = "Age Distribution",
       subtitle = sprintf("Mean=%.1f, Median=%.0f", mean(d_raw$age, na.rm = TRUE),
                          median(d_raw$age, na.rm = TRUE)),
       x = "Age (years)", y = "Count") +
  theme_minimal(base_size = 12)
ggsave("outputs/eda/age_hist.pdf", p5, width = 8, height = 5)
cat("  Saved: outputs/eda/age_hist.pdf\n")

# ============================================================
# 5. Correlation of vitals with acuity
# ============================================================
cat("\n[5] Correlation of vitals with ESI...\n")
cor_results <- sapply(vitals, function(v) {
  if (all(is.na(d_raw[[v]]))) return(NA_real_)
  cor(d_raw[[v]], d_raw$target_triage_acuity, use = "complete.obs", method = "spearman")
})
cor_df <- data.table(variable = names(cor_results), spearman_corr = cor_results)
cor_df <- cor_df[order(-abs(spearman_corr))]
cat("  Spearman correlation with ESI:\n")
print(cor_df)
fwrite(cor_df, "outputs/eda/vital_correlation.csv")

# ============================================================
# 6. Chief complaint keyword analysis
# ============================================================
cat("\n[6] Chief complaint keyword frequency...\n")
cc <- d_raw$chief_complaint_text[!is.na(d_raw$chief_complaint_text)]
cat(sprintf("  Non-null chief complaints: %d / %d (%.1f%%)\n",
            length(cc), nrow(d_raw), 100 * length(cc) / nrow(d_raw)))

# Check key symptoms
keywords <- c("chest pain", "shortness of breath", "abdominal pain",
              "headache", "fever", "vomiting", "cough", "back pain",
              "dizziness", "syncope", "seizure", "bleeding", "trauma",
              "nausea", "weakness", "rash", "laceration", "fall")
for (kw in keywords) {
  count <- sum(grepl(kw, cc, ignore.case = TRUE))
  pct <- 100 * count / length(cc)
  cat(sprintf("  %-20s: %5d (%.1f%%)\n", kw, count, pct))
}

cat("\nDONE.\n")
