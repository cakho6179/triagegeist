suppressMessages({
  library(data.table)
  library(ggplot2)
})

cat("========================================\n")
cat("BIAS / FAIRNESS ANALYSIS (Section 17)\n")
cat("========================================\n\n")

d <- fread("data/raw/nhamcs_data_2018_22.csv")
cat(sprintf("Loaded %d rows from raw data\n", nrow(d)))

# ============================================================
# Distribution analysis by demographic groups
# ============================================================
cat("\n[1] Acuity distribution by demographic groups...\n")
dir.create("outputs/bias", showWarnings = FALSE)

# By age group
d[, age_group := cut(age, breaks = c(0, 18, 40, 65, 120),
                     labels = c("0-17", "18-39", "40-64", "65+"))]
cat("  Age group distribution:\n")
print(d[, .N, by = age_group])

# By sex
cat("\n  Sex distribution:\n")
print(d[, .N, by = sex])

# By race
cat("\n  Race distribution:\n")
print(d[, .N, by = race][order(-N)])

# By insurance
cat("\n  Insurance distribution:\n")
print(d[, .N, by = insurance][order(-N)])

# ============================================================
# Acuity by group
# ============================================================
cat("\n[2] Mean acuity by demographic groups...\n")
cat("\n  Mean acuity by age group:\n")
print(d[, .(mean_acuity = mean(target_triage_acuity, na.rm = TRUE),
           median_wait = median(wait_time_minutes, na.rm = TRUE),
           N = .N), by = age_group])

cat("\n  Mean acuity by sex:\n")
print(d[, .(mean_acuity = mean(target_triage_acuity, na.rm = TRUE),
           median_wait = median(wait_time_minutes, na.rm = TRUE),
           N = .N), by = sex])

cat("\n  Mean acuity by race:\n")
print(d[, .(mean_acuity = mean(target_triage_acuity, na.rm = TRUE),
           median_wait = median(wait_time_minutes, na.rm = TRUE),
           N = .N), by = race][order(-N)])

cat("\n  Mean acuity by insurance:\n")
print(d[, .(mean_acuity = mean(target_triage_acuity, na.rm = TRUE),
           median_wait = median(wait_time_minutes, na.rm = TRUE),
           N = .N), by = insurance][order(-N)])

cat("\n  Mean acuity by EMS arrival:\n")
print(d[, .(mean_acuity = mean(target_triage_acuity, na.rm = TRUE),
           median_wait = median(wait_time_minutes, na.rm = TRUE),
           N = .N), by = ems_arrival])

# ============================================================
# Chi-square tests
# ============================================================
cat("\n[3] Chi-square tests of independence...\n")
chi_test <- function(var) {
  if (!is.factor(d[[var]])) d[[var]] <- as.factor(d[[var]])
  ct <- table(d[[var]], d$target_triage_acuity)
  ct <- ct[rowSums(ct) > 0, colSums(ct) > 0, drop = FALSE]
  test <- suppressWarnings(chisq.test(ct))
  sprintf("  %s vs acuity: chi2=%.2f, df=%d, p=%.2e, V=%.3f",
          var, test$statistic, test$parameter, test$p.value,
          sqrt(test$statistic / (sum(ct) * (min(dim(ct)) - 1))))
}

cat(chi_test("age_group"), "\n")
cat(chi_test("sex"), "\n")
cat(chi_test("race"), "\n")
cat(chi_test("insurance"), "\n")
cat(chi_test("ems_arrival"), "\n")

# ============================================================
# Kruskal-Wallis for wait time
# ============================================================
cat("\n[4] Kruskal-Wallis tests for wait time...\n")
kw_test <- function(var) {
  groups <- split(d$wait_time_minutes, d[[var]])
  groups <- groups[sapply(groups, function(x) length(x) > 0 && !all(is.na(x)))]
  test <- suppressWarnings(kruskal.test(groups))
  sprintf("  %s vs wait_time: H=%.2f, df=%d, p=%.2e",
          var, test$statistic, test$parameter, test$p.value)
}

cat(kw_test("age_group"), "\n")
cat(kw_test("sex"), "\n")
cat(kw_test("race"), "\n")
cat(kw_test("insurance"), "\n")
cat(kw_test("ems_arrival"), "\n")

# ============================================================
# Visualizations
# ============================================================
cat("\n[5] Generating plots...\n")

# Plot 1: Acuity by age group
p1 <- ggplot(d[!is.na(age_group)], aes(x = age_group, fill = factor(target_triage_acuity))) +
  geom_bar(position = "fill") +
  scale_fill_brewer(palette = "RdYlGn", name = "ESI") +
  labs(title = "Acuity Distribution by Age Group",
       x = "Age Group", y = "Proportion") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")
ggsave("outputs/bias/acuity_by_age.pdf", p1, width = 8, height = 5)
cat("  Saved: outputs/bias/acuity_by_age.pdf\n")

# Plot 2: Acuity by EMS
p2 <- ggplot(d[!is.na(ems_arrival)], aes(x = ems_arrival, fill = factor(target_triage_acuity))) +
  geom_bar(position = "fill") +
  scale_fill_brewer(palette = "RdYlGn", name = "ESI") +
  labs(title = "Acuity Distribution by EMS Arrival",
       x = "EMS Arrival", y = "Proportion") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")
ggsave("outputs/bias/acuity_by_ems.pdf", p2, width = 8, height = 5)
cat("  Saved: outputs/bias/acuity_by_ems.pdf\n")

# Plot 3: Wait time by acuity
p3 <- ggplot(d[!is.na(wait_time_minutes) & !is.na(target_triage_acuity)],
             aes(x = factor(target_triage_acuity), y = wait_time_minutes,
                 fill = factor(target_triage_acuity))) +
  geom_boxplot(outlier.size = 0.5) +
  scale_fill_brewer(palette = "RdYlGn", name = "ESI") +
  labs(title = "Wait Time by Acuity Level",
       subtitle = "Lower acuity → shorter wait expected",
       x = "Acuity (1=Most Urgent)", y = "Wait Time (minutes)") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none") +
  coord_cartesian(ylim = c(0, 200))
ggsave("outputs/bias/wait_by_acuity.pdf", p3, width = 8, height = 5)
cat("  Saved: outputs/bias/wait_by_acuity.pdf\n")

# Plot 4: Wait time by insurance
p4 <- ggplot(d[!is.na(wait_time_minutes) & !is.na(insurance)],
             aes(x = reorder(insurance, wait_time_minutes, FUN = median, na.rm = TRUE),
                 y = wait_time_minutes, fill = insurance)) +
  geom_boxplot(outlier.size = 0.5) +
  labs(title = "Wait Time by Insurance Type",
       x = "Insurance", y = "Wait Time (minutes)") +
  theme_minimal(base_size = 11) +
  theme(legend.position = "none", axis.text.x = element_text(angle = 30, hjust = 1)) +
  coord_cartesian(ylim = c(0, 200))
ggsave("outputs/bias/wait_by_insurance.pdf", p4, width = 9, height = 5)
cat("  Saved: outputs/bias/wait_by_insurance.pdf\n")

# Plot 5: Acuity by race
p5 <- ggplot(d[!is.na(race)], aes(x = reorder(race, target_triage_acuity, FUN = mean),
                                   fill = factor(target_triage_acuity))) +
  geom_bar(position = "fill") +
  scale_fill_brewer(palette = "RdYlGn", name = "ESI") +
  labs(title = "Acuity Distribution by Race",
       x = "Race", y = "Proportion") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom", axis.text.x = element_text(angle = 30, hjust = 1))
ggsave("outputs/bias/acuity_by_race.pdf", p5, width = 9, height = 5)
cat("  Saved: outputs/bias/acuity_by_race.pdf\n")

# Plot 6: Wait time by sex
p6 <- ggplot(d[!is.na(wait_time_minutes) & !is.na(sex)],
             aes(x = sex, y = wait_time_minutes, fill = sex)) +
  geom_boxplot(outlier.size = 0.5) +
  labs(title = "Wait Time by Sex",
       x = "Sex", y = "Wait Time (minutes)") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none") +
  coord_cartesian(ylim = c(0, 200))
ggsave("outputs/bias/wait_by_sex.pdf", p6, width = 7, height = 5)
cat("  Saved: outputs/bias/wait_by_sex.pdf\n")

# ============================================================
# Save summary
# ============================================================
cat("\n[6] Saving bias analysis summary...\n")
summary_stats <- list(
  age_group = d[, .(mean_acuity = mean(target_triage_acuity, na.rm = TRUE),
                    median_wait = median(wait_time_minutes, na.rm = TRUE),
                    N = .N), by = age_group],
  sex = d[, .(mean_acuity = mean(target_triage_acuity, na.rm = TRUE),
              median_wait = median(wait_time_minutes, na.rm = TRUE),
              N = .N), by = sex],
  race = d[, .(mean_acuity = mean(target_triage_acuity, na.rm = TRUE),
               median_wait = median(wait_time_minutes, na.rm = TRUE),
               N = .N), by = race][order(-N)],
  insurance = d[, .(mean_acuity = mean(target_triage_acuity, na.rm = TRUE),
                    median_wait = median(wait_time_minutes, na.rm = TRUE),
                    N = .N), by = insurance][order(-N)],
  ems = d[, .(mean_acuity = mean(target_triage_acuity, na.rm = TRUE),
              median_wait = median(wait_time_minutes, na.rm = TRUE),
              N = .N), by = ems_arrival]
)
saveRDS(summary_stats, "outputs/r_bias_summary.rds")
cat("  Saved: outputs/r_bias_summary.rds\n")
cat("\nDONE.\n")
