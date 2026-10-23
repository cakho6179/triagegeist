# ====================================================================
# TRIAGEGEIST - Master R Pipeline
# Runs all 5 parts of the proposal in R
# ====================================================================
# Usage:
#   "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" R/run_all.R
# ====================================================================

cat("\n")
cat("==============================================================\n")
cat("   TRIAGEGEIST - R Analysis Pipeline (5 bai)\n")
cat("==============================================================\n\n")
cat("Bai 1: EDA & Data Understanding\n")
cat("Bai 2: Modeling (E1-E5)\n")
cat("Bai 3: Ablation (E6)\n")
cat("Bai 4: Explainability & Bias\n")
cat("Bai 5: Wait Time Analysis\n")
cat("\n")

t_start <- Sys.time()

# ====================================================================
# Bai 1: EDA
# ====================================================================
cat("\n[BAI 1/5] =========================================\n")
cat("[BAI 1/5] Running EDA...\n")
cat("[BAI 1/5] =========================================\n")
source("R/eda.R")
cat(sprintf("\n[BAI 1/5] DONE in %.1f min\n",
            as.numeric(difftime(Sys.time(), t_start, units = "mins"))))

# ====================================================================
# Bai 2: Experiments E1-E5
# ====================================================================
cat("\n[BAI 2/5] =========================================\n")
cat("[BAI 2/5] Running Experiments E1-E5...\n")
cat("[BAI 2/5] =========================================\n")
source("R/experiments.R")
cat(sprintf("\n[BAI 2/5] DONE in %.1f min\n",
            as.numeric(difftime(Sys.time(), t_start, units = "mins"))))

# ====================================================================
# Bai 3: Ablation (E6) - already part of experiments.R
# See outputs/r_experiments_summary.csv for E6 results
# ====================================================================
cat("\n[BAI 3/5] =========================================\n")
cat("[BAI 3/5] Ablation E6 results (already computed in Bai 2)\n")
cat("[BAI 3/5] =========================================\n")
e6_summary <- fread("outputs/r_experiments_summary.csv")
print(e6_summary[grep("E6", Experiment)])
cat(sprintf("\n[BAI 3/5] DONE in %.1f min\n",
            as.numeric(difftime(Sys.time(), t_start, units = "mins"))))

# ====================================================================
# Bai 4: Explainability + Bias
# ====================================================================
cat("\n[BAI 4/5] =========================================\n")
cat("[BAI 4/5] Running SHAP analysis...\n")
cat("[BAI 4/5] =========================================\n")
source("R/compute_shap.R")
cat(sprintf("\n[BAI 4/5] SHAP done at %.1f min\n",
            as.numeric(difftime(Sys.time(), t_start, units = "mins"))))

cat("\n[BAI 4/5] Running Bias/Fairness analysis...\n")
source("R/bias_analysis.R")
cat(sprintf("\n[BAI 4/5] DONE in %.1f min\n",
            as.numeric(difftime(Sys.time(), t_start, units = "mins"))))

# ====================================================================
# Bai 5: Wait Time Analysis
# ====================================================================
cat("\n[BAI 5/5] =========================================\n")
cat("[BAI 5/5] Running Wait Time Analysis...\n")
cat("[BAI 5/5] =========================================\n")
source("R/wait_time_analysis.R")
cat(sprintf("\n[BAI 5/5] DONE in %.1f min\n",
            as.numeric(difftime(Sys.time(), t_start, units = "mins"))))

# ====================================================================
# Done
# ====================================================================
elapsed <- as.numeric(difftime(Sys.time(), t_start, units = "mins"))
cat("\n")
cat("==============================================================\n")
cat(sprintf("   ALL 5 BAI COMPLETE in %.1f minutes\n", elapsed))
cat("==============================================================\n")
cat("\nOutputs:\n")
cat("  outputs/eda/        - EDA plots and summaries\n")
cat("  outputs/bias/       - Bias/fairness plots\n")
cat("  outputs/wait/       - Wait time analysis plots\n")
cat("  outputs/r_experiments_summary.csv - E1-E6 results\n")
cat("  outputs/r_best_xgboost_model.rds  - Best model\n")
cat("  outputs/r_xgb_thresholds.rds       - Best thresholds\n")
cat("  outputs/r_shap_summary.pdf         - SHAP summary\n")
cat("==============================================================\n")
