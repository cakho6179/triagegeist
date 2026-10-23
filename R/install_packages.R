# Install R packages for Triagegeist explainable modeling
packages <- c(
  # Data manipulation
  "data.table", "dplyr", "tidyr", "stringr", "janitor",
  # Visualization
  "ggplot2", "corrplot", "gridExtra", "patchwork", "ggridges",
  # Modeling
  "xgboost", "ranger", "randomForest", "rpart", "rpart.plot",
  # Interpretation / Explainability
  "DALEX", "DALEXtra", "vip", "pdp", "shapr", "iml",
  # Evaluation
  "caret", "Metrics"
)

to_install <- packages[!packages %in% installed.packages()[,"Package"]]
if (length(to_install) > 0) {
  cat("Installing:", paste(to_install, collapse=", "), "\n")
  install.packages(to_install, repos="https://cloud.r-project.org")
} else {
  cat("All packages already installed.\n")
}
