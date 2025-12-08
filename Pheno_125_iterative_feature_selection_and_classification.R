###############################################################################
# Generic Data Analysis Pipeline: Application to pain QST data
#
# This script loads, preprocesses, and analyzes datasets for classification.
# It handles feature selection, correlation analysis, and machine learning
# classification with multiple algorithms and confidence intervals.
###############################################################################

# --- Libraries ----------------------------------------------------------------
library(parallel)
library(randomForest)
library(caret)
library(C50)
library(partykit)
library(pbmcapply)
library(Boruta)
library(reshape2)
library(pROC)
library(dplyr)
library(glmnet)
library(car)
library(tidyr)
library(ggplot2)
library(patchwork)

###############################################################################
# Configuration Parameters
###############################################################################

# External functions
FUNCTIONS_FILE_PATH <- "/home/joern/.Datenplatte/Joerns Dateien/Aktuell/ABCPython/08AnalyseProgramme/R/ABC2way/feature_selection_and_classification_functions.R"

# Dataset name
DATASET_NAME <- "Pheno_125"
EXPERIMENTS_DIR <- "/home/joern/.Datenplatte/Joerns Dateien/Aktuell/ABCPython/08AnalyseProgramme/R/ABC2way/"

# Dataset paths (now only training + validation CSVs)
base_path <- "/home/joern/Dokumente/PainGenesDrugs"
r_path <- "08AnalyseProgramme/R"

train_file <- file.path(base_path, r_path, "PainThresholds_scaled_Training.csv")
val_file <- file.path(base_path, r_path, "PainThresholds_scaled_Test.csv")

# Analysis parameters
SEED <- 42
noise_factor <- 0.2
CORRELATION_METHOD <- "pearson"
CORRELATION_LIMIT <- 0.9
Boruta_tentative_in <- FALSE
use_nyt <- TRUE
tune_RF <- TRUE
tune_KNN <- TRUE
tune_SVM <- TRUE
mtry_12only <- FALSE

use_curated <- FALSE
use_roc_auc <- FALSE
training_and_validation_subsplits <- TRUE
TRAINING_PARTITION_SIZE <- 0.8
VALIDATION_PARTITION_SIZE <- 0.8

max_iterations <- 5
RUN_ONE_ADDITIONAL_ITERATION <- FALSE

# Dataset-specific configuration  
DATASET_COLUMN_NAMES <- c(
  "Heat", "Pressure", "Current", "Heat_Capsaicin",
  "Capsaicin_Effect_Heat", "Cold", "Cold_Menthol", "Menthol_Effect_Cold",
  "vonFrey", "vonFrey_Capsaicin", "Capsaicin_Effect_vonFrey"
)

CURATED_COLUMN_NAMES <- c(
  "Heat", "Pressure", "Current", "Heat_Capsaicin",
  "Cold", "Cold_Menthol",
  "vonFrey", "vonFrey_Capsaicin"
)

COLUMNS_COLINEAR <- NULL # Will be determined later

# Noise addition parameters
ADD_NOISE_COLUMN <- TRUE
NOISE_COLUMN_SOURCE <- "Pressure"
NOISE_COLUMN_NAME <- "Pressure2"

###############################################################################
# Load External Functions
###############################################################################
if (file.exists(FUNCTIONS_FILE_PATH)) {
  source(FUNCTIONS_FILE_PATH)
} else {
  stop(paste("Functions file not found:", FUNCTIONS_FILE_PATH))
}

set_working_directory(EXPERIMENTS_DIR)

###############################################################################
# Check data files existence
###############################################################################
if (!file.exists(train_file)) {
  stop(paste("Training data file not found:", train_file))
}
if (!file.exists(val_file)) {
  stop(paste("Validation data file not found:", val_file))
}


###############################################################################
# Load data and modify files when needed
###############################################################################

# --- Define paths ---
base_path <- "/home/joern/Dokumente/PainGenesDrugs"
r_path <- "08AnalyseProgramme/R"

train_file <- file.path(base_path, r_path, "PainThresholds_scaled_Training.csv")
val_file <- file.path(base_path, r_path, "PainThresholds_scaled_Test.csv")

# --- Load datasets ---
train_df <- read.csv(train_file, row.names = 1)
val_df <- read.csv(val_file, row.names = 1)

# --- Sizes to later reseparate ---
n_train <- nrow(train_df)
n_val <- nrow(val_df)

# --- Extract features + targets ---
train_features <- train_df[, - ncol(train_df)]
train_target <- train_df[, ncol(train_df)]

val_features <- val_df[, - ncol(val_df)]
val_target <- val_df[, ncol(val_df)]

# --- Combine into pain_data + target_data ---
pain_data <- rbind(train_features, val_features)
target_data <- c(train_target, val_target)

# --- Rename columns once, for the combined dataset ---
pain_data <- rename_pain_data_columns(pain_data, DATASET_COLUMN_NAMES)

# --- Add noise ONCE (so all subsets share the same noise values) ---
set.seed(SEED)
pain_data[[NOISE_COLUMN_NAME]] <- pain_data[[NOISE_COLUMN_SOURCE]] +
  rnorm(
    n = nrow(pain_data),
    mean = 0,
    sd = abs(pain_data[[NOISE_COLUMN_SOURCE]]) * noise_factor
  )

# --- Reseparate into train/validation, consistent with original splits ---
training_data_original <- pain_data[1:n_train,]
validation_data_original <- pain_data[(n_train + 1):(n_train + n_val),]

training_target <- target_data[1:n_train]
validation_target <- target_data[(n_train + 1):(n_train + n_val)]

# --- Optional curation after renaming + noise ---
if (use_curated) {
  training_data_original <- training_data_original[, CURATED_COLUMN_NAMES]
  validation_data_original <- validation_data_original[, CURATED_COLUMN_NAMES]
  pain_data <- pain_data[, CURATED_COLUMN_NAMES]
}

# Save data file for use in other analyses
Pheno_125_prepared_data <- cbind.data.frame(Target = target_data, pain_data)
# write.csv(Pheno_125_prepared_data, "Pheno_125_prepared_data.csv")

###############################################################################
# Run analysis sequentially
###############################################################################

start_time <- Sys.time()
Pheno125_results_list <- run_feature_selection_iterations()
end_time <- Sys.time()
elapsed_time <- end_time - start_time
print(elapsed_time)

###############################################################################
# Combine plots from 'Full Dataset' configuration
###############################################################################

library(patchwork)

# Function to combine and save plots for a given iteration or "Full dataset"
combine_and_save_plots <- function(Pheno125_results_list, iteration = "Full dataset", add_file_string = "", ylim = NULL) {
  # Extract plots based on iteration
  plots <- if (iteration == "Full dataset") {
    Pheno125_results_list$Phase_0_Full$plots
  } else {
    Pheno125_results_list$Phase_3_Final_Selected$plots
  }

  matrix_plot <- plots$matrix
  summary_plot <- plots$summary
  boruta_plot <- plots$boruta
  if (!is.null(ylim)) lasso_plot <- plots$lasso + ylim(ylim_full) else lasso_plot <- plots$lasso

  # Compose right column (matrix + summary stacked)
  right_column <- matrix_plot / summary_plot + plot_layout(heights = c(2, 1))

  # Compose full plot layout
  combined_plot <- boruta_plot + lasso_plot + right_column +
    plot_layout(widths = c(2, 2, 1)) +
    plot_annotation(tag_levels = 'A')

  print(combined_plot)

  # Compose filenames
  suffix <- if (iteration == "Full dataset") "" else paste0("_", iteration)
  png_file <- paste0(DATASET_NAME, "_feature_selection_comparison", suffix, add_file_string, ".png")
  svg_file <- paste0(DATASET_NAME, "_feature_selection_comparison", suffix, add_file_string, ".svg")

  # Save to files
  ggsave(png_file, plot = combined_plot, width = 14, height = 8, dpi = 300)
  ggsave(svg_file, plot = combined_plot, width = 14, height = 8)

  invisible(combined_plot)
}

# For full dataset
combine_and_save_plots(Pheno125_results_list$results_list, "Full dataset")

# Get y axis limits for LASSO plot to forward it to plot 2
# ylim_full <- ggplot_build(Pheno125_results_list$results_list$Phase_0_Full$plots$lasso)$layout$panel_scales_y[[1]]$range$range
ylim_full <- c(0,0.8)

# For first iteration of curated subset
combine_and_save_plots(Pheno125_results_list$results_list, "other")

###############################################################################
# Run logistic regression on all datasets, collect results
###############################################################################

cat("\n", paste(rep("=", 80), collapse = ""), "\n")
cat("SIMPLE LOGISTIC REGRESSION ANALYSIS\n")
cat("\n", paste(rep("=", 80), collapse = ""), "\n")

datasets_to_test <- Pheno125_results_list$results_list$`Full Dataset`$datasets_to_test

logistic_results <- list()

# Run logistic regression per dataset split
for (name in names(datasets_to_test)) {
  if (nrow(datasets_to_test[[name]]$train) > 0 && ncol(datasets_to_test[[name]]$train) > 0) {
    logistic_results[[name]] <- run_single_logistic_regression(
      datasets_to_test[[name]]$train,
      training_target,
      name
    )
  } else {
    cat(sprintf("\nSkipping %s - no data available\n", name))
  }
}

# Run logistic regression variants on original data
for (i in 1:2) {
  if (i == 2) sink(paste0(DATASET_NAME, "_lr_orig_output", ".txt"))

  r1 <- run_single_logistic_regression(
    train_data = pain_data,
    train_target = target_data,
    dataset_name = "Original unsplit modified"
  )
  if (is.null(COLUMNS_COLINEAR)) {
    COLUMNS_COLINEAR <- rownames(alias(r1)$Complete)
    cat("\nCOLUMNS_COLINEAR\n", COLUMNS_COLINEAR)
  }

  run_single_logistic_regression(
    train_data = pain_data[, !names(pain_data) %in% c("Pressure2")],
    train_target = target_data,
    dataset_name = "Original unsplit unmodified"
  )

  run_single_logistic_regression(
    train_data = training_data_original,
    train_target = train_target,
    dataset_name = "Training split modified"
  )

  run_single_logistic_regression(
    train_data = training_data_original[, !names(training_data_original) %in% c("Pressure2")],
    train_target = train_target,
    dataset_name = "Training split unmodified"
  )

  run_single_logistic_regression(
    train_data = pain_data[, !names(pain_data) %in% c("Pressure2", names(which(is.na(r1$coefficients))))],
    train_target = target_data,
    dataset_name = "Original unsplit modified VIF removed"
  )

  run_single_logistic_regression(
    train_data = pain_data[, names(pain_data) %in% c("Pressure2", COLUMNS_COLINEAR)],
    train_target = target_data,
    dataset_name = "Original unsplit only modifed variables"
  )

  if (i == 2) sink()
}

cat("\n", paste(rep("=", 80), collapse = ""), "\n")
cat("Logistic regression analysis completed!\n")
cat("\n", paste(rep("=", 80), collapse = ""), "\n")

###############################################################################
# Calculate Cohen's d and t-tests for both datasets, then plot
###############################################################################

# Calculate Cohen's d and t-tests - assumed function calculate_cohens_d_with_ttest() available
cohens_d_original <- calculate_cohens_d_with_ttest(pain_data, target_data, "Original dataset")
cohens_d_training <- calculate_cohens_d_with_ttest(training_data_original, training_target, "Training dataset")

# Plot combined Cohen's d effect sizes
cohens_d_results <- plot_cohens_d(
  cohens_d_list = list(cohens_d_original, cohens_d_training),
  dataset_names = c("Original dataset", "Training dataset")
)

print(cohens_d_results$plot)

# Save plot
ggsave(
  plot = cohens_d_results$plot,
  filename = paste0(DATASET_NAME, "_cohens_d_with_ttests", ".svg"),
  width = 13,
  height = 7
)

# Print summary statistics
cat("\n=== COHEN'S D AND T-TEST SUMMARY ===\n")
summary_stats <- cohens_d_results$combined_data %>%
  group_by(Dataset) %>%
  summarise(
    Mean_Cohens_d = mean(abs(Cohens_d), na.rm = TRUE),
    Median_Cohens_d = median(abs(Cohens_d), na.rm = TRUE),
    Max_Cohens_d = max(abs(Cohens_d), na.rm = TRUE),
    Variables_Significant = sum(p_value < 0.05, na.rm = TRUE),
    Variables_Small_Effect = sum(abs(Cohens_d) >= 0.2 & abs(Cohens_d) < 0.5, na.rm = TRUE),
    Variables_Medium_Effect = sum(abs(Cohens_d) >= 0.5 & abs(Cohens_d) < 0.8, na.rm = TRUE),
    Variables_Large_Effect = sum(abs(Cohens_d) >= 0.8, na.rm = TRUE),
    .groups = "drop"
  )
print(summary_stats)

# Print detailed Cohen's d results
cat("\n=== DETAILED RESULTS ===\n")
detailed_results <- cohens_d_results$combined_data %>%
  arrange(desc(abs(Cohens_d))) %>%
  mutate(
    Effect_Size_Category = case_when(
      abs(Cohens_d) < 0.2 ~ "Negligible",
      abs(Cohens_d) < 0.5 ~ "Small",
      abs(Cohens_d) < 0.8 ~ "Medium",
      TRUE ~ "Large"
    )
  ) %>%
  select(Variable, Dataset, Cohens_d, CI_lower, CI_upper, t_statistic, p_value,
         p_label = p_label, Effect_Size_Category)
print(detailed_results)

# Save detailed Cohen's d results to CSV
write.csv(
  detailed_results,
  paste0(DATASET_NAME, "_cohens_d_with_ttests_results", ".csv"),
  row.names = FALSE
)

cat("\nFiles saved:\n")
cat("- cohens_d_with_ttests.svg\n")
cat("- cohens_d_with_ttests_results.csv\n")
