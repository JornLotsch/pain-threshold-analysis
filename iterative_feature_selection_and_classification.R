###############################################################################
# Generic Data Analysis Pipeline: Application to pain QST data
#
# This script loads, preprocesses, and analyzes datasets for classification.
# It handles feature selection, correlation analysis, and machine learning
# classification with multiple algorithms and confidence intervals.
###############################################################################

# --- Libraries ----------------------------------------------------------------
library(parallel)
library(opdisDownsampling)
library(randomForest)
library(caret)
library(pbmcapply)
library(Boruta)
library(reshape2)
library(pROC)
library(dplyr)
library(glmnet)
library(car)
library(tidyr)
library(ggplot2)
library(patchwork)   # for plot layout composition

###############################################################################
# Configuration Parameters (Modify these for your dataset)
###############################################################################

# External functions file path
FUNCTIONS_FILE_PATH <- "/home/joern/.Datenplatte/Joerns Dateien/Aktuell/ABCPython/08AnalyseProgramme/R/ABC2way/feature_selection_and_classification_functions.R"

# File paths - 
DATA_FILE_PATH <- "/home/joern/Dokumente/PainGenesDrugs/08AnalyseProgramme/R/PainThresholdsData_transformed_imputed.csv"
TARGET_FILE_PATH <- "/home/joern/Dokumente/PainGenesDrugs/08AnalyseProgramme/R/PainThresholds.csv"

# Analysis parameters
SEED <- 42
noise_factor <- 0.05
CORRELATION_METHOD <- "pearson"
CORRELATION_LIMIT <- 0.9
Boruta_tentative_in <- FALSE
use_nyt <- TRUE
tune_RF <- TRUE
use_curated <- FALSE
use_roc_auc <- FALSE
training_and_validation_subsplits <- TRUE
TRAINING_PARTITION_SIZE <- 0.8
VALIDATION_PARTITION_SIZE <- 0.8

# Dataset-specific configuration - 
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

COLUMNS_COLINEAR <- NULL  # Will be determined 

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

###############################################################################
# Main Execution
###############################################################################

# Check existence of data files
if (!file.exists(DATA_FILE_PATH)) {
  stop(paste("Data file not found:", DATA_FILE_PATH))
}
if (!file.exists(TARGET_FILE_PATH)) {
  stop(paste("Target file not found:", TARGET_FILE_PATH))
}

###############################################################################
# Train/Validation Split
###############################################################################

# Load data using external functions
pain_data <- load_pain_thresholds_data(DATA_FILE_PATH)
pain_data <- rename_pain_data_columns(pain_data, DATASET_COLUMN_NAMES)
target_data <- load_target_data(TARGET_FILE_PATH)

# Duplicate pressure variable with small noise addition
set.seed(SEED)
pain_data$Pressure2 <- pain_data$Pressure +
  runif(length(pain_data$Pressure),
        min = -abs(pain_data$Pressure) * noise_factor,
        max = abs(pain_data$Pressure) * noise_factor)

# Use only curated variables if specified
if (use_curated) pain_data <- pain_data[, CURATED_COLUMN_NAMES]

# Split into training and validation using opdisDownsampling package
data_split <- opdisDownsampling::opdisDownsampling(
  pain_data,
  Cls = target_data,
  Size = 0.8,
  Seed = SEED,
  nTrials = 2000000,
  MaxCores = parallel::detectCores() - 1
)

training_data_original <- data_split$ReducedData[, 1:(ncol(data_split$ReducedData) - 1)]
training_target <- data_split$ReducedData$Cls

validation_data_original <- data_split$RemovedData[, 1:(ncol(data_split$RemovedData) - 1)]
validation_target <- data_split$RemovedData$Cls

###############################################################################
# Run analysis sequentially
###############################################################################
# Run full dataset analysis first
full_config <- list(
  name = "Full dataset",
  use_curated = FALSE,
  curated_names = NULL
)

full_results <- run_analysis_pipeline(
  training_data_actual = training_data_original,
  training_target = training_target,
  validation_data_actual = validation_data_original,
  validation_target = validation_target,
  use_curated_subset = FALSE,
  curated_names = NULL,
  add_file_string = "_full"
)

# Initialize storage for all results
all_results_feature_selection <- list()
all_results_feature_selection[["Full dataset"]] <- full_results
print(all_results_feature_selection[[1]]$plots$matrix)

# Initialize curated features based on full results
boruta_res <- get_boruta_features(full_results$boruta_results$finalDecision, Boruta_tentative_in)
boruta_selected <- boruta_res$selected
lasso_selected <- full_results$lasso_results$selected
available_features <- names(training_data_original)
curated_features <- setdiff(available_features, union(boruta_selected, lasso_selected))

# Set max iterations and initialize counter
max_iterations <- 5
iteration <- 0

# Force start loop condition
classification_success_values <- c(1)

# Initialize list to hold each curated iteration's results separately
all_curated_results <- list()

while (length(curated_features) > 0 && any(classification_success_values != 0) && iteration < max_iterations) {
  iteration <- iteration + 1
  cat(sprintf("Iteration %d: running curated subset with %d features\n", iteration, length(curated_features)))
  
  curated_results <- run_analysis_pipeline(
    training_data_actual = training_data_original,
    training_target = training_target,
    validation_data_actual = validation_data_original,
    validation_target = validation_target,
    use_curated_subset = TRUE,
    curated_names = curated_features,
    add_file_string = paste0("_curated_iter", iteration)
  )
  
  # Save this iteration's results indexed by iteration
  all_curated_results[[paste0("Iter_", iteration)]] <- curated_results
  print(curated_results$plots$matrix)
  
  results_table <- curated_results$results_table
  
  # Identify rows with "rejected" in Dataset name (case insensitive)
  rejected_indices <- grepl("rejected", results_table$Dataset, ignore.case = TRUE)
  
  # Only continue if any rejected dataset has Classification_Success == 1
  continue_iteration <- any(results_table$Classification_Success[rejected_indices] == 1)
  
  if (continue_iteration) {
    # Remove selected features
    boruta_res <- get_boruta_features(curated_results$boruta_results$finalDecision, Boruta_tentative_in)
    boruta_selected <- boruta_res$selected
    lasso_selected <- curated_results$lasso_results$selected
    curated_features <- setdiff(curated_features, union(boruta_selected, lasso_selected))
    classification_success_values <- results_table$Classification_Success
  } else {
    cat("No rejected datasets with Classification_Success == 1, stopping iteration.\n")
    # Set to all zero to break while loop
    classification_success_values <- rep(0, length(results_table$Classification_Success))
  }
}

# # Run last iteration manually after the loop if desired
# boruta_res <- get_boruta_features(curated_results$boruta_results$finalDecision, Boruta_tentative_in)
# boruta_selected <- boruta_res$selected
# lasso_selected <- curated_results$lasso_results$selected
# curated_features <- setdiff(curated_features, union(boruta_selected, lasso_selected))
# 
# if (!(length(curated_features) > 0 && any(classification_success_values != 0) && iteration < max_iterations)) {
#   iteration <- iteration + 1
#   cat(sprintf("Final iteration %d: running curated subset with %d features\n", iteration, length(curated_features)))
#   curated_results <- run_analysis_pipeline(
#     training_data_actual = training_data_original,
#     training_target = training_target,
#     validation_data_actual = validation_data_original,
#     validation_target = validation_target,
#     use_curated_subset = TRUE,
#     curated_names = curated_features,
#     add_file_string = paste0("_curated_iter", iteration + 1)
#   )
#   all_curated_results[[paste0("Iter_", iteration)]] <- curated_results
#   print(curated_results$plots$matrix)
# }

# Store all curated iteration results in main results list
all_results_feature_selection[["Curated subset iterations"]] <- all_curated_results

if (iteration == max_iterations && any(classification_success_values != 0)) {
  cat("Max iterations reached but classification success is still 1 for some datasets.\n")
}

# Extract full results table and add a column to identify the iteration
full_results_table <- all_results_feature_selection[["Full dataset"]]$results_table
full_results_table$Iteration <- "Full dataset"

# Initialize combined table starting with full dataset results
combined_results_table <- full_results_table

# Append all curated iteration results tables with iteration names
for(iter_name in names(all_results_feature_selection[["Curated subset iterations"]])) {
  iter_table <- all_results_feature_selection[["Curated subset iterations"]][[iter_name]]$results_table
  iter_table$Iteration <- iter_name
  combined_results_table <- rbind(combined_results_table, iter_table)
}

# Save final results tables to disk
write.csv(combined_results_table,
          paste0("ML_results_df_table.csv"))

cat("\n=== ANALYSIS COMPLETE ===\n")
cat("All results stored in 'all_results_feature_selection' list\n")
cat("Each curated iteration available in all_results_feature_selection$`Curated subset iterations`$Iter_X\n")


###############################################################################
# Combine plots from 'Full Dataset' configuration
###############################################################################

library(patchwork)

# Function to combine and save plots for a given iteration or "Full dataset"
combine_and_save_plots <- function(results_list, iteration = "Full dataset", add_file_string = "") {
  # Extract plots based on iteration
  plots <- if (iteration == "Full dataset") {
    results_list[["Full dataset"]]$plots
  } else {
    results_list[["Curated subset iterations"]][[iteration]]$plots
  }
  
  matrix_plot  <- plots$matrix
  summary_plot <- plots$summary
  boruta_plot  <- plots$boruta
  lasso_plot   <- plots$lasso
  
  # Compose right column (matrix + summary stacked)
  right_column <- matrix_plot / summary_plot + plot_layout(heights = c(2, 1))
  
  # Compose full plot layout
  combined_plot <- boruta_plot + lasso_plot + right_column + 
    plot_layout(widths = c(2, 2, 1)) +
    plot_annotation(tag_levels = 'A')
  
  print(combined_plot)
  
  # Compose filenames
  suffix <- if (iteration == "Full dataset") "" else paste0("_", iteration)
  png_file <- paste0("feature_selection_comparison", suffix, add_file_string, ".png")
  svg_file <- paste0("feature_selection_comparison", suffix, add_file_string, ".svg")
  
  # Save to files
  ggsave(png_file, plot = combined_plot, width = 14, height = 8, dpi = 300)
  ggsave(svg_file, plot = combined_plot, width = 14, height = 8)
  
  invisible(combined_plot)
}

# Example usage:
# For full dataset
combine_and_save_plots(all_results_feature_selection, "Full dataset")

# For first iteration of curated subset
combine_and_save_plots(all_results_feature_selection, "Iter_1")

###############################################################################
# Run logistic regression on all datasets, collect results
###############################################################################

cat("\n", paste(rep("=", 80), collapse = ""), "\n")
cat("SIMPLE LOGISTIC REGRESSION ANALYSIS\n")
cat("\n", paste(rep("=", 80), collapse = ""), "\n")

datasets_to_test <- all_results_feature_selection$`Full Dataset`$datasets_to_test

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
  if (i == 2) sink(paste0("lr_orig_output", ".txt"))
  
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
    train_data = pain_data[, names(pain_data) %in% c(COLUMNS_COLINEAR)],
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
  filename = paste0("cohens_d_with_ttests", ".svg"),
  width = 10,
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
  paste0("cohens_d_with_ttests_results",  ".csv"),
  row.names = FALSE
)

cat("\nFiles saved:\n")
cat("- cohens_d_with_ttests.svg\n")
cat("- cohens_d_with_ttests_results.csv\n")
