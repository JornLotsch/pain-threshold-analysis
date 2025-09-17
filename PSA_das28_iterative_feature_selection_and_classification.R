###############################################################################
# Generic Data Analysis Pipeline: Application to PSA_das28crp data
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

# External functions (still needed for rename_pain_data_columns and others)
FUNCTIONS_FILE_PATH <- "/home/joern/.Datenplatte/Joerns Dateien/Aktuell/ABCPython/08AnalyseProgramme/R/ABC2way/feature_selection_and_classification_functions.R"

# Dataset name
DATASET_NAME <- "PSA_das28crp"
EXPERIMENTS_DIR <- "/home/joern/.Datenplatte/Joerns Dateien/Aktuell/ABCPython/08AnalyseProgramme/R/ABC2way/"

# PSA das28crp data paths
base_path_psa <- "/home/joern/Aktuell/RheumaMetabolomicsFFM"
r_path_psa <- "08AnalyseProgramme/R/OLD"

psa_base_path <- file.path(base_path_psa, r_path_psa)

training_file_psa <- file.path(psa_base_path, "TrainingTestData_das28crp_all_items.csv")
validation_file_psa <- file.path(psa_base_path, "ValidationData_das28crp_all_items.csv")
full_data_file_psa <- file.path(psa_base_path, "PSA_das28_all_items.csv")

# Analysis parameters
SEED <- 42
noise_factor <- 0.2 # Not used here but kept for code consistency if needed later
CORRELATION_METHOD <- "pearson"
CORRELATION_LIMIT <- 0.9
Boruta_tentative_in <- FALSE
use_nyt <- TRUE
tune_RF <- TRUE
mtry_12only <- FALSE
max_iterations <- 5

use_curated <- FALSE
use_roc_auc <- FALSE
training_and_validation_subsplits <- TRUE
TRAINING_PARTITION_SIZE <- 0.8
VALIDATION_PARTITION_SIZE <- 0.8

# PSA dataset does not require renaming or noise addition here as per initial code

###############################################################################
# Load External Functions and set actual path
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
if (!file.exists(training_file_psa)) {
  stop(paste("PSA training data file not found:", training_file_psa))
}
if (!file.exists(validation_file_psa)) {
  stop(paste("PSA validation data file not found:", validation_file_psa))
}
if (!file.exists(full_data_file_psa)) {
  stop(paste("PSA full data file not found:", full_data_file_psa))
}
###############################################################################
# Load PSA data and preprocess similarly to PainThreshold pipeline
###############################################################################
# # Load full PSA data 
dfDAS28crp <- read.csv(full_data_file_psa, row.names = 1)
dfDAS28crp <- dfDAS28crp[, !names(dfDAS28crp) %in% c("visit_da_tjc28", "visit_da_sjc28")]
# # Create target factor based on remission status
dfDAS28crp_Target <- as.factor(ifelse(dfDAS28crp$ps_a_score == "remission", 0, 1))
dfDAS28crp <- dfDAS28crp[, !names(dfDAS28crp) %in% c("Cls", "Target", "ps_a_score")]

# Load training set
dfDAS28crp_training <- read.csv(training_file_psa, row.names = 1)
dfDAS28crp_training <- dfDAS28crp_training[, !names(dfDAS28crp_training) %in% c("visit_da_tjc28", "visit_da_sjc28")]
# Create training target factor similarly
dfDAS28crp_training_Target <- as.factor(ifelse(dfDAS28crp_training$Cls == "remission", 0, 1))
dfDAS28crp_training <- dfDAS28crp_training[, !names(dfDAS28crp_training) %in% c("Cls", "Target", "ps_a_score")]


# Load validation set
dfDAS28crp_validation <- read.csv(validation_file_psa, row.names = 1)
dfDAS28crp_validation <- dfDAS28crp_validation[, !names(dfDAS28crp_validation) %in% c("visit_da_tjc28", "visit_da_sjc28")]
dfDAS28crp_validation_Target <- as.factor(ifelse(dfDAS28crp_validation$Cls == "remission", 0, 1))
dfDAS28crp_validation <- dfDAS28crp_validation[, !names(dfDAS28crp_validation) %in% c("Cls", "Target", "ps_a_score")]

# Assign to pipeline variables expected downstream
training_data_original <- dfDAS28crp_training
training_target <- dfDAS28crp_training_Target
validation_data_original <- dfDAS28crp_validation
validation_target <- dfDAS28crp_validation_Target

###############################################################################
# Run analysis sequentially
###############################################################################

results_list <- run_feature_selection_iterations()

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

  matrix_plot <- plots$matrix + theme(axis.text.y = element_text(size = 3))
  summary_plot <- plots$summary
  boruta_plot <- plots$boruta + theme(axis.text.x = element_text(size = 5))
  lasso_plot <- plots$lasso + theme(axis.text.x = element_text(size = 5)) + ylim(0, 3)

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
combine_and_save_plots(results_list$results_list, "Full dataset")

###############################################################################
# Run logistic regression on all datasets, collect results
###############################################################################

cat("\n", paste(rep("=", 80), collapse = ""), "\n")
cat("SIMPLE LOGISTIC REGRESSION ANALYSIS\n")
cat("\n", paste(rep("=", 80), collapse = ""), "\n")

datasets_to_test <- results_list$results_list$`Full dataset`$datasets_to_test

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
    train_data = dfDAS28crp,
    train_target = as.factor(dfDAS28crp_Target),
    dataset_name = "Original unsplit"
  )
  if (is.null(COLUMNS_COLINEAR)) {
    COLUMNS_COLINEAR <- rownames(alias(r1)$Complete)
    cat("\nCOLUMNS_COLINEAR\n", COLUMNS_COLINEAR)
  }

  run_single_logistic_regression(
    train_data = dfDAS28crp[, !names(dfDAS28crp) %in% COLUMNS_COLINEAR],
    train_target = dfDAS28crp_Target,
    dataset_name = "Original unsplit colinear variables removed"
  )

  if (i == 2) sink()
}

cat("\n", paste(rep("=", 80), collapse = ""), "\n")
cat("Logistic regression analysis completed!\n")
cat("\n", paste(rep("=", 80), collapse = ""), "\n")

