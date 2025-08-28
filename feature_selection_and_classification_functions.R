###############################################################################
# Feature Selection and Classification Pipeline
#
# PURPOSE: Main utilities for data import, preprocessing, 
#          feature selection, and classification for pain threshold analysis.
###############################################################################

# --- Import Required Libraries ------------------------------------------------

library(parallel)           # Enables parallel processing
library(opdisDownsampling)  # Down/upsampling for balanced datasets
library(randomForest)       # Random Forest ML algorithm
library(caret)              # Comprehensive ML toolkit
library(pbmcapply)          # Parallel map with progress bar
library(Boruta)             # Wrapper for Boruta feature selector
library(reshape2)           # Reshape data functions (e.g., 'melt')
library(pROC)               # AUC/ROC analysis
library(dplyr)              # Data wrangling
library(glmnet)             # LASSO/logistic regression utilities
library(car)                # Linear and generalized linear model diagnostic tools

###############################################################################
# Utility and Preprocessing Functions ------------------------------------------
###############################################################################

# Load pain thresholds (features)
load_pain_thresholds_data <- function(file_path) {
  # Loads feature data from CSV. Assumes first column is row names.
  read.csv(file_path, row.names = 1)
}

# Load target vector/class labels
load_target_data <- function(file_path) {
  # Loads target labels from CSV. Assumes column "Target" present.
  read.csv(file_path, row.names = 1)$Target
}

# Rename columns for data harmonization
rename_pain_data_columns <- function(data, new_names) {
  # Assign new column names for consistency.
  names(data) <- new_names
  data
}

# Identify highly correlated variables for removal (caret)
find_variables_to_drop_caret <- function(data, method = CORRELATION_METHOD, cutoff = CORRELATION_LIMIT) {
  # Reports indices and names of highly correlated features (for data thinning).
  corr_matrix <- cor(data, method = method)
  high_corr_indices <- findCorrelation(corr_matrix, cutoff = cutoff)
  high_corr_names <- names(data)[high_corr_indices]
  cat("Highly correlated variables found:\n")
  print(high_corr_names)
  list(
    correlation_matrix = corr_matrix,
    high_correlation_indices = high_corr_indices,
    high_correlation_names = high_corr_names,
    vars_to_drop = high_corr_names
  )
}

# Standard plot theme for ggplot outputs
nyt_theme <- function() {
  theme_minimal(base_family = "Helvetica") +
    theme(
      text = element_text(color = "black"),
      plot.title = element_text(size = 12, face = "plain", hjust = 0),
      plot.subtitle = element_text(size = 11, face = "plain", hjust = 0),
      plot.caption = element_text(size = 10, color = "gray40", hjust = 0),
      axis.title = element_text(size = 11, face = "plain"),
      axis.text = element_text(size = 10),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.major.y = element_line(color = "gray85", size = 0.4),
      axis.line = element_blank(),
      legend.position = "bottom",
      legend.title = element_blank(),
      legend.text = element_text(size = 10),
      plot.margin = ggplot2::margin(10, 10, 10, 10)
    )
}

###############################################################################
# Classification Runner with Cross-Validation ----------------------------------

run_classifier_multiple_times <- function(train_data, train_target, test_data, test_target,
                                          classifier_type = "RF", n_runs = 100) {
  
  train_df_all <- as.data.frame(train_data)
  test_df_all <- as.data.frame(test_data)
  
  # Check for invalid values in features
  if (any(!is.finite(as.matrix(train_df_all)))) stop("Training data contains NA/NaN/Inf")
  if (any(!is.finite(as.matrix(test_df_all)))) stop("Test data contains NA/NaN/Inf")
  
  ba_results <- numeric(n_runs)
  auc_results <- numeric(n_runs)
  
  # Quick tune RF
  if (classifier_type == "RF" && tune_RF) {
    if (ncol(train_df_all) > 1) {
      mtry_values <- c(1, 2)
    } else {
      mtry_values <- 1
    }
    ntree_values <- c(500, 1000)
    
    grid <- expand.grid(mtry = mtry_values, ntree = ntree_values)
    grid$error <- NA
    
    # Store errors per run per grid point
    errors_matrix <- matrix(NA, nrow = nrow(grid), ncol = 10)
    
    for (run in 1:10) {
      set.seed(SEED + run)  # Different seed each run
      
      for (i in 1:nrow(grid)) {
        model <- randomForest(
          x = train_df_all,
          y = as.factor(train_target),
          mtry = grid$mtry[i],
          ntree = grid$ntree[i]
        )
        errors_matrix[i, run] <- mean(model$err.rate[, 1])  # OOB classification error
      }
    }
    
    # Compute median error across runs for each hyperparameter combo
    grid$median_error <- apply(errors_matrix, 1, median)
    
    # Select best hyperparameters based on median error
    best <- grid[which.min(grid$median_error), ]
  }
  
  
  for (i in 1:n_runs) {
    tryCatch({
      
      set.seed(i) # Different seed for each run
      
      if (training_and_validation_subsplits) {
        # Training and validation splits
        inTraining <- caret::createDataPartition(
          train_target,
          p = TRAINING_PARTITION_SIZE,
          list = FALSE
        )
        train_df <- train_df_all[inTraining, , drop = FALSE]  # keep as data frame
        train_target_factor <- as.factor(train_target[inTraining])
        
        inValidation <- caret::createDataPartition(
          test_target,
          p = VALIDATION_PARTITION_SIZE,
          list = FALSE
        )
        test_df <- test_df_all[inValidation, , drop = FALSE]  # keep as data frame
        test_target_factor <- as.factor(test_target[inValidation])
      } else {
        train_df <- train_df_all
        train_target_factor <- as.factor(train_target)
        test_df <- test_df_all
        test_target_factor <- as.factor(test_target)
      }
      levels(test_target_factor) <- levels(train_target_factor)
      
      if (classifier_type == "RF") {
        # Random Forest
        if (length(levels(train_target_factor)) < 2) stop("Less than 2 classes in training data")
        
        if (tune_RF) {
          model <- randomForest(
            x = train_df,
            y = train_target_factor,
            mtry = best$mtry,
            ntree = best$ntree
          )
        } else {
          model <- randomForest(x = train_df, y = train_target_factor, ntree = 500)
        }
        
        pred <- predict(model, test_df, type = "class")
        prob <- predict(model, test_df, type = "prob")
        
        if (ncol(prob) >= 2) {
          roc_obj <- pROC::roc(as.numeric(test_target_factor), prob[, 2], quiet = TRUE)
          auc_results[i] <- as.numeric(roc_obj$auc)
        } else {
          auc_results[i] <- NA
        }
        
        cm <- caret::confusionMatrix(pred, test_target_factor)
        ba_results[i] <- cm$byClass["Balanced Accuracy"]
        
      } else if (classifier_type == "LR") {
        # Logistic Regression
        if (length(levels(train_target_factor)) < 2) stop("Less than 2 classes in training data")
        
        lr_train_data <- train_df
        lr_train_data$target <- train_target_factor
        
        formula_str <- if (ncol(train_data) == 1) {
          paste("target ~", names(train_data)[1])
        } else {
          "target ~ ."
        }
        
        model <- glm(as.formula(formula_str), data = lr_train_data, family = binomial)
        prob_vec <- predict(model, test_df, type = "response")
        
        target_levels <- levels(train_target_factor)
        pred <- as.factor(ifelse(prob_vec > 0.5,
                                 target_levels[2],
                                 target_levels[1]))
        
        roc_obj <- pROC::roc(as.numeric(test_target_factor), prob_vec, quiet = TRUE)
        auc_results[i] <- as.numeric(roc_obj$auc)
        
        cm <- caret::confusionMatrix(pred, test_target_factor)
        ba_results[i] <- cm$byClass["Balanced Accuracy"]
        
      } else if (classifier_type == "KNN") {
        # K Nearest Neighbors (caret)
        knn_train_data <- train_df
        knn_train_data$target <- as.factor(train_target_factor)
        
        if (length(levels(knn_train_data$target)) < 2) stop("Less than 2 classes in training data")
        
        levels(knn_train_data$target) <- c("Class0", "Class1")
        
        ctrl <- caret::trainControl(method = "cv", number = 5,
                                    classProbs = TRUE, summaryFunction = twoClassSummary)
        
        model <- caret::train(
          target ~ ., data = knn_train_data,
          method = "knn",
          trControl = ctrl,
          metric = "ROC",
          preProcess = c("center", "scale"),
          tuneLength = 5
        )
        
        pred <- predict(model, test_df)
        prob <- predict(model, test_df, type = "prob")
        
        levels(test_target_factor) <- c("Class0", "Class1")
        
        roc_obj <- pROC::roc(as.numeric(test_target_factor), prob[, "Class1"], quiet = TRUE)
        auc_results[i] <- as.numeric(roc_obj$auc)
        
        cm <- caret::confusionMatrix(pred, test_target_factor)
        ba_results[i] <- cm$byClass["Balanced Accuracy"]
        
      } else {
        stop("Unsupported classifier_type")
      }
      
    }, error = function(e) {
      ba_results[i] <<- NA
      auc_results[i] <<- NA
    })
    
    if (i %% 20 == 0) cat(".") # Progress indicator
  }
  
  valid_ba <- ba_results[!is.na(ba_results)]
  valid_auc <- auc_results[!is.na(auc_results)]
  
  if (length(valid_ba) < 10) {
    return(list(
      ba_mean = NA, ba_ci_lower = NA, ba_ci_upper = NA,
      auc_mean = NA, auc_ci_lower = NA, auc_ci_upper = NA,
      n_successful = length(valid_ba)
    ))
  }
  
  ba_ci_lower <- stats::quantile(valid_ba, 0.025, na.rm = TRUE)
  ba_ci_upper <- stats::quantile(valid_ba, 0.975, na.rm = TRUE)
  auc_ci_lower <- stats::quantile(valid_auc, 0.025, na.rm = TRUE)
  auc_ci_upper <- stats::quantile(valid_auc, 0.975, na.rm = TRUE)
  
  return(list(
    ba_mean = mean(valid_ba, na.rm = TRUE),
    ba_ci_lower = ba_ci_lower,
    ba_ci_upper = ba_ci_upper,
    auc_mean = mean(valid_auc, na.rm = TRUE),
    auc_ci_lower = auc_ci_lower,
    auc_ci_upper = auc_ci_upper,
    n_successful = length(valid_ba)
  ))
}

# Classification function with 100 runs
quick_classify_100_runs <- function(train_data, train_target, test_data, test_target, dataset_name) {
  
  cat(sprintf("\n--- %s (%d features) ---\n", dataset_name, ncol(train_data)))
  
  if (ncol(train_data) == 0 || nrow(train_data) == 0) {
    cat("No data available - skipping\n")
    return(NULL)
  }
  
  results <- list()
  
  # === RANDOM FOREST ===
  cat("Running Random Forest 100 times")
  rf_results <- run_classifier_multiple_times(train_data, train_target, test_data, test_target, "RF", 100)
  cat(sprintf(" (%d successful runs)\n", rf_results$n_successful))
  
  results$RF <- rf_results
  
  if (!is.na(rf_results$ba_mean)) {
    cat(sprintf("RF - BA: %.3f [%.3f, %.3f], AUC: %.3f [%.3f, %.3f]\n",
                rf_results$ba_mean, rf_results$ba_ci_lower, rf_results$ba_ci_upper,
                rf_results$auc_mean, rf_results$auc_ci_lower, rf_results$auc_ci_upper))
  } else {
    cat("RF - Failed to get valid results\n")
  }
  
  # === LOGISTIC REGRESSION ===
  cat("Running Logistic Regression 100 times")
  lr_results <- run_classifier_multiple_times(train_data, train_target, test_data, test_target, "LR", 100)
  cat(sprintf(" (%d successful runs)\n", lr_results$n_successful))
  
  results$LR <- lr_results
  
  if (!is.na(lr_results$ba_mean)) {
    cat(sprintf("LR - BA: %.3f [%.3f, %.3f], AUC: %.3f [%.3f, %.3f]\n",
                lr_results$ba_mean, lr_results$ba_ci_lower, lr_results$ba_ci_upper,
                lr_results$auc_mean, lr_results$auc_ci_lower, lr_results$auc_ci_upper))
  } else {
    cat("LR - Failed to get valid results\n")
  }
  
  # === K NEAREST NEIGHBORS ===
  cat("Running KNN 100 times")
  knn_results <- run_classifier_multiple_times(train_data, train_target, test_data, test_target, "KNN", 100)
  cat(sprintf(" (%d successful runs)\n", knn_results$n_successful))
  
  results$KNN <- knn_results
  
  if (!is.na(knn_results$ba_mean)) {
    cat(sprintf("KNN - BA: %.3f [%.3f, %.3f], AUC: %.3f [%.3f, %.3f]\n",
                knn_results$ba_mean, knn_results$ba_ci_lower, knn_results$ba_ci_upper,
                knn_results$auc_mean, knn_results$auc_ci_lower, knn_results$auc_ci_upper))
  } else {
    cat("KNN - Failed to get valid results\n")
  }
  
  return(results)
}


create_results_table <- function(test_results, datasets_to_test) {
  # Define columns based on flag
  if (use_roc_auc) {
    results_df <- data.frame(
      Dataset = character(),
      Features = numeric(),
      RF_BA = character(),
      RF_AUC = character(),
      LR_BA = character(),
      LR_AUC = character(),
      KNN_BA = character(),
      KNN_AUC = character(),
      Classification_Success = integer(),
      stringsAsFactors = FALSE
    )
  } else {
    results_df <- data.frame(
      Dataset = character(),
      Features = numeric(),
      RF_BA = character(),
      LR_BA = character(),
      KNN_BA = character(),
      Classification_Success = integer(),
      stringsAsFactors = FALSE
    )
  }
  
  for (name in names(test_results)) {
    if (!is.null(test_results[[name]])) {
      
      # ---- RF ----
      rf_res <- test_results[[name]]$RF
      if (!is.na(rf_res$ba_mean)) {
        rf_ba_str <- sprintf("%.3f [%.3f, %.3f]", rf_res$ba_mean, rf_res$ba_ci_lower, rf_res$ba_ci_upper)
        if (use_roc_auc) {
          rf_auc_str <- sprintf("%.3f [%.3f, %.3f]", rf_res$auc_mean, rf_res$auc_ci_lower, rf_res$auc_ci_upper)
        }
      } else {
        rf_ba_str <- "NA"
        if (use_roc_auc) {
          rf_auc_str <- "NA"
        }
      }
      
      # ---- LR ----
      lr_res <- test_results[[name]]$LR
      if (!is.na(lr_res$ba_mean)) {
        lr_ba_str <- sprintf("%.3f [%.3f, %.3f]", lr_res$ba_mean, lr_res$ba_ci_lower, lr_res$ba_ci_upper)
        if (use_roc_auc) {
          lr_auc_str <- sprintf("%.3f [%.3f, %.3f]", lr_res$auc_mean, lr_res$auc_ci_lower, lr_res$auc_ci_upper)
        }
      } else {
        lr_ba_str <- "NA"
        if (use_roc_auc) {
          lr_auc_str <- "NA"
        }
      }
      
      # ---- KNN ----
      knn_res <- test_results[[name]]$KNN
      if (!is.na(knn_res$ba_mean)) {
        knn_ba_str <- sprintf("%.3f [%.3f, %.3f]", knn_res$ba_mean, knn_res$ba_ci_lower, knn_res$ba_ci_upper)
        if (use_roc_auc) {
          knn_auc_str <- sprintf("%.3f [%.3f, %.3f]", knn_res$auc_mean, knn_res$auc_ci_lower, knn_res$auc_ci_upper)
        }
      } else {
        knn_ba_str <- "NA"
        if (use_roc_auc) {
          knn_auc_str <- "NA"
        }
      }
      
      # ---- Success flag ----
      if (use_roc_auc) {
        success_flag <- any(
          rf_res$ba_ci_lower > 0.5, rf_res$auc_ci_lower > 0.5,
          lr_res$ba_ci_lower > 0.5, lr_res$auc_ci_lower > 0.5,
          knn_res$ba_ci_lower > 0.5, knn_res$auc_ci_lower > 0.5,
          na.rm = TRUE
        )
      } else {
        success_flag <- any(
          rf_res$ba_ci_lower > 0.5,
          lr_res$ba_ci_lower > 0.5,
          knn_res$ba_ci_lower > 0.5,
          na.rm = TRUE
        )
      }
      
      # Append row based on flag
      if (use_roc_auc) {
        results_df <- rbind(results_df, data.frame(
          Dataset = name,
          Features = ncol(datasets_to_test[[name]]$train),
          RF_BA = rf_ba_str,
          RF_AUC = rf_auc_str,
          LR_BA = lr_ba_str,
          LR_AUC = lr_auc_str,
          KNN_BA = knn_ba_str,
          KNN_AUC = knn_auc_str,
          Classification_Success = as.integer(success_flag),
          stringsAsFactors = FALSE
        ))
      } else {
        results_df <- rbind(results_df, data.frame(
          Dataset = name,
          Features = ncol(datasets_to_test[[name]]$train),
          RF_BA = rf_ba_str,
          LR_BA = lr_ba_str,
          KNN_BA = knn_ba_str,
          Classification_Success = as.integer(success_flag),
          stringsAsFactors = FALSE
        ))
      }
    }
  }
  return(results_df)
}



###############################################################################
# Feature Selection: Boruta and LASSO -----------------------------------------

# Boruta feature selection: wrapper for default config
run_boruta <- function(x, y)
  Boruta(x, y = as.factor(y), maxRuns = 100)

# Prepare Boruta output for plotting
prepare_boruta_plot_data <- function(boruta_res) {
  # Reshape importance measures, add decision status for coloring
  imp_long <- reshape2::melt(boruta_res$ImpHistory)
  colnames(imp_long) <- c("Iteration", "Feature", "Importance")
  decisions <- data.frame(Decision = boruta_res$finalDecision)
  decisions$Feature <- rownames(decisions)
  imp_long$Color <- decisions$Decision[match(imp_long$Feature, decisions$Feature)]
  imp_long$Color <- factor(imp_long$Color, levels = c(levels(decisions$Decision), "Shadow"))
  imp_long$Color[is.na(imp_long$Color)] <- "Shadow"
  list(importance = imp_long, decisions = decisions)
}

# Visualize Boruta results
plot_boruta <- function(plot_data, title) {
  ggplot(
    plot_data$importance,
    aes(x = reorder(Feature, Importance), y = Importance, color = Color, fill = Color)
  ) +
    geom_boxplot(alpha = .3) +
    theme_light() +
    labs(
      title = title,
      fill = "Decision", color = "Decision",
      x = "Features", y = "Importance"
    ) +
    theme(
      axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1, size = 10),
      legend.position = c(.2, .8),
      legend.background = element_rect(fill = alpha("white", 0.5)),
      legend.key = element_rect(fill = alpha("white", 0.5)),
      plot.title = element_text(size = 12)
    ) +
    scale_color_manual(
      values = c("Confirmed" = "chartreuse4", "Tentative" = "#E69F00", "Rejected" = "salmon", "Shadow" = "grey50")
    ) +
    scale_fill_manual(
      values = c("Confirmed" = "chartreuse4", "Tentative" = "#E69F00", "Rejected" = "salmon", "Shadow" = "grey50")
    )
}

# Helper function to retrive the Boruta features later
# Get Boruta selected and rejected features
get_boruta_features <- function(boruta_decisions, tentative_in = TRUE) {
  if (tentative_in) {
    selected_levels <- c("Confirmed", "Tentative")
    rejected_levels <- "Rejected"
  } else {
    selected_levels <- "Confirmed"
    rejected_levels <- c("Rejected", "Tentative")
  }
  list(
    selected = names(boruta_decisions[boruta_decisions %in% selected_levels]),
    rejected = names(boruta_decisions[boruta_decisions %in% rejected_levels])
  )
}

# LASSO regression: returns selected features and model
run_LASSO <- function(x, y) {
  lasso_actual <- tryCatch({
    x_matrix <- as.matrix(x)
    y_factor <- as.factor(y)
    cv_lasso <- cv.glmnet(x_matrix, y_factor, family = "binomial", alpha = 1, nfolds = 5)
    lasso_coef <- coef(cv_lasso, s = "lambda.min")
    selected_vars <- rownames(lasso_coef)[abs(lasso_coef[, 1]) > 0][-1] # exclude intercept
    list(selected = selected_vars, model = cv_lasso)
  }, error = function(e) {
    cat("Error in LASSO:", e$message, "\n")
    NULL
  })
}

# Prepare LASSO output for barplotting
prepare_lasso_plot_data <- function(lasso_res, feature_names) {
  if (is.null(lasso_res)) return(NULL)
  
  lasso_coef <- coef(lasso_res$model, s = "lambda.min")
  coef_df <- data.frame(
    Feature = rownames(lasso_coef)[-1],
    Coefficient = as.numeric(lasso_coef[-1, 1]),
    stringsAsFactors = FALSE
  )
  missing_features <- setdiff(feature_names, coef_df$Feature)
  if (length(missing_features) > 0) {
    missing_df <- data.frame(
      Feature = missing_features,
      Coefficient = 0,
      stringsAsFactors = FALSE
    )
    coef_df <- rbind(coef_df, missing_df)
  }
  coef_df$Decision <- ifelse(abs(coef_df$Coefficient) > 0, "Selected", "Rejected")
  coef_df$AbsCoefficient <- abs(coef_df$Coefficient)
  coef_df$AbsCoefficient[coef_df$AbsCoefficient == 0] <- 0.001
  return(coef_df)
}

# Visualize LASSO results as barplot
plot_lasso <- function(plot_data, title) {
  if (is.null(plot_data)) {
    return(ggplot() +
             ggtitle("LASSO Failed") +
             theme_void())
  }
  ggplot(plot_data, aes(x = reorder(Feature, AbsCoefficient), y = AbsCoefficient,
                        color = Decision, fill = Decision)) +
    geom_col(alpha = 0.3) +
    theme_light() +
    labs(
      title = title,
      fill = "Decision", color = "Decision",
      x = "Features", y = "Absolute Coefficient Value"
    ) +
    theme(
      axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1, size = 10),
      legend.position.inside = TRUE, legend.position = c(.2, .8),
      legend.background = element_rect(fill = alpha("white", 0.5)),
      legend.key = element_rect(fill = alpha("white", 0.5)),
      plot.title = element_text(size = 12)
    ) +
    scale_color_manual(
      values = c("Selected" = "chartreuse4", "Rejected" = "salmon")
    ) +
    scale_fill_manual(
      values = c("Selected" = "chartreuse4", "Rejected" = "salmon")
    ) +
    ylim(0,1.5)
}

###############################################################################
# --- Simple Logistic Regression Analysis ---
###############################################################################

# Function to run single logistic regression and print summary
run_single_logistic_regression <- function(train_data, train_target, dataset_name) {
  
  cat(sprintf("\n=== %s - Logistic Regression Summary ===\n", dataset_name))
  cat(sprintf("Dataset: %d features, %d samples\n", ncol(train_data), nrow(train_data)))
  
  # Skip if no data
  if (ncol(train_data) == 0 || nrow(train_data) == 0) {
    cat("No data available - skipping\n")
    return(NULL)
  }
  
  tryCatch({
    # Prepare data for logistic regression
    lr_train_data <- train_data
    lr_train_data$target <- as.factor(train_target)
    
    # Create formula
    if (ncol(train_data) == 1) {
      formula_str <- paste("target ~", names(train_data)[1])
    } else {
      formula_str <- "target ~ ."
    }
    
    # Fit logistic regression model
    model <- glm(as.formula(formula_str), data = lr_train_data, family = binomial)
    
    # Print model summary
    print(summary(model))
    
    return(model)
    
  }, error = function(e) {
    cat("Error fitting logistic regression:", e$message, "\n")
    return(NULL)
  })
}

###############################################################################
# --- Cohen's d effect sizes ---
###############################################################################

# Cohen's d Analysis and Visualization
library(effsize)
library(ggplot2)
library(dplyr)
library(tidyr)


cat("Creating Cohen's d analysis with t-tests and visualization...\n")

# Function to calculate Cohen's d with confidence intervals and t-test
calculate_cohens_d_with_ttest <- function(data, target, dataset_name) {
  results <- data.frame(
    Variable = character(),
    Cohens_d = numeric(),
    CI_lower = numeric(),
    CI_upper = numeric(),
    t_statistic = numeric(),
    p_value = numeric(),
    Dataset = character(),
    stringsAsFactors = FALSE
  )
  
  target_factor <- as.factor(target)
  
  for (var in names(data)) {
    tryCatch({
      # Split data by target groups
      group1 <- data[target_factor == levels(target_factor)[1], var]
      group2 <- data[target_factor == levels(target_factor)[2], var]
      
      # Calculate Cohen's d with confidence interval
      cohens_result <- cohen.d(group1, group2, conf.level = 0.95)
      
      # Perform t-test
      ttest_result <- t.test(group1, group2)
      
      results <- rbind(results, data.frame(
        Variable = var,
        Cohens_d = cohens_result$estimate,
        CI_lower = cohens_result$conf.int[1],
        CI_upper = cohens_result$conf.int[2],
        t_statistic = ttest_result$statistic,
        p_value = ttest_result$p.value,
        Dataset = dataset_name,
        stringsAsFactors = FALSE
      ))
    }, error = function(e) {
      cat(sprintf("Error calculating Cohen's d for variable %s in %s: %s\n", var, dataset_name, e$message))
    })
  }
  
  return(results)
}

library(ggplot2)
library(dplyr)

plot_cohens_d <- function(cohens_d_list, dataset_names = NULL) {
  # Input validation
  if (!is.list(cohens_d_list)) stop("Input cohens_d_list must be a list of data frames.")
  if (!is.null(dataset_names) && length(dataset_names) != length(cohens_d_list)) {
    stop("Length of dataset_names must be same as length of cohens_d_list or NULL.")
  }
  
  # Assign dataset labels to each data frame
  labeled_list <- Map(function(df, name) {
    df$Dataset <- if (is.null(name)) unique(df$Dataset)[1] else name
    df
  }, cohens_d_list, if (is.null(dataset_names)) rep(NA, length(cohens_d_list)) else dataset_names)
  
  # Combine all Cohen's d data frames
  all_cohens_d <- dplyr::bind_rows(labeled_list)
  
  # Add significance labels and position info
  all_cohens_d <- all_cohens_d %>%
    dplyr::mutate(
      p_label = dplyr::case_when(
        p_value < 0.001 ~ "***",
        p_value < 0.01  ~ "**",
        p_value < 0.05  ~ "*",
        p_value < 0.1   ~ ".",
        TRUE            ~ ""
      ),
      t_label = sprintf("t=%.2f%s", t_statistic, p_label),
      text_y_position = ifelse(Cohens_d >= 0, CI_upper + 0.1, CI_lower - 0.1),
      text_hjust = ifelse(Cohens_d >= 0, 0, 1)
    )
  
  # Use original ordering from first dataset's effect sizes (absolute)
  variable_order <- cohens_d_list[[1]] %>%
    dplyr::arrange(dplyr::desc(abs(Cohens_d))) %>%
    dplyr::pull(Variable)
  
  all_cohens_d$Variable <- factor(all_cohens_d$Variable, levels = rev(variable_order))
  
  # Define fill colors, accommodating dynamic dataset names
  unique_datasets <- unique(all_cohens_d$Dataset)
  palette <- c("dodgerblue", "chartreuse3", "orangered", "goldenrod", "purple", "cyan", "magenta", "gray40")
  fill_colors <- setNames(palette[seq_along(unique_datasets)], unique_datasets)
  
  # Build plot
  cohens_d_plot <- ggplot(all_cohens_d, aes(x = reorder(Variable, abs(Cohens_d)),
                                            y = Cohens_d,
                                            fill = Dataset)) +
    geom_col(position = position_dodge(width = 0.7), width = 0.6, alpha = 0.9) +
    geom_errorbar(aes(ymin = CI_lower, ymax = CI_upper),
                  position = position_dodge(width = 0.7),
                  width = 0.2,
                  color = "grey50") +
    geom_text(aes(y = text_y_position, label = t_label, hjust = text_hjust),
              position = position_dodge(width = 0.7),
              size = 3.2,
              family = "sans",
              color = "black") +
    geom_hline(yintercept = 0, linetype = "solid", color = "black", linewidth = 0.4) +
    geom_hline(yintercept = c(-0.2, 0.2), linetype = "dashed", color = "grey70") +
    geom_hline(yintercept = c(-0.5, 0.5), linetype = "dashed", color = "grey60") +
    geom_hline(yintercept = c(-0.8, 0.8), linetype = "dashed", color = "grey50") +
    scale_fill_manual(name = NULL, values = fill_colors) +
    coord_flip() +
    labs(title = "Cohen's d effect sizes",
         subtitle = "Variables ordered by descending effect size\nStars indicate t-test significance",
         x = NULL,
         y = "Cohen's d",
         caption = "Dashed lines represent standard effect size thresholds (0.2, 0.5, 0.8).") +
    theme_minimal(base_family = "sans")
  
  if (use_nyt) {
    cohens_d_plot <- cohens_d_plot + nyt_theme()
  }
  
  return(list(plot = cohens_d_plot, combined_data = all_cohens_d))
}

###############################################################################
# --- Analysis Pipeline Function ---
###############################################################################

run_analysis_pipeline <- function(processed_data = NULL, use_curated_subset = FALSE,
                                  curated_names = NULL, add_file_string = "",
                                  training_data_actual = NULL, training_target = NULL,
                                  validation_data_actual = NULL, validation_target = NULL) {
  
  # Check if pre-split data is provided
  if (!is.null(training_data_actual) && !is.null(training_target) &&
      !is.null(validation_data_actual) && !is.null(validation_target)) {
    
    # Use pre-split data
    cat("Using pre-split training/validation data...\n")
    
    # Apply curated subset if requested to pre-split data
    if (use_curated_subset && !is.null(curated_names)) {
      available_curated_train <- intersect(curated_names, names(training_data_actual))
      available_curated_valid <- intersect(curated_names, names(validation_data_actual))
      
      if (length(available_curated_train) > 0 && length(available_curated_valid) > 0) {
        training_data_actual <- training_data_actual[, available_curated_train, drop = FALSE]
        validation_data_actual <- validation_data_actual[, available_curated_valid, drop = FALSE]
        cat("Applied curated subset to pre-split data:", paste(names(training_data_actual), collapse = ", "), "\n")
      }
    }
    
  } else if (!is.null(processed_data)) {
    
    # Use full dataset and perform splitting
    data <- processed_data$data
    target <- processed_data$target
    
    # Apply curated subset if requested
    if (use_curated_subset && !is.null(curated_names)) {
      available_curated <- intersect(curated_names, names(data))
      if (length(available_curated) > 0) {
        data <- data[, available_curated, drop = FALSE]
        cat("Using curated subset:", paste(names(data), collapse = ", "), "\n")
      }
    }
    
    # Split into training/validation sets
    cat("Splitting data into training/validation sets...\n")
    data_split <- opdisDownsampling::opdisDownsampling(
      data, Cls = target, Size = 0.8, Seed = SEED, nTrials = 2000000,
      MaxCores = parallel::detectCores() - 1
    )
    
    training_data_actual <- data_split$ReducedData[, 1:(ncol(data_split$ReducedData) - 1)]
    training_target <- data_split$ReducedData$Cls
    
    validation_data_actual <- data_split$RemovedData[, 1:(ncol(data_split$RemovedData) - 1)]
    validation_target <- data_split$RemovedData$Cls
    
  } else {
    stop("Either 'processed_data' or all four split parameters (training_data_actual, training_target, validation_data_actual, validation_target) must be provided")
  }
  
  cat("=== Classification Analysis with 100 Runs CI ===\n")
  cat("Training data:", nrow(training_data_actual), "rows,", ncol(training_data_actual), "features\n")
  cat("Validation data:", nrow(validation_data_actual), "rows,", ncol(validation_data_actual), "features\n")
  
  # Run Boruta feature selection
  cat("\nRunning Boruta feature selection...\n")
  set.seed(SEED)
  boruta_actual <- run_boruta(x = training_data_actual, y = as.factor(training_target))
  
  # Create and save Boruta plot
  cat("Creating Boruta visualization plot...\n")
  boruta_plot_data <- prepare_boruta_plot_data(boruta_actual)
  boruta_plot <- plot_boruta(boruta_plot_data, "Boruta Feature Importance - Training Data")
  if (use_nyt) boruta_plot <- boruta_plot + nyt_theme() +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))
  
  boruta_decisions <- boruta_actual$finalDecision
  boruta_res <- get_boruta_features(boruta_decisions, Boruta_tentative_in)
  
  boruta_selected <- boruta_res$selected
  boruta_rejected <- boruta_res$rejected

  
  # # Handle case where no features are selected by Boruta
  # if (length(boruta_selected) == 0) {
  #   cat("Warning: Boruta selected NO features! Using all features as backup.\n")
  #   boruta_selected <- names(training_data_actual)
  #   boruta_rejected <- character(0)
  # }
  
  cat("Boruta selected features:", length(boruta_selected), "\n")
  cat("Selected:", paste(boruta_selected, collapse = ", "), "\n")
  cat("Boruta rejected features:", length(boruta_rejected), "\n")
  cat("Rejected:", paste(boruta_rejected, collapse = ", "), "\n")
  
  # Run LASSO feature selection
  cat("\nRunning LASSO feature selection...\n")
  set.seed(SEED)
  lasso_actual <- run_LASSO(x = training_data_actual, y = as.factor(training_target))
  
  # Create and save LASSO plot
  lasso_plot_data <- prepare_lasso_plot_data(lasso_actual, names(training_data_actual))
  lasso_plot <- plot_lasso(lasso_plot_data, "LASSO Feature Selection - Training Data")
  if (use_nyt) lasso_plot <- lasso_plot + nyt_theme() +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))
  
  # Get LASSO selected and rejected features
  if (!is.null(lasso_actual)) {
    lasso_selected <- lasso_actual$selected
    lasso_rejected <- setdiff(names(training_data_actual), lasso_selected)
    
    # if (length(lasso_selected) == 0) {
    #   cat("Warning: LASSO selected NO features! Using all features as backup.\n")
    #   lasso_selected <- names(training_data_actual)
    #   lasso_rejected <- character(0)
    # }
  } else {
    cat("LASSO failed - using all features as backup.\n")
    lasso_selected <- names(training_data_actual)
    lasso_rejected <- character(0)
  }
  
  cat("LASSO selected features:", length(lasso_selected), "\n")
  cat("Selected:", paste(lasso_selected, collapse = ", "), "\n")
  cat("LASSO rejected features:", length(lasso_rejected), "\n")
  cat("Rejected:", paste(lasso_rejected, collapse = ", "), "\n")
  
  # Create feature selection matrix plot
  cat("\nCreating feature selection matrix plot...\n")
  all_features <- names(training_data_actual)
  
  selection_matrix <- data.frame(
    Feature        = all_features,
    Boruta_Selected = as.integer(all_features %in% boruta_res$selected),
    LASSO_Selected  = as.integer(all_features %in% lasso_plot_data$Feature[lasso_plot_data$Decision != "Rejected"]),
    stringsAsFactors = FALSE
  )
  
  selection_matrix$Selection_Category <- with(selection_matrix, {
    ifelse(Boruta_Selected == 1 & LASSO_Selected == 1, "Both",
           ifelse(Boruta_Selected == 1 & LASSO_Selected == 0, "Boruta only",
                  ifelse(Boruta_Selected == 0 & LASSO_Selected == 1, "LASSO only", "Neither")))
  })
  
  # Convert to long format for plotting
  selection_long <- selection_matrix %>%
    select(Feature, Boruta_Selected, LASSO_Selected) %>%
    pivot_longer(cols = c(Boruta_Selected, LASSO_Selected),
                 names_to = "Method",
                 values_to = "Selected") %>%
    mutate(Method = gsub("_Selected", "", Method))
  
  # Create matrix heatmap
  matrix_plot <- ggplot(selection_long, aes(x = Method, y = Feature, fill = factor(Selected))) +
    geom_tile(color = "white", linewidth = 0.5, alpha = .7) +
    scale_fill_manual(
      values = c("0" = "lightgray", "1" = "chartreuse4"),
      labels = c("0" = "Not Selected", "1" = "Selected"),
      name = "Selection Status"
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 0, hjust = 0.5, size = 12),
      axis.text.y = element_text(size = 10),
      plot.title = element_text(size = 14, hjust = 0.5),
      legend.position = "bottom", legend.direction = "vertical",
      panel.grid = element_blank()
    ) +
    labs(
      title = "Feature Selection Matrix",
      x = "Selection Method",
      y = "Features"
    )
  if (use_nyt) matrix_plot <- matrix_plot + nyt_theme()
  
  # Create summary bar plot
  summary_data <- selection_matrix %>%
    count(Selection_Category) %>%
    mutate(Selection_Category = factor(Selection_Category,
                                       levels = c("Both", "Boruta only", "LASSO only", "Neither")))
  
  summary_plot <- ggplot(summary_data, aes(x = Selection_Category, y = n, fill = Selection_Category)) +
    geom_bar(stat = "identity", alpha = 0.8) +
    geom_text(aes(label = n), vjust = 0.3, size = 4) +
    scale_fill_manual(
      values = c("Both" = "chartreuse4", "Boruta only" = "#56B4E9",
                 "LASSO only" = "#E69F00", "Neither" = "salmon")
    ) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.1))) + # Add 10% space at top
    theme_light() +
    theme(
      plot.title = element_text(size = 14, hjust = 0.5),
      legend.position = "none",
      axis.text.x = element_text(angle = 90, hjust = 1)
    ) +
    labs(
      title = "Feature selection summary",
      x = "Selection category",
      y = "Number of features"
    )
  if (use_nyt) summary_plot <- summary_plot + nyt_theme() + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))

  # Run classification experiments
  cat("\nRunning classification experiments...\n")
  
  # Define datasets to test
  datasets_to_test <- list(
    "All_Features" = list(
      train = training_data_actual,
      test = validation_data_actual
    ),
    "Boruta_Selected" = list(
      train = if (length(boruta_selected) > 0) training_data_actual[, boruta_selected, drop = FALSE] else data.frame(),
      test = if (length(boruta_selected) > 0) validation_data_actual[, boruta_selected, drop = FALSE] else data.frame()
    ),
    "Boruta_Rejected" = list(
      train = if (length(boruta_rejected) > 0) training_data_actual[, boruta_rejected, drop = FALSE] else data.frame(),
      test = if (length(boruta_rejected) > 0) validation_data_actual[, boruta_rejected, drop = FALSE] else data.frame()
    ),
    "LASSO_Selected" = list(
      train = if (length(lasso_selected) > 0) training_data_actual[, lasso_selected, drop = FALSE] else data.frame(),
      test = if (length(lasso_selected) > 0) validation_data_actual[, lasso_selected, drop = FALSE] else data.frame()
    ),
    "LASSO_Rejected" = list(
      train = if (length(lasso_rejected) > 0) training_data_actual[, lasso_rejected, drop = FALSE] else data.frame(),
      test = if (length(lasso_rejected) > 0) validation_data_actual[, lasso_rejected, drop = FALSE] else data.frame()
    ),
    "Boruta_LASSO_Selected" = list(
      train = if (length(boruta_selected) > 0 || length(lasso_selected) > 0) 
        training_data_actual[, union(boruta_selected, lasso_selected), drop = FALSE] else data.frame(),
      test = if (length(boruta_selected) > 0 || length(lasso_selected) > 0) 
        validation_data_actual[, union(boruta_selected, lasso_selected), drop = FALSE] else data.frame()
    ),
    "Boruta_LASSO_Rejected" = list(
      train = if (length(boruta_rejected) > 0 || length(lasso_rejected) > 0)  
        training_data_actual[, setdiff(names(training_data_actual), union(boruta_selected, lasso_selected)), drop = FALSE] else data.frame(),
      test = if (length(boruta_rejected) > 0 || length(lasso_rejected) > 0)
        validation_data_actual[, setdiff(names(training_data_actual), union(boruta_selected, lasso_selected)), drop = FALSE] else data.frame()
    ),
    "Only_high_correlated" = list(
      train = data.frame(Pressure2 = training_data_actual[find_variables_to_drop_caret(data = training_data_actual, method = CORRELATION_METHOD, cutoff = CORRELATION_LIMIT)$vars_to_drop]),
      test = data.frame(Pressure2 = validation_data_actual[find_variables_to_drop_caret(data = training_data_actual, method = CORRELATION_METHOD, cutoff = CORRELATION_LIMIT)$vars_to_drop])
    )
  )
  
  # Run classification tests
  test_results <- list()
  for (dataset_name in names(datasets_to_test)) {
    dataset <- datasets_to_test[[dataset_name]]
    test_results[[dataset_name]] <- quick_classify_100_runs(
      train_data =  dataset$train, train_target = training_target, test_data = dataset$test, test_target = validation_target, dataset_name = dataset_name
    )
  }
  
  # Create and display results table
  if (length(test_results) > 0) {
    cat("\n=== RESULTS SUMMARY ===\n")
    results_table <- create_results_table(test_results, datasets_to_test)
    print(results_table)
  }
  
  # Return results for further analysis if needed
  return(list(
    boruta_results = boruta_actual,
    lasso_results = lasso_actual,
    classification_results = test_results,
    results_table = results_table,
    datasets_to_test = datasets_to_test,
    plots = list(
      boruta = boruta_plot,
      lasso = lasso_plot,
      matrix = matrix_plot,
      summary = summary_plot
      
    )
  ))
}

