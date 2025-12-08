###############################################################################
# Feature Selection and Classification Pipeline
#
# PURPOSE: Main utilities for data import, preprocessing, 
#          feature selection, and classification for pain threshold analysis.
###############################################################################

# --- Import Required Libraries ------------------------------------------------

library(parallel) # Enables parallel processing
library(opdisDownsampling) # Down/upsampling for balanced datasets
library(randomForest) # Random Forest ML algorithm
library(caret) # Comprehensive ML toolkit
library(pbmcapply) # Parallel map with progress bar
library(Boruta) # Wrapper for Boruta feature selector
library(reshape2) # Reshape data functions (e.g., 'melt')
library(pROC) # AUC/ROC analysis
library(dplyr) # Data wrangling
library(glmnet) # LASSO/logistic regression utilities
library(car) # Linear and generalized linear model diagnostic tools
library(R.utils) # For timeout

###############################################################################
# Set working directory  ------------------------------------------
###############################################################################

set_working_directory <- function(EXPERIMENTS_DIR = NULL) {
  # Attempt to set working directory to script location (for RStudio)
  tryCatch({
    if (requireNamespace("rstudioapi", quietly = TRUE) &&
        rstudioapi::isAvailable("getSourceEditorContext")) {
      script_path <- rstudioapi::getSourceEditorContext()$path
      if (!is.null(script_path) && nzchar(script_path)) {
        setwd(dirname(script_path))
        cat("Working directory set to script location:", getwd(), "\n")
      }
    }
  }, error = function(e) {
    message("Unable to set working directory automatically: ", e$message)
  })

  # If EXPERIMENTS_DIR is provided, try switching to it
  if (!is.null(EXPERIMENTS_DIR)) {
    if (dir.exists(EXPERIMENTS_DIR)) {
      tryCatch({
        setwd(EXPERIMENTS_DIR)
        cat("Working directory changed to:", getwd(), "\n")
      }, error = function(e) {
        warning("Failed to change to experiments directory: ", e$message)
        cat("Continuing with current directory:", getwd(), "\n")
      })
    } else {
      warning("Experiments directory does not exist: ", EXPERIMENTS_DIR)
      cat("Continuing with current directory:", getwd(), "\n")
    }
  }

  # Verify current working directory
  cat("Current working directory:", getwd(), "\n")

  invisible(getwd()) # return the working dir without printing
}


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

# Timeout configuration parameters
PER_ITERATION_TIMEOUT <- 60  # seconds - hard limit per single run
MIN_SUCCESSFUL_RUNS <- 10     # minimum required for valid statistics
EARLY_STOP_CHECK <- 20        # check after this many runs for early abort

# Timeout function for Linux (fork-based)
eval_with_timeout <- function(expr, envir = parent.frame(), timeout, on_timeout = c("error", "warning", "silent")) {
  # substitute expression so it is not executed as soon it is used
  expr <- substitute(expr)

  # match on_timeout
  on_timeout <- match.arg(on_timeout)

  # execute expr in separate fork
  myfork <- parallel::mcparallel({
    eval(expr, envir = envir)
  }, silent = FALSE)

  # wait max n seconds for a result.
  myresult <- parallel::mccollect(myfork, wait = FALSE, timeout = timeout)
  # kill fork after collect has returned
  tools::pskill(myfork$pid, tools::SIGKILL)
  tools::pskill(-1 * myfork$pid, tools::SIGKILL)

  # clean up:
  parallel::mccollect(myfork, wait = FALSE)

  # timeout?
  if (is.null(myresult)) {
    if (on_timeout == "error") {
      stop("reached elapsed time limit")
    } else if (on_timeout == "warning") {
      warning("reached elapsed time limit")
    } else if (on_timeout == "silent") {
      myresult <- NA
    }
  }

  # move this to distinguish between timeout and NULL returns
  myresult <- myresult[[1]]

  if ("try-error" %in% class(myresult)) {
    stop(attr(myresult, "condition"))
  }

  # send the buffered response
  return(myresult)
}

run_classifier_multiple_times <- function(train_data, train_target, test_data, test_target,
                                          classifier_type = "RF", n_runs = 100) {
  train_df_all <- as.data.frame(train_data)
  test_df_all <- as.data.frame(test_data)
  # Check for invalid values in features
  if (any(!is.finite(as.matrix(train_df_all)))) stop("Training data contains NA/NaN/Inf")
  if (any(!is.finite(as.matrix(test_df_all)))) stop("Test data contains NA/NaN/Inf")

  ba_results <- numeric(n_runs)
  auc_results <- numeric(n_runs)
  n_timeouts <- 0
  n_errors <- 0
  n_successful <- 0

  # Quick tune RF - do once before loop
  if (classifier_type == "RF" && tune_RF) {
    if (!mtry_12only) {
      mtry_values <- unique(round(c(2, sqrt(ncol(train_df_all)), ncol(train_df_all) / 2)))
    } else {
      if (ncol(train_df_all) > 1) {
        mtry_values <- c(1, 2)
      } else {
        mtry_values <- 1
      }
    }
    ntree_values <- c(200, 500, 1000)
    grid <- expand.grid(mtry = mtry_values, ntree = ntree_values)
    grid$error <- NA
    errors_matrix <- matrix(NA, nrow = nrow(grid), ncol = 10)
    for (run in 1:10) {
      set.seed(SEED + run)
      for (i in 1:nrow(grid)) {
        tryCatch({
          model <- randomForest(
            x = train_df_all,
            y = as.factor(train_target),
            mtry = grid$mtry[i],
            ntree = grid$ntree[i]
          )
          errors_matrix[i, run] <- mean(model$err.rate[, 1])
        }, error = function(e) {
          errors_matrix[i, run] <<- NA
        })
      }
    }
    grid$median_error <- apply(errors_matrix, 1, median, na.rm = TRUE)
    best <- grid[which.min(grid$median_error),]
  }

  # Quick tune KNN - do once before loop
  if (classifier_type == "KNN" && tune_KNN) {
    cat("Tuning KNN hyperparameters...")

    best_knn_k <- tryCatch({
      eval_with_timeout({
        # Prepare data for tuning
        knn_tune_data <- train_df_all
        knn_tune_data$target <- as.factor(train_target)
        n_classes <- length(levels(knn_tune_data$target))

        # Rename classes to valid names
        new_levels <- paste0("Class", seq_len(n_classes))
        levels(knn_tune_data$target) <- new_levels

        # Setup CV control for tuning
        if (n_classes == 2) {
          ctrl_tune <- caret::trainControl(method = "cv", number = 5,
                                           classProbs = TRUE,
                                           summaryFunction = twoClassSummary,
                                           allowParallel = FALSE)
          metric_tune <- "ROC"
        } else {
          ctrl_tune <- caret::trainControl(method = "cv", number = 5,
                                           classProbs = TRUE,
                                           allowParallel = FALSE)
          metric_tune <- "Accuracy"
        }

        set.seed(SEED)
        knn_tune_model <- suppressWarnings(
          caret::train(
            target ~ ., data = knn_tune_data,
            method = "knn",
            trControl = ctrl_tune,
            metric = metric_tune,
            preProcess = c("center", "scale"),
            tuneGrid = data.frame(k = c(3, 5, 7, 9, 11, 13))
          )
        )
        knn_tune_model$bestTune$k
      }, timeout = PER_ITERATION_TIMEOUT * 5, on_timeout = "error")  # 5x longer for tuning
    }, error = function(e) {
      cat(" FAILED (using default k=5)\n")
      5  # fallback default
    })

    if (!is.null(best_knn_k)) {
      cat(sprintf(" best k = %d\n", best_knn_k))
    }
  }

  # Quick tune SVM - do once before loop
  if (classifier_type == "SVM" && tune_SVM) {
    cat("Tuning SVM hyperparameters...")

    best_svm_params <- tryCatch({
      eval_with_timeout({
        # Prepare data for tuning
        svm_tune_data <- train_df_all
        svm_tune_data$target <- as.factor(train_target)
        n_classes <- length(levels(svm_tune_data$target))

        # Rename classes to valid names
        new_levels <- paste0("Class", seq_len(n_classes))
        levels(svm_tune_data$target) <- new_levels

        # Setup CV control for tuning
        if (n_classes == 2) {
          ctrl_tune <- caret::trainControl(method = "cv", number = 5,
                                           classProbs = TRUE,
                                           summaryFunction = twoClassSummary,
                                           allowParallel = FALSE)
          metric_tune <- "ROC"
        } else {
          ctrl_tune <- caret::trainControl(method = "cv", number = 5,
                                           classProbs = TRUE,
                                           allowParallel = FALSE)
          metric_tune <- "Accuracy"
        }

        set.seed(SEED)
        svm_tune_model <- suppressWarnings(
          caret::train(
            target ~ ., data = svm_tune_data,
            method = "svmRadial",
            trControl = ctrl_tune,
            metric = metric_tune,
            preProcess = c("center", "scale"),
            tuneGrid = expand.grid(C = c(0.1, 1, 10), sigma = c(0.01, 0.1, 1))
          )
        )
        list(C = svm_tune_model$bestTune$C, sigma = svm_tune_model$bestTune$sigma)
      }, timeout = PER_ITERATION_TIMEOUT * 5, on_timeout = "error")  # 5x longer for tuning
    }, error = function(e) {
      cat(" FAILED (using defaults C=1, sigma=0.1)\n")
      list(C = 1, sigma = 0.1)  # fallback defaults
    })

    if (!is.null(best_svm_params)) {
      best_svm_C <- best_svm_params$C
      best_svm_sigma <- best_svm_params$sigma
      cat(sprintf(" best C = %.2f, sigma = %.3f\n", best_svm_C, best_svm_sigma))
    }
  }

  # Now run 100 iterations with fixed parameters
  for (i in 1:n_runs) {
    # Early stopping check
    if (i == EARLY_STOP_CHECK && n_successful == 0) {
      cat(sprintf("\n[Early stop: 0 successful runs after %d attempts]\n", EARLY_STOP_CHECK))
      break
    }

    run_result <- tryCatch({
      # Wrap entire iteration in timeout
      eval_with_timeout({
        set.seed(i)
        if (training_and_validation_subsplits) {
          inTraining <- caret::createDataPartition(
            train_target,
            p = TRAINING_PARTITION_SIZE,
            list = FALSE
          )
          train_df <- train_df_all[inTraining,, drop = FALSE]
          train_target_factor <- as.factor(train_target[inTraining])
          inValidation <- caret::createDataPartition(
            test_target,
            p = VALIDATION_PARTITION_SIZE,
            list = FALSE
          )
          test_df <- test_df_all[inValidation,, drop = FALSE]
          test_target_factor <- as.factor(test_target[inValidation])
        } else {
          train_df <- train_df_all
          train_target_factor <- as.factor(train_target)
          test_df <- test_df_all
          test_target_factor <- as.factor(test_target)
        }
        levels(test_target_factor) <- levels(train_target_factor)

        # Get number of classes
        n_classes <- length(levels(train_target_factor))

        if (classifier_type == "RF") {
          if (n_classes < 2) stop("Less than 2 classes in training data")

          # Use randomForest with tuned parameters
          if (tune_RF && exists("best_rf")) {
            model <- randomForest(
              x = train_df,
              y = train_target_factor,
              mtry = best_rf$mtry,
              ntree = 1000
            )
          } else {
            model <- randomForest(
              x = train_df,
              y = train_target_factor,
              ntree = 1000
            )
          }

          pred <- predict(model, test_df, type = "class")
          prob <- predict(model, test_df, type = "prob")

          # Calculate AUC for both binary and multi-class
          if (n_classes == 2) {
            roc_obj <- pROC::roc(as.numeric(test_target_factor), prob[, 2], quiet = TRUE)
            auc_val <- as.numeric(roc_obj$auc)
          } else if (n_classes > 2) {
            roc_obj <- pROC::multiclass.roc(as.numeric(test_target_factor), prob, quiet = TRUE)
            auc_val <- as.numeric(roc_obj$auc)
          } else {
            auc_val <- NA
          }

          cm <- caret::confusionMatrix(pred, test_target_factor)
          ba_val <- cm$byClass["Balanced Accuracy"]

          list(ba = ba_val, auc = auc_val, status = "success")

        } else if (classifier_type == "LR") {
          if (n_classes < 2) stop("Less than 2 classes in training data")

          if (n_classes == 2) {
            lr_train_data <- train_df
            lr_train_data$target <- as.factor(train_target_factor)
            formula_str <- if (ncol(train_data) == 1) {
              paste("target ~", names(train_data)[1])
            } else {
              "target ~ ."
            }
            model <- glm(as.formula(formula_str), data = lr_train_data, family = binomial)
            prob_vec <- predict(model, test_df, type = "response")
            target_levels <- levels(train_target_factor)
            pred <- as.factor(ifelse(prob_vec > 0.5, target_levels[2], target_levels[1]))
            roc_obj <- pROC::roc(as.numeric(test_target_factor), prob_vec, quiet = TRUE)
            auc_val <- as.numeric(roc_obj$auc)
          } else {
            if (!requireNamespace("nnet", quietly = TRUE)) {
              stop("nnet package required for multi-class logistic regression")
            }
            lr_train_data <- train_df
            lr_train_data$target <- as.factor(train_target_factor)
            formula_str <- if (ncol(train_data) == 1) {
              paste("target ~", names(train_data)[1])
            } else {
              "target ~ ."
            }
            model <- nnet::multinom(as.formula(formula_str), data = lr_train_data, trace = FALSE)
            pred <- predict(model, test_df, type = "class")
            prob_mat <- predict(model, test_df, type = "probs")

            if (!is.matrix(prob_mat)) {
              prob_mat <- matrix(prob_mat, nrow = 1)
            }

            roc_obj <- pROC::multiclass.roc(as.numeric(test_target_factor), prob_mat, quiet = TRUE)
            auc_val <- as.numeric(roc_obj$auc)
            pred <- factor(pred, levels = levels(train_target_factor))
          }

          cm <- caret::confusionMatrix(pred, test_target_factor)
          ba_val <- cm$byClass["Balanced Accuracy"]

          list(ba = ba_val, auc = auc_val, status = "success")

        } else if (classifier_type == "KNN") {
          knn_train_data <- train_df
          knn_train_data$target <- as.factor(train_target_factor)
          if (length(levels(knn_train_data$target)) < 2) stop("Less than 2 classes in training data")

          original_levels <- levels(knn_train_data$target)
          n_classes_knn <- length(original_levels)

          new_levels <- paste0("Class", seq_len(n_classes_knn))
          levels(knn_train_data$target) <- new_levels

          # Use method = "none" with pre-tuned k (no internal CV)
          ctrl <- caret::trainControl(method = "none",
                                      classProbs = TRUE,
                                      allowParallel = FALSE)

          # Use tuned k if available, else default to 5
          k_value <- if(tune_KNN && exists("best_knn_k")) best_knn_k else 5

          model <- suppressWarnings(
            caret::train(
              target ~ ., data = knn_train_data,
              method = "knn",
              trControl = ctrl,
              preProcess = c("center", "scale"),
              tuneGrid = data.frame(k = k_value)
            )
          )

          test_target_factor_knn <- test_target_factor
          levels(test_target_factor_knn) <- new_levels

          pred <- predict(model, test_df)
          prob <- predict(model, test_df, type = "prob")

          if (n_classes_knn == 2) {
            roc_obj <- pROC::roc(as.numeric(test_target_factor_knn), prob[, new_levels[2]], quiet = TRUE)
            auc_val <- as.numeric(roc_obj$auc)
          } else if (n_classes_knn > 2) {
            roc_obj <- pROC::multiclass.roc(as.numeric(test_target_factor_knn), prob, quiet = TRUE)
            auc_val <- as.numeric(roc_obj$auc)
          } else {
            auc_val <- NA
          }

          cm <- caret::confusionMatrix(pred, test_target_factor_knn)
          ba_val <- cm$byClass["Balanced Accuracy"]

          list(ba = ba_val, auc = auc_val, status = "success")

        } else if (classifier_type == "SVM") {
          svm_train_data <- train_df
          svm_train_data$target <- as.factor(train_target_factor)
          if (length(levels(svm_train_data$target)) < 2) stop("Less than 2 classes in training data")

          original_levels <- levels(svm_train_data$target)
          n_classes_svm <- length(original_levels)

          new_levels <- paste0("Class", seq_len(n_classes_svm))
          levels(svm_train_data$target) <- new_levels

          # Use method = "none" with pre-tuned C and sigma (no internal CV)
          ctrl <- caret::trainControl(method = "none",
                                      classProbs = TRUE,
                                      allowParallel = FALSE)

          # Use tuned parameters if available, else defaults
          C_value <- if(tune_SVM && exists("best_svm_C")) best_svm_C else 1
          sigma_value <- if(tune_SVM && exists("best_svm_sigma")) best_svm_sigma else 0.1

          model <- suppressWarnings(
            caret::train(
              target ~ ., data = svm_train_data,
              method = "svmRadial",
              trControl = ctrl,
              preProcess = c("center", "scale"),
              tuneGrid = data.frame(C = C_value, sigma = sigma_value)
            )
          )

          test_target_factor_svm <- test_target_factor
          levels(test_target_factor_svm) <- new_levels

          pred <- predict(model, test_df)
          prob <- predict(model, test_df, type = "prob")

          if (n_classes_svm == 2) {
            roc_obj <- pROC::roc(as.numeric(test_target_factor_svm), prob[, new_levels[2]], quiet = TRUE)
            auc_val <- as.numeric(roc_obj$auc)
          } else if (n_classes_svm > 2) {
            roc_obj <- pROC::multiclass.roc(as.numeric(test_target_factor_svm), prob, quiet = TRUE)
            auc_val <- as.numeric(roc_obj$auc)
          } else {
            auc_val <- NA
          }

          cm <- caret::confusionMatrix(pred, test_target_factor_svm)
          ba_val <- cm$byClass["Balanced Accuracy"]

          list(ba = ba_val, auc = auc_val, status = "success")

        } else if (classifier_type == "C50") {
          if (!requireNamespace("C50", quietly = TRUE)) stop("C50 package not installed")
          if (n_classes < 2) stop("Less than 2 classes in training data")
          train_data_c50 <- train_df
          train_data_c50$target <- as.factor(train_target_factor)
          formula_str <- if (ncol(train_df) == 1) {
            paste("target ~", names(train_df)[1])
          } else {
            "target ~ ."
          }
          model <- C50::C5.0(as.formula(formula_str), data = train_data_c50)
          pred <- predict(model, test_df, type = "class")
          prob <- predict(model, test_df, type = "prob")

          if (n_classes == 2) {
            roc_obj <- pROC::roc(as.numeric(test_target_factor), prob[, 2], quiet = TRUE)
            auc_val <- as.numeric(roc_obj$auc)
          } else if (n_classes > 2) {
            roc_obj <- pROC::multiclass.roc(as.numeric(test_target_factor), prob, quiet = TRUE)
            auc_val <- as.numeric(roc_obj$auc)
          } else {
            auc_val <- NA
          }

          cm <- caret::confusionMatrix(pred, test_target_factor)
          ba_val <- cm$byClass["Balanced Accuracy"]

          list(ba = ba_val, auc = auc_val, status = "success")

        } else {
          stop("Unsupported classifier_type")
        }
      }, timeout = PER_ITERATION_TIMEOUT, on_timeout = "error")

    }, error = function(e) {
      # Check if timeout error
      if (grepl("reached elapsed time limit", e$message)) {
        n_timeouts <<- n_timeouts + 1
        list(ba = NA, auc = NA, status = "timeout")
      } else {
        n_errors <<- n_errors + 1
        list(ba = NA, auc = NA, status = "error")
      }
    })

    # Store results
    if (!is.null(run_result)) {
      ba_results[i] <- run_result$ba
      auc_results[i] <- run_result$auc
      if (run_result$status == "success") {
        n_successful <- n_successful + 1
      }
    } else {
      ba_results[i] <- NA
      auc_results[i] <- NA
      n_errors <- n_errors + 1
    }

    if (i %% 20 == 0) cat(".")
  }

  valid_ba <- ba_results[!is.na(ba_results)]
  valid_auc <- auc_results[!is.na(auc_results)]

  # Check minimum successful runs threshold
  if (length(valid_ba) < MIN_SUCCESSFUL_RUNS) {
    return(list(
      ba_median = NA, ba_ci_lower = NA, ba_ci_upper = NA,
      auc_median = NA, auc_ci_lower = NA, auc_ci_upper = NA,
      n_successful = length(valid_ba),
      n_timeouts = n_timeouts,
      n_errors = n_errors
    ))
  }

  ba_ci_lower <- stats::quantile(valid_ba, 0.025, na.rm = TRUE)
  ba_ci_upper <- stats::quantile(valid_ba, 0.975, na.rm = TRUE)
  auc_ci_lower <- stats::quantile(valid_auc, 0.025, na.rm = TRUE)
  auc_ci_upper <- stats::quantile(valid_auc, 0.975, na.rm = TRUE)

  return(list(
    ba_median = median(valid_ba, na.rm = TRUE),
    ba_ci_lower = ba_ci_lower,
    ba_ci_upper = ba_ci_upper,
    auc_median = median(valid_auc, na.rm = TRUE),
    auc_ci_lower = auc_ci_lower,
    auc_ci_upper = auc_ci_upper,
    n_successful = length(valid_ba),
    n_timeouts = n_timeouts,
    n_errors = n_errors
  ))
}


# Helper function to check for zero variance
has_zero_variance <- function(data) {
  if (ncol(data) == 0 || nrow(data) == 0) return(TRUE)
  variances <- apply(data, 2, var, na.rm = TRUE)
  all(variances == 0 | is.na(variances))  # Changed: any() → all()
}

# Classification function with 100 runs
quick_classify_100_runs <- function(train_data, train_target, test_data, test_target, dataset_name) {
  cat(sprintf("\n--- %s (%d features) ---\n", dataset_name, ncol(train_data)))
  if (ncol(train_data) == 0 || nrow(train_data) == 0) {
    cat("No data available - skipping\n")
    return(NULL)
  }

  # Check for zero variance features - return chance performance
  if (has_zero_variance(train_data)) {
    cat("Zero variance detected - returning chance performance (BA = 0.5, AUC = 0.5)\n")
    return(list(
      ba_median = 0.5, ba_ci_lower = 0.5, ba_ci_upper = 0.5,
      auc_median = 0.5, auc_ci_lower = 0.5, auc_ci_upper = 0.5,
      n_successful = 100
    ))
  }

  results <- list()

  # RF
  cat("Running Random Forest 100 times")
  rf_results <- run_classifier_multiple_times(train_data, train_target, test_data, test_target, "RF", 100)
  cat(sprintf(" (%d successful, %d timeouts, %d errors)\n",
              rf_results$n_successful, rf_results$n_timeouts, rf_results$n_errors))
  results$RF <- rf_results
  if (!is.null(rf_results$ba_median) && length(rf_results$ba_median) > 0 && !is.na(rf_results$ba_median)) {
    cat(sprintf("RF - BA: %.3f [%.3f, %.3f], AUC: %.3f [%.3f, %.3f]\n",
                rf_results$ba_median, rf_results$ba_ci_lower, rf_results$ba_ci_upper,
                rf_results$auc_median, rf_results$auc_ci_lower, rf_results$auc_ci_upper))
  } else {
    cat("RF - Failed to get valid results\n")
  }

  # LR
  cat("Running Logistic Regression 100 times")
  lr_results <- run_classifier_multiple_times(train_data, train_target, test_data, test_target, "LR", 100)
  cat(sprintf(" (%d successful, %d timeouts, %d errors)\n",
              lr_results$n_successful, lr_results$n_timeouts, lr_results$n_errors))
  results$LR <- lr_results
  if (!is.null(lr_results$ba_median) && length(lr_results$ba_median) > 0 && !is.na(lr_results$ba_median)) {
    cat(sprintf("LR - BA: %.3f [%.3f, %.3f], AUC: %.3f [%.3f, %.3f]\n",
                lr_results$ba_median, lr_results$ba_ci_lower, lr_results$ba_ci_upper,
                lr_results$auc_median, lr_results$auc_ci_lower, lr_results$auc_ci_upper))
  } else {
    cat("LR - Failed to get valid results\n")
  }

  # KNN
  cat("Running KNN 100 times")
  knn_results <- run_classifier_multiple_times(train_data, train_target, test_data, test_target, "KNN", 100)
  cat(sprintf(" (%d successful, %d timeouts, %d errors)\n",
              knn_results$n_successful, knn_results$n_timeouts, knn_results$n_errors))
  results$KNN <- knn_results
  if (!is.null(knn_results$ba_median) && length(knn_results$ba_median) > 0 && !is.na(knn_results$ba_median)) {
    cat(sprintf("KNN - BA: %.3f [%.3f, %.3f], AUC: %.3f [%.3f, %.3f]\n",
                knn_results$ba_median, knn_results$ba_ci_lower, knn_results$ba_ci_upper,
                knn_results$auc_median, knn_results$auc_ci_lower, knn_results$auc_ci_upper))
  } else {
    cat("KNN - Failed to get valid results\n")
  }

  # C5.0
  cat("Running C5.0 100 times")
  c50_results <- run_classifier_multiple_times(train_data, train_target, test_data, test_target, "C50", 100)
  cat(sprintf(" (%d successful, %d timeouts, %d errors)\n",
              c50_results$n_successful, c50_results$n_timeouts, c50_results$n_errors))
  results$C50 <- c50_results
  if (!is.null(c50_results$ba_median) && length(c50_results$ba_median) > 0 && !is.na(c50_results$ba_median)) {
    cat(sprintf("C5.0 - BA: %.3f [%.3f, %.3f], AUC: %.3f [%.3f, %.3f]\n",
                c50_results$ba_median, c50_results$ba_ci_lower, c50_results$ba_ci_upper,
                c50_results$auc_median, c50_results$auc_ci_lower, c50_results$auc_ci_upper))
  } else {
    cat("C5.0 - Failed to get valid results\n")
  }

  # SVM
  cat("Running SVM 100 times")
  svm_results <- run_classifier_multiple_times(train_data, train_target, test_data, test_target, "SVM", 100)
  cat(sprintf(" (%d successful, %d timeouts, %d errors)\n",
              svm_results$n_successful, svm_results$n_timeouts, svm_results$n_errors))
  results$SVM <- svm_results
  if (!is.null(svm_results$ba_median) && length(svm_results$ba_median) > 0 && !is.na(svm_results$ba_median)) {
    cat(sprintf("SVM - BA: %.3f [%.3f, %.3f], AUC: %.3f [%.3f, %.3f]\n",
                svm_results$ba_median, svm_results$ba_ci_lower, svm_results$ba_ci_upper,
                svm_results$auc_median, svm_results$auc_ci_lower, svm_results$auc_ci_upper))
  } else {
    cat("SVM - Failed to get valid results\n")
  }

  return(results)
}

create_results_table <- function(test_results, datasets_to_test) {
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
      C50_BA = character(),
      C50_AUC = character(),
      SVM_BA = character(),
      SVM_AUC = character(),
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
      C50_BA = character(),
      SVM_BA = character(),
      Classification_Success = integer(),
      stringsAsFactors = FALSE
    )
  }

  for (name in names(test_results)) {
    # Skip NULL results (e.g., empty datasets)
    if (is.null(test_results[[name]])) {
      next
    }

    rf_res <- test_results[[name]]$RF
    # Check if rf_res exists and has valid ba_median
    if (!is.null(rf_res) && length(rf_res$ba_median) > 0 && !is.na(rf_res$ba_median)) {
      rf_ba_str <- sprintf("%.3f [%.3f, %.3f]", rf_res$ba_median, rf_res$ba_ci_lower, rf_res$ba_ci_upper)
      if (use_roc_auc) rf_auc_str <- sprintf("%.3f [%.3f, %.3f]", rf_res$auc_median, rf_res$auc_ci_lower, rf_res$auc_ci_upper) else rf_auc_str <- NULL
    } else {
      rf_ba_str <- "NA"
      if (use_roc_auc) rf_auc_str <- "NA" else rf_auc_str <- NULL
    }

    lr_res <- test_results[[name]]$LR
    if (!is.null(lr_res) && length(lr_res$ba_median) > 0 && !is.na(lr_res$ba_median)) {
      lr_ba_str <- sprintf("%.3f [%.3f, %.3f]", lr_res$ba_median, lr_res$ba_ci_lower, lr_res$ba_ci_upper)
      if (use_roc_auc) lr_auc_str <- sprintf("%.3f [%.3f, %.3f]", lr_res$auc_median, lr_res$auc_ci_lower, lr_res$auc_ci_upper) else lr_auc_str <- NULL
    } else {
      lr_ba_str <- "NA"
      if (use_roc_auc) lr_auc_str <- "NA" else lr_auc_str <- NULL
    }

    knn_res <- test_results[[name]]$KNN
    if (!is.null(knn_res) && length(knn_res$ba_median) > 0 && !is.na(knn_res$ba_median)) {
      knn_ba_str <- sprintf("%.3f [%.3f, %.3f]", knn_res$ba_median, knn_res$ba_ci_lower, knn_res$ba_ci_upper)
      if (use_roc_auc) knn_auc_str <- sprintf("%.3f [%.3f, %.3f]", knn_res$auc_median, knn_res$auc_ci_lower, knn_res$auc_ci_upper) else knn_auc_str <- NULL
    } else {
      knn_ba_str <- "NA"
      if (use_roc_auc) knn_auc_str <- "NA" else knn_auc_str <- NULL
    }

    c50_res <- test_results[[name]]$C50
    if (!is.null(c50_res) && length(c50_res$ba_median) > 0 && !is.na(c50_res$ba_median)) {
      c50_ba_str <- sprintf("%.3f [%.3f, %.3f]", c50_res$ba_median, c50_res$ba_ci_lower, c50_res$ba_ci_upper)
      if (use_roc_auc) c50_auc_str <- sprintf("%.3f [%.3f, %.3f]", c50_res$auc_median, c50_res$auc_ci_lower, c50_res$auc_ci_upper) else c50_auc_str <- NULL
    } else {
      c50_ba_str <- "NA"
      if (use_roc_auc) c50_auc_str <- "NA" else c50_auc_str <- NULL
    }

    svm_res <- test_results[[name]]$SVM
    if (!is.null(svm_res) && length(svm_res$ba_median) > 0 && !is.na(svm_res$ba_median)) {
      svm_ba_str <- sprintf("%.3f [%.3f, %.3f]", svm_res$ba_median, svm_res$ba_ci_lower, svm_res$ba_ci_upper)
      if (use_roc_auc) svm_auc_str <- sprintf("%.3f [%.3f, %.3f]", svm_res$auc_median, svm_res$auc_ci_lower, svm_res$auc_ci_upper) else svm_auc_str <- NULL
    } else {
      svm_ba_str <- "NA"
      if (use_roc_auc) svm_auc_str <- "NA" else svm_auc_str <- NULL
    }

    if (use_roc_auc) {
      success_flag <- any(
        !is.null(rf_res) && length(rf_res$ba_ci_lower) > 0 && !is.na(rf_res$ba_ci_lower) && rf_res$ba_ci_lower > 0.5,
        !is.null(rf_res) && length(rf_res$auc_ci_lower) > 0 && !is.na(rf_res$auc_ci_lower) && rf_res$auc_ci_lower > 0.5,
        !is.null(lr_res) && length(lr_res$ba_ci_lower) > 0 && !is.na(lr_res$ba_ci_lower) && lr_res$ba_ci_lower > 0.5,
        !is.null(lr_res) && length(lr_res$auc_ci_lower) > 0 && !is.na(lr_res$auc_ci_lower) && lr_res$auc_ci_lower > 0.5,
        !is.null(knn_res) && length(knn_res$ba_ci_lower) > 0 && !is.na(knn_res$ba_ci_lower) && knn_res$ba_ci_lower > 0.5,
        !is.null(knn_res) && length(knn_res$auc_ci_lower) > 0 && !is.na(knn_res$auc_ci_lower) && knn_res$auc_ci_lower > 0.5,
        !is.null(c50_res) && length(c50_res$ba_ci_lower) > 0 && !is.na(c50_res$ba_ci_lower) && c50_res$ba_ci_lower > 0.5,
        !is.null(c50_res) && length(c50_res$auc_ci_lower) > 0 && !is.na(c50_res$auc_ci_lower) && c50_res$auc_ci_lower > 0.5,
        !is.null(svm_res) && length(svm_res$ba_ci_lower) > 0 && !is.na(svm_res$ba_ci_lower) && svm_res$ba_ci_lower > 0.5,
        !is.null(svm_res) && length(svm_res$auc_ci_lower) > 0 && !is.na(svm_res$auc_ci_lower) && svm_res$auc_ci_lower > 0.5
      )
    } else {
      success_flag <- any(
        !is.null(rf_res) && length(rf_res$ba_ci_lower) > 0 && !is.na(rf_res$ba_ci_lower) && rf_res$ba_ci_lower > 0.5,
        !is.null(lr_res) && length(lr_res$ba_ci_lower) > 0 && !is.na(lr_res$ba_ci_lower) && lr_res$ba_ci_lower > 0.5,
        !is.null(knn_res) && length(knn_res$ba_ci_lower) > 0 && !is.na(knn_res$ba_ci_lower) && knn_res$ba_ci_lower > 0.5,
        !is.null(c50_res) && length(c50_res$ba_ci_lower) > 0 && !is.na(c50_res$ba_ci_lower) && c50_res$ba_ci_lower > 0.5,
        !is.null(svm_res) && length(svm_res$ba_ci_lower) > 0 && !is.na(svm_res$ba_ci_lower) && svm_res$ba_ci_lower > 0.5
      )
    }
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
        C50_BA = c50_ba_str,
        C50_AUC = c50_auc_str,
        SVM_BA = svm_ba_str,
        SVM_AUC = svm_auc_str,
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
        C50_BA = c50_ba_str,
        SVM_BA = svm_ba_str,
        Classification_Success = as.integer(success_flag),
        stringsAsFactors = FALSE
      ))
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
    )
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


# cat("Creating Cohen's d analysis with t-tests and visualization...\n")

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
        p_value < 0.01 ~ "**",
        p_value < 0.05 ~ "*",
        p_value < 0.1 ~ ".",
        TRUE ~ ""
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
    Feature = all_features,
    Boruta_Selected = as.integer(all_features %in% boruta_res$selected),
    LASSO_Selected = as.integer(all_features %in% lasso_plot_data$Feature[lasso_plot_data$Decision != "Rejected"]),
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

  # Attempt to get highly correlated vars to drop; handle errors if cor() fails
  high_corr_vars <- character(0) # default empty

  try({
    res <- find_variables_to_drop_caret(
      data = training_data_actual,
      method = CORRELATION_METHOD,
      cutoff = CORRELATION_LIMIT
    )
    if (!is.null(res) && length(res$vars_to_drop) > 0) {
      high_corr_vars <- res$vars_to_drop
    }
  }, silent = TRUE)

  # Build datasets_to_test list
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
      train = if ((length(boruta_selected) + length(lasso_selected)) > 0)
        training_data_actual[, union(boruta_selected, lasso_selected), drop = FALSE] else data.frame(),
      test = if ((length(boruta_selected) + length(lasso_selected)) > 0)
        validation_data_actual[, union(boruta_selected, lasso_selected), drop = FALSE] else data.frame()
    ),
    "Boruta_LASSO_Rejected" = list(
      train = if ((length(boruta_rejected) + length(lasso_rejected)) > 0)
        training_data_actual[, setdiff(names(training_data_actual), union(boruta_selected, lasso_selected)), drop = FALSE] else data.frame(),
      test = if ((length(boruta_rejected) + length(lasso_rejected)) > 0)
        validation_data_actual[, setdiff(names(training_data_actual), union(boruta_selected, lasso_selected)), drop = FALSE] else data.frame()
    ),
    "Only_high_correlated" = list(
      train = if (length(high_corr_vars) > 0) training_data_actual[, high_corr_vars, drop = FALSE] else data.frame(),
      test = if (length(high_corr_vars) > 0) validation_data_actual[, high_corr_vars, drop = FALSE] else data.frame()
    )
  )


  # Run classification tests
  test_results <- list()
  for (dataset_name in names(datasets_to_test)) {
    dataset <- datasets_to_test[[dataset_name]]
    test_results[[dataset_name]] <- quick_classify_100_runs(
      train_data = dataset$train, train_target = training_target, test_data = dataset$test, test_target = validation_target, dataset_name = dataset_name
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

###############################################################################
# --- Iterative Analysis Function ---
###############################################################################

###############################################################################
# --- Streamlined Minimal Feature Selection Function ---
###############################################################################

run_feature_selection_iterations <- function() {
  all_results_feature_selection <- list()

  # --- Phase 0: Initial Analysis ---
  cat("\n=== PHASE 0: Initial Feature Selection Analysis ===\n")
  full_results <- run_analysis_pipeline(
    training_data_actual = training_data_original,
    training_target = training_target,
    validation_data_actual = validation_data_original,
    validation_target = validation_target,
    use_curated_subset = FALSE,
    curated_names = NULL,
    add_file_string = "_phase0_full"
  )
  all_results_feature_selection[["Phase_0_Full"]] <- full_results
  if (!is.null(full_results$plots$matrix)) print(full_results$plots$matrix)

  results_table_full <- full_results$results_table
  available_features <- names(training_data_original)

  # Find ALL successful datasets from Phase 0
  successful_datasets <- results_table_full[results_table_full$Classification_Success == 1, ]

  # CRITICAL CHECK: If NO dataset in Phase 0 succeeded, stop
  if (nrow(successful_datasets) == 0) {
    cat("\n")
    cat(paste(rep("!", 80), collapse = ""), "\n")
    cat("!!! CRITICAL: No Classifiable Feature Set Found !!!\n")
    cat(paste(rep("!", 80), collapse = ""), "\n")
    cat("\nALL feature sets tested in Phase 0 failed to achieve classification success\n")
    cat("(lower CI for BA > 0.5), including:\n")
    cat("  - Complete feature set (All_Features)\n")
    cat("  - Boruta-selected features\n")
    cat("  - LASSO-selected features\n")
    cat("  - All other tested combinations\n\n")
    cat("This indicates:\n")
    cat("  1. The dataset does not contain sufficient information for classification\n")
    cat("  2. The target variable is not predictable from these features\n")
    cat("  3. Sample size may be insufficient for reliable classification\n")
    cat("  4. Data quality issues or excessive noise may be present\n\n")
    cat(sprintf("Total features available: %d\n", ncol(training_data_original)))
    cat(sprintf("Feature subsets evaluated: %d\n", nrow(results_table_full)))
    cat(sprintf("Classifiers tested per subset: RF, LR, KNN, C5.0 (100 bootstrap iterations each)\n\n"))
    cat("RECOMMENDATION: Review data quality, target definition, and sample size.\n")
    cat("The dataset is not suitable for classification in its current form.\n")
    cat(paste(rep("!", 80), collapse = ""), "\n\n")

    return(list(
      results_list = all_results_feature_selection,
      combined_table = results_table_full,
      final_features = NULL,
      classification_failed = TRUE
    ))
  }

  # If we reach here, at least ONE feature set in Phase 0 succeeded
  # Find the smallest successful set to start minimization
  smallest_idx <- which.min(successful_datasets$Features)
  smallest_successful_dataset <- successful_datasets[smallest_idx, ]
  smallest_successful_name <- smallest_successful_dataset$Dataset

  # Extract actual features from this dataset
  if (smallest_successful_name %in% names(full_results$datasets_to_test)) {
    starting_features <- colnames(full_results$datasets_to_test[[smallest_successful_name]]$train)
  } else {
    cat("\nCannot extract features from smallest successful set. Using Boruta+LASSO union.\n")
    boruta_res <- get_boruta_features(full_results$boruta_results$finalDecision, Boruta_tentative_in)
    lasso_selected <- full_results$lasso_results$selected
    starting_features <- union(boruta_res$selected, lasso_selected)
  }

  # Warning if smallest successful set is the full feature set
  if (smallest_successful_name == "All_Features") {
    cat("\n")
    cat(paste(rep("!", 80), collapse = ""), "\n")
    cat("!!! WARNING: Feature Selection Limitation !!!\n")
    cat(paste(rep("!", 80), collapse = ""), "\n")
    cat("\nThe smallest successful feature set is the COMPLETE feature set.\n")
    cat("No feature selection subsets (Boruta, LASSO, etc.) achieved success.\n\n")
    cat("This indicates:\n")
    cat("  1. All features may be necessary for classification\n")
    cat("  2. Feature selection methods did not identify sufficient subsets\n")
    cat("  3. Strong interdependencies among features\n\n")
    cat(sprintf("Total features: %d\n", length(starting_features)))
    cat("\nIMPLICATION: Proceeding with backward elimination on full feature set.\n")
    cat(paste(rep("!", 80), collapse = ""), "\n\n")

    full_set_required_warning <- TRUE
  } else {
    full_set_required_warning <- FALSE
  }

  cat(sprintf("\nSmallest successful set from Phase 0: '%s'\n", smallest_successful_name))
  cat(sprintf("  Contains %d features: %s\n", length(starting_features),
              paste(starting_features, collapse = ", ")))

  # --- Phase 1: Minimize the Successful Set ---
  cat("\n=== PHASE 1: Minimizing Feature Set ===\n")
  cat("Attempting to remove features one-by-one while maintaining classification success...\n\n")
  
  current_features <- starting_features
  removed_features <- c()
  
  while (length(current_features) > 1) {
    cat(sprintf("Current set size: %d features\n", length(current_features)))
    
    # Test removing each feature
    best_removal <- NULL
    best_removal_ba <- -Inf
    can_remove_any <- FALSE
    
    for (feature_to_test in current_features) {
      remaining <- setdiff(current_features, feature_to_test)
      
      cat(sprintf("  Testing removal of '%s' ... ", feature_to_test))
      
      test_results <- quick_classify_100_runs(
        train_data = training_data_original[, remaining, drop = FALSE],
        train_target = training_target,
        test_data = validation_data_original[, remaining, drop = FALSE],
        test_target = validation_target,
        dataset_name = paste0("Minimize_Without_", feature_to_test)
      )
      
      # Check if ANY classifier still succeeds (lower CI > 0.5)
      still_succeeds <- any(
        test_results$RF$ba_ci_lower > 0.5,
        test_results$LR$ba_ci_lower > 0.5,
        test_results$KNN$ba_ci_lower > 0.5,
        test_results$C50$ba_ci_lower > 0.5,
        na.rm = TRUE
      )
      
      mdn_ba <- median(c(test_results$RF$ba_median, test_results$LR$ba_median,
                       test_results$KNN$ba_median, test_results$C50$ba_median), na.rm = TRUE)
      
      if (still_succeeds) {
        cat(sprintf("CAN remove (BA=%.3f, success maintained)\n", mdn_ba))
        can_remove_any <- TRUE
        if (mdn_ba > best_removal_ba) {
          best_removal_ba <- mdn_ba
          best_removal <- feature_to_test
        }
      } else {
        cat(sprintf("CANNOT remove (BA=%.3f, would lose classification)\n", mdn_ba))
      }
    }
    
    if (can_remove_any && !is.null(best_removal)) {
      cat(sprintf("\n→ Removing '%s' (best candidate, BA=%.3f)\n\n", best_removal, best_removal_ba))
      current_features <- setdiff(current_features, best_removal)
      removed_features <- c(removed_features, best_removal)
    } else {
      cat("\n→ Cannot remove any more features without losing classification success.\n")
      break
    }
  }
  
  minimal_features <- current_features
  cat(sprintf("\nPhase 1 complete. Minimal feature set: %d features\n", length(minimal_features)))
  cat(sprintf("  Features: %s\n", paste(minimal_features, collapse = ", ")))
  cat(sprintf("  Removed: %d features\n", length(removed_features)))

  # --- Phase 2: Rescue Wrongly Rejected Features ---
  cat("\n=== PHASE 2: Testing Rejected Features for Rescue ===\n")

  rejected_features <- setdiff(available_features, minimal_features)
  rescued_features <- c()

  if (length(rejected_features) > 0) {
    cat(sprintf("Testing %d rejected features individually in parallel...\n\n", length(rejected_features)))

    # Parallelize individual feature testing
    rescue_results <- pbmcapply::pbmclapply(rejected_features, function(feature) {
      # Check for zero variance before testing
      feature_data <- training_data_original[, feature, drop = FALSE]
      if (has_zero_variance(feature_data)) {
        return(list(feature = feature, rescued = FALSE, reason = "zero_variance"))
      }

      test_results <- quick_classify_100_runs(
        train_data = feature_data,
        train_target = training_target,
        test_data = validation_data_original[, feature, drop = FALSE],
        test_target = validation_target,
        dataset_name = paste0("Rescue_Test_", feature)
      )

      # Check if ANY classifier succeeds (lower CI > 0.5)
      can_classify <- any(
        test_results$RF$ba_ci_lower > 0.5,
        test_results$LR$ba_ci_lower > 0.5,
        test_results$KNN$ba_ci_lower > 0.5,
        test_results$C50$ba_ci_lower > 0.5,
        na.rm = TRUE
      )

      min_ci <- min(c(test_results$RF$ba_ci_lower, test_results$LR$ba_ci_lower,
                      test_results$KNN$ba_ci_lower, test_results$C50$ba_ci_lower), na.rm = TRUE)

      return(list(feature = feature, rescued = can_classify, min_ci = min_ci))
    }, mc.cores = parallel::detectCores() - 1)

    # Process results
    for (result in rescue_results) {
      if (!is.null(result$reason) && result$reason == "zero_variance") {
        cat(sprintf("  %s: zero variance - skipped (BA ≤ 0.5)\n", result$feature))
      } else if (result$rescued) {
        cat(sprintf("  %s: RESCUED (min lower CI=%.4f > 0.5)\n", result$feature, result$min_ci))
        rescued_features <- c(rescued_features, result$feature)
      } else {
        cat(sprintf("  %s: stays rejected (min lower CI=%.4f ≤ 0.5)\n", result$feature, result$min_ci))
      }
    }

    if (length(rescued_features) > 0) {
      cat(sprintf("\nRescued %d features: %s\n", length(rescued_features),
                  paste(rescued_features, collapse = ", ")))
    } else {
      cat("\nNo features rescued.\n")
    }
  } else {
    cat("No rejected features to test (all features in minimal set).\n")
  }
  
  # Final feature sets
  final_selected_features <- union(minimal_features, rescued_features)
  final_rejected_features <- setdiff(available_features, final_selected_features)
  
  cat(sprintf("\nPhase 2 complete.\n"))
  cat(sprintf("  Minimal set: %d features\n", length(minimal_features)))
  cat(sprintf("  Rescued: %d features\n", length(rescued_features)))
  cat(sprintf("  Final selected: %d features\n", length(final_selected_features)))
  cat(sprintf("  Final rejected: %d features\n", length(final_rejected_features)))

  # --- Phase 3: Final Verification ---
  cat("\n=== PHASE 3: Final Verification ===\n")

  # Test final selected features
  cat(sprintf("\nTesting final selected feature set (%d features)...\n", length(final_selected_features)))
  final_selected_results <- run_analysis_pipeline(
    training_data_actual = training_data_original,
    training_target = training_target,
    validation_data_actual = validation_data_original,
    validation_target = validation_target,
    use_curated_subset = TRUE,
    curated_names = final_selected_features,
    add_file_string = "_phase3_final_selected"
  )
  all_results_feature_selection[["Phase_3_Final_Selected"]] <- final_selected_results
  if (!is.null(final_selected_results$plots$matrix)) print(final_selected_results$plots$matrix)

  selected_success <- any(final_selected_results$results_table$Classification_Success == 1)
  cat(sprintf("\n✓ Final selected features: Classification %s\n",
              if(selected_success) "SUCCESSFUL ✓" else "FAILED ✗ (ERROR!)"))

  # Test final rejected features using full pipeline analysis
  final_rejected_results <- NULL
  additional_rescued <- c()

  if (length(final_rejected_features) > 0) {
    cat(sprintf("\n=== PHASE 3b: Analyzing Rejected Feature Set (%d features) ===\n",
                length(final_rejected_features)))
    cat("Running full feature selection pipeline on rejected features...\n")

    final_rejected_results <- run_analysis_pipeline(
      training_data_actual = training_data_original,
      training_target = training_target,
      validation_data_actual = validation_data_original,
      validation_target = validation_target,
      use_curated_subset = TRUE,
      curated_names = final_rejected_features,
      add_file_string = "_phase3_final_rejected"
    )
    all_results_feature_selection[["Phase_3_Final_Rejected"]] <- final_rejected_results
    if (!is.null(final_rejected_results$plots$matrix)) print(final_rejected_results$plots$matrix)

    rejected_success <- any(final_rejected_results$results_table$Classification_Success == 1)
    cat(sprintf("\n✓ Final rejected features: Classification %s\n",
                if(rejected_success) "SUCCESSFUL ✗ (WARNING!)" else "FAILED ✓ (correct)"))

    # If rejected features can classify, examine which subsets are responsible
    if (rejected_success) {
      cat("\n!!! WARNING: Rejected features can still classify as a group! !!!\n")
      cat("Analyzing feature selection results to identify responsible features...\n\n")

      # Extract features identified by Boruta/LASSO within rejected set
      rejected_boruta_res <- get_boruta_features(
        final_rejected_results$boruta_results$finalDecision,
        Boruta_tentative_in
      )
      rejected_lasso_selected <- if (!is.null(final_rejected_results$lasso_results)) {
        final_rejected_results$lasso_results$selected
      } else {
        character(0)
      }

      # Union of features selected by either method within rejected set
      features_to_rescue <- unique(c(rejected_boruta_res$selected, rejected_lasso_selected))

      if (length(features_to_rescue) > 0) {
        cat(sprintf("Feature selection within rejected set identified %d features:\n",
                    length(features_to_rescue)))
        cat(sprintf("  %s\n", paste(features_to_rescue, collapse = ", ")))
        cat("\nVerifying these features individually for rescue in parallel...\n")

        # Parallelize individual verification
        verify_results <- pbmcapply::pbmclapply(features_to_rescue, function(feature) {
          # Check for zero variance before testing
          feature_data <- training_data_original[, feature, drop = FALSE]
          if (has_zero_variance(feature_data)) {
            return(list(feature = feature, rescued = FALSE, reason = "zero_variance"))
          }

          test_results <- quick_classify_100_runs(
            train_data = feature_data,
            train_target = training_target,
            test_data = validation_data_original[, feature, drop = FALSE],
            test_target = validation_target,
            dataset_name = paste0("Rescue_Verify_", feature)
          )

          can_classify <- any(
            test_results$RF$ba_ci_lower > 0.5,
            test_results$LR$ba_ci_lower > 0.5,
            test_results$KNN$ba_ci_lower > 0.5,
            test_results$C50$ba_ci_lower > 0.5,
            na.rm = TRUE
          )

          min_ci <- min(c(test_results$RF$ba_ci_lower, test_results$LR$ba_ci_lower,
                          test_results$KNN$ba_ci_lower, test_results$C50$ba_ci_lower), na.rm = TRUE)

          return(list(feature = feature, rescued = can_classify, min_ci = min_ci))
        }, mc.cores = parallel::detectCores() - 1)

        # Process results
        for (result in verify_results) {
          if (!is.null(result$reason) && result$reason == "zero_variance") {
            cat(sprintf("  %s: zero variance - skipped (BA ≤ 0.5)\n", result$feature))
          } else if (result$rescued) {
            cat(sprintf("  %s: RESCUED (min lower CI=%.4f > 0.5)\n", result$feature, result$min_ci))
            additional_rescued <- c(additional_rescued, result$feature)
          } else {
            cat(sprintf("  %s: stays rejected (min lower CI=%.4f ≤ 0.5)\n", result$feature, result$min_ci))
          }
        }

        if (length(additional_rescued) > 0) {
          cat(sprintf("\nRescued %d features from rejected set analysis: %s\n",
                      length(additional_rescued), paste(additional_rescued, collapse = ", ")))
        } else {
          cat("\nNo features from rejected set analysis could be individually rescued.\n")
          cat("Classification success may be due to feature interactions not captured individually.\n")
        }

      } else {
        cat("Feature selection within rejected set identified NO specific features.\n")
        cat("This suggests complex feature interactions are responsible for classification.\n")
        cat("\nFalling back to individual feature testing in parallel...\n\n")

        # Parallelize fallback testing
        fallback_results <- pbmcapply::pbmclapply(final_rejected_features, function(feature) {
          # Check for zero variance before testing
          feature_data <- training_data_original[, feature, drop = FALSE]
          if (has_zero_variance(feature_data)) {
            return(list(feature = feature, rescued = FALSE, reason = "zero_variance"))
          }

          test_results <- quick_classify_100_runs(
            train_data = feature_data,
            train_target = training_target,
            test_data = validation_data_original[, feature, drop = FALSE],
            test_target = validation_target,
            dataset_name = paste0("Fallback_Rescue_", feature)
          )

          can_classify <- any(
            test_results$RF$ba_ci_lower > 0.5,
            test_results$LR$ba_ci_lower > 0.5,
            test_results$KNN$ba_ci_lower > 0.5,
            test_results$C50$ba_ci_lower > 0.5,
            na.rm = TRUE
          )

          min_ci <- min(c(test_results$RF$ba_ci_lower, test_results$LR$ba_ci_lower,
                          test_results$KNN$ba_ci_lower, test_results$C50$ba_ci_lower), na.rm = TRUE)

          return(list(feature = feature, rescued = can_classify, min_ci = min_ci))
        }, mc.cores = parallel::detectCores() - 1)

        # Process results
        for (result in fallback_results) {
          if (!is.null(result$reason) && result$reason == "zero_variance") {
            cat(sprintf("  %s: zero variance - skipped (BA ≤ 0.5)\n", result$feature))
          } else if (result$rescued) {
            cat(sprintf("  %s: RESCUED (min lower CI=%.4f > 0.5)\n", result$feature, result$min_ci))
            additional_rescued <- c(additional_rescued, result$feature)
          } else {
            cat(sprintf("  %s: stays rejected (min lower CI=%.4f ≤ 0.5)\n", result$feature, result$min_ci))
          }
        }

        # NEW WARNING CODE - INSERT HERE
        if (length(additional_rescued) == 0) {
          cat("\n")
          cat(paste(rep("!", 80), collapse = ""), "\n")
          cat("!!! CRITICAL WARNING: Feature Selection Method Limitation Detected !!!\n")
          cat(paste(rep("!", 80), collapse = ""), "\n")
          cat("\nThe rejected feature set as a group achieves classification success,\n")
          cat("but BOTH Boruta and LASSO failed to identify responsible features,\n")
          cat("and NO individual feature can classify alone.\n\n")
          cat("This indicates:\n")
          cat("  1. Complex feature interactions not captured by current methods\n")
          cat("  2. Potential synergistic effects among multiple rejected features\n")
          cat("  3. Limitations of both wrapper and embedded feature selection approaches\n\n")
          cat(sprintf("Rejected features (%d total):\n", length(final_rejected_features)))
          cat(sprintf("  %s\n", paste(final_rejected_features, collapse = ", ")))
          cat("\nIMPLICATION: The final 'rejected' set may contain features that,\n")
          cat("while individually insufficient, collectively contribute to classification\n")
          cat("through interactions that neither Boruta nor LASSO can decompose.\n\n")
          cat("RECOMMENDATION: Consider these rejected features as 'potentially informative\n")
          cat("in combination' rather than 'definitively uninformative'. Further investigation\n")
          cat("of higher-order feature interactions may be warranted.\n")
          cat(paste(rep("!", 80), collapse = ""), "\n\n")

          # Add flag to return value to indicate this condition
          warning_issued <- TRUE
        } else {
          warning_issued <- FALSE
        }
      }

      # If any features were rescued, update and re-verify
      if (length(additional_rescued) > 0) {
        cat("\n=== Re-running Phase 3 with Updated Feature Sets ===\n")

        # Update final feature sets
        final_selected_features <- union(final_selected_features, additional_rescued)
        final_rejected_features <- setdiff(final_rejected_features, additional_rescued)

        # Re-run final verification with updated sets
        cat(sprintf("\nRe-testing updated final selected feature set (%d features)...\n",
                    length(final_selected_features)))
        final_selected_results <- run_analysis_pipeline(
          training_data_actual = training_data_original,
          training_target = training_target,
          validation_data_actual = validation_data_original,
          validation_target = validation_target,
          use_curated_subset = TRUE,
          curated_names = final_selected_features,
          add_file_string = "_phase3_final_selected_updated"
        )
        all_results_feature_selection[["Phase_3_Final_Selected_Updated"]] <- final_selected_results

        if (length(final_rejected_features) > 0) {
          cat(sprintf("\nRe-testing updated final rejected feature set (%d features)...\n",
                      length(final_rejected_features)))
          final_rejected_results <- run_analysis_pipeline(
            training_data_actual = training_data_original,
            training_target = training_target,
            validation_data_actual = validation_data_original,
            validation_target = validation_target,
            use_curated_subset = TRUE,
            curated_names = final_rejected_features,
            add_file_string = "_phase3_final_rejected_updated"
          )
          all_results_feature_selection[["Phase_3_Final_Rejected_Updated"]] <- final_rejected_results

          rejected_success_updated <- any(final_rejected_results$results_table$Classification_Success == 1)
          cat(sprintf("\n✓ Updated final rejected features: Classification %s\n",
                      if(rejected_success_updated) "SUCCESSFUL ✗ (still WARNING!)" else "FAILED ✓ (now correct)"))
        } else {
          cat("\nNo rejected features remaining after rescue.\n")
        }
      }
    }
  } else {
    cat("\nNo rejected features to test (all features selected).\n")
  }
  
  # --- Compile Results ---
  add_features_column <- function(results_table, datasets_to_test) {
    results_table$Features_Used <- sapply(results_table$Dataset, function(dataset_name) {
      if (dataset_name %in% names(datasets_to_test)) {
        features <- colnames(datasets_to_test[[dataset_name]]$train)
        if (length(features) > 0) {
          return(paste(features, collapse = "; "))
        }
      }
      return("")
    })
    return(results_table)
  }
  
  combined_results_table <- add_features_column(results_table_full, full_results$datasets_to_test)
  combined_results_table$Phase <- "Phase_0_Full"
  
  final_selected_table <- add_features_column(final_selected_results$results_table, 
                                              final_selected_results$datasets_to_test)
  final_selected_table$Phase <- "Phase_3_Final_Selected"
  combined_results_table <- rbind(combined_results_table, final_selected_table)
  
  if (!is.null(final_rejected_results)) {
    final_rejected_table <- add_features_column(final_rejected_results$results_table, 
                                                final_rejected_results$datasets_to_test)
    final_rejected_table$Phase <- "Phase_3_Final_Rejected"
    combined_results_table <- rbind(combined_results_table, final_rejected_table)
  }
  
  col_order <- c(setdiff(names(combined_results_table), "Features_Used"), "Features_Used")
  combined_results_table <- combined_results_table[, col_order]
  
  write.csv(combined_results_table,
            paste0(DATASET_NAME, "_minimal_feature_selection_results.csv"), row.names = FALSE)
  
  # Save feature selection history
  feature_history <- list(
    starting_set = starting_features,
    starting_set_name = smallest_successful_name,
    minimal_features = minimal_features,
    removed_in_minimization = removed_features,
    rescued_features = rescued_features,
    final_selected = final_selected_features,
    final_rejected = final_rejected_features
  )
  
  saveRDS(feature_history, paste0(DATASET_NAME, "_minimal_feature_history.rds"))
  
  # Final summary
  cat("\n" , paste(rep("=", 70), collapse = ""), "\n")
  cat("=== MINIMAL FEATURE SELECTION COMPLETE ===\n")
  cat(paste(rep("=", 70), collapse = ""), "\n\n")
  cat(sprintf("Starting from smallest successful set ('%s'): %d features\n", 
              smallest_successful_name, length(starting_features)))
  cat(sprintf("After minimization: %d features\n", length(minimal_features)))
  cat(sprintf("Rescued from rejected: %d features\n", length(rescued_features)))
  cat(sprintf("\nFINAL SELECTED FEATURES: %d\n", length(final_selected_features)))
  cat(sprintf("  %s\n", paste(final_selected_features, collapse = ", ")))
  cat(sprintf("\nFINAL REJECTED FEATURES: %d\n", length(final_rejected_features)))
  if (length(final_rejected_features) > 0 && length(final_rejected_features) <= 100) {
    cat(sprintf("  %s\n", paste(final_rejected_features, collapse = ", ")))
  }
  cat("\n")

  return(list(
    results_list = all_results_feature_selection,
    combined_table = combined_results_table,
    final_features = final_selected_features,
    excluded_features = final_rejected_features,
    feature_history = feature_history,
    fs_limitation_warning = if(exists("warning_issued")) warning_issued else FALSE
  ))
}

cat("Main functions loaded\n")