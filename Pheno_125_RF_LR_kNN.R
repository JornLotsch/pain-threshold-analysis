###############################################################################
# Pain Thresholds Data Analysis
# 
# This script loads, preprocesses, and analyzes pain threshold data.
# It handles variable duplication (with optional noise), correlation analysis, 
# variable reduction, and visualization using Venn diagrams.
###############################################################################

###############################################################################
# Quick Test with 95% CI from 100 Model Runs
###############################################################################

# --- Libraries ---
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

# --- Constants ---
PAIN_DATA_FILE_PATH <- "/home/joern/Dokumente/PainGenesDrugs/08AnalyseProgramme/R/PainThresholdsData_transformed_imputed.csv"
TARGET_FILE_PATH <- "/home/joern/Dokumente/PainGenesDrugs/08AnalyseProgramme/R/PainThresholds.csv"

SEED <- 42 # or 12
noise_factor <- 0.05

CORRELATION_METHOD <- "pearson"
CORRELATION_LIMIT <- 0.9

Boruta_tentative_in <- FALSE

use_nyt <- TRUE

tune_RF <- TRUE

PAIN_DATA_COLUMN_NAMES <- c(
  "Heat", "Pressure", "Current", "Heat_Capsaicin",
  "Capsaicin_Effect_Heat", "Cold", "Cold_Menthol", "Menthol_Effect_Cold",
  "vonFrey", "vonFrey_Capsaicin", "Capsaicin_Effect_vonFrey"
)

PAIN_DATA_CURATED_COLUMN_NAMES <- c(
  "Heat", "Pressure", "Current", "Heat_Capsaicin",
  "Cold", "Cold_Menthol", 
  "vonFrey", "vonFrey_Capsaicin"
)

PAIN_DATA_TO_REMOVE <- c(
  "Pressure", "Current","Menthol_Effect_Cold",  "vonFrey_Capsaicin"
)

PAIN_DATA_COLINEAR <- c(
  "Capsaicin_Effect_Heat", "Menthol_Effect_Cold", "Capsaicin_Effect_vonFrey", "Pressure2"
)


# PAIN_DATA_CURATED_COLUMN_NAMES <- setdiff(PAIN_DATA_COLUMN_NAMES, PAIN_DATA_TO_REMOVE)
PAIN_DATA_CURATED_COLUMN_NAMES <- c(
  "Heat",
  "Heat_Capsaicin",
  "Capsaicin_Effect_Heat",
  "Cold",
  "Cold_Menthol",
  "vonFrey",
  "Capsaicin_Effect_vonFrey"
)

use_curated <- c(FALSE, TRUE)

for (use_curated in use_curated) {
  ifelse (use_curated, add_file_string <- "_use_curated", add_file_string <- "")
  
  ###############################################################################
  # --- Utility Functions ---
  ###############################################################################
  
  # Load pain thresholds dataset
  load_pain_thresholds_data <- function(file_path) {
    read.csv(file_path, row.names = 1)
  }
  
  # Load target variable
  load_target_data <- function(file_path) {
    read.csv(file_path, row.names = 1)$Target
  }
  
  # Rename dataset columns
  rename_pain_data_columns <- function(data, new_names) {
    names(data) <- new_names
    data
  }
  
  # Identify highly correlated variables using caret::findCorrelation
  find_variables_to_drop_caret <- function(data, method = CORRELATION_METHOD, cutoff = CORRELATION_LIMIT) {
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
  
  nyt_theme <- function() {
    theme_minimal(base_family = "Helvetica") +
      theme(
        text = element_text(color = "black"),
        plot.title = element_text(size = 12, face = "plain", hjust = 0), # left aligned, no bold
        plot.subtitle = element_text(size = 11, face = "plain", hjust = 0),
        plot.caption = element_text(size = 10, color = "gray40", hjust = 0),
        axis.title = element_text(size = 11, face = "plain"),
        # NB: don't touch angle/rotation here
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
  # --- Train/Validation Split ---
  ###############################################################################
  
  # Load data
  pain_data <- load_pain_thresholds_data(PAIN_DATA_FILE_PATH)
  pain_data <- rename_pain_data_columns(pain_data, PAIN_DATA_COLUMN_NAMES)
  target_data <- load_target_data(TARGET_FILE_PATH)
  
  # # Duplicate pressure variable with tiny noise addition
  set.seed(SEED)
  pain_data$Pressure2 <- pain_data$Pressure +
    runif(length(pain_data$Pressure), min = -abs(pain_data$Pressure) *
            noise_factor, max = abs(pain_data$Pressure) * noise_factor)
  
  # Use only curated variables
  if (use_curated) pain_data <- pain_data[,PAIN_DATA_CURATED_COLUMN_NAMES]
  
  # Split into training/validation sets (using opdisDownsampling)
  data_split <- opdisDownsampling::opdisDownsampling(
    pain_data, Cls = target_data, Size = 0.8, Seed = 42, nTrials = 2000000, MaxCores = parallel::detectCores() - 1
  )
  
  training_data_original <- data_split$ReducedData[, 1:(ncol(data_split$ReducedData) - 1)]
  training_target <- data_split$ReducedData$Cls
  
  validation_data_original <- data_split$RemovedData[, 1:(ncol(data_split$RemovedData) - 1)]
  validation_target <- data_split$RemovedData$Cls
  
  
  run_classifier_multiple_times <- function(train_data, train_target, test_data, test_target,
                                            classifier_type = "RF", n_runs = 100) {
    
    ba_results <- numeric(n_runs)
    auc_results <- numeric(n_runs)
    
    # Quick tune RF
    if (classifier_type == "RF" && tune_RF) {
      # Quick grid for both mtry and ntree
      mtry_values <- c(1,2) #unique(round(c(2, sqrt(ncol(train_df)), ncol(train_df) / 2)))
      ntree_values <- c(500, 1000)
      
      results <- expand.grid(mtry = mtry_values, ntree = ntree_values)
      results$error <- NA
      
      for (i in 1:nrow(results)) {
        model <- randomForest(
          x = train_df,
          y = as.factor(train_target),
          mtry = results$mtry[i],
          ntree = results$ntree[i]
        )
        results$error[i] <- mean(model$err.rate[, 1]) # OOB error for classification
      }
      
      # Find the best parameter set
      best <- results[which.min(results$error),]
    }
    
    for (i in 1:n_runs) {
      tryCatch({
        set.seed(i) # Different seed for each run
        
        train_df <- as.data.frame(train_data)
        test_df <- as.data.frame(test_data)
        
        # Check for invalid values in features
        if (any(!is.finite(as.matrix(train_df)))) stop("Training data contains NA/NaN/Inf")
        if (any(!is.finite(as.matrix(test_df)))) stop("Test data contains NA/NaN/Inf")
        
        if (classifier_type == "RF") {
          # Random Forest
          train_target_factor <- as.factor(train_target)
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
          
          test_target_factor <- as.factor(test_target)
          levels(test_target_factor) <- levels(train_target_factor)
          
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
          train_target_factor <- as.factor(train_target)
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
          
          test_target_factor <- as.factor(test_target)
          levels(test_target_factor) <- target_levels
          
          roc_obj <- pROC::roc(as.numeric(test_target_factor), prob_vec, quiet = TRUE)
          auc_results[i] <- as.numeric(roc_obj$auc)
          
          cm <- caret::confusionMatrix(pred, test_target_factor)
          ba_results[i] <- cm$byClass["Balanced Accuracy"]
          
        } else if (classifier_type == "KNN") {
          # K Nearest Neighbors (caret)
          knn_train_data <- train_df
          knn_train_data$target <- as.factor(train_target)
          
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
          
          test_target_factor <- as.factor(test_target)
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
  
  
  
  # Use the original training/validation data
  cat("=== Quick Classification Test with 100 Runs CI ===\n")
  cat("Training data:", nrow(training_data_original), "rows,", ncol(training_data_original), "features\n")
  cat("Validation data:", nrow(validation_data_original), "rows,", ncol(validation_data_original), "features\n")
  
  # Run Boruta on original training data
  cat("\nRunning Boruta feature selection...\n")
  set.seed(SEED)
  boruta_original <- Boruta(x = training_data_original, y = as.factor(training_target), maxRuns = 100)
  
  # Add Boruta visualization
  cat("\nCreating Boruta visualization plot...\n")
  
  # Prepare Boruta results for plotting
  prepare_boruta_plot_data <- function(boruta_res) {
    # Melt importance history
    imp_long <- reshape2::melt(boruta_res$ImpHistory)
    colnames(imp_long) <- c("Iteration", "Feature", "Importance")
    
    # Extract final decisions
    decisions <- data.frame(Decision = boruta_res$finalDecision)
    decisions$Feature <- rownames(decisions)
    
    # Assign color categories
    imp_long$Color <- decisions$Decision[match(imp_long$Feature, decisions$Feature)]
    imp_long$Color <- factor(imp_long$Color, levels = c(levels(decisions$Decision), "Shadow"))
    imp_long$Color[is.na(imp_long$Color)] <- "Shadow"
    
    list(importance = imp_long, decisions = decisions)
  }
  
  # Plot Boruta results
  plot_boruta <- function(plot_data, title) {
    ggplot(
      boruta_plot_data$importance,
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
  
  # Create and save Boruta plot
  boruta_plot_data <- prepare_boruta_plot_data(boruta_original)
  boruta_plot <- plot_boruta(boruta_plot_data, "Boruta Feature Importance - Training Data")
  if (use_nyt) boruta_plot <- boruta_plot + nyt_theme() + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))
  
  # Get Boruta selected and rejected features
  boruta_decisions <- boruta_original$finalDecision
  # Boruta_tentative_in is TRUE if Tentative should be treated as Selected
  if (Boruta_tentative_in) {
    boruta_selected <- names(boruta_decisions[boruta_decisions %in% c("Confirmed", "Tentative")])
    boruta_rejected <- names(boruta_decisions[boruta_decisions == "Rejected"])
  } else {
    boruta_selected <- names(boruta_decisions[boruta_decisions == "Confirmed"])
    boruta_rejected <- names(boruta_decisions[boruta_decisions %in% c("Rejected", "Tentative")])
  }
  
  # Handle case where no features are selected by Boruta
  if (length(boruta_selected) == 0) {
    cat("Warning: Boruta selected NO features! Using all features as backup.\n")
    boruta_selected <- names(training_data_original)
    boruta_rejected <- character(0)
  }
  
  cat("Boruta selected features:", length(boruta_selected), "\n")
  cat("Selected:", paste(boruta_selected, collapse = ", "), "\n")
  cat("Boruta rejected features:", length(boruta_rejected), "\n")
  cat("Rejected:", paste(boruta_rejected, collapse = ", "), "\n")
  
  # Run LASSO on original training data
  cat("\nRunning LASSO feature selection...\n")
  set.seed(SEED)
  lasso_original <- tryCatch({
    x_matrix <- as.matrix(training_data_original)
    y_factor <- as.factor(training_target)
    
    # Run cross-validated LASSO
    cv_lasso <- cv.glmnet(x_matrix, y_factor, family = "binomial", alpha = 1, nfolds = 5)
    
    # Get coefficients at lambda.min
    lasso_coef <- coef(cv_lasso, s = "lambda.min")
    selected_vars <- rownames(lasso_coef)[abs(lasso_coef[, 1]) > 0][-1] # Remove intercept
    
    list(selected = selected_vars, model = cv_lasso)
  }, error = function(e) {
    cat("Error in LASSO:", e$message, "\n")
    NULL
  })
  
  # Prepare LASSO results for plotting
  prepare_lasso_plot_data <- function(lasso_res, feature_names) {
    if (is.null(lasso_res)) {
      return(NULL)
    }
    
    # Get coefficients at lambda.min
    lasso_coef <- coef(lasso_res$model, s = "lambda.min")
    coef_df <- data.frame(
      Feature = rownames(lasso_coef)[-1], # Remove intercept
      Coefficient = as.numeric(lasso_coef[-1, 1]),
      stringsAsFactors = FALSE
    )
    
    # Add all features that weren't selected (coefficient = 0)
    missing_features <- setdiff(feature_names, coef_df$Feature)
    if (length(missing_features) > 0) {
      missing_df <- data.frame(
        Feature = missing_features,
        Coefficient = 0,
        stringsAsFactors = FALSE
      )
      coef_df <- rbind(coef_df, missing_df)
    }
    
    # Create decision categories
    coef_df$Decision <- ifelse(abs(coef_df$Coefficient) > 0, "Selected", "Rejected")
    coef_df$AbsCoefficient <- abs(coef_df$Coefficient)
    coef_df$AbsCoefficient[coef_df$AbsCoefficient == 0] <- 0.001
    
    return(coef_df)
  }
  
  # Plot LASSO results
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
  
  # Create and save LASSO plot
  lasso_plot_data <- prepare_lasso_plot_data(lasso_original, names(training_data_original))
  lasso_plot <- plot_lasso(lasso_plot_data, "LASSO Feature Selection - Training Data")
  if (use_nyt) lasso_plot <- lasso_plot + nyt_theme() + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))
  
  
  # Get LASSO selected and rejected features
  if (!is.null(lasso_original)) {
    lasso_selected <- lasso_original$selected
    lasso_rejected <- setdiff(names(training_data_original), lasso_selected)
    
    # Handle case where no features are selected by LASSO
    if (length(lasso_selected) == 0) {
      cat("Warning: LASSO selected NO features! Using all features as backup.\n")
      lasso_selected <- names(training_data_original)
      lasso_rejected <- character(0)
    }
  } else {
    cat("LASSO failed - using all features as backup.\n")
    lasso_selected <- names(training_data_original)
    lasso_rejected <- character(0)
  }
  
  cat("LASSO selected features:", length(lasso_selected), "\n")
  cat("Selected:", paste(lasso_selected, collapse = ", "), "\n")
  cat("LASSO rejected features:", length(lasso_rejected), "\n")
  cat("Rejected:", paste(lasso_rejected, collapse = ", "), "\n")
  
  # Create feature selection matrix plot
  cat("\nCreating feature selection matrix plot...\n")
  
  # Get all features
  all_features <- names(training_data_original)
  
  # Create binary selection matrix
  selection_matrix <- data.frame(
    Feature = all_features,
    Boruta_Selected = ifelse(all_features %in% boruta_plot_data$decisions$Feature[boruta_plot_data$decisions$Decision != "Rejected"], 1, 0),
    LASSO_Selected = ifelse(all_features %in% lasso_plot_data$Feature[lasso_plot_data$Decision != "Rejected"], 1, 0),
    stringsAsFactors = FALSE
  )
  
  # Add selection categories for visualization
  selection_matrix$Selection_Category <- with(selection_matrix, {
    ifelse(Boruta_Selected == 1 & LASSO_Selected == 1, "Both",
           ifelse(Boruta_Selected == 1 & LASSO_Selected == 0, "Boruta only",
                  ifelse(Boruta_Selected == 0 & LASSO_Selected == 1, "LASSO only", "Neither")))
  })
  
  # Convert to long format for plotting
  library(tidyr)
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
      title = "Feature selection matrix",
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
      values = c("Both" = "chartreuse4", "Boruta Only" = "#56B4E9",
                 "LASSO Only" = "#E69F00", "Neither" = "salmon")
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
  
  # Combine plots
  library(patchwork)
  
  right_column <- matrix_plot / summary_plot +
    plot_layout(heights = c(2, 1)) # top plot twice the height of bottom one
  
  combined_selection_plot <- boruta_plot + lasso_plot +
    right_column +
    plot_layout(widths = c(2, 2, 1)) + # left-right widths
    plot_annotation(tag_levels = 'A')
  
  # Save the combined plot
  ggsave(
    plot = combined_selection_plot,
    filename = paste0("feature_selection_comparison", add_file_string, ".png"),
    width = 14,
    height = 8,
    dpi = 300
  )
  
  ggsave(
    plot = combined_selection_plot,
    filename = paste0("feature_selection_comparison", add_file_string, ".svg"),
    width = 14,
    height = 8
  )
  
  # Display the combined plot
  print(combined_selection_plot)
  
  # Print selection matrix
  cat("\n=== FEATURE SELECTION MATRIX ===\n")
  print(selection_matrix)
  
  # Print selection summary
  cat("\n=== SELECTION SUMMARY ===\n")
  cat(sprintf("Features selected by both methods: %d\n", sum(selection_matrix$Selection_Category == "Both")))
  cat(sprintf("Features selected by Boruta only: %d\n", sum(selection_matrix$Selection_Category == "Boruta Only")))
  cat(sprintf("Features selected by LASSO only: %d\n", sum(selection_matrix$Selection_Category == "LASSO Only")))
  cat(sprintf("Features selected by neither: %d\n", sum(selection_matrix$Selection_Category == "Neither")))
  cat(sprintf("Total agreement rate: %.1f%%\n",
              100 * sum(selection_matrix$Selection_Category %in% c("Both", "Neither")) / nrow(selection_matrix)))
  
  # Save selection matrix to CSV
  write.csv(selection_matrix, paste0("feature_selection_matrix", add_file_string, ".csv"), row.names = FALSE)
  
  cat("\nPlots and data saved as:\n")
  cat("- feature_selection_comparison.png\n")
  cat("- feature_selection_comparison.svg\n")
  cat("- feature_selection_matrix.csv\n")
  
  # Create five datasets for testing
  datasets_test <- list(
    "All_Features" = list(
      train = training_data_original,
      test = validation_data_original
    ),
    "Boruta_Selected" = list(
      train = if (length(boruta_selected) > 0) training_data_original[, boruta_selected, drop = FALSE] else data.frame(),
      test = if (length(boruta_selected) > 0) validation_data_original[, boruta_selected, drop = FALSE] else data.frame()
    ),
    "Boruta_Rejected" = list(
      train = if (length(boruta_rejected) > 0) training_data_original[, boruta_rejected, drop = FALSE] else data.frame(),
      test = if (length(boruta_rejected) > 0) validation_data_original[, boruta_rejected, drop = FALSE] else data.frame()
    ),
    "LASSO_Selected" = list(
      train = if (length(lasso_selected) > 0) training_data_original[, lasso_selected, drop = FALSE] else data.frame(),
      test = if (length(lasso_selected) > 0) validation_data_original[, lasso_selected, drop = FALSE] else data.frame()
    ),
    "LASSO_Rejected" = list(
      train = if (length(lasso_rejected) > 0) training_data_original[, lasso_rejected, drop = FALSE] else data.frame(),
      test = if (length(lasso_rejected) > 0) validation_data_original[, lasso_rejected, drop = FALSE] else data.frame()
    ),
    "Boruta_LASSO_Selected" = list(
      train = if (length(boruta_selected) > 0 || length(lasso_selected) > 0) 
        training_data_original[, union(boruta_selected, lasso_selected), drop = FALSE] else data.frame(),
      test = if (length(boruta_selected) > 0 || length(lasso_selected) > 0) 
        validation_data_original[, union(boruta_selected, lasso_selected), drop = FALSE] else data.frame()
    ),
    "Boruta_LASSO_Rejected" = list(
      train = if (length(boruta_rejected) > 0 || length(lasso_rejected) > 0)  
        training_data_original[, setdiff(names(training_data_original), union(boruta_selected, lasso_selected)), drop = FALSE] else data.frame(),
      test = if (length(boruta_rejected) > 0 || length(lasso_rejected) > 0)
        validation_data_original[, setdiff(names(training_data_original), union(boruta_selected, lasso_selected)), drop = FALSE] else data.frame()
    ),
    "Only_high_correlated" = list(
      train = data.frame(Pressure2 = training_data_original[find_variables_to_drop_caret(data = training_data_original, method = CORRELATION_METHOD, cutoff = CORRELATION_LIMIT)$vars_to_drop]),
      test = data.frame(Pressure2 = validation_data_original[find_variables_to_drop_caret(data = training_data_original, method = CORRELATION_METHOD, cutoff = CORRELATION_LIMIT)$vars_to_drop])
    )
  )
  
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
  
  # Run classification on all three datasets
  cat("\n", paste(rep("=", 60), collapse = ""), "\n")
  test_results <- list()
  for (name in names(datasets_test)) {
    if (nrow(datasets_test[[name]]$train) > 0 && ncol(datasets_test[[name]]$train) > 0) {
      test_results[[name]] <- quick_classify_100_runs(
        datasets_test[[name]]$train,
        training_target,
        datasets_test[[name]]$test,
        validation_target,
        name
      )
    } else {
      cat(sprintf("\nSkipping %s - no data available\n", name))
    }
  }
  
  # Create results table with confidence intervals
  cat("\n=== RESULTS SUMMARY (100 runs, 95% CI) ===\n")
  results_df <- data.frame(
    Dataset = character(),
    Features = numeric(),
    RF_BA = character(),
    RF_AUC = character(),
    LR_BA = character(),
    LR_AUC = character(),
    KNN_BA = character(),
    KNN_AUC = character(),
    stringsAsFactors = FALSE
  )
  
  for (name in names(test_results)) {
    if (!is.null(test_results[[name]])) {
      
      # ---- RF ----
      rf_res <- test_results[[name]]$RF
      if (!is.na(rf_res$ba_mean)) {
        rf_ba_str <- sprintf("%.3f [%.3f, %.3f]", rf_res$ba_mean, rf_res$ba_ci_lower, rf_res$ba_ci_upper)
        rf_auc_str <- sprintf("%.3f [%.3f, %.3f]", rf_res$auc_mean, rf_res$auc_ci_lower, rf_res$auc_ci_upper)
      } else {
        rf_ba_str <- "NA"
        rf_auc_str <- "NA"
      }
      
      # ---- LR ----
      lr_res <- test_results[[name]]$LR
      if (!is.na(lr_res$ba_mean)) {
        lr_ba_str <- sprintf("%.3f [%.3f, %.3f]", lr_res$ba_mean, lr_res$ba_ci_lower, lr_res$ba_ci_upper)
        lr_auc_str <- sprintf("%.3f [%.3f, %.3f]", lr_res$auc_mean, lr_res$auc_ci_lower, lr_res$auc_ci_upper)
      } else {
        lr_ba_str <- "NA"
        lr_auc_str <- "NA"
      }
      
      # ---- KNN ----
      knn_res <- test_results[[name]]$KNN
      if (!is.na(knn_res$ba_mean)) {
        knn_ba_str <- sprintf("%.3f [%.3f, %.3f]", knn_res$ba_mean, knn_res$ba_ci_lower, knn_res$ba_ci_upper)
        knn_auc_str <- sprintf("%.3f [%.3f, %.3f]", knn_res$auc_mean, knn_res$auc_ci_lower, knn_res$auc_ci_upper)
      } else {
        knn_ba_str <- "NA"
        knn_auc_str <- "NA"
      }
      
      # Append to results table
      results_df <- rbind(results_df, data.frame(
        Dataset = name,
        Features = ncol(datasets_test[[name]]$train),
        RF_BA = rf_ba_str,
        RF_AUC = rf_auc_str,
        LR_BA = lr_ba_str,
        LR_AUC = lr_auc_str,
        KNN_BA = knn_ba_str,
        KNN_AUC = knn_auc_str
      ))
    }
  }
  
  print(results_df)
  write.csv(results_df, paste0("ML_results_df", add_file_string, ".csv"))
  
  # Statistical comparison
  cat("\n=== STATISTICAL COMPARISON ===\n")
  for (name in names(test_results)) {
    if (!is.null(test_results[[name]])) {
      rf_res <- test_results[[name]]$RF
      lr_res <- test_results[[name]]$LR
      knn_res <- test_results[[name]]$KNN
      
      cat(sprintf("\n%s:\n", name))
      
      # ---- BA comparisons ----
      if (!is.na(rf_res$ba_mean) && !is.na(lr_res$ba_mean)) {
        cat(" RF vs LR (BA):\n")
        ba_overlap <- max(rf_res$ba_ci_lower, lr_res$ba_ci_lower) <= min(rf_res$ba_ci_upper, lr_res$ba_ci_upper)
        cat(sprintf("   RF BA: %.3f [%.3f, %.3f]\n", rf_res$ba_mean, rf_res$ba_ci_lower, rf_res$ba_ci_upper))
        cat(sprintf("   LR BA: %.3f [%.3f, %.3f]\n", lr_res$ba_mean, lr_res$ba_ci_lower, lr_res$ba_ci_upper))
        if (!ba_overlap) {
          if (rf_res$ba_mean > lr_res$ba_mean) cat("   --> RF significantly better\n") else cat("   --> LR significantly better\n")
        } else cat("   --> No significant difference\n")
      }
      
      if (!is.na(rf_res$ba_mean) && !is.na(knn_res$ba_mean)) {
        cat(" RF vs KNN (BA):\n")
        ba_overlap <- max(rf_res$ba_ci_lower, knn_res$ba_ci_lower) <= min(rf_res$ba_ci_upper, knn_res$ba_ci_upper)
        cat(sprintf("   RF BA: %.3f [%.3f, %.3f]\n", rf_res$ba_mean, rf_res$ba_ci_lower, rf_res$ba_ci_upper))
        cat(sprintf("   KNN BA: %.3f [%.3f, %.3f]\n", knn_res$ba_mean, knn_res$ba_ci_lower, knn_res$ba_ci_upper))
        if (!ba_overlap) {
          if (rf_res$ba_mean > knn_res$ba_mean) cat("   --> RF significantly better\n") else cat("   --> KNN significantly better\n")
        } else cat("   --> No significant difference\n")
      }
      
      if (!is.na(lr_res$ba_mean) && !is.na(knn_res$ba_mean)) {
        cat(" LR vs KNN (BA):\n")
        ba_overlap <- max(lr_res$ba_ci_lower, knn_res$ba_ci_lower) <= min(lr_res$ba_ci_upper, knn_res$ba_ci_upper)
        cat(sprintf("   LR BA: %.3f [%.3f, %.3f]\n", lr_res$ba_mean, lr_res$ba_ci_lower, lr_res$ba_ci_upper))
        cat(sprintf("   KNN BA: %.3f [%.3f, %.3f]\n", knn_res$ba_mean, knn_res$ba_ci_lower, knn_res$ba_ci_upper))
        if (!ba_overlap) {
          if (lr_res$ba_mean > knn_res$ba_mean) cat("   --> LR significantly better\n") else cat("   --> KNN significantly better\n")
        } else cat("   --> No significant difference\n")
      }
    }
  }
  
  cat("\n=== BORUTA SUMMARY ===\n")
  print(table(boruta_original$finalDecision))
  
  cat("\nTest completed with 100-run confidence intervals!\n")
  
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
  
  # Run logistic regression on all three datasets
  cat("\n", paste(rep("=", 80), collapse = ""), "\n")
  cat("SIMPLE LOGISTIC REGRESSION ANALYSIS")
  cat("\n", paste(rep("=", 80), collapse = ""), "\n")
  
  # Run on each dataset
  for (name in names(datasets_test)) {
    if (nrow(datasets_test[[name]]$train) > 0 && ncol(datasets_test[[name]]$train) > 0) {
      run_single_logistic_regression(
        datasets_test[[name]]$train,
        training_target,
        name
      )
    } else {
      cat(sprintf("\nSkipping %s - no data available\n", name))
    }
  }
  
  # Run once on the original data set with and without added variable
  for (i in 1:2) {
    if (i == 2)  sink(paste0("lr_orig_output", add_file_string, ".txt"))
    r1 <- run_single_logistic_regression(train_data = pain_data,
                                         train_target = target_data, dataset_name = "Original unsplit modified")
    run_single_logistic_regression(train_data = pain_data[, !names(pain_data) %in% c("Pressure2")],
                                   train_target = target_data, dataset_name = "Original unsplit unmodified")
    run_single_logistic_regression(train_data = training_data_original,
                                   train_target = train_target, dataset_name = "Training split modified")
    run_single_logistic_regression(train_data = training_data_original[, !names(training_data_original) %in% c("Pressure2")],
                                   train_target = train_target, dataset_name = "Training split unmodified")
    run_single_logistic_regression(train_data = pain_data[, !names(pain_data) %in% c("Pressure2", names(which(is.na(r1$coefficients))))],
                                   train_target = target_data, dataset_name = "Original unsplit modified VIF removed")
    run_single_logistic_regression(train_data = pain_data[, names(pain_data) %in% c(PAIN_DATA_COLINEAR)],
                                   train_target = target_data, dataset_name = "Original unsplit only modifed variables")
    
    
    if (i == 2) sink()
  }
  
  df_orig_4_lr <- cbind.data.frame(pain_data, Target = target_data)
  df_orig_4_lr <- df_orig_4_lr[,!names(df_orig_4_lr) %in% c("Capsaicin_Effect_Heat", "Menthol_Effect_Cold", "Capsaicin_Effect_vonFrey")]
  orig_lr_res <- glm(Target ~., data = df_orig_4_lr, family = binomial )
  alias(orig_lr_res)
  
  car::vif(orig_lr_res)
  
  cat("\n", paste(rep("=", 80), collapse = ""), "\n")
  cat("Logistic regression analysis completed!")
  cat("\n", paste(rep("=", 80), collapse = ""), "\n")
  
  
  
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
  
  # Calculate Cohen's d and t-tests for both datasets
  cohens_d_original <- calculate_cohens_d_with_ttest(pain_data, target_data, "Original dataset")
  cohens_d_training <- calculate_cohens_d_with_ttest(training_data_original, training_target, "Training dataset")
  
  # Combine results
  all_cohens_d <- rbind(cohens_d_original, cohens_d_training)
  
  # Add significance labels and text positioning
  all_cohens_d <- all_cohens_d %>%
    mutate(
      p_label = case_when(
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
  
  # Get overall sorting order based on absolute Cohen's d values (using original dataset)
  variable_order <- cohens_d_original %>%
    arrange(desc(abs(Cohens_d))) %>%
    pull(Variable)
  
  # Apply the ordering to combined data
  all_cohens_d$Variable <- factor(all_cohens_d$Variable, levels = rev(variable_order))
  
  # Create the plot with t-test results
  cohens_d_plot <- ggplot(all_cohens_d, aes(x = reorder(Variable, abs(Cohens_d)),
                                            y = Cohens_d, fill = Dataset)) +
    geom_col(position = position_dodge(width = 0.7), width = 0.6, alpha = 0.9) +
    geom_errorbar(
      aes(ymin = CI_lower, ymax = CI_upper),
      position = position_dodge(width = 0.7),
      width = 0.2,
      color = "grey50"
    ) +
    geom_text(
      aes(y = text_y_position, label = t_label, hjust = text_hjust),
      position = position_dodge(width = 0.7),
      size = 3.2,
      family = "serif",
      color = "black"
    ) +
    geom_hline(yintercept = 0, linetype = "solid", color = "black", size = 0.4) +
    geom_hline(yintercept = c(-0.2, 0.2), linetype = "dashed", color = "grey70") +
    geom_hline(yintercept = c(-0.5, 0.5), linetype = "dashed", color = "grey60") +
    geom_hline(yintercept = c(-0.8, 0.8), linetype = "dashed", color = "grey50") +
    scale_fill_manual(
      values = c("Original dataset" = "#9ecae1", "Training dataset" = "#fdae6b"),
      name = NULL
    ) +
    coord_flip() +
    labs(
      title = "Cohen's d effect sizes",
      subtitle = "Variables ordered by descending effect size\nStars indicate t-test significance",
      x = NULL,
      y = "Cohen's d",
      caption = "Dashed lines represent standard effect size thresholds (0.2, 0.5, 0.8)."
    ) +
    theme_minimal(base_family = "sans")
  
  if (use_nyt) cohens_d_plot <- cohens_d_plot + nyt_theme()
  
  # Print the plot
  print(cohens_d_plot)
  
  # Save the plots
  ggsave(
    plot = cohens_d_plot,
    filename = paste0("cohens_d_with_ttests", add_file_string, ".svg"),
    width = 10,
    height = 7
  )
  
  
  # Print summary statistics
  cat("\n=== COHEN'S D AND T-TEST SUMMARY ===\n")
  summary_stats <- all_cohens_d %>%
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
  
  # Print detailed results
  cat("\n=== DETAILED RESULTS ===\n")
  detailed_results <- all_cohens_d %>%
    arrange(desc(abs(Cohens_d))) %>%
    mutate(
      Effect_Size_Category = case_when(
        abs(Cohens_d) < 0.2 ~ "Negligible",
        abs(Cohens_d) < 0.5 ~ "Small",
        abs(Cohens_d) < 0.8 ~ "Medium",
        TRUE ~ "Large"
      )
    ) %>%
    select(Variable, Dataset, Cohens_d, CI_lower, CI_upper, t_statistic, p_value, p_label, Effect_Size_Category)
  
  print(detailed_results)
  
  # Save results to CSV
  write.csv(detailed_results, paste0("cohens_d_with_ttests_results", add_file_string, ".csv"), row.names = FALSE)
  
  cat("\nFiles saved:\n")
  cat("- cohens_d_with_ttests.png\n")
  cat("- cohens_d_with_ttests_rotated.png\n")
  cat("- cohens_d_with_ttests_results.csv\n")
  
  print(cor(pain_data))
}