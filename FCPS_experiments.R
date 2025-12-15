###############################################################################
# FCPS Data Analysis
# 
###############################################################################

library(FCPS)
library(opdisDownsampling)
library(randomForest)
library(caret)
library(pbmcapply)
library(plotly)

###############################################################################
# Configuration Parameters 
###############################################################################

SEED <- 42 # or 12
training_and_validation_subsplits <- TRUE
TRAINING_PARTITION_SIZE <- 0.8
VALIDATION_PARTITION_SIZE <- 0.8

tune_RF <- FALSE
tune_KNN <- TRUE
tune_SVM <- TRUE


DATASET_NAME <- "Atom"
EXPERIMENTS_DIR <- "/home/joern/.Datenplatte/Joerns Dateien/Aktuell/ABCPython/08AnalyseProgramme/R/ABC2way/"
# External functions
FUNCTIONS_FILE_PATH <- "/home/joern/.Datenplatte/Joerns Dateien/Aktuell/ABCPython/08AnalyseProgramme/R/ABC2way/feature_selection_and_classification_functions.R"

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
# Prepare and downsample data once
###############################################################################

FCPS_df_original <- data.frame(Target = as.factor(FCPS::Atom$Cls), FCPS::Atom$Data)
ds_result <- opdisDownsampling::opdisDownsampling(
  Data = FCPS_df_original[, -1],
  Cls = FCPS_df_original$Target,
  Size = 0.8, Seed = SEED, nTrials = 2000000, MaxCores = parallel::detectCores() - 1
)

train <- ds_result$ReducedData
valid <- ds_result$RemovedData
train$Cls <- as.factor(train$Cls)
valid$Cls <- as.factor(valid$Cls)

###############################################################################
# Plot the data set
###############################################################################

# Create 3D scatter plot with margins and perspective -------------------
# Extract first 3 numeric features; adjust if needed
x <- FCPS_df_original[, 2]
y <- FCPS_df_original[, 3]
z <- FCPS_df_original[, 4]
classes <- FCPS_df_original$Target

# Add 10% padding to axis ranges for margins
add_padding <- function(vec, pad_ratio = 0.1) {
  r <- range(vec, na.rm = TRUE)
  pad <- diff(r) * pad_ratio
  c(r[1] - pad, r[2] + pad)
}

x_range <- add_padding(x)
y_range <- add_padding(y)
z_range <- add_padding(z)

# Standard camera perspective view
camera_view <- list(eye = list(x = 1.5, y = 1.5, z = 1.0))

# Build plotly 3D scatter plot
p <- plot_ly(
  x = ~x, y = ~y, z = ~z,
  color = ~classes,
  colors = c("#1f77b4", "#ff7f0e", "#2ca02c"), # adjust colors based on classes
  type = 'scatter3d',
  mode = 'markers',
  marker = list(size = 5)
) %>%
  layout(
    scene = list(
      xaxis = list(title = 'X', range = x_range),
      yaxis = list(title = 'Y', range = y_range),
      zaxis = list(title = 'Z', range = z_range),
      camera = camera_view
    ),
    legend = list(title = list(text = '<b>Class</b>'))
  )

# Display the plot
p <- p %>% layout(showlegend = FALSE)
p

###############################################################################
# Statistical analysis 
###############################################################################

# Fit logistic regression on complete set
model_lr <- glm(as.factor(Cls) ~ ., data = train, family = binomial)

# Print logistic regression summary
print(summary(model_lr))

# Fit logistic regression on training set
model_lr_orig <- glm(as.factor(Target) ~ ., data = FCPS_df_original, family = binomial)

# Print logistic regression summary
print(summary(model_lr_orig))

# Univariate t-tests comparing features by class
univariate_tests <- apply(within(train, rm(Cls)), 2, function(x) t.test(x ~ train$Cls))

###############################################################################
# Machine learning analysis 
###############################################################################

# Helper: flatten byClass matrix or vector into a named vector
flatten_byClass <- function(byClass, class_levels) {
  if (is.null(nrow(byClass))) {
    # two-class: prefix positive class metrics
    pos_class <- class_levels[2]
    names(byClass) <- paste0(pos_class, "_", names(byClass))
    return(byClass)
  } else {
    # multi-class: flatten matrix prefixing class labels
    flattened <- c()
    for (cls in rownames(byClass)) {
      cls_metrics <- byClass[cls,]
      names(cls_metrics) <- paste0(cls, "_", names(cls_metrics))
      flattened <- c(flattened, cls_metrics)
    }
    return(flattened)
  }
}

# Quick tune RF  - do once before loop
if (tune_RF) {
  mtry_values <- c(1, 2)
  ntree_values <- c(100, 200, 500, 1000, 1500)
  
  results <- expand.grid(mtry = mtry_values, ntree = ntree_values)
  results$error <- NA
  
  for (i in 1:nrow(results)) {
    set.seed(SEED)
    model <- randomForest(as.factor(Cls) ~ ., data = train,
                          mtry = results$mtry[i],
                          ntree = results$ntree[i]
    )
    results$error[i] <- mean(model$err.rate[, 1]) # OOB error for classification
  }
  
  # Find the best parameter set for RF
  best_rf <- results[which.min(results$error),]
}

# Quick tune SVM - do once before loop
if (tune_SVM) {svm_tune_data <- within(train, rm(Cls))
svm_tune_data$target <- as.factor(train$Cls)
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

# Find the best parameter set for SVM
best_svm <- list(C = svm_tune_model$bestTune$C, sigma = svm_tune_model$bestTune$sigma)
}

run_one_iteration <- function(train_df, valid_df, seed) {
  set.seed(seed)
  # train Logistic Regression
  lr_model <- glm(Cls ~ ., data = train_df, family = binomial)
  
  # ---- Penalized Logistic Regression (Elastic Net) ----
  # Prepare data
  set.seed(seed)
  x_train <- as.matrix(train_df[, setdiff(names(train_df), "Cls")])
  y_train_fac <- as.factor(train_df$Cls)
  n_classes <- length(levels(y_train_fac))
  
  if (n_classes < 2) stop("Less than 2 classes in training data")
  
  if (n_classes == 2) {
    x_valid <- as.matrix(valid_df[, setdiff(names(valid_df), "Cls")])
    y_valid_fac <- as.factor(valid_df$Cls)
    
    # Convert to numeric (0/1)
    y_train_num <- as.numeric(y_train_fac) - 1
    y_valid_num <- as.numeric(y_valid_fac) - 1
    
    # Fit elastic-net penalized logistic regression
    plr_model <- glmnet::glmnet(
      x = x_train,
      y = y_train_num,
      family = "binomial",
      alpha = 0.5 # Elastic net mixing parameter
    )
    
    # Select a lambda (you could use cv.glmnet for tuning)
    lambda_use <- plr_model$lambda[length(plr_model$lambda)]
    
    # Predicted probabilities (for class "1")
    plr_prob <- as.numeric(
      predict(plr_model, newx = x_valid, s = lambda_use, type = "response")
    )
    
    class_levels <- levels(y_train_fac)
    plr_pred <- factor(
      ifelse(plr_prob > 0.5, class_levels[2], class_levels[1]),
      levels = class_levels
    )
    
    # confusion matrix
    cm_plr <- caret::confusionMatrix(
      factor(valid_df$Cls, levels = class_levels), 
      plr_pred, 
      mode = "everything"
    )
    
    byClass_plr <- flatten_byClass(cm_plr$byClass, class_levels)
    overall_plr <- cm_plr$overall[c("Accuracy", "Kappa")]
    plr_stats <- c(overall_plr, byClass_plr)
  } else {
    warning("pLR currently supports only binary classification.")
    plr_stats <- NA
  }
  
  # ---- Random Forest ----
  set.seed(seed)
  if (tune_RF && exists("best_rf")) {
    rf_model <- randomForest::randomForest(
      Cls ~ ., data = train_df,
      mtry = best_rf$mtry,
      ntree = best_rf$ntree
    )
  } else {
    rf_model <- randomForest::randomForest(Cls ~ ., data = train_df)
  }
  
  # ---- KNN ----
  set.seed(seed)
  train_knn <- train_df
  train_knn$Cls <- as.factor(train_knn$Cls)
  levels(train_knn$Cls) <- c("Class0", "Class1")
  
  ctrl <- caret::trainControl(
    method = "cv", number = 5,
    classProbs = TRUE, summaryFunction = twoClassSummary
  )
  
  set.seed(seed)
  knn_model <- caret::train(
    Cls ~ ., data = train_knn,
    method = "knn",
    trControl = ctrl,
    metric = "ROC",
    preProcess = c("center", "scale"),
    tuneLength = 5
  )
  
  # ---- C5.0 ----
  set.seed(seed)
  c50_model <- C50::C5.0(Cls ~ ., data = train_df)
  
  # ---- SVM ----
  train_svm <- train_df
  train_svm$Cls <- as.factor(train_svm$Cls)
  original_levels <- levels(train_svm$Cls)
  n_classes_svm <- length(original_levels)
  new_levels <- paste0("Class", seq_len(n_classes_svm))
  levels(train_svm$Cls) <- new_levels
  
  ctrl <- caret::trainControl(
    method = "none",
    classProbs = TRUE,
    allowParallel = FALSE
  )
  
  C_value <- if (tune_SVM && exists("best_svm")) best_svm$C else 1
  sigma_value <- if (tune_SVM && exists("best_svm")) best_svm$sigma else 0.1
  
  set.seed(seed)
  svm_model <- suppressWarnings(
    caret::train(
      Cls ~ ., data = train_svm,
      method = "svmRadial",
      trControl = ctrl,
      preProcess = c("center", "scale"),
      tuneGrid = data.frame(C = C_value, sigma = sigma_value)
    )
  )
  
  # ---- Predictions ----
  lr_prob <- predict(lr_model, valid_df, type = "response")
  class_levels <- levels(FCPS_df_original$Target)
  
  if (length(class_levels) == 2) {
    lr_pred <- factor(ifelse(lr_prob > 0.5, class_levels[2], class_levels[1]), levels = class_levels)
  } else {
    lr_pred <- factor(class_levels[1], levels = class_levels)
  }
  
  rf_pred <- predict(rf_model, valid_df)
  
  valid_knn <- valid_df
  valid_knn$Cls <- as.factor(valid_knn$Cls)
  levels(valid_knn$Cls) <- c("Class0", "Class1")
  knn_pred <- predict(knn_model, valid_knn)
  
  c50_pred <- predict(c50_model, valid_df)
  
  valid_svm <- valid_df
  valid_svm$Cls <- as.factor(valid_svm$Cls)
  levels(valid_svm$Cls) <- c("Class1", "Class2")
  svm_pred <- predict(svm_model, valid_df)
  
  # ---- Confusion Matrices ----
  cm_rf <- caret::confusionMatrix(factor(valid_df$Cls, levels = class_levels), rf_pred, mode = "everything")
  cm_lr <- caret::confusionMatrix(factor(valid_df$Cls, levels = class_levels), lr_pred, mode = "everything")
  cm_knn <- caret::confusionMatrix(valid_knn$Cls, knn_pred, mode = "everything")
  cm_c50 <- caret::confusionMatrix(valid_df$Cls, c50_pred, mode = "everything")
  cm_svm <- caret::confusionMatrix(valid_svm$Cls, svm_pred, mode = "everything")
  
  # ---- Flatten stats ----
  byClass_rf <- flatten_byClass(cm_rf$byClass, class_levels)
  byClass_lr <- flatten_byClass(cm_lr$byClass, class_levels)
  byClass_knn <- flatten_byClass(cm_knn$byClass, levels(valid_knn$Cls))
  byClass_c50 <- flatten_byClass(cm_c50$byClass, class_levels)
  byClass_svm <- flatten_byClass(cm_svm$byClass, levels(valid_svm$Cls))
  
  overall_rf <- cm_rf$overall[c("Accuracy", "Kappa")]
  overall_lr <- cm_lr$overall[c("Accuracy", "Kappa")]
  overall_knn <- cm_knn$overall[c("Accuracy", "Kappa")]
  overall_c50 <- cm_c50$overall[c("Accuracy", "Kappa")]
  overall_svm <- cm_svm$overall[c("Accuracy", "Kappa")]
  
  rf_stats <- c(overall_rf, byClass_rf)
  lr_stats <- c(overall_lr, byClass_lr)
  knn_stats <- c(overall_knn, byClass_knn)
  c50_stats <- c(overall_c50, byClass_knn)
  svm_stats <- c(overall_svm, byClass_svm)
  
  # ---- Return results ----
  list(
    Logistic = lr_stats,
    PenalizedLogistic = plr_stats,
    RandomForest = rf_stats,
    KNN = knn_stats,
    C50 = c50_stats,
    SVM = svm_stats
  )
}

# Main execution
# Run 100 iterations in parallel -----------------------------------------
n_runs <- 100
set.seed(SEED)
seeds <- SEED:(SEED + n_runs)

results_list <- pbmcapply::pbmclapply(seeds, function(seed) {
  set.seed(seed)
  if (training_and_validation_subsplits) {
    inTraining <- caret::createDataPartition(
      train$Cls,
      p = TRAINING_PARTITION_SIZE,
      list = FALSE
    )
    train_df <- train[inTraining,, drop = FALSE]
    inValidation <- caret::createDataPartition(
      valid$Cls,
      p = VALIDATION_PARTITION_SIZE,
      list = FALSE
    )
    valid_df <- valid[inValidation,, drop = FALSE]
  } else {
    train_df <- train
    valid_df <- valid
  }
  
  run_one_iteration(train_df, valid_df, seed)
  
}, mc.cores = parallel::detectCores() - 1)

# Convert list results into data frames ----------------------------------
extract_df <- function(results, model_name) {
  vals <- lapply(results, `[[`, model_name)
  df <- do.call(rbind, lapply(vals, unlist))
  df <- as.data.frame(df, stringsAsFactors = FALSE)
  df[] <- lapply(df, as.numeric)
  df
}

df_rf <- extract_df(results_list, "RandomForest")
df_lr <- extract_df(results_list, "Logistic")
df_plr <- extract_df(results_list, "PenalizedLogistic")
df_knn <- extract_df(results_list, "KNN")
df_c50 <- extract_df(results_list, "C50")
df_svm <- extract_df(results_list, "SVM")

# Compute summary statistics (median, 2.5th and 97.5th percentiles) ------
summary_stats <- function(df) {
  data.frame(
    Metric = colnames(df),
    Median = apply(df, 2, median, na.rm = TRUE),
    CI_lower = apply(df, 2, quantile, 0.025, na.rm = TRUE),
    CI_upper = apply(df, 2, quantile, 0.975, na.rm = TRUE)
  )
}

summary_rf <- summary_stats(df_rf)
summary_lr <- summary_stats(df_lr)
summary_plr <- summary_stats(df_plr)
summary_knn <- summary_stats(df_knn)
summary_c50 <- summary_stats(df_c50)
summary_svm <- summary_stats(df_svm)

###############################################################################
# Show all results and write them to a text file 
###############################################################################

# View summarized statistics
for (i in 1:2) {
  if (i == 2) sink(paste0(DATASET_NAME, "_lr_and_ml_output", ".txt"))
  
  cat("\n\nRF ML summary\n")
  print(summary_rf)
  cat("\n\nLR ML summary\n")
  print(summary_lr)
  cat("\n\npLR ML summary\n")
  print(summary_plr)
  cat("\n\nKNN ML summary\n")
  print(summary_knn)
  cat("\n\nC5.0 ML summary\n")
  print(summary_c50)
  cat("\n\nSVM summary\n")
  print(summary_svm)
  cat("\n\nStandard statistical logistic regression summary\n")
  print(summary(model_lr_orig))
  
  if (i == 2) sink()
}

cat("\n", paste(rep("=", 80), collapse = ""), "\n")
cat("FCPS analysis completed!\n")
cat("\n", paste(rep("=", 80), collapse = ""), "\n")

###############################################################################
# PENALIZED LOGISTIC REGRESSION ANALYSIS 
###############################################################################

cat("\n", paste(rep("=", 80), collapse = ""), "\n")
cat("PENALIZED LOGISTIC REGRESSION ANALYSIS\n")
cat(paste(rep("=", 80), collapse = ""), "\n\n")

# Run penalized analysis and capture to file
sink(paste0(DATASET_NAME, "_penalized_lr_output.txt"))
pen_res <- run_penalized_logistic_regression_all(
  train_data = FCPS_df_original[,-1],
  train_target = as.factor(FCPS_df_original$Target),
  dataset_name = "Original unsplit - Penalized (ridge/lasso/elastic)",
  alpha_elastic = 0.5,
  nfolds = 5,
  ridge_threshold = 0.05,
  seed = 123
)
sink()

# ========== EXPORT RESULTS TO CSV FOR PAPER ==========
# Save the comparison table as CSV (perfect for paper tables)
write.csv(pen_res$coef_table,
          paste0(DATASET_NAME, "_penalized_comparison_table.csv"),
          row.names = FALSE)

cat("\n", paste(rep("=", 80), collapse = ""), "\n")
cat("ANALYSIS COMPLETED! Files created:\n")
cat(sprintf("  - %s_lr_orig_output.txt\n", DATASET_NAME))
cat(sprintf("  - %s_penalized_lr_output.txt\n", DATASET_NAME))
cat(sprintf("  - %s_penalized_comparison_table.csv\n", DATASET_NAME))
cat(sprintf("  - %s_selection_summary.csv\n", DATASET_NAME))
cat(paste(rep("=", 80), collapse = ""), "\n")


###############################################################################
# BORUTA ANALYSIS
###############################################################################

cat("\n", paste(rep("=", 80), collapse = ""), "\n")
cat("BORUTA ANALYSIS\n")
cat(paste(rep("=", 80), collapse = ""), "\n\n")

# Run Boruta analysis and log output to a text file
sink(paste0(DATASET_NAME, "_boruta_output.txt"))

set.seed(123)  # For reproducibility
boruta_res <- Boruta::Boruta(
  x = FCPS_df_original[, -1], 
  y = as.factor(FCPS_df_original$Target), 
  doTrace = 1
)

sink()  # Stop capturing output

# Get final variable importance decisions
boruta_decisions <- boruta_res$finalDecision

# Save summary to CSV
boruta_summary <- data.frame(
  Feature = names(boruta_decisions),
  Decision = as.character(boruta_decisions),
  stringsAsFactors = FALSE
)

write.csv(
  boruta_summary,
  paste0(DATASET_NAME, "_boruta_summary.csv"),
  row.names = FALSE
)

# Save feature importance plot to file (optional, for paper figures)
pdf(paste0(DATASET_NAME, "_boruta_plot.pdf"))
plot(boruta_res, las = 2, cex.axis = 0.7, main = "Boruta Feature Importance")
dev.off()

# Print top features to console
cat("\nTop Confirmed Features:\n")
print(names(boruta_res$finalDecision[boruta_res$finalDecision == "Confirmed"]))

cat("\n", paste(rep("=", 80), collapse = ""), "\n")
cat("ANALYSIS COMPLETED! Files created:\n")
cat(sprintf("  - %s_lr_orig_output.txt\n", DATASET_NAME))
cat(sprintf("  - %s_penalized_lr_output.txt\n", DATASET_NAME))
cat(sprintf("  - %s_penalized_comparison_table.csv\n", DATASET_NAME))
cat(sprintf("  - %s_selection_summary.csv\n", DATASET_NAME))
cat(sprintf("  - %s_boruta_output.txt\n", DATASET_NAME))
cat(sprintf("  - %s_boruta_summary.csv\n", DATASET_NAME))
cat(sprintf("  - %s_boruta_plot.pdf\n", DATASET_NAME))
cat(paste(rep("=", 80), collapse = ""), "\n")
