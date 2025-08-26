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

SEED <- 42 # or 12


# Prepare and downsample data once --------------------------------------
df_original <- data.frame(Target = as.factor(FCPS::Atom$Cls), FCPS::Atom$Data)
ds_result <- opdisDownsampling::opdisDownsampling(
  Data = df_original[,-1], 
  Cls = df_original$Target, 
  Size = 0.8, Seed = SEED, nTrials = 2000000, MaxCores = parallel::detectCores() - 1
)

train <- ds_result$ReducedData
valid <- ds_result$RemovedData
train$Cls <- as.factor(train$Cls)
valid$Cls <- as.factor(valid$Cls)

# Create 3D scatter plot with margins and perspective -------------------

# Extract first 3 numeric features; adjust if needed
x <- df_original[,2]
y <- df_original[,3]
z <- df_original[,4]
classes <- df_original$Target

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
  colors = c("#1f77b4", "#ff7f0e", "#2ca02c"),  # adjust colors based on classes
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
p

# Statistical analysis ---------------------------------------------------

# Fit logistic regression on complte set
model_lr <- glm(Cls ~ ., data = train, family = binomial)

# Print logistic regression summary
print(summary(model_lr))

# Fit logistic regression on training set
model_lr_orig <- glm(Target ~ ., data = df_original, family = binomial)

# Print logistic regression summary
print(summary(model_lr_orig))

# Univariate t-tests comparing features by class
univariate_tests <- apply(within(train, rm(Cls)), 2, function(x) t.test(x ~ train$Cls))

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
      cls_metrics <- byClass[cls, ]
      names(cls_metrics) <- paste0(cls, "_", names(cls_metrics))
      flattened <- c(flattened, cls_metrics)
    }
    return(flattened)
  }
}

# Quick tune RF
  mtry_values <- c(1,2) 
  ntree_values <- c(500, 1000)
  
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
  
  # Find the best parameter set
  best <- results[which.min(results$error),]


run_one_iteration <- function(seed) {
  # Train Logistic Regression
  set.seed(seed)
  lr_model <- glm(Cls ~ ., data = train, family = binomial)
  
  # Train Random Forest
  set.seed(seed)
  rf_model <- randomForest::randomForest(Cls ~ ., data = train, mtry = best$mtry, ntree = best$ntree)
  
  # Train KNN using caret (with preprocessing)
  # Ensure levels are valid factor names
  set.seed(seed)
  train_knn <- train
  train_knn$Cls <- as.factor(train_knn$Cls)
  levels(train_knn$Cls) <- c("Class0", "Class1")  # adjust if classes differ
  
  ctrl <- caret::trainControl(method = "cv", number = 5,
                              classProbs = TRUE, summaryFunction = twoClassSummary)
  
  set.seed(seed)
  knn_model <- caret::train(
    Cls ~ ., data = train_knn,
    method = "knn",
    trControl = ctrl,
    metric = "ROC",
    preProcess = c("center", "scale"),
    tuneLength = 5
  )
  
  # Predict with all models on validation set (valid)
  # For LR (binary probability)
  lr_prob <- predict(lr_model, valid, type = "response")
  class_levels <- levels(df_original$Target)
  
  if (length(class_levels) == 2) {
    lr_pred <- factor(ifelse(lr_prob > 0.5, class_levels[2], class_levels[1]), levels = class_levels)
  } else {
    lr_pred <- factor(class_levels[1], levels = class_levels)
  }
  
  rf_pred <- predict(rf_model, valid)
  
  # For KNN: rename valid$Cls factor to match KNN train levels
  valid_knn <- valid
  valid_knn$Cls <- as.factor(valid_knn$Cls)
  levels(valid_knn$Cls) <- c("Class0", "Class1")
  
  knn_pred <- predict(knn_model, valid_knn)
  
  # Confusion matrices
  cm_lr <- caret::confusionMatrix(factor(valid$Cls, levels = class_levels), lr_pred, mode = "everything")
  cm_rf <- caret::confusionMatrix(factor(valid$Cls, levels = class_levels), rf_pred, mode = "everything")
  cm_knn <- caret::confusionMatrix(valid_knn$Cls, knn_pred, mode = "everything")
  
  # Flatten 'byClass' stats helper function assumed present
  byClass_lr <- flatten_byClass(cm_lr$byClass, class_levels)
  byClass_rf <- flatten_byClass(cm_rf$byClass, class_levels)
  byClass_knn <- flatten_byClass(cm_knn$byClass, levels(valid_knn$Cls))
  
  overall_lr <- cm_lr$overall[c("Accuracy", "Kappa")]
  overall_rf <- cm_rf$overall[c("Accuracy", "Kappa")]
  overall_knn <- cm_knn$overall[c("Accuracy", "Kappa")]
  
  lr_stats <- c(overall_lr, byClass_lr)
  rf_stats <- c(overall_rf, byClass_rf)
  knn_stats <- c(overall_knn, byClass_knn)
  
  list(Logistic = lr_stats, RandomForest = rf_stats, KNN = knn_stats)
}


# Run 100 iterations in parallel -----------------------------------------
n_runs <- 100
set.seed(SEED)
seeds <- SEED:(SEED+n_runs)   

results_list <- pbmcapply::pbmclapply(seeds, run_one_iteration, mc.cores = parallel::detectCores()-1)

# Convert list results into data frames ----------------------------------
extract_df <- function(results, model_name) {
  vals <- lapply(results, `[[`, model_name)
  df <- do.call(rbind, lapply(vals, unlist))
  df <- as.data.frame(df, stringsAsFactors = FALSE)
  df[] <- lapply(df, as.numeric)
  df
}

df_lr <- extract_df(results_list, "Logistic")
df_rf <- extract_df(results_list, "RandomForest")
df_knn <- extract_df(results_list, "KNN")  

# Compute summary statistics (median, 2.5th and 97.5th percentiles) ------
summary_stats <- function(df) {
  data.frame(
    Metric = colnames(df),
    Median = apply(df, 2, median, na.rm = TRUE),
    CI_lower = apply(df, 2, quantile, 0.025, na.rm = TRUE),
    CI_upper = apply(df, 2, quantile, 0.975, na.rm = TRUE)
  )
}

summary_lr <- summary_stats(df_lr)
summary_rf <- summary_stats(df_rf)
summary_knn <- summary_stats(df_knn)   

# View summarized statistics
summary_lr
summary_rf
summary_knn    
