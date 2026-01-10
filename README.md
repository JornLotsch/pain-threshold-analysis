# Resolving interpretation challenges in machine learning feature selection with an iterative approach in biomedical pain data

## Four-Phase Iterative Feature Selection and Classification Pipeline

This project provides a comprehensive four-phase iterative machine learning framework that systematically identifies the truly minimal sufficient feature set for classification while ensuring no valuable features are overlooked through rescue mechanisms and explicit detection of feature selection method limitations.

## Overview

This repository provides a rigorous framework for analyzing classification datasets through an iterative four-phase approach combining multiple feature selection methods and machine learning algorithms. Originally developed for pain threshold phenotype prediction using multimodal sensory data, the pipeline has been generalized for use with any tabular classification dataset.

The framework addresses a critical challenge in machine learning: feature selection methods may erroneously reject individually-strong features due to redundancy masking, interaction effects, or stochastic variability. Our four-phase approach systematically minimizes the feature set while implementing rescue mechanisms to recover falsely rejected features and explicitly flagging cases where feature selection methods fail to decompose complex interactions.

## Table of Contents

- [Key Features](#key-features)
- [Algorithm Workflow](#algorithm-workflow)
  - [Overview of the Four-Phase Approach](#overview-of-the-four-phase-approach)
  - [Phase 0: Initial Analysis](#phase-0-initial-analysis)
  - [Phase 1: Minimization of the Feature Set](#phase-1-minimization-of-the-feature-set)
  - [Phase 2: Individual Rescue of Rejected Features](#phase-2-individual-rescue-of-rejected-features)
  - [Phase 3: Verification and Analysis of the Rejected Set](#phase-3-verification-and-analysis-of-the-rejected-set)
- [Classification Performance Criteria](#classification-performance-criteria)
- [Installation & Requirements](#installation--requirements)
- [Input Data Format](#input-data-format)
- [Scripts](#scripts)
- [Key Capabilities](#key-capabilities)
- [Example Results Structure](#example-results-structure)
- [Handling Edge Cases](#handling-edge-cases)
- [Dependencies](#dependencies)
- [Additional Information](#additional-information)
- [Scientific Rationale](#scientific-rationale)
- [License](#license)
- [Citation](#citation)

## Key Features

- **Four-phase iterative approach**: Initial comprehensive analysis (Phase 0), systematic minimization through backward elimination (Phase 1), individual rescue of rejected features (Phase 2), and hierarchical verification with iterative re-analysis (Phase 3)
- **Multiple feature selection methods**: Boruta algorithm (nonlinear importance via Random Forest) and LASSO regularization (linear contributions with sparsity)
- **Comprehensive classifier ensemble**: Random Forest, Logistic Regression, k-Nearest Neighbors, and C5.0 Decision Trees
- **Rigorous statistical validation**: 100-run Monte Carlo cross-validation with 95% confidence intervals
- **Rescue mechanisms**: Individual testing of rejected features and hierarchical re-analysis of rejected feature sets
- **Limitation detection**: Explicit warnings when feature selection methods fail to identify responsible features in successful sets
- **Effect size analysis**: Cohen's d calculations with statistical testing
- **Comprehensive visualization**: Feature selection matrices, importance plots, and effect size visualizations
- **Flexible configuration**: Easy adaptation to different datasets and analysis requirements

## Algorithm Workflow

![Flowchart of the four-phase feature selection framework](FlowChart.drawio.svg)

**Figure 1:** Flowchart of the four-phase feature selection framework. The pipeline systematically minimizes feature redundancy while preserving valuable features through four sequential phases: Phase 0 (initial analysis), Phase 1 (minimization), Phase 2 (individual rescue), and Phase 3 (verification with iterative rescue).

### Overview of the Four-Phase Approach

To identify the truly minimal sufficient feature set for classification, the analytical framework is organized into four phases that systematically minimize redundancy while ensuring no valuable features are overlooked. First, an initial analysis identifies candidate features and establishes baseline performance. Second, selected features are systematically removed and models are retrained to test whether classification can still be achieved. Third, features that failed to support classification are examined individually to confirm their lack of contribution. Together, these phases enable a structured evaluation of both selected and rejected features.

### Phase 0: Initial Analysis

In Phase 0, Boruta and LASSO are applied to the full training dataset. For **Boruta**, features are deemed relevant when they appear in the "confirmed" group. For **LASSO**, features are deemed relevant when their coefficient is larger than 0.

Multiple feature subset combinations are then tested with five classifiers (random forest, logistic regression, k-nearest neighbors, C5.0 decision tree, and support vector machines):
- All features
- Boruta-selected features
- LASSO-selected features  
- Boruta-rejected features
- LASSO-rejected features
- Unions and intersections of Boruta and LASSO selections

For each feature combination, classifiers are tuned by grid search for suitable hyperparameters (e.g., number of trees in random forests, number of neighbors in kNN). Classification success is defined as any classifier exhibiting a lower bound of the 95% confidence interval for balanced accuracy exceeding 0.5, based on 100 Monte-Carlo cross-validation iterations.

**Termination conditions:**
- If no feature subset (including the full feature set) achieves classification success, the algorithm terminates and the dataset is deemed not classifiable
- If the smallest successful subset is identical to the full feature set, all features are considered necessary and the procedure stops
- Otherwise, the smallest successful subset is passed to Phase 1

### Phase 1: Minimization of the Feature Set

Starting from the smallest successful feature set identified in Phase 0, Phase 1 employs backward elimination to further minimize the feature set. Features are sequentially tested for removal, with each feature whose absence still maintains classification success being permanently removed. This iterative process continues until no further features can be removed without loss of classification success, yielding a minimal feature set capable of supporting successful classification.

### Phase 2: Individual Rescue of Rejected Features

Phase 2 implements a rescue mechanism to avoid incorrectly discarding potentially relevant variables. Each feature rejected during Phases 0 and 1 is tested individually to determine whether it can support classification on its own. Features demonstrating independent classification success (lower 95% CI for balanced accuracy > 0.5) are rescued and reintegrated into the selected feature set.

This phase is essential because feature selection procedures can discard individually strong predictors due to redundancy masking, interaction effects, or stochastic variability. For example, a feature "A" may have independent classification ability but be rejected if features "B" and "C" together form a stronger combined predictor. Rescuing such features preserves scientifically valuable information and addresses the critical problem that features failing to pass feature selection can nonetheless classify successfully when used alone.

### Phase 3: Verification and Analysis of the Rejected Set

After Phases 0–2, the selected feature set should contain all features enabling successful classification while rejected features should not. Phase 3 verifies this assumption by jointly assessing both sets using the same logic as Phase 0.

First, the selected feature set (including all rescued variables) is re-evaluated by running all feature selectors and classifiers on the complete group and across all subset combinations. Next, the rejected feature set is tested identically. If no classifier on no subset achieves performance above chance, the procedure terminates with the final partition.

However, if at least one classifier on one subset achieves classification success, the full Phase 0 pipeline is rerun on the rejected subset:
- Boruta and LASSO are reapplied to the rejected features
- All resulting combinations are tested
- Features selected by either method undergo individual testing; those showing independent success (lower CI > 0.5) are rescued and added to the selected set
- Verification repeats iteratively until the rejected set no longer classifies successfully

In rare cases where the rejected set retains group-level success but neither Boruta nor LASSO identifies features and no individual feature classifies alone, a warning is issued, flagging these features as potentially informative only through complex interactions undetectable by both methods.

## Classification Performance Criteria

- **Metric**: Balanced accuracy (robust to class imbalance)
- **Validation**: 100 runs Monte Carlo cross-validation (80% training, 20% validation)
- **Success criterion**: Lower bound of 95% confidence interval for balanced accuracy > 0.5
- **Requirement**: At least one classifier must achieve success criterion

### Return Value Structure

`run_feature_selection_iterations()` returns a nested list with the following structure:

<pre>
results_list
├── $results_list
│   └── $`Full dataset`
│       ├── $final_selected_features       # Character vector: minimal + rescued features
│       ├── $final_rejected_features       # Character vector: definitively rejected features
│       ├── $minimal_feature_set          # Character vector: Phase 1 minimal set
│       ├── $rescued_features             # Character vector: Phase 2 rescued features
│       ├── $phase0_results               # Data frame: all Phase 0 combinations tested
│       ├── $datasets_to_test             # List: feature subsets used in analysis
│       ├── $plots                        # List of ggplot objects
│       │   ├── $matrix                   # Feature selection matrix heatmap
│       │   ├── $summary                  # Classification performance summary
│       │   ├── $boruta                   # Boruta importance plot
│       │   └── $lasso                    # LASSO coefficients plot
│       ├── $ML_results                   # Data frame: classification performance metrics
│       ├── $warning_flags                # List: edge case indicators
│       │   ├── $dataset_unsuitable       # Logical: no successful classification
│       │   ├── $all_features_necessary   # Logical: cannot reduce feature set
│       │   └── $method_limitation_detected # Logical: Boruta/LASSO failed to decompose
│       └── $feature_selection_history    # Data frame: tracking feature status across phases
│
└── $execution_details
    ├── $timestamp
    ├── $configuration
    └── $iterations_performed
</pre>

## Scripts

### Core Pipeline

- **`feature_selection_and_classification_functions.R`**: Core utility functions implementing the three-phase iterative framework
    - `run_feature_selection_iterations()`: Main three-phase pipeline with rescue mechanisms
    - `run_analysis_pipeline()`: Phase 0 comprehensive feature selection and classification
    - `quick_classify_100_runs()`: 100-run Monte Carlo validation
    - `run_boruta()`, `run_LASSO()`: Feature selection implementations
    - Visualization functions for feature selection matrices and importance plots

### Pain Threshold Analysis (12 Features)

- **`Pheno_125_iterative_feature_selection_and_classification.R`**: Main analysis pipeline for pain threshold dataset
- **`Pheno_125_correlation.R`**: Correlation analysis and heatmap visualization
- **`Pheno_125_VIF.R`**: Variance Inflation Factor analysis for multicollinearity detection

### Psoriatic Arthritis Analysis

- **`PSA_das28_iterative_feature_selection_and_classification.R`**: Analysis pipeline for PsA dataset

### Synthetic Dataset Validation

- **`FCPS_experiments.R`**: Validation on FCPS clustering datasets (e.g., Atom dataset)

## Key Capabilities

### Feature Selection

- **Boruta algorithm**: Wrapper method capturing nonlinear importance and interactions via Random Forest
- **LASSO regularization**: Embedded method capturing linear contributions with L1 sparsity penalty
- **Complementary detection**: At least one method identifies features contributing to classification
- **Correlation analysis**: Identify and handle multicollinear features
- **Hierarchical rescue**: Progressive feature elimination with systematic recovery mechanisms

### Classification Methods

- **Random Forest**: Ensemble method with hyperparameter tuning (mtry, ntree)
- **Logistic Regression**: Linear classifier with binomial family
- **k-Nearest Neighbors**: Instance-based learning with centering/scaling preprocessing
- **C5.0 Decision Trees**: Rule-based classifier for interpretable models
- **Support Vector Machines**: Margin-based hyperplane classifier (radial kernel) with hyperparameter traing (C, sigma)

### Statistical Analysis

- **Monte-Carlo validation**: 100 runs with 2.5th and 97.5th percentile confidence intervals
- **Effect size calculation**: Cohen's d with confidence intervals and t-tests
- **Performance metrics**: Balanced accuracy, AUC-ROC, confusion matrices
- **Significance testing**: t-tests for feature differences between classes

### Output and Reporting

- **Results tables**: Comprehensive CSV files with all feature set combinations and performance metrics
- **Feature history**: RDS files tracking feature selection decisions across phases
- **Visualizations**: Feature selection matrices, importance plots, effect size plots
- **Warning flags**: Explicit indicators for dataset unsuitability or method limitations
- **Limitation transparency**: Critical warnings when feature selection methods fail to decompose interactions

## Example Results Structure

The pipeline generates comprehensive results including:

1. **Phase 0 Results**: All tested feature combinations with classification performance
2. **Minimal Feature Set**: Features surviving Phase 1 backward elimination
3. **Rescued Features**: Features recovered in Phase 2 individual testing
4. **Final Selected Features**: Union of minimal set and rescued features
5. **Final Rejected Features**: Features failing all rescue attempts
6. **Warning Flags**: Indicators for edge cases and method limitations
7. **Feature Selection History**: Complete tracking of feature status through all phases

## Handling Edge Cases

The pipeline explicitly handles and reports:

1. **No successful classification**: Terminates with dataset unsuitability warning
2. **Full feature set required**: Flags when feature selection fails to create successful subsets
3. **Complex interactions undetected**: Warning when rejected features succeed as group but no individual features or subsets can be identified by Boruta/LASSO
4. **Rescue mechanism iterations**: Continues until rejected set definitively fails or all rescuable features are recovered

## Installation & Requirements

### System Requirements

- **R version**: 4.0 or higher (recommended 4.3+)
- **Operating System**: Linux or Unix (macOS supported); **not tested on Windows**
- **RAM**: Minimum 4GB (8GB+ recommended for large datasets)
- **Processing**: Parallel processing recommended; CPU cores utilized: 4-8

**Note**: This code has been developed and tested exclusively on Linux/Unix systems. Windows compatibility has not been verified; users on Windows systems may encounter issues with parallel processing or file path handling.

### Package Installation

Install required packages from CRAN:

```r
packages <- c(
  "parallel",           # Base R: parallelization
  "opdisDownsampling",  # Balanced dataset splitting
  "randomForest",       # Random Forest implementation
  "caret",              # Machine learning framework
  "pbmcapply",          # Parallel processing with progress bars
  "Boruta",             # Boruta feature selection
  "reshape2",           # Data manipulation
  "pROC",               # ROC/AUC analysis
  "dplyr",              # Data wrangling
  "glmnet",             # LASSO/Ridge/Elastic Net regression
  "car",                # GLM diagnostics and VIF
  "effsize",            # Effect size calculations
  "ggplot2",            # Visualization
  "tidyr",              # Data tidying
  "C50"                 # C5.0 Decision Trees
)

# Install from CRAN
install.packages(packages)

# For SVM functionality (if needed)
install.packages("kernlab")
```

### Special Package: opdisDownsampling

**Status**: CRAN package  
**Purpose**: Balanced dataset splitting and downsampling for cross-validation

Installed automatically with the CRAN installation command above.

### Verify Installation

Test that all packages are loaded correctly:

```r
# Load all required packages
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
library(effsize)
library(ggplot2)
library(tidyr)
library(C50)

cat("All packages loaded successfully!\n")
```

### Version Compatibility

| Package | Tested Version | Minimum Version |
|---------|---|---|
| caret | 6.0+ | 6.0 |
| randomForest | 4.6+ | 4.6 |
| glmnet | 4.1+ | 4.0 |
| Boruta | 7.0+ | 7.0 |
| C50 | 0.1+ | 0.1 |
| pROC | 1.18+ | 1.15 |

## Input Data Format

### Expected Structure

- **Rows**: Individual samples/observations
- **Columns**: Numeric or categorical features + a class/outcome variable
- **Class variable**: Binary classification target (two distinct classes)
- **Missing values**: Should be handled prior to analysis (listwise deletion, imputation, or feature-specific handling as appropriate)
- **Class imbalance**: The pipeline uses balanced accuracy metrics which are robust to imbalance; extreme imbalance may require prior resampling

### Preparation

Users should prepare their data according to these requirements before adapting the analysis scripts. Implementation details for handling specific data characteristics (missing values, scaling, normalization) are provided in the core functions and example analysis scripts. See [Scripts](#scripts) section for example implementations on real datasets.

## Dependencies

```r
library(parallel)           # Parallel processing
library(opdisDownsampling)  # Balanced dataset splitting
library(randomForest)       # Random Forest algorithm
library(caret)              # ML toolkit
library(pbmcapply)          # Parallel progress bars
library(Boruta)             # Boruta feature selection
library(reshape2)           # Data reshaping
library(pROC)               # ROC/AUC analysis
library(dplyr)              # Data wrangling
library(glmnet)             # LASSO regression
library(car)                # GLM diagnostics
library(effsize)            # Cohen's d calculation
library(ggplot2)            # Visualization
library(tidyr)              # Data tidying
library(C50)                # C5.0 decision trees
```

## Additional Information

### Complementary Statistical Analyses

Beyond the machine learning pipeline, the repository includes functions for traditional statistical analyses to complement and validate feature selection results:

#### Standard and Penalized Logistic Regression

- **`run_single_logistic_regression()`**: Fits standard logistic regression models with complete coefficient summaries, p-values, and diagnostic statistics
- **`run_penalized_logistic_regression_all()`**: Implements Ridge (α=0), LASSO (α=1), and Elastic Net (α=0.5) penalized regression with cross-validated lambda selection
    - Compares coefficients across all three penalty types
    - Identifies features selected by each method (non-zero coefficients for LASSO/Elastic Net, threshold-based for Ridge)
    - Outputs comparison tables suitable for publication

#### Effect Size Analysis

- **`calculate_cohens_d_with_ttest()`**: Calculates Cohen's d effect sizes with 95% confidence intervals and paired t-tests for all features
- **`plot_cohens_d()`**: Visualizes effect sizes across multiple datasets with significance indicators
    - Categorizes effects as negligible (|d| < 0.2), small (0.2-0.5), medium (0.5-0.8), or large (≥ 0.8)
    - Highlights statistically significant differences

These analyses are typically applied to:
- Selected feature subsets from the four-phase pipeline
- Original complete dataset (with/without multicollinear variables)
- Training and validation splits separately
- Comparing modified vs. unmodified feature sets (e.g., with/without noise variables)

**Output files:**
- `[DATASET_NAME]_lr_orig_output.txt`: Complete logistic regression summaries
- `[DATASET_NAME]_penalized_lr_output.txt`: Penalized regression results with optimal lambdas
- `[DATASET_NAME]_penalized_comparison_table.csv`: Coefficient comparison across Ridge/LASSO/Elastic Net
- `[DATASET_NAME]_cohens_d_with_ttests_results.csv`: Effect sizes with confidence intervals and p-values
- `[DATASET_NAME]_cohens_d_with_ttests.svg`: Effect size visualization

### SPSS Statistical Analyses for Comparison
- **Pheno_125_SPSS_regression_complete_dataset.pdf**: Regression analysis results of pain threshold data
- **PSA_das28crp_SPSS_regression_complete_dataset.pdf**: Regression analysis results of psoriatic arthritis phenotype data

## Scientific Rationale

The four-phase iterative structure (Phases 0–3) was deliberately designed to address the challenge of identifying minimal sufficient feature sets while preserving scientifically valuable information:

1. **Why separate Phase 0 (comprehensive testing) from Phases 1–3 (iterative refinement)?**
   - Phase 0 exhaustively tests all feature combinations from two complementary methods (Boruta + LASSO), establishing a complete baseline of what can and cannot classify
   - Phases 1–3 progressively minimize the feature set with systematic recovery mechanisms
   - This separation ensures we don't prematurely eliminate features before understanding the full landscape of possibilities

2. **Why include Phase 1 backward elimination?**
   - Provides efficient systematic removal of redundant features while maintaining classification success
   - Identifies the truly minimal subset from Phase 0's successful candidates
   - Computationally more efficient than iterating Phase 0 repeatedly

3. **Why include Phase 2 individual rescue?**
   - Addresses the core problem that feature selection procedures can reject individually strong predictors due to redundancy masking, interaction effects, or stochastic variability
   - A feature "A" may have genuine independent classification ability but be rejected if features "B" and "C" together form a stronger combined predictor
   - Preserves scientifically valuable features with independent predictive power that would otherwise be lost
   - Directly targets the critical finding that features failing standard feature selection can nonetheless classify successfully when used alone

4. **Why hierarchical Phase 3 re-analysis and iterative rescue?**
   - Detects when rejected features succeed only through complex interactions undetectable by individual feature selection methods
   - Systematically applies Phase 0 logic to rejected feature sets, identifying any additional rescuable features
   - Explicitly flags cases where features are informative only through interactions both methods fail to decompose
   - Maintains scientific transparency about method limitations
   - Iterates until rejected sets definitively fail, ensuring complete exploration of the feature space

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0).

## Citation

Lötsch, J, Himmelspach A, Kringel D. (2025). Resolving interpretation challenges in machine learning feature selection with an iterative approach in biomedical pain data. [European Journal of Pain 2025 (accepted)]