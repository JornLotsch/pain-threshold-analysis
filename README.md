# Resolving interpretation challenges in machine learning feature selection with an iterative approach in biomedical pain data

## Three-Phase Iterative Feature Selection and Classification Pipeline: 

This project provides a comprehensive three-phase iterative machine learning framework that systematically identifies the truly minimal sufficient feature set for classification while ensuring no valuable features are overlooked through rescue mechanisms and explicit detection of feature selection method limitations.

## Overview

This repository provides a rigorous framework for analyzing classification datasets through an iterative three-phase approach combining multiple feature selection methods and machine learning algorithms. Originally developed for pain threshold phenotype prediction using multimodal sensory data, the pipeline has been generalized for use with any tabular classification dataset.

The framework addresses a critical challenge in machine learning: feature selection methods may erroneously reject individually-strong features due to redundancy masking, interaction effects, or stochastic variability. Our three-phase approach systematically minimizes the feature set while implementing rescue mechanisms to recover falsely rejected features and explicitly flagging cases where feature selection methods fail to decompose complex interactions.

## Key Features

- **Three-phase iterative approach**: Comprehensive feature selection (Phase 0), greedy backward elimination (Phase 1), and individual feature rescue with hierarchical verification (Phase 2-3)
- **Multiple feature selection methods**: Boruta algorithm (nonlinear importance via Random Forest) and LASSO regularization (linear contributions with sparsity)
- **Comprehensive classifier ensemble**: Random Forest, Logistic Regression, k-Nearest Neighbors, and C5.0 Decision Trees
- **Rigorous statistical validation**: 100-run Monte Carlo cross-validation with 95% confidence intervals
- **Rescue mechanisms**: Individual testing of rejected features and hierarchical re-analysis of rejected feature sets
- **Limitation detection**: Explicit warnings when feature selection methods fail to identify responsible features in successful sets
- **Effect size analysis**: Cohen's d calculations with statistical testing
- **Comprehensive visualization**: Feature selection matrices, importance plots, and effect size visualizations
- **Flexible configuration**: Easy adaptation to different datasets and analysis requirements

## Algorithm Workflow

![Image](FlowChart.drawio.svg)
**Figure:** Flowchart of the four-phase feature selection framework. Blue elements represent decision points and final outputs. Gray boxes delineate Phases 0–3. The iterative loop in Phase 3 (right) continues until rejected features no longer enable classification.


### Phase 0: Comprehensive Initial Feature Selection

1. Apply Boruta algorithm and LASSO regularization to full **training** dataset
2. Create multiple feature subset combinations:
    - All features
    - Boruta-selected features **("confirmed" only)**
    - Boruta-rejected features
    - LASSO-selected features **(non-zero coefficients at lambda.min)**
    - LASSO-rejected features
    - Union of Boruta and LASSO selections
    - Intersection of Boruta and LASSO selections
3. Test all combinations with five classifiers (RF, LR, KNN, C5.0, SVM) **tuned by grid search for hyperparameters**
4. Identify smallest feature set achieving classification success (lower CI for balanced accuracy > 0.5)
5. If no feature set succeeds, **including all features**, terminate with dataset unsuitability warning
6. If **smallest successful subset is identical to full feature set**, terminate and return **"all features necessary"**; otherwise pass to Phase 1

### Phase 1: Greedy Backward Elimination

1. Start with smallest successful feature set from Phase 0
2. Iteratively test removal of each individual feature
3. Permanently remove features whose absence maintains classification success
4. Continue until no further features can be removed without classification failure
5. Result: Minimal feature set necessary for classification

### Phase 2: Individual Feature Rescue

1. Test each rejected feature individually for independent predictive ability
2. Rescue any feature demonstrating classification success alone (lower CI > 0.5)
3. Prevents loss of individually-strong features masked by redundancy or interactions
4. Critical for scientific completeness: features with independent predictive power are valuable even if overshadowed during combined testing

### Phase 3: Hierarchical Verification and Iterative Rescue

1. Verify final selected feature set achieves classification success
2. Test if rejected feature set (as a group) can still classify
3. **If rejected set succeeds**: Apply complete Phase 0 pipeline to rejected features:
    - Run Boruta and LASSO on rejected feature set
    - Test all subset combinations (Boruta-selected, LASSO-selected, unions, intersections)
    - Individually verify features identified by either method
    - Rescue features demonstrating independent classification success
4. **If no features identified but set succeeds**: Issue critical warning about feature selection method limitations
5. Iterate rescue mechanism until rejected set fails classification
6. Flag cases where complex feature interactions cannot be decomposed by feature selection methods

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
- Selected feature subsets from the three-phase pipeline
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

The three-phase structure was deliberately chosen over alternative approaches:

1. **Why not iterate Phases 0 and 1 repeatedly?**
   - Computationally inefficient (repeated expensive Boruta/LASSO analyses)
   - Provides diminishing returns (Phase 1 already exhaustively minimizes)
   - Omits individual testing that prevents false rejection

2. **Why include Phase 2 individual rescue?**
   - Feature A with independent classification ability (CI > 0.5) might be rejected if features B and C together form stronger predictor
   - A's individual importance is masked despite standalone predictive value
   - Scientifically valuable to retain all individually-predictive features

3. **Why hierarchical Phase 3 re-analysis?**
   - Detects when rejected features succeed only through complex interactions
   - Applies systematic feature selection to decompose responsible features
   - Explicitly flags cases where both Boruta and LASSO fail
   - Maintains scientific transparency about method limitations

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0).

## Citation

Lötsch, J, Himmelspach A, Kringel D. (2025). Resolving interpretation challenges in machine learning feature selection with an iterative approach in biomedical pain data. [European Journal of Pain 2025 (in revision)]