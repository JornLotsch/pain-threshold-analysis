# Feature selection and machine learning classification pipeline

This project provides an iterative machine-learning framework using ensemble feature selection and multiple classifiers to identify trait-relevant variables in pain-related datasets, ensuring interpretability by testing classification success on both selected and unselected features.

## Overview

This repository provides a flexible framework for analyzing classification datasets through multiple feature selection methods and machine learning algorithms. Originally developed for pain threshold phenotype prediction using multimodal sensory data, the pipeline has been generalized for use with any tabular classification dataset.

## Key features

- **Multiple feature selection methods**: Boruta algorithm and LASSO regularization
- **Iterative analysis**: Automated iterative feature refinement with stopping criteria
- **Multiple classifiers**: Random forest, logistic regression, and k-nearest neighbors
- **Statistical validation**: 100-run bootstrap with 95% confidence intervals
- **Effect size analysis**: Cohen's d calculations with statistical testing
- **Comprehensive visualization**: Feature selection matrices, importance plots, and effect size visualizations
- **Flexible configuration**: Easy adaptation to different datasets and analysis requirements

## Scripts

### Core pipeline

- Core utility functions and classification algorithms: `feature_selection_and_classification_functions.R`

### Pain threshold specific scripts

- Main analysis pipeline with iterative feature refinement for pain threshold data set: `Pheno_125_iterative_feature_selection_and_classification.R`
- Correlation analysis and heatmap visualization for pain threshold data set: `Pheno_125_correlation.R`
- Variance inflation factor analysis for multicollinearity detection for pain threshold data set: `Pheno_125_VIF.R`

### PsA data specific scripts

- Main analysis pipeline with iterative feature refinement for PsA data set: `PSA_das28_iterative_feature_selection_and_classification.R`

### Analysis of synthetic data set

- Example analysis on FCPS clustering datasets demonstrating pipeline flexibility: `FCPS_experiments.R`

## Algorithm workflow

1. **Data loading & preprocessing**: Load features and targets, handle missing data, optional column renaming
2. **Initial feature selection**: Run Boruta and LASSO on full dataset
3. **Iterative refinement**:
    - Create feature subsets based on selection results
    - Test classification performance on multiple feature combinations
    - Remove selected features and repeat if "rejected" features show good performance
    - Continue until stopping criteria met (max iterations, no successful classification on rejected features)
4. **Final analysis**: Generate comprehensive results tables, visualizations, and statistical summaries

## Key capabilities

### Feature selection

- **Boruta algorithm**: Wrapper method comparing feature importance against shadow features
- **LASSO regularization**: L1 penalty for automatic feature selection
- **Correlation analysis**: Identify and handle multicollinear features
- **Iterative refinement**: Progressive feature elimination based on performance

### Classification methods

- **Random forests**: Ensemble method with hyperparameter tuning
- **Logistic regression**: Linear classifier with regularization options
- **k-nearest neighbors**: Instance-based learning with preprocessing

### Statistical analysis

- **Bootstrap validation**: 100 runs with confidence intervals
- **Effect size calculation**: Cohen's d with confidence intervals
- **Performance metrics**: Balanced accuracy, AUC-ROC, confusion matrices
- **Significance testing**: t-tests for feature differences between classes

## Additional information

### SPSS statistical analyses for comparison

- **Pheno_125_SPSS_regression_complete_dataset.pdf**: Regression analysis results of pain threshold data
- **PSA_das28crp_SPSS_regression_complete_dataset.pdf**: Regression analysis results of psoriatic arthritis phenotype data

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0).

## Citation

Lötsch, J, Himmelspach A, Kringel D. (2025). Resolving interpretation challenges in machine learning feature selection with an iterative approach in biomedical pain data. [submitted]
