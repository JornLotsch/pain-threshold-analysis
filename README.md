# Feature Selection and Machine Learning Classification Pipeline
A comprehensive R pipeline for automated feature selection and classification analysis on tabular datasets, with iterative refinement capabilities.
## Overview
This repository provides a flexible framework for analyzing classification datasets through multiple feature selection methods and machine learning algorithms. Originally developed for pain threshold phenotype prediction using multimodal sensory data, the pipeline has been generalized for use with any tabular classification dataset.
## Key Features
- **Multiple Feature Selection Methods**: Boruta algorithm and LASSO regularization
- **Iterative Analysis**: Automated iterative feature refinement with stopping criteria
- **Multiple Classifiers**: Random Forest, Logistic Regression, and k-Nearest Neighbors
- **Statistical Validation**: 100-run bootstrap with 95% confidence intervals
- **Effect Size Analysis**: Cohen's d calculations with statistical testing
- **Comprehensive Visualization**: Feature selection matrices, importance plots, and effect size visualizations
- **Flexible Configuration**: Easy adaptation to different datasets and analysis requirements

## Scripts
### Core Pipeline
- - Main analysis pipeline with iterative feature refinement `iterative_feature_selection_and_classification.R`
- - Core utility functions and classification algorithms `feature_selection_and_classification_functions.R`

### Pain Threshold Specific Scripts
- - Correlation analysis and heatmap visualization `Pheno_125_correlation.R`
- - Variance Inflation Factor analysis for multicollinearity detection `Pheno_125_VIF.R`
- - Basic classification without iterative refinement `Pheno_125_RF_LR_kNN.R`

### Experimental Analysis
- - Example analysis on FCPS clustering datasets demonstrating pipeline flexibility `FCPS_experiments.R`

## Algorithm Workflow
1. **Data Loading & Preprocessing**: Load features and targets, handle missing data, optional column renaming
2. **Initial Feature Selection**: Run Boruta and LASSO on full dataset
3. **Iterative Refinement**:
    - Create feature subsets based on selection results
    - Test classification performance on multiple feature combinations
    - Remove selected features and repeat if "rejected" features show good performance
    - Continue until stopping criteria met (max iterations, no successful classification on rejected features)

4. **Final Analysis**: Generate comprehensive results tables, visualizations, and statistical summaries

## Key Capabilities
### Feature Selection
- **Boruta Algorithm**: Wrapper method comparing feature importance against shadow features
- **LASSO Regularization**: L1 penalty for automatic feature selection
- **Correlation Analysis**: Identify and handle multicollinear features
- **Iterative Refinement**: Progressive feature elimination based on performance

### Classification Methods
- **Random Forest**: Ensemble method with hyperparameter tuning
- **Logistic Regression**: Linear classifier with regularization options
- : Instance-based learning with preprocessing **k-Nearest Neighbors**

### Statistical Analysis
- **Bootstrap Validation**: 100 runs with confidence intervals
- **Effect Size Calculation**: Cohen's d with confidence intervals
- **Performance Metrics**: Balanced accuracy, AUC-ROC, confusion matrices
- **Significance Testing**: t-tests for feature differences between classes
