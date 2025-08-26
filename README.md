# Pain Threshold Analysis

Analysis code for pain threshold phenotype prediction using multimodal sensory data.

## Overview

This repository contains R scripts for analyzing pain threshold data across different stimulus modalities (thermal, mechanical, electrical, and sensitization effects). The analysis includes correlation assessment, multicollinearity detection, and machine learning classification approaches.

## Scripts

- `Pheno_125_correlation.R` - Correlation analysis and heatmap visualization of pain threshold variables grouped by stimulus type
- `Pheno_125_VIF.R` - Variance Inflation Factor analysis to identify collinearity and aliasing issues in logistic regression models
- `Pheno_125_RF_LR_kNN.R` - Machine learning classification using Random Forest, Logistic Regression, and k-Nearest Neighbors with feature selection via Boruta and LASSO
- `FCPS_experiments.R` - Additional experimental analysis on a data set where regression fails

## Data

The analysis uses pain threshold measurements across 11 variables:
- Thermal stimuli: Heat, Cold, with and without capsaicin/menthol
- Mechanical stimuli: Pressure, von Frey filaments, with capsaicin effects
- Electrical stimuli: Current thresholds
- Sensitization effects: Capsaicin and menthol modulation

## Dependencies

Key R packages: `ComplexHeatmap`, `randomForest`, `caret`, `Boruta`, `glmnet`, `car`, `pROC`, `viridis`, `ggthemes`

## Usage

Scripts are designed to run independently. Update file paths in the constants section to match your data location.

## Output

- Correlation heatmaps (SVG format)
- VIF analysis heatmaps
- Feature importance plots
- Classification performance metrics with confidence intervals

## Note

This code accompanies a manuscript submission. Data files are not included in this repository.