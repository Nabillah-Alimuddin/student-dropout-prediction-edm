# PROJECT CONTEXT

## Project Title

Early Student Dropout Prediction Using XGBoost, SMOTE-ENN, and SHAP Explainability

---

# Research Overview

This project aims to develop a machine learning model capable of predicting students at risk of dropping out from higher education institutions.

The study focuses on binary classification:

* Dropout = 1
* Graduate = 0

The objective is to maximize the identification of students who are likely to drop out while maintaining balanced classification performance.

---

# Research Questions

## RQ1

Can XGBoost combined with SMOTE-ENN outperform traditional machine learning algorithms in predicting student dropout?

## RQ2

Is the performance improvement statistically significant compared to baseline models?

---

# Dataset

Dataset source:

Student Performance / Higher Education Student Dataset

Target variable:

Target

Original classes:

* Dropout
* Graduate
* Enrolled

For this research:

* Dropout → 1
* Graduate → 0
* Enrolled records are removed

This transforms the problem into a binary classification task.

---

# Research Pipeline

## Phase 1 — Data Preparation

Objectives:

* Load dataset
* Check dimensions
* Check missing values
* Examine class distribution
* Remove "Enrolled" class
* Encode target labels

Outputs:

* Binary classification dataset
* Clean target labels

---

## Phase 2 — Data Splitting

Method:

Stratified Train-Test Split

Configuration:

* Training = 80%
* Testing = 20%

Reason:

Preserve class proportions across train and test sets.

---

## Phase 3 — Feature Scaling

Method:

StandardScaler

Objectives:

* Normalize feature ranges
* Improve model stability
* Prepare data for resampling

Outputs:

* X_train_scaled
* X_test_scaled

---

## Phase 4 — Imbalanced Data Handling

Method:

SMOTE-ENN

Components:

### SMOTE

Creates synthetic samples for minority class.

### ENN (Edited Nearest Neighbors)

Removes noisy and potentially mislabeled observations.

Objectives:

* Balance class distribution
* Improve minority class detection

Outputs:

* X_resampled
* y_resampled

---

## Phase 5 — Hyperparameter Optimization

Method:

Optuna

Optimization target:

Maximize F1-Score for Dropout class.

Parameters searched:

* n_estimators
* max_depth
* learning_rate
* subsample
* colsample_bytree
* gamma
* min_child_weight

Outputs:

* Best parameter configuration

---

## Phase 6 — Model Training

Algorithm:

XGBoost Classifier

Training data:

SMOTE-ENN resampled dataset

Outputs:

* Final trained model

---

## Phase 7 — Model Evaluation

Metrics:

* Accuracy
* Balanced Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

Priority metric:

F1 Score for Dropout class

Reason:

Dropout detection is the primary objective.

---

## Phase 8 — Threshold Optimization

Purpose:

Find the probability threshold that produces the best balance between:

* Precision
* Recall
* F1 Score

Default threshold:

0.50

Experiment range:

0.05 – 0.95

---

## Phase 9 — Baseline Comparison

Models compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Baseline XGBoost
5. Proposed XGBoost + SMOTE-ENN

Purpose:

Evaluate whether the proposed approach provides meaningful improvements.

---

## Phase 10 — Statistical Significance Testing

Method:

McNemar Test

Purpose:

Determine whether performance differences between models are statistically significant.

Significance level:

α = 0.05

---

## Phase 11 — Error Analysis

Method:

Confusion Matrix

Analyze:

* True Positive
* True Negative
* False Positive
* False Negative

Purpose:

Understand classification mistakes and model behavior.

---

## Phase 12 — Explainable AI

Method:

SHAP (SHapley Additive exPlanations)

Analyses:

### SHAP Summary Plot

Global feature importance.

### SHAP Beeswarm Plot

Feature influence distribution.

### SHAP Dependence Plot

Relationship between feature values and prediction outputs.

Purpose:

Explain why the model predicts a student as likely to drop out.

---

## Phase 13 — Robustness Validation

Method:

10-Fold Cross Validation

Metrics:

* F1 Score
* ROC-AUC
* Balanced Accuracy

Purpose:

Measure model stability and generalization performance.

---

## Phase 14 — Computational Efficiency

Measure execution time for:

* Scaling
* SMOTE-ENN
* Hyperparameter tuning
* Training
* SHAP computation
* Cross validation

Purpose:

Assess computational cost of the proposed framework.

---

# Explainability Requirements

The project emphasizes explainability.

Any modification must preserve:

* SHAP compatibility
* Feature importance analysis
* Interpretability of predictions

Avoid black-box modifications that reduce explainability.

---

# Development Rules

When modifying this project:

1. Do not change research objectives.
2. Do not alter target encoding.
3. Keep binary classification setting.
4. Maintain SMOTE-ENN preprocessing.
5. Preserve SHAP explainability pipeline.
6. Preserve statistical testing procedures.
7. Explain all major code changes before implementation.
8. Prioritize reproducibility and academic validity.

---

# Success Criteria

A successful model should:

* Achieve strong F1 Score for Dropout class.
* Improve minority class detection.
* Demonstrate statistically significant improvement.
* Remain explainable through SHAP.
* Show stable performance through cross-validation.

---

# Expected Final Deliverables

* Trained XGBoost model
* Evaluation report
* Baseline comparison report
* McNemar significance results
* SHAP explainability results
* Cross-validation analysis
* Research conclusions
