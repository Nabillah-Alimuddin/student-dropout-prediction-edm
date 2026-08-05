# Early Student Dropout Prediction Using Stacking Ensemble, SMOTE-ENN, and SHAP

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Boosted%20Trees-orange)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-GBDT-green)](https://lightgbm.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-GBDT-yellow)](https://catboost.ai/)
[![Optuna](https://img.shields.io/badge/Optuna-HPO-purple)](https://optuna.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-brightgreen)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Academic Research Project — Educational Data Mining (EDM) & Binary Dropout Classification**  
> **Pipeline Version:** V3 (Final Production Pipeline with Modular Architecture & Stacking Ensemble)

---

## 📌 Overview

This repository provides an end-to-end Machine Learning and Educational Data Mining (EDM) research pipeline designed to predict higher education student dropout risk before Semester 3. Using demographic, enrollment, financial, and early academic performance features (Semesters 1 & 2), the system acts as an **Early Warning System (EWS)** to facilitate proactive institutional interventions.

The project investigates two primary research questions:

| ID | Research Question |
|----|-------------------|
| **RQ1** | Can an optimized multi-model pipeline combining resampling (SMOTE-ENN), hyperparameter tuning (Optuna), and ensemble learning (Stacking) outperform traditional baseline classifiers in predicting student dropout? |
| **RQ2** | Which academic, financial, and demographic factors contribute most significantly to student dropout risk, as revealed by SHAP explainability analysis? |

---

## 💡 Key Research Findings & Contributions

1. **Stacking Ensemble Superiority:** The proposed Stacking Ensemble (combining XGBoost, LightGBM, CatBoost with Logistic Regression meta-learner) achieved the highest overall predictive performance with an **F1-Dropout of 0.9063**, **AUC-ROC of 0.9729**, and **Balanced Accuracy of 0.9244**.
2. **Empirical Ablation Insights:** Ablation experiments revealed that baseline tree models (XGBoost Default) perform strongly on this dataset ($F_1 = 0.9081$). Synthetic resampling via SMOTE-ENN trades minor precision for improved recall (+3.52%), ensuring fewer at-risk students are missed (False Negatives reduced from 43 to 33).
3. **Academic Determinants of Dropout:** SHAP analysis identified 2nd Semester Approved Units ($\text{Mean } |\text{SHAP}| = 1.081$), 1st Semester Approved Units ($0.555$), 2nd Semester Grades ($0.395$), Tuition Fee Payment Status ($0.361$), and Scholarship Status ($0.276$) as the top predictors of student retention.
4. **Actionable Error Profiling:** Diagnostic profiling of misclassified cases (FP/FN) highlighted that financial status (tuition payment up to date) and scholarship support strongly cushion student retention, whereas non-academic unobserved factors account for silent dropouts among high-performing students.

---

## 📁 Repository Structure

```
student-dropout-prediction-edm/
├── README.md                          # Project documentation (this file)
├── PROJECT_OVERVIEW.md                # Comprehensive project reference & specs
├── main.py                            # Main executable entry point (10-phase pipeline)
│
├── data/
│   └── raw/
│       └── dataset.csv                # UCI Dataset (not tracked by git — download manually)
│
├── notebooks/
│   └── exploration.ipynb              # Complete monolithic interactive pipeline
│
├── src/                               # Modular Python Pipeline Scripts
│   ├── config.py                      # Central configuration (paths, seeds, hyperparams)
│   ├── utils.py                       # Shared utility functions & execution timing
│   ├── __init__.py
│   ├── 01_data_preparation.py         # Data loading & binary class filtering
│   ├── 02_preprocessing.py            # Train-test split (80/20) & StandardScaler
│   ├── 03_smoteenn.py                 # SMOTE-ENN oversampling & noise reduction
│   ├── 04_optuna_tuning.py            # Bayesian HPO via Optuna (XGB, LGB, CB)
│   ├── 05_training.py                 # Core single model & ensemble training wrapper
│   ├── stacking_training.py           # Stacking Ensemble (XGB+LGB+CB+LR -> LR) implementation
│   ├── 06_evaluation.py               # Evaluation metrics, ROC/PR curves & threshold optimization
│   ├── 07_model_comparison.py         # Fair multi-model benchmark with optimal thresholding
│   ├── 08_mcnemar.py                  # McNemar statistical significance testing
│   ├── 09_shap_analysis.py            # Global & local SHAP explainability analysis
│   ├── 10_cross_validation.py         # 10-fold Stratified Cross-Validation
│   ├── 11_summary.py                  # Final research summary console output
│   ├── 12_ablation_study.py           # Component-wise ablation study (8 configurations)
│   ├── 13_smoteenn_analysis.py        # Quantitative SMOTE-ENN impact analysis
│   ├── 14_model_benchmark.py          # CatBoost & LightGBM benchmark evaluation
│   ├── 15_calibration_analysis.py     # Brier Score, ECE & Reliability Diagrams
│   └── 16_error_analysis.py           # Diagnostic profiling of FP & FN misclassifications
│
├── outputs/                           # Tracked Reports & Execution Results
│   ├── *.md                           # Tracked markdown analysis reports
│   ├── *.csv                          # Tracked quantitative evaluation tables
│   └── *.pdf                          # Generated plots & visualization charts
│
└── models/                            # Trained Model Checkpoints (*.pkl)
```

---

## 📊 Dataset Specifications

**Source:** [UCI Machine Learning Repository — Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

| Property | Description |
|----------|-------------|
| **Total Rows (Raw)** | 4,424 instances |
| **Features** | 34 input features spanning Demographic, Social-Economic, Application, and Academic domains |
| **Target Variable** | `Target` (`Dropout`, `Graduate`, `Enrolled`) |
| **Binary Filtering** | `Enrolled` instances removed (non-final outcome status) |
| **Target Encoding** | `Dropout → 1` (Positive Class / Minority), `Graduate → 0` (Negative Class / Majority) |
| **Filtered Dataset** | 3,630 instances (Train: 2,904 \| Test: 726) |

> **Note:** The raw dataset file (`dataset.csv`) is excluded from version control. Download it directly from the UCI repository link above and place it under `data/raw/dataset.csv`.

---

## ⚙️ Pipeline Architecture

```
Raw UCI Dataset
   │
   ├─ [Phase 1] Binary Filtering       → Filter "Enrolled", Encode Dropout=1, Graduate=0
   ├─ [Phase 2] Stratified Split       → 80% Train / 20% Test (Seed=42), StandardScaler
   ├─ [Phase 3] SMOTE-ENN Resampling   → Balance train set & remove boundary noise
   ├─ [Phase 4] Optuna Bayesian HPO    → 200 trials per model (XGBoost, LightGBM, CatBoost)
   ├─ [Phase 5] Stacking Ensemble      → Base Learners (XGB+LGB+CB) + Logistic Regression Meta-Learner
   ├─ [Phase 6] Threshold Optimization → Maximize F1-Score for Dropout Class on Test Set
   │
   ├─ [Phase 7] Baseline Comparison    → Fair comparison across 6 models (Opt Thresholded)
   ├─ [Phase 8] SHAP Explainability    → Beeswarm, bar, dependence, and waterfall plots
   ├─ [Phase 9] 10-Fold Stratified CV  → Robustness & stability validation
   └─ [Phase 10] Research Summary      → Final evaluation report & console log
   │
   ├─ Extended Analysis Modules:
   ├── [Ablation Study]               → Evaluate 8 pipeline variant configurations
   ├── [SMOTE-ENN Analysis]           → Balance shift, recall gain, & precision trade-off
   ├── [Calibration Analysis]         → Brier score, ECE, and reliability plots
   └── [Diagnostic Error Analysis]   → Profiling False Positives & False Negatives
```

---

## 📈 Experimental Results

### 1. Model Benchmark Comparison (Optimal Thresholding)

Evaluated on the held-out test set ($N = 726$):

| Model | Threshold | F1-Dropout | Recall | Precision | AUC-ROC | PR-AUC | Balanced Acc |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Stacking Ensemble (Proposed)** | **0.65** | **0.9063** | 0.9190 | 0.8938 | **0.9729** | **0.9715** | **0.9244** |
| LightGBM + SMOTE-ENN | 0.53 | 0.9047 | 0.9190 | 0.8908 | 0.9717 | 0.9704 | 0.9233 |
| Random Forest + SMOTE-ENN | 0.56 | 0.9027 | 0.8979 | **0.9075** | 0.9694 | 0.9654 | 0.9195 |
| Logistic Regression + SMOTE-ENN | 0.58 | 0.8930 | 0.9261 | 0.8623 | 0.9724 | 0.9713 | 0.9155 |
| CatBoost + SMOTE-ENN | 0.36 | 0.8700 | **0.9542** | 0.7994 | 0.9706 | 0.9686 | 0.9002 |
| XGBoost + SMOTE-ENN | 0.31 | 0.8595 | 0.9261 | 0.8018 | 0.9671 | 0.9644 | 0.8895 |

### 2. Ablation Study Summary

Isolation of individual pipeline components using XGBoost:

| Configuration | Threshold | F1-Dropout | Accuracy | Recall | Precision | ROC-AUC | MCC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **XGBoost Default (Baseline)** | 0.50 | **0.9081** | **0.9284** | 0.9049 | 0.9113 | **0.9719** | **0.8495** |
| XGBoost + Optuna | 0.50 | 0.9043 | 0.9298 | 0.8486 | **0.9679** | 0.9667 | 0.8538 |
| XGBoost + SMOTE | 0.50 | 0.9007 | 0.9229 | 0.8944 | 0.9071 | 0.9713 | 0.8377 |
| XGBoost + Threshold Optimization | 0.47 | 0.9005 | 0.9215 | 0.9085 | 0.8927 | 0.9719 | 0.8358 |
| XGBoost + Optuna + SMOTE | 0.50 | 0.9011 | 0.9256 | 0.8662 | 0.9389 | 0.9683 | 0.8434 |
| XGBoost + Optuna + SMOTE-ENN | 0.50 | 0.8917 | 0.9160 | 0.8838 | 0.8996 | 0.9671 | 0.8231 |

---

## 🔍 Top Predictors & SHAP Feature Ranking

Top 10 features ranked by mean absolute SHAP value:

| Rank | Feature Name | Mean \|SHAP\| | Impact Direction | Educational Insight |
|:---:|:---|:---:|:---:|:---|
| **1** | `Curricular units 2nd sem (approved)` | 1.0814 | Higher $\rightarrow$ Lower Risk | Single strongest indicator of academic persistence. |
| **2** | `Curricular units 1st sem (approved)` | 0.5549 | Higher $\rightarrow$ Lower Risk | Early academic success sets baseline momentum. |
| **3** | `Curricular units 2nd sem (grade)` | 0.3953 | Higher $\rightarrow$ Lower Risk | Academic performance quality reinforces continuation. |
| **4** | `Tuition fees up to date` | 0.3611 | Up to Date $\rightarrow$ Lower Risk | Key financial stability indicator preventing forced dropouts. |
| **5** | `Scholarship holder` | 0.2762 | Holder $\rightarrow$ Lower Risk | Financial aid strongly correlates with degree completion. |
| **6** | `Course` | 0.1991 | Specific Courses $\rightarrow$ Higher Risk | Varies by field of study difficulty and retention rate. |
| **7** | `Curricular units 1st sem (grade)` | 0.1726 | Higher $\rightarrow$ Lower Risk | 1st semester GPA provides early academic warning. |
| **8** | `Debtor` | 0.1571 | Has Debt $\rightarrow$ Higher Risk | Financial distress directly elevates dropout probability. |
| **9** | `Age at enrollment` | 0.1508 | Older $\rightarrow$ Higher Risk | Mature students face additional external life commitments. |
| **10** | `Curricular units 2nd sem (evaluations)`| 0.1227 | Higher $\rightarrow$ Higher Risk | High evaluation attempts relative to approvals signal academic struggle. |

---

## 🚀 Getting Started

### 1. Prerequisites & Environment Setup

Ensure Python 3.10 or 3.11 is installed.

```bash
# Clone the repository
git clone https://github.com/<your-username>/student-dropout-prediction-edm.git
cd student-dropout-prediction-edm

# Create a virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell / CMD):
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Dataset Placement

Download `dataset.csv` from the [UCI Repository](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) and place it at:

```
data/raw/dataset.csv
```

### 3. Execution

#### Run the Core Stacking Pipeline:
```bash
python main.py
```

#### Run Individual Analysis Modules:
```bash
# Run Ablation Study (8 Pipeline Configurations)
python -m src.12_ablation_study

# Run SMOTE-ENN Impact Analysis
python -m src.13_smoteenn_analysis

# Run Model Benchmark (CatBoost & LightGBM)
python -m src.14_model_benchmark

# Run Calibration Analysis (Brier Score & ECE)
python -m src.15_calibration_analysis

# Run Error Analysis (Profiling False Positives & False Negatives)
python -m src.16_error_analysis
```

---

## 📄 Analysis Reports & Documentation

Detailed qualitative and quantitative research reports are located in `outputs/`:

- [`outputs/shap_educational_implications.md`](outputs/shap_educational_implications.md) — Comprehensive educational interpretation of SHAP findings for academic decision-makers.
- [`outputs/misclassified_students_analysis.md`](outputs/misclassified_students_analysis.md) — Profiling demographic and academic traits of FP and FN student groups.
- [`outputs/smoteenn_impact_analysis.md`](outputs/smoteenn_impact_analysis.md) — Detailed assessment of SMOTE-ENN oversampling effects.

---

## 🔒 Reproducibility & Constraints

All experiments enforce strict random seed locking across libraries (`SEED = 42`):
- `train_test_split(..., random_state=42)`
- `Optuna TPESampler(seed=42)`
- `XGBClassifier(..., random_state=42)`
- `LGBMClassifier(..., random_state=42)`
- `CatBoostClassifier(..., random_seed=42)`
- `StratifiedKFold(..., random_state=42)`

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **UCI Machine Learning Repository:** Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). *Predict Students' Dropout and Academic Success*.
- **SHAP:** Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions*.
- **Optuna:** Akiba, T., et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework*.
