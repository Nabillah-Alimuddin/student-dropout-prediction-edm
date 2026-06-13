# Early Student Dropout Prediction Using XGBoost, SMOTE-ENN, and SHAP

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Boosted%20Trees-orange)](https://xgboost.readthedocs.io/)
[![Optuna](https://img.shields.io/badge/Optuna-HPO-purple)](https://optuna.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-green)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Academic Research Project — Binary Classification for Higher Education Student Dropout Prediction**  
> Pipeline Version: V3 (Final)

---

## 📌 Overview

This repository contains a complete machine learning research pipeline for predicting student dropout in higher education using demographic, enrollment, and early academic performance data (Semesters 1 & 2).

The research is structured around two core questions:

| ID | Research Question |
|----|-------------------|
| **RQ1** | Can XGBoost combined with SMOTE-ENN outperform traditional ML algorithms in predicting student dropout? |
| **RQ2** | Is the performance improvement statistically significant compared to baseline models? |

The practical objective is an **early warning system**: predict dropout risk *before* Semester 3 to enable timely institutional intervention.

---

## 📁 Repository Structure

```
riset/
├── README.md                        # Project overview (this file)
├── PROJECT_OVERVIEW.md              # Detailed pipeline documentation
├── enhacementRoadmap.md             # Research enhancement roadmap
├── researchPipeline.md              # Pipeline design notes
├── main.py                          # Entry point for the core pipeline
│
├── data/
│   └── raw/
│       └── dataset.csv              # UCI dataset (not tracked by git — add manually)
│
├── notebooks/
│   └── exploration.ipynb            # Complete monolithic pipeline (V3 Final)
│
├── src/
│   ├── config.py                    # Central config (paths, seeds, hyperparams)
│   ├── utils.py                     # Shared utility functions
│   ├── __init__.py
│   ├── 01_data_preparation.py       # Load & binary-filter dataset
│   ├── 02_preprocessing.py          # StandardScaler, train-test split
│   ├── 03_smoteenn.py               # SMOTE-ENN resampling
│   ├── 04_optuna_tuning.py          # Bayesian HPO with Optuna (200 trials)
│   ├── 05_training.py               # XGBoost model training
│   ├── 06_evaluation.py             # Core metrics, ROC, threshold analysis
│   ├── 07_model_comparison.py       # Baseline model comparison (5 models)
│   ├── 08_mcnemar.py                # McNemar statistical significance test
│   ├── 09_shap_analysis.py          # SHAP explainability (global + individual)
│   ├── 10_cross_validation.py       # 10-fold stratified CV
│   ├── 11_summary.py                # Research summary report
│   ├── 12_ablation_study.py         # Ablation: impact of each component
│   ├── 13_smoteenn_analysis.py      # SMOTE-ENN impact analysis
│   ├── 14_model_benchmark.py        # CatBoost benchmark + Optuna tuning
│   ├── 15_calibration_analysis.py   # Brier Score, ECE, reliability diagrams
│   └── 16_error_analysis.py         # TP/TN/FP/FN error profiling
│
├── outputs/
│   ├── *.md                         # Tracked: analysis reports (markdown)
│   └── *.pdf / *.csv                # NOT tracked: generated plots & metrics
│
└── models/
    └── *.pkl                        # NOT tracked: large binary model files
```

---

## 📊 Dataset

**Source:** [UCI ML Repository — Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

| Property | Value |
|----------|-------|
| File | `data/raw/dataset.csv` |
| Total Rows (before filter) | ~4,424 |
| Features | 34 input columns |
| Target Column | `Target` |
| Classes (original) | `Dropout`, `Graduate`, `Enrolled` |
| Filtering | `Enrolled` removed (non-final status) |
| Binary Encoding | `Dropout → 1` (positive), `Graduate → 0` (negative) |

> **Note:** The raw dataset is NOT included in this repository. Download it from the UCI link above and place it at `data/raw/dataset.csv`.

---

## ⚙️ Pipeline Architecture

```
Raw CSV
  │
  ├─ [Phase 0] Binary Filtering      → Remove "Enrolled" rows
  ├─ [Phase 1] Target Encoding       → Dropout=1, Graduate=0
  ├─ [Phase 2] Stratified Split      → 80% train / 20% test (seed=42)
  ├─ [Phase 3] StandardScaler        → Fit on train, transform both
  ├─ [Phase 4] SMOTE-ENN             → Oversample + denoise training data
  ├─ [Phase 5] Optuna HPO            → 200 trials, 5-fold CV, maximize F1-Dropout
  ├─ [Phase 6] XGBoost Training      → Best params from Optuna
  │
  ├─ [Phase 7]  Evaluation           → F1, AUC, Balanced Acc, MCC, Kappa
  ├─ [Phase 8]  Baseline Comparison  → DT, LR, RF, XGB-base vs. Proposed
  ├─ [Phase 9]  McNemar Test         → Statistical significance (α=0.05)
  ├─ [Phase 10] Error Analysis       → Confusion matrix + TP/FP/FN/TN profiling
  ├─ [Phase 11] SHAP Analysis        → Beeswarm, bar, dependence, waterfall plots
  ├─ [Phase 12] 10-Fold CV           → Robustness validation
  ├─ [Phase 13] Computational Log    → Execution time per phase
  ├─ [Phase 14] Research Summary     → Answers to RQ1 & RQ2
  │
  ├─ [Extension 1] CatBoost Benchmark    → Default + Optuna-tuned, McNemar
  ├─ [Extension 2] Calibration Analysis  → Brier Score, ECE, reliability diagram
  └─ [Extension 3] Error Profile         → Predictive error profiling per group
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Create & activate virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Download the UCI dataset and place it at:

```
data/raw/dataset.csv
```

### 5. Run the pipeline

```bash
# Full core pipeline
python main.py

# Individual analysis scripts
python -m src.14_model_benchmark   # CatBoost benchmark
python -m src.15_calibration_analysis
python -m src.16_error_analysis
```

Or open `notebooks/exploration.ipynb` for the complete interactive pipeline.

---

## 📦 Dependencies

| Library | Purpose |
|---------|---------|
| `xgboost` | Primary classifier (XGBClassifier) |
| `catboost` | Benchmark classifier |
| `optuna` | Bayesian hyperparameter optimization |
| `shap` | Model explainability (TreeExplainer) |
| `imbalanced-learn` | SMOTE-ENN resampling |
| `scikit-learn` | Preprocessing, metrics, CV, baselines |
| `statsmodels` | McNemar statistical test |
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` / `seaborn` | Visualization |

Generate a `requirements.txt` with:

```bash
pip freeze > requirements.txt
```

---

## 📈 Key Results (V3 Final)

| Model | F1-Dropout | AUC-ROC | Balanced Acc |
|-------|-----------|---------|--------------|
| Decision Tree | — | — | — |
| Logistic Regression | **0.9107** | — | — |
| Random Forest | — | — | — |
| XGBoost Baseline | — | — | — |
| **XGBoost + SMOTE-ENN (Proposed)** | — | — | — |
| CatBoost Default | — | — | — |
| CatBoost Tuned | — | — | — |

> Detailed results are in `outputs/model_benchmark_results.csv` and `outputs/calibration_metrics.csv`.

---

## 🔍 Analysis Reports (Tracked)

| File | Description |
|------|-------------|
| [`outputs/shap_educational_implications.md`](outputs/shap_educational_implications.md) | Evidence-based SHAP interpretation for educational practitioners |
| [`outputs/misclassified_students_analysis.md`](outputs/misclassified_students_analysis.md) | FP/FN error profiling with demographic & academic breakdown |
| [`outputs/smoteenn_impact_analysis.md`](outputs/smoteenn_impact_analysis.md) | Quantitative impact of SMOTE-ENN on model performance |

---

## 🔒 Reproducibility

All experiments use `SEED = 42` throughout:

- `train_test_split(..., random_state=42)`
- `Optuna TPESampler(seed=42)`
- `XGBClassifier(..., random_state=42)`
- `CatBoostClassifier(..., random_seed=42)`
- `StratifiedKFold(..., random_state=42)`

---

## 📋 Research Constraints

1. **Do not** change target encoding (`Dropout=1`, `Graduate=0`)
2. **Maintain** binary classification (no multiclass)
3. **Preserve** SMOTE-ENN preprocessing pipeline
4. **Preserve** SHAP explainability pipeline
5. **Preserve** McNemar statistical testing procedures
6. **Prioritize** reproducibility and academic validity
7. All major code changes must be justified before implementation

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- Dataset: Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). *Predict Students' Dropout and Academic Success*. UCI Machine Learning Repository.
- SHAP: Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions*.
- Optuna: Akiba, T., et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework*.
