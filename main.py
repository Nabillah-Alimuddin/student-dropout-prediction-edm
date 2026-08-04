"""
main.py — Entry point for the Student Dropout Prediction Pipeline.

Executes the full research pipeline in 10 sequential phases:
  Phase 1:  Data Loading & Preparation
  Phase 2:  Preprocessing & Scaling
  Phase 3:  SMOTE-ENN Resampling
  Phase 4:  Optuna Hyperparameter Tuning
  Phase 5:  Stacking Ensemble Training
  Phase 6:  Model Evaluation & Threshold Optimization
  Phase 7:  Fair Baseline Model Comparison
  Phase 8:  SHAP Explainability Analysis
  Phase 9:  10-Fold Stratified Cross-Validation
  Phase 10: Final Research Summary

Usage:
    python main.py
"""

import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
import importlib

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Set random seeds globally
SEED = 42
np.random.seed(SEED)

# Set matplotlib backend for non-interactive environments
import matplotlib
matplotlib.use("Agg")

from src.config import SEED, OUTPUT_DIR, MODEL_DIR
from src.utils import reset_waktu_log, print_separator


def main():
    """Execute the full student dropout prediction pipeline in 10 clean phases."""

    pipeline_start = time.time()
    reset_waktu_log()

    print("=" * 70)
    print("  STUDENT DROPOUT PREDICTION PIPELINE")
    print("  Stacking (XGB+LGB+CB+LR) + SMOTE-ENN + SHAP Explainability")
    print("  Version: V3 (Binary Classification — Final)")
    print("=" * 70)
    print(f"\n  Output directory: {OUTPUT_DIR}")
    print(f"  Model directory:  {MODEL_DIR}")
    print(f"  Random seed:      {SEED}")

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 1: Data Loading & Preparation
    # ═══════════════════════════════════════════════════════════════════════
    data_preparation = importlib.import_module("src.01_data_preparation")
    load_and_prepare_data = data_preparation.load_and_prepare_data
    X, y, df = load_and_prepare_data()

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 2: Preprocessing (Split & Scaling)
    # ═══════════════════════════════════════════════════════════════════════
    preprocessing = importlib.import_module("src.02_preprocessing")
    split_and_scale = preprocessing.split_and_scale
    X_train, X_test, X_train_sc, X_test_sc, y_train, y_test, scaler = split_and_scale(X, y)

    # Save train-test split summary CSV
    split_summary = pd.DataFrame([
        {"Set": "Train",
         "Samples": int(X_train.shape[0]),
         "Features": int(X_train.shape[1]),
         "Dropout (1)": int((y_train == 1).sum()),
         "Graduate (0)": int((y_train == 0).sum()),
         "Dropout %": round((y_train == 1).mean() * 100, 2),
         "Graduate %": round((y_train == 0).mean() * 100, 2)},
        {"Set": "Test",
         "Samples": int(X_test.shape[0]),
         "Features": int(X_test.shape[1]),
         "Dropout (1)": int((y_test == 1).sum()),
         "Graduate (0)": int((y_test == 0).sum()),
         "Dropout %": round((y_test == 1).mean() * 100, 2),
         "Graduate %": round((y_test == 0).mean() * 100, 2)},
    ])
    split_summary_path = os.path.join(OUTPUT_DIR, "train_test_split_summary.csv")
    split_summary.to_csv(split_summary_path, index=False)
    print(f"  📄 Train-test split summary saved to {split_summary_path}")

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 3: SMOTE-ENN Resampling
    # ═══════════════════════════════════════════════════════════════════════
    smoteenn = importlib.import_module("src.03_smoteenn")
    apply_smoteenn = smoteenn.apply_smoteenn
    X_res, y_res = apply_smoteenn(X_train_sc, y_train)

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 4: Optuna Hyperparameter Optimization
    # ═══════════════════════════════════════════════════════════════════════
    optuna_tuning = importlib.import_module("src.04_optuna_tuning")
    run_optuna_tuning = optuna_tuning.run_optuna_tuning
    run_optuna_tuning_lgbm = optuna_tuning.run_optuna_tuning_lgbm
    run_optuna_tuning_catboost = optuna_tuning.run_optuna_tuning_catboost

    best_params_xgb, study_xgb = run_optuna_tuning(X_train, y_train)
    best_params_lgbm, study_lgbm = run_optuna_tuning_lgbm(X_train, y_train)
    best_params_cb, study_cb = run_optuna_tuning_catboost(X_train, y_train)

    best_params_stacking = {
        "xgb": best_params_xgb,
        "lgb": best_params_lgbm,
        "cat": best_params_cb
    }

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 5: Model Training (Stacking Ensemble)
    # ═══════════════════════════════════════════════════════════════════════
    training = importlib.import_module("src.05_training")
    train_final_model_stacking = training.train_final_model_stacking
    model = train_final_model_stacking(X_res, y_res, best_params_xgb, best_params_lgbm, best_params_cb)

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 6: Model Evaluation & Threshold Optimization
    # ═══════════════════════════════════════════════════════════════════════
    evaluation = importlib.import_module("src.06_evaluation")
    evaluate_model = evaluation.evaluate_model
    eval_results = evaluate_model(
        model, X_train, y_train, X_test_sc, y_test, X_res, y_res
    )

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 7: Fair Baseline Model Comparison
    # ═══════════════════════════════════════════════════════════════════════
    model_comparison = importlib.import_module("src.07_model_comparison")
    compare_models = model_comparison.compare_models
    comparison_df, models_dict, predictions_dict = compare_models(
        model, X_train_sc, y_train, X_test_sc, y_test,
        optimal_threshold=eval_results["optimal_threshold"],
        X_res=X_res, y_res=y_res,
        best_params_xgb=best_params_xgb,
        best_params_lgbm=best_params_lgbm,
        best_params_cb=best_params_cb
    )

    plot_logistic_regression_confusion_matrix = model_comparison.plot_logistic_regression_confusion_matrix
    plot_logistic_regression_confusion_matrix(models_dict, predictions_dict, y_test)

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 8: SHAP Explainability Analysis
    # ═══════════════════════════════════════════════════════════════════════
    shap_analysis = importlib.import_module("src.09_shap_analysis")
    run_shap_analysis = shap_analysis.run_shap_analysis
    shap_values, top_features = run_shap_analysis(
        model, X_test_sc, y_test, eval_results["y_pred"]
    )

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 9: 10-Fold Stratified Cross-Validation
    # ═══════════════════════════════════════════════════════════════════════
    cross_validation = importlib.import_module("src.10_cross_validation")
    run_cross_validation = cross_validation.run_cross_validation
    cv_results = run_cross_validation(X_train, y_train, best_params_stacking)

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 10: Final Research Summary
    # ═══════════════════════════════════════════════════════════════════════
    summary = importlib.import_module("src.11_summary")
    print_research_summary = summary.print_research_summary
    print_research_summary(
        eval_results, comparison_df,
        cv_results, top_features
    )

    # ─── Total pipeline execution time ────────────────────────────────────
    total_time = time.time() - pipeline_start
    print(f"\n  🏁 Total pipeline execution time: {total_time:.2f}s "
          f"({total_time/60:.2f}min)")


if __name__ == "__main__":
    main()
