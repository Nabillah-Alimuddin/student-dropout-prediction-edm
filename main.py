"""
main.py — Entry point for the Student Dropout Prediction Pipeline.

Executes the full research pipeline in correct order:
  Phase 0-1:  Data Loading & Binary Filtering & Target Encoding
  Phase 2-3:  Stratified Split & Feature Scaling
  Phase 4:    SMOTE-ENN Resampling
  Phase 5:    Optuna Hyperparameter Optimization
  Phase 6:    Final Model Training
  Phase 7:    Model Evaluation (Metrics, Learning Curve, ROC, Threshold)
  Phase 8:    Baseline Model Comparison
  Phase 9:    McNemar Statistical Test
  Phase 10:   Confusion Matrix & Error Analysis
  Phase 11:   SHAP Explainability Analysis
  Phase 12:   10-Fold Cross-Validation
  Phase 13:   Computational Efficiency Summary
  Phase 14:   Research Summary

Usage:
    python main.py
"""

import sys
import os
import time
import warnings
import numpy as np

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
    """Execute the full student dropout prediction pipeline."""

    pipeline_start = time.time()
    reset_waktu_log()

    print("=" * 70)
    print("  STUDENT DROPOUT PREDICTION PIPELINE")
    print("  XGBoost + SMOTE-ENN + SHAP Explainability")
    print("  Version: V3 (Binary Classification — Final)")
    print("=" * 70)
    print(f"\n  Output directory: {OUTPUT_DIR}")
    print(f"  Model directory:  {MODEL_DIR}")
    print(f"  Random seed:      {SEED}")

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 0-1: Data Loading & Preparation
    # ═══════════════════════════════════════════════════════════════════════
    import importlib
    data_preparation = importlib.import_module("src.01_data_preparation")
    load_and_prepare_data = data_preparation.load_and_prepare_data
    X, y, df = load_and_prepare_data()

    # ═══════════════════════════════════════════════════════════════════════
    # ═══════════════════════════════════════════════════════════════════════
    # Phase 2-3: Preprocessing (Split + Scaling)
    # ═══════════════════════════════════════════════════════════════════════
    preprocessing = importlib.import_module("src.02_preprocessing")
    split_and_scale = preprocessing.split_and_scale
    X_train, X_test, X_train_sc, X_test_sc, y_train, y_test, scaler = split_and_scale(X, y)

    # ─── Save train-test split summary to CSV ───────────────────────────────
    import pandas as pd
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
    # Phase 4: SMOTE-ENN
    # ═══════════════════════════════════════════════════════════════════════
    smoteenn = importlib.import_module("src.03_smoteenn")
    apply_smoteenn = smoteenn.apply_smoteenn
    X_res, y_res = apply_smoteenn(X_train_sc, y_train)

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 5: Optuna Tuning
    # ═══════════════════════════════════════════════════════════════════════
    optuna_tuning = importlib.import_module("src.04_optuna_tuning")
    run_optuna_tuning = optuna_tuning.run_optuna_tuning
    # Pass original unscaled training data — StandardScaler and SMOTE-ENN are 
    # applied inside each CV fold via ImbPipeline, preventing scaling and SMOTE leakage.
    best_params, study = run_optuna_tuning(X_train, y_train)

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 6: Model Training
    # ═══════════════════════════════════════════════════════════════════════
    training = importlib.import_module("src.05_training")
    train_final_model = training.train_final_model
    # train_final_model menerima data resampled 100%:
    model = train_final_model(X_res, y_res, best_params)

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 7 + 10: Evaluation + Confusion Matrix
    # ═══════════════════════════════════════════════════════════════════════
    evaluation = importlib.import_module("src.06_evaluation")
    evaluate_model = evaluation.evaluate_model
    # X_train (unscaled) passed to perform leakage-free OOF threshold tuning
    eval_results = evaluate_model(
        model, X_train, y_train, X_test_sc, y_test, X_res, y_res
    )

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 8: Model Comparison
    # ═══════════════════════════════════════════════════════════════════════
    model_comparison = importlib.import_module("src.07_model_comparison")
    compare_models = model_comparison.compare_models
    # Pass the optimal threshold found for the proposed model
    comparison_df, models_dict, predictions_dict = compare_models(
        model, X_train_sc, y_train, X_test_sc, y_test,
        optimal_threshold=eval_results["optimal_threshold"]
    )

    # ─── Confusion Matrix: Logistic Regression ───────────────────────────────
    plot_logistic_regression_confusion_matrix = model_comparison.plot_logistic_regression_confusion_matrix
    plot_logistic_regression_confusion_matrix(models_dict, predictions_dict, y_test)

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 9: McNemar Test
    # ═══════════════════════════════════════════════════════════════════════
    mcnemar = importlib.import_module("src.08_mcnemar")
    run_mcnemar_tests = mcnemar.run_mcnemar_tests
    mcnemar_results = run_mcnemar_tests(y_test, predictions_dict)

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 11: SHAP Analysis
    # ═══════════════════════════════════════════════════════════════════════
    shap_analysis = importlib.import_module("src.09_shap_analysis")
    run_shap_analysis = shap_analysis.run_shap_analysis
    shap_values, top_features = run_shap_analysis(
        model, X_test_sc, y_test, eval_results["y_pred"]
    )

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 12: Cross-Validation
    # ═══════════════════════════════════════════════════════════════════════
    cross_validation = importlib.import_module("src.10_cross_validation")
    run_cross_validation = cross_validation.run_cross_validation
    # Pass original unscaled training data — StandardScaler and SMOTE-ENN are 
    # applied inside each CV fold via ImbPipeline for a completely leakage-free estimate.
    cv_results = run_cross_validation(X_train, y_train, best_params)

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 12.5: Ablation Study
    # ═══════════════════════════════════════════════════════════════════════
    ablation_study = importlib.import_module("src.12_ablation_study")
    run_ablation_study = ablation_study.run_ablation_study
    ablation_df = run_ablation_study(X_train, y_train, X_test, y_test, scaler, best_params)

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 12.6: SMOTE-ENN Impact Analysis
    # ═══════════════════════════════════════════════════════════════════════
    smoteenn_analysis = importlib.import_module("src.13_smoteenn_analysis")
    run_smoteenn_analysis = smoteenn_analysis.run_smoteenn_analysis
    # ─── Threshold Sensitivity Analysis (Recall‑first Early‑Warning) ────────────────────────
    run_threshold_analysis = smoteenn_analysis.threshold_sensitivity_analysis
    threshold_df, optimal_thr = run_threshold_analysis(model, X_test_sc, y_test)
    # Save threshold analysis results
    threshold_df.to_csv(f"{OUTPUT_DIR}/threshold_sensitivity_analysis.csv", index=False)
    print(f"  📄 Threshold analysis CSV saved to {OUTPUT_DIR}/threshold_sensitivity_analysis.csv")
    if optimal_thr is not None:
        print(f"  🎯 Optimal threshold (Recall ≥ 0.90): {optimal_thr}")
    else:
        print("  ⚠️ No threshold achieved Recall ≥ 0.90")
    

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 13: Timing Summary
    # ═══════════════════════════════════════════════════════════════════════
    summary = importlib.import_module("src.11_summary")
    print_timing_summary = summary.print_timing_summary
    print_research_summary = summary.print_research_summary
    timing_df = print_timing_summary()

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 14: Research Summary
    # ═══════════════════════════════════════════════════════════════════════
    print_research_summary(
        eval_results, comparison_df, mcnemar_results,
        cv_results, top_features
    )

    # ─── Total pipeline time ──────────────────────────────────────────────
    total_time = time.time() - pipeline_start
    print(f"\n  🏁 Total pipeline execution time: {total_time:.2f}s "
          f"({total_time/60:.2f}min)")


if __name__ == "__main__":
    main()
