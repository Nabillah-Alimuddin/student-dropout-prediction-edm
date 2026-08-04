"""
11_summary.py — Phase 14: Final Research Summary.

Prints a comprehensive summary comparing Proposed Stacking Ensemble against
ALL benchmark models (Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost)
under identical experimental conditions.
"""

import time
import numpy as np
import pandas as pd

from src.utils import print_separator


def print_research_summary(eval_results, comparison_df, cv_results, top_features):
    """
    Phase 14: Comprehensive Research Summary comparing Proposed Stacking against ALL baselines.

    Args:
        eval_results (dict): Evaluation metrics and results.
        comparison_df (pd.DataFrame): Model comparison table across all models.
        cv_results (pd.DataFrame): Cross-validation results.
        top_features (list): Top SHAP features.
    """
    print_separator("PHASE 14: RESEARCH SUMMARY & OBJECTIVE COMPARISON")

    # ─── Full Model Comparison Table ─────────────────────────────────────
    print(f"\n  ═══ Fair Model Benchmark Comparison (SMOTE-ENN + Threshold Opt) ═══")
    if comparison_df is not None and not comparison_df.empty:
        print(comparison_df.to_string(index=False))

    # ─── Optimal threshold ────────────────────────────────────────────────
    if "optimal_threshold" in eval_results:
        print(f"\n  ═══ Optimal Threshold ═══")
        print(f"    Threshold: {eval_results['optimal_threshold']:.2f}")
        print(f"    F1 at optimal: {eval_results['optimal_f1']:.4f}")

    # ─── Top SHAP features ───────────────────────────────────────────────
    print(f"\n  ═══ Top 5 SHAP Features (Proposed Model) ═══")
    for i, feat in enumerate(top_features[:5], 1):
        print(f"    {i}. {feat}")

    # ─── Cross-validation robustness ──────────────────────────────────────
    if cv_results is not None and not cv_results.empty:
        print(f"\n  ═══ 10-Fold Cross-Validation Robustness (Proposed Model) ═══")
        for col in ["F1-Dropout", "AUC-ROC", "Balanced Acc", "MCC"]:
            vals = cv_results[col]
            print(f"    {col:<15}: {vals.mean():.4f} ± {vals.std():.4f} (Min: {vals.min():.4f}, Max: {vals.max():.4f})")

    print(f"\n  ✅ Pipeline execution successfully completed.")
