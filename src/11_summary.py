"""
11_summary.py — Phase 13 & 14: Computational Efficiency + Research Summary.

Produces timing summary table and answers research questions RQ1 and RQ2.
"""

import time
import numpy as np
import pandas as pd

from src.utils import get_waktu_log, catat_waktu, print_separator


def print_timing_summary():
    """
    Phase 13: Print computational efficiency summary.

    Returns:
        timing_df (pd.DataFrame): Timing log for all phases.
    """
    print_separator("PHASE 13: COMPUTATIONAL EFFICIENCY")

    waktu_log = get_waktu_log()
    if not waktu_log:
        print("  No timing data recorded.")
        return pd.DataFrame()

    timing_df = pd.DataFrame(waktu_log)

    print(f"\n  {'Phase':<30} {'Seconds':>10} {'Minutes':>10}")
    print(f"  {'─'*52}")
    for _, row in timing_df.iterrows():
        print(f"  {row['Fase']:<30} {row['Detik']:>10.2f} {row['Menit']:>10.4f}")

    total_sec = timing_df["Detik"].sum()
    total_min = timing_df["Menit"].sum()
    print(f"  {'─'*52}")
    print(f"  {'TOTAL':<30} {total_sec:>10.2f} {total_min:>10.4f}")

    return timing_df


def print_research_summary(eval_results, comparison_df, mcnemar_results,
                           cv_results, top_features):
    """
    Phase 14: Research summary answering RQ1 and RQ2.

    Args:
        eval_results (dict): Evaluation metrics and results.
        comparison_df (pd.DataFrame): Model comparison table.
        mcnemar_results (pd.DataFrame): McNemar test results.
        cv_results (pd.DataFrame): Cross-validation results.
        top_features (list): Top SHAP features.
    """
    print_separator("PHASE 14: RESEARCH SUMMARY")

    # ─── RQ1: Can XGBoost + SMOTE-ENN outperform baselines? ──────────────
    print(f"\n  ═══ RQ1: Performance Comparison ═══")
    proposed = comparison_df[comparison_df["Model"] == "XGBoost + SMOTE-ENN (Proposed)"]
    baseline_xgb = comparison_df[comparison_df["Model"] == "XGBoost (Baseline)"]

    if not proposed.empty and not baseline_xgb.empty:
        for metric in ["F1-Dropout", "Recall", "Precision", "AUC-ROC", "Balanced Acc"]:
            prop_val = proposed[metric].values[0]
            base_val = baseline_xgb[metric].values[0]
            delta = prop_val - base_val
            direction = "↑" if delta > 0 else "↓" if delta < 0 else "="
            print(f"    {metric:<15}: Proposed={prop_val:.4f}  "
                  f"Baseline={base_val:.4f}  Δ={delta:+.4f} {direction}")

    # ─── RQ2: Statistical significance ────────────────────────────────────
    print(f"\n  ═══ RQ2: Statistical Significance ═══")
    if mcnemar_results is not None and not mcnemar_results.empty:
        for _, row in mcnemar_results.iterrows():
            print(f"    {row['Comparison']}: p={row['p-value']:.6f} → "
                  f"{row[mcnemar_results.columns[-1]]}")

    # ─── Optimal threshold ────────────────────────────────────────────────
    if "optimal_threshold" in eval_results:
        print(f"\n  ═══ Optimal Threshold ═══")
        print(f"    Threshold: {eval_results['optimal_threshold']:.2f}")
        print(f"    F1 at optimal: {eval_results['optimal_f1']:.4f}")

    # ─── Top SHAP features ───────────────────────────────────────────────
    print(f"\n  ═══ Top 5 SHAP Features ═══")
    for i, feat in enumerate(top_features[:5], 1):
        print(f"    {i}. {feat}")

    # ─── Cross-validation robustness ──────────────────────────────────────
    if cv_results is not None and not cv_results.empty:
        print(f"\n  ═══ Cross-Validation Robustness ═══")
        for col in ["F1-Dropout", "AUC-ROC", "Balanced Acc", "MCC"]:
            vals = cv_results[col]
            print(f"    {col}: {vals.mean():.4f} ± {vals.std():.4f}")

    # ─── Output files ────────────────────────────────────────────────────
    print(f"\n  ═══ Generated PDF & Analysis Outputs ═══")
    pdf_files = [
        "v3_distribusi_binary.pdf",
        "v3_learning_curve.pdf",
        "v3_roc_curve.pdf",
        "v3_threshold_optimization.pdf",
        "v3_perbandingan_metrik.pdf",
        "v3_precision_recall_curve.pdf",
        "v3_confusion_matrix.pdf",
        "v3_shap_beeswarm.pdf",
        "v3_shap_global_bar.pdf",
        "v3_shap_dep_*.pdf (top 3 features)",
        "v3_shap_waterfall_dropout.pdf",
        "v3_shap_waterfall_graduate.pdf",
        "v3_ablation_study.pdf",
        "v3_smoteenn_confusion_matrices.pdf",
        "ablation_results.csv",
        "smoteenn_impact_metrics.csv",
        "smoteenn_impact_analysis.md"
    ]
    for f in pdf_files:
        print(f"    📄 {f}")

    print(f"\n  ✅ Pipeline complete.")
