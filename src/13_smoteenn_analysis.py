"""
13_smoteenn_analysis.py — Phase 12.6: SMOTE-ENN Impact Investigation.

Compares class distributions and model performance (metrics and confusion matrices)
between XGBoost Baseline (no resampling) and XGBoost + SMOTE-ENN (resampled)
to analyze the exact impact of SMOTE-ENN.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
)
from xgboost import XGBClassifier

from src.config import SEED, OUTPUT_DIR
from src.utils import catat_waktu, save_pdf, print_separator


def run_smoteenn_analysis(X_train_sc, y_train, X_res, y_res, X_test_sc, y_test):
    """
    Analyzes the impact of SMOTE-ENN resampling on class distribution and model performance.

    Args:
        X_train_sc (pd.DataFrame): Scaled training features (no resampling).
        y_train (pd.Series): Training target (no resampling).
        X_res (pd.DataFrame): Resampled training features.
        y_res (pd.Series): Resampled training target.
        X_test_sc (pd.DataFrame): Scaled test features.
        y_test (pd.Series): Test target.
    """
    print_separator("PHASE 12.6: SMOTE-ENN IMPACT ANALYSIS")
    mulai = time.time()

    # ─── 1. Class Distribution Comparison ─────────────────────────────────
    dist_before = y_train.value_counts()
    dist_after = y_res.value_counts()

    print("  Class Distribution Comparison:")
    print(f"    Before Resampling:")
    print(f"      Graduate (0): {dist_before.get(0, 0)} ({dist_before.get(0, 0)/len(y_train)*100:.1f}%)")
    print(f"      Dropout  (1): {dist_before.get(1, 0)} ({dist_before.get(1, 0)/len(y_train)*100:.1f}%)")
    print(f"      Imbalance Ratio: {dist_before.get(1, 0)/dist_before.get(0, 0):.4f}")
    
    print(f"    After SMOTE-ENN:")
    print(f"      Graduate (0): {dist_after.get(0, 0)} ({dist_after.get(0, 0)/len(y_res)*100:.1f}%)")
    print(f"      Dropout  (1): {dist_after.get(1, 0)} ({dist_after.get(1, 0)/len(y_res)*100:.1f}%)")
    print(f"      Imbalance Ratio: {dist_after.get(1, 0)/dist_after.get(0, 0):.4f}")
    print()

    # ─── 2. Model Training & Evaluation ───────────────────────────────────
    # XGBoost Baseline (no resampling, threshold = 0.5)
    model_baseline = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=SEED, use_label_encoder=False,
        eval_metric="logloss"
    )
    model_baseline.fit(X_train_sc, y_train, verbose=False)
    y_pred_base = model_baseline.predict(X_test_sc)
    
    # XGBoost + SMOTE-ENN (resampled, threshold = 0.5)
    model_resampled = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=SEED, use_label_encoder=False,
        eval_metric="logloss"
    )
    model_resampled.fit(X_res, y_res, verbose=False)
    y_pred_res = model_resampled.predict(X_test_sc)

    # Calculate metrics
    metrics_base = {
        "Precision (Dropout)": precision_score(y_test, y_pred_base, pos_label=1),
        "Recall (Dropout)": recall_score(y_test, y_pred_base, pos_label=1),
        "F1-Score (Dropout)": f1_score(y_test, y_pred_base, pos_label=1),
        "MCC": matthews_corrcoef(y_test, y_pred_base)
    }

    metrics_res = {
        "Precision (Dropout)": precision_score(y_test, y_pred_res, pos_label=1),
        "Recall (Dropout)": recall_score(y_test, y_pred_res, pos_label=1),
        "F1-Score (Dropout)": f1_score(y_test, y_pred_res, pos_label=1),
        "MCC": matthews_corrcoef(y_test, y_pred_res)
    }

    comparison_df = pd.DataFrame([metrics_base, metrics_res], index=["XGBoost Baseline", "XGBoost + SMOTE-ENN"])
    print("  Performance Comparison (Threshold = 0.5):")
    print(comparison_df.to_string())
    print()

    # Save comparison metrics
    comparison_df.to_csv(f"{OUTPUT_DIR}/smoteenn_impact_metrics.csv")

    # ─── 3. Side-by-Side Confusion Matrices ───────────────────────────────
    cm_base = confusion_matrix(y_test, y_pred_base)
    cm_res = confusion_matrix(y_test, y_pred_res)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Baseline CM
    sns.heatmap(cm_base, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0],
                annot_kws={"size": 14, "weight": "bold"})
    axes[0].set_title("XGBoost Baseline\n(No Resampling, Imbalanced)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Predicted Label", fontsize=11)
    axes[0].set_ylabel("True Label", fontsize=11)
    axes[0].set_xticklabels(["Graduate (0)", "Dropout (1)"])
    axes[0].set_yticklabels(["Graduate (0)", "Dropout (1)"])

    # SMOTE-ENN CM
    sns.heatmap(cm_res, annot=True, fmt="d", cmap="Oranges", cbar=False, ax=axes[1],
                annot_kws={"size": 14, "weight": "bold"})
    axes[1].set_title("XGBoost + SMOTE-ENN\n(With Resampling, Balanced)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Predicted Label", fontsize=11)
    axes[1].set_ylabel("True Label", fontsize=11)
    axes[1].set_xticklabels(["Graduate (0)", "Dropout (1)"])
    axes[1].set_yticklabels(["Graduate (0)", "Dropout (1)"])

    plt.tight_layout()
    plot_path = "v3_smoteenn_confusion_matrices.pdf"
    save_pdf(fig, plot_path)
    plt.close(fig)
    print(f"  📄 Confusion matrix comparison saved to {OUTPUT_DIR}/{plot_path}")

    # ─── 4. Detailed Scientific Analysis/Discussion ──────────────────────
    print("\n  🔍 Scientific Analysis of SMOTE-ENN Impact:")
    
    # Analyze changes
    delta_recall = metrics_res["Recall (Dropout)"] - metrics_base["Recall (Dropout)"]
    delta_prec = metrics_res["Precision (Dropout)"] - metrics_base["Precision (Dropout)"]
    delta_f1 = metrics_res["F1-Score (Dropout)"] - metrics_base["F1-Score (Dropout)"]
    
    analysis_text = []
    analysis_text.append("### Impact of SMOTE-ENN on Student Dropout Prediction\n")
    analysis_text.append(f"1. **Class Balance shift**: SMOTE-ENN resampled the training set class ratio (Dropout:Graduate) from {dist_before.get(1, 0)/dist_before.get(0, 0):.3f} to {dist_after.get(1, 0)/dist_after.get(0, 0):.3f}. This allows the XGBoost classifier to learn from a balanced representation of both outcomes, mitigating the majority-class bias.\n")
    
    if delta_recall > 0:
        analysis_text.append(f"2. **Recall Improvement**: Applying SMOTE-ENN increased the Recall of the Dropout class by {delta_recall*100:+.2f}%. This means the model successfully identified more actual dropout cases (reducing False Negatives from {cm_base[1, 0]} to {cm_res[1, 0]}), which is critical for early warning intervention.\n")
    else:
        analysis_text.append(f"2. **Recall Change**: Recall of the Dropout class changed by {delta_recall*100:+.2f}% (False Negatives: Baseline={cm_base[1, 0]}, SMOTE-ENN={cm_res[1, 0]}).\n")
        
    if delta_prec < 0:
        analysis_text.append(f"3. **Precision Trade-off**: As expected with synthetic oversampling, there is a minor trade-off in Precision of {delta_prec*100:+.2f}%, leading to an increase in False Positives from {cm_base[0, 1]} to {cm_res[0, 1]}. In educational institutions, a False Positive (misidentifying a graduate student as risk of dropout) is generally preferred over a False Negative (missing a dropout student entirely).\n")
    else:
        analysis_text.append(f"3. **Precision Change**: Precision of the Dropout class changed by {delta_prec*100:+.2f}% (False Positives: Baseline={cm_base[0, 1]}, SMOTE-ENN={cm_res[0, 1]}).\n")
        
    analysis_text.append(f"4. **Overall Metric (F1-Score & MCC)**: The overall F1-Score changed by {delta_f1*100:+.2f}%. This indicates whether the harmonic mean of precision and recall improves with resampling. Combined with the Matthews Correlation Coefficient (MCC) change, we can assess if the resampling has a net positive effect on the classifier's discriminative ability across both classes.\n")

    analysis_markdown = "".join(analysis_text)
    print("".join(["    " + line for line in analysis_markdown.splitlines(keepends=True)]))

    # Save analysis as md file
    with open(f"{OUTPUT_DIR}/smoteenn_impact_analysis.md", "w") as f:
        f.write(analysis_markdown)
    print(f"  💾 Analysis report saved to {OUTPUT_DIR}/smoteenn_impact_analysis.md")

    catat_waktu("SMOTE-ENN Impact Analysis", mulai)
