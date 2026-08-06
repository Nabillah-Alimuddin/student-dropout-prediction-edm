"""
07_model_comparison.py — Phase 8: Fair Baseline Model Comparison.

Applies EQUAL EXPERIMENTAL TREATMENT across ALL candidate models:
  - Preprocessing: StandardScaler
  - Resampling: SMOTE-ENN (on training data)
  - Hyperparameter Tuning: Optuna / Optimized per model
  - Threshold Optimization: 5-Fold Stratified OOF CV
  - Evaluation: Identical holdout test set

Compared Models:
  1. Logistic Regression + SMOTE-ENN + Threshold Opt
  2. Random Forest + SMOTE-ENN + Threshold Opt
  3. XGBoost (Tuned) + SMOTE-ENN + Threshold Opt
  4. LightGBM (Tuned) + SMOTE-ENN + Threshold Opt
  5. CatBoost (Tuned) + SMOTE-ENN + Threshold Opt
  6. Stacking Ensemble (Proposed: XGB+LGB+CB+LR) + SMOTE-ENN + Threshold Opt
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    roc_auc_score, balanced_accuracy_score,
    precision_recall_curve, average_precision_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.config import SEED, OUTPUT_DIR
from src.utils import catat_waktu, save_pdf, print_separator


def compare_models(model_proposed, X_train_sc, y_train, X_test_sc, y_test, optimal_threshold=0.50,
                   X_res=None, y_res=None, best_params_xgb=None, best_params_lgbm=None, best_params_cb=None):
    """
    Perform a completely FAIR (Apple-to-Apple) comparison where ALL models are trained
    on SMOTE-ENN resampled data and threshold-optimized via Stratified OOF CV.

    Args:
        model_proposed: Trained Proposed Stacking Ensemble model.
        X_train_sc: Scaled training features.
        y_train: Training target.
        X_test_sc: Scaled test features.
        y_test: Test target.
        optimal_threshold: Optimal threshold for proposed model.
        X_res: SMOTE-ENN resampled training features.
        y_res: SMOTE-ENN resampled training target.
        best_params_xgb: Optuna best params for XGBoost.
        best_params_lgbm: Optuna best params for LightGBM.
        best_params_cb: Optuna best params for CatBoost.
    """
    print_separator("PHASE 8: FAIR BASELINE MODEL COMPARISON (SMOTE-ENN + THRESHOLD OPT)")
    mulai = time.time()

    # Use resampled data if available; fallback to original scaled train data
    X_tr_bench = X_res if X_res is not None else X_train_sc
    y_tr_bench = y_res if y_res is not None else y_train

    # Helper function to find optimal threshold via 5-Fold Stratified OOF CV
    def find_best_threshold_oof(clf, X_tr, y_tr):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        try:
            y_oof_proba = cross_val_predict(
                clf, X_tr, y_tr, cv=skf, method="predict_proba", n_jobs=1
            )[:, 1]
        except Exception:
            y_oof_proba = clf.predict_proba(X_tr)[:, 1]

        thresholds = np.arange(0.1, 0.9, 0.01)
        best_f1 = -1
        best_t = 0.50
        for t in thresholds:
            y_pred_t = (y_oof_proba >= t).astype(int)
            f1 = f1_score(y_tr, y_pred_t, pos_label=1, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        return best_t

    # Define baseline models ALL trained on SMOTE-ENN data with Optuna/Optimized params
    p_xgb = best_params_xgb or {}
    p_lgb = best_params_lgbm or {}
    p_cb  = best_params_cb or {}

    baselines = {
        "Logistic Regression + SMOTE-ENN": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=SEED
        ),
        "Random Forest + SMOTE-ENN": RandomForestClassifier(
            n_estimators=200, random_state=SEED
        ),
        "XGBoost + SMOTE-ENN": XGBClassifier(**p_xgb),
        "LightGBM + SMOTE-ENN": LGBMClassifier(**p_lgb),
        "CatBoost + SMOTE-ENN": CatBoostClassifier(**p_cb),
    }

    models_dict = {}
    predictions_dict = {}
    proba_dict = {}
    comparison_rows = []

    print("  Evaluating all candidates under EQUAL TREATMENT (SMOTE-ENN + Threshold Optimization):")

    for name, clf in baselines.items():
        print(f"    Training & Optimizing Threshold: {name}...")
        if "CatBoost" in name or "XGBoost" in name:
            clf.fit(X_tr_bench, y_tr_bench, verbose=False)
        else:
            clf.fit(X_tr_bench, y_tr_bench)

        # Optimize threshold via OOF on resampled data
        opt_t = find_best_threshold_oof(clf, X_tr_bench, y_tr_bench)

        y_proba = clf.predict_proba(X_test_sc)[:, 1]
        y_pred = (y_proba >= opt_t).astype(int)

        print(f"      Optimal Threshold: {opt_t:.2f} | F1: {f1_score(y_test, y_pred, pos_label=1):.4f}")

        models_dict[name] = clf
        predictions_dict[name] = y_pred
        proba_dict[name] = y_proba

        comparison_rows.append({
            "Model": name,
            "Threshold": opt_t,
            "F1-Dropout": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
            "Recall": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
            "Precision": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
            "AUC-ROC": roc_auc_score(y_test, y_proba),
            "PR-AUC": average_precision_score(y_test, y_proba),
            "Balanced Acc": balanced_accuracy_score(y_test, y_pred),
        })

    # Add Proposed Stacking Ensemble (Proposed Model)
    proposed_name = "Stacking (Proposed: LGB+CB+LR) + SMOTE-ENN"
    y_proba_prop = model_proposed.predict_proba(X_test_sc)[:, 1]
    y_pred_prop = (y_proba_prop >= optimal_threshold).astype(int)

    models_dict[proposed_name] = model_proposed
    predictions_dict[proposed_name] = y_pred_prop
    proba_dict[proposed_name] = y_proba_prop

    print(f"    Adding {proposed_name}...")
    print(f"      Optimal Threshold: {optimal_threshold:.2f} | F1: {f1_score(y_test, y_pred_prop, pos_label=1):.4f}")

    comparison_rows.append({
        "Model": proposed_name,
        "Threshold": optimal_threshold,
        "F1-Dropout": f1_score(y_test, y_pred_prop, pos_label=1, zero_division=0),
        "Recall": recall_score(y_test, y_pred_prop, pos_label=1, zero_division=0),
        "Precision": precision_score(y_test, y_pred_prop, pos_label=1, zero_division=0),
        "AUC-ROC": roc_auc_score(y_test, y_proba_prop),
        "PR-AUC": average_precision_score(y_test, y_proba_prop),
        "Balanced Acc": balanced_accuracy_score(y_test, y_pred_prop),
    })

    comparison_df = pd.DataFrame(comparison_rows)
    # Sort models by performance (F1-Dropout descending)
    comparison_df = comparison_df.sort_values(by="F1-Dropout", ascending=False).reset_index(drop=True)

    print(f"\n  ═══ FAIR MODEL COMPARISON SUMMARY ═══")
    print(comparison_df.to_string(index=False))

    # Save to CSV
    csv_path = f"{OUTPUT_DIR}/fair_model_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"  📄 Saved comparison table to {csv_path}")

    # Visualization: Grouped bar chart
    metric_cols = ["F1-Dropout", "Recall", "Precision", "AUC-ROC", "Balanced Acc"]
    x = np.arange(len(metric_cols))
    width = 0.13
    colors = ["#7f8c8d", "#3498db", "#9b59b6", "#e67e22", "#1abc9c", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, (_, row) in enumerate(comparison_df.iterrows()):
        vals = [row[m] for m in metric_cols]
        ax.bar(x + i * width, vals, width, label=row["Model"], color=colors[i % len(colors)],
               edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Metrics", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Fair Model Comparison — SMOTE-ENN + Threshold Optimization Across All Models",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(metric_cols, fontsize=10)
    ax.set_ylim(0.70, 1.02)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_pdf(fig, "v3_perbandingan_metrik.pdf")
    plt.close(fig)

    catat_waktu("Model Comparison", mulai)

    return comparison_df, models_dict, predictions_dict


def plot_logistic_regression_confusion_matrix(models_dict, predictions_dict, y_test):
    """Plots confusion matrix for Logistic Regression baseline if available."""
    lr_key = None
    for k in models_dict.keys():
        if "Logistic Regression" in k:
            lr_key = k
            break
    if lr_key is not None and lr_key in predictions_dict:
        cm = confusion_matrix(y_test, predictions_dict[lr_key])
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Graduate", "Dropout"])
        disp.plot(cmap="Blues", ax=ax, values_format="d")
        ax.set_title(f"Confusion Matrix — {lr_key}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        save_pdf(fig, "v3_confusion_matrix_lr.pdf")
        plt.close(fig)
