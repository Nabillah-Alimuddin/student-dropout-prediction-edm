"""
07_model_comparison.py — Phase 8: Baseline Model Comparison.

Compares 5 models on the same test set:
  1. Decision Tree (class_weight='balanced')
  2. Logistic Regression (class_weight='balanced', max_iter=1000)
  3. Random Forest (class_weight='balanced', n_estimators=100)
  4. XGBoost Baseline (default: n_estimators=100, max_depth=6, lr=0.1)
  5. XGBoost + SMOTE-ENN (proposed, Optuna-tuned)
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    roc_auc_score, balanced_accuracy_score,
    precision_recall_curve, average_precision_score
)

from src.config import SEED
from src.utils import catat_waktu, save_pdf, print_separator


def compare_models(model_proposed, X_train_sc, y_train, X_test_sc, y_test, optimal_threshold=0.50):
    """
    Train 4 baseline models and compare with the proposed model.
    All models are evaluated at their respective optimal thresholds (tuned via Stratified OOF CV)
    for a fair, apple-to-apple comparison.

    Args:
        model_proposed: Trained proposed XGBoost model.
        X_train_sc: Scaled training features (original, not resampled).
        y_train: Training target (original, not resampled).
        X_test_sc: Scaled test features.
        y_test: Test target.
        optimal_threshold: The optimal probability threshold for proposed model (already optimized).

    Returns:
        comparison_df (pd.DataFrame): Comparison metrics for all models.
        models_dict (dict): Dictionary of {name: trained_model}.
        predictions_dict (dict): Dictionary of {name: y_pred}.
    """
    print_separator("PHASE 8: BASELINE MODEL COMPARISON")
    mulai = time.time()

    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    # Helper function to find optimal threshold using Stratified OOF CV
    def find_best_threshold_oof(clf, X, y):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        try:
            y_oof_proba = cross_val_predict(
                clf, X, y, cv=skf, method="predict_proba", n_jobs=-1
            )[:, 1]
        except Exception:
            y_oof_proba = cross_val_predict(
                clf, X, y, cv=skf, method="predict_proba"
            )[:, 1]
        
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_f1 = -1
        best_t = 0.50
        for t in thresholds:
            y_pred_t = (y_oof_proba >= t).astype(int)
            f1 = f1_score(y, y_pred_t, pos_label=1, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        return best_t

    # ─── Define baseline models ───────────────────────────────────────────
    baselines = {
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=SEED
        ),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=SEED
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced", n_estimators=100, random_state=SEED
        ),
        "XGBoost (Baseline)": XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=SEED, use_label_encoder=False,
            eval_metric="logloss"
        ),
    }

    # ─── Train baselines on original scaled data ──────────────────────────
    models_dict = {}
    predictions_dict = {}
    proba_dict = {}
    comparison_rows = []

    for name, clf in baselines.items():
        print(f"  Training & Optimizing Threshold: {name}...")
        clf.fit(X_train_sc, y_train)
        
        # Optimize threshold using Stratified OOF CV
        opt_t = find_best_threshold_oof(clf, X_train_sc, y_train)
        
        y_proba = clf.predict_proba(X_test_sc)[:, 1]
        y_pred = (y_proba >= opt_t).astype(int)

        print(f"    Optimal Threshold: {opt_t:.2f}")

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

    # ─── Add proposed model ───────────────────────────────────────────────
    y_proba_prop = model_proposed.predict_proba(X_test_sc)[:, 1]
    y_pred_prop = (y_proba_prop >= optimal_threshold).astype(int)
    models_dict["XGBoost + SMOTE-ENN (Proposed)"] = model_proposed
    predictions_dict["XGBoost + SMOTE-ENN (Proposed)"] = y_pred_prop
    proba_dict["XGBoost + SMOTE-ENN (Proposed)"] = y_proba_prop

    print(f"  Adding Proposed Model (XGBoost + SMOTE-ENN)...")
    print(f"    Optimal Threshold: {optimal_threshold:.2f}")

    comparison_rows.append({
        "Model": "XGBoost + SMOTE-ENN (Proposed)",
        "Threshold": optimal_threshold,
        "F1-Dropout": f1_score(y_test, y_pred_prop, pos_label=1, zero_division=0),
        "Recall": recall_score(y_test, y_pred_prop, pos_label=1, zero_division=0),
        "Precision": precision_score(y_test, y_pred_prop, pos_label=1, zero_division=0),
        "AUC-ROC": roc_auc_score(y_test, y_proba_prop),
        "PR-AUC": average_precision_score(y_test, y_proba_prop),
        "Balanced Acc": balanced_accuracy_score(y_test, y_pred_prop),
    })

    comparison_df = pd.DataFrame(comparison_rows)

    print(f"\n  Model Comparison Results (All Optimized):")
    print(comparison_df.to_string(index=False))

    # ─── Visualization: Grouped bar chart ─────────────────────────────────
    metric_cols = ["F1-Dropout", "Recall", "Precision", "AUC-ROC", "Balanced Acc", "PR-AUC"]
    x = np.arange(len(metric_cols))
    width = 0.15
    colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, (_, row) in enumerate(comparison_df.iterrows()):
        vals = [row[m] for m in metric_cols]
        ax.bar(x + i * width, vals, width, label=row["Model"], color=colors[i],
               edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Metrics", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — All Metrics", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(metric_cols, fontsize=10)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_pdf(fig, "v3_perbandingan_metrik.pdf")
    plt.close(fig)

    # ─── Visualization: Precision-Recall Curves ───────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    for name, y_proba in proba_dict.items():
        prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        ax.plot(rec_vals, prec_vals, lw=2, label=f"{name} (AP={ap:.4f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve — All Models", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_pdf(fig, "v3_precision_recall_curve.pdf")
    plt.close(fig)

    catat_waktu("Model Comparison", mulai)

    return comparison_df, models_dict, predictions_dict


def plot_logistic_regression_confusion_matrix(models_dict, predictions_dict, y_test):
    """
    Generate and save the Logistic Regression confusion matrix as a PDF.

    Args:
        models_dict (dict): Dictionary of {name: trained_model} from compare_models.
        predictions_dict (dict): Dictionary of {name: y_pred} from compare_models.
        y_test: True test labels.

    Returns:
        cm (np.array): The 2x2 confusion matrix.
    """
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, classification_report

    print_separator("LOGISTIC REGRESSION — CONFUSION MATRIX")

    name = "Logistic Regression"
    if name not in predictions_dict:
        print(f"  ⚠️  '{name}' not found in predictions_dict. Skipping.")
        return None

    y_pred_lr = predictions_dict[name]

    cm = confusion_matrix(y_test, y_pred_lr)
    cm_norm = confusion_matrix(y_test, y_pred_lr, normalize="true")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Logistic Regression — Confusion Matrix", fontsize=15, fontweight="bold", y=1.01)

    # Absolute counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", ax=axes[0],
                xticklabels=["Graduate", "Dropout"],
                yticklabels=["Graduate", "Dropout"])
    axes[0].set_title("Confusion Matrix (Counts)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # Normalized proportions
    sns.heatmap(cm_norm, annot=True, fmt=".3f", cmap="Oranges", ax=axes[1],
                xticklabels=["Graduate", "Dropout"],
                yticklabels=["Graduate", "Dropout"])
    axes[1].set_title("Confusion Matrix (Normalized)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    save_pdf(fig, "v3_confusion_matrix_lr.pdf")
    plt.close(fig)

    # Print breakdown
    tn, fp, fn, tp = cm.ravel()
    print(f"  TP (Correctly predicted Dropout)  : {tp}")
    print(f"  TN (Correctly predicted Graduate) : {tn}")
    print(f"  FP (Graduate predicted as Dropout): {fp}")
    print(f"  FN (Dropout predicted as Graduate): {fn}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred_lr, target_names=["Graduate", "Dropout"]))
    print(f"  📄 Saved: outputs/v3_confusion_matrix_lr.pdf")

    return cm
