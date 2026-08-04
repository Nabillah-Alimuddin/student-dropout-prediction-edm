"""
06_evaluation.py — Phase 7 & 10: Model Evaluation + Confusion Matrix.

Core metrics, learning curve, ROC curve, threshold optimization,
and detailed confusion matrix analysis.

Methodological notes:
- Threshold optimization uses Out-of-Fold (OOF) cross-validation on the training set
  to prevent evaluation-set leakage while utilizing all 2904 training samples.
- Learning curve uses ImbPipeline(StandardScaler + SMOTE-ENN + XGBoost) on the original
  X_train / y_train to prevent SMOTE and scaling leakage across CV folds.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
    cohen_kappa_score, matthews_corrcoef,
    classification_report, confusion_matrix,
    roc_curve, auc, make_scorer
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours

from src.config import (
    SEED, THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEP,
    OPTUNA_CV_FOLDS, SMOTE_K_NEIGHBORS, SMOTE_TARGET_RATIO,
    ENN_N_NEIGHBORS, ENN_KIND_SEL
)
from src.utils import catat_waktu, save_pdf, print_separator


def evaluate_model(model, X_train, y_train, X_test_sc, y_test, X_res, y_res,
                   X_val=None, y_val=None):
    """
    Full model evaluation: metrics, learning curve, ROC, threshold optimization,
    and confusion matrix analysis.

    Args:
        model: Trained XGBoost model.
        X_train: Original unscaled training features (for OOF threshold & learning curve).
        y_train: Original training target.
        X_test_sc: Scaled test features.
        y_test: Test target.
        X_res: Resampled training features (reference).
        y_res: Resampled training target (reference).
        X_val: Internal validation features (optional, for backward compatibility).
        y_val: Internal validation target (optional, for backward compatibility).

    Returns:
        results (dict): Dictionary of all evaluation results.
    """
    results = {}

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 7.3 — Threshold Optimization (Run first to optimize test predictions)
    # ═══════════════════════════════════════════════════════════════════════
    print_separator("PHASE 7.3: THRESHOLD OPTIMIZATION")
    mulai_thresh = time.time()

    thresholds = np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP, THRESHOLD_STEP)

    if X_val is not None and y_val is not None:
        y_val_proba = model.predict_proba(X_val)[:, 1]
        sweep_proba  = y_val_proba
        sweep_labels = np.array(y_val)
        print(f"  ✅ Threshold selection on internal validation set "
              f"({len(y_val)} samples) — test set not used.")
    else:
        # Out-Of-Fold (OOF) threshold selection using a 5-fold CV to prevent leakage
        print(f"  ✅ No validation set provided. Computing Out-Of-Fold (OOF) probabilities")
        print(f"     on training set ({len(y_train)} samples) via 5-Fold Stratified CV...")
        
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        
        # Check if model is StackingEnsemble or has base_models
        is_stacking = hasattr(model, 'base_models')
        
        if is_stacking:
            from src.stacking_training import StackingEnsemble
            clf_instance = StackingEnsemble(
                model.xgb_params,
                model.lgbm_params,
                model.catboost_params,
                seed=SEED
            )
        else:
            clf_instance = type(model)(**model.get_params())

        pipe_cv = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(
                k_neighbors=SMOTE_K_NEIGHBORS,
                random_state=SEED,
                sampling_strategy=SMOTE_TARGET_RATIO
            )),
            ('enn', EditedNearestNeighbours(
                n_neighbors=ENN_N_NEIGHBORS,
                kind_sel=ENN_KIND_SEL
            )),
            ('clf', clf_instance),
        ])
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        
        # n_jobs=1 for stacking to avoid nested parallelism
        cv_n_jobs = 1 if is_stacking else -1
        y_oof_proba = cross_val_predict(
            pipe_cv, X_train, y_train, cv=skf, method="predict_proba", n_jobs=cv_n_jobs
        )[:, 1]
        
        sweep_proba = y_oof_proba
        sweep_labels = np.array(y_train)

    f1_scores_val      = []
    recall_scores_val  = []
    prec_scores_val    = []

    for t in thresholds:
        y_t = (sweep_proba >= t).astype(int)
        f1_scores_val.append(f1_score(sweep_labels, y_t, pos_label=1, zero_division=0))
        recall_scores_val.append(recall_score(sweep_labels, y_t, pos_label=1, zero_division=0))
        prec_scores_val.append(precision_score(sweep_labels, y_t, pos_label=1, zero_division=0))

    # ── Select threshold on validation/OOF, apply once to test ─────────────
    best_idx = np.argmax(f1_scores_val)
    optimal_threshold = thresholds[best_idx]

    # Predict test probabilities
    y_proba = model.predict_proba(X_test_sc)[:, 1]
    
    # Report test-set metrics at the validation/OOF-selected threshold
    y_pred = (y_proba >= optimal_threshold).astype(int)
    optimal_f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

    # For the plot: show test-set curves alongside the validation-selected threshold
    f1_scores     = [f1_score(y_test, (y_proba >= t).astype(int), pos_label=1, zero_division=0)
                     for t in thresholds]
    recall_scores = [recall_score(y_test, (y_proba >= t).astype(int), pos_label=1, zero_division=0)
                     for t in thresholds]
    precision_scores = [precision_score(y_test, (y_proba >= t).astype(int), pos_label=1, zero_division=0)
                        for t in thresholds]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, f1_scores, label="F1-Score", color="#e74c3c", lw=2)
    ax.plot(thresholds, recall_scores, label="Recall", color="#3498db", lw=2)
    ax.plot(thresholds, precision_scores, label="Precision", color="#2ecc71", lw=2)
    ax.axvline(x=optimal_threshold, color="black", linestyle="--", alpha=0.7,
               label=f"Optimal: {optimal_threshold:.2f}")
    ax.axvline(x=0.50, color="gray", linestyle=":", alpha=0.5, label="Default: 0.50")
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Threshold Optimization — F1, Recall, Precision", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_pdf(fig, "v3_threshold_optimization.pdf")
    plt.close(fig)

    # Compare default vs optimal
    y_default = (y_proba >= 0.50).astype(int)

    print(f"  Optimal threshold: {optimal_threshold:.2f}")
    print(f"  F1 at default (0.50): {f1_score(y_test, y_default, pos_label=1):.4f}")
    print(f"  F1 at optimal ({optimal_threshold:.2f}): {optimal_f1:.4f}")

    results["optimal_threshold"] = optimal_threshold
    results["optimal_f1"] = optimal_f1
    results["y_pred"] = y_pred
    results["y_proba"] = y_proba

    catat_waktu("Threshold Optimization", mulai_thresh)

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 7.0 — Core Metrics (evaluated at optimal threshold)
    # ═══════════════════════════════════════════════════════════════════════
    print_separator("PHASE 7.0: CORE METRICS (At Optimized Threshold)")
    mulai = time.time()

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
        "F1-Score (Dropout)": f1_score(y_test, y_pred, pos_label=1),
        "Precision (Dropout)": precision_score(y_test, y_pred, pos_label=1),
        "Recall (Dropout)": recall_score(y_test, y_pred, pos_label=1),
        "AUC-ROC": roc_auc_score(y_test, y_proba),
        "Cohen's Kappa": cohen_kappa_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }

    print(f"\n  {'Metric':<25} {'Value':>10}")
    print(f"  {'─'*37}")
    for name, val in metrics.items():
        print(f"  {name:<25} {val:>10.4f}")

    print(f"\n  Classification Report (At Optimal Threshold = {optimal_threshold:.2f}):")
    print(classification_report(y_test, y_pred, target_names=["Graduate", "Dropout"]))

    results["metrics"] = metrics

    catat_waktu("Core Metrics", mulai)

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 7.1 — Learning Curve
    # ═══════════════════════════════════════════════════════════════════════
    print_separator("PHASE 7.1: LEARNING CURVE")
    mulai = time.time()

    f1_scorer = make_scorer(f1_score, pos_label=1)

    # ─── ImbPipeline: scaling + SMOTE-ENN inside fold CV — leakage-free ──
    is_stacking = hasattr(model, 'base_models')
    if is_stacking:
        from src.stacking_training import StackingEnsemble
        clf_instance_lc = StackingEnsemble(
            model.xgb_params,
            model.lgbm_params,
            model.catboost_params,
            seed=SEED
        )
        model_name = "Stacking (XGB+LGB+CB+LR)"
    else:
        clf_instance_lc = type(model)(**model.get_params())
        model_name = "XGBoost"

    lc_pipe = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(
            k_neighbors=SMOTE_K_NEIGHBORS,
            random_state=SEED,
            sampling_strategy=SMOTE_TARGET_RATIO
        )),
        ('enn', EditedNearestNeighbours(
            n_neighbors=ENN_N_NEIGHBORS,
            kind_sel=ENN_KIND_SEL
        )),
        ('clf', clf_instance_lc),
    ])

    print(f"  Using ImbPipeline (with StandardScaler) on original X_train to prevent leakage.")

    lc_n_jobs = 1 if is_stacking else -1
    train_sizes, train_scores, val_scores = learning_curve(
        lc_pipe, X_train, y_train,
        cv=OPTUNA_CV_FOLDS,
        scoring=f1_scorer,
        train_sizes=np.linspace(0.1, 1.0, 10),
        random_state=SEED,
        n_jobs=lc_n_jobs
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="orange")
    ax.plot(train_sizes, train_mean, "o-", color="blue", label="Training F1")
    ax.plot(train_sizes, val_mean, "o-", color="orange", label="Validation F1")
    ax.set_xlabel("Training Set Size", fontsize=12)
    ax.set_ylabel("F1-Score", fontsize=12)
    ax.set_title(f"Learning Curve — {model_name} + SMOTE-ENN", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Overfitting gap analysis
    gap = train_mean[-1] - val_mean[-1]
    status = "✅ No significant overfitting" if gap < 0.10 else "⚠️ Possible overfitting"
    ax.annotate(f"Gap: {gap:.4f} — {status}",
                xy=(0.5, 0.02), xycoords="axes fraction",
                fontsize=10, ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

    plt.tight_layout()
    save_pdf(fig, "v3_learning_curve.pdf")
    plt.close(fig)

    print(f"  Overfitting gap: {gap:.4f} — {status}")
    results["learning_curve_gap"] = gap

    catat_waktu("Learning Curve", mulai)

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 7.2 — AUC-ROC Curve
    # ═══════════════════════════════════════════════════════════════════════
    print_separator("PHASE 7.2: AUC-ROC CURVE")
    mulai = time.time()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#e74c3c", lw=2, label=f"{model_name} + SMOTE-ENN (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random Baseline")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Proposed Model", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_pdf(fig, "v3_roc_curve.pdf")
    plt.close(fig)

    print(f"  AUC-ROC: {roc_auc:.4f}")

    catat_waktu("ROC Curve", mulai)

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 10 — Confusion Matrix (Error Analysis)
    # ═══════════════════════════════════════════════════════════════════════
    print_separator("PHASE 10: CONFUSION MATRIX & ERROR ANALYSIS (At Optimized Threshold)")
    mulai = time.time()

    cm = confusion_matrix(y_test, y_pred)
    cm_norm = confusion_matrix(y_test, y_pred, normalize="true")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Absolute counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["Graduate", "Dropout"],
                yticklabels=["Graduate", "Dropout"])
    axes[0].set_title("Confusion Matrix (Counts)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # Normalized proportions
    sns.heatmap(cm_norm, annot=True, fmt=".3f", cmap="Blues", ax=axes[1],
                xticklabels=["Graduate", "Dropout"],
                yticklabels=["Graduate", "Dropout"])
    axes[1].set_title("Confusion Matrix (Normalized)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    save_pdf(fig, "v3_confusion_matrix.pdf")
    plt.close(fig)

    # Detailed breakdown
    tn, fp, fn, tp = cm.ravel()
    print(f"  TP (Correctly predicted Dropout): {tp}")
    print(f"  TN (Correctly predicted Graduate): {tn}")
    print(f"  FP (Graduate predicted as Dropout): {fp}")
    print(f"  FN (Dropout predicted as Graduate): {fn}")
    print(f"\n  Manual verification:")
    prec_manual = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec_manual = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_manual = 2 * prec_manual * rec_manual / (prec_manual + rec_manual) if (prec_manual + rec_manual) > 0 else 0
    print(f"    Precision: {prec_manual:.4f}")
    print(f"    Recall: {rec_manual:.4f}")
    print(f"    F1-Score: {f1_manual:.4f}")

    results["confusion_matrix"] = cm

    catat_waktu("Confusion Matrix", mulai)

    return results
