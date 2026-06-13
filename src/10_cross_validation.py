"""
10_cross_validation.py — Phase 12: 10-Fold Stratified Cross-Validation.

Robustness validation using a leakage-free approach:
- CV is applied to the ORIGINAL scaled training set (X_train_sc, y_train).
- SMOTE-ENN is wrapped inside an ImbPipeline and executed WITHIN each
  training fold — synthetic samples NEVER appear in validation folds.
- Reports F1-Dropout, AUC-ROC, Balanced Accuracy, MCC per fold.
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, roc_auc_score, balanced_accuracy_score, matthews_corrcoef,
    make_scorer
)
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours

from src.config import SEED, CV_FOLDS, SMOTE_K_NEIGHBORS, SMOTE_TARGET_RATIO, ENN_N_NEIGHBORS, ENN_KIND_SEL
from src.utils import catat_waktu, print_separator


def run_cross_validation(X_train, y_train, best_params):
    """
    Perform 10-fold stratified cross-validation on the ORIGINAL training set.

    StandardScaler and SMOTE-ENN are applied INSIDE each fold via ImbPipeline, preventing
    synthetic samples or scaled values from leaking into validation folds. This is the
    correct, publishable approach for robustness validation.

    Args:
        X_train (pd.DataFrame): Original training features (NOT scaled/resampled).
        y_train (pd.Series): Original training target (NOT resampled).
        best_params (dict): Best hyperparameters from Optuna.

    Returns:
        cv_results (pd.DataFrame): Per-fold and summary statistics.
    """
    print_separator("PHASE 12: 10-FOLD STRATIFIED CROSS-VALIDATION")
    mulai = time.time()

    print(f"  Input: original X_train ({X_train.shape[0]} samples) — NOT pre-scaled/resampled.")
    print(f"  StandardScaler & SMOTE-ENN applied INSIDE each fold via ImbPipeline (leakage-free).")

    # ─── Build leakage-free pipeline ─────────────────────────────────────
    pipe = ImbPipeline([
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
        ('clf', XGBClassifier(**best_params)),
    ])

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    scoring = {
        'f1':               make_scorer(f1_score, pos_label=1),
        'roc_auc':          'roc_auc',
        'balanced_accuracy': 'balanced_accuracy',
        'mcc':              make_scorer(matthews_corrcoef),
    }

    raw = cross_validate(
        pipe, X_train, y_train,
        cv=skf, scoring=scoring,
        return_train_score=False, n_jobs=-1
    )

    fold_results = []
    for fold in range(CV_FOLDS):
        fold_metrics = {
            "Fold":         fold + 1,
            "F1-Dropout":   raw['test_f1'][fold],
            "AUC-ROC":      raw['test_roc_auc'][fold],
            "Balanced Acc": raw['test_balanced_accuracy'][fold],
            "MCC":          raw['test_mcc'][fold],
        }
        fold_results.append(fold_metrics)
        print(f"  Fold {fold+1:2d}: F1={fold_metrics['F1-Dropout']:.4f}  "
              f"AUC={fold_metrics['AUC-ROC']:.4f}  "
              f"BA={fold_metrics['Balanced Acc']:.4f}  "
              f"MCC={fold_metrics['MCC']:.4f}")

    cv_results = pd.DataFrame(fold_results)

    # ─── Summary statistics ───────────────────────────────────────────────
    print(f"\n  {'Metric':<15} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'─'*47}")
    for col in ["F1-Dropout", "AUC-ROC", "Balanced Acc", "MCC"]:
        vals = cv_results[col]
        print(f"  {col:<15} {vals.mean():>8.4f} {vals.std():>8.4f} "
              f"{vals.min():>8.4f} {vals.max():>8.4f}")

    catat_waktu("Cross-Validation", mulai)

    return cv_results
