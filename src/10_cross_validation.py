"""
10_cross_validation.py — Phase 12: 10-Fold Stratified Cross-Validation.

Robustness validation using a leakage-free approach:
- CV is applied to the ORIGINAL unscaled training set (X_train, y_train).
- StandardScaler and SMOTE-ENN are wrapped inside an ImbPipeline and executed WITHIN each
  training fold — synthetic samples NEVER appear in validation folds.
- Reports F1-Dropout, AUC-ROC, Balanced Accuracy, MCC per fold cleanly without NaN.
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, roc_auc_score, balanced_accuracy_score, matthews_corrcoef
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
    synthetic samples or scaled values from leaking into validation folds.

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
    is_stacking = ("xgb_params" in best_params) or ("xgb" in best_params and "lgb" in best_params)
    
    if is_stacking:
        from src.stacking_training import StackingEnsemble
        xgb_p = best_params.get("xgb", best_params.get("xgb_params", best_params))
        lgb_p = best_params.get("lgb", best_params.get("lgbm_params", {}))
        cat_p = best_params.get("cat", best_params.get("catboost_params", {}))
        print("  [CV] Using StackingEnsemble (XGB+LGB+CB+LR) inside ImbPipeline.")
        clf_cv = StackingEnsemble(xgb_p, lgb_p, cat_p, seed=SEED)
    else:
        clf_cv = XGBClassifier(**best_params)

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
        ('clf', clf_cv),
    ])

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr_f, X_val_f = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr_f, y_val_f = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Fit leakage-free pipeline on fold training data
        pipe.fit(X_tr_f, y_tr_f)

        y_val_pred = pipe.predict(X_val_f)
        y_val_proba = pipe.predict_proba(X_val_f)[:, 1]

        f1 = f1_score(y_val_f, y_val_pred, pos_label=1, zero_division=0)
        auc = roc_auc_score(y_val_f, y_val_proba)
        ba = balanced_accuracy_score(y_val_f, y_val_pred)
        mcc = matthews_corrcoef(y_val_f, y_val_pred)

        fold_metrics = {
            "Fold":         fold + 1,
            "F1-Dropout":   f1,
            "AUC-ROC":      auc,
            "Balanced Acc": ba,
            "MCC":          mcc,
        }
        fold_results.append(fold_metrics)
        print(f"  Fold {fold+1:2d}: F1={f1:.4f}  AUC={auc:.4f}  BA={ba:.4f}  MCC={mcc:.4f}")

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
