"""
10_cross_validation.py — Phase 12: 10-Fold Stratified Cross-Validation.

Robustness validation using a leakage-free approach:
- CV is applied to the ORIGINAL unscaled training set (X_train, y_train).
- StandardScaler and SMOTE-ENN are wrapped inside an ImbPipeline and executed WITHIN each
  training fold — synthetic samples NEVER appear in validation folds.
- Reports F1-Dropout, AUC-ROC, Balanced Accuracy, MCC per fold cleanly without NaN.
"""

import time
import os
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

from src.config import (
    SEED, CV_FOLDS, SMOTE_K_NEIGHBORS, SMOTE_TARGET_RATIO,
    ENN_N_NEIGHBORS, ENN_KIND_SEL, THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEP
)
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
        use_xgb = best_params.get("use_xgb", True)
        base_desc = "XGB+LGB+CB+LR" if use_xgb else "LGB+CB+LR"
        print(f"  [CV] Using StackingEnsemble ({base_desc}) with internal SMOTE-ENN.")
        clf_cv = StackingEnsemble(xgb_p, lgb_p, cat_p, seed=SEED, apply_resampling=True, use_xgb=use_xgb)
        
        # StackingEnsemble handles SMOTE-ENN internally; pipeline only scales
        pipe = ImbPipeline([
            ('scaler', StandardScaler()),
            ('clf', clf_cv),
        ])
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
    oof_proba = np.zeros(X_train.shape[0])

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr_f, X_val_f = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr_f, y_val_f = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Fit leakage-free pipeline on fold training data
        pipe.fit(X_tr_f, y_tr_f)

        y_val_proba = pipe.predict_proba(X_val_f)[:, 1]
        oof_proba[val_idx] = y_val_proba

        print(f"  Fold {fold+1:2d} fitted.")
        if is_stacking:
            clf_cv_fitted = pipe.named_steps['clf']
            meta_coefs = clf_cv_fitted.meta_learner_.coef_[0]
            if clf_cv_fitted.use_xgb:
                print(f"         Meta weights -> XGB: {meta_coefs[0]:.4f}, LGB: {meta_coefs[1]:.4f}, CB: {meta_coefs[2]:.4f}, LR: {meta_coefs[3]:.4f}")
            else:
                print(f"         Meta weights -> LGB: {meta_coefs[0]:.4f}, CB: {meta_coefs[1]:.4f}, LR: {meta_coefs[2]:.4f}")

    # Sweep thresholds on OOF predictions to find the optimal threshold
    thresholds = np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP, THRESHOLD_STEP)
    best_f1 = -1
    opt_t = 0.50
    for t in thresholds:
        y_pred_t = (oof_proba >= t).astype(int)
        f1_t = f1_score(y_train, y_pred_t, pos_label=1, zero_division=0)
        if f1_t > best_f1:
            best_f1 = f1_t
            opt_t = t

    print(f"\n  [CV] Optimal OOF Threshold Selected: {opt_t:.2f} (Overall OOF F1-Score: {best_f1:.4f})")
    print(f"  [CV] Re-evaluating fold metrics at optimal threshold ({opt_t:.2f}):")

    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        y_val_f = y_train.iloc[val_idx]
        y_val_proba = oof_proba[val_idx]
        y_val_pred = (y_val_proba >= opt_t).astype(int)

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

    # Save CV results to CSV
    from src.config import OUTPUT_DIR
    cv_csv_path = os.path.join(OUTPUT_DIR, "cross_validation_results.csv")
    cv_results.to_csv(cv_csv_path, index=False)
    print(f"  📄 Saved CV results to {cv_csv_path}")

    catat_waktu("Cross-Validation", mulai)

    return cv_results
