"""
stacking_training.py — Level 0 Base Learners (XGB+LGB+CB+LR) + Level 1 Meta Learner.

Architecture:
    Level 0: XGBoost, LightGBM, CatBoost, LogisticRegression (diverse ensemble)
    Level 1: LogisticRegression meta-learner (trained on OOF probabilities)
    
Rationale for including LR as base learner:
    Dataset features show near-linear monotonic relationships (SHAP correlations
    -0.98, -0.97, -0.91 for top features). LR captures this linear signal
    optimally. Combined with tree-based learners that capture interactions,
    the ensemble is robust to both linear and nonlinear patterns.

Leakage Prevention (V3.1 fix):
    When apply_resampling=True, SMOTE-ENN is applied INSIDE each OOF fold
    so that synthetic SMOTE neighbors never leak across fold boundaries.
    This prevents optimistically biased OOF predictions and ensures that
    meta-learner weights and threshold selection are trustworthy.
"""

import time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from src.config import (
    SEED, SMOTE_K_NEIGHBORS, SMOTE_TARGET_RATIO, ENN_N_NEIGHBORS, ENN_KIND_SEL
)
from src.utils import catat_waktu, print_separator


class StackingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Custom Stacking Ensemble with base learners (XGB+LGB+CB+LR or any subset)
    and a Logistic Regression meta-learner.
    
    Implements full scikit-learn estimator API for compatibility with
    ImbPipeline, cross_val_predict, cross_validate, etc.
    
    Parameters
    ----------
    apply_resampling : bool, default=False
        If True, SMOTE-ENN is applied INSIDE each OOF fold during fit(),
        preventing synthetic sample leakage. Use True for standalone
        training (main pipeline). Use False when wrapped inside an
        ImbPipeline that already handles resampling (e.g., cross-validation).
    use_xgb : bool, default=True
        If True, includes XGBoost as a base learner. If False, XGBoost is dropped
        for parsimony (3-learner ensemble: LGBM+CatBoost+LR).
    use_lr : bool, default=True
        If True, includes Logistic Regression as a base learner. If False, LR is dropped.
    """
    def __init__(self, xgb_params=None, lgbm_params=None, catboost_params=None,
                 seed=SEED, apply_resampling=False, use_xgb=True, use_lr=True):
        self.xgb_params = xgb_params or {}
        self.lgbm_params = lgbm_params or {}
        self.catboost_params = catboost_params or {}
        self.seed = seed
        self.apply_resampling = apply_resampling
        self.use_xgb = use_xgb
        self.use_lr = use_lr

    def fit(self, X, y):
        """
        Fit all base models and meta-learner.
        
        1. Generate Out-Of-Fold probabilities from base learners via 5-fold CV
           (with SMOTE-ENN per fold if apply_resampling=True)
        2. Train meta-learner on those OOF probabilities
        3. Refit all base learners on 100% of the data
           (with SMOTE-ENN on full data if apply_resampling=True)
        """
        X_arr = np.array(X)
        y_arr = np.array(y)
        
        self.classes_ = np.unique(y_arr)
        self.n_features_in_ = X_arr.shape[1]
        
        # Initialize base models
        self.xgb_ = XGBClassifier(**self.xgb_params) if self.use_xgb else None
        self.lgb_ = LGBMClassifier(**self.lgbm_params)
        self.cat_ = CatBoostClassifier(**self.catboost_params)
        self.lr_ = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=self.seed
        ) if self.use_lr else None
        
        # Initialize meta-learner
        self.meta_learner_ = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=self.seed
        )
        
        # Convenience reference for SHAP analysis
        self.base_models = {
            'lgb': self.lgb_,
            'cat': self.cat_
        }
        if self.use_xgb:
            self.base_models['xgb'] = self.xgb_
        if self.use_lr:
            self.base_models['lr'] = self.lr_
        
        # 5-fold CV to generate OOF meta-features
        n_base = len(self.base_models)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        oof_preds = np.zeros((X_arr.shape[0], n_base))
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_arr, y_arr)):
            X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
            y_tr = y_arr[train_idx]
            
            # Apply SMOTE-ENN inside fold if requested (leakage-free)
            if self.apply_resampling:
                smote = SMOTE(
                    k_neighbors=SMOTE_K_NEIGHBORS,
                    random_state=self.seed,
                    sampling_strategy=SMOTE_TARGET_RATIO
                )
                enn = EditedNearestNeighbours(
                    n_neighbors=ENN_N_NEIGHBORS,
                    kind_sel=ENN_KIND_SEL
                )
                X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
                X_tr, y_tr = enn.fit_resample(X_tr_res, y_tr_res)
            
            # Temporary fold models
            f_lgb = LGBMClassifier(**self.lgbm_params)
            f_cat = CatBoostClassifier(**self.catboost_params)
            
            f_lgb.fit(X_tr, y_tr)
            f_cat.fit(X_tr, y_tr, verbose=False)
            
            col_idx = 0
            if self.use_xgb:
                f_xgb = XGBClassifier(**self.xgb_params)
                f_xgb.fit(X_tr, y_tr, verbose=False)
                oof_preds[val_idx, col_idx] = f_xgb.predict_proba(X_val)[:, 1]
                col_idx += 1
                
            oof_preds[val_idx, col_idx] = f_lgb.predict_proba(X_val)[:, 1]
            oof_preds[val_idx, col_idx + 1] = f_cat.predict_proba(X_val)[:, 1]
            col_idx += 2
            
            if self.use_lr:
                f_lr  = LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=self.seed
                )
                f_lr.fit(X_tr, y_tr)
                oof_preds[val_idx, col_idx] = f_lr.predict_proba(X_val)[:, 1]
        
        # Fit meta-learner on OOF predictions
        self.meta_learner_.fit(oof_preds, y_arr)
        
        # Refit all base models on full data
        if self.apply_resampling:
            # Resample full data for final refit
            smote_full = SMOTE(
                k_neighbors=SMOTE_K_NEIGHBORS,
                random_state=self.seed,
                sampling_strategy=SMOTE_TARGET_RATIO
            )
            enn_full = EditedNearestNeighbours(
                n_neighbors=ENN_N_NEIGHBORS,
                kind_sel=ENN_KIND_SEL
            )
            X_full_res, y_full_res = smote_full.fit_resample(X_arr, y_arr)
            X_refit, y_refit = enn_full.fit_resample(X_full_res, y_full_res)
        else:
            X_refit, y_refit = X, y
        
        if self.use_xgb:
            self.xgb_.fit(X_refit, y_refit, verbose=False)
        self.lgb_.fit(X_refit, y_refit)
        self.cat_.fit(X_refit, y_refit, verbose=False)
        if self.use_lr:
            self.lr_.fit(X_refit, y_refit)
        
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities using stacking meta-learner.
        """
        preds = []
        if self.use_xgb:
            p_xgb = self.xgb_.predict_proba(X)[:, 1]
            preds.append(p_xgb)
        p_lgb = self.lgb_.predict_proba(X)[:, 1]
        p_cat = self.cat_.predict_proba(X)[:, 1]
        preds.extend([p_lgb, p_cat])
        if self.use_lr:
            p_lr = self.lr_.predict_proba(X)[:, 1]
            preds.append(p_lr)
            
        meta_features = np.column_stack(preds)
        return self.meta_learner_.predict_proba(meta_features)

    def predict(self, X):
        """
        Predict class labels (threshold=0.5).
        """
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)

    def get_params(self, deep=True):
        return {
            "xgb_params": self.xgb_params,
            "lgbm_params": self.lgbm_params,
            "catboost_params": self.catboost_params,
            "seed": self.seed,
            "apply_resampling": self.apply_resampling,
            "use_xgb": self.use_xgb,
            "use_lr": self.use_lr
        }
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


def train_stacking_ensemble(X_train_sc, y_train, best_params_xgb, best_params_lgbm, best_params_catboost, use_xgb=True, use_lr=True):
    """
    Train the stacking ensemble on scaled (but NOT resampled) training data.
    
    SMOTE-ENN is applied INSIDE each OOF fold to prevent synthetic sample
    leakage across fold boundaries (V3.1 leakage fix).
    
    Args:
        X_train_sc: Scaled training features (NOT resampled by SMOTE-ENN).
        y_train: Original training target.
        best_params_xgb: Optuna-tuned XGBoost hyperparameters.
        best_params_lgbm: Optuna-tuned LightGBM hyperparameters.
        best_params_catboost: Optuna-tuned CatBoost hyperparameters.
        use_xgb: If True, include XGBoost in base models.
        use_lr: If True, include Logistic Regression in base models.
    
    Returns:
        ensemble: Trained StackingEnsemble.
    """
    print_separator("PHASE 6: TRAINING STACKING ENSEMBLE")
    mulai = time.time()
    
    base_names = []
    if use_xgb:
        base_names.append("XGBoost")
    base_names.extend(["LightGBM", "CatBoost"])
    if use_lr:
        base_names.append("LogisticRegression")
        
    print(f"  Base learners: {' + '.join(base_names)}")
    print("  Meta-learner:  LogisticRegression (class_weight=balanced)")
    print("  ⚠️  SMOTE-ENN applied INSIDE each OOF fold (leakage-free, V3.1 fix)")
    
    ensemble = StackingEnsemble(
        best_params_xgb, best_params_lgbm, best_params_catboost,
        seed=SEED, apply_resampling=True, use_xgb=use_xgb, use_lr=use_lr
    )
    ensemble.fit(X_train_sc, y_train)
    
    # Print and save meta-learner coefficients for Bab IV discussion
    coefs = ensemble.meta_learner_.coef_[0]
    intercept = ensemble.meta_learner_.intercept_[0]
    
    rows = []
    col_idx = 0
    if use_xgb:
        rows.append({"Base Learner": "XGBoost", "Coefficient (Weight)": round(float(coefs[col_idx]), 4)})
        col_idx += 1
    rows.append({"Base Learner": "LightGBM", "Coefficient (Weight)": round(float(coefs[col_idx]), 4)})
    col_idx += 1
    rows.append({"Base Learner": "CatBoost", "Coefficient (Weight)": round(float(coefs[col_idx]), 4)})
    col_idx += 1
    if use_lr:
        rows.append({"Base Learner": "LogisticRegression", "Coefficient (Weight)": round(float(coefs[col_idx]), 4)})
        col_idx += 1
    rows.append({"Base Learner": "Intercept", "Coefficient (Weight)": round(float(intercept), 4)})
    coef_df = pd.DataFrame(rows)
    
    from src.config import OUTPUT_DIR
    coef_path = f"{OUTPUT_DIR}/meta_learner_coefficients.csv"
    coef_df.to_csv(coef_path, index=False)

    print(f"\n  Meta-learner coefficients (Weights):")
    col_idx = 0
    if use_xgb:
        print(f"    XGBoost:             {coefs[col_idx]:.4f}")
        col_idx += 1
    print(f"    LightGBM:            {coefs[col_idx]:.4f}")
    col_idx += 1
    print(f"    CatBoost:            {coefs[col_idx]:.4f}")
    col_idx += 1
    if use_lr:
        print(f"    LogisticRegression:  {coefs[col_idx]:.4f}")
        col_idx += 1
    print(f"    Intercept:           {intercept:.4f}")
    print(f"  📄 Saved meta-learner coefficients to {coef_path}")
    
    print("  ✅ Stacking ensemble trained successfully (leakage-free OOF).")
    catat_waktu("Stacking Training", mulai)
    return ensemble
