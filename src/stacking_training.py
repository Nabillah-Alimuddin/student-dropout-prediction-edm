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

Scaling Consistency (V3.2 fix):
    When apply_scaling=True, StandardScaler is fitted INSIDE each OOF fold
    on the fold training split only, matching the scaling regime used during
    Optuna hyperparameter tuning (ImbPipeline with per-fold scaler).
    This eliminates the mild scaling leakage from global pre-scaling.
"""

import time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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
    apply_scaling : bool, default=False
        If True, StandardScaler is fitted INSIDE each OOF fold during fit(),
        matching the scaling regime used during Optuna tuning. The final
        scaler (fitted on 100% training data) is stored as self.scaler_
        and auto-applied in predict_proba()/predict().
        Use True for standalone training when receiving unscaled data.
        Use False when wrapped inside an ImbPipeline that already scales.
    use_xgb : bool, default=True
        If True, includes XGBoost as a base learner. If False, XGBoost is dropped
        for parsimony (3-learner ensemble: LGBM+CatBoost+LR).
    use_lr : bool, default=True
        If True, includes Logistic Regression as a base learner. If False, LR is dropped.
    """
    def __init__(self, xgb_params=None, lgbm_params=None, catboost_params=None,
                 seed=SEED, apply_resampling=False, apply_scaling=False,
                 use_xgb=True, use_lr=True):
        self.xgb_params = xgb_params or {}
        self.lgbm_params = lgbm_params or {}
        self.catboost_params = catboost_params or {}
        self.seed = seed
        self.apply_resampling = apply_resampling
        self.apply_scaling = apply_scaling
        self.use_xgb = use_xgb
        self.use_lr = use_lr
        self.scaler_ = None  # Will be set if apply_scaling=True

    def fit(self, X, y):
        """
        Fit all base models and meta-learner.
        
        1. Generate Out-Of-Fold probabilities from base learners via 5-fold CV
           (with per-fold StandardScaler if apply_scaling=True,
            with per-fold SMOTE-ENN if apply_resampling=True)
        2. Train meta-learner on those OOF probabilities
        3. Refit all base learners on 100% of the data
           (with global StandardScaler and SMOTE-ENN as applicable)
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
            
            # Apply per-fold scaling if requested (matches Optuna regime)
            if self.apply_scaling:
                fold_scaler = StandardScaler()
                X_tr = fold_scaler.fit_transform(X_tr)
                X_val = fold_scaler.transform(X_val)
            
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
        
        # Prepare data for final refit on 100% of data
        X_refit = X_arr.copy()
        y_refit = y_arr.copy()
        
        # Apply global scaling for final refit (and save scaler for predict)
        if self.apply_scaling:
            self.scaler_ = StandardScaler()
            X_refit = self.scaler_.fit_transform(X_refit)
        
        # Resample full data for final refit
        if self.apply_resampling:
            smote_full = SMOTE(
                k_neighbors=SMOTE_K_NEIGHBORS,
                random_state=self.seed,
                sampling_strategy=SMOTE_TARGET_RATIO
            )
            enn_full = EditedNearestNeighbours(
                n_neighbors=ENN_N_NEIGHBORS,
                kind_sel=ENN_KIND_SEL
            )
            X_full_res, y_full_res = smote_full.fit_resample(X_refit, y_refit)
            X_refit, y_refit = enn_full.fit_resample(X_full_res, y_full_res)
        
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
        Auto-scales input if self.scaler_ was fitted during training.
        """
        X_input = np.array(X)
        if self.scaler_ is not None:
            X_input = self.scaler_.transform(X_input)
        
        preds = []
        if self.use_xgb:
            p_xgb = self.xgb_.predict_proba(X_input)[:, 1]
            preds.append(p_xgb)
        p_lgb = self.lgb_.predict_proba(X_input)[:, 1]
        p_cat = self.cat_.predict_proba(X_input)[:, 1]
        preds.extend([p_lgb, p_cat])
        if self.use_lr:
            p_lr = self.lr_.predict_proba(X_input)[:, 1]
            preds.append(p_lr)
            
        meta_features = np.column_stack(preds)
        return self.meta_learner_.predict_proba(meta_features)

    def predict(self, X):
        """
        Predict class labels (threshold=0.5).
        Auto-scales input if self.scaler_ was fitted during training.
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
            "apply_scaling": self.apply_scaling,
            "use_xgb": self.use_xgb,
            "use_lr": self.use_lr
        }
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


def train_stacking_ensemble(X_train, y_train, best_params_xgb, best_params_lgbm, best_params_catboost, use_xgb=True, use_lr=True):
    """
    Train the stacking ensemble on UNSCALED training data.
    
    StandardScaler is fitted INSIDE each OOF fold to match the scaling regime
    used during Optuna hyperparameter tuning (V3.2 scaling consistency fix).
    SMOTE-ENN is applied INSIDE each OOF fold to prevent synthetic sample
    leakage across fold boundaries (V3.1 leakage fix).
    
    Args:
        X_train: Original training features (NOT pre-scaled, NOT resampled).
        y_train: Original training target.
        best_params_xgb: Optuna-tuned XGBoost hyperparameters.
        best_params_lgbm: Optuna-tuned LightGBM hyperparameters.
        best_params_catboost: Optuna-tuned CatBoost hyperparameters.
        use_xgb: If True, include XGBoost in base models.
        use_lr: If True, include Logistic Regression in base models.
    
    Returns:
        ensemble: Trained StackingEnsemble (with internal scaler).
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
    print("  ⚠️  StandardScaler applied INSIDE each OOF fold (V3.2 scaling consistency fix)")
    print("  ⚠️  SMOTE-ENN applied INSIDE each OOF fold (leakage-free, V3.1 fix)")
    
    ensemble = StackingEnsemble(
        best_params_xgb, best_params_lgbm, best_params_catboost,
        seed=SEED, apply_resampling=True, apply_scaling=True,
        use_xgb=use_xgb, use_lr=use_lr
    )
    ensemble.fit(X_train, y_train)
    
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
    
    print("  ✅ Stacking ensemble trained successfully (leakage-free OOF, consistent scaling).")
    catat_waktu("Stacking Training", mulai)
    return ensemble
