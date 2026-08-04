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
from src.config import SEED
from src.utils import catat_waktu, print_separator


class StackingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Custom Stacking Ensemble with 4 base learners (XGB+LGB+CB+LR)
    and a Logistic Regression meta-learner.
    
    Implements full scikit-learn estimator API for compatibility with
    ImbPipeline, cross_val_predict, cross_validate, etc.
    """
    def __init__(self, xgb_params=None, lgbm_params=None, catboost_params=None, seed=SEED):
        self.xgb_params = xgb_params or {}
        self.lgbm_params = lgbm_params or {}
        self.catboost_params = catboost_params or {}
        self.seed = seed

    def fit(self, X, y):
        """
        Fit all base models and meta-learner.
        
        1. Generate Out-Of-Fold probabilities from 4 base learners via 5-fold CV
        2. Train meta-learner on those OOF probabilities
        3. Refit all base learners on 100% of the data
        """
        X_arr = np.array(X)
        y_arr = np.array(y)
        
        self.classes_ = np.unique(y_arr)
        self.n_features_in_ = X_arr.shape[1]
        
        # Initialize base models
        self.xgb_ = XGBClassifier(**self.xgb_params)
        self.lgb_ = LGBMClassifier(**self.lgbm_params)
        self.cat_ = CatBoostClassifier(**self.catboost_params)
        self.lr_ = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=self.seed
        )
        
        # Initialize meta-learner
        self.meta_learner_ = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=self.seed
        )
        
        # Convenience reference for SHAP analysis
        self.base_models = {
            'xgb': self.xgb_,
            'lgb': self.lgb_,
            'cat': self.cat_,
            'lr':  self.lr_
        }
        
        # 5-fold CV to generate OOF meta-features (no leakage)
        n_base = 4
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        oof_preds = np.zeros((X_arr.shape[0], n_base))
        
        for train_idx, val_idx in skf.split(X_arr, y_arr):
            X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
            y_tr = y_arr[train_idx]
            
            # Temporary fold models
            f_xgb = XGBClassifier(**self.xgb_params)
            f_lgb = LGBMClassifier(**self.lgbm_params)
            f_cat = CatBoostClassifier(**self.catboost_params)
            f_lr  = LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=self.seed
            )
            
            f_xgb.fit(X_tr, y_tr, verbose=False)
            f_lgb.fit(X_tr, y_tr)
            f_cat.fit(X_tr, y_tr, verbose=False)
            f_lr.fit(X_tr, y_tr)
            
            oof_preds[val_idx, 0] = f_xgb.predict_proba(X_val)[:, 1]
            oof_preds[val_idx, 1] = f_lgb.predict_proba(X_val)[:, 1]
            oof_preds[val_idx, 2] = f_cat.predict_proba(X_val)[:, 1]
            oof_preds[val_idx, 3] = f_lr.predict_proba(X_val)[:, 1]
        
        # Fit meta-learner on OOF predictions
        self.meta_learner_.fit(oof_preds, y_arr)
        
        # Refit all base models on full data
        self.xgb_.fit(X, y, verbose=False)
        self.lgb_.fit(X, y)
        self.cat_.fit(X, y, verbose=False)
        self.lr_.fit(X, y)
        
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities using stacking meta-learner.
        """
        p_xgb = self.xgb_.predict_proba(X)[:, 1]
        p_lgb = self.lgb_.predict_proba(X)[:, 1]
        p_cat = self.cat_.predict_proba(X)[:, 1]
        p_lr  = self.lr_.predict_proba(X)[:, 1]
        
        meta_features = np.column_stack([p_xgb, p_lgb, p_cat, p_lr])
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
            "seed": self.seed
        }
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


def train_stacking_ensemble(X_res, y_res, best_params_xgb, best_params_lgbm, best_params_catboost):
    """Train the stacking ensemble on resampled data."""
    print_separator("PHASE 6: TRAINING STACKING ENSEMBLE")
    mulai = time.time()
    
    print("  Base learners: XGBoost + LightGBM + CatBoost + LogisticRegression")
    print("  Meta-learner:  LogisticRegression (class_weight=balanced)")
    
    ensemble = StackingEnsemble(best_params_xgb, best_params_lgbm, best_params_catboost, seed=SEED)
    ensemble.fit(X_res, y_res)
    
    # Print and save meta-learner coefficients for Bab IV discussion
    coefs = ensemble.meta_learner_.coef_[0]
    intercept = ensemble.meta_learner_.intercept_[0]
    
    coef_df = pd.DataFrame([
        {"Base Learner": "XGBoost", "Coefficient (Weight)": round(float(coefs[0]), 4)},
        {"Base Learner": "LightGBM", "Coefficient (Weight)": round(float(coefs[1]), 4)},
        {"Base Learner": "CatBoost", "Coefficient (Weight)": round(float(coefs[2]), 4)},
        {"Base Learner": "LogisticRegression", "Coefficient (Weight)": round(float(coefs[3]), 4)},
        {"Base Learner": "Intercept", "Coefficient (Weight)": round(float(intercept), 4)},
    ])
    
    from src.config import OUTPUT_DIR
    coef_path = f"{OUTPUT_DIR}/meta_learner_coefficients.csv"
    coef_df.to_csv(coef_path, index=False)

    print(f"\n  Meta-learner coefficients (Weights):")
    print(f"    XGBoost:             {coefs[0]:.4f}")
    print(f"    LightGBM:            {coefs[1]:.4f}")
    print(f"    CatBoost:            {coefs[2]:.4f}")
    print(f"    LogisticRegression:  {coefs[3]:.4f}")
    print(f"    Intercept:           {intercept:.4f}")
    print(f"  📄 Saved meta-learner coefficients to {coef_path}")
    
    print("  ✅ Stacking ensemble trained successfully.")
    catat_waktu("Stacking Training", mulai)
    return ensemble
