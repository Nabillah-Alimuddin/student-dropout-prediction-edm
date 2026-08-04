"""
05b_stacking_training.py — Level 0 Base Learners and Level 1 Meta Learner.
"""

import time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from src.config import SEED
from src.utils import catat_waktu, print_separator

class StackingEnsemble:
    """
    Custom Stacking Ensemble implementing scikit-learn compatible interface.
    """
    def __init__(self, xgb_params, lgbm_params, catboost_params, seed=SEED):
        self.xgb_params = xgb_params
        self.lgbm_params = lgbm_params
        self.catboost_params = catboost_params
        self.seed = seed
        
        self.xgb = XGBClassifier(**xgb_params)
        self.lgb = LGBMClassifier(**lgbm_params)
        self.cat = CatBoostClassifier(**catboost_params)
        
        # Meta learner
        self.meta_learner = LogisticRegression(class_weight="balanced", random_state=seed)
        self.base_models = {
            'xgb': self.xgb,
            'lgb': self.lgb,
            'cat': self.cat
        }

    def fit(self, X_res, y_res):
        """
        Fit base models on resampled data.
        Generates Out-Of-Fold predictions to train the Meta-Learner.
        Then refits base models on 100% of resampled data.
        """
        X_res_arr = np.array(X_res)
        y_res_arr = np.array(y_res)
        
        # 5-fold CV to generate meta features for the training set (no leakage)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        
        oof_preds = np.zeros((X_res_arr.shape[0], 3))
        
        for train_idx, val_idx in skf.split(X_res_arr, y_res_arr):
            X_tr, X_val = X_res_arr[train_idx], X_res_arr[val_idx]
            y_tr, y_val = y_res_arr[train_idx], y_res_arr[val_idx]
            
            # Temporary fold models
            fold_xgb = XGBClassifier(**self.xgb_params)
            fold_lgb = LGBMClassifier(**self.lgbm_params)
            fold_cat = CatBoostClassifier(**self.catboost_params)
            
            fold_xgb.fit(X_tr, y_tr, verbose=False)
            fold_lgb.fit(X_tr, y_tr)
            fold_cat.fit(X_tr, y_tr, verbose=False)
            
            oof_preds[val_idx, 0] = fold_xgb.predict_proba(X_val)[:, 1]
            oof_preds[val_idx, 1] = fold_lgb.predict_proba(X_val)[:, 1]
            oof_preds[val_idx, 2] = fold_cat.predict_proba(X_val)[:, 1]
            
        # Fit meta learner on OOF predictions
        self.meta_learner.fit(oof_preds, y_res_arr)
        
        # Refit base models on full data
        self.xgb.fit(X_res, y_res, verbose=False)
        self.lgb.fit(X_res, y_res)
        self.cat.fit(X_res, y_res, verbose=False)
        return self

    def predict_proba(self, X):
        """
        Predict probability of dropout (class 1) using the stacking meta-learner.
        """
        # Generate meta features from base models
        p_xgb = self.xgb.predict_proba(X)[:, 1]
        p_lgb = self.lgb.predict_proba(X)[:, 1]
        p_cat = self.cat.predict_proba(X)[:, 1]
        
        meta_features = np.column_stack([p_xgb, p_lgb, p_cat])
        return self.meta_learner.predict_proba(meta_features)

    def predict(self, X):
        """
        Predict class labels.
        """
        # Default to 0.5 threshold
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)

    def get_params(self, deep=True):
        return {
            "xgb_params": self.xgb_params,
            "lgbm_params": self.lgbm_params,
            "catboost_params": self.catboost_params,
            "seed": self.seed
        }


def train_stacking_ensemble(X_res, y_res, best_params_xgb, best_params_lgbm, best_params_catboost):
    print_separator("PHASE 6: TRAINING STACKING ENSEMBLE")
    mulai = time.time()
    
    ensemble = StackingEnsemble(best_params_xgb, best_params_lgbm, best_params_catboost, seed=SEED)
    ensemble.fit(X_res, y_res)
    
    print("  ✅ Stacking ensemble and base models trained successfully.")
    catat_waktu("Stacking Training", mulai)
    return ensemble
