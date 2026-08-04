"""
04_optuna_tuning.py — Phase 5: Hyperparameter Optimization with Optuna.

Uses TPE sampler with MedianPruner, 200 trials, 5-fold StratifiedKFold CV,
optimizing F1-Score for the Dropout class (pos_label=1).

Methodological note:
- Receives the ORIGINAL scaled training data (X_train_sc, y_train), NOT the
  pre-resampled X_res/y_res. SMOTE-ENN is applied INSIDE each CV fold via
  ImbPipeline, preventing synthetic samples from leaking into validation folds.
"""

import time
import numpy as np
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, make_scorer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours

from src.config import (
    SEED, OPTUNA_N_TRIALS, OPTUNA_TIMEOUT,
    OPTUNA_CV_FOLDS, OPTUNA_STARTUP_TRIALS, OPTUNA_WARMUP_STEPS,
    SMOTE_K_NEIGHBORS, SMOTE_TARGET_RATIO, ENN_N_NEIGHBORS, ENN_KIND_SEL
)
from src.utils import catat_waktu, print_separator


def run_optuna_tuning(X_train, y_train):
    """
    Run Optuna hyperparameter optimization for XGBoost.

    Uses the original unscaled training data. StandardScaler and SMOTE-ENN are 
    applied inside each CV fold via ImbPipeline, preventing any leakage.

    Args:
        X_train (pd.DataFrame): Original training features (NOT scaled/resampled).
        y_train (pd.Series): Original training target (NOT resampled).

    Returns:
        best_params (dict): Best hyperparameters found by Optuna.
        study (optuna.Study): The completed Optuna study object.
    """
    print_separator("PHASE 5: OPTUNA HYPERPARAMETER OPTIMIZATION")
    mulai = time.time()

    # ─── Objective function ───────────────────────────────────────────────
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 700),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 30),
            "gamma": trial.suggest_float("gamma", 0.0, 10.0),
            "max_delta_step": trial.suggest_int("max_delta_step", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1.0, 30.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 30.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 2.0),
            # Fixed parameters
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": SEED,
            "use_label_encoder": False,
        }

        # ── Scaling + SMOTE-ENN + XGBoost pipeline — resampling inside each fold ──
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
            ('clf', XGBClassifier(**params)),
        ])

        skf = StratifiedKFold(
            n_splits=OPTUNA_CV_FOLDS, shuffle=True, random_state=SEED
        )

        f1_scorer = make_scorer(f1_score, pos_label=1)
        # CV on original training data; SMOTE-ENN & scaling applied per fold inside pipe
        scores = cross_val_score(
            pipe, X_train, y_train, cv=skf, scoring=f1_scorer, n_jobs=-1
        )

        return scores.mean()

    # ─── Configure Optuna ─────────────────────────────────────────────────
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=OPTUNA_STARTUP_TRIALS,
        n_warmup_steps=OPTUNA_WARMUP_STEPS
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner
    )

    # ─── Run optimization ─────────────────────────────────────────────────
    print(f"  Running {OPTUNA_N_TRIALS} trials (timeout: {OPTUNA_TIMEOUT}s)...")
    print(f"  Sampler: TPE (seed={SEED})")
    print(f"  Pruner: MedianPruner (startup={OPTUNA_STARTUP_TRIALS}, warmup={OPTUNA_WARMUP_STEPS})")
    print(f"  CV: {OPTUNA_CV_FOLDS}-fold StratifiedKFold (SMOTE-ENN & scaling inside each fold via ImbPipeline)")
    print(f"  Objective: Maximize F1-Score (Dropout)")
    print(f"  Input: original X_train (NOT pre-scaled/resampled) — leakage-free")
    print()

    study.optimize(
        objective,
        n_trials=OPTUNA_N_TRIALS,
        timeout=OPTUNA_TIMEOUT,
        show_progress_bar=True
    )

    # ─── Report results ───────────────────────────────────────────────────
    best_params = study.best_params.copy()
    # Add fixed params
    best_params.update({
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": SEED,
        "use_label_encoder": False,
    })

    print(f"\n  Best trial: #{study.best_trial.number}")
    print(f"  Best F1-Score (CV, leakage-free): {study.best_value:.6f}")
    print(f"\n  Best parameters:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    catat_waktu("Optuna Tuning", mulai)

    return best_params, study


def run_optuna_tuning_lgbm(X_train, y_train):
    """
    Run Optuna hyperparameter optimization for LightGBM.
    """
    from lightgbm import LGBMClassifier
    print_separator("PHASE 5 (LGBM): OPTUNA HYPERPARAMETER OPTIMIZATION FOR LIGHTGBM")
    mulai = time.time()

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 700),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 30.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
            "random_state": SEED,
            "verbosity": -1
        }

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
            ('clf', LGBMClassifier(**params)),
        ])

        skf = StratifiedKFold(
            n_splits=OPTUNA_CV_FOLDS, shuffle=True, random_state=SEED
        )

        f1_scorer = make_scorer(f1_score, pos_label=1)
        scores = cross_val_score(
            pipe, X_train, y_train, cv=skf, scoring=f1_scorer, n_jobs=-1
        )

        return scores.mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # We use a lower number of trials or custom config for fast execution if needed
    from src.config import LGBM_N_TRIALS, LGBM_TIMEOUT
    print(f"  Running {LGBM_N_TRIALS} trials (timeout: {LGBM_TIMEOUT}s)...")

    study.optimize(objective, n_trials=LGBM_N_TRIALS, timeout=LGBM_TIMEOUT, show_progress_bar=True)

    best_params = study.best_params.copy()
    best_params.update({
        "random_state": SEED,
        "verbosity": -1
    })

    print(f"\n  Best trial: #{study.best_trial.number}")
    print(f"  Best F1-Score (CV, leakage-free): {study.best_value:.6f}")
    catat_waktu("Optuna Tuning LGBM", mulai)
    return best_params, study


def run_optuna_tuning_catboost(X_train, y_train):
    """
    Run Optuna hyperparameter optimization for CatBoost.
    """
    from catboost import CatBoostClassifier
    print_separator("PHASE 5 (CatBoost): OPTUNA HYPERPARAMETER OPTIMIZATION FOR CATBOOST")
    mulai = time.time()

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 100, 700),
            "depth": trial.suggest_int("depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "random_seed": SEED,
            "verbose": False
        }

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
            ('clf', CatBoostClassifier(**params)),
        ])

        skf = StratifiedKFold(
            n_splits=OPTUNA_CV_FOLDS, shuffle=True, random_state=SEED
        )

        f1_scorer = make_scorer(f1_score, pos_label=1)
        scores = cross_val_score(
            pipe, X_train, y_train, cv=skf, scoring=f1_scorer, n_jobs=-1
        )

        return scores.mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    from src.config import CATBOOST_N_TRIALS, CATBOOST_TIMEOUT
    print(f"  Running {CATBOOST_N_TRIALS} trials (timeout: {CATBOOST_TIMEOUT}s)...")

    study.optimize(objective, n_trials=CATBOOST_N_TRIALS, timeout=CATBOOST_TIMEOUT, show_progress_bar=True)

    best_params = study.best_params.copy()
    best_params.update({
        "random_seed": SEED,
        "verbose": False
    })

    print(f"\n  Best trial: #{study.best_trial.number}")
    print(f"  Best F1-Score (CV, leakage-free): {study.best_value:.6f}")
    catat_waktu("Optuna Tuning CatBoost", mulai)
    return best_params, study

