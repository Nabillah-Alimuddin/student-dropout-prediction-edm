"""
19_multiseed_robustness.py — Poin 3: Multi-Seed Robustness Analysis.

Runs the proposed Stacking Ensemble model and the runner-up CatBoost model
on 5 different train-test split seeds to evaluate if the Stacking Ensemble's
superiority is robust or just a split coincidence.
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, balanced_accuracy_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from catboost import CatBoostClassifier

from src.config import (
    SEED, OUTPUT_DIR, TEST_SIZE,
    SMOTE_K_NEIGHBORS, SMOTE_TARGET_RATIO, ENN_N_NEIGHBORS, ENN_KIND_SEL
)
from src.utils import catat_waktu, print_separator
from src.stacking_training import StackingEnsemble

def run_multiseed_robustness(X, y, best_params_xgb, best_params_lgbm, best_params_cb, seeds=[42, 123, 456, 789, 2024]):
    """
    Train and evaluate Proposed Stacking and Runner-Up CatBoost on 5 different seeds.
    
    Args:
        X (pd.DataFrame): Raw features.
        y (pd.Series): Target.
        best_params_xgb (dict): Best parameters for XGB.
        best_params_lgbm (dict): Best parameters for LGBM.
        best_params_cb (dict): Best parameters for CatBoost.
        seeds (list): List of seeds to evaluate.
        
    Returns:
        summary_df (pd.DataFrame): Comparison summary.
    """
    print_separator("POIN 3: MULTI-SEED ROBUSTNESS ANALYSIS (5 DIFFERENT SPLITS)")
    mulai = time.time()
    
    results = []
    
    # Define helper function to find optimal threshold via OOF CV
    def find_best_threshold_oof(model_instance, X_tr, y_tr, is_stacking=False):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        n_jobs = 1 if is_stacking else -1
        try:
            y_oof_proba = cross_val_predict(
                model_instance, X_tr, y_tr, cv=skf, method="predict_proba", n_jobs=n_jobs
            )[:, 1]
        except Exception:
            y_oof_proba = model_instance.predict_proba(X_tr)[:, 1]

        thresholds = np.arange(0.1, 0.9, 0.01)
        best_f1 = -1
        best_t = 0.50
        for t in thresholds:
            y_pred_t = (y_oof_proba >= t).astype(int)
            f1 = f1_score(y_tr, y_pred_t, pos_label=1, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        return best_t

    for seed in seeds:
        print(f"\n  --- Running experiments for Seed: {seed} ---")
        
        # 1. Split data (Stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
        )
        
        # 2. Setup Models
        
        # A. Proposed Stacking (LGB+CB+LR) with internal scaler and SMOTE-ENN
        # V3.2: Scaling and resampling are handled internally
        model_stacking = StackingEnsemble(
            xgb_params=best_params_xgb,
            lgbm_params=best_params_lgbm,
            catboost_params=best_params_cb,
            seed=seed,
            apply_resampling=True,
            apply_scaling=True,
            use_xgb=False,  # Proposed model has use_xgb=False
            use_lr=True
        )
        
        # Optimize threshold for Stacking via OOF on raw X_train
        opt_t_stacking = find_best_threshold_oof(model_stacking, X_train, y_train, is_stacking=True)
        model_stacking.fit(X_train, y_train)
        
        # Predict on unscaled test
        y_proba_stacking = model_stacking.predict_proba(X_test)[:, 1]
        y_pred_stacking = (y_proba_stacking >= opt_t_stacking).astype(int)
        f1_stacking = f1_score(y_test, y_pred_stacking, pos_label=1, zero_division=0)
        rec_stacking = recall_score(y_test, y_pred_stacking, pos_label=1, zero_division=0)
        prec_stacking = precision_score(y_test, y_pred_stacking, pos_label=1, zero_division=0)
        
        # B. Runner-Up CatBoost via ImbPipeline
        p_cb = best_params_cb.copy()
        p_cb["verbose"] = 0
        p_cb["random_seed"] = seed
        
        pipe_cb = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(
                k_neighbors=SMOTE_K_NEIGHBORS,
                random_state=seed,
                sampling_strategy=SMOTE_TARGET_RATIO
            )),
            ('enn', EditedNearestNeighbours(
                n_neighbors=ENN_N_NEIGHBORS,
                kind_sel=ENN_KIND_SEL
            )),
            ('clf', CatBoostClassifier(**p_cb)),
        ])
        
        # Optimize threshold for CatBoost via OOF on raw X_train
        opt_t_cb = find_best_threshold_oof(pipe_cb, X_train, y_train, is_stacking=False)
        pipe_cb.fit(X_train, y_train)
        
        # Predict on unscaled test
        y_proba_cb = pipe_cb.predict_proba(X_test)[:, 1]
        y_pred_cb = (y_proba_cb >= opt_t_cb).astype(int)
        f1_cb = f1_score(y_test, y_pred_cb, pos_label=1, zero_division=0)
        rec_cb = recall_score(y_test, y_pred_cb, pos_label=1, zero_division=0)
        prec_cb = precision_score(y_test, y_pred_cb, pos_label=1, zero_division=0)
        
        print(f"    Proposed Stacking  -> Threshold: {opt_t_stacking:.2f} | F1: {f1_stacking:.4f} | Recall: {rec_stacking:.4f} | Precision: {prec_stacking:.4f}")
        print(f"    Runner-up CatBoost -> Threshold: {opt_t_cb:.2f} | F1: {f1_cb:.4f} | Recall: {rec_cb:.4f} | Precision: {prec_cb:.4f}")
        
        results.append({
            "Seed": seed,
            "Proposed Stacking F1": f1_stacking,
            "Proposed Stacking Recall": rec_stacking,
            "Proposed Stacking Precision": prec_stacking,
            "Runner-up CatBoost F1": f1_cb,
            "Runner-up CatBoost Recall": rec_cb,
            "Runner-up CatBoost Precision": prec_cb,
        })
        
    df_results = pd.DataFrame(results)
    
    # Calculate Summary Stats
    summary_rows = [
        {
            "Seed": "Mean",
            "Proposed Stacking F1": df_results["Proposed Stacking F1"].mean(),
            "Proposed Stacking Recall": df_results["Proposed Stacking Recall"].mean(),
            "Proposed Stacking Precision": df_results["Proposed Stacking Precision"].mean(),
            "Runner-up CatBoost F1": df_results["Runner-up CatBoost F1"].mean(),
            "Runner-up CatBoost Recall": df_results["Runner-up CatBoost Recall"].mean(),
            "Runner-up CatBoost Precision": df_results["Runner-up CatBoost Precision"].mean(),
        },
        {
            "Seed": "Std",
            "Proposed Stacking F1": df_results["Proposed Stacking F1"].std(),
            "Proposed Stacking Recall": df_results["Proposed Stacking Recall"].std(),
            "Proposed Stacking Precision": df_results["Proposed Stacking Precision"].std(),
            "Runner-up CatBoost F1": df_results["Runner-up CatBoost F1"].std(),
            "Runner-up CatBoost Recall": df_results["Runner-up CatBoost Recall"].std(),
            "Runner-up CatBoost Precision": df_results["Runner-up CatBoost Precision"].std(),
        }
    ]
    df_summary = pd.concat([df_results, pd.DataFrame(summary_rows)], ignore_index=True)
    
    csv_path = os.path.join(OUTPUT_DIR, "multiseed_robustness_summary.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"\n  📄 Saved multi-seed robustness summary to {csv_path}")
    
    # Display summary
    print("\n  === MULTI-SEED ROBUSTNESS RESULTS ===")
    print(df_summary.tail(2).to_string(index=False))
    
    # Generate visualization
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(seeds))
    width = 0.35
    
    ax.bar(x - width/2, df_results["Proposed Stacking F1"], width, label="Proposed Stacking (LGB+CB+LR)", color="#e74c3c", edgecolor="black", linewidth=0.5)
    ax.bar(x + width/2, df_results["Runner-up CatBoost F1"], width, label="Runner-up CatBoost", color="#3498db", edgecolor="black", linewidth=0.5)
    
    ax.set_title("F1-Score Stability Across 5 Different Train-Test Splits", fontsize=12, fontweight="bold")
    ax.set_xlabel("Split Seed")
    ax.set_ylabel("F1-Score (Dropout)")
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.set_ylim(0.85, 0.95)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    
    # Add mean lines
    mean_stacking = df_results["Proposed Stacking F1"].mean()
    mean_cb = df_results["Runner-up CatBoost F1"].mean()
    ax.axhline(mean_stacking, color="#e74c3c", linestyle="--", alpha=0.7, label=f"Proposed Mean: {mean_stacking:.4f}")
    ax.axhline(mean_cb, color="#3498db", linestyle="--", alpha=0.7, label=f"Runner-up Mean: {mean_cb:.4f}")
    ax.legend(loc="lower right")
    
    plt.tight_layout()
    plot_filename = "v3_multiseed_robustness.pdf"
    from src.utils import save_pdf
    save_pdf(fig, plot_filename)
    plt.close(fig)
    print(f"  📄 Saved multi-seed robustness plot to {OUTPUT_DIR}/{plot_filename}")
    
    catat_waktu("Multi-Seed Robustness Analysis", mulai)
    return df_summary
