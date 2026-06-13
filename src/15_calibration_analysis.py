"""
15_calibration_analysis.py — Priority 3: Calibration Analysis.

Evaluates probability calibration of Logistic Regression, XGBoost Baseline,
XGBoost Proposed, CatBoost Default, and CatBoost Tuned on the test set.
Computes Brier Score and Expected Calibration Error (ECE), and plots reliability diagrams.
"""

import time
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

# Import configuration and utilities from src
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    SEED, OUTPUT_DIR, MODEL_PATH, CATBOOST_MODEL_PATH
)
from src.utils import catat_waktu, save_pdf, print_separator, reset_waktu_log


def expected_calibration_error(y_true, y_proba, n_bins=10):
    """
    Computes the Expected Calibration Error (ECE).
    """
    y_true = np.array(y_true)
    y_proba = np.array(y_proba)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Handle last bin boundary inclusive
        if i == n_bins - 1:
            in_bin = (y_proba >= bin_lower) & (y_proba <= bin_upper)
        else:
            in_bin = (y_proba >= bin_lower) & (y_proba < bin_upper)
            
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_proba[in_bin])
            ece += (bin_size / len(y_true)) * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
    return ece


def main():
    reset_waktu_log()
    start_time = time.time()

    print("=" * 70)
    print("  PROBABILITY CALIBRATION ANALYSIS (BRIER SCORE & ECE)")
    print("=" * 70)

    # ─── Load Data ────────────────────────────────────────────────────────
    import importlib
    data_preparation = importlib.import_module("src.01_data_preparation")
    preprocessing = importlib.import_module("src.02_preprocessing")
    smoteenn = importlib.import_module("src.03_smoteenn")

    load_and_prepare_data = data_preparation.load_and_prepare_data
    split_and_scale = preprocessing.split_and_scale
    apply_smoteenn = smoteenn.apply_smoteenn

    X, y, df = load_and_prepare_data()
    X_train, X_test, X_train_sc, X_test_sc, y_train, y_test, scaler = split_and_scale(X, y)
    X_res, y_res = apply_smoteenn(X_train_sc, y_train)

    # ─── Load or Train Models ─────────────────────────────────────────────
    print_separator("LOADING AND TRAINING MODELS FOR CALIBRATION")
    
    # 1. Logistic Regression (Standard Baseline)
    lr_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
    lr_model.fit(X_train_sc, y_train)
    
    # 2. XGBoost Baseline
    xgb_baseline = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=SEED, use_label_encoder=False, eval_metric="logloss"
    )
    xgb_baseline.fit(X_train_sc, y_train)
    
    # 3. XGBoost Proposed
    if os.path.exists(MODEL_PATH):
        print(f"  Loading XGBoost Proposed from {MODEL_PATH}...")
        xgb_proposed = joblib.load(MODEL_PATH)
    else:
        print("  ⚠️ XGBoost Proposed model not found. Training default XGBoost on resampled data...")
        xgb_proposed = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=SEED, use_label_encoder=False, eval_metric="logloss"
        )
        xgb_proposed.fit(X_res, y_res)
        
    # 4. CatBoost Default
    cb_default = CatBoostClassifier(random_seed=SEED, verbose=0, thread_count=-1)
    cb_default.fit(X_res, y_res)
    
    # 5. CatBoost Tuned
    if os.path.exists(CATBOOST_MODEL_PATH):
        print(f"  Loading CatBoost Tuned from {CATBOOST_MODEL_PATH}...")
        cb_tuned = joblib.load(CATBOOST_MODEL_PATH)
    else:
        print("  ⚠️ CatBoost Tuned model not found. Training default CatBoost on resampled data...")
        cb_tuned = cb_default

    models = {
        "Logistic Regression": lr_model,
        "XGBoost Baseline": xgb_baseline,
        "XGBoost + SMOTE-ENN (Proposed)": xgb_proposed,
        "CatBoost (Default)": cb_default,
        "CatBoost + SMOTE-ENN (Tuned)": cb_tuned
    }

    # ─── Calibration Evaluation ──────────────────────────────────────────
    print_separator("EVALUATING CALIBRATION METRICS ON TEST SET")
    
    calibration_metrics = []
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Perfect calibration reference line
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated", alpha=0.7)
    
    colors = ["#7f8c8d", "#e67e22", "#e74c3c", "#f1c40f", "#2ecc71"]
    
    for idx, (name, clf) in enumerate(models.items()):
        y_proba = clf.predict_proba(X_test_sc)[:, 1]
        
        # Calculate metrics
        brier = brier_score_loss(y_test, y_proba)
        ece = expected_calibration_error(y_test, y_proba, n_bins=10)
        
        calibration_metrics.append({
            "Model": name,
            "Brier Score": round(brier, 6),
            "Expected Calibration Error (ECE)": round(ece, 6)
        })
        
        # Get calibration curve data
        prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy="uniform")
        
        ax.plot(prob_pred, prob_true, "s-", label=f"{name} (Brier={brier:.4f}, ECE={ece:.4f})",
                color=colors[idx], markersize=6, linewidth=1.5)
        
    metrics_df = pd.DataFrame(calibration_metrics)
    csv_path = os.path.join(OUTPUT_DIR, "calibration_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"\n  ✅ Calibration metrics saved to {csv_path}")
    print(metrics_df.to_string(index=False))
    
    # Plot formatting
    ax.set_xlabel("Mean Predicted Probability (Confidence)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Fraction of Positives (Accuracy)", fontsize=12, fontweight="bold")
    ax.set_title("Calibration Curves (Reliability Diagram)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    
    plt.tight_layout()
    plot_path = "v3_calibration_analysis.pdf"
    save_pdf(fig, plot_path)
    plt.close(fig)
    
    total_time = time.time() - start_time
    print(f"\n  🏁 Calibration analysis complete. Total time: {total_time:.2f}s")


if __name__ == "__main__":
    main()
