"""
15_calibration_analysis.py — Poin 6: Model Calibration Analysis (ECE & Brier Score).

Evaluates the probability calibration of the proposed Stacking Ensemble and the
baseline models on the test set. Computes Expected Calibration Error (ECE) and
Brier Score, and generates a reliability diagram (calibration curves).
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from src.config import SEED, OUTPUT_DIR, MODEL_PATH, DATA_PATH
from src.utils import catat_waktu, save_pdf, print_separator

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

def run_calibration_analysis(models_dict, X_test, X_test_sc, y_test):
    """
    Evaluates probability calibration of baseline and proposed models.
    
    Args:
        models_dict (dict): Dictionary mapping model names to fitted model/pipeline objects.
        X_test (pd.DataFrame): Unscaled test features (for Proposed Stacking).
        X_test_sc (pd.DataFrame): Scaled test features (for baselines).
        y_test (pd.Series): Test labels.
        
    Returns:
        metrics_df (pd.DataFrame): Calibration metrics dataframe.
    """
    print_separator("POIN 6: PROBABILITY CALIBRATION ANALYSIS (BRIER SCORE & ECE)")
    mulai = time.time()
    
    calibration_metrics = []
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Perfect calibration reference line
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated", alpha=0.7)
    
    # Define colors for different models to match comparison tables
    colors = {
        "Stacking (Proposed: LGB+CB+LR) + SMOTE-ENN": "#e74c3c", # Red
        "CatBoost + SMOTE-ENN": "#2ecc71",                       # Green
        "LightGBM + SMOTE-ENN": "#f1c40f",                       # Yellow
        "XGBoost + SMOTE-ENN": "#9b59b6",                        # Purple
        "Random Forest + SMOTE-ENN": "#3498db",                   # Blue
        "Logistic Regression + SMOTE-ENN": "#7f8c8d"             # Grey
    }
    
    print("  Evaluating calibration metrics on test set:")
    for name, clf in models_dict.items():
        # Determine correct feature set to avoid double-scaling
        if "Proposed" in name or "Stacking" in name:
            # Stacking proposed model auto-scales internally (apply_scaling=True)
            y_proba = clf.predict_proba(X_test)[:, 1]
        else:
            # Baseline models are scaled externally
            y_proba = clf.predict_proba(X_test_sc)[:, 1]
            
        brier = brier_score_loss(y_test, y_proba)
        ece = expected_calibration_error(y_test, y_proba, n_bins=10)
        
        calibration_metrics.append({
            "Model": name,
            "Brier Score": round(brier, 6),
            "Expected Calibration Error (ECE)": round(ece, 6)
        })
        
        # Calculate reliability diagram curve
        prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy="uniform")
        
        color = colors.get(name, "#34495e")
        ax.plot(prob_pred, prob_true, "s-", label=f"{name} (Brier={brier:.4f}, ECE={ece:.4f})",
                color=color, markersize=6, linewidth=1.5)
        
    metrics_df = pd.DataFrame(calibration_metrics)
    metrics_df = metrics_df.sort_values("Expected Calibration Error (ECE)").reset_index(drop=True)
    
    csv_path = os.path.join(OUTPUT_DIR, "calibration_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"\n  Result Summary Table (Sorted by lowest ECE):")
    print(metrics_df.to_string(index=False))
    print(f"\n  📄 Calibration metrics saved to {csv_path}")
    
    # Plot formatting
    ax.set_xlabel("Mean Predicted Probability (Confidence)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Fraction of Positives (Accuracy)", fontsize=12, fontweight="bold")
    ax.set_title("Calibration Curves (Reliability Diagram)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    
    plt.tight_layout()
    plot_path = "v3_calibration_analysis.pdf"
    save_pdf(fig, plot_path)
    plt.close(fig)
    print(f"  📄 Calibration curve plot saved to {OUTPUT_DIR}/{plot_path}")
    
    catat_waktu("Calibration Analysis", mulai)
    return metrics_df

# Standard main block to allow standalone execution
if __name__ == "__main__":
    # If run standalone, train standard models to verify
    print("Running Calibration Analysis Standalone...")
    import importlib
    data_preparation = importlib.import_module("src.01_data_preparation")
    preprocessing = importlib.import_module("src.02_preprocessing")
    smoteenn = importlib.import_module("src.03_smoteenn")
    
    X, y, df = data_preparation.load_and_prepare_data()
    X_train, X_test, X_train_sc, X_test_sc, y_train, y_test, scaler = preprocessing.split_and_scale(X, y)
    
    # Simple dictionary for testing
    from sklearn.linear_model import LogisticRegression
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import EditedNearestNeighbours
    
    pipe_lr = ImbPipeline([
        ('smote', SMOTE(random_state=SEED)),
        ('enn', EditedNearestNeighbours()),
        ('clf', LogisticRegression(random_state=SEED))
    ])
    pipe_lr.fit(X_train_sc, y_train)
    
    test_dict = {
        "Logistic Regression + SMOTE-ENN": pipe_lr
    }
    
    run_calibration_analysis(test_dict, X_test, X_test_sc, y_test)
