"""
18_bootstrap_ci.py — Poin 2: Bootstrap Confidence Interval for Test F1-Score.

Calculates the 95% Bootstrap Confidence Interval for the F1-Score on the test set
(1,000 resamples of the test set) and compares it with the 10-Fold CV results.
Provides empirical evidence of whether the CV-Test gap is within statistical variance.
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from src.config import OUTPUT_DIR, SEED
from src.utils import catat_waktu, print_separator

def run_bootstrap_ci(y_test, y_pred, cv_results, n_bootstrap=1000, seed=SEED):
    """
    Compute bootstrap confidence interval for test F1-score and compare with CV.
    
    Args:
        y_test (array-like): Ground truth labels for the test set.
        y_pred (array-like): Predicted labels for the test set.
        cv_results (pd.DataFrame): 10-Fold CV results.
        n_bootstrap (int): Number of bootstrap resamples.
        seed (int): Random seed for reproducibility.
        
    Returns:
        results_dict (dict): Dictionary of computed stats.
    """
    print_separator("POIN 2: BOOTSTRAP CONFIDENCE INTERVAL (CV vs TEST GAP ANALYSIS)")
    mulai = time.time()
    
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    n_samples = len(y_test)
    
    # Set seed
    rng = np.random.default_rng(seed)
    
    # Bootstrap resampling
    bootstrap_f1s = []
    print(f"  Performing {n_bootstrap} bootstrap resamples of the test set ({n_samples} samples)...")
    for _ in range(n_bootstrap):
        # Sample indices with replacement
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        y_test_resampled = y_test[indices]
        y_pred_resampled = y_pred[indices]
        
        # Calculate F1-Score for Dropout (class 1)
        f1 = f1_score(y_test_resampled, y_pred_resampled, pos_label=1, zero_division=0)
        bootstrap_f1s.append(f1)
        
    bootstrap_f1s = np.array(bootstrap_f1s)
    
    # Calculate 95% Confidence Interval
    ci_lower = np.percentile(bootstrap_f1s, 2.5)
    ci_upper = np.percentile(bootstrap_f1s, 97.5)
    test_f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    
    print(f"  Test Set F1-Score: {test_f1:.4f}")
    print(f"  95% Bootstrap Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # CV Stats
    cv_mean = cv_results["F1-Dropout"].mean()
    cv_std = cv_results["F1-Dropout"].std()
    cv_range_lower = cv_mean - cv_std
    cv_range_upper = cv_mean + cv_std
    print(f"  10-Fold CV F1-Score: {cv_mean:.4f} ± {cv_std:.4f} (Range: [{cv_range_lower:.4f}, {cv_range_upper:.4f}])")
    
    # Overlap analysis
    # Overlap occurs if the CV mean range intersects with the bootstrap CI
    overlap = not (cv_range_upper < ci_lower or cv_range_lower > ci_upper)
    overlap_status = "✅ YES (Overlap)" if overlap else "❌ NO (No Overlap)"
    print(f"  Overlap between Bootstrap 95% CI and CV Mean±Std: {overlap_status}")
    
    if overlap:
        conclusion = (
            "Evidence supports that the gap between CV and Test F1-Score is due to "
            "sampling variance of the test set, NOT overfitting to the test set."
        )
    else:
        conclusion = (
            "The gap between CV and Test F1-Score lies outside normal sampling variance, "
            "suggesting possible split-specific variance or mild overfitting."
        )
    print(f"  Conclusion: {conclusion}")
    
    # Save results
    results_df = pd.DataFrame([{
        "Metric": "F1-Dropout",
        "Test F1-Score": round(test_f1, 4),
        "Bootstrap 95% CI Lower": round(ci_lower, 4),
        "Bootstrap 95% CI Upper": round(ci_upper, 4),
        "CV Mean": round(cv_mean, 4),
        "CV Std": round(cv_std, 4),
        "CV Range Lower": round(cv_range_lower, 4),
        "CV Range Upper": round(cv_range_upper, 4),
        "Overlap": overlap,
        "Conclusion": conclusion
    }])
    
    csv_path = os.path.join(OUTPUT_DIR, "bootstrap_ci_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"  📄 Saved bootstrap CI results to {csv_path}")
    
    # Generate visualization
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(bootstrap_f1s, bins=30, color="#3498db", alpha=0.6, edgecolor="black", label="Bootstrap F1-Scores")
    ax.axvline(test_f1, color="red", linestyle="-", linewidth=2.5, label=f"Test F1 ({test_f1:.4f})")
    ax.axvline(ci_lower, color="red", linestyle="--", linewidth=1.5, label="Bootstrap 95% CI")
    ax.axvline(ci_upper, color="red", linestyle="--", linewidth=1.5)
    
    # CV shaded region
    ax.axvspan(cv_range_lower, cv_range_upper, color="#2ecc71", alpha=0.3, label=f"CV Mean±Std ({cv_mean:.4f}±{cv_std:.4f})")
    ax.axvline(cv_mean, color="green", linestyle="-", linewidth=2)
    
    ax.set_title("Bootstrap Distribution of Test F1-Score vs 10-Fold CV Range", fontsize=12, fontweight="bold")
    ax.set_xlabel("F1-Score (Dropout)")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_filename = "v3_bootstrap_ci_comparison.pdf"
    from src.utils import save_pdf
    save_pdf(fig, plot_filename)
    plt.close(fig)
    print(f"  📄 Saved bootstrap distribution plot to {OUTPUT_DIR}/{plot_filename}")
    
    catat_waktu("Bootstrap CI Analysis", mulai)
    
    return {
        "test_f1": test_f1,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "overlap": overlap,
        "conclusion": conclusion
    }
