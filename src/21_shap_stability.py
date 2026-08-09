"""
21_shap_stability.py — Poin 5: Bootstrap SHAP Stability Analysis.

Evaluates the stability of SHAP feature importance rankings by bootstrapping
the test set (1000 resamples with replacement) and recalculating the mean
absolute SHAP values. Provides statistics on ranking stability and frequency of
appearance in the top-5 feature list.
"""

import os
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from src.config import OUTPUT_DIR, SEED
from src.utils import catat_waktu, print_separator, save_pdf

def run_shap_stability(X_test, shap_values, n_bootstrap=1000, seed=SEED):
    """
    Perform bootstrap stability analysis on pre-calculated SHAP values.
    
    Args:
        X_test (pd.DataFrame): Test features.
        shap_values (np.ndarray): Pre-calculated SHAP values for X_test.
        n_bootstrap (int): Number of bootstrap resamples.
        seed (int): Random seed for reproducibility.
        
    Returns:
        stability_df (pd.DataFrame): Summary of feature stability metrics.
    """
    print_separator("POIN 5: BOOTSTRAP SHAP STABILITY ANALYSIS")
    mulai = time.time()
    
    n_samples = X_test.shape[0]
    n_features = X_test.shape[1]
    feature_names = X_test.columns.tolist()
    
    # Calculate original ranking
    orig_mean_abs = np.abs(shap_values).mean(axis=0)
    orig_rank_indices = np.argsort(orig_mean_abs)[::-1]
    orig_ranking = {feature_names[idx]: i+1 for i, idx in enumerate(orig_rank_indices)}
    
    # Bootstrap ranking simulation
    rng = np.random.default_rng(seed)
    bootstrap_ranks = {name: [] for name in feature_names}
    bootstrap_means = {name: [] for name in feature_names}
    
    print(f"  Simulating {n_bootstrap} bootstrap resamples of SHAP values ({n_samples} samples)...")
    
    for _ in range(n_bootstrap):
        # Resample indices with replacement
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        resampled_shap = shap_values[indices]
        
        # Calculate mean absolute SHAP for this resample
        resampled_mean_abs = np.abs(resampled_shap).mean(axis=0)
        
        # Rank features (descending order)
        # Note: argpartition or argsort can be used. argsort is clean here.
        rank_indices = np.argsort(resampled_mean_abs)[::-1]
        
        for rank, feat_idx in enumerate(rank_indices):
            feat_name = feature_names[feat_idx]
            bootstrap_ranks[feat_name].append(rank + 1) # 1-indexed rank
            bootstrap_means[feat_name].append(resampled_mean_abs[feat_idx])
            
    # Compile stability statistics
    stability_rows = []
    for name in feature_names:
        ranks = np.array(bootstrap_ranks[name])
        means = np.array(bootstrap_means[name])
        
        in_top_5 = np.sum(ranks <= 5) / n_bootstrap * 100
        in_top_10 = np.sum(ranks <= 10) / n_bootstrap * 100
        
        stability_rows.append({
            "Feature": name,
            "Original Mean |SHAP|": orig_mean_abs[feature_names.index(name)],
            "Original Rank": orig_ranking[name],
            "Mean Bootstrap Rank": np.mean(ranks),
            "Std Bootstrap Rank": np.std(ranks),
            "Min Bootstrap Rank": np.min(ranks),
            "Max Bootstrap Rank": np.max(ranks),
            "Freq in Top 5 (%)": in_top_5,
            "Freq in Top 10 (%)": in_top_10,
        })
        
    df_stability = pd.DataFrame(stability_rows)
    df_stability = df_stability.sort_values(by="Original Rank").reset_index(drop=True)
    
    # Save to CSV
    csv_path = os.path.join(OUTPUT_DIR, "shap_stability_results.csv")
    df_stability.to_csv(csv_path, index=False)
    print(f"  📄 Saved SHAP stability results to {csv_path}")
    
    # Print top-10 stability
    print("\n  Top 10 Feature Stability Summary:")
    print(df_stability.head(10)[["Feature", "Original Rank", "Mean Bootstrap Rank", "Std Bootstrap Rank", "Freq in Top 5 (%)"]].to_string(index=False))
    
    # Generate visualization
    import matplotlib.pyplot as plt
    
    # Plot top 10 features' rank distributions as boxplots
    top_10_features = df_stability.head(10)["Feature"].tolist()
    plot_data = [bootstrap_ranks[feat] for feat in top_10_features]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Horizontal boxplot
    bp = ax.boxplot(plot_data, vert=False, patch_artist=True, labels=top_10_features)
    
    # Coloring boxplots nicely
    for patch in bp['boxes']:
        patch.set_facecolor('#3498db')
        patch.set_alpha(0.6)
        patch.set_edgecolor('black')
        
    for median in bp['medians']:
        median.set_color('red')
        median.set_linewidth(2)
        
    ax.set_title("Bootstrap Distribution of SHAP Feature Ranks (Top 10 Features)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Feature Rank (Lower is more important)")
    ax.invert_yaxis()  # Put the top feature at the top
    ax.grid(axis="x", alpha=0.3)
    
    plt.tight_layout()
    plot_filename = "v3_shap_stability_boxplot.pdf"
    save_pdf(fig, plot_filename)
    plt.close(fig)
    print(f"  📄 Saved SHAP stability boxplot to {OUTPUT_DIR}/{plot_filename}")
    
    catat_waktu("SHAP Stability Analysis", mulai)
    return df_stability
