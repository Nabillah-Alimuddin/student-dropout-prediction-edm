"""
20_cost_sensitive_threshold.py — Poin 4: Cost-Sensitive Threshold Optimization.

Performs cost-sensitive threshold selection using expected cost minimization
where False Negatives (FN) are assumed to be r times more costly than
False Positives (FP) (r = 1, 2, 3, 5, 7, 10).
The primary default ratio is 3:1 (FN:FP), with 5:1 as a sensitivity check.
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
from src.config import OUTPUT_DIR
from src.utils import catat_waktu, print_separator, save_pdf

def run_cost_sensitive_analysis(y_test, y_proba, optimal_threshold_f1, cost_ratios=[1, 2, 3, 5, 7, 10]):
    """
    Perform cost-sensitive analysis for predicted probabilities.
    
    Args:
        y_test (array-like): Ground truth labels.
        y_proba (array-like): Predicted probabilities.
        optimal_threshold_f1 (float): F1-optimized threshold (e.g. 0.69).
        cost_ratios (list): List of cost ratios r (FN:FP).
        
    Returns:
        df_summary (pd.DataFrame): Threshold results per cost ratio.
    """
    print_separator("POIN 4: COST-SENSITIVE THRESHOLD OPTIMIZATION (EXPECTED COST MINIMIZATION)")
    mulai = time.time()
    
    y_test = np.array(y_test)
    y_proba = np.array(y_proba)
    
    thresholds = np.arange(0.01, 1.00, 0.01)
    results = []
    
    # Calculate costs for all thresholds and all ratios
    for r in cost_ratios:
        best_cost = float("inf")
        best_t = 0.50
        
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            # Cost function: FP * 1 + FN * r
            cost = fp * 1.0 + fn * float(r)
            
            if cost < best_cost:
                best_cost = cost
                best_t = t
                
        # Calculate metrics at the cost-optimal threshold
        y_pred_opt = (y_proba >= best_t).astype(int)
        rec_opt = recall_score(y_test, y_pred_opt, pos_label=1, zero_division=0)
        prec_opt = precision_score(y_test, y_pred_opt, pos_label=1, zero_division=0)
        f1_opt = f1_score(y_test, y_pred_opt, pos_label=1, zero_division=0)
        cm_opt = confusion_matrix(y_test, y_pred_opt)
        _, fp_opt, fn_opt, _ = cm_opt.ravel()
        
        # Calculate baseline costs (at default threshold 0.50 and F1-optimal threshold)
        y_pred_def = (y_proba >= 0.50).astype(int)
        cm_def = confusion_matrix(y_test, y_pred_def)
        _, fp_def, fn_def, _ = cm_def.ravel()
        cost_def = fp_def * 1.0 + fn_def * float(r)
        
        y_pred_f1 = (y_proba >= optimal_threshold_f1).astype(int)
        cm_f1 = confusion_matrix(y_test, y_pred_f1)
        _, fp_f1, fn_f1, _ = cm_f1.ravel()
        cost_f1 = fp_f1 * 1.0 + fn_f1 * float(r)
        
        # Savings compared to F1 threshold and default 0.50 threshold
        saving_vs_def = (cost_def - best_cost) / cost_def * 100 if cost_def > 0 else 0
        saving_vs_f1 = (cost_f1 - best_cost) / cost_f1 * 100 if cost_f1 > 0 else 0
        
        print(f"  Ratio {r}:1 (FN:FP) -> Optimal Threshold: {best_t:.2f} | Recall: {rec_opt:.4f} | FP: {fp_opt}, FN: {fn_opt}")
        print(f"             Total Cost: {best_cost:.1f} (Saved vs default 0.50: {saving_vs_def:.1f}%, vs F1-opt: {saving_vs_f1:.1f}%)")
        
        results.append({
            "Cost Ratio (FN:FP)": f"{r}:1",
            "Ratio Value": r,
            "Optimal Threshold": round(best_t, 2),
            "Recall": round(rec_opt, 4),
            "Precision": round(prec_opt, 4),
            "F1-Dropout": round(f1_opt, 4),
            "False Positives (FP)": fp_opt,
            "False Negatives (FN)": fn_opt,
            "Optimal Cost": round(best_cost, 1),
            "Default 0.50 Cost": round(cost_def, 1),
            "F1-Opt Cost": round(cost_f1, 1),
            "Savings vs Default (%)": round(saving_vs_def, 2),
            "Savings vs F1-Opt (%)": round(saving_vs_f1, 2)
        })
        
    df_results = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, "cost_sensitive_thresholds.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n  📄 Saved cost-sensitive thresholds to {csv_path}")
    
    # Generate visualization
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ["#7f8c8d", "#3498db", "#2ecc71", "#e74c3c"]
    selected_ratios = [1, 3, 5, 10]
    
    for idx, r in enumerate(selected_ratios):
        costs = []
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            cm = confusion_matrix(y_test, y_pred)
            _, fp, fn, _ = cm.ravel()
            costs.append(fp * 1.0 + fn * float(r))
            
        costs = np.array(costs)
        min_idx = np.argmin(costs)
        min_t = thresholds[min_idx]
        min_c = costs[min_idx]
        
        label = f"Ratio {r}:1 (Min Cost Threshold = {min_t:.2f})"
        ax.plot(thresholds, costs, color=colors[idx % len(colors)], linewidth=2, label=label)
        ax.scatter(min_t, min_c, color="black", s=50, zorder=5)
        
    ax.axvline(optimal_threshold_f1, color="red", linestyle="--", alpha=0.7, label=f"F1-Max Threshold = {optimal_threshold_f1:.2f}")
    ax.axvline(0.50, color="blue", linestyle=":", alpha=0.5, label="Default Threshold = 0.50")
    
    ax.set_title("Expected Intervention Cost vs Probability Threshold", fontsize=12, fontweight="bold")
    ax.set_xlabel("Probability Threshold")
    ax.set_ylabel("Expected Total Cost (FP Cost = 1, FN Cost = Ratio)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_filename = "v3_cost_sensitive_curves.pdf"
    save_pdf(fig, plot_filename)
    plt.close(fig)
    print(f"  📄 Saved expected cost curves plot to {OUTPUT_DIR}/{plot_filename}")
    
    catat_waktu("Cost-Sensitive Threshold Analysis", mulai)
    return df_results
