"""
22_subgroup_fairness.py — Poin 7: Subgroup Fairness Analysis.

Analyzes the fairness of the proposed Stacking Ensemble model predictions
across key demographic and socioeconomic subgroups: Gender, Age at enrollment,
Scholarship holder, and Debtor. Evaluates Recall, Precision, and F1-Score
for each subgroup, computes disparity ratios (four-fifths rule), and performs
statistical significance tests on prediction differences.
"""

import os
import time
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.metrics import recall_score, precision_score, f1_score
from src.config import OUTPUT_DIR
from src.utils import catat_waktu, print_separator, save_pdf

def run_subgroup_fairness(model, X_test, y_test, y_pred):
    """
    Perform subgroup fairness analysis on the test set.
    
    Args:
        model: Trained StackingEnsemble model.
        X_test (pd.DataFrame): Unscaled test features.
        y_test (pd.Series): Test labels.
        y_pred (np.ndarray): Model predictions on the test set.
        
    Returns:
        fairness_df (pd.DataFrame): Detailed subgroup metrics.
    """
    print_separator("POIN 7: SUBGROUP FAIRNESS ANALYSIS")
    mulai = time.time()
    
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    
    # 1. Define subgroups mapping
    # Note: Gender (1=male; 0=female), Scholarship holder (1=yes; 0=no), Debtor (1=yes; 0=no)
    subgroup_definitions = {
        "Gender": {
            "column": "Gender",
            "mapping": {0: "Female", 1: "Male"}
        },
        "Scholarship holder": {
            "column": "Scholarship holder",
            "mapping": {0: "Non-Holder", 1: "Scholarship Holder"}
        },
        "Debtor": {
            "column": "Debtor",
            "mapping": {0: "Non-Debtor", 1: "Debtor"}
        }
    }
    
    fairness_rows = []
    
    # Analyze binary subgroups
    for group_name, info in subgroup_definitions.items():
        col = info["column"]
        mapping = info["mapping"]
        
        if col not in X_test.columns:
            print(f"  ⚠️ Column {col} not found in test features. Skipping...")
            continue
            
        group_vals = X_test[col].values
        
        # Calculate metrics for each subgroup
        subgroup_metrics = {}
        for val, label in mapping.items():
            mask = (group_vals == val)
            n_sub = np.sum(mask)
            
            if n_sub == 0:
                continue
                
            y_t_sub = y_test[mask]
            y_p_sub = y_pred[mask]
            
            rec = recall_score(y_t_sub, y_p_sub, pos_label=1, zero_division=0)
            prec = precision_score(y_t_sub, y_p_sub, pos_label=1, zero_division=0)
            f1 = f1_score(y_t_sub, y_p_sub, pos_label=1, zero_division=0)
            n_dropout = np.sum(y_t_sub == 1)
            
            subgroup_metrics[label] = {
                "Recall": rec,
                "Precision": prec,
                "F1-Score": f1,
                "N": n_sub,
                "N_Dropout": n_dropout
            }
            
            fairness_rows.append({
                "Group Type": group_name,
                "Subgroup": label,
                "Total Samples (N)": n_sub,
                "Actual Dropouts": n_dropout,
                "Recall (Dropout)": round(rec, 4),
                "Precision (Dropout)": round(prec, 4),
                "F1-Score (Dropout)": round(f1, 4)
            })
            
        # Statistical test & disparity for this group type
        labels = list(subgroup_metrics.keys())
        if len(labels) == 2:
            label1, label2 = labels[0], labels[1]
            rec1 = subgroup_metrics[label1]["Recall"]
            rec2 = subgroup_metrics[label2]["Recall"]
            
            # Disparity ratio (min / max)
            disp_ratio = min(rec1, rec2) / max(rec1, rec2) if max(rec1, rec2) > 0 else 1.0
            four_fifths_rule = "✅ Fair" if disp_ratio >= 0.80 else "⚠️ Disparate Impact"
            
            # Statistical test for recall: form a contingency table of Correct (TP) vs Incorrect (FN) for actual dropouts
            # Group 1 actual dropouts
            mask1 = (group_vals == list(mapping.keys())[0]) & (y_test == 1)
            tp1 = np.sum(y_pred[mask1] == 1)
            fn1 = np.sum(y_pred[mask1] == 0)
            
            # Group 2 actual dropouts
            mask2 = (group_vals == list(mapping.keys())[1]) & (y_test == 1)
            tp2 = np.sum(y_pred[mask2] == 1)
            fn2 = np.sum(y_pred[mask2] == 0)
            
            table = np.array([[tp1, fn1], [tp2, fn2]])
            
            # Use Fisher's exact test since cells can be small
            _, p_val = fisher_exact(table)
            sig_diff = "Yes (Significant)" if p_val < 0.05 else "No"
            
            print(f"  {group_name} Fairness Check:")
            print(f"    {label1} Recall: {rec1:.4f} (N={subgroup_metrics[label1]['N']}, Dropouts={subgroup_metrics[label1]['N_Dropout']})")
            print(f"    {label2} Recall: {rec2:.4f} (N={subgroup_metrics[label2]['N']}, Dropouts={subgroup_metrics[label2]['N_Dropout']})")
            print(f"    Disparity Ratio: {disp_ratio:.4f} ({four_fifths_rule}) | Fisher's Test p-val: {p_val:.4f} (Sig Diff: {sig_diff})")
            
            # Store in the rows
            for row in fairness_rows:
                if row["Group Type"] == group_name:
                    row["Recall Disparity Ratio"] = round(disp_ratio, 4)
                    row["Recall Fair (4/5 Rule)"] = four_fifths_rule
                    row["Fisher Exact p-value"] = round(p_val, 4)
                    row["Significant Difference"] = sig_diff

    # 2. Analyze continuous Age at enrollment binned: <=20, 21-25, >25
    age_col = "Age at enrollment"
    if age_col in X_test.columns:
        age_vals = X_test[age_col].values
        
        # Bin age
        age_bins = []
        for age in age_vals:
            if age <= 20:
                age_bins.append("<=20")
            elif age <= 25:
                age_bins.append("21-25")
            else:
                age_bins.append(">25")
        age_bins = np.array(age_bins)
        
        unique_bins = ["<=20", "21-25", ">25"]
        age_metrics = {}
        
        for label in unique_bins:
            mask = (age_bins == label)
            n_sub = np.sum(mask)
            
            if n_sub == 0:
                continue
                
            y_t_sub = y_test[mask]
            y_p_sub = y_pred[mask]
            
            rec = recall_score(y_t_sub, y_p_sub, pos_label=1, zero_division=0)
            prec = precision_score(y_t_sub, y_p_sub, pos_label=1, zero_division=0)
            f1 = f1_score(y_t_sub, y_p_sub, pos_label=1, zero_division=0)
            n_dropout = np.sum(y_t_sub == 1)
            
            age_metrics[label] = {
                "Recall": rec,
                "Precision": prec,
                "F1-Score": f1,
                "N": n_sub,
                "N_Dropout": n_dropout
            }
            
            fairness_rows.append({
                "Group Type": "Age Group",
                "Subgroup": label,
                "Total Samples (N)": n_sub,
                "Actual Dropouts": n_dropout,
                "Recall (Dropout)": round(rec, 4),
                "Precision (Dropout)": round(prec, 4),
                "F1-Score (Dropout)": round(f1, 4)
            })
            
        # Calculate disparity for Age
        rec_vals = [age_metrics[l]["Recall"] for l in unique_bins if l in age_metrics]
        if len(rec_vals) > 1:
            disp_ratio = min(rec_vals) / max(rec_vals) if max(rec_vals) > 0 else 1.0
            four_fifths_rule = "✅ Fair" if disp_ratio >= 0.80 else "⚠️ Disparate Impact"
            
            # Chi-squared test for multiple age groups
            contingency_list = []
            for label in unique_bins:
                mask_age = (age_bins == label) & (y_test == 1)
                tp = np.sum(y_pred[mask_age] == 1)
                fn = np.sum(y_pred[mask_age] == 0)
                contingency_list.append([tp, fn])
                
            table = np.array(contingency_list)
            # Check if all row sums are > 0 to avoid zero division in chi2
            if np.all(table.sum(axis=1) > 0):
                try:
                    _, p_val, _, _ = chi2_contingency(table)
                    sig_diff = "Yes (Significant)" if p_val < 0.05 else "No"
                except Exception:
                    p_val = np.nan
                    sig_diff = "Error"
            else:
                p_val = np.nan
                sig_diff = "Insufficient Data"
                
            print(f"  Age Group Fairness Check:")
            for label in unique_bins:
                if label in age_metrics:
                    print(f"    {label} Recall: {age_metrics[label]['Recall']:.4f} (N={age_metrics[label]['N']}, Dropouts={age_metrics[label]['N_Dropout']})")
            print(f"    Disparity Ratio: {disp_ratio:.4f} ({four_fifths_rule}) | Chi-squared Test p-val: {p_val:.4f} (Sig Diff: {sig_diff})")
            
            # Store in the rows
            for row in fairness_rows:
                if row["Group Type"] == "Age Group":
                    row["Recall Disparity Ratio"] = round(disp_ratio, 4)
                    row["Recall Fair (4/5 Rule)"] = four_fifths_rule
                    row["Fisher/Chi2 p-value"] = round(p_val, 4) if not np.isnan(p_val) else None
                    row["Significant Difference"] = sig_diff
                    
    df_fairness = pd.DataFrame(fairness_rows)
    csv_path = os.path.join(OUTPUT_DIR, "subgroup_fairness_results.csv")
    df_fairness.to_csv(csv_path, index=False)
    print(f"\n  📄 Saved subgroup fairness results to {csv_path}")
    
    # Generate visualization
    import matplotlib.pyplot as plt
    
    # Filter unique subgroups for plotting
    plot_df = df_fairness[["Subgroup", "Recall (Dropout)", "Precision (Dropout)", "F1-Score (Dropout)"]].copy()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(plot_df["Subgroup"]))
    width = 0.25
    
    ax.bar(x - width, plot_df["Recall (Dropout)"], width, label="Recall (Sensitivity)", color="#e74c3c", edgecolor="black", linewidth=0.5)
    ax.bar(x, plot_df["Precision (Dropout)"], width, label="Precision", color="#3498db", edgecolor="black", linewidth=0.5)
    ax.bar(x + width, plot_df["F1-Score (Dropout)"], width, label="F1-Score", color="#2ecc71", edgecolor="black", linewidth=0.5)
    
    ax.set_title("Proposed Model Performance Across Subgroups (Fairness Evaluation)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Subgroup")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Subgroup"], rotation=15)
    ax.set_ylim(0.5, 1.05)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    
    # Draw reference recall line
    overall_recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    ax.axhline(overall_recall, color="black", linestyle="--", alpha=0.7, label=f"Overall Recall ({overall_recall:.4f})")
    ax.legend(loc="lower right")
    
    plt.tight_layout()
    plot_filename = "v3_subgroup_fairness.pdf"
    save_pdf(fig, plot_filename)
    plt.close(fig)
    print(f"  📄 Saved subgroup fairness plot to {OUTPUT_DIR}/{plot_filename}")
    
    catat_waktu("Subgroup Fairness Analysis", mulai)
    return df_fairness
