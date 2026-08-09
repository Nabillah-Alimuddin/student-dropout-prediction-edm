"""
13_smoteenn_analysis.py — Phase 12.6: SMOTE-ENN Impact Investigation.

Compares class distributions and model performance (metrics and confusion matrices)
between XGBoost Baseline (no resampling) and XGBoost + SMOTE-ENN (resampled)
to analyze the exact impact of SMOTE-ENN.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    balanced_accuracy_score,
)
from xgboost import XGBClassifier

from src.config import SEED, OUTPUT_DIR
from src.utils import catat_waktu, save_pdf, print_separator


def run_smoteenn_analysis(X_train_sc, y_train, X_res, y_res, X_test_sc, y_test, best_params):
    """
    Analyzes the impact of SMOTE-ENN resampling on class distribution and model performance.
    Compares Config B (XGBoost + Optuna, No Resampling) and Config G (XGBoost + Optuna + SMOTE-ENN)
    at a uniform threshold of 0.50 to isolate the exact effect of SMOTE-ENN.

    Args:
        X_train_sc (pd.DataFrame): Scaled training features (no resampling).
        y_train (pd.Series): Training target (no resampling).
        X_res (pd.DataFrame): Resampled training features.
        y_res (pd.Series): Resampled training target.
        X_test_sc (pd.DataFrame): Scaled test features.
        y_test (pd.Series): Test target.
        best_params (dict): Best parameters from Optuna tuning.
    """
    print_separator("PHASE 12.6: SMOTE-ENN IMPACT ANALYSIS")
    mulai = time.time()

    # ─── 1. Class Distribution Comparison ─────────────────────────────────
    dist_before = y_train.value_counts()
    dist_after = y_res.value_counts()

    print("  Class Distribution Comparison:")
    print(f"    Before Resampling:")
    print(f"      Graduate (0): {dist_before.get(0, 0)} ({dist_before.get(0, 0)/len(y_train)*100:.1f}%)")
    print(f"      Dropout  (1): {dist_before.get(1, 0)} ({dist_before.get(1, 0)/len(y_train)*100:.1f}%)")
    print(f"      Imbalance Ratio: {dist_before.get(1, 0)/dist_before.get(0, 0):.4f}")
    
    print(f"    After SMOTE-ENN:")
    print(f"      Graduate (0): {dist_after.get(0, 0)} ({dist_after.get(0, 0)/len(y_res)*100:.1f}%)")
    print(f"      Dropout  (1): {dist_after.get(1, 0)} ({dist_after.get(1, 0)/len(y_res)*100:.1f}%)")
    print(f"      Imbalance Ratio: {dist_after.get(1, 0)/dist_after.get(0, 0):.4f}")
    print()

    # ─── 2. Model Training & Evaluation ───────────────────────────────────
    # Config B: XGBoost + Optuna (No Resampling, Threshold = 0.50)
    model_baseline = XGBClassifier(**best_params)
    print("  Training Config B: XGBoost + Optuna (No Resampling)...")
    model_baseline.fit(X_train_sc, y_train, verbose=False)
    y_proba_base = model_baseline.predict_proba(X_test_sc)[:, 1]
    y_pred_base = (y_proba_base >= 0.50).astype(int)
    
    # Config G: XGBoost + Optuna + SMOTE-ENN (Resampled, Threshold = 0.50)
    model_resampled = XGBClassifier(**best_params)
    print("  Training Config G: XGBoost + Optuna + SMOTE-ENN (With Resampling)...")
    model_resampled.fit(X_res, y_res, verbose=False)
    y_proba_res = model_resampled.predict_proba(X_test_sc)[:, 1]
    y_pred_res = (y_proba_res >= 0.50).astype(int)

    # Calculate metrics
    metrics_base = {
        "Threshold": 0.50,
        "Precision (Dropout)": precision_score(y_test, y_pred_base, pos_label=1),
        "Recall (Dropout)": recall_score(y_test, y_pred_base, pos_label=1),
        "F1-Score (Dropout)": f1_score(y_test, y_pred_base, pos_label=1),
        "MCC": matthews_corrcoef(y_test, y_pred_base)
    }

    metrics_res = {
        "Threshold": 0.50,
        "Precision (Dropout)": precision_score(y_test, y_pred_res, pos_label=1),
        "Recall (Dropout)": recall_score(y_test, y_pred_res, pos_label=1),
        "F1-Score (Dropout)": f1_score(y_test, y_pred_res, pos_label=1),
        "MCC": matthews_corrcoef(y_test, y_pred_res)
    }

    comparison_df = pd.DataFrame([metrics_base, metrics_res], index=["XGBoost + Optuna (Config B)", "XGBoost + Optuna + SMOTE-ENN (Config G)"])
    print("  Performance Comparison (Threshold = 0.50):")
    print(comparison_df.to_string())
    print()

    # Save comparison metrics
    comparison_df.to_csv(f"{OUTPUT_DIR}/smoteenn_impact_metrics.csv")

    # ─── 3. Side-by-Side Confusion Matrices ───────────────────────────────
    cm_base = confusion_matrix(y_test, y_pred_base)
    cm_res = confusion_matrix(y_test, y_pred_res)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Baseline CM
    sns.heatmap(cm_base, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0],
                annot_kws={"size": 14, "weight": "bold"})
    axes[0].set_title("XGBoost + Optuna (Config B)\n(No Resampling, Thresh=0.50)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Predicted Label", fontsize=11)
    axes[0].set_ylabel("True Label", fontsize=11)
    axes[0].set_xticklabels(["Graduate (0)", "Dropout (1)"])
    axes[0].set_yticklabels(["Graduate (0)", "Dropout (1)"])

    # SMOTE-ENN CM
    sns.heatmap(cm_res, annot=True, fmt="d", cmap="Oranges", cbar=False, ax=axes[1],
                annot_kws={"size": 14, "weight": "bold"})
    axes[1].set_title("XGBoost + Optuna + SMOTE-ENN (Config G)\n(With Resampling, Thresh=0.50)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Predicted Label", fontsize=11)
    axes[1].set_ylabel("True Label", fontsize=11)
    axes[1].set_xticklabels(["Graduate (0)", "Dropout (1)"])
    axes[1].set_yticklabels(["Graduate (0)", "Dropout (1)"])

    plt.tight_layout()
    plot_path = "v3_smoteenn_confusion_matrices.pdf"
    save_pdf(fig, plot_path)
    plt.close(fig)
    print(f"  📄 Confusion matrix comparison saved to {OUTPUT_DIR}/{plot_path}")

    # ─── 4. Detailed Scientific Analysis/Discussion ──────────────────────
    print("\n  🔍 Scientific Analysis of SMOTE-ENN Impact:")
    
    # Analyze changes
    delta_recall = metrics_res["Recall (Dropout)"] - metrics_base["Recall (Dropout)"]
    delta_prec = metrics_res["Precision (Dropout)"] - metrics_base["Precision (Dropout)"]
    delta_f1 = metrics_res["F1-Score (Dropout)"] - metrics_base["F1-Score (Dropout)"]
    
    analysis_text = []
    analysis_text.append("### Impact of SMOTE-ENN on Student Dropout Prediction\n")
    analysis_text.append(f"1. **Class Balance shift**: SMOTE-ENN resampled the training set class ratio (Dropout:Graduate) from {dist_before.get(1, 0)/dist_before.get(0, 0):.3f} to {dist_after.get(1, 0)/dist_after.get(0, 0):.3f}. This allows the XGBoost classifier to learn from a balanced representation of both outcomes, mitigating the majority-class bias.\n")
    
    if delta_recall > 0:
        analysis_text.append(f"2. **Recall Improvement**: Applying SMOTE-ENN increased the Recall of the Dropout class by {delta_recall*100:+.2f}% (from {metrics_base['Recall (Dropout)']*100:.2f}% to {metrics_res['Recall (Dropout)']*100:.2f}%). This means the model successfully identified more actual dropout cases (reducing False Negatives from {cm_base[1, 0]} to {cm_res[1, 0]}), which is critical for early warning intervention.\n")
    else:
        analysis_text.append(f"2. **Recall Change**: Recall of the Dropout class changed by {delta_recall*100:+.2f}% (False Negatives: Config B={cm_base[1, 0]}, Config G={cm_res[1, 0]}).\n")
        
    if delta_prec < 0:
        analysis_text.append(f"3. **Precision Trade-off**: As expected with synthetic oversampling, there is a minor trade-off in Precision of {delta_prec*100:+.2f}% (from {metrics_base['Precision (Dropout)']*100:.2f}% to {metrics_res['Precision (Dropout)']*100:.2f}%), leading to an increase in False Positives from {cm_base[0, 1]} to {cm_res[0, 1]}. In educational institutions, a False Positive (misidentifying a graduate student as risk of dropout) is generally preferred over a False Negative (missing a dropout student entirely).\n")
    else:
        analysis_text.append(f"3. **Precision Change**: Precision of the Dropout class changed by {delta_prec*100:+.2f}% (False Positives: Config B={cm_base[0, 1]}, Config G={cm_res[0, 1]}).\n")
        
    analysis_text.append(f"4. **Overall Metric (F1-Score & MCC)**: The overall F1-Score changed by {delta_f1*100:+.2f}%. This indicates whether the harmonic mean of precision and recall improves with resampling. Combined with the Matthews Correlation Coefficient (MCC) change, we can assess if the resampling has a net positive effect on the classifier's discriminative ability across both classes.\n")

    analysis_markdown = "".join(analysis_text)
    print("".join(["    " + line for line in analysis_markdown.splitlines(keepends=True)]))

    # Save analysis as md file
    with open(f"{OUTPUT_DIR}/smoteenn_impact_analysis.md", "w") as f:
        f.write(analysis_markdown)
    print(f"  💾 Analysis report saved to {OUTPUT_DIR}/smoteenn_impact_analysis.md")

    catat_waktu("SMOTE-ENN Impact Analysis", mulai)


# ----------------------------------------------------------------------
# Threshold Sensitivity Analysis (Recall‑first Early‑Warning)
# ----------------------------------------------------------------------
def threshold_sensitivity_analysis(
    model,
    X_test,
    y_test,
    thresholds: list[float] | None = None,
    pos_label: int = 1,
    seed: int = 42,
    pdf_name: str = "threshold_sensitivity_analysis.pdf",
) -> tuple[pd.DataFrame, float | None]:
    """
    Evaluates Recall, Precision, F1‑Dropout and Balanced Accuracy across a range of
    probability thresholds and visualises the results.

    Parameters
    ----------
    model : StackingEnsemble or XGBClassifier
        Trained binary classifier.
    X_test : array‑like
        Test features (unscaled if model has internal scaling).
    y_test : array‑like
        True test labels.
    thresholds : list of float, optional
        Threshold values to evaluate.  Default = np.arange(0.30, 0.71, 0.05).
    pos_label : int, default 1
        Label representing the *Dropout* class.
    seed : int, default 42
        Random seed (kept for reproducibility).
    pdf_name : str, default "threshold_sensitivity_analysis.pdf"
        Filename (saved under OUTPUT_DIR) for the PDF plot.

    Returns
    -------
    metrics_df : pd.DataFrame
        Rows per threshold with columns: Threshold, Recall, Precision,
        F1‑Dropout, Balanced_Acc.
    optimal_thr : float | None
        Smallest threshold where Recall ≥ 0.90 (or None if not reached).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if thresholds is None:
        thresholds = np.arange(0.30, 0.71, 0.05)

    # 1️⃣ Compute probabilities once (auto-scales if model has internal scaler)
    proba = model.predict_proba(X_test)[:, 1]

    # 2️⃣ Evaluate metrics for each threshold
    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        rec = recall_score(y_test, pred, pos_label=pos_label, zero_division=0)
        prec = precision_score(y_test, pred, pos_label=pos_label, zero_division=0)
        f1 = f1_score(y_test, pred, pos_label=pos_label, zero_division=0)
        bal = balanced_accuracy_score(y_test, pred)
        rows.append({
            "Threshold": round(t, 3),
            "Recall": rec,
            "Precision": prec,
            "F1‑Dropout": f1,
            "Balanced_Acc": bal,
        })
    metrics_df = pd.DataFrame(rows)

    # 3️⃣ Determine optimal threshold (Recall ≥ 0.90)
    opt_candidates = metrics_df[metrics_df["Recall"] >= 0.90]
    optimal_thr = opt_candidates["Threshold"].min() if not opt_candidates.empty else None

    # 4️⃣ Plot all metrics
    plt.figure(figsize=(10, 6))
    for metric, style, color in zip(
        ["Recall", "Precision", "F1‑Dropout", "Balanced_Acc"],
        ["-o", "--s", "-.^", "--d"],
        ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"],
    ):
        plt.plot(
            metrics_df["Threshold"],
            metrics_df[metric],
            style,
            color=color,
            linewidth=2,
            markersize=6,
            label=metric,
        )

    if optimal_thr is not None:
        plt.axvline(
            optimal_thr,
            color="#f1c40f",
            linestyle=":",
            linewidth=2,
            label=f"Optimal (Recall≥0.90) – {optimal_thr:.2f}",
        )
        plt.text(
            optimal_thr + 0.01,
            0.05,
            f"{optimal_thr:.2f}",
            rotation=90,
            color="#f1c40f",
            fontsize=10,
            verticalalignment="bottom",
        )

    plt.title(
        "Threshold Sensitivity Analysis – Early‑Warning (Recall‑first)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Probability Threshold", fontsize=12)
    plt.ylabel("Metric Score", fontsize=12)
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=10, loc="lower left")
    plt.tight_layout()

    # 5️⃣ Save figure and CSV
    plot_path = pdf_name
    save_pdf(plt.gcf(), plot_path)
    plt.close()
    csv_path = "threshold_sensitivity_metrics.csv"
    metrics_df.to_csv(f"{OUTPUT_DIR}/{csv_path}", index=False)
    print(f"📊 Plot saved to {OUTPUT_DIR}/{plot_path}")
    print(f"🗃️ Metrics table saved to {OUTPUT_DIR}/{csv_path}")

    return metrics_df, optimal_thr

# ----------------------------------------------------------------------
# Error Profile Analysis for Proposed Model (XGBoost + SMOTE‑ENN)
# ----------------------------------------------------------------------
def error_profile_analysis(
    model: XGBClassifier,
    X_test_raw: pd.DataFrame,
    X_test_sc,
    y_test,
    threshold: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate an error‑profile table for the proposed model.

    The function classifies each of the *726* test samples into TP, TN, FP, FN
    based on the supplied ``threshold`` (default 0.5). For each error category
    it computes the mean value of the 10 raw features of interest and returns a
    ``pd.DataFrame`` where rows are categories and columns are the feature
    means.

    Parameters
    ----------
    model : XGBClassifier
        Trained XGBoost model (the SMOTE‑ENN‑resampled version).
    X_test_raw : pd.DataFrame
        Original, **unscaled** test features – required for meaningful mean
        values.
    X_test_sc : array‑like
        Scaled test features used for prediction.
    y_test : array‑like
        True binary labels.
    threshold : float, optional
        Probability cut‑off for converting probabilities to class labels.
    seed : int, optional
        Random seed (kept for reproducibility; not actively used here).

    Returns
    -------
    pd.DataFrame
        Table with rows ``[TP, TN, FP, FN]`` and columns for the selected
        features containing the average raw values.
    """
    # 1️⃣ Predict probabilities and convert to hard labels
    proba = model.predict_proba(X_test_sc)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    # 2️⃣ Define masks for each error type
    tp_mask = (y_test == 1) & (y_pred == 1)
    tn_mask = (y_test == 0) & (y_pred == 0)
    fp_mask = (y_test == 0) & (y_pred == 1)
    fn_mask = (y_test == 1) & (y_pred == 0)

    masks = {
        "TP": tp_mask,
        "TN": tn_mask,
        "FP": fp_mask,
        "FN": fn_mask,
    }

    # 3️⃣ Features of interest
    features = [
        "Curricular units 2nd sem (approved)",
        "Curricular units 1st sem (approved)",
        "Curricular units 2nd sem (grade)",
        "Tuition fees up to date",
        "Scholarship holder",
        "Course",
        "Curricular units 1st sem (grade)",
        "Debtor",
        "Age at enrollment",
        "Curricular units 2nd sem (evaluations)",
    ]

    # 4️⃣ Compute mean raw values per category
    rows = []
    for cat, mask in masks.items():
        if mask.sum() == 0:
            mean_vals = {f: np.nan for f in features}
        else:
            mean_vals = X_test_raw.loc[mask, features].mean().to_dict()
        row = {"Category": cat, **mean_vals, "Count": int(mask.sum())}
        rows.append(row)

    profile_df = pd.DataFrame(rows).set_index("Category")

    # 5️⃣ Save results and print a brief interpretation for FN
    csv_path = "error_profile_proposed_model.csv"
    profile_df.to_csv(f"{OUTPUT_DIR}/{csv_path}")
    print(f"📄 Error‑profile table saved to {OUTPUT_DIR}/{csv_path}")
    print(profile_df)

    # Interpretation focusing on FN
    fn_means = profile_df.loc["FN", features]
    tp_means = profile_df.loc["TP", features]
    diff = fn_means - tp_means
    print("\n🧐 Interpretation of False‑Negatives (undetected dropouts):")
    for f in features:
        d = diff[f]
        if pd.isna(d):
            continue
        if d > 0:
            print(f"  • {f}: FN samples have higher average ({fn_means[f]:.2f}) than TP ({tp_means[f]:.2f}).")
        elif d < 0:
            print(f"  • {f}: FN samples have lower average ({fn_means[f]:.2f}) than TP ({tp_means[f]:.2f}).")

    return profile_df
