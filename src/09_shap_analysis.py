"""
09_shap_analysis.py — Phase 11: SHAP Explainability Analysis.

TreeExplainer for XGBoost: beeswarm, global bar, top-20 ranking table (CSV),
dependence plots (top 3 features), and individual waterfall plots.
"""

import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from src.config import SEED, OUTPUT_DIR
from src.utils import catat_waktu, save_pdf, print_separator


def run_shap_analysis(model, X_test_sc, y_test, y_pred):
    """
    Full SHAP explainability analysis for the trained XGBoost model.

    Args:
        model: Trained XGBoost model.
        X_test_sc (pd.DataFrame): Scaled test features.
        y_test (pd.Series): Test target.
        y_pred (np.array): Model predictions on test set.

    Returns:
        shap_values: SHAP values array.
        top_features (list): Top feature names by importance.
    """
    print_separator("PHASE 11: SHAP EXPLAINABILITY ANALYSIS")
    mulai = time.time()

    # ─── Compute SHAP values ──────────────────────────────────────────────
    is_stacking = hasattr(model, 'base_models')
    if is_stacking:
        if 'xgb' in model.base_models:
            target_model = model.base_models['xgb']
            model_name = "XGBoost"
        elif 'lgb' in model.base_models:
            target_model = model.base_models['lgb']
            model_name = "LightGBM"
        else:
            target_model = model.base_models['cat']
            model_name = "CatBoost"
        print(f"  [SHAP] Running SHAP analysis on {model_name} base learner inside Stacking Ensemble.")
        print(f"  [SHAP] Note: Interpretasi dilakukan terhadap base learner karena SHAP belum secara langsung merepresentasikan keseluruhan keputusan meta-learner.")
    else:
        target_model = model

    explainer = shap.TreeExplainer(target_model)
    shap_values = explainer.shap_values(X_test_sc)

    print(f"  SHAP values computed for {X_test_sc.shape[0]} test samples.")
    print(f"  Convention: SHAP > 0 = increases dropout risk")
    print(f"              SHAP < 0 = decreases dropout risk")

    # ─── 1. Beeswarm Plot (Global) ────────────────────────────────────────
    print(f"\n  Generating beeswarm plot...")
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_sc, plot_type="dot",
                      max_display=20, show=False)
    plt.title("SHAP Beeswarm Plot — Feature Influence Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_pdf(plt.gcf(), "v3_shap_beeswarm.pdf")
    plt.close("all")

    # ─── 2. Global Bar Chart ──────────────────────────────────────────────
    print(f"  Generating global bar chart...")
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_sc, plot_type="bar",
                      max_display=15, show=False)
    plt.title("SHAP Global Feature Importance (Mean |SHAP|)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_pdf(plt.gcf(), "v3_shap_global_bar.pdf")
    plt.close("all")

    # ─── 3. Top 20 Feature Ranking Table ─────────────────────────────────
    print(f"\n  Top 20 Features by Mean |SHAP|:")
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Calculate correlation to find true direction of relationship
    correlations = []
    for i, col in enumerate(X_test_sc.columns):
        if np.std(X_test_sc[col]) == 0 or np.std(shap_values[:, i]) == 0:
            correlations.append(0.0)
        else:
            correlations.append(np.corrcoef(X_test_sc[col], shap_values[:, i])[0, 1])

    feature_importance = pd.DataFrame({
        "Feature": X_test_sc.columns,
        "Mean |SHAP|": mean_abs_shap,
        "Correlation": correlations,
        "Interpretation": ["↑ Feature = ↑ Dropout Risk" if c > 0 else "↑ Feature = ↓ Dropout Risk"
                          for c in correlations]
    }).sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)

    # Add rank column (1-indexed)
    feature_importance.insert(0, "Rank", range(1, len(feature_importance) + 1))

    top_20 = feature_importance.head(20)
    print(top_20.to_string(index=False))

    # Save top-20 ranking to CSV
    csv_top20_path = os.path.join(OUTPUT_DIR, "shap_top20_ranking.csv")
    top_20.to_csv(csv_top20_path, index=False)
    print(f"\n  ✅ SHAP top-20 ranking saved to {csv_top20_path}")

    top_features = feature_importance["Feature"].tolist()

    # ─── 4. Dependence Plots (Top 3 features) ────────────────────────────
    print(f"\n  Generating dependence plots for top 3 features...")
    for i, feat in enumerate(top_features[:3]):
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.dependence_plot(feat, shap_values, X_test_sc, show=False, ax=ax)
        ax.set_title(f"SHAP Dependence — {feat}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        safe_name = feat.replace(" ", "_").replace("/", "_").lower()
        save_pdf(fig, f"v3_shap_dep_{safe_name}.pdf")
        plt.close(fig)

    # ─── 5. Waterfall Plot — Dropout instance ─────────────────────────────
    print(f"\n  Generating waterfall plot (Dropout instance)...")
    y_test_arr = np.array(y_test)
    y_pred_arr = np.array(y_pred)

    expected_val = explainer.expected_value
    # expected_value can be list or scalar depending on shap/xgb versions
    if isinstance(expected_val, (list, np.ndarray)):
        expected_val = expected_val[0]

    # Find correctly predicted Dropout
    dropout_indices = np.where((y_test_arr == 1) & (y_pred_arr == 1))[0]
    if len(dropout_indices) > 0:
        idx = dropout_indices[0]
        fig, ax = plt.subplots(figsize=(12, 8))
        shap_explanation = shap.Explanation(
            values=shap_values[idx],
            base_values=expected_val,
            data=X_test_sc.iloc[idx].values,
            feature_names=X_test_sc.columns.tolist()
        )
        shap.waterfall_plot(shap_explanation, max_display=15, show=False)
        plt.title("SHAP Waterfall — Correctly Predicted Dropout", fontsize=13, fontweight="bold")
        plt.tight_layout()
        save_pdf(plt.gcf(), "v3_shap_waterfall_dropout.pdf")
        plt.close("all")
    else:
        print("  ⚠️ No correctly predicted Dropout found for waterfall.")

    # ─── 6. Waterfall Plot — Graduate instance ────────────────────────────
    print(f"  Generating waterfall plot (Graduate instance)...")
    graduate_indices = np.where((y_test_arr == 0) & (y_pred_arr == 0))[0]
    if len(graduate_indices) > 0:
        idx = graduate_indices[0]
        fig, ax = plt.subplots(figsize=(12, 8))
        shap_explanation = shap.Explanation(
            values=shap_values[idx],
            base_values=expected_val,
            data=X_test_sc.iloc[idx].values,
            feature_names=X_test_sc.columns.tolist()
        )
        shap.waterfall_plot(shap_explanation, max_display=15, show=False)
        plt.title("SHAP Waterfall — Correctly Predicted Graduate", fontsize=13, fontweight="bold")
        plt.tight_layout()
        save_pdf(plt.gcf(), "v3_shap_waterfall_graduate.pdf")
        plt.close("all")
    else:
        print("  ⚠️ No correctly predicted Graduate found for waterfall.")

    catat_waktu("SHAP Analysis", mulai)

    return shap_values, top_features
