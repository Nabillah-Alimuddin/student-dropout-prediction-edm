"""
17_base_learner_ablation.py — Base Learner Ablation Study.

Compares stacking ensemble configurations with different base learner subsets
to determine whether all base learners contribute meaningfully:

  4-learner: XGBoost + LightGBM + CatBoost + LogisticRegression (current)
  3-learner: LightGBM + CatBoost + LogisticRegression (drop XGBoost — proposed)
  2-learner: LightGBM + CatBoost (minimal tree ensemble)

Each variant uses the unified, leakage-free training and threshold optimization
procedure identical to the main pipeline.

Includes McNemar test between 4-learner and 3-learner to test whether
dropping XGBoost causes a statistically significant change in error pattern.
"""

import time
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    roc_auc_score, balanced_accuracy_score, matthews_corrcoef
)
from statsmodels.stats.contingency_tables import mcnemar

from src.config import (
    SEED, OUTPUT_DIR, ALPHA,
    THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEP
)
from src.utils import catat_waktu, save_pdf, print_separator
from src.stacking_training import StackingEnsemble


def run_base_learner_ablation(X_train, y_train, X_test, y_test,
                               best_params_xgb, best_params_lgbm, best_params_cb):
    """
    Run base learner ablation study: 4-learner vs 3-learner vs 2-learner.
    
    All variants use identical leakage-free procedure:
    - StandardScaler per OOF fold (V3.2 scaling consistency fix)
    - SMOTE-ENN per OOF fold (V3.1 leakage fix)
    - Outer OOF threshold sweep matching the main pipeline
    
    Args:
        X_train: Original training features (NOT pre-scaled, NOT resampled).
        y_train: Original training target.
        X_test: Original test features (NOT pre-scaled).
        y_test: Test target.
        best_params_xgb, best_params_lgbm, best_params_cb: Optuna hyperparameters.
    
    Returns:
        ablation_df: DataFrame with comparison results.
    """
    print_separator("PHASE 11: BASE LEARNER ABLATION (4 vs 3 vs 2 LEARNER)")
    mulai = time.time()
    
    configs = [
        ("4-learner (XGB+LGB+CB+LR)", StackingEnsemble(
            xgb_params=best_params_xgb,
            lgbm_params=best_params_lgbm,
            catboost_params=best_params_cb,
            seed=SEED,
            apply_resampling=True,
            apply_scaling=True,
            use_xgb=True,
            use_lr=True
        )),
        ("3-learner (LGB+CB+LR)", StackingEnsemble(
            xgb_params=best_params_xgb,
            lgbm_params=best_params_lgbm,
            catboost_params=best_params_cb,
            seed=SEED,
            apply_resampling=True,
            apply_scaling=True,
            use_xgb=False,
            use_lr=True
        )),
        ("2-learner (LGB+CB)", StackingEnsemble(
            xgb_params=best_params_xgb,
            lgbm_params=best_params_lgbm,
            catboost_params=best_params_cb,
            seed=SEED,
            apply_resampling=True,
            apply_scaling=True,
            use_xgb=False,
            use_lr=False
        )),
    ]
    
    results = []
    all_predictions = {}  # For McNemar
    
    for label, model in configs:
        print(f"\n  ── Training: {label} ──")
        
        # 1. Optimize threshold via unbiased outer OOF sweep
        print("    Optimizing threshold via unbiased 5-Fold Stratified CV...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        y_oof_proba = cross_val_predict(
            model, X_train, y_train, cv=skf, method="predict_proba", n_jobs=1
        )[:, 1]
        
        thresholds = np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP, THRESHOLD_STEP)
        best_f1 = -1
        optimal_t = 0.50
        for t in thresholds:
            y_pred_t = (y_oof_proba >= t).astype(int)
            f1_t = f1_score(y_train, y_pred_t, pos_label=1, zero_division=0)
            if f1_t > best_f1:
                best_f1 = f1_t
                optimal_t = t
                
        # 2. Fit final model on 100% training data
        print("    Fitting final model on full training set...")
        model.fit(X_train, y_train)
        
        # 3. Predict on test set (model auto-scales via internal scaler)
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= optimal_t).astype(int)
        
        # Compute metrics
        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        auc_roc = roc_auc_score(y_test, y_proba)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        # Meta-learner coefficients
        coefs = model.meta_learner_.coef_[0]
        base_names = []
        if model.use_xgb:
            base_names.append("XGBoost")
        base_names.extend(["LightGBM", "CatBoost"])
        if model.use_lr:
            base_names.append("LogisticRegression")
            
        coef_str = ", ".join([f"{n}: {c:.4f}" for n, c in zip(base_names, coefs)])
        
        results.append({
            "Configuration": label,
            "Threshold": optimal_t,
            "F1-Dropout": f1,
            "Recall": rec,
            "Precision": prec,
            "AUC-ROC": auc_roc,
            "Balanced Acc": bal_acc,
            "MCC": mcc,
            "Meta-Learner Coefficients": coef_str,
        })
        
        all_predictions[label] = y_pred
        
        print(f"    Optimal OOF Threshold: {optimal_t:.2f}")
        print(f"    F1-Dropout: {f1:.4f} | Recall: {rec:.4f} | Precision: {prec:.4f}")
        print(f"    AUC-ROC: {auc_roc:.4f} | Bal Acc: {bal_acc:.4f} | MCC: {mcc:.4f}")
        print(f"    Meta-Learner Weights: {coef_str}")
    
    ablation_df = pd.DataFrame(results)
    
    # ── McNemar Test: 4-learner vs 3-learner ─────────────────────────────
    print(f"\n  ── McNemar Test: 4-learner vs 3-learner ──")
    y_pred_4 = all_predictions["4-learner (XGB+LGB+CB+LR)"]
    y_pred_3 = all_predictions["3-learner (LGB+CB+LR)"]
    y_true = np.array(y_test)
    
    correct_4 = (y_pred_4 == y_true)
    correct_3 = (y_pred_3 == y_true)
    
    a = np.sum(correct_4 & correct_3)     # Both correct
    b = np.sum(correct_4 & ~correct_3)    # 4-learner correct, 3-learner wrong
    c = np.sum(~correct_4 & correct_3)    # 4-learner wrong, 3-learner correct
    d = np.sum(~correct_4 & ~correct_3)   # Both wrong
    
    table = np.array([[a, b], [c, d]])
    use_exact = (b + c) < 25
    result = mcnemar(table, exact=use_exact)
    
    significant = "✅ Yes" if result.pvalue < ALPHA else "❌ No"
    
    print(f"    Contingency: a={a}, b={b}, c={c}, d={d}")
    print(f"    Method: {'Exact' if use_exact else 'Chi-squared'}")
    print(f"    Statistic: {result.statistic:.4f}, p-value: {result.pvalue:.6f}")
    print(f"    Significant (α={ALPHA}): {significant}")
    
    mcnemar_row = {
        "Comparison": "4-learner vs 3-learner",
        "b (4✓ 3✗)": b,
        "c (4✗ 3✓)": c,
        "Method": "Exact" if use_exact else "Chi-squared",
        "Statistic": round(result.statistic, 4),
        "p-value": round(result.pvalue, 6),
        f"Significant (α={ALPHA})": significant,
    }
    
    # Determine recommendation
    f1_4 = ablation_df[ablation_df["Configuration"].str.contains("4-learner")]["F1-Dropout"].values[0]
    f1_3 = ablation_df[ablation_df["Configuration"].str.contains("3-learner")]["F1-Dropout"].values[0]
    delta_f1 = f1_4 - f1_3
    
    print(f"\n  ── Recommendation ──")
    print(f"    F1 delta (4-learner − 3-learner): {delta_f1:+.4f}")
    
    if result.pvalue >= ALPHA:
        if abs(delta_f1) < 0.005:
            recommendation = ("3-learner (parsimoni): performa identik, "
                             "drop XGBoost untuk model yang lebih ringkas")
        else:
            recommendation = ("Perbedaan tidak signifikan secara statistik. "
                             "Pertimbangkan 3-learner untuk parsimoni model.")
    else:
        if delta_f1 > 0:
            recommendation = "Pertahankan 4-learner: perbedaan signifikan, 4-learner unggul."
        else:
            recommendation = "Drop ke 3-learner: perbedaan signifikan, 3-learner justru lebih baik!"
    
    print(f"    → {recommendation}")
    
    # ── Save results ─────────────────────────────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, "base_learner_ablation.csv")
    ablation_df.to_csv(csv_path, index=False)
    print(f"\n  📄 Ablation results saved to {csv_path}")
    
    mcnemar_abl_path = os.path.join(OUTPUT_DIR, "mcnemar_4vs3_learner.csv")
    pd.DataFrame([mcnemar_row]).to_csv(mcnemar_abl_path, index=False)
    print(f"  📄 McNemar (4 vs 3) saved to {mcnemar_abl_path}")
    
    # ── Summary table ────────────────────────────────────────────────────
    print(f"\n  ═══ BASE LEARNER ABLATION SUMMARY ═══")
    display_cols = ["Configuration", "Threshold", "F1-Dropout", "Recall", 
                    "Precision", "AUC-ROC", "Balanced Acc", "MCC"]
    print(ablation_df[display_cols].to_string(index=False))
    
    # ── Visualization ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    metric_cols = ["F1-Dropout", "Recall", "Precision", "AUC-ROC", "Balanced Acc", "MCC"]
    x = np.arange(len(metric_cols))
    width = 0.25
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    
    for idx, row_idx in enumerate(ablation_df.index):
        row = ablation_df.loc[row_idx]
        vals = [row[m] for m in metric_cols]
        ax.bar(x + idx * width, vals, width, label=row["Configuration"],
               color=colors[idx], edgecolor="black", linewidth=0.5)
    
    ax.set_xlabel("Metrics", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Base Learner Ablation — 4 vs 3 vs 2 Learner Stacking",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_cols, fontsize=10)
    ax.set_ylim(0.70, 1.02)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    
    ax.annotate(
        f"McNemar 4v3: p={result.pvalue:.4f} ({significant})\n{recommendation}",
        xy=(0.5, 0.02), xycoords="axes fraction",
        fontsize=8.5, ha="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray")
    )
    
    plt.tight_layout()
    save_pdf(fig, "v3_base_learner_ablation.pdf")
    plt.close(fig)
    
    catat_waktu("Base Learner Ablation", mulai)
    
    return ablation_df


# ─── Standalone execution ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import os
    
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    
    import warnings
    warnings.filterwarnings("ignore")
    
    from src.utils import reset_waktu_log
    reset_waktu_log()
    
    print("=" * 70)
    print("  BASE LEARNER ABLATION STUDY (STANDALONE)")
    print("=" * 70)
    
    import importlib
    data_prep = importlib.import_module("src.01_data_preparation")
    load_and_prepare_data = data_prep.load_and_prepare_data
    
    preprocessing = importlib.import_module("src.02_preprocessing")
    split_and_scale = preprocessing.split_and_scale
    
    X, y, df = load_and_prepare_data()
    X_train, X_test, X_train_sc, X_test_sc, y_train, y_test, scaler = split_and_scale(X, y)
    
    import joblib
    
    params_path = os.path.join(OUTPUT_DIR, "optuna_best_params.pkl")
    if os.path.exists(params_path):
        saved = joblib.load(params_path)
        p_xgb = saved.get("xgb", {})
        p_lgb = saved.get("lgb", {})
        p_cb = saved.get("cat", {})
    else:
        print("  ⚠️  No saved Optuna params found. Using defaults.")
        p_xgb = {"random_state": SEED, "use_label_encoder": False, "eval_metric": "logloss"}
        p_lgb = {"random_state": SEED, "verbosity": -1}
        p_cb = {"random_seed": SEED, "verbose": False}
    
    ablation_df = run_base_learner_ablation(
        X_train, y_train, X_test, y_test,
        p_xgb, p_lgb, p_cb
    )
