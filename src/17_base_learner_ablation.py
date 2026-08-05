"""
17_base_learner_ablation.py — Base Learner Ablation Study.

Compares stacking ensemble configurations with different base learner subsets
to determine whether all 4 base learners contribute meaningfully:

  4-learner: XGBoost + LightGBM + CatBoost + LogisticRegression (current)
  3-learner: LightGBM + CatBoost + LogisticRegression (drop XGBoost)
  2-learner: LightGBM + CatBoost (minimal tree ensemble)

Each variant uses identical, leakage-free training procedure:
  - 5-fold Stratified OOF with SMOTE-ENN inside each fold
  - LogisticRegression meta-learner
  - Threshold optimization via OOF probability sweep

Includes McNemar test between 4-learner and 3-learner to test whether
dropping XGBoost causes a statistically significant change in error pattern.

Motivation:
  In V3, XGBoost's meta-learner coefficient was near zero (-0.0229),
  suggesting it contributes negligibly. This ablation provides empirical
  evidence to justify retaining or dropping XGBoost.
"""

import time
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    roc_auc_score, balanced_accuracy_score, matthews_corrcoef
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from statsmodels.stats.contingency_tables import mcnemar

from src.config import (
    SEED, OUTPUT_DIR, ALPHA,
    SMOTE_K_NEIGHBORS, SMOTE_TARGET_RATIO, ENN_N_NEIGHBORS, ENN_KIND_SEL,
    THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEP
)
from src.utils import catat_waktu, save_pdf, print_separator


def _get_base_learner_factories(config_name, p_xgb, p_lgb, p_cb, seed):
    """Return an ordered dict of {name: factory_fn} for a given configuration."""
    lr_factory = lambda: LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=seed
    )
    
    if config_name == "4-learner":
        return {
            "XGBoost": lambda: XGBClassifier(**p_xgb),
            "LightGBM": lambda: LGBMClassifier(**p_lgb),
            "CatBoost": lambda: CatBoostClassifier(**p_cb),
            "LogisticRegression": lr_factory,
        }
    elif config_name == "3-learner":
        return {
            "LightGBM": lambda: LGBMClassifier(**p_lgb),
            "CatBoost": lambda: CatBoostClassifier(**p_cb),
            "LogisticRegression": lr_factory,
        }
    elif config_name == "2-learner":
        return {
            "LightGBM": lambda: LGBMClassifier(**p_lgb),
            "CatBoost": lambda: CatBoostClassifier(**p_cb),
        }
    else:
        raise ValueError(f"Unknown config: {config_name}")


def _train_stacking_variant(X_train_sc, y_train, factories, seed):
    """
    Train a stacking variant with SMOTE-ENN inside each OOF fold.
    
    Returns:
        final_models: List of refitted base learner models (on full resampled data).
        meta_learner: Trained LogisticRegression meta-learner.
        oof_meta_proba: OOF-level stacking probabilities (for threshold optimization).
    """
    X_arr = np.array(X_train_sc)
    y_arr = np.array(y_train)
    
    n_base = len(factories)
    base_names = list(factories.keys())
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_preds = np.zeros((X_arr.shape[0], n_base))
    
    for train_idx, val_idx in skf.split(X_arr, y_arr):
        X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
        y_tr = y_arr[train_idx]
        
        # SMOTE-ENN inside fold (leakage-free)
        smote = SMOTE(
            k_neighbors=SMOTE_K_NEIGHBORS,
            random_state=seed,
            sampling_strategy=SMOTE_TARGET_RATIO
        )
        enn = EditedNearestNeighbours(
            n_neighbors=ENN_N_NEIGHBORS,
            kind_sel=ENN_KIND_SEL
        )
        X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
        X_tr_clean, y_tr_clean = enn.fit_resample(X_tr_res, y_tr_res)
        
        # Fit fold models and collect OOF predictions
        for j, (name, factory) in enumerate(factories.items()):
            m = factory()
            if "XGBoost" in name or "CatBoost" in name:
                m.fit(X_tr_clean, y_tr_clean, verbose=False)
            else:
                m.fit(X_tr_clean, y_tr_clean)
            oof_preds[val_idx, j] = m.predict_proba(X_val)[:, 1]
    
    # Train meta-learner on OOF predictions
    meta_learner = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=seed
    )
    meta_learner.fit(oof_preds, y_arr)
    
    # OOF stacking probabilities for threshold optimization
    oof_meta_proba = meta_learner.predict_proba(oof_preds)[:, 1]
    
    # Refit all base models on full data with SMOTE-ENN
    smote_full = SMOTE(
        k_neighbors=SMOTE_K_NEIGHBORS,
        random_state=seed,
        sampling_strategy=SMOTE_TARGET_RATIO
    )
    enn_full = EditedNearestNeighbours(
        n_neighbors=ENN_N_NEIGHBORS,
        kind_sel=ENN_KIND_SEL
    )
    X_full_res, y_full_res = smote_full.fit_resample(X_arr, y_arr)
    X_final, y_final = enn_full.fit_resample(X_full_res, y_full_res)
    
    final_models = []
    for name, factory in factories.items():
        m = factory()
        if "XGBoost" in name or "CatBoost" in name:
            m.fit(X_final, y_final, verbose=False)
        else:
            m.fit(X_final, y_final)
        final_models.append(m)
    
    return final_models, meta_learner, oof_meta_proba


def _predict_stacking(final_models, meta_learner, X_test):
    """Predict probabilities using stacking variant."""
    X_arr = np.array(X_test)
    meta_features = np.column_stack([
        m.predict_proba(X_arr)[:, 1] for m in final_models
    ])
    return meta_learner.predict_proba(meta_features)[:, 1]


def _find_optimal_threshold_oof(y_true, y_proba):
    """Find optimal F1 threshold via sweep on OOF probabilities."""
    thresholds = np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP, THRESHOLD_STEP)
    best_f1 = -1
    best_t = 0.50
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t


def run_base_learner_ablation(X_train_sc, y_train, X_test_sc, y_test,
                               best_params_xgb, best_params_lgbm, best_params_cb):
    """
    Run base learner ablation study: 4-learner vs 3-learner vs 2-learner.
    
    All variants use identical leakage-free procedure (SMOTE-ENN per OOF fold).
    Includes McNemar test between 4-learner and 3-learner.
    
    Args:
        X_train_sc: Scaled training features (NOT resampled).
        y_train: Original training target.
        X_test_sc: Scaled test features.
        y_test: Test target.
        best_params_xgb, best_params_lgbm, best_params_cb: Optuna hyperparameters.
    
    Returns:
        ablation_df: DataFrame with comparison results.
    """
    print_separator("PHASE 11: BASE LEARNER ABLATION (4 vs 3 vs 2 LEARNER)")
    mulai = time.time()
    
    configs = [
        ("4-learner (XGB+LGB+CB+LR)", "4-learner"),
        ("3-learner (LGB+CB+LR)", "3-learner"),
        ("2-learner (LGB+CB)", "2-learner"),
    ]
    
    results = []
    all_predictions = {}  # For McNemar
    
    for label, config_name in configs:
        print(f"\n  ── Training: {label} ──")
        
        factories = _get_base_learner_factories(
            config_name, best_params_xgb, best_params_lgbm, best_params_cb, SEED
        )
        
        final_models, meta_learner, oof_meta_proba = _train_stacking_variant(
            X_train_sc, y_train, factories, SEED
        )
        
        # Threshold optimization using OOF stacking probabilities
        optimal_t = _find_optimal_threshold_oof(np.array(y_train), oof_meta_proba)
        
        # Test predictions
        y_proba = _predict_stacking(final_models, meta_learner, X_test_sc)
        y_pred = (y_proba >= optimal_t).astype(int)
        
        # Compute metrics
        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        auc_roc = roc_auc_score(y_test, y_proba)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        # Meta-learner coefficients
        coefs = meta_learner.coef_[0]
        base_names = list(factories.keys())
        coef_str = ", ".join([f"{n}: {c:.4f}" for n, c in zip(base_names, coefs)])
        
        results.append({
            "Configuration": label,
            "N Base Learners": len(factories),
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
        
        print(f"    Threshold: {optimal_t:.2f}")
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
    
    for idx, (_, row) in enumerate(ablation_df.iterrows()):
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
    
    # Add annotation for McNemar result
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
    """Run ablation study standalone (re-does data prep + Optuna from saved params)."""
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
    
    # Re-run data preparation
    import importlib
    data_prep = importlib.import_module("src.01_data_preparation")
    load_and_prepare_data = data_prep.load_and_prepare_data
    
    preprocessing = importlib.import_module("src.02_preprocessing")
    split_and_scale = preprocessing.split_and_scale
    
    X, y, df = load_and_prepare_data()
    X_train, X_test, X_train_sc, X_test_sc, y_train, y_test, scaler = split_and_scale(X, y)
    
    # Load saved Optuna best params (if available) or use defaults
    import joblib
    
    params_path = os.path.join(OUTPUT_DIR, "optuna_best_params.pkl")
    if os.path.exists(params_path):
        saved = joblib.load(params_path)
        p_xgb = saved.get("xgb", {})
        p_lgb = saved.get("lgb", {})
        p_cb = saved.get("cat", {})
    else:
        print("  ⚠️  No saved Optuna params found. Using defaults.")
        print("     Run 'python main.py' first for full pipeline.")
        p_xgb = {"random_state": SEED, "use_label_encoder": False, "eval_metric": "logloss"}
        p_lgb = {"random_state": SEED, "verbosity": -1}
        p_cb = {"random_seed": SEED, "verbose": False}
    
    ablation_df = run_base_learner_ablation(
        X_train_sc, y_train, X_test_sc, y_test,
        p_xgb, p_lgb, p_cb
    )
