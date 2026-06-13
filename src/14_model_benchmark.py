"""
14_model_benchmark.py — Priority 2: CatBoost Benchmarking & Extended McNemar Tests.

Trains, tunes, and evaluates CatBoost against existing models using default threshold (0.50)
to avoid unnecessary complexity. Performs expanded McNemar significance tests.
"""

import time
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    roc_auc_score, balanced_accuracy_score, matthews_corrcoef,
    make_scorer
)
from statsmodels.stats.contingency_tables import mcnemar
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours

# Import configuration and utilities from src
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    SEED, OUTPUT_DIR, MODEL_DIR, MODEL_PATH,
    CATBOOST_N_TRIALS, CATBOOST_TIMEOUT, CATBOOST_MODEL_PATH,
    SMOTE_K_NEIGHBORS, SMOTE_TARGET_RATIO, ENN_N_NEIGHBORS, ENN_KIND_SEL,
    OPTUNA_CV_FOLDS, ALPHA
)
from src.utils import catat_waktu, save_pdf, print_separator, reset_waktu_log


def run_catboost_tuning(X_train, y_train):
    """
    Tune CatBoostClassifier using Optuna.
    StandardScaler and SMOTE-ENN are applied inside each CV fold via ImbPipeline.
    """
    print_separator("TUNING CATBOOST HYPERPARAMETERS")
    mulai = time.time()

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 100, 500),
            "depth": trial.suggest_int("depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "random_seed": SEED,
            "verbose": 0,
            "thread_count": -1
        }

        pipe = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(
                k_neighbors=SMOTE_K_NEIGHBORS,
                random_state=SEED,
                sampling_strategy=SMOTE_TARGET_RATIO
            )),
            ('enn', EditedNearestNeighbours(
                n_neighbors=ENN_N_NEIGHBORS,
                kind_sel=ENN_KIND_SEL
            )),
            ('clf', CatBoostClassifier(**params)),
        ])

        skf = StratifiedKFold(n_splits=OPTUNA_CV_FOLDS, shuffle=True, random_state=SEED)
        f1_scorer = make_scorer(f1_score, pos_label=1)
        
        scores = cross_val_score(
            pipe, X_train, y_train, cv=skf, scoring=f1_scorer, n_jobs=1
        )
        return scores.mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    print(f"  Running {CATBOOST_N_TRIALS} trials (timeout: {CATBOOST_TIMEOUT}s)...")
    study.optimize(objective, n_trials=CATBOOST_N_TRIALS, timeout=CATBOOST_TIMEOUT)

    best_params = study.best_params.copy()
    best_params.update({
        "random_seed": SEED,
        "verbose": 0,
        "thread_count": -1
    })

    print(f"\n  Best trial: #{study.best_trial.number}")
    print(f"  Best F1-Score (CV): {study.best_value:.6f}")
    print(f"  Best parameters:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    catat_waktu("CatBoost Tuning", mulai)
    return best_params


def perform_mcnemar_test(y_test, pred_a, pred_b, name_a, name_b):
    """Perform pairwise McNemar test between two sets of predictions."""
    correct_a = (pred_a == y_test)
    correct_b = (pred_b == y_test)

    b = np.sum(correct_a & ~correct_b)  # A correct, B wrong
    c = np.sum(~correct_a & correct_b)  # A wrong, B correct
    a = np.sum(correct_a & correct_b)   # Both correct
    d = np.sum(~correct_a & ~correct_b)  # Both wrong

    table = np.array([[a, b], [c, d]])
    use_exact = (b + c) < 25
    result = mcnemar(table, exact=use_exact)
    significant = "Yes" if result.pvalue < ALPHA else "No"

    return {
        "Comparison": f"{name_a} vs {name_b}",
        "b": b,
        "c": c,
        "Method": "Exact" if use_exact else "Chi-squared",
        "Statistic": round(result.statistic, 4),
        "p-value": round(result.pvalue, 6),
        "Significant": significant
    }


def main():
    reset_waktu_log()
    start_time = time.time()

    print("=" * 70)
    print("  CATBOOST MODEL BENCHMARKING & EXPANDED McNEMAR TESTS")
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

    # ─── Train CatBoost Default ───────────────────────────────────────────
    print_separator("TRAINING CATBOOST DEFAULT")
    cb_default = CatBoostClassifier(random_seed=SEED, verbose=0, thread_count=-1)
    cb_default.fit(X_res, y_res)
    print("  CatBoost Default model trained successfully.")

    # ─── Tune CatBoost with Optuna ────────────────────────────────────────
    best_params = run_catboost_tuning(X_train, y_train)

    # ─── Train CatBoost Tuned ─────────────────────────────────────────────
    print_separator("TRAINING TUNED CATBOOST")
    cb_tuned = CatBoostClassifier(**best_params)
    cb_tuned.fit(X_res, y_res)
    joblib.dump(cb_tuned, CATBOOST_MODEL_PATH)
    print(f"  Tuned CatBoost model trained and saved to {CATBOOST_MODEL_PATH}")

    # ─── Load or Train XGBoost Proposed ───────────────────────────────────
    print_separator("PREPARING OTHER MODELS FOR COMPARISON")
    if os.path.exists(MODEL_PATH):
        print(f"  Loading existing tuned XGBoost Proposed model from {MODEL_PATH}...")
        xgb_proposed = joblib.load(MODEL_PATH)
    else:
        print("  ⚠️ Tuned XGBoost Proposed model not found. Training with default parameters...")
        xgb_proposed = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=SEED, use_label_encoder=False, eval_metric="logloss"
        )
        xgb_proposed.fit(X_res, y_res)

    # Define all models to evaluate (all evaluated at default threshold = 0.50)
    models = {
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED),
        "Random Forest": RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=SEED),
        "XGBoost (Baseline)": XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=SEED, use_label_encoder=False, eval_metric="logloss"
        ),
        "XGBoost + SMOTE-ENN (Proposed)": xgb_proposed,
        "CatBoost (Default)": cb_default,
        "CatBoost + SMOTE-ENN (Tuned)": cb_tuned
    }

    # Train baseline classifiers
    for name in ["Decision Tree", "Logistic Regression", "Random Forest", "XGBoost (Baseline)"]:
        print(f"  Training baseline: {name}...")
        models[name].fit(X_train_sc, y_train)

    # ─── Evaluate Models at 0.50 Threshold ────────────────────────────────
    print_separator("EVALUATING BENCHMARKED MODELS (Threshold = 0.50)")
    results_rows = []
    predictions = {}
    probabilities = {}

    for name, clf in models.items():
        # XGBoost Proposed and CatBoost (Default & Tuned) are trained on X_res (resampled)
        # Baselines are trained on X_train_sc (original scaled)
        y_proba = clf.predict_proba(X_test_sc)[:, 1]
        y_pred = (y_proba >= 0.50).astype(int)

        predictions[name] = y_pred
        probabilities[name] = y_proba

        f1 = f1_score(y_test, y_pred, pos_label=1)
        rec = recall_score(y_test, y_pred, pos_label=1)
        prec = precision_score(y_test, y_pred, pos_label=1)
        auc_roc = roc_auc_score(y_test, y_proba)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        results_rows.append({
            "Model": name,
            "F1-Dropout": f1,
            "Recall": rec,
            "Precision": prec,
            "AUC-ROC": auc_roc,
            "Balanced Acc": bal_acc,
            "MCC": mcc
        })

    benchmark_df = pd.DataFrame(results_rows)
    csv_path = os.path.join(OUTPUT_DIR, "model_benchmark_results.csv")
    benchmark_df.to_csv(csv_path, index=False)
    print(f"\n  ✅ Benchmark results saved to {csv_path}")
    print(benchmark_df.to_string(index=False))

    # ─── Comparative Chart ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 8))
    metrics_to_plot = ["F1-Dropout", "Recall", "Precision", "AUC-ROC", "Balanced Acc", "MCC"]
    x = np.arange(len(metrics_to_plot))
    width = 0.11
    colors = ["#7f8c8d", "#3498db", "#9b59b6", "#e67e22", "#1abc9c", "#f1c40f", "#e74c3c"]

    for idx, (_, row) in enumerate(benchmark_df.iterrows()):
        vals = [row[m] for m in metrics_to_plot]
        ax.bar(x + idx * width, vals, width, label=row["Model"],
               color=colors[idx], edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Evaluation Metrics", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Model Benchmarking — Baseline, XGBoost, and CatBoost", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * 3)
    ax.set_xticklabels(metrics_to_plot, fontsize=10)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_path = "v3_model_benchmark_comparison.pdf"
    save_pdf(fig, plot_path)
    plt.close(fig)

    # ─── Expanded McNemar Significance Tests ──────────────────────────────
    print_separator("EXPANDED McNEMAR SIGNIFICANCE TESTS")
    mcnemar_comparisons = [
        ("Logistic Regression", "XGBoost + SMOTE-ENN (Proposed)"),
        ("Logistic Regression", "CatBoost + SMOTE-ENN (Tuned)"),
        ("CatBoost + SMOTE-ENN (Tuned)", "XGBoost + SMOTE-ENN (Proposed)")
    ]

    mc_results = []
    for model_a, model_b in mcnemar_comparisons:
        if model_a in predictions and model_b in predictions:
            res = perform_mcnemar_test(
                y_test, predictions[model_a], predictions[model_b], model_a, model_b
            )
            mc_results.append(res)
            print(f"  {res['Comparison']}: p-value = {res['p-value']:.6f} (Significant: {res['Significant']})")

    mc_df = pd.DataFrame(mc_results)
    mc_csv_path = os.path.join(OUTPUT_DIR, "mcnemar_benchmark_results.csv")
    mc_df.to_csv(mc_csv_path, index=False)
    print(f"  ✅ McNemar benchmark results saved to {mc_csv_path}")

    total_time = time.time() - start_time
    print(f"\n  🏁 Benchmark script complete. Total time: {total_time:.2f}s")


if __name__ == "__main__":
    main()
