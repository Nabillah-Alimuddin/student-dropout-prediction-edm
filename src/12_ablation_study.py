"""
12_ablation_study.py — Phase 12.5: Ablation Study.

Evaluates 8 different configurations to isolate the performance contribution
of each pipeline component (Optuna, SMOTE, ENN, Threshold Optimization).
Produces a summary CSV and a comparative plot.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, matthews_corrcoef
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours

from src.config import (
    SEED, SMOTE_K_NEIGHBORS, SMOTE_TARGET_RATIO,
    ENN_N_NEIGHBORS, ENN_KIND_SEL,
    THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEP,
    OUTPUT_DIR
)
from src.utils import catat_waktu, save_pdf, print_separator


def run_ablation_study(X_train, y_train, X_test, y_test, scaler, best_params):
    """
    Runs the ablation study for the 8 configurations.

    Args:
        X_train (pd.DataFrame): Raw training features.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Raw test features.
        y_test (pd.Series): Test target.
        scaler (StandardScaler): Scaler fitted on X_train.
        best_params (dict): Best parameters from Optuna tuning.
    """
    print_separator("PHASE 12.5: ABLATION STUDY")
    mulai = time.time()

    # Pre-scale test set using the fitted scaler
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    X_train_sc = pd.DataFrame(
        scaler.transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    # Define common objects
    smote = SMOTE(
        k_neighbors=SMOTE_K_NEIGHBORS,
        random_state=SEED,
        sampling_strategy=SMOTE_TARGET_RATIO
    )
    enn = EditedNearestNeighbours(
        n_neighbors=ENN_N_NEIGHBORS,
        kind_sel=ENN_KIND_SEL
    )

    # Resampled datasets
    # C/F: SMOTE only
    X_train_smote, y_train_smote = smote.fit_resample(X_train_sc, y_train)
    # D/G/H: SMOTE-ENN
    X_train_smote_temp, y_train_smote_temp = smote.fit_resample(X_train_sc, y_train)
    X_train_se, y_train_se = enn.fit_resample(X_train_smote_temp, y_train_smote_temp)

    # Define models
    def get_default_xgb():
        return XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=SEED, use_label_encoder=False,
            eval_metric="logloss"
        )

    def get_tuned_xgb():
        return XGBClassifier(**best_params)

    # Threshold optimization function using Stratified OOF CV
    def optimize_threshold(pipeline, X_tr, y_tr):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        y_oof_proba = cross_val_predict(
            pipeline, X_tr, y_tr, cv=skf, method="predict_proba", n_jobs=-1
        )[:, 1]
        
        thresholds = np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP, THRESHOLD_STEP)
        best_f1 = -1
        best_t = 0.50
        for t in thresholds:
            y_pred_t = (y_oof_proba >= t).astype(int)
            f1 = f1_score(y_tr, y_pred_t, pos_label=1, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        return best_t

    # Define the configurations
    configs = {
        "A": {
            "name": "XGBoost Default",
            "model": get_default_xgb(),
            "X_tr": X_train_sc,
            "y_tr": y_train,
            "optimize_thresh": False,
            "pipeline_for_thresh": None
        },
        "B": {
            "name": "XGBoost + Optuna",
            "model": get_tuned_xgb(),
            "X_tr": X_train_sc,
            "y_tr": y_train,
            "optimize_thresh": False,
            "pipeline_for_thresh": None
        },
        "C": {
            "name": "XGBoost + SMOTE",
            "model": get_default_xgb(),
            "X_tr": X_train_smote,
            "y_tr": y_train_smote,
            "optimize_thresh": False,
            "pipeline_for_thresh": None
        },
        "D": {
            "name": "XGBoost + SMOTE-ENN",
            "model": get_default_xgb(),
            "X_tr": X_train_se,
            "y_tr": y_train_se,
            "optimize_thresh": False,
            "pipeline_for_thresh": None
        },
        "E": {
            "name": "XGBoost + Threshold Opt",
            "model": get_default_xgb(),
            "X_tr": X_train_sc,
            "y_tr": y_train,
            "optimize_thresh": True,
            "pipeline_for_thresh": ImbPipeline([
                ('scaler', StandardScaler()),
                ('clf', get_default_xgb())
            ])
        },
        "F": {
            "name": "XGBoost + Optuna + SMOTE",
            "model": get_tuned_xgb(),
            "X_tr": X_train_smote,
            "y_tr": y_train_smote,
            "optimize_thresh": False,
            "pipeline_for_thresh": None
        },
        "G": {
            "name": "XGBoost + Optuna + SMOTE-ENN",
            "model": get_tuned_xgb(),
            "X_tr": X_train_se,
            "y_tr": y_train_se,
            "optimize_thresh": False,
            "pipeline_for_thresh": None
        },
        "H": {
            "name": "XGBoost + Optuna + SMOTE-ENN + Threshold Opt (Proposed)",
            "model": get_tuned_xgb(),
            "X_tr": X_train_se,
            "y_tr": y_train_se,
            "optimize_thresh": True,
            "pipeline_for_thresh": ImbPipeline([
                ('scaler', StandardScaler()),
                ('smote', SMOTE(k_neighbors=SMOTE_K_NEIGHBORS, random_state=SEED, sampling_strategy=SMOTE_TARGET_RATIO)),
                ('enn', EditedNearestNeighbours(n_neighbors=ENN_N_NEIGHBORS, kind_sel=ENN_KIND_SEL)),
                ('clf', get_tuned_xgb())
            ])
        }
    }

    results = []

    for fid, cfg in configs.items():
        print(f"  Running configuration {fid}: {cfg['name']}...")
        model = cfg["model"]
        X_tr = cfg["X_tr"]
        y_tr = cfg["y_tr"]

        # Fit model on config's training data
        model.fit(X_tr, y_tr, verbose=False)

        # Determine threshold
        if cfg["optimize_thresh"] and cfg["pipeline_for_thresh"] is not None:
            # Find optimal threshold via OOF on ORIGINAL unscaled data
            opt_t = optimize_threshold(cfg["pipeline_for_thresh"], X_train, y_train)
            print(f"    Optimal OOF threshold: {opt_t:.2f}")
            
            # IMPORTANT: refit the pipeline on full training data and use IT
            # for prediction — ensures consistency between threshold search
            # and final predictions
            cfg["pipeline_for_thresh"].fit(X_train, y_train)
            y_proba = cfg["pipeline_for_thresh"].predict_proba(X_test)[:, 1]
        else:
            opt_t = 0.50
            y_proba = model.predict_proba(X_test_sc)[:, 1]

        # Predictions at threshold
        y_pred = (y_proba >= opt_t).astype(int)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)
        mcc = matthews_corrcoef(y_test, y_pred)

        results.append({
            "Config ID": fid,
            "Configuration": cfg["name"],
            "Threshold": opt_t,
            "Accuracy": acc,
            "Balanced Accuracy": bal_acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Dropout": f1,
            "ROC-AUC": roc_auc,
            "MCC": mcc
        })

    ablation_df = pd.DataFrame(results)

    # Save to CSV
    csv_path = f"{OUTPUT_DIR}/ablation_results.csv"
    ablation_df.to_csv(csv_path, index=False)
    print(f"\n  ✅ Ablation study results saved to {csv_path}")
    print(ablation_df.to_string(index=False))

    # Plot results
    fig, ax = plt.subplots(figsize=(14, 8))
    metrics_to_plot = ["F1-Dropout", "Recall", "Precision", "ROC-AUC", "Balanced Accuracy", "MCC"]
    x = np.arange(len(metrics_to_plot))
    width = 0.10

    # Modern harmonious color palette
    colors = ["#7f8c8d", "#3498db", "#9b59b6", "#e67e22", "#1abc9c", "#f1c40f", "#e74c3c", "#2ecc71"]

    for idx, row in ablation_df.iterrows():
        vals = [row[m] for m in metrics_to_plot]
        ax.bar(x + idx * width, vals, width, label=f"({row['Config ID']}) {row['Configuration']}",
               color=colors[idx], edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Evaluation Metrics", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Ablation Study — Component Influence Analysis", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * 3.5)
    ax.set_xticklabels(metrics_to_plot, fontsize=10)
    ax.legend(fontsize=9, loc="lower left", bbox_to_anchor=(0.0, 0.0))
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_path = "v3_ablation_study.pdf"
    save_pdf(fig, plot_path)
    plt.close(fig)
    print(f"  📄 Ablation study comparative chart saved to {OUTPUT_DIR}/{plot_path}")

    catat_waktu("Ablation Study", mulai)

    return ablation_df
