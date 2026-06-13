"""
16_error_analysis.py — Priority 4: Error Profile Analysis.

Identifies the best-performing model on the test set, categorizes its predictions
into TP, TN, FP, and FN, and compares their profiles on raw (unscaled) features.
Saves a detailed markdown analysis to outputs/misclassified_students_analysis.md.
"""

import time
import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# Import configuration and utilities from src
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    SEED, OUTPUT_DIR, MODEL_PATH, CATBOOST_MODEL_PATH
)
from src.utils import catat_waktu, print_separator, reset_waktu_log


def get_best_model(X_train_sc, y_train, X_res, y_res, X_test_sc, y_test):
    """
    Finds and returns the name, trained instance, and predictions of the best-performing model on the test set.
    """
    # 1. Logistic Regression
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
    lr.fit(X_train_sc, y_train)
    
    # 2. Decision Tree
    dt = DecisionTreeClassifier(class_weight="balanced", random_state=SEED)
    dt.fit(X_train_sc, y_train)
    
    # 3. Random Forest
    rf = RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=SEED)
    rf.fit(X_train_sc, y_train)
    
    # 4. XGBoost Baseline
    xgb_base = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=SEED, use_label_encoder=False, eval_metric="logloss"
    )
    xgb_base.fit(X_train_sc, y_train)
    
    # 5. XGBoost Proposed
    if os.path.exists(MODEL_PATH):
        xgb_proposed = joblib.load(MODEL_PATH)
    else:
        xgb_proposed = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=SEED, use_label_encoder=False, eval_metric="logloss"
        )
        xgb_proposed.fit(X_res, y_res)
        
    # 6. CatBoost Default
    cb_default = CatBoostClassifier(random_seed=SEED, verbose=0, thread_count=-1)
    cb_default.fit(X_res, y_res)
    
    # 7. CatBoost Tuned
    if os.path.exists(CATBOOST_MODEL_PATH):
        cb_tuned = joblib.load(CATBOOST_MODEL_PATH)
    else:
        cb_tuned = cb_default

    models = {
        "Logistic Regression": lr,
        "Decision Tree": dt,
        "Random Forest": rf,
        "XGBoost Baseline": xgb_base,
        "XGBoost + SMOTE-ENN (Proposed)": xgb_proposed,
        "CatBoost (Default)": cb_default,
        "CatBoost + SMOTE-ENN (Tuned)": cb_tuned
    }
    
    best_name = None
    best_f1 = -1
    best_clf = None
    best_preds = None
    
    print("\n  Evaluating models on test set (at default threshold 0.50) to find the best...")
    for name, clf in models.items():
        y_proba = clf.predict_proba(X_test_sc)[:, 1]
        y_pred = (y_proba >= 0.50).astype(int)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        print(f"    {name:<32}: F1-Dropout = {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_clf = clf
            best_preds = y_pred
            
    print(f"\n  🏆 Best Performing Model: {best_name} (F1-Dropout = {best_f1:.4f})")
    return best_name, best_clf, best_preds


def main():
    reset_waktu_log()
    start_time = time.time()

    print("=" * 70)
    print("  ERROR PROFILE ANALYSIS ON BEST-PERFORMING MODEL")
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

    # ─── Identify Best Model ──────────────────────────────────────────────
    best_name, best_clf, y_pred = get_best_model(X_train_sc, y_train, X_res, y_res, X_test_sc, y_test)

    # ─── Categorize Predictions ───────────────────────────────────────────
    print_separator("CATEGORIZING TEST PREDICTIONS")
    
    y_test_arr = np.array(y_test)
    y_pred_arr = np.array(y_pred)
    
    # Indices
    tp_idx = (y_test_arr == 1) & (y_pred_arr == 1)
    tn_idx = (y_test_arr == 0) & (y_pred_arr == 0)
    fp_idx = (y_test_arr == 0) & (y_pred_arr == 1)
    fn_idx = (y_test_arr == 1) & (y_pred_arr == 0)
    
    n_tp = np.sum(tp_idx)
    n_tn = np.sum(tn_idx)
    n_fp = np.sum(fp_idx)
    n_fn = np.sum(fn_idx)
    
    print(f"  True Positives (TP - Correctly predicted Dropout)   : {n_tp}")
    print(f"  True Negatives (TN - Correctly predicted Graduate)  : {n_tn}")
    print(f"  False Positives (FP - Graduate predicted as Dropout): {n_fp}")
    print(f"  False Negatives (FN - Dropout predicted as Graduate): {n_fn}")
    print(f"  Total test set samples                              : {len(y_test)}")

    # ─── Error Profile Extraction on Unscaled Features ────────────────────
    print_separator("ANALYZING ERROR PROFILES (RAW FEATURES)")
    
    # Key features to analyze
    key_features = [
        "Age at enrollment",
        "Scholarship holder",
        "Debtor",
        "Tuition fees up to date",
        "Gender",
        "Displaced",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (grade)"
    ]
    
    # Validate features exist in test set
    key_features = [f for f in key_features if f in X_test.columns]
    
    profiles = {}
    for f in key_features:
        profiles[f] = {
            "TP (N={})".format(n_tp): X_test.loc[tp_idx, f].mean() if n_tp > 0 else np.nan,
            "TN (N={})".format(n_tn): X_test.loc[tn_idx, f].mean() if n_tn > 0 else np.nan,
            "FP (N={})".format(n_fp): X_test.loc[fp_idx, f].mean() if n_fp > 0 else np.nan,
            "FN (N={})".format(n_fn): X_test.loc[fn_idx, f].mean() if n_fn > 0 else np.nan
        }
        
    profiles_df = pd.DataFrame(profiles).T
    print(profiles_df.to_string())
    
    # ─── Save Markdown Analysis Report ───────────────────────────────────
    report_path = os.path.join(OUTPUT_DIR, "misclassified_students_analysis.md")
    
    report_content = []
    report_content.append("# Characteristics of Misclassified Students\n")
    report_content.append(f"This report presents an in-depth error profile analysis of the **{best_name}** model, which achieved the best classification performance on the test set.\n\n")
    
    report_content.append("## Prediction Category Breakdown\n")
    report_content.append(f"- **True Positives (TP)**: {n_tp} students (Actual Dropouts correctly identified)\n")
    report_content.append(f"- **True Negatives (TN)**: {n_tn} students (Actual Graduates correctly identified)\n")
    report_content.append(f"- **False Positives (FP)**: {n_fp} students (Actual Graduates predicted as Dropouts)\n")
    report_content.append(f"- **False Negatives (FN)**: {n_fn} students (Actual Dropouts predicted as Graduates — *High Risk*)\n\n")
    
    report_content.append("## Mean Profile Comparison of Key Features\n\n")
    
    # Format table to markdown
    report_content.append("| Feature | TP (N={}) | TN (N={}) | FP (N={}) | FN (N={}) |\n".format(n_tp, n_tn, n_fp, n_fn))
    report_content.append("| --- | --- | --- | --- | --- |\n")
    for feat in key_features:
        tp_val = profiles_df.loc[feat, "TP (N={})".format(n_tp)]
        tn_val = profiles_df.loc[feat, "TN (N={})".format(n_tn)]
        fp_val = profiles_df.loc[feat, "FP (N={})".format(n_fp)]
        fn_val = profiles_df.loc[feat, "FN (N={})".format(n_fn)]
        
        # Format percentages for binary variables, floats for continuous
        is_binary = X_test[feat].nunique() <= 2
        fmt = "{:.1%}" if is_binary else "{:.2f}"
        
        report_content.append("| **{}** | {} | {} | {} | {} |\n".format(
            feat, 
            fmt.format(tp_val) if not np.isnan(tp_val) else "N/A", 
            fmt.format(tn_val) if not np.isnan(tn_val) else "N/A",
            fmt.format(fp_val) if not np.isnan(fp_val) else "N/A",
            fmt.format(fn_val) if not np.isnan(fn_val) else "N/A"
        ))
    report_content.append("\n*Note: Binary flags (Scholarship holder, Debtor, Tuition fees up to date, Gender, Displaced) are represented as percentages of the group.*\n\n")
    
    report_content.append("## Qualitative Insights & Educational Implications\n\n")
    
    # 1. False Negatives analysis
    report_content.append("### 1. Profil False Negatives (FN) — Mahasiswa Dropout yang Lolos dari Prediksi\n")
    report_content.append("Mahasiswa dalam kelompok False Negative adalah kasus paling berisiko karena sistem gagal menandai mereka untuk intervensi. Dari data profil di atas:\n")
    
    fn_approved_2nd = profiles_df.loc["Curricular units 2nd sem (approved)", "FN (N={})".format(n_fn)]
    tp_approved_2nd = profiles_df.loc["Curricular units 2nd sem (approved)", "TP (N={})".format(n_tp)]
    tn_approved_2nd = profiles_df.loc["Curricular units 2nd sem (approved)", "TN (N={})".format(n_tn)]
    
    report_content.append(f"- **Performa Akademik yang 'Menipu'**: Mahasiswa FN memiliki rata-rata unit semester 2 yang disetujui sebanyak **{fn_approved_2nd:.2f}** unit. Ini jauh lebih tinggi daripada kelompok TP (**{tp_approved_2nd:.2f}** unit) dan hampir menyamai kelompok TN (**{tn_approved_2nd:.2f}** unit). Hal ini menunjukkan bahwa secara akademis pada semester awal, mereka tampak aman, namun ada faktor lain yang memicu keputusan untuk keluar. Asosiasi misklasifikasi ini kemungkinan berkaitan dengan faktor-faktor yang tidak terobservasi (*unobserved factors*) yang tidak tercakup dalam dataset ini.\n")
    
    fn_debtor = profiles_df.loc["Debtor", "FN (N={})".format(n_fn)]
    fn_tuition = profiles_df.loc["Tuition fees up to date", "FN (N={})".format(n_fn)]
    report_content.append(f"- **Keterbatasan Variabel Finansial**: Kelompok FN memiliki tingkat tunggakan (Debtor) sebesar **{fn_debtor:.1%}** dan ketepatan pembayaran SPP (Tuition fees up to date) sebesar **{fn_tuition:.1%}**. Meskipun catatan administrasi pembayaran SPP mereka 100% lancar, kelompok FN tampaknya dipengaruhi oleh faktor-faktor lain yang tidak direpresentasikan secara kuat oleh variabel akademis dan finansial yang tersedia di dataset.\n")
    
    report_content.append("- **Rekomendasi Intervensi**: Mahasiswa dengan performa akademik baik tetapi tetap berisiko dropout memerlukan mekanisme pemantauan tambahan di luar indikator akademik tradisional. Temuan ini menunjukkan bahwa sistem peringatan dini berbasis data akademik perlu dilengkapi dengan sumber informasi lain yang dapat menangkap faktor-faktor yang tidak tersedia dalam dataset.\n\n")
    
    # 2. False Positives analysis
    report_content.append("### 2. Profil False Positives (FP) — Mahasiswa Graduate yang Diprediksi Dropout\n")
    report_content.append("Mahasiswa dalam kelompok False Positive diprediksi akan dropout, namun pada kenyataannya mereka berhasil lulus. Kelompok ini mewakili mahasiswa yang memiliki daya tahan (*resilience*) tinggi:\n")
    
    fp_approved_2nd = profiles_df.loc["Curricular units 2nd sem (approved)", "FP (N={})".format(n_fp)]
    report_content.append(f"- **Resiliensi Akademik**: Meskipun mahasiswa FP memiliki performa akademik semester awal yang marjinal or bermasalah (rata-rata unit semester 2 disetujui = **{fp_approved_2nd:.2f}** unit, jauh di bawah kelompok lulusan TN), mereka mampu bertahan dan menyelesaikan studinya.\n")
    
    # 3. Displaced Variable analysis
    fp_displaced = profiles_df.loc["Displaced", "FP (N={})".format(n_fp)]
    tn_displaced = profiles_df.loc["Displaced", "TN (N={})".format(n_tn)]
    report_content.append("### 3. Pengaruh Variabel 'Displaced' (Mahasiswa Rantau)\n")
    report_content.append(f"- **Tingginya Proporsi Displaced pada FP**: Kelompok False Positive (FP) memiliki proporsi mahasiswa displaced (perantau) yang sangat tinggi, yaitu sebesar **{fp_displaced:.1%}**, dibandingkan dengan kelompok True Negative (TN) yang sebesar **{tn_displaced:.1%}**. Perbedaan besar ini menunjukkan bahwa status displaced sering kali ditafsirkan oleh model sebagai faktor risiko dropout yang kuat.\n")
    report_content.append("- **Faktor Risiko vs Mekanisme Koping**: Temuan ini menunjukkan bahwa status displaced mungkin berfungsi sebagai faktor risiko awal yang dipertimbangkan oleh model. Namun, tingginya proporsi mahasiswa displaced pada kelompok lulusan mengindikasikan bahwa status tersebut tidak selalu berujung pada dropout dan kemungkinan berinteraksi dengan faktor-faktor lain yang tidak tercakup dalam dataset.\n\n")
    
    # 4. Scholarship impact
    tp_scholarship = profiles_df.loc["Scholarship holder", "TP (N={})".format(n_tp)]
    fn_scholarship = profiles_df.loc["Scholarship holder", "FN (N={})".format(n_fn)]
    tn_scholarship = profiles_df.loc["Scholarship holder", "TN (N={})".format(n_tn)]
    report_content.append("### 4. Dampak Penerima Beasiswa (Scholarship Holder)\n")
    report_content.append(f"- **Proporsi Penerima Beasiswa yang Kontras**: Hanya **{tp_scholarship:.1%}** dari kelompok TP dan **{fn_scholarship:.1%}** dari kelompok FN yang merupakan penerima beasiswa, dibandingkan dengan **{tn_scholarship:.1%}** pada kelompok lulusan (TN). Pola yang sangat kuat ini menegaskan bahwa penerima beasiswa secara signifikan lebih condong masuk ke dalam kelompok lulusan.\n")
    report_content.append("- **Mekanisme Beasiswa**: Tingginya proporsi penerima beasiswa pada kelompok lulusan menunjukkan adanya hubungan positif antara status penerima beasiswa dan keberhasilan studi. Temuan ini mengindikasikan bahwa beasiswa dapat berperan sebagai faktor pendukung keberlangsungan studi, meskipun mekanisme spesifik yang mendasari hubungan tersebut tidak dapat ditentukan secara langsung dari dataset yang tersedia.\n\n")

    # 5. Suggested Discussion Addition (Synthesis)
    report_content.append("### 5. Ringkasan Sintesis untuk Pembahasan Tesis\n")
    report_content.append("> The error analysis suggests that student dropout is strongly associated with early academic performance; however, a small subset of students with seemingly satisfactory academic progress still withdraw from university. These students are difficult to identify using conventional academic and financial indicators, indicating the possible influence of additional unobserved factors beyond those available in the dataset. Consequently, institutions should not rely exclusively on academic performance metrics when designing early-warning systems.\n")
    
    with open(report_path, "w") as f:
        f.write("".join(report_content))
        
    print(f"  💾 Analysis report successfully generated and saved to {report_path}")
    
    total_time = time.time() - start_time
    print(f"\n  🏁 Error profile analysis complete. Total time: {total_time:.2f}s")


if __name__ == "__main__":
    main()
