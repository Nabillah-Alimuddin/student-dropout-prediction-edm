"""
08_mcnemar.py — Phase 9: McNemar Statistical Significance Test.

Pairwise McNemar test: proposed model vs. each baseline.
Tests whether error pattern differences are statistically significant.
Uses exact test when (b+c) < 25, chi-squared approximation otherwise.
"""

import time
import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

from src.config import ALPHA
from src.utils import catat_waktu, print_separator


def run_mcnemar_tests(y_test, predictions_dict):
    """
    Perform pairwise McNemar tests: proposed model vs. each baseline.

    Args:
        y_test: Test target values.
        predictions_dict (dict): {model_name: y_pred} for all models.

    Returns:
        mcnemar_results (pd.DataFrame): McNemar test results table.
    """
    print_separator("PHASE 9: McNEMAR STATISTICAL SIGNIFICANCE TEST")
    mulai = time.time()

    # Find the proposed model name dynamically
    proposed_candidates = [
        "Stacking (Proposed) + SMOTE-ENN",
        "Stacking (XGB+LGB+CB) + SMOTE-ENN",
        "XGBoost + SMOTE-ENN (Proposed)",
        "XGBoost + SMOTE-ENN"
    ]
    proposed_name = None
    for cand in proposed_candidates:
        if cand in predictions_dict:
            proposed_name = cand
            break
            
    if proposed_name is None:
        # Fallback: find any key containing "Proposed" (very specific!)
        for key in predictions_dict.keys():
            if "Proposed" in key:
                proposed_name = key
                break
                
    if proposed_name is None:
        # Fallback: find any key containing "Stacking"
        for key in predictions_dict.keys():
            if "Stacking" in key:
                proposed_name = key
                break
                
    if proposed_name is None:
        # Final fallback: last key
        proposed_name = list(predictions_dict.keys())[-1]

    y_pred_proposed = predictions_dict[proposed_name]

    results_rows = []

    for name, y_pred_baseline in predictions_dict.items():
        if name == proposed_name:
            continue

        # Build contingency table
        # correct_proposed & wrong_baseline (b)
        # wrong_proposed & correct_baseline (c)
        correct_proposed = (y_pred_proposed == y_test)
        correct_baseline = (y_pred_baseline == y_test)

        b = np.sum(correct_proposed & ~correct_baseline)  # Proposed correct, baseline wrong
        c = np.sum(~correct_proposed & correct_baseline)  # Proposed wrong, baseline correct

        # Build 2x2 contingency table
        a = np.sum(correct_proposed & correct_baseline)   # Both correct
        d = np.sum(~correct_proposed & ~correct_baseline)  # Both wrong

        table = np.array([[a, b], [c, d]])

        # Choose exact or chi-squared
        use_exact = (b + c) < 25
        result = mcnemar(table, exact=use_exact)

        significant = "✅ Yes" if result.pvalue < ALPHA else "❌ No"

        results_rows.append({
            "Comparison": f"Proposed vs. {name}",
            "b (Proposed✓ Baseline✗)": b,
            "c (Proposed✗ Baseline✓)": c,
            "Method": "Exact" if use_exact else "Chi-squared",
            "Statistic": round(result.statistic, 4),
            "p-value": round(result.pvalue, 6),
            f"Significant (α={ALPHA})": significant,
        })

        print(f"\n  Proposed vs. {name}:")
        print(f"    b={b}, c={c} → {'Exact' if use_exact else 'Chi-sq'}")
        print(f"    Statistic: {result.statistic:.4f}, p-value: {result.pvalue:.6f}")
        print(f"    Significant: {significant}")

    mcnemar_results = pd.DataFrame(results_rows)

    print(f"\n  Summary Table:")
    print(mcnemar_results.to_string(index=False))

    catat_waktu("McNemar Test", mulai)

    return mcnemar_results
