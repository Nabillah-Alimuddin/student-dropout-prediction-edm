"""
03_smoteenn.py — Phase 4: SMOTE-ENN Binary Resampling

Mengikuti implementasi notebook V3 FINAL:
1. SMOTE oversampling kelas Dropout
2. Edited Nearest Neighbours (ENN) cleaning
3. Diterapkan HANYA pada training set

Tidak ada perubahan metodologi dari notebook.
"""

import time
import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours

from src.config import (
    SEED,
    SMOTE_K_NEIGHBORS,
    SMOTE_TARGET_RATIO,
    ENN_N_NEIGHBORS,
    ENN_KIND_SEL
)

from src.utils import (
    catat_waktu,
    print_separator
)


def apply_smoteenn(X_train_sc, y_train):
    """
    Apply SMOTE-ENN exactly as implemented in notebook.

    Parameters
    ----------
    X_train_sc : pd.DataFrame
        Scaled training features.

    y_train : pd.Series
        Training target.

    Returns
    -------
    X_res : pd.DataFrame
        Resampled training features.

    y_res : pd.Series
        Resampled training target.
    """

    print_separator("PHASE 4: SMOTE-ENN BINARY")

    mulai = time.time()

    # ============================================================
    # BEFORE RESAMPLING
    # ============================================================
    n_dropout_train = (y_train == 1).sum()
    n_graduate_train = (y_train == 0).sum()

    print("Before SMOTE-ENN:")
    print(f"  Graduate (0): {n_graduate_train}")
    print(f"  Dropout  (1): {n_dropout_train}")

    # ============================================================
    # SMOTE
    # ============================================================
    smote = SMOTE(
        k_neighbors=SMOTE_K_NEIGHBORS,
        random_state=SEED,
        sampling_strategy=SMOTE_TARGET_RATIO
    )

    # ============================================================
    # ENN
    # ============================================================
    enn = EditedNearestNeighbours(
        n_neighbors=ENN_N_NEIGHBORS,
        kind_sel=ENN_KIND_SEL
    )

    # ============================================================
    # STEP 1 : SMOTE
    # ============================================================
    X_smote, y_smote = smote.fit_resample(
        X_train_sc,
        y_train
    )

    # ============================================================
    # STEP 2 : ENN
    # ============================================================
    X_res, y_res = enn.fit_resample(
        X_smote,
        y_smote
    )

    # ============================================================
    # CONVERT TO DATAFRAME
    # ============================================================
    X_res = pd.DataFrame(
        X_res,
        columns=X_train_sc.columns
    )

    y_res = pd.Series(
        y_res,
        name=y_train.name if y_train.name else "Target_enc"
    )

    # ============================================================
    # AFTER RESAMPLING
    # ============================================================
    n_res_grad = (y_res == 0).sum()
    n_res_drop = (y_res == 1).sum()

    rasio = n_res_drop / n_res_grad

    print("\nAfter SMOTE-ENN:")
    print(f"  Graduate (0): {n_res_grad}")
    print(f"  Dropout  (1): {n_res_drop}")
    print(f"  Ratio        : {rasio:.3f} ({rasio*100:.1f}%)")

    status = (
        "✅ Balanced"
        if rasio >= 0.80
        else "⚠️ Need inspection"
    )

    print(f"  Status       : {status}")

    catat_waktu(
        "SMOTE-ENN Binary",
        mulai
    )

    return X_res, y_res