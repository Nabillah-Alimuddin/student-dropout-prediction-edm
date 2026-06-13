"""
02_preprocessing.py — Phase 2 & 3: Stratified Train-Test Split and Feature Scaling.

Splits data 80/20 with stratification, then applies StandardScaler
(fit on train only, transform both).
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import SEED, TEST_SIZE
from src.utils import catat_waktu, print_separator


def split_and_scale(X, y):
    """
    Stratified train-test split followed by StandardScaler.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Binary target.

    Returns:
        X_train (pd.DataFrame): Raw training features.
        X_test (pd.DataFrame): Raw test features.
        X_train_sc (pd.DataFrame): Scaled training features.
        X_test_sc (pd.DataFrame): Scaled test features.
        y_train (pd.Series): Training target.
        y_test (pd.Series): Test target.
        scaler (StandardScaler): Fitted scaler object.
    """
    print_separator("PHASE 2: STRATIFIED TRAIN-TEST SPLIT")
    mulai = time.time()

    # ─── Stratified Split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set:     {X_test.shape[0]} samples")
    print(f"  Train class distribution:")
    print(f"    Dropout (1): {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.1f}%)")
    print(f"    Graduate (0): {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.1f}%)")
    print(f"  Test class distribution:")
    print(f"    Dropout (1): {(y_test == 1).sum()} ({(y_test == 1).mean()*100:.1f}%)")
    print(f"    Graduate (0): {(y_test == 0).sum()} ({(y_test == 0).mean()*100:.1f}%)")

    catat_waktu("Train-Test Split", mulai)

    # ─── Feature Scaling ──────────────────────────────────────────────────
    print_separator("PHASE 3: FEATURE SCALING (StandardScaler)")
    mulai = time.time()

    scaler = StandardScaler()

    # Fit on training data ONLY, transform both
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    print(f"  Scaler fitted on training data only.")
    print(f"  Scaled training shape: {X_train_sc.shape}")
    print(f"  Scaled test shape:     {X_test_sc.shape}")

    catat_waktu("Feature Scaling", mulai)

    return X_train, X_test, X_train_sc, X_test_sc, y_train, y_test, scaler
