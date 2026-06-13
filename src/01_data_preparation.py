"""
01_data_preparation.py — Phase 0 & 1: Data Loading, Binary Filtering, Target Encoding.

Loads the raw CSV, removes 'Enrolled' records, encodes target as binary
(Dropout=1, Graduate=0), and separates features from target.
"""

import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.config import DATA_PATH, SEED, TARGET_COL
from src.utils import catat_waktu, save_pdf, print_separator


def load_and_prepare_data():
    """
    Load dataset, filter to binary classification, encode target.

    Returns:
        X (pd.DataFrame): Feature matrix (all columns except target).
        y (pd.Series): Binary target (Dropout=1, Graduate=0).
        df (pd.DataFrame): The filtered DataFrame with encoded target.
    """
    print_separator("PHASE 0-1: DATA LOADING & BINARY FILTERING")
    mulai = time.time()

    # ─── Load CSV ─────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    print(f"  Raw dataset shape: {df.shape}")
    print(f"  Missing values: {df.isnull().sum().sum()}")

    # ─── Original class distribution ──────────────────────────────────────
    print(f"\n  Original class distribution:")
    print(df[TARGET_COL].value_counts().to_string().replace("\n", "\n  "))

    # ─── Filter: Remove 'Enrolled' — keep only Graduate & Dropout ────────
    df = df[df[TARGET_COL] != "Enrolled"].copy()
    print(f"\n  After removing 'Enrolled': {df.shape}")

    # ─── Binary class distribution ────────────────────────────────────────
    distribusi = df[TARGET_COL].value_counts()
    print(f"\n  Binary class distribution:")
    print(distribusi.to_string().replace("\n", "\n  "))

    # ─── Visualize binary distribution (bar + pie) ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    colors = ["#2ecc71", "#e74c3c"]
    distribusi.plot(kind="bar", ax=axes[0], color=colors, edgecolor="black")
    axes[0].set_title("Distribusi Kelas (Binary)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Kelas")
    axes[0].set_ylabel("Jumlah")
    axes[0].tick_params(axis="x", rotation=0)
    for i, v in enumerate(distribusi.values):
        axes[0].text(i, v + 20, str(v), ha="center", fontweight="bold")

    # Pie chart
    axes[1].pie(distribusi.values, labels=distribusi.index,
                autopct="%1.1f%%", colors=colors, startangle=90,
                explode=[0.05] * len(distribusi))
    axes[1].set_title("Proporsi Kelas (Binary)", fontsize=14, fontweight="bold")

    plt.tight_layout()
    save_pdf(fig, "v3_distribusi_binary.pdf")
    plt.close(fig)

    # ─── Imbalance ratio ─────────────────────────────────────────────────
    minority = distribusi.min()
    majority = distribusi.max()
    rasio = minority / majority
    print(f"\n  Imbalance ratio (minority/majority): {rasio:.4f}")

    # ─── Target Encoding ─────────────────────────────────────────────────
    print_separator("PHASE 1: TARGET ENCODING")
    df["Target_enc"] = df[TARGET_COL].map({"Dropout": 1, "Graduate": 0})
    print(f"  Encoding: Dropout → 1, Graduate → 0")

    # ─── Separate features and target ─────────────────────────────────────
    y = df["Target_enc"].copy()
    X = df.drop(columns=[TARGET_COL, "Target_enc"]).copy()

    print(f"  Features: {X.shape[1]} columns")
    print(f"  Class distribution:")
    print(f"    Dropout (1): {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")
    print(f"    Graduate (0): {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")

    catat_waktu("Data Loading & Preparation", mulai)

    return X, y, df
