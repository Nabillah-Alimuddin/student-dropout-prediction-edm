"""
config.py — Central configuration for the Student Dropout Prediction Pipeline.
All paths, constants, and random seeds are defined here.
"""

import os

# ─── Random Seed ───────────────────────────────────────────────────────────────
SEED = 42

# ─── Project Root (parent of src/) ────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─── Data Paths ───────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "dataset.csv")

# ─── Output Directories ──────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# ─── Create directories if they don't exist ───────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── Model Save Path ─────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_binary.pkl")

# ─── Target Column ───────────────────────────────────────────────────────────
TARGET_COL = "Target"

# ─── Train-Test Split ────────────────────────────────────────────────────────
TEST_SIZE = 0.20

# ─── SMOTE-ENN Configuration ─────────────────────────────────────────────────
SMOTE_K_NEIGHBORS = 5
SMOTE_TARGET_RATIO = 0.9  # Target 90% of majority count
ENN_N_NEIGHBORS = 3
ENN_KIND_SEL = "mode"

# ─── Optuna Configuration ────────────────────────────────────────────────────
OPTUNA_N_TRIALS = 200
OPTUNA_TIMEOUT = 7200  # seconds (2 hours)
OPTUNA_CV_FOLDS = 5
OPTUNA_STARTUP_TRIALS = 15
OPTUNA_WARMUP_STEPS = 5

CATBOOST_N_TRIALS = 50
CATBOOST_TIMEOUT = 1800  # seconds (30 minutes)
CATBOOST_MODEL_PATH = os.path.join(MODEL_DIR, "catboost_tuned.pkl")

# ─── LightGBM Configuration ──────────────────────────────────────────────────
LGBM_N_TRIALS = 100
LGBM_TIMEOUT = 3600  # seconds (1 hour)
LGBM_MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_tuned.pkl")

# ─── Cross-Validation ────────────────────────────────────────────────────────
CV_FOLDS = 10

# ─── Significance Level ──────────────────────────────────────────────────────
ALPHA = 0.05

# ─── Threshold Sweep ─────────────────────────────────────────────────────────
THRESHOLD_MIN = 0.05
THRESHOLD_MAX = 0.95
THRESHOLD_STEP = 0.01

