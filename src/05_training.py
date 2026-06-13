"""
05_training.py — Phase 6: Final Model Training.

Trains the final model on 100% of the resampled training data (X_res, y_res).
Saves the trained model to models/xgb_binary.pkl.
"""

import time
import joblib
from xgboost import XGBClassifier

from src.config import MODEL_PATH
from src.utils import catat_waktu, print_separator


def train_final_model(X_res, y_res, best_params):
    """
    Train the final XGBoost model with best Optuna parameters on 100% of resampled data.

    Args:
        X_res (pd.DataFrame): Resampled training features (SMOTE-ENN applied, scaled).
        y_res (pd.Series): Resampled training target.
        best_params (dict): Best hyperparameters from Optuna.

    Returns:
        model (XGBClassifier): Trained XGBoost model.
    """
    print_separator("PHASE 6: FINAL MODEL TRAINING")
    mulai = time.time()

    model = XGBClassifier(**best_params)
    print(f"\n  Training XGBoost dengan Optuna-optimized parameters pada 100% data resampled...")
    
    model.fit(
        X_res, y_res,
        verbose=False
    )

    print(f"  ✅ Model trained successfully on 100% of training data.")

    # ─── Save model ───────────────────────────────────────────────────────
    joblib.dump(model, MODEL_PATH)
    print(f"  💾 Model saved: {MODEL_PATH}")

    catat_waktu("Model Training", mulai)

    return model
