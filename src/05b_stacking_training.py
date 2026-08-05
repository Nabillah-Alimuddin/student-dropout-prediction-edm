"""
05b_stacking_training.py — Deprecated alias for src.stacking_training.

This module re-exports StackingEnsemble and train_stacking_ensemble
from src.stacking_training for backward compatibility.
"""

from src.stacking_training import StackingEnsemble, train_stacking_ensemble

__all__ = ["StackingEnsemble", "train_stacking_ensemble"]
