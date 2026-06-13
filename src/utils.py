"""
utils.py — Utility functions for the Student Dropout Prediction Pipeline.
Provides timing, saving, and display helpers.
"""

import time
import os
import matplotlib.pyplot as plt
from src.config import OUTPUT_DIR


# ─── Timing Utility ──────────────────────────────────────────────────────────
_waktu_log = []


def catat_waktu(nama_fase, mulai):
    """Record execution time for a pipeline phase."""
    durasi = time.time() - mulai
    _waktu_log.append({
        "Fase": nama_fase,
        "Detik": round(durasi, 2),
        "Menit": round(durasi / 60, 4)
    })
    print(f"  ⏱  {nama_fase}: {durasi:.2f} detik ({durasi/60:.4f} menit)")
    return durasi


def get_waktu_log():
    """Return the accumulated timing log."""
    return _waktu_log


def reset_waktu_log():
    """Reset the timing log."""
    global _waktu_log
    _waktu_log = []


# ─── PDF Save Helper ─────────────────────────────────────────────────────────
def save_pdf(fig, filename):
    """Save a matplotlib figure as PDF to the outputs directory."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, format="pdf", bbox_inches="tight", dpi=300)
    print(f"  📄 Saved: {filepath}")
    return filepath


# ─── Display Helper ───────────────────────────────────────────────────────────
def print_separator(title=""):
    """Print a visual separator for console output."""
    print(f"\n{'='*70}")
    if title:
        print(f"  {title}")
        print(f"{'='*70}")
