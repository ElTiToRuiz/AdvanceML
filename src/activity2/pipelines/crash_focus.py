# crash_focus.py
# ─────────────────────────────────────────────────────────────────────────────
# Activity 2 — Block E: crash-focused analysis.
#
# Loads the Phase-2 winner bundle and pushes recall on the crash class as
# far as it can go, subject to a precision floor. Five sub-analyses:
#   §4.1  Probability calibration (Platt scaling on the Phase-2 XGBoost)
#   §4.2  Asymmetric-cost re-tuning (Optuna 100 trials, crash 10x weight)
#   §4.3  Threshold tuning + PR curve on the test set
#   §4.4  Binary reframe (crash vs not-crash) as side analysis
#   §4.5  Long-only "exit on signal" backtest vs buy-and-hold
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os
import warnings

import joblib
import numpy as np
import pandas as pd

from ..config import (
    PRECISION_FLOOR_CRASH,
    REGIME_ORDER,
    REPORT_CRASH_FOCUS_DIR,
    REPORT_MODELS_DIR,
)
from ..evaluation.operational import (
    CalibratedClassifierWrapper,
    plot_reliability,
    reliability_diagram,
)


warnings.filterwarnings("ignore")
TARGET_CLASS = "crash"


# ─── I/O ─────────────────────────────────────────────────────────────────────


def _load_bundle() -> dict:
    path = os.path.join(REPORT_MODELS_DIR, "winner_bundle.joblib")
    bundle = joblib.load(path)
    print(f"  Loaded winner bundle: {bundle['winner_name']} "
          f"(imp={bundle['winner_imp']}, imb={bundle['winner_imb']})")
    return bundle


# ─── §4.1 — Calibration ─────────────────────────────────────────────────────


def run_calibration(bundle: dict) -> CalibratedClassifierWrapper:
    """
    Apply Platt scaling on top of the Phase-2 XGBoost using val. Plot the
    reliability diagram before and after for the crash class on test.
    """
    print("\n— §4.1 Calibration —")

    base = bundle["winner_model"]
    X_val, y_val = bundle["X_val"], bundle["y_val"]
    X_test, y_test = bundle["X_test"], bundle["y_test"]

    # Reliability — BEFORE
    proba_before = base.predict_proba(X_test)
    crash_idx_before = list(base.classes_).index(TARGET_CLASS)
    rel_before = reliability_diagram(
        y_test.astype(str).values, proba_before[:, crash_idx_before],
        TARGET_CLASS, n_bins=10,
    )
    plot_reliability(
        rel_before, "Reliability — XGBoost raw probabilities", TARGET_CLASS,
        os.path.join(REPORT_CRASH_FOCUS_DIR, "calibration_before.png"),
    )

    # Calibrate on val, render reliability AFTER on test
    calibrated = CalibratedClassifierWrapper(base, X_val, y_val)
    proba_after = calibrated.predict_proba(X_test)
    crash_idx_after = list(calibrated.classes_).index(TARGET_CLASS)
    rel_after = reliability_diagram(
        y_test.astype(str).values, proba_after[:, crash_idx_after],
        TARGET_CLASS, n_bins=10,
    )
    plot_reliability(
        rel_after, "Reliability — Platt-scaled XGBoost", TARGET_CLASS,
        os.path.join(REPORT_CRASH_FOCUS_DIR, "calibration_after.png"),
    )

    # Persist the calibrated model so later sub-analyses can reuse it
    joblib.dump(calibrated, os.path.join(REPORT_CRASH_FOCUS_DIR, "crash_xgb_calibrated.joblib"))
    print(f"  Calibrated model saved → crash_xgb_calibrated.joblib")
    return calibrated


# ─── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 60)
    print("Activity 2 — Block E: crash-focused analysis")
    print("=" * 60)
    os.makedirs(REPORT_CRASH_FOCUS_DIR, exist_ok=True)

    bundle = _load_bundle()
    calibrated = run_calibration(bundle)
    # §4.2-§4.5 added in later tasks
    return None


if __name__ == "__main__":
    main()
