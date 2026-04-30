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

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.utils.class_weight import compute_sample_weight

from ..config import (
    ASYMMETRIC_COST_WEIGHTS,
    CRASH_OPTUNA_TRIALS,
    RANDOM_SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    XGB_OPTUNA_RANGES,
)
from ..data.loader import load_with_nans
from ..evaluation.operational import (
    asymmetric_sample_weights, evaluate_at_threshold, find_operating_point,
)
from ..models.xgboost_clf import XGBoostModel
from ..preprocessing.imputation import add_mnar_indicators, build_imputer


warnings.filterwarnings("ignore")
TARGET_CLASS = "crash"
optuna.logging.set_verbosity(optuna.logging.WARNING)


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


# ─── Shared prepare (replicates Block C imputation, no resampling) ──────────


def _prepare_imputed_splits(winner_imp: str):
    """
    Reproduce the imputed train/val/test splits used in Phase 2,
    WITHOUT applying the imbalance resampling (Block E uses sample
    weights instead).
    """
    df = load_with_nans()
    y = df["regime"].dropna()
    X = df.loc[y.index].drop(columns=["regime"])
    if "SPY_price" in X.columns:
        X = X.drop(columns=["SPY_price"])
    X = X.select_dtypes(include="number")

    if winner_imp.endswith("+mnar_indicator"):
        X = add_mnar_indicators(X)
        method = winner_imp.replace("+mnar_indicator", "")
    else:
        method = winner_imp

    n = len(X); n_tr = int(n * TRAIN_RATIO); n_va_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    X_tr, X_va, X_te = X.iloc[:n_tr], X.iloc[n_tr:n_va_end], X.iloc[n_va_end:]
    y_tr, y_va, y_te = y.iloc[:n_tr], y.iloc[n_tr:n_va_end], y.iloc[n_va_end:]

    imp = build_imputer(method).fit(X_tr)
    return imp.transform(X_tr), y_tr, imp.transform(X_va), y_va, imp.transform(X_te), y_te


# ─── §4.2 — Asymmetric-cost re-tuning ───────────────────────────────────────


def run_asymmetric_retune(bundle: dict) -> XGBoostModel:
    """
    Re-tune XGBoost with explicit per-class cost weights. The Optuna
    objective is recall on the crash class on val, with a hard
    constraint that crash precision >= PRECISION_FLOOR_CRASH (otherwise
    the trial returns 0 → the optimiser learns to avoid useless models).
    """
    print("\n— §4.2 Asymmetric re-tuning —")
    X_tr, y_tr, X_va, y_va, X_te, y_te = _prepare_imputed_splits(bundle["winner_imp"])
    sw_train = asymmetric_sample_weights(y_tr, ASYMMETRIC_COST_WEIGHTS)

    history = []

    def objective(trial):
        params = {
            "learning_rate":    trial.suggest_float("learning_rate",    *XGB_OPTUNA_RANGES["learning_rate"]),
            "max_depth":        trial.suggest_int(  "max_depth",        *XGB_OPTUNA_RANGES["max_depth"]),
            "n_estimators":     trial.suggest_int(  "n_estimators",     *XGB_OPTUNA_RANGES["n_estimators"]),
            "subsample":        trial.suggest_float("subsample",        *XGB_OPTUNA_RANGES["subsample"]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *XGB_OPTUNA_RANGES["colsample_bytree"]),
            "reg_alpha":        trial.suggest_float("reg_alpha",        *XGB_OPTUNA_RANGES["reg_alpha"], log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda",       *XGB_OPTUNA_RANGES["reg_lambda"], log=True),
        }
        model = XGBoostModel(**params).fit(X_tr, y_tr, sample_weight=sw_train)
        proba_val = model.predict_proba(X_va)
        crash_idx = list(model.classes_).index(TARGET_CLASS)
        op = find_operating_point(
            y_va.astype(str).values, proba_val[:, crash_idx],
            target_class=TARGET_CLASS,
            precision_floor=PRECISION_FLOOR_CRASH,
            detector_name="asym",
        )
        score = op.recall if op.precision >= PRECISION_FLOOR_CRASH else 0.0
        history.append({**params, "trial": trial.number,
                        "val_recall_crash": op.recall,
                        "val_precision_crash": op.precision,
                        "score": score})
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=RANDOM_SEED),
        pruner=MedianPruner(),
    )
    study.optimize(objective, n_trials=CRASH_OPTUNA_TRIALS, show_progress_bar=False)
    pd.DataFrame(history).to_csv(
        os.path.join(REPORT_CRASH_FOCUS_DIR, "asymmetric_optuna_history.csv"),
        index=False,
    )

    best_params = study.best_params
    print(f"  Optuna best score (val recall_crash @ precision>=0.10): {study.best_value:.4f}")
    print(f"  Best params: {best_params}")

    model = XGBoostModel(**best_params).fit(X_tr, y_tr, sample_weight=sw_train)
    joblib.dump(model, os.path.join(REPORT_CRASH_FOCUS_DIR, "crash_xgb_asymmetric.joblib"))
    return model


# ─── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 60)
    print("Activity 2 — Block E: crash-focused analysis")
    print("=" * 60)
    os.makedirs(REPORT_CRASH_FOCUS_DIR, exist_ok=True)

    bundle = _load_bundle()
    calibrated = run_calibration(bundle)
    asymmetric = run_asymmetric_retune(bundle)
    # §4.3-§4.5 added in later tasks
    return None


if __name__ == "__main__":
    main()
