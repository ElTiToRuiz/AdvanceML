# imbalance.py
# ─────────────────────────────────────────────────────────────────────────────
# Block B — imbalance handling strategies.
#
# Each strategy is exposed as one of three kinds:
#
#   1) Resampler:  modifies (X_train, y_train) before fit.
#                  SMOTE / ADASYN / SMOTE+ENN / RandomUnderSampler.
#                  Function signature: (X_train, y_train) -> (X_res, y_res)
#
#   2) Reweight:   leaves the data alone; produces sample_weight or
#                  passes class_weight='balanced' to the model.
#
#   3) Threshold:  post-training; takes predict_proba on val + true val
#                  labels and finds the per-class decision thresholds
#                  that maximise macro-F1.
#
# This file gives every strategy its own builder and a uniform
# `build_strategy(name)` entry point used by the pipelines.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight

from ..config import RANDOM_SEED


# ─── Reweighting ─────────────────────────────────────────────────────────────


def balanced_sample_weight(y: pd.Series) -> np.ndarray:
    """Per-row weights inversely proportional to class frequency."""
    return compute_sample_weight(class_weight="balanced", y=y)


# ─── Resampling ──────────────────────────────────────────────────────────────


def _resample(name: str, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply a TRAIN-ONLY resampler. Returns the resampled (X, y).

    Resampler choice:
        smote        → SMOTE (k=5)
        adasyn       → ADASYN (focuses on hard regions)
        smote_enn    → SMOTE + Edited Nearest Neighbours cleaning
        rus          → RandomUnderSampler (drop majority-class rows)

    NOTE: SMOTE family interpolates in feature space — it MUST run on
    the imputed feature matrix, never on raw NaN-containing data.
    """
    sampler_map = {
        "smote":      SMOTE(random_state=RANDOM_SEED, k_neighbors=5),
        "adasyn":     ADASYN(random_state=RANDOM_SEED, n_neighbors=5),
        "smote_enn":  SMOTEENN(random_state=RANDOM_SEED),
        "rus":        RandomUnderSampler(random_state=RANDOM_SEED),
    }
    sampler = sampler_map[name]
    X_res, y_res = sampler.fit_resample(X.values, y.values)
    return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)


# ─── Threshold tuning ────────────────────────────────────────────────────────


def tune_thresholds(
    y_val: np.ndarray,
    proba_val: np.ndarray,
    classes: list[str],
) -> dict:
    """
    Find per-class probability shifts on val that maximise macro-F1.

    Strategy: grid search additive shifts to the rare-class probabilities
    (crash, correction). The argmax of the shifted probabilities becomes
    the new prediction.

    Returns a dict {class_name: float} of additive shifts to predict_proba.
    """
    classes = list(classes)
    base = np.argmax(proba_val, axis=1)
    base_pred = np.array(classes)[base]
    best_macro = f1_score(y_val, base_pred, labels=classes, average="macro", zero_division=0)
    best_shifts = {c: 0.0 for c in classes}

    grid = np.linspace(-0.20, 0.20, 21)
    for s_crash in grid:
        for s_correction in grid:
            shifted = proba_val.copy()
            shifted[:, classes.index("crash")]      += s_crash
            shifted[:, classes.index("correction")] += s_correction
            pred = np.array(classes)[np.argmax(shifted, axis=1)]
            f1 = f1_score(y_val, pred, labels=classes, average="macro", zero_division=0)
            if f1 > best_macro:
                best_macro = f1
                best_shifts = {c: 0.0 for c in classes}
                best_shifts["crash"] = float(s_crash)
                best_shifts["correction"] = float(s_correction)
    return best_shifts


def apply_thresholds(proba: np.ndarray, classes: list[str], shifts: dict) -> np.ndarray:
    """Apply per-class additive shifts to a predict_proba matrix and argmax."""
    out = proba.copy()
    for i, c in enumerate(classes):
        out[:, i] += shifts.get(c, 0.0)
    return np.array(classes)[np.argmax(out, axis=1)]


# ─── Public entry: resolve a strategy name to flags ──────────────────────────


@dataclass
class ImbalanceStrategy:
    """The output of `build_strategy(name)` — what each pipeline applies."""
    name: str
    resampler: str | None = None      # one of {smote, adasyn, smote_enn, rus} or None
    use_class_weight: bool = False    # pass class_weight='balanced' to the model
    use_sample_weight: bool = False   # pass balanced sample_weight to .fit()
    use_threshold: bool = False       # post-fit threshold tuning


def build_strategy(name: str) -> ImbalanceStrategy:
    name = name.lower()
    if name == "untouched":
        return ImbalanceStrategy(name="untouched")
    if name == "class_weight":
        return ImbalanceStrategy(name="class_weight", use_class_weight=True)
    if name in {"smote", "adasyn", "smote_enn", "rus"}:
        return ImbalanceStrategy(name=name, resampler=name)
    if name == "threshold":
        return ImbalanceStrategy(name="threshold", use_threshold=True)
    raise ValueError(f"Unknown imbalance strategy: {name}")
