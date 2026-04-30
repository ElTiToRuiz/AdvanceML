# metrics.py
# ─────────────────────────────────────────────────────────────────────────────
# Classification metrics suite for Activity 2.
#
# Primary metric:
#   - macro_f1       Average of per-class F1. Insensitive to class
#                    frequency, so the rare classes matter as much as
#                    the common ones. Pick winners with this.
#
# Imbalance-aware single-number metrics (class notes recommend both):
#   - mcc            Matthews Correlation Coefficient. Symmetric, in [-1, 1].
#   - g_mean         Geometric mean of per-class recalls. Collapses to 0
#                    if ANY class has zero recall — a hard penalty for
#                    "ignore the rare class" predictors.
#
# Secondary:
#   - balanced_accuracy   Macro recall (= G-Mean's arithmetic counterpart)
#   - per_class_pr        Precision and recall per class as a tidy DataFrame
#   - accuracy            Reported "for completeness", NEVER used to pick.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    recall_score,
)

from ..config import REGIME_ORDER


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, labels=REGIME_ORDER, average="macro", zero_division=0))


def mcc(y_true, y_pred) -> float:
    return float(matthews_corrcoef(y_true, y_pred))


def g_mean(y_true, y_pred) -> float:
    """Geometric mean of per-class recalls. 0 if any class has zero recall."""
    recalls = recall_score(y_true, y_pred, labels=REGIME_ORDER, average=None, zero_division=0)
    if np.any(recalls == 0):
        return 0.0
    return float(np.exp(np.mean(np.log(recalls))))


def balanced_accuracy(y_true, y_pred) -> float:
    return float(balanced_accuracy_score(y_true, y_pred))


def accuracy(y_true, y_pred) -> float:
    return float(accuracy_score(y_true, y_pred))


def per_class_pr(y_true, y_pred) -> pd.DataFrame:
    p, r, f, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=REGIME_ORDER, zero_division=0
    )
    return pd.DataFrame(
        {"precision": p, "recall": r, "f1": f, "support": sup.astype(int)},
        index=REGIME_ORDER,
    )


def confusion(y_true, y_pred) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=REGIME_ORDER)
    return pd.DataFrame(cm, index=REGIME_ORDER, columns=REGIME_ORDER)


def compute_all(y_true, y_pred) -> dict[str, float]:
    """One-call summary used by every pipeline."""
    return {
        "macro_f1":           macro_f1(y_true, y_pred),
        "balanced_accuracy":  balanced_accuracy(y_true, y_pred),
        "mcc":                mcc(y_true, y_pred),
        "g_mean":             g_mean(y_true, y_pred),
        "accuracy":           accuracy(y_true, y_pred),
        "recall_crash":       float(recall_score(y_true, y_pred, labels=["crash"],      average="macro", zero_division=0)),
        "recall_correction":  float(recall_score(y_true, y_pred, labels=["correction"], average="macro", zero_division=0)),
        "recall_normal":      float(recall_score(y_true, y_pred, labels=["normal"],     average="macro", zero_division=0)),
        "recall_rally":       float(recall_score(y_true, y_pred, labels=["rally"],      average="macro", zero_division=0)),
    }
