# base.py
# ─────────────────────────────────────────────────────────────────────────────
# Abstract interface that every Activity-2 multi-class classifier must
# implement. Mirrors the philosophy of src/activity1/models/base.py
# (Forecaster) but adapted to classification.
#
# Purpose: enforce train/val/test discipline at the code level, so pipelines
# can iterate over heterogeneous models without if/else dispatch.
#
#   model.fit(X_train, y_train)   → sees ONLY the training split
#   model.predict(X_val)          → hard class labels for hyperparameter tuning
#   model.predict_proba(X_test)   → probabilities for threshold tuning
#                                    and calibration
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Classifier(ABC):
    """Base class for every Activity-2 multi-class regime classifier."""

    #: Short identifier used in metric tables and plot legends.
    name: str = "classifier"

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> "Classifier":
        """
        Fit on the training split ONLY.

        `sample_weight` is the lever for cost-sensitive imbalance handling
        (Block B): pass per-row weights computed from class frequencies
        and sklearn-style estimators will weight the loss accordingly.

        Must return self to allow chaining.
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Hard class predictions, returned as a 1-D numpy array of strings."""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Class probabilities — required for threshold modification and
        calibration plots. Shape (n_samples, n_classes), columns ordered
        by `self.classes_` (set at fit time).
        """

    #: Class labels in the order used by predict_proba columns.
    #: Subclasses must populate this in `fit()` (typically via the
    #: underlying sklearn estimator's `classes_` attribute).
    classes_: np.ndarray = None  # type: ignore[assignment]
