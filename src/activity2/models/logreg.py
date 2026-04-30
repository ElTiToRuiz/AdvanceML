# logreg.py
# ─────────────────────────────────────────────────────────────────────────────
# Multinomial logistic regression wrapper.
#
# The "softmax" baseline of multi-class ML: linear decision boundaries
# in feature space, one weight vector per class, joined by a softmax.
# Chosen as the LINEAR ladder rung — anything more complex (RF, XGBoost)
# must beat this to justify the extra capacity.
#
# Standardisation is done internally: a StandardScaler is fitted on
# X_train and applied to X at predict time. This lets imputation /
# imbalance pipelines pass raw features without thinking about scaling.
#
# Note on the sklearn 1.8 API: `penalty` and `n_jobs` were deprecated.
# The new API uses `l1_ratio` (0 = pure L2, 1 = pure L1, between =
# elasticnet) which is what the grid in config.py uses.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .base import Classifier
from ..config import RANDOM_SEED


class LogRegClassifier(Classifier):
    name = "logreg"

    def __init__(
        self,
        C: float = 1.0,
        l1_ratio: float = 0.0,
        class_weight: str | dict | None = None,
        max_iter: int = 2000,
        solver: str = "saga",
    ):
        self.C = C
        self.l1_ratio = l1_ratio
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.solver = solver

    def fit(self, X_train, y_train, sample_weight=None):
        self._scaler = StandardScaler().fit(X_train)
        Xs = self._scaler.transform(X_train)
        self._model = LogisticRegression(
            C=self.C,
            l1_ratio=self.l1_ratio,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            solver=self.solver,
            random_state=RANDOM_SEED,
        )
        self._model.fit(Xs, y_train, sample_weight=sample_weight)
        self.classes_ = self._model.classes_
        return self

    def predict(self, X):
        return self._model.predict(self._scaler.transform(X))

    def predict_proba(self, X):
        return self._model.predict_proba(self._scaler.transform(X))
