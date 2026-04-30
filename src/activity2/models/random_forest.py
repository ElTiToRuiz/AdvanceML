# random_forest.py
# ─────────────────────────────────────────────────────────────────────────────
# Random Forest wrapper.
#
# Bagging ensemble of decision trees, each trained on a bootstrap
# sample with feature sub-sampling. Picks up non-linear interactions
# without manual feature engineering — useful here because Figure 9
# of the EDA showed the rare classes occupy outer regions of (SPY_ret,
# VIX_ret) space that a linear boundary cannot capture.
#
# No standardisation — trees are scale-invariant.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .base import Classifier
from ..config import RANDOM_SEED


class RandomForestModel(Classifier):
    name = "random_forest"

    def __init__(
        self,
        n_estimators: int = 400,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        max_features: str | float | None = "sqrt",
        class_weight: str | dict | None = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight

    def fit(self, X_train, y_train, sample_weight=None):
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            class_weight=self.class_weight,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        self._model.fit(X_train, y_train, sample_weight=sample_weight)
        self.classes_ = self._model.classes_
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)
