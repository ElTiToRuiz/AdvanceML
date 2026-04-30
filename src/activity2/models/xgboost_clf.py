# xgboost_clf.py
# ─────────────────────────────────────────────────────────────────────────────
# Gradient-boosted trees wrapper (XGBoost).
#
# Each tree corrects the residuals of the previous ensemble. With L1/L2
# regularisation and early stopping XGBoost is the strongest off-the-
# shelf model for tabular multi-class problems — the planning doc
# explicitly bets on it as the winner.
#
# Class labels (strings) are encoded once at fit-time because xgboost's
# multi:softprob requires integer y. We hide the encoding behind the
# Classifier interface so callers always see string predictions.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from .base import Classifier
from ..config import RANDOM_SEED, REGIME_ORDER


class XGBoostModel(Classifier):
    name = "xgboost"

    def __init__(
        self,
        n_estimators: int = 400,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda

    def fit(self, X_train, y_train, sample_weight=None):
        # Encode class strings to ints, freeze label order so it matches
        # REGIME_ORDER (so SHAP / confusion-matrix axes stay consistent).
        self._encoder = LabelEncoder().fit(REGIME_ORDER)
        y_int = self._encoder.transform(y_train)
        self._model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        self._model.fit(X_train, y_int, sample_weight=sample_weight)
        self.classes_ = self._encoder.classes_
        return self

    def predict(self, X):
        y_int = self._model.predict(X)
        return self._encoder.inverse_transform(y_int)

    def predict_proba(self, X):
        return self._model.predict_proba(X)

    @property
    def booster(self):
        """Access to the underlying XGBoost estimator — needed for Tree SHAP."""
        return self._model
