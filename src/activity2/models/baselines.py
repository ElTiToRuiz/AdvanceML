# baselines.py
# ─────────────────────────────────────────────────────────────────────────────
# Trivial baselines that any real model must beat. They expose the
# same Classifier interface as the ML wrappers.
#
#   AlwaysMostCommon  →  predict the majority training class for every row.
#                        On this dataset (~50% normal, ~28% rally,
#                        ~18% correction, ~4% crash) this hits ~50%
#                        accuracy and 0% recall on crash. The whole
#                        plan is designed to expose this exact failure.
#
#   StratifiedRandom  →  draw predictions according to the training-class
#                        prior. Macro-F1 ≈ 0.25 — a sanity floor even for
#                        the rare classes.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import numpy as np
import pandas as pd

from .base import Classifier
from ..config import RANDOM_SEED, REGIME_ORDER


class AlwaysMostCommon(Classifier):
    name = "baseline_most_common"

    def fit(self, X_train, y_train, sample_weight=None):
        counts = pd.Series(y_train).value_counts()
        self._mode = counts.idxmax()
        self.classes_ = np.array(REGIME_ORDER)
        prior = np.array([counts.get(c, 0) for c in self.classes_], dtype=float)
        self._prior = prior / prior.sum()
        return self

    def predict(self, X):
        return np.full(len(X), self._mode, dtype=object)

    def predict_proba(self, X):
        return np.tile(self._prior, (len(X), 1))


class StratifiedRandom(Classifier):
    name = "baseline_stratified"

    def __init__(self, seed: int = RANDOM_SEED):
        self.seed = seed

    def fit(self, X_train, y_train, sample_weight=None):
        counts = pd.Series(y_train).value_counts()
        self.classes_ = np.array(REGIME_ORDER)
        prior = np.array([counts.get(c, 0) for c in self.classes_], dtype=float)
        self._prior = prior / prior.sum()
        self._rng = np.random.default_rng(self.seed)
        return self

    def predict(self, X):
        return self._rng.choice(self.classes_, size=len(X), p=self._prior).astype(object)

    def predict_proba(self, X):
        return np.tile(self._prior, (len(X), 1))


def all_baselines() -> list[Classifier]:
    return [AlwaysMostCommon(), StratifiedRandom()]
