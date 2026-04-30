# imputation.py
# ─────────────────────────────────────────────────────────────────────────────
# Block A — imputation methods.
#
# Each builder returns a (fit, transform) pair encapsulated as an object
# with `.fit(X_train)` (fits on TRAIN ONLY) and `.transform(X)` (applies
# to any split). The contract guarantees no information from val/test
# leaks into the imputer's parameters.
#
# Methods listed in the planning doc:
#   - mean / median  →  univariate baselines (sklearn SimpleImputer)
#   - ffill          →  finance convention; bfill fallback for leading gaps
#   - linear         →  pandas linear interpolation, bfill for leading gaps
#   - knn (k=5)      →  multivariate, sklearn KNNImputer
#   - mice           →  multivariate, sklearn IterativeImputer
#
# Plus the missingness-indicator helper for the MNAR columns
# (GLD_ret, USO_ret, UUP_ret) — see `add_mnar_indicators()`.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  ← side-effect import
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.impute import IterativeImputer

from ..config import RANDOM_SEED


# ─── MNAR columns: assets that did not exist before their inception ──────────
# The 3 retail ETFs that show structural pre-launch NaNs in graph04.
MNAR_COLUMNS = ["GLD_ret", "USO_ret", "UUP_ret"]


# ─── Imputer wrappers ────────────────────────────────────────────────────────
# Each wrapper exposes .fit(X_train) and .transform(X). All operate on
# numeric features only; non-numeric columns must be excluded by the caller.


class _SklearnImputer:
    """Adapter for any sklearn imputer that follows fit/transform."""

    def __init__(self, name: str, estimator):
        self.name = name
        self._estimator = estimator
        self._columns: list[str] = []

    def fit(self, X_train: pd.DataFrame) -> "_SklearnImputer":
        self._columns = list(X_train.columns)
        self._estimator.fit(X_train.values)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        arr = self._estimator.transform(X[self._columns].values)
        return pd.DataFrame(arr, index=X.index, columns=self._columns)


class _PandasImputer:
    """Imputer for ffill/linear that uses pandas semantics directly."""

    def __init__(self, name: str, method: str):
        assert method in {"ffill", "linear"}
        self.name = name
        self._method = method
        self._train_means: pd.Series | None = None  # for filling truly leading NaNs

    def fit(self, X_train: pd.DataFrame) -> "_PandasImputer":
        # We learn nothing from X_train for ffill/linear except a mean fallback
        # for any value that remains NaN after ffill+bfill (rare but possible
        # if a column is entirely NaN at the start of a split).
        self._train_means = X_train.mean(numeric_only=True)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        if self._method == "ffill":
            out = out.ffill().bfill()
        else:  # linear
            out = out.interpolate(method="linear", limit_direction="both")
        # final fallback if a whole column is NaN in this split
        out = out.fillna(self._train_means)
        return out


# ─── Factory ─────────────────────────────────────────────────────────────────


def build_imputer(name: str):
    """
    Build one of the imputers in the planning doc by name.

    Available: 'mean', 'median', 'ffill', 'linear', 'knn', 'mice'.
    """
    name = name.lower()
    if name == "mean":
        return _SklearnImputer("mean", SimpleImputer(strategy="mean"))
    if name == "median":
        return _SklearnImputer("median", SimpleImputer(strategy="median"))
    if name == "ffill":
        return _PandasImputer("ffill", "ffill")
    if name == "linear":
        return _PandasImputer("linear", "linear")
    if name == "knn":
        return _SklearnImputer("knn", KNNImputer(n_neighbors=5))
    if name == "mice":
        return _SklearnImputer(
            "mice",
            IterativeImputer(max_iter=10, random_state=RANDOM_SEED, sample_posterior=False),
        )
    raise ValueError(f"Unknown imputer: {name}")


# ─── MNAR indicator helper ───────────────────────────────────────────────────


def add_mnar_indicators(
    X: pd.DataFrame,
    columns: Iterable[str] = MNAR_COLUMNS,
) -> pd.DataFrame:
    """
    Append a binary `was_missing_<col>` column for each MNAR column.

    Captures the structural fact: "this asset did not exist yet". For
    MNAR data the missingness pattern itself is informative and gets
    lost once the imputer fills the NaN.
    """
    out = X.copy()
    for c in columns:
        if c in out.columns:
            out[f"was_missing_{c}"] = out[c].isna().astype(np.int8)
    return out
