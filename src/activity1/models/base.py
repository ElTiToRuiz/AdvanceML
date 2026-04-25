# base.py
# ─────────────────────────────────────────────────────────────────────────────
# Abstract interface that every Activity-1 forecaster must implement.
#
# Purpose: enforce the train / val / test discipline at the code level.
#
#   model.fit(train)          → sees ONLY the training split
#   model.predict(val)        → used to tune hyperparameters
#   model.predict(test)       → touched exactly ONCE for the final report
#
# All models (Naive baselines, SARIMAX, LSTM, Chronos-2) share this API so
# the pipelines and the evaluation code are model-agnostic.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd


class Forecaster(ABC):
    """Base class for every 1-step-ahead forecaster of SPY next-day log-return."""

    #: Short identifier used in metric tables and plot legends.
    name: str = "forecaster"

    # ── Training ────────────────────────────────────────────────────────────

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> "Forecaster":
        """
        Fit the model on the training split ONLY.

        The DataFrame passed in must be the processed training set
        (with at least `SPY_ret` and `target`, plus optional exogenous
        columns such as `TNX_yield`). Implementations are free to ignore
        any column they do not need.

        Subclasses should store the data they need in `self._train` so
        the default `observe()` can extend it without refitting.

        Must return `self` to allow chaining: `model.fit(train).predict(val)`.
        """

    # ── Inference ───────────────────────────────────────────────────────────

    @abstractmethod
    def predict(self, context: pd.DataFrame) -> pd.Series:
        """
        Produce 1-step-ahead forecasts for every row of `context`.

        The returned Series is aligned with `context.index`: the value
        at row t is the forecast for day t+1 (directly comparable with
        `context['target'][t]`).

        Implementations must NOT use information from after each row t
        when forecasting row t+1 — i.e. no peeking into the future.
        """

    # ── History extension ───────────────────────────────────────────────────

    def observe(self, new_data: pd.DataFrame) -> "Forecaster":
        """
        Extend the model's known history with `new_data` WITHOUT
        retraining or refitting.

        Use case: between `predict(val)` and `predict(test)`, call
        `observe(val)` so that test-time forecasts have a contiguous
        train + val history (matches realistic deployment, where val
        data has already been observed by the time test starts).

        Default implementation: append `new_data` to `self._train`
        and de-duplicate by index. Subclasses with extra state
        (e.g. cached scalers fitted on train) may override but must
        leave fit-time statistics unchanged — only the OBSERVED
        history grows; learned parameters do not.
        """
        if not hasattr(self, "_train"):
            raise RuntimeError(
                f"{type(self).__name__}.observe() called before .fit()"
            )
        combined = pd.concat([self._train, new_data])
        self._train = combined[~combined.index.duplicated(keep="first")].sort_index()
        return self
