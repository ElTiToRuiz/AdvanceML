# sarimax.py
# ─────────────────────────────────────────────────────────────────────────────
# SARIMAX forecaster for SPY next-day log-return.
#
# SARIMAX(p, d, q)(P, D, Q, s)  with exogenous variables.
#
#   - AR(p)  : "tomorrow depends on the last p returns"
#   - I(d)   : d differences. Returns are already stationary (ADF p<0.01),
#              so we default to d=0 on the return series.
#   - MA(q)  : corrects the last q forecast errors.
#   - X      : exogenous variables. Default: TNX_chg (daily change in the
#              10-year Treasury yield, in percentage points). We use the
#              CHANGE because the level is non-stationary, so a stationary
#              endogenous series paired with a non-stationary exog would
#              break SARIMAX assumptions.
#
# Why ^TNX and not IEF:  the 10-year yield is the causal, theoretically
# motivated variable (risk-free rate in DCF / CAPM). IEF price and ^TNX
# yield carry near-identical information (corr ≈ -1 on daily changes), so
# including both would create multicollinearity with no benefit.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
from typing import Tuple, Sequence

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .base import Forecaster


class SarimaxForecaster(Forecaster):
    """SARIMAX(p,d,q)(P,D,Q,s) with optional exogenous columns."""

    def __init__(
        self,
        order: Tuple[int, int, int]      = (1, 0, 1),
        seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
        exog_cols: Sequence[str]          = ("TNX_chg",),
    ) -> None:
        self.order          = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.exog_cols      = list(exog_cols) if exog_cols else []

        # Compact human-readable name used in metric tables and plots
        tag_exog = "_".join(self.exog_cols) if self.exog_cols else "none"
        self.name = (
            f"sarimax_{self.order[0]}{self.order[1]}{self.order[2]}"
            f"_exog_{tag_exog}"
        )

    # ── Fit ──────────────────────────────────────────────────────────────

    def fit(self, train: pd.DataFrame) -> "SarimaxForecaster":
        """Fit on train only. Stores fitted state for rolling prediction."""
        self._train = train.copy()
        y_train     = train["SPY_ret"]
        exog_train  = train[self.exog_cols] if self.exog_cols else None

        self._model = SARIMAX(
            endog=y_train,
            exog=exog_train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            # Returns often don't satisfy strict invertibility; we relax both
            # to let the optimiser converge on the real MLE.
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._fitted = self._model.fit(disp=False, maxiter=200)
        return self

    # ── Predict ──────────────────────────────────────────────────────────

    def predict(self, context: pd.DataFrame) -> pd.Series:
        """
        Rolling 1-step-ahead forecast for every row of `context`.

        Implementation:
          1. Concatenate train endog + context endog (and same for exog).
          2. Re-apply the fitted parameters to the extended series with
             `.apply(...)` — this does NOT re-fit, it only updates the
             filter state.
          3. Call `get_prediction(dynamic=False)` so each in-sample forecast
             uses the ACTUAL past observations (not previously predicted
             ones). This is exactly a rolling 1-step-ahead forecast.

        The value at row t of the returned Series is the forecast for day
        t+1 (comparable with context['target'][t]).
        """
        y_full = pd.concat([self._train["SPY_ret"], context["SPY_ret"]])
        y_full = y_full[~y_full.index.duplicated(keep="first")].sort_index()

        exog_full = None
        if self.exog_cols:
            exog_full = pd.concat(
                [self._train[self.exog_cols], context[self.exog_cols]]
            )
            exog_full = exog_full[~exog_full.index.duplicated(keep="first")]
            exog_full = exog_full.sort_index()

        applied = self._fitted.apply(endog=y_full, exog=exog_full, refit=False)

        # In-sample 1-step-ahead predictions for the context window.
        # These correspond to the model's forecast at time t for time t+1
        # given all observations up to and including t.
        pred_in_sample = applied.predict(start=0, end=len(y_full) - 1)

        # Shift by -1: pred_in_sample at index t+1 is the forecast made using
        # info up to t, which is the forecast "at time t for day t+1".
        # We want row t to hold the forecast FOR day t+1, so we shift forward
        # by one position.
        pred_shifted = pred_in_sample.shift(-1)

        return pred_shifted.loc[context.index].rename(self.name)
