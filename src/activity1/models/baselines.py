# baselines.py
# ─────────────────────────────────────────────────────────────────────────────
# Four naive forecasters for SPY next-day log-return.
#
# All four implement the Forecaster interface:
#   - fit(train)     stores the training DataFrame in self._train
#   - observe(val)   extends self._train with new history (inherited default)
#   - predict(ctx)   concatenates self._train["SPY_ret"] with ctx["SPY_ret"],
#                    then shifts / expands / rolls to produce a 1-step-ahead
#                    forecast for every row of ctx
#
# None of these baselines use IEF or ^TNX. That is intentional — their job
# is to serve as a lower bound the exogenous-aware models must beat.
# Any gain from SARIMAX / LSTM / Chronos-2 over these numbers quantifies
# the value added by the yield (^TNX) variable.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import pandas as pd

from .base import Forecaster


# ─── helpers ─────────────────────────────────────────────────────────────────

def _history_then_context(train_ret: pd.Series, ctx_ret: pd.Series) -> pd.Series:
    """
    Concatenate train history with the context return series, keeping only
    the first occurrence of any duplicated index (train takes precedence).
    """
    full = pd.concat([train_ret, ctx_ret])
    return full[~full.index.duplicated(keep="first")].sort_index()


# ─── 1. Naive ────────────────────────────────────────────────────────────────

class NaiveForecaster(Forecaster):
    """
    ŷ_{t+1} = y_t
    "Tomorrow's return equals today's return." Pure momentum assumption.
    """

    name = "naive"

    def fit(self, train: pd.DataFrame) -> "NaiveForecaster":
        self._train = train.copy()
        return self

    def predict(self, context: pd.DataFrame) -> pd.Series:
        # Naive only needs ctx["SPY_ret"] — history is irrelevant for
        # ŷ_{t+1} = y_t. observe() still works (updates self._train) but
        # has no effect on this model's predictions.
        return context["SPY_ret"].copy().rename(self.name)


# ─── 2. Seasonal Naive ───────────────────────────────────────────────────────

class SeasonalNaiveForecaster(Forecaster):
    """
    ŷ_{t+1} = y_{t+1-m}
    Tomorrow equals the return `m` trading days ago. Default m=5 assumes
    weekly seasonality (one business week).
    """

    def __init__(self, m: int = 5) -> None:
        self.m = m
        self.name = f"seasonal_naive_m{m}"

    def fit(self, train: pd.DataFrame) -> "SeasonalNaiveForecaster":
        self._train = train.copy()
        return self

    def predict(self, context: pd.DataFrame) -> pd.Series:
        full = _history_then_context(self._train["SPY_ret"], context["SPY_ret"])
        # target[t] = y_{t+1} → forecast is y_{t+1-m} = y.shift(m-1) at row t
        pred = full.shift(self.m - 1)
        return pred.loc[context.index].rename(self.name)


# ─── 3. Drift / Mean ─────────────────────────────────────────────────────────

class DriftMeanForecaster(Forecaster):
    """
    ŷ_{t+1} = (1/t) · Σ_{i=1..t} y_i     (expanding historical mean)

    On a stationary return series Hyndman's drift method collapses to the
    historical mean. For equity markets this still matters: SPY has a small
    but persistent positive drift (~0.03%/day) that the other baselines miss.
    """

    name = "drift_mean"

    def fit(self, train: pd.DataFrame) -> "DriftMeanForecaster":
        self._train = train.copy()
        return self

    def predict(self, context: pd.DataFrame) -> pd.Series:
        full = _history_then_context(self._train["SPY_ret"], context["SPY_ret"])
        pred = full.expanding(min_periods=1).mean()
        return pred.loc[context.index].rename(self.name)


# ─── 4. Moving Average ───────────────────────────────────────────────────────

class MovingAverageForecaster(Forecaster):
    """
    ŷ_{t+1} = (1/w) · Σ_{i=t-w+1..t} y_i  (rolling mean over the last w days)

    Unlike DriftMean, this has no long-memory bias — the forecast adapts
    to new regimes within `window` days. Default 20 ≈ one trading month.
    """

    def __init__(self, window: int = 20) -> None:
        self.window = window
        self.name = f"ma_{window}"

    def fit(self, train: pd.DataFrame) -> "MovingAverageForecaster":
        self._train = train.copy()
        return self

    def predict(self, context: pd.DataFrame) -> pd.Series:
        full = _history_then_context(self._train["SPY_ret"], context["SPY_ret"])
        pred = full.rolling(window=self.window, min_periods=self.window).mean()
        return pred.loc[context.index].rename(self.name)


# ─── Convenience factory ─────────────────────────────────────────────────────

def all_baselines(*, m: int = 5, ma_window: int = 20) -> list[Forecaster]:
    """Return fresh (un-fit) instances of the four baselines."""
    return [
        NaiveForecaster(),
        SeasonalNaiveForecaster(m=m),
        DriftMeanForecaster(),
        MovingAverageForecaster(window=ma_window),
    ]
