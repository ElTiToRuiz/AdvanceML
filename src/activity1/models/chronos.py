# chronos.py
# ─────────────────────────────────────────────────────────────────────────────
# Chronos-2 zero-shot forecaster for SPY next-day log-return.
#
# Chronos-2 (Amazon Science, Oct 2025) is a 120M-parameter encoder-only
# foundation model for time-series forecasting. Key property: ZERO-SHOT —
# no training on our data is done. The pretrained weights (trained on
# millions of diverse time series) are applied directly to our SPY returns.
#
# Unlike Chronos v1, v2 natively supports covariates, so we can pass
# TNX_chg as a past / future covariate — making Chronos-2 directly
# comparable to SARIMAX-with-exog and to LSTM with multiple features.
#
# Caveat on future_covariates:  we feed TNX_chg[t+1] as a future covariate
# when predicting SPY_ret[t+1]. This matches the SARIMAX setup where the
# contemporaneous exogenous is provided to the Kalman filter when
# computing the 1-step-ahead forecast; both models therefore use the
# same information when the TNX variant is enabled.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline

from .base import Forecaster


class Chronos2Forecaster(Forecaster):
    """Zero-shot forecaster wrapping the Chronos-2 HuggingFace model."""

    def __init__(
        self,
        model_id:          str = "amazon/chronos-2",
        past_covariates:   Sequence[str] = (),
        context_length:    int = 1024,
        batch_size:        int = 16,
        seed:              int = 42,
    ) -> None:
        self.model_id        = model_id
        self.past_covariates = list(past_covariates)
        self.context_length  = context_length
        self.batch_size      = batch_size
        self.seed            = seed

        tag = "_".join(past_covariates) if past_covariates else "none"
        self.name = f"chronos2_cov_{tag}"

    # ── Fit (no training — just load the model and store history) ──────

    def fit(self, train: pd.DataFrame) -> "Chronos2Forecaster":
        torch.manual_seed(self.seed)
        self._train = train.copy()
        self._pipeline = Chronos2Pipeline.from_pretrained(
            self.model_id,
            device_map="cpu",
            torch_dtype=torch.float32,
        )
        return self

    # ── Predict ────────────────────────────────────────────────────────

    def predict(self, context: pd.DataFrame) -> pd.Series:
        full = pd.concat([self._train, context])
        full = full[~full.index.duplicated(keep="first")].sort_index()

        y = full["SPY_ret"].values.astype(np.float32)
        covs: dict = {c: full[c].values.astype(np.float32) for c in self.past_covariates}

        ctx_pos = np.flatnonzero(full.index.isin(context.index))

        inputs: list[dict] = []
        valid_pos: list[int] = []

        for pos in ctx_pos:
            if pos + 1 >= len(full):
                # No future covariate available for the very last row.
                continue
            start = max(0, pos + 1 - self.context_length)
            item = {
                "target": torch.from_numpy(y[start : pos + 1].copy()),
            }
            if self.past_covariates:
                item["past_covariates"] = {
                    c: torch.from_numpy(covs[c][start : pos + 1].copy())
                    for c in self.past_covariates
                }
                item["future_covariates"] = {
                    c: torch.from_numpy(covs[c][pos + 1 : pos + 2].copy())
                    for c in self.past_covariates
                }
            inputs.append(item)
            valid_pos.append(pos)

        if not inputs:
            return pd.Series(index=context.index, dtype=float, name=self.name)

        print(f"    running {len(inputs)} Chronos-2 forecasts "
              f"(batch_size={self.batch_size}, context_length={self.context_length}) ...")
        preds_list = self._pipeline.predict(
            inputs,
            prediction_length=1,
            batch_size=self.batch_size,
        )

        point_preds = [_to_point_forecast(p) for p in preds_list]

        out = pd.Series(index=context.index, dtype=float, name=self.name)
        out.loc[full.index[valid_pos]] = point_preds
        return out


# ─── Fine-tuned variant ──────────────────────────────────────────────────────

class FineTunedChronos2Forecaster(Chronos2Forecaster):
    """
    Same as Chronos2Forecaster, but `fit(train)` runs a short fine-tuning
    pass on the training split instead of just loading pretrained weights.

    Defaults to LoRA (Low-Rank Adaptation) because full fine-tuning of a
    120M model on CPU is prohibitively slow. LoRA trains a small set of
    adapter matrices while keeping the base weights frozen — orders of
    magnitude cheaper and still effective for domain adaptation.
    """

    def __init__(
        self,
        model_id:            str = "amazon/chronos-2",
        past_covariates:     Sequence[str] = (),
        context_length:      int = 1024,
        batch_size:          int = 16,
        finetune_mode:       str = "lora",     # "lora" | "full"
        learning_rate:       float = 1e-5,     # LoRA likes it higher
        num_steps:           int = 200,
        finetune_batch_size: int = 8,
        seed:                int = 42,
    ) -> None:
        super().__init__(
            model_id=model_id,
            past_covariates=past_covariates,
            context_length=context_length,
            batch_size=batch_size,
            seed=seed,
        )
        self.finetune_mode       = finetune_mode
        self.learning_rate       = learning_rate
        self.num_steps           = num_steps
        self.finetune_batch_size = finetune_batch_size

        tag = "_".join(past_covariates) if past_covariates else "none"
        self.name = f"chronos2_ft_{finetune_mode}_cov_{tag}"

    def fit(self, train: pd.DataFrame) -> "FineTunedChronos2Forecaster":
        torch.manual_seed(self.seed)
        self._train = train.copy()

        base = Chronos2Pipeline.from_pretrained(
            self.model_id,
            device_map="cpu",
            torch_dtype=torch.float32,
        )

        # Build a single-series fine-tuning input. The trainer samples
        # random context/target windows from this series internally.
        y_train = train["SPY_ret"].values.astype(np.float32)
        item: dict = {"target": torch.from_numpy(y_train.copy())}

        if self.past_covariates:
            item["past_covariates"] = {
                c: torch.from_numpy(train[c].values.astype(np.float32))
                for c in self.past_covariates
            }
            # future_covariates is only used as a SCHEMA hint during training
            # (it tells the model which covariates are known into the future).
            # The actual values are ignored, so a 1-element placeholder is fine.
            item["future_covariates"] = {
                c: torch.zeros(1, dtype=torch.float32)
                for c in self.past_covariates
            }

        print(
            f"    fine-tuning Chronos-2  mode={self.finetune_mode}  "
            f"steps={self.num_steps}  lr={self.learning_rate}  "
            f"batch_size={self.finetune_batch_size} ..."
        )
        self._pipeline = base.fit(
            inputs=[item],
            prediction_length=1,
            finetune_mode=self.finetune_mode,
            learning_rate=self.learning_rate,
            num_steps=self.num_steps,
            batch_size=self.finetune_batch_size,
            disable_data_parallel=True,
            remove_printer_callback=True,
        )
        return self


# ─── helpers ─────────────────────────────────────────────────────────────────

def _to_point_forecast(p) -> float:
    """
    Chronos-2 may return samples, quantiles, or means depending on version.
    This helper reduces whatever comes back to a single point forecast
    (median if multiple values, otherwise the single value).
    """
    t = p if isinstance(p, torch.Tensor) else torch.as_tensor(p)
    t = t.squeeze()
    if t.ndim == 0:
        return float(t.item())
    # 1-D: samples or quantiles → take the median
    return float(t.median().item())
