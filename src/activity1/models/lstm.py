# lstm.py
# ─────────────────────────────────────────────────────────────────────────────
# LSTM forecaster for SPY next-day log-return.
#
# Architecture:
#   input features (default: SPY_ret, TNX_chg)
#       → LSTM(hidden, num_layers, dropout)
#       → Linear(hidden → 1)
#       → next-day log-return prediction
#
# Training:
#   - Sliding windows of length `seq_len` built from the train split.
#   - Loss: MSE on the target log-return.
#   - Optimiser: Adam.
#   - Early stopping: if val loss has not improved for `patience` epochs,
#     revert to the best checkpoint and stop.
#
# Rolling 1-step-ahead inference:
#   At row t of `context`, the prediction uses the last `seq_len` feature
#   rows ending at t (from train + observed context). No peeking.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
from typing import Sequence
import copy

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .base import Forecaster


# ─── Torch module ────────────────────────────────────────────────────────────

class _LSTMNet(nn.Module):
    def __init__(self, n_features: int, hidden: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)            # (B, T, H)
        last   = out[:, -1, :]           # take last time step
        return self.head(last).squeeze(-1)  # (B,)


# ─── Forecaster wrapper ──────────────────────────────────────────────────────

class LSTMForecaster(Forecaster):
    """PyTorch LSTM wrapped in the Forecaster interface."""

    def __init__(
        self,
        features:   Sequence[str] = ("SPY_ret", "TNX_chg"),
        seq_len:    int   = 20,
        hidden:     int   = 64,
        num_layers: int   = 2,
        dropout:    float = 0.2,
        epochs:     int   = 50,
        patience:   int   = 5,
        lr:         float = 1e-3,
        batch_size: int   = 128,
        seed:       int   = 42,
    ) -> None:
        self.features   = list(features)
        self.seq_len    = seq_len
        self.hidden     = hidden
        self.num_layers = num_layers
        self.dropout    = dropout
        self.epochs     = epochs
        self.patience   = patience
        self.lr         = lr
        self.batch_size = batch_size
        self.seed       = seed
        self.name       = (
            f"lstm_h{hidden}_L{num_layers}_seq{seq_len}"
            f"_feat_{'_'.join(features)}"
        )

    # ── Internal helpers ────────────────────────────────────────────────

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        return (x - self._feat_mean) / self._feat_std

    def _make_windows(self, feats: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Build sliding windows of length seq_len.
        X[i] = feats[i : i + seq_len]            (seq_len × n_features)
        y[i] = target[i + seq_len - 1]           (forecast for the NEXT day,
                                                  which is stored in `target`)
        """
        n = len(feats) - self.seq_len + 1
        X = np.stack([feats[i : i + self.seq_len] for i in range(n)])
        y = target[self.seq_len - 1 : self.seq_len - 1 + n]
        mask = ~np.isnan(y)
        return X[mask], y[mask]

    # ── Fit ─────────────────────────────────────────────────────────────

    def fit(self, train: pd.DataFrame) -> "LSTMForecaster":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self._train = train.copy()

        # Standardise features using TRAIN-ONLY statistics (no leakage)
        feats_train = train[self.features].values.astype(np.float32)
        self._feat_mean = feats_train.mean(axis=0)
        self._feat_std  = feats_train.std(axis=0) + 1e-8
        feats_std = self._standardize(feats_train)

        target_train = train["target"].values.astype(np.float32)

        X, y = self._make_windows(feats_std, target_train)

        # Last 15% of training windows as internal val for early stopping
        n_val_internal = max(1, int(len(X) * 0.15))
        X_tr, X_vl = X[:-n_val_internal], X[-n_val_internal:]
        y_tr, y_vl = y[:-n_val_internal], y[-n_val_internal:]

        X_tr_t = torch.from_numpy(X_tr)
        y_tr_t = torch.from_numpy(y_tr)
        X_vl_t = torch.from_numpy(X_vl)
        y_vl_t = torch.from_numpy(y_vl)

        loader = DataLoader(
            TensorDataset(X_tr_t, y_tr_t),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self._net = _LSTMNet(len(self.features), self.hidden, self.num_layers, self.dropout)
        opt       = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        loss_fn   = nn.MSELoss()

        best_state: dict | None = None
        best_val   = float("inf")
        stale      = 0

        for epoch in range(1, self.epochs + 1):
            self._net.train(mode=True)
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(self._net(xb), yb)
                loss.backward()
                opt.step()

            self._net.train(mode=False)
            with torch.no_grad():
                vl = loss_fn(self._net(X_vl_t), y_vl_t).item()

            if vl < best_val - 1e-6:
                best_val   = vl
                best_state = copy.deepcopy(self._net.state_dict())
                stale      = 0
            else:
                stale += 1

            if epoch == 1 or epoch % 5 == 0 or stale == 0:
                print(f"    epoch {epoch:>3} | val_loss={vl:.6e}  best={best_val:.6e}  stale={stale}")

            if stale >= self.patience:
                print(f"    early stopping at epoch {epoch}")
                break

        if best_state is not None:
            self._net.load_state_dict(best_state)
        self._net.train(mode=False)
        return self

    # ── Predict ─────────────────────────────────────────────────────────

    def predict(self, context: pd.DataFrame) -> pd.Series:
        """
        Rolling 1-step-ahead: for each row t in context, build a window of
        length seq_len ending at t (using train + observed context up to t),
        run it through the network, and emit the forecast for day t+1.
        """
        full = pd.concat([self._train, context])
        full = full[~full.index.duplicated(keep="first")].sort_index()

        feats      = full[self.features].values.astype(np.float32)
        feats_std  = self._standardize(feats)

        # Precompute windows for every context index
        idx_full   = full.index
        ctx_pos    = np.flatnonzero(idx_full.isin(context.index))

        windows: list[np.ndarray] = []
        valid_pos: list[int] = []
        for pos in ctx_pos:
            start = pos - self.seq_len + 1
            if start < 0:
                continue           # not enough history
            windows.append(feats_std[start : pos + 1])
            valid_pos.append(pos)

        if not windows:
            return pd.Series(index=context.index, dtype=float, name=self.name)

        X_t = torch.from_numpy(np.stack(windows))
        with torch.no_grad():
            preds = self._net(X_t).numpy()

        out = pd.Series(index=context.index, dtype=float, name=self.name)
        pred_index = idx_full[valid_pos]
        out.loc[pred_index] = preds
        return out
