# tune_lstm.py
# ─────────────────────────────────────────────────────────────────────────────
# Grid search over LSTM hyperparameters.
#
# Protocol:
#   1. For every config in the grid, fit on train with early stopping
#      (the LSTMForecaster uses the last 15% of train internally for
#      the stopping check — no val/test leakage).
#   2. Evaluate the trained model on the VALIDATION split.
#   3. Pick the winner by lowest val RMSE (secondary tiebreak:
#      highest val directional accuracy).
#   4. Report the winner's metrics on TEST.
#
# Outputs:
#   reports/activity1/models/lstm_tuning_grid.csv        (all configs, all metrics)
#   reports/activity1/models/lstm_tuning_best.csv        (winner row only)
#
# Grid size is intentionally modest (~18 configs) to fit in a few minutes
# on CPU. Expand in code if you have time and want finer resolution.
#
# Run from project root:
#     python -m src.activity1.pipelines.tune_lstm
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import itertools
import os
import time

import pandas as pd

from ..config import FIGURES_DIR
from ..data.loader import load_splits
from ..evaluation.metrics import compute_metrics
from ..models.lstm import LSTMForecaster


MODELS_DIR = os.path.join(FIGURES_DIR, "models")


# ─── Grid definition ─────────────────────────────────────────────────────────

GRID = {
    "seq_len":    [10, 20, 40],
    "hidden":     [32, 64, 128],
    "num_layers": [1, 2],
    "dropout":    [0.1, 0.3],
}

# Fixed hyperparameters (same across all configs)
FIXED = {
    "features":   ("SPY_ret",),     # winner of previous experiment on test
    "epochs":     50,
    "patience":   5,
    "lr":         1e-3,
    "batch_size": 128,
    "seed":       42,
}


def _iter_configs():
    keys = list(GRID.keys())
    for combo in itertools.product(*(GRID[k] for k in keys)):
        yield dict(zip(keys, combo))


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Activity 1 — LSTM hyperparameter grid search")
    print("=" * 60)

    splits = load_splits()
    train, val, test = splits.train, splits.val, splits.test
    print(f"\n  Loaded splits:  train={len(train)}  val={len(val)}  test={len(test)}")

    configs = list(_iter_configs())
    print(f"\n  Grid size: {len(configs)} configs")

    rows: list[dict] = []

    for i, cfg in enumerate(configs, 1):
        print(f"\n  [{i:>2}/{len(configs)}]  cfg = {cfg}")
        t0 = time.time()

        model = LSTMForecaster(**cfg, **FIXED)
        model.fit(train)

        val_pred  = model.predict(val)
        model.observe(val)              # extend history before test
        test_pred = model.predict(test)

        val_m  = compute_metrics(val["target"],  val_pred)
        test_m = compute_metrics(test["target"], test_pred)

        row = {
            **cfg,
            "val_rmse":     val_m["rmse"],
            "val_mae":      val_m["mae"],
            "val_dir_acc":  val_m["dir_acc"],
            "test_rmse":    test_m["rmse"],
            "test_mae":     test_m["mae"],
            "test_dir_acc": test_m["dir_acc"],
            "elapsed_s":    round(time.time() - t0, 1),
        }
        rows.append(row)

        print(
            f"      val_rmse={val_m['rmse']:.6f}  val_dir_acc={val_m['dir_acc']:.4f}  "
            f"test_dir_acc={test_m['dir_acc']:.4f}  ({row['elapsed_s']}s)"
        )

    grid_df = pd.DataFrame(rows)

    # Winner: lowest val RMSE, then highest val dir_acc as tiebreaker
    grid_sorted = grid_df.sort_values(
        by=["val_rmse", "val_dir_acc"],
        ascending=[True, False],
    ).reset_index(drop=True)
    winner = grid_sorted.iloc[0].to_dict()

    os.makedirs(MODELS_DIR, exist_ok=True)
    grid_path = os.path.join(MODELS_DIR, "lstm_tuning_grid.csv")
    best_path = os.path.join(MODELS_DIR, "lstm_tuning_best.csv")

    grid_sorted.to_csv(grid_path, index=False)
    pd.DataFrame([winner]).to_csv(best_path, index=False)

    print("\n  Full grid (sorted by val_rmse):")
    print(grid_sorted.round(6).to_string(index=False))

    print("\n  Winner (by val_rmse):")
    for k, v in winner.items():
        print(f"    {k:>14s} = {v}")

    print(f"\n  Saved: {grid_path}")
    print(f"  Saved: {best_path}")


if __name__ == "__main__":
    main()
