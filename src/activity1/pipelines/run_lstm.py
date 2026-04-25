# run_lstm.py
# ─────────────────────────────────────────────────────────────────────────────
# LSTM pipeline for Activity 1.
#
# Trains a 2-layer LSTM on train, monitors internal early-stopping loss,
# then evaluates with rolling 1-step-ahead forecasts on val and test.
#
# Run from project root:
#     python -m src.activity1.pipelines.run_lstm
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os

from ..config import FIGURES_DIR
from ..data.loader import load_splits
from ..evaluation.metrics import compute_metrics, metrics_table
from ..evaluation.plots import plot_actual_vs_predicted
from ..models.lstm import LSTMForecaster


MODELS_DIR = os.path.join(FIGURES_DIR, "models")


def main() -> None:
    print("=" * 55)
    print("Activity 1 — LSTM")
    print("=" * 55)

    splits = load_splits()
    train, val, test = splits.train, splits.val, splits.test
    print(f"\n  Loaded splits:  train={len(train)}  val={len(val)}  test={len(test)}")

    # ── Models: isolate the effect of adding TNX_chg as a second feature ──
    models = [
        LSTMForecaster(features=("SPY_ret",)),
        LSTMForecaster(features=("SPY_ret", "TNX_chg")),
    ]

    val_preds:  dict = {}
    test_preds: dict = {}
    results:    dict = {}

    for model in models:
        print(f"\n  Training {model.name} ...")
        model.fit(train)
        val_pred  = model.predict(val)
        model.observe(val)              # extend history before test
        test_pred = model.predict(test)

        val_preds[model.name]  = val_pred
        test_preds[model.name] = test_pred

        results[model.name] = {
            "val":  compute_metrics(val["target"],  val_pred),
            "test": compute_metrics(test["target"], test_pred),
        }

    # ── Metrics ────────────────────────────────────────────────────────────
    table = metrics_table(results)
    os.makedirs(MODELS_DIR, exist_ok=True)
    csv_path = os.path.join(MODELS_DIR, "lstm_metrics.csv")
    table.to_csv(csv_path)

    print("\n  Metrics:")
    print(table.round(6).to_string())
    print(f"\n  Saved: {csv_path}")

    # ── Plots ──────────────────────────────────────────────────────────────
    plot_actual_vs_predicted(
        y_true=val["target"],
        preds=val_preds,
        title="Activity 1 — LSTM on VALIDATION (SPY next-day log-return)",
        out_path=os.path.join(MODELS_DIR, "lstm_val.png"),
    )
    plot_actual_vs_predicted(
        y_true=test["target"],
        preds=test_preds,
        title="Activity 1 — LSTM on TEST (SPY next-day log-return)",
        out_path=os.path.join(MODELS_DIR, "lstm_test.png"),
    )

    print("\n  Done.")


if __name__ == "__main__":
    main()
