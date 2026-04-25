# run_baselines.py
# ─────────────────────────────────────────────────────────────────────────────
# Entry point for the naive-baseline block of Activity 1.
#
# Steps:
#   1. Load train / val / test splits.
#   2. Fit each baseline on the TRAIN split only (enforced by fit/predict API).
#   3. Predict on val and test.
#   4. Compute RMSE, MAE and Directional Accuracy per baseline per split.
#   5. Save:
#        - reports/activity1/models/baselines_metrics.csv
#        - reports/activity1/models/baselines_val.png
#        - reports/activity1/models/baselines_test.png
#
# Run from project root:
#     python -m src.activity1.pipelines.run_baselines
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os

from ..config import FIGURES_DIR
from ..data.loader import load_splits
from ..evaluation.metrics import compute_metrics, metrics_table
from ..evaluation.plots import plot_actual_vs_predicted
from ..models.baselines import all_baselines


MODELS_DIR = os.path.join(FIGURES_DIR, "models")


def main() -> None:
    print("=" * 55)
    print("Activity 1 — Naive baselines")
    print("=" * 55)

    splits = load_splits()
    train, val, test = splits.train, splits.val, splits.test
    print(f"\n  Loaded splits:  train={len(train)}  val={len(val)}  test={len(test)}")

    # ── 1. Fit each baseline on TRAIN, predict on VAL and TEST ─────────────
    models = all_baselines()
    val_preds:  dict = {}
    test_preds: dict = {}
    results:    dict = {}

    for model in models:
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
        print(f"  ✓  {model.name}")

    # ── 2. Save metrics table ──────────────────────────────────────────────
    table = metrics_table(results)
    os.makedirs(MODELS_DIR, exist_ok=True)
    csv_path = os.path.join(MODELS_DIR, "baselines_metrics.csv")
    table.to_csv(csv_path)

    print("\n  Metrics:")
    print(table.round(6).to_string())
    print(f"\n  Saved: {csv_path}")

    # ── 3. Plot actual vs predicted per split ──────────────────────────────
    plot_actual_vs_predicted(
        y_true=val["target"],
        preds=val_preds,
        title="Activity 1 — Baselines on VALIDATION (SPY next-day log-return)",
        out_path=os.path.join(MODELS_DIR, "baselines_val.png"),
    )
    plot_actual_vs_predicted(
        y_true=test["target"],
        preds=test_preds,
        title="Activity 1 — Baselines on TEST (SPY next-day log-return)",
        out_path=os.path.join(MODELS_DIR, "baselines_test.png"),
    )

    print("\n  Done.")


if __name__ == "__main__":
    main()
