# run_chronos.py
# ─────────────────────────────────────────────────────────────────────────────
# Chronos-2 pipeline for Activity 1.
#
# Zero-shot comparison of two Chronos-2 configurations:
#   1. univariate — only SPY log-returns
#   2. with TNX_chg as a past+future covariate
#
# Runs rolling 1-step-ahead forecasts on val and test. Because Chronos-2
# is 120M params on CPU, this pipeline is noticeably slower than the others
# (minutes, not seconds).
#
# Run from project root:
#     python -m src.activity1.pipelines.run_chronos
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os
import time

from ..config import FIGURES_DIR
from ..data.loader import load_splits
from ..evaluation.metrics import compute_metrics, metrics_table
from ..evaluation.plots import plot_actual_vs_predicted
from ..models.chronos import Chronos2Forecaster


MODELS_DIR = os.path.join(FIGURES_DIR, "models")


def main() -> None:
    print("=" * 55)
    print("Activity 1 — Chronos-2 (zero-shot foundation model)")
    print("=" * 55)

    splits = load_splits()
    train, val, test = splits.train, splits.val, splits.test
    print(f"\n  Loaded splits:  train={len(train)}  val={len(val)}  test={len(test)}")

    models = [
        Chronos2Forecaster(past_covariates=()),
        Chronos2Forecaster(past_covariates=("TNX_chg",)),
    ]

    val_preds:  dict = {}
    test_preds: dict = {}
    results:    dict = {}

    for model in models:
        print(f"\n  Loading {model.name} ...")
        t0 = time.time()
        model.fit(train)
        print(f"    model loaded in {time.time()-t0:.1f}s")

        print(f"  Predicting validation ...")
        t0 = time.time()
        val_pred = model.predict(val)
        print(f"    val done in {time.time()-t0:.1f}s")

        model.observe(val)              # extend history before test

        print(f"  Predicting test ...")
        t0 = time.time()
        test_pred = model.predict(test)
        print(f"    test done in {time.time()-t0:.1f}s")

        val_preds[model.name]  = val_pred
        test_preds[model.name] = test_pred

        results[model.name] = {
            "val":  compute_metrics(val["target"],  val_pred),
            "test": compute_metrics(test["target"], test_pred),
        }

    table = metrics_table(results)
    os.makedirs(MODELS_DIR, exist_ok=True)
    csv_path = os.path.join(MODELS_DIR, "chronos_metrics.csv")
    table.to_csv(csv_path)

    print("\n  Metrics:")
    print(table.round(6).to_string())
    print(f"\n  Saved: {csv_path}")

    plot_actual_vs_predicted(
        y_true=val["target"],
        preds=val_preds,
        title="Activity 1 — Chronos-2 on VALIDATION (SPY next-day log-return)",
        out_path=os.path.join(MODELS_DIR, "chronos_val.png"),
    )
    plot_actual_vs_predicted(
        y_true=test["target"],
        preds=test_preds,
        title="Activity 1 — Chronos-2 on TEST (SPY next-day log-return)",
        out_path=os.path.join(MODELS_DIR, "chronos_test.png"),
    )

    print("\n  Done.")


if __name__ == "__main__":
    main()
