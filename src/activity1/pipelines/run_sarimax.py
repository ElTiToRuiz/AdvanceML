# run_sarimax.py
# ─────────────────────────────────────────────────────────────────────────────
# SARIMAX pipeline for Activity 1.
#
# Compares three SARIMAX variants to isolate the contribution of the
# exogenous TNX_chg input:
#
#   1. sarimax_101_exog_none   — pure ARIMA(1,0,1), NO exogenous
#   2. sarimax_101_exog_TNX_chg — adds daily yield change
#   3. sarimax_202_exog_TNX_chg — slightly richer AR/MA dynamics
#
# The first two together answer the key economic question of the project:
# "Does the 10-year yield actually improve SPY forecasts?"
#
# Run from project root:
#     python -m src.activity1.pipelines.run_sarimax
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os

from ..config import FIGURES_DIR
from ..data.loader import load_splits
from ..evaluation.metrics import compute_metrics, metrics_table
from ..evaluation.plots import plot_actual_vs_predicted
from ..models.sarimax import SarimaxForecaster


MODELS_DIR = os.path.join(FIGURES_DIR, "models")


def main() -> None:
    print("=" * 55)
    print("Activity 1 — SARIMAX")
    print("=" * 55)

    splits = load_splits()
    train, val, test = splits.train, splits.val, splits.test
    print(f"\n  Loaded splits:  train={len(train)}  val={len(val)}  test={len(test)}")

    # ── Models to compare ──────────────────────────────────────────────────
    models = [
        SarimaxForecaster(order=(1, 0, 1), exog_cols=()),                 # no exog
        SarimaxForecaster(order=(1, 0, 1), exog_cols=("TNX_chg",)),       # + yield
        SarimaxForecaster(order=(2, 0, 2), exog_cols=("TNX_chg",)),       # richer
    ]

    val_preds:  dict = {}
    test_preds: dict = {}
    results:    dict = {}

    for model in models:
        print(f"\n  Fitting {model.name} ...")
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
        print(f"    AIC={model._fitted.aic:.1f}  BIC={model._fitted.bic:.1f}")

    # ── Metrics table ──────────────────────────────────────────────────────
    table = metrics_table(results)
    os.makedirs(MODELS_DIR, exist_ok=True)
    csv_path = os.path.join(MODELS_DIR, "sarimax_metrics.csv")
    table.to_csv(csv_path)

    print("\n  Metrics:")
    print(table.round(6).to_string())
    print(f"\n  Saved: {csv_path}")

    # ── Plots ──────────────────────────────────────────────────────────────
    plot_actual_vs_predicted(
        y_true=val["target"],
        preds=val_preds,
        title="Activity 1 — SARIMAX on VALIDATION (SPY next-day log-return)",
        out_path=os.path.join(MODELS_DIR, "sarimax_val.png"),
    )
    plot_actual_vs_predicted(
        y_true=test["target"],
        preds=test_preds,
        title="Activity 1 — SARIMAX on TEST (SPY next-day log-return)",
        out_path=os.path.join(MODELS_DIR, "sarimax_test.png"),
    )

    print("\n  Done.")


if __name__ == "__main__":
    main()
