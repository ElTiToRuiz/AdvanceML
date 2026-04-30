# final_evaluation.py
# ─────────────────────────────────────────────────────────────────────────────
# Block D — touch the test set ONCE.
#
# Loads the winner bundle written by tune_models.py, runs predict + every
# metric, writes:
#   reports/activity2/final/final_metrics.csv
#   reports/activity2/final/per_class_metrics.csv
#   reports/activity2/final/confusion_matrix.png
#   reports/activity2/final/per_class_metrics_bar.png
#   reports/activity2/final/calibration_curve.png       (one-vs-rest, crash)
#   reports/activity2/shap/shap_summary.png             (Tree SHAP or LR proxy)
#   reports/activity2/shap/shap_importance.csv
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os
import warnings

import joblib
import pandas as pd

from ..config import REPORT_FINAL_DIR, REPORT_MODELS_DIR, REPORT_SHAP_DIR
from ..evaluation.metrics import compute_all, confusion, per_class_pr
from ..evaluation.plots import (
    calibration_curve_plot, confusion_matrix_plot, per_class_metrics_bar,
)
from ..evaluation.shap_explainer import explain


warnings.filterwarnings("ignore")


def main():
    print("=" * 60)
    print("Activity 2 — Block D: final test-set evaluation")
    print("=" * 60)

    os.makedirs(REPORT_FINAL_DIR, exist_ok=True)
    os.makedirs(REPORT_SHAP_DIR, exist_ok=True)

    bundle = joblib.load(os.path.join(REPORT_MODELS_DIR, "winner_bundle.joblib"))
    model = bundle["winner_model"]
    X_te = bundle["X_test"]
    y_te = bundle["y_test"].astype(str).values

    print(f"  Winner model: {bundle['winner_name']}")
    print(f"  Imputation:   {bundle['winner_imp']}")
    print(f"  Imbalance:    {bundle['winner_imb']}")
    print(f"  Test size:    {len(y_te)} rows")

    # ── Metrics ─────────────────────────────────────────────────────────────
    pred = model.predict(X_te).astype(str)
    metrics = compute_all(y_te, pred)
    metrics["winner_model"] = bundle["winner_name"]
    metrics["winner_imp"]   = bundle["winner_imp"]
    metrics["winner_imb"]   = bundle["winner_imb"]

    print("\n  Test-set metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k:20s}  {v:.4f}")
        else:
            print(f"    {k:20s}  {v}")

    pd.DataFrame([metrics]).to_csv(os.path.join(REPORT_FINAL_DIR, "final_metrics.csv"), index=False)

    # ── Plots ───────────────────────────────────────────────────────────────
    cm = confusion(y_te, pred)
    pcpr = per_class_pr(y_te, pred)
    pcpr.to_csv(os.path.join(REPORT_FINAL_DIR, "per_class_metrics.csv"))

    confusion_matrix_plot(
        cm, f"Confusion matrix — {bundle['winner_name']}",
        os.path.join(REPORT_FINAL_DIR, "confusion_matrix.png"),
    )
    per_class_metrics_bar(
        pcpr, f"Per-class precision / recall / F1 — {bundle['winner_name']}",
        os.path.join(REPORT_FINAL_DIR, "per_class_metrics_bar.png"),
    )

    # Calibration on the rare class
    proba = model.predict_proba(X_te)
    crash_idx = list(model.classes_).index("crash")
    calibration_curve_plot(
        y_te, proba[:, crash_idx], "crash",
        os.path.join(REPORT_FINAL_DIR, "calibration_curve.png"),
    )

    # ── SHAP / explainability ───────────────────────────────────────────────
    explain(model, X_te, REPORT_SHAP_DIR)

    print("\n  All artifacts written under reports/activity2/final/ and shap/")


if __name__ == "__main__":
    main()
