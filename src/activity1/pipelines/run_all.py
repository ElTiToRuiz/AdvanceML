# run_all.py
# ─────────────────────────────────────────────────────────────────────────────
# Unified Activity-1 comparison pipeline.
#
# Runs the full family of forecasters (Naive baselines + SARIMAX + LSTM +
# Chronos-2) and produces three layers of analysis:
#
#   1) Point metrics     (RMSE, MAE, Dir.Acc.) — basic ML evaluation
#   2) Threshold curves  (confidence-aware Dir.Acc. vs coverage)
#   3) P&L simulation    (long/short equity curves vs buy-and-hold SPY)
#
# All outputs go to reports/activity1/models/.
#
# Run from project root:
#     python -m src.activity1.pipelines.run_all
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os
import time
from typing import List

import pandas as pd

from ..config import FIGURES_DIR
from ..data.loader import load_splits
from ..evaluation.metrics     import compute_metrics, metrics_table, threshold_curve
from ..evaluation.plots       import (
    plot_actual_vs_predicted, plot_threshold_curves, plot_ic_bars,
)
from ..evaluation.backtesting import (
    simulate_pnl, buy_and_hold_equity, pnl_summary_table, plot_equity_curves,
)
from ..models.base     import Forecaster
from ..models.baselines import all_baselines
from ..models.sarimax   import SarimaxForecaster
from ..models.lstm      import LSTMForecaster
from ..models.chronos   import Chronos2Forecaster


MODELS_DIR = os.path.join(FIGURES_DIR, "models")


# ─── Model set ───────────────────────────────────────────────────────────────

def build_models() -> List[Forecaster]:
    """
    One representative configuration per model family.
    Drift/MA baselines are the cheap suelo; SARIMAX / LSTM / Chronos-2 use
    the same exogenous signal (TNX_chg) or omit it where that previously
    proved better on test.
    """
    return [
        *all_baselines(),
        SarimaxForecaster(order=(1, 0, 1), exog_cols=("TNX_chg",)),
        LSTMForecaster(features=("SPY_ret",)),
        Chronos2Forecaster(past_covariates=("TNX_chg",)),
    ]


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Activity 1 — Unified forecasting comparison")
    print("=" * 60)

    splits = load_splits()
    train, val, test = splits.train, splits.val, splits.test
    print(f"\n  Loaded splits:  train={len(train)}  val={len(val)}  test={len(test)}")

    models = build_models()

    val_preds:  dict = {}
    test_preds: dict = {}
    point_results:    dict = {}

    # ── Fit + predict each model ──────────────────────────────────────────
    # Order is important:
    #   1. fit(train)
    #   2. predict(val)            ← val forecast made with train history
    #   3. observe(val)            ← model now knows val happened
    #   4. predict(test)           ← test forecast uses train + val as past
    # This matches realistic deployment: by the time test starts, val has
    # already been observed.
    for model in models:
        print(f"\n  ── {model.name} ──")
        t0 = time.time()
        model.fit(train)
        t_fit = time.time() - t0

        t0 = time.time()
        val_pred  = model.predict(val)
        t_val = time.time() - t0

        model.observe(val)

        t0 = time.time()
        test_pred = model.predict(test)
        t_test = time.time() - t0

        val_preds[model.name]  = val_pred
        test_preds[model.name] = test_pred

        point_results[model.name] = {
            "val":  compute_metrics(val["target"],  val_pred),
            "test": compute_metrics(test["target"], test_pred),
        }
        print(f"    fit {t_fit:5.1f}s | val {t_val:5.1f}s | test {t_test:5.1f}s")

    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Layer 1: point metrics ────────────────────────────────────────────
    table = metrics_table(point_results)
    csv_path = os.path.join(MODELS_DIR, "all_metrics.csv")
    table.to_csv(csv_path)

    print("\n  Point metrics:")
    print(table.round(6).to_string())
    print(f"\n  Saved: {csv_path}")

    plot_actual_vs_predicted(
        y_true=val["target"],  preds=val_preds,
        title="Activity 1 — All models on VALIDATION (SPY next-day log-return)",
        out_path=os.path.join(MODELS_DIR, "all_val.png"),
    )
    plot_actual_vs_predicted(
        y_true=test["target"], preds=test_preds,
        title="Activity 1 — All models on TEST (SPY next-day log-return)",
        out_path=os.path.join(MODELS_DIR, "all_test.png"),
    )

    # ── Layer 2: threshold curves ─────────────────────────────────────────
    print("\n  Computing threshold curves ...")
    val_curves  = {name: threshold_curve(val["target"],  p) for name, p in val_preds.items()}
    test_curves = {name: threshold_curve(test["target"], p) for name, p in test_preds.items()}

    # Save curves in long format for the report
    def _flatten(curves: dict, split: str) -> pd.DataFrame:
        frames = []
        for name, df in curves.items():
            if df.empty:
                continue
            d = df.copy()
            d["model"] = name
            d["split"] = split
            frames.append(d)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    curves_df = pd.concat(
        [_flatten(val_curves, "val"), _flatten(test_curves, "test")],
        ignore_index=True,
    )
    curves_path = os.path.join(MODELS_DIR, "threshold_curves.csv")
    curves_df.to_csv(curves_path, index=False)
    print(f"  Saved: {curves_path}")

    plot_threshold_curves(
        curves=val_curves,
        title="Activity 1 — Accuracy vs Coverage on VALIDATION",
        out_path=os.path.join(MODELS_DIR, "threshold_val.png"),
    )
    plot_threshold_curves(
        curves=test_curves,
        title="Activity 1 — Accuracy vs Coverage on TEST",
        out_path=os.path.join(MODELS_DIR, "threshold_test.png"),
    )

    # ── IC bar charts (Pearson + Spearman) ────────────────────────────────
    plot_ic_bars(
        metrics_split=table.xs("val", level="split"),
        title="Activity 1 — Information Coefficient on VALIDATION",
        out_path=os.path.join(MODELS_DIR, "ic_val.png"),
    )
    plot_ic_bars(
        metrics_split=table.xs("test", level="split"),
        title="Activity 1 — Information Coefficient on TEST",
        out_path=os.path.join(MODELS_DIR, "ic_test.png"),
    )

    # ── Layer 3: P&L backtest ─────────────────────────────────────────────
    print("\n  Running P&L backtest (threshold=0, always long/short) ...")
    pnl_results: dict = {}
    val_equity_curves:  dict = {}
    test_equity_curves: dict = {}

    for name in val_preds:
        pnl_results[name] = {
            "val":  simulate_pnl(val["target"],  val_preds[name],  threshold=0.0),
            "test": simulate_pnl(test["target"], test_preds[name], threshold=0.0),
        }
        val_equity_curves[name]  = pnl_results[name]["val"]["equity_curve"]
        test_equity_curves[name] = pnl_results[name]["test"]["equity_curve"]

    pnl_table = pnl_summary_table(pnl_results)
    pnl_path  = os.path.join(MODELS_DIR, "pnl_summary.csv")
    pnl_table.to_csv(pnl_path)

    print("\n  P&L summary:")
    print(pnl_table.round(3).to_string())
    print(f"\n  Saved: {pnl_path}")

    plot_equity_curves(
        curves=val_equity_curves,
        buy_and_hold=buy_and_hold_equity(val["target"]),
        title="Activity 1 — Equity curves on VALIDATION (long/short, threshold=0)",
        out_path=os.path.join(MODELS_DIR, "pnl_val.png"),
    )
    plot_equity_curves(
        curves=test_equity_curves,
        buy_and_hold=buy_and_hold_equity(test["target"]),
        title="Activity 1 — Equity curves on TEST (long/short, threshold=0)",
        out_path=os.path.join(MODELS_DIR, "pnl_test.png"),
    )

    # ── Leaderboard 1: SKILL above the trivial drift bet ──────────────────
    test_board = (
        table.xs("test", level="split")
             .sort_values("skill", ascending=False)
             .round(6)
    )
    print("\n  Leaderboard — test, sorted by SKILL (dir_acc - trivial):")
    print("  (skill ≈ 0 means the model just rides the drift)")
    print(test_board[["dir_acc", "trivial_acc", "skill"]].to_string())

    # ── Leaderboard 2: MAGNITUDE-AWARE skill (the days that matter) ───────
    mag_board = (
        table.xs("test", level="split")
             .sort_values("ic_spearman", ascending=False)
             .round(4)
    )
    print("\n  Leaderboard — test, sorted by IC (information coefficient):")
    print("  (weighted_acc weights days by |actual return|; ic > 0 = real signal)")
    print(mag_board[["weighted_acc", "ic_pearson", "ic_spearman"]].to_string())

    # ── Leaderboard 3: P&L (Sharpe) ───────────────────────────────────────
    pnl_board = (
        pnl_table.xs("test", level="split")
                 .sort_values("sharpe", ascending=False)
                 .round(3)
    )
    print("\n  Leaderboard — test, sorted by Sharpe:")
    print(pnl_board[["ann_return_pct", "ann_vol_pct", "sharpe",
                     "max_drawdown_pct", "final_equity"]].to_string())

    print("\n  Done.")


if __name__ == "__main__":
    main()
