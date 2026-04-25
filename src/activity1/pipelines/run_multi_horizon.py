# run_multi_horizon.py
# ─────────────────────────────────────────────────────────────────────────────
# Multi-horizon forecasting analysis for Activity 1.
#
# Compares the same models at three forecast horizons:
#
#     h = 1  day   — tomorrow's SPY log-return
#     h = 5  days  — cumulative log-return over the next trading week
#     h = 30 days  — cumulative log-return over the next ~6 weeks
#
# The hypothesis we want to test: "longer horizons are more tractable
# because short-term noise averages out".  If Dir.Acc. rises monotonically
# with h while models stay the same, the claim is supported.
#
# Caveats:
#   - Cumulative windows OVERLAP (consecutive targets share h-1 days),
#     so at h=30 consecutive samples are not independent. Metrics are
#     still informative in relative terms across models, but absolute
#     numbers should be read with care.
#   - Chronos-2 is NOT included here — it would need prediction_length=h,
#     which requires a separate code path. Left out for simplicity; the
#     comparison is already meaningful with baselines + LSTM.
#
# Outputs:
#   reports/activity1/models/multi_horizon_metrics.csv
#   reports/activity1/models/multi_horizon_dir_acc.png
#
# Run from project root:
#     python -m src.activity1.pipelines.run_multi_horizon
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..config import FIGURES_DIR
from ..data.loader import load_splits
from ..evaluation.metrics import compute_metrics
from ..evaluation.style   import apply_style, model_color, GRAY
from ..models.baselines import all_baselines
from ..models.lstm import LSTMForecaster

apply_style()


MODELS_DIR = os.path.join(FIGURES_DIR, "models")
HORIZONS   = [1, 5, 30]


# ─── Build h-day horizon splits ──────────────────────────────────────────────

def _rebuild_for_horizon(train, val, test, h: int):
    """
    Replace SPY_ret with its h-day cumulative log-return (rolling sum of h)
    and target with the NEXT h-day cumulative (target shifted by -h).
    Everything else in the DataFrame stays the same.
    """
    full_ret = pd.concat([train["SPY_ret"], val["SPY_ret"], test["SPY_ret"]])
    full_ret = full_ret[~full_ret.index.duplicated(keep="first")].sort_index()

    hday        = full_ret.rolling(h).sum()           # past h-day cumulative
    hday_target = full_ret.rolling(h).sum().shift(-h) # next h-day cumulative

    def rebuild(df):
        out = df.copy()
        out["SPY_ret"] = hday.loc[df.index]
        out["target"]  = hday_target.loc[df.index]
        return out

    return rebuild(train), rebuild(val), rebuild(test)


def _build_models():
    """Fresh model instances for every horizon (so fit state does not leak)."""
    return [
        *all_baselines(),
        LSTMForecaster(features=("SPY_ret",)),
    ]


# ─── Plot ────────────────────────────────────────────────────────────────────

def _plot_dir_acc_vs_horizon(df: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, split in zip(axes, ["val", "test"]):
        sub = df[df["split"] == split]
        for model in sub["model"].unique():
            rows = sub[sub["model"] == model].sort_values("horizon")
            ax.plot(
                rows["horizon"], rows["dir_acc"] * 100,
                marker="o", markersize=8, linewidth=1.8,
                color=model_color(model), label=model, alpha=0.9,
            )
        ax.axhline(50, color=GRAY, linestyle="--", alpha=0.7, label="random (50%)")
        ax.set_xlabel("Forecast horizon (trading days)")
        ax.set_title(f"Directional Accuracy — {split.upper()}")
        ax.set_xticks(HORIZONS)
    axes[0].set_ylabel("Directional Accuracy (%)")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "Activity 1 — Multi-horizon forecasting · longer horizon ⇒ less noise",
        fontsize=14, fontweight="bold",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Activity 1 — Multi-horizon forecasting")
    print("=" * 60)

    splits = load_splits()
    train, val, test = splits.train, splits.val, splits.test
    print(f"\n  Loaded splits:  train={len(train)}  val={len(val)}  test={len(test)}")

    rows: list[dict] = []

    for h in HORIZONS:
        print(f"\n  ── Horizon h = {h} ──")
        trn_h, val_h, test_h = _rebuild_for_horizon(train, val, test, h)

        # Drop train rows whose SPY_ret is NaN (first h-1 rows) so models
        # receive a clean history.
        trn_h = trn_h.dropna(subset=["SPY_ret", "target"])

        models = _build_models()
        for model in models:
            print(f"    fitting {model.name} ...")
            model.fit(trn_h)
            val_pred  = model.predict(val_h)
            model.observe(val_h)        # extend history before test
            test_pred = model.predict(test_h)

            val_m  = compute_metrics(val_h["target"],  val_pred)
            test_m = compute_metrics(test_h["target"], test_pred)

            rows.append({"horizon": h, "model": model.name, "split": "val",  **val_m})
            rows.append({"horizon": h, "model": model.name, "split": "test", **test_m})

            print(
                f"      val_dir={val_m['dir_acc']:.3f}  "
                f"test_dir={test_m['dir_acc']:.3f}  "
                f"val_rmse={val_m['rmse']:.5f}  test_rmse={test_m['rmse']:.5f}"
            )

    df = pd.DataFrame(rows)
    os.makedirs(MODELS_DIR, exist_ok=True)
    csv_path = os.path.join(MODELS_DIR, "multi_horizon_metrics.csv")
    df.to_csv(csv_path, index=False)

    print("\n  Results (sorted by horizon, split, model):")
    pivot = df.pivot_table(
        index=["split", "model"], columns="horizon",
        values="dir_acc"
    ).round(4)
    print(pivot.to_string())
    print(f"\n  Saved: {csv_path}")

    _plot_dir_acc_vs_horizon(
        df,
        out_path=os.path.join(MODELS_DIR, "multi_horizon_dir_acc.png"),
    )


if __name__ == "__main__":
    main()
