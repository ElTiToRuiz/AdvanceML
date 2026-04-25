# plots.py
# ─────────────────────────────────────────────────────────────────────────────
# Evaluation plots shared across every Activity-1 forecasting model.
# Uses the project-wide style from .style for visual consistency with the
# EDA charts and the backtesting equity curves.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt

from .style import apply_style, model_color, INK, GRAY, LGRAY

apply_style()


# ─── Actual vs predicted ─────────────────────────────────────────────────────

def plot_actual_vs_predicted(
    y_true: pd.Series,
    preds: dict,
    title: str,
    out_path: str,
) -> None:
    """One figure with actual returns + every model's predictions overlaid."""
    fig, ax = plt.subplots(figsize=(13, 5.5))

    ax.plot(
        y_true.index, y_true.values * 100,
        label="actual", color=INK, linewidth=1.0, alpha=0.55,
    )
    for name, y_pred in preds.items():
        ax.plot(
            y_pred.index, y_pred.values * 100,
            label=name, color=model_color(name),
            alpha=0.85, linewidth=1.0,
        )

    ax.axhline(0, color=GRAY, linewidth=0.6)
    ax.set_title(title)
    ax.set_ylabel("SPY log-return (%)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper right", ncol=2, fontsize=8)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── Information Coefficient (IC) bar chart ──────────────────────────────────

def plot_ic_bars(
    metrics_split: pd.DataFrame,
    title:    str,
    out_path: str,
) -> None:
    """
    Side-by-side bar chart of Pearson IC and Spearman IC per model.
    Industry reference lines at IC = 0.05 ("decent") and IC = 0.10
    ("exceptional") help the reader interpret the bars.
    """
    df = metrics_split.copy()
    if "model" in df.columns:
        df = df.set_index("model")

    df = df.sort_values("ic_pearson", ascending=True)
    models = df.index.tolist()
    n      = len(models)

    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.55 * n)))

    y = list(range(n))
    bar_h = 0.36

    pearson_colors  = [model_color(m) for m in models]
    spearman_colors = pearson_colors

    bars_p = ax.barh(
        [yi + bar_h / 2 for yi in y], df["ic_pearson"],
        height=bar_h, color=pearson_colors, alpha=0.95, edgecolor="white",
        label="Pearson IC",
    )
    bars_s = ax.barh(
        [yi - bar_h / 2 for yi in y], df["ic_spearman"],
        height=bar_h, color=spearman_colors, alpha=0.45, edgecolor="white",
        label="Spearman IC (rank)",
    )

    # Industry reference lines
    for x, label in [(0.0, "noise"), (0.05, "decent"), (0.10, "exceptional")]:
        ax.axvline(x, color=GRAY, linestyle="--", alpha=0.5, linewidth=0.8)
        ax.text(
            x, n - 0.4, f"{label}\n({x:+.2f})",
            ha="center", va="bottom", fontsize=8, color=GRAY,
        )
    for x in [-0.05, -0.10]:
        ax.axvline(x, color=GRAY, linestyle="--", alpha=0.4, linewidth=0.7)

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=9)
    ax.set_xlabel("Information Coefficient (correlation with actual returns)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(min(-0.12, df["ic_pearson"].min() * 1.1, df["ic_spearman"].min() * 1.1),
                max( 0.15, df["ic_pearson"].max() * 1.1, df["ic_spearman"].max() * 1.1))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── Threshold curves ────────────────────────────────────────────────────────

def plot_threshold_curves(
    curves:   dict,
    title:    str,
    out_path: str,
) -> None:
    """Accuracy-vs-coverage curves for several models."""
    fig, ax = plt.subplots(figsize=(11, 6))

    for name, df in curves.items():
        if df.empty:
            continue
        df = df.sort_values("coverage")
        ax.plot(
            df["coverage"] * 100,
            df["accuracy"] * 100,
            marker="o", markersize=4,
            color=model_color(name),
            alpha=0.9, linewidth=1.4,
            label=name,
        )

    ax.axhline(50, color=GRAY, linestyle="--", alpha=0.7,
               label="random (50%)")

    ax.set_xlabel("Coverage (% of days kept · |pred| > threshold)")
    ax.set_ylabel("Directional Accuracy (%)")
    ax.set_title(title)
    ax.legend(loc="best", ncol=2, fontsize=8)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")
