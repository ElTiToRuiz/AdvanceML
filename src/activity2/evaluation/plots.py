# plots.py
# ─────────────────────────────────────────────────────────────────────────────
# Reusable plots for Block A / B / C / D outputs.
#
#   confusion_matrix_plot(cm, title)      → 4×4 heatmap with cell counts
#   per_class_metrics_bar(per_class)      → grouped bar chart of P / R / F1
#   calibration_curve_plot(...)           → reliability curve for the rare class
#   bar_macro_f1(df, by, title)           → horizontal bars used in Block A / B
#
# All plots use the shared style and the regime / model colour maps.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import (
    INK, REGIME_COLORS, apply_style, model_color, regime_color,
)
from ..config import REGIME_ORDER


apply_style()


def confusion_matrix_plot(cm: pd.DataFrame, title: str, save_to: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm.values, cmap="Blues")
    ax.set_xticks(range(len(REGIME_ORDER)))
    ax.set_yticks(range(len(REGIME_ORDER)))
    ax.set_xticklabels(REGIME_ORDER, rotation=20, ha="right")
    ax.set_yticklabels(REGIME_ORDER)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    vmax = cm.values.max()
    for i in range(len(REGIME_ORDER)):
        for j in range(len(REGIME_ORDER)):
            v = int(cm.values[i, j])
            ax.text(
                j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > vmax / 2 else INK, fontsize=10,
            )
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_to)
    plt.close(fig)


def per_class_metrics_bar(per_class: pd.DataFrame, title: str, save_to: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    metrics = ["precision", "recall", "f1"]
    x = np.arange(len(REGIME_ORDER))
    width = 0.27
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 1) * width, per_class[m].values, width=width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(REGIME_ORDER)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_to)
    plt.close(fig)


def bar_macro_f1(df: pd.DataFrame, by: str, score_col: str, title: str, save_to: str) -> None:
    """
    Horizontal bar chart of `score_col` indexed by `by` (e.g. 'method').

    Used by Block A (imputation comparison) and Block B (imbalance comparison).
    """
    df_sorted = df.sort_values(score_col)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(df_sorted))))
    ax.barh(
        df_sorted[by],
        df_sorted[score_col],
        color=[model_color(m) for m in df_sorted[by]],
    )
    ax.set_xlabel(score_col)
    ax.set_title(title)
    ax.set_xlim(0, max(0.6, df_sorted[score_col].max() * 1.1))
    for i, v in enumerate(df_sorted[score_col].values):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_to)
    plt.close(fig)


def calibration_curve_plot(
    y_true: np.ndarray,
    proba_class: np.ndarray,
    class_label: str,
    save_to: str,
    n_bins: int = 10,
) -> None:
    """One-vs-rest calibration curve for a single class (defaults to 'crash')."""
    y_bin = (y_true == class_label).astype(int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(proba_class, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        rows.append({
            "p_mean":   float(proba_class[mask].mean()),
            "freq":     float(y_bin[mask].mean()),
            "n":        int(mask.sum()),
        })
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfectly calibrated")
    ax.plot(df["p_mean"], df["freq"], "-o", color=regime_color(class_label), label=f"{class_label}")
    ax.set_xlabel(f"Mean predicted P({class_label})")
    ax.set_ylabel("Empirical frequency")
    ax.set_title(f"Calibration curve — {class_label}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_to)
    plt.close(fig)
