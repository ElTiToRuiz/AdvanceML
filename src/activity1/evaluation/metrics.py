# metrics.py
# ─────────────────────────────────────────────────────────────────────────────
# Evaluation metrics shared across every Activity-1 forecasting model.
#
# Point metrics (all on SPY log-return):
#   - RMSE                 Root Mean Squared Error
#   - MAE                  Mean Absolute Error (robust to outliers)
#   - Directional Accuracy % of days where sign(pred) == sign(actual)
#
# Skill-above-drift metrics:
#   - up_rate              % of days with positive actual return.
#                          This is the accuracy of the trivial
#                          "always-up" predictor — the natural baseline
#                          on a market with positive long-run drift.
#   - trivial_acc          max(up_rate, 1 - up_rate) — accuracy of the
#                          best CONSTANT predictor (always-up or
#                          always-down, whichever is more frequent).
#   - skill_above_drift    dir_acc - trivial_acc.  Positive = the model
#                          adds genuine predictive skill above the
#                          trivial drift bet.  Near 0 = the apparent
#                          "accuracy" was just market drift, not skill.
#
# Confidence-aware metrics:
#   - Thresholded Dir. Acc.  only keep days where |pred| > threshold,
#                            reports (accuracy, coverage).
#   - Threshold curve        the same scan over many thresholds → smooth
#                            trade-off between coverage and accuracy.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import numpy as np
import pandas as pd


# ─── Point metrics ───────────────────────────────────────────────────────────

def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def directional_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Share of days where sign(pred) == sign(actual). Days with actual = 0
    are excluded (extremely rare on SPY log-returns).
    """
    mask = y_true != 0
    correct = np.sign(y_true[mask]) == np.sign(y_pred[mask])
    return float(correct.mean()) if mask.any() else float("nan")


def up_rate(y_true: pd.Series) -> float:
    """Share of days with positive actual return — accuracy of 'always up'."""
    mask = y_true.notna() & (y_true != 0)
    return float((y_true[mask] > 0).mean()) if mask.any() else float("nan")


def trivial_baseline_acc(y_true: pd.Series) -> float:
    """
    Accuracy of the best CONSTANT predictor: max(up_rate, 1 - up_rate).
    Equivalent to "always bet on the most frequent class".
    """
    u = up_rate(y_true)
    return max(u, 1.0 - u)


def skill_above_drift(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Directional accuracy MINUS the trivial-baseline accuracy.
    Positive ⇒ model adds predictive skill beyond the drift bet.
    Near 0   ⇒ what looked like "accuracy" was just market drift.
    """
    return directional_accuracy(y_true, y_pred) - trivial_baseline_acc(y_true)


# ─── Magnitude-aware metrics (quant finance lens) ─────────────────────────────

def magnitude_weighted_dir_acc(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Directional accuracy weighted by |actual return|.

    Big-move days dominate the score, tiny-move days are nearly free.
    A model that gets BIG moves right but misses small ones can score
    high here while looking mediocre on plain Dir.Acc.

    Equivalent to:
        Σ_t  1[sign(pred_t)==sign(y_t)] · |y_t|
        ─────────────────────────────────────
                        Σ_t  |y_t|
    """
    weights = y_true.abs()
    correct = (np.sign(y_true) == np.sign(y_pred)).astype(float)
    total   = float(weights.sum())
    return float((correct * weights).sum() / total) if total > 0 else float("nan")


def pearson_ic(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Information Coefficient (Pearson). Linear correlation between the
    model's predicted return and the actual return.

    Industry rules of thumb:
      |IC| < 0.02 → noise
      |IC| ≈ 0.05 → decent signal
      |IC| > 0.10 → exceptional signal
    """
    return float(y_true.corr(y_pred, method="pearson"))


def spearman_ic(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Rank-IC (Spearman). Robust to outliers — only the ORDER of the
    predictions matters, not the magnitudes. Often preferred when
    the model's output scale is not directly comparable to the target.
    """
    return float(y_true.corr(y_pred, method="spearman"))


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Drop rows where either series is NaN, then compute all metrics."""
    mask = y_true.notna() & y_pred.notna()
    y_t, y_p = y_true[mask], y_pred[mask]
    return {
        "n":           int(mask.sum()),
        "rmse":        rmse(y_t, y_p),
        "mae":         mae(y_t, y_p),
        "dir_acc":     directional_accuracy(y_t, y_p),
        "trivial_acc": trivial_baseline_acc(y_t),
        "skill":       skill_above_drift(y_t, y_p),
        "weighted_acc": magnitude_weighted_dir_acc(y_t, y_p),
        "ic_pearson":  pearson_ic(y_t, y_p),
        "ic_spearman": spearman_ic(y_t, y_p),
    }


def metrics_table(results: dict) -> pd.DataFrame:
    """Turn {model: {split: metrics}} into a flat (model, split) DataFrame."""
    rows = []
    for model, splits in results.items():
        for split, m in splits.items():
            rows.append({"model": model, "split": split, **m})
    return pd.DataFrame(rows).set_index(["model", "split"])


# ─── Confidence-aware metrics ────────────────────────────────────────────────

def thresholded_directional_accuracy(
    y_true: pd.Series,
    y_pred: pd.Series,
    threshold: float,
) -> tuple[float, float]:
    """
    Only keep days where |y_pred| > `threshold`, then compute directional
    accuracy on that subset.

    Returns
    -------
    accuracy : float  — share of kept days where sign matched (NaN if none)
    coverage : float  — share of all (non-NaN) days kept
    """
    mask = y_true.notna() & y_pred.notna()
    y_t, y_p = y_true[mask], y_pred[mask]

    confident = y_p.abs() > threshold
    if confident.sum() == 0:
        return float("nan"), 0.0

    correct = np.sign(y_t[confident]) == np.sign(y_p[confident])
    return float(correct.mean()), float(confident.mean())


def threshold_curve(
    y_true:   pd.Series,
    y_pred:   pd.Series,
    n_points: int = 20,
) -> pd.DataFrame:
    """
    Sweep thresholds at percentiles of |y_pred| and report
    (threshold, accuracy, coverage, n) at each step.

    The result is a smooth accuracy-vs-coverage curve:
    - leftmost point    → smallest threshold → 100% coverage, baseline acc
    - rightmost point   → largest threshold  → smallest coverage, hopefully
                                               higher accuracy
    """
    mask = y_true.notna() & y_pred.notna()
    if mask.sum() == 0:
        return pd.DataFrame(columns=["threshold", "accuracy", "coverage", "n"])

    abs_pred = y_pred[mask].abs()
    # zero threshold + percentile grid so the curve spans 100%→~5% coverage
    thresholds = np.concatenate(
        ([0.0], np.quantile(abs_pred, np.linspace(0.05, 0.95, n_points)))
    )

    rows = []
    for t in thresholds:
        acc, cov = thresholded_directional_accuracy(y_true, y_pred, t)
        n_kept  = int(((y_pred.abs() > t) & mask).sum())
        rows.append(
            {"threshold": float(t), "accuracy": acc, "coverage": cov, "n": n_kept}
        )
    return pd.DataFrame(rows)
