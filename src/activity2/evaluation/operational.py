# operational.py
# ─────────────────────────────────────────────────────────────────────────────
# Block E helpers: probability calibration, sample-weight construction,
# threshold search under a precision floor, PR-curve overlay, and the
# long-only "exit on signal" backtest.
#
# Pure functions only — no file I/O. The pipeline owns persistence.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import precision_recall_curve

from .style import AMBER, BLUE, GREEN, PURPLE, REGIME_COLORS, apply_style, regime_color


apply_style()


# ─── Sample weights ──────────────────────────────────────────────────────────


def asymmetric_sample_weights(y: pd.Series, weight_map: dict[str, float]) -> np.ndarray:
    """
    Build a per-row sample_weight array from a {class_name: weight} map.

    The Phase-2 path used `class_weight='balanced'` which weights each
    class by inverse frequency. Block E uses an explicit cost-asymmetric
    map: missing a crash is much more expensive than missing a rally.
    """
    return np.array([float(weight_map[str(c)]) for c in y], dtype=float)


# ─── Calibration ─────────────────────────────────────────────────────────────


class CalibratedClassifierWrapper:
    """
    Thin wrapper that applies CalibratedClassifierCV with `cv='prefit'`
    on top of an already-fitted Classifier. Exposes the same .predict /
    .predict_proba / .classes_ surface so it slots into our pipelines.

    The `method='sigmoid'` choice = Platt scaling. Faster than isotonic
    on small val sets and parametric (less prone to overfit val noise).
    """

    def __init__(self, base_classifier, X_val: pd.DataFrame, y_val: pd.Series):
        self.name = f"{base_classifier.name}_calibrated"
        self.base = base_classifier
        # CalibratedClassifierCV needs an sklearn-style estimator. We
        # pass the underlying booster wrapped in FrozenEstimator (sklearn
        # 1.8 replacement for the deprecated cv='prefit' option), so the
        # calibrator only fits the Platt-scaling layer on top.
        underlying = getattr(base_classifier, "booster", base_classifier._model)
        # XGBoostModel uses LabelEncoder; we need the integer-encoded y
        # for that branch but original strings for sklearn estimators.
        if hasattr(base_classifier, "_encoder"):
            y_for_cal = base_classifier._encoder.transform(y_val)
        else:
            y_for_cal = y_val.values if hasattr(y_val, "values") else y_val
        # sklearn 1.8 removed cv='prefit'. The replacement pattern is to wrap
        # the already-fitted estimator in FrozenEstimator, which tells the
        # calibrator to skip refitting on the calibration set.
        self._calibrator = CalibratedClassifierCV(
            estimator=FrozenEstimator(underlying), method="sigmoid",
        ).fit(X_val.values, y_for_cal)
        # Recover string class labels (CalibratedClassifierCV.classes_ is ints
        # when the base estimator was trained on encoded labels).
        if hasattr(base_classifier, "_encoder"):
            self.classes_ = base_classifier._encoder.inverse_transform(
                self._calibrator.classes_
            )
        else:
            self.classes_ = self._calibrator.classes_

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.array(self.classes_)[np.argmax(proba, axis=1)]

    def predict_proba(self, X) -> np.ndarray:
        arr = X.values if hasattr(X, "values") else X
        return self._calibrator.predict_proba(arr)


def reliability_diagram(
    y_true: np.ndarray,
    proba_class: np.ndarray,
    class_label: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Bin proba_class into n_bins, return per-bin (predicted mean, empirical
    frequency, count). Caller decides how to plot.
    """
    y_bin = (np.asarray(y_true) == class_label).astype(int)
    proba_class = np.asarray(proba_class)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(proba_class, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        rows.append({
            "p_mean": float(proba_class[mask].mean()),
            "freq":   float(y_bin[mask].mean()),
            "n":      int(mask.sum()),
        })
    return pd.DataFrame(rows)


def plot_reliability(
    rel_df: pd.DataFrame,
    title: str,
    class_label: str,
    save_to: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfectly calibrated")
    ax.plot(rel_df["p_mean"], rel_df["freq"], "-o",
            color=regime_color(class_label), label=class_label)
    ax.set_xlabel(f"Mean predicted P({class_label})")
    ax.set_ylabel("Empirical frequency")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_to)
    plt.close(fig)


# ─── Threshold search ────────────────────────────────────────────────────────


@dataclass
class OperatingPoint:
    """The chosen decision rule for one detector at a fixed precision floor."""
    detector:        str
    threshold:       float
    recall:          float
    precision:       float
    n_alarms:        int


def find_operating_point(
    y_val: np.ndarray,
    proba_class_val: np.ndarray,
    target_class: str,
    precision_floor: float,
    detector_name: str = "?",
) -> OperatingPoint:
    """
    Pick the threshold on `proba_class_val` that maximises
    target-class recall on val SUBJECT TO precision >= precision_floor.

    If no threshold satisfies the precision floor, fall back to the
    threshold with the HIGHEST achievable precision (the one closest to
    the floor we wanted). The returned `OperatingPoint.recall` will be
    whatever recall that fallback threshold produces — it is NOT
    guaranteed to be zero.
    """
    y_true_bin = (np.asarray(y_val) == target_class).astype(int)
    precision, recall, thresholds = precision_recall_curve(y_true_bin, proba_class_val)
    # precision_recall_curve drops the trailing endpoint of `thresholds`, so
    # zip with the matching slices.
    candidates = []
    for t, p, r in zip(thresholds, precision[:-1], recall[:-1]):
        if p >= precision_floor:
            candidates.append((float(t), float(r), float(p)))
    if candidates:
        threshold, rec, prec = max(candidates, key=lambda x: x[1])
    else:
        # Fallback: argmax precision (closest to floor we can get)
        idx = int(np.argmax(precision[:-1]))
        threshold = float(thresholds[idx])
        rec = float(recall[idx])
        prec = float(precision[idx])
    n_alarms = int((proba_class_val >= threshold).sum())
    return OperatingPoint(
        detector=detector_name,
        threshold=threshold,
        recall=rec,
        precision=prec,
        n_alarms=n_alarms,
    )


def evaluate_at_threshold(
    y_true: np.ndarray,
    proba_class: np.ndarray,
    threshold: float,
    target_class: str,
) -> dict[str, float]:
    """Return recall / precision / n_alarms for the target class on a held-out split."""
    y_bin = (np.asarray(y_true) == target_class).astype(int)
    pred_bin = (np.asarray(proba_class) >= threshold).astype(int)
    tp = int(((pred_bin == 1) & (y_bin == 1)).sum())
    fp = int(((pred_bin == 1) & (y_bin == 0)).sum())
    fn = int(((pred_bin == 0) & (y_bin == 1)).sum())
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return {
        "recall":    float(recall),
        "precision": float(precision),
        "n_alarms":  int((pred_bin == 1).sum()),
        "tp": tp, "fp": fp, "fn": fn,
    }


# ─── PR curve overlay ───────────────────────────────────────────────────────


def plot_pr_overlay(
    series: list[tuple[str, np.ndarray, np.ndarray]],   # (name, y_true_bin, proba)
    operating_points: list[OperatingPoint],
    target_class: str,
    save_to: str,
) -> None:
    """
    Overlay the PR curve of every detector on the same axes, and mark
    each detector's chosen operating point with a star.
    """
    palette = [BLUE, AMBER, GREEN, PURPLE][: len(series)]
    fig, ax = plt.subplots(figsize=(8, 6))
    for (name, y_bin, proba), color, op in zip(series, palette, operating_points):
        precision, recall, _ = precision_recall_curve(y_bin, proba)
        ax.plot(recall, precision, "-", color=color, linewidth=1.4, label=name)
        ax.plot(op.recall, op.precision, marker="*", color=color, markersize=14,
                markeredgecolor="black", markeredgewidth=0.6)
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_xlabel(f"Recall on {target_class}")
    ax.set_ylabel(f"Precision on {target_class}")
    ax.set_title(f"Precision-Recall on the {target_class} class — test set")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_to)
    plt.close(fig)


# ─── Backtest ───────────────────────────────────────────────────────────────


def simulate_exit_strategy(
    spy_log_returns: pd.Series,        # daily log-returns of SPY, indexed by date
    crash_signal: pd.Series,           # 1 if model fires crash for THAT day, 0 otherwise
) -> pd.DataFrame:
    """
    Long-only, daily decisions:
      - if crash_signal[t] == 1 → hold cash on day t (return = 0)
      - else                    → hold SPY on day t (return = spy_log_returns[t])

    Both series must be aligned on the same date index.

    Returns a DataFrame with columns: spy_ret, signal, strat_ret,
    cumret_strat, cumret_buyhold.
    """
    aligned = pd.concat(
        [spy_log_returns.rename("spy_ret"), crash_signal.rename("signal")],
        axis=1, join="inner",
    ).dropna()
    aligned["strat_ret"] = np.where(aligned["signal"] == 1, 0.0, aligned["spy_ret"])
    aligned["cumret_strat"]   = np.exp(aligned["strat_ret"].cumsum())
    aligned["cumret_buyhold"] = np.exp(aligned["spy_ret"].cumsum())
    return aligned


def backtest_summary(
    bt: pd.DataFrame,
    y_true_class: pd.Series,            # categorical regime labels, same index as bt
    target_class: str = "crash",
) -> dict[str, float]:
    """
    One-call summary of a simulated strategy.

    Sharpe is annualised assuming 252 trading days/year.
    Drawdown is on the strategy log-cumulative path.
    """
    n = len(bt)
    if n == 0:
        return {}
    annual_factor = np.sqrt(252.0)
    s_strat = bt["strat_ret"]
    s_bh    = bt["spy_ret"]

    # Note: cash-holding days are zero-return days that reduce the std
    # of strat_ret, which can mechanically inflate `sharpe_strat`. This
    # is the correct measure for a risk-off strategy but should be
    # interpreted alongside max_drawdown, not in isolation.
    sharpe_strat = float(s_strat.mean() / s_strat.std() * annual_factor) if s_strat.std() > 0 else 0.0
    sharpe_bh    = float(s_bh.mean()    / s_bh.std()    * annual_factor) if s_bh.std()    > 0 else 0.0

    cum = bt["cumret_strat"]
    drawdown = float((cum / cum.cummax() - 1.0).min())
    drawdown_bh = float((bt["cumret_buyhold"] / bt["cumret_buyhold"].cummax() - 1.0).min())

    crash_days  = (y_true_class.reindex(bt.index) == target_class).astype(int)
    n_crashes_total  = int(crash_days.sum())
    n_crashes_evaded = int(((crash_days == 1) & (bt["signal"] == 1)).sum())
    n_crashes_missed = int(((crash_days == 1) & (bt["signal"] == 0)).sum())
    n_false_alarms   = int(((crash_days == 0) & (bt["signal"] == 1)).sum())

    return {
        "n_days":              n,
        "final_eq_strat":      float(bt["cumret_strat"].iloc[-1]),
        "final_eq_buyhold":    float(bt["cumret_buyhold"].iloc[-1]),
        "ann_return_strat":    float(np.exp(s_strat.mean() * 252.0) - 1.0),
        "ann_return_buyhold":  float(np.exp(s_bh.mean() * 252.0) - 1.0),
        "sharpe_strat":        sharpe_strat,
        "sharpe_buyhold":      sharpe_bh,
        "max_drawdown_strat":  drawdown,
        "max_drawdown_buyhold": drawdown_bh,
        "n_signals":           int(bt["signal"].sum()),
        "n_crashes_total":     n_crashes_total,
        "n_crashes_evaded":    n_crashes_evaded,
        "n_crashes_missed":    n_crashes_missed,
        "n_false_alarms":      n_false_alarms,
    }


def plot_equity_curve(bt: pd.DataFrame, save_to: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(bt.index, bt["cumret_buyhold"], color="#64748B",
            linewidth=1.4, label="buy & hold")
    ax.plot(bt.index, bt["cumret_strat"], color=REGIME_COLORS["crash"],
            linewidth=1.6, label="exit on crash signal")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return ($1 → ?)")
    ax.set_title("Backtest equity curve — test window")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(save_to)
    plt.close(fig)
