# backtesting.py
# ─────────────────────────────────────────────────────────────────────────────
# Simple long/short backtester used to translate model predictions into a
# P&L curve — "¿cuánto palmas/ganas siguiendo el modelo?".
#
# Strategy for every day t:
#
#     pos[t] = +1   if pred[t] >  +threshold   (long next day)
#              -1   if pred[t] <  -threshold   (short next day)
#               0   otherwise                   (flat, out of the market)
#
# Realised strategy log-return on day t+1:
#
#     strat_ret[t] = pos[t] · actual_ret[t+1]
#                  = pos[t] · y_true[t]
#     (recall our convention:  y_true[t] = SPY_ret[t+1])
#
# Equity assumes continuous compounding of log-returns, so the curve
# grows as exp(cumsum(strat_ret)).
#
# This is a deliberately NAIVE backtest: no transaction costs, no slippage,
# no risk sizing, no short-borrow costs. It is a diagnostic for comparing
# ML models — not a real trading simulation.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .style import apply_style, model_color, INK, GRAY

apply_style()


TRADING_DAYS_PER_YEAR = 252


# ─── Core simulation ─────────────────────────────────────────────────────────

def simulate_pnl(
    y_true:    pd.Series,
    y_pred:    pd.Series,
    threshold: float = 0.0,
) -> dict:
    """
    Run the long/short strategy described above and return a dict with
    summary metrics plus the full equity curve (pd.Series).
    """
    mask = y_true.notna() & y_pred.notna()
    y_t = y_true[mask].astype(float)
    y_p = y_pred[mask].astype(float)

    pos = pd.Series(0.0, index=y_t.index)
    pos[y_p > +threshold] = +1.0
    pos[y_p < -threshold] = -1.0

    strat_ret = pos * y_t
    n = len(strat_ret)

    if n == 0:
        return {
            "n_days":           0,
            "active_pct":       0.0,
            "hit_rate":         float("nan"),
            "total_return_pct": 0.0,
            "ann_return_pct":   0.0,
            "ann_vol_pct":      0.0,
            "sharpe":           0.0,
            "max_drawdown_pct": 0.0,
            "final_equity":     1.0,
            "equity_curve":     pd.Series(dtype=float),
        }

    equity = np.exp(strat_ret.cumsum())

    active_mask = pos != 0
    n_active    = int(active_mask.sum())
    hit_rate    = (
        float((strat_ret[active_mask] > 0).sum() / n_active)
        if n_active > 0 else float("nan")
    )

    ann_factor = TRADING_DAYS_PER_YEAR / n
    std        = float(strat_ret.std())

    return {
        "n_days":           n,
        "active_pct":       float(n_active / n * 100),
        "hit_rate":         hit_rate,
        "total_return_pct": float((equity.iloc[-1] - 1) * 100),
        "ann_return_pct":   float((equity.iloc[-1] ** ann_factor - 1) * 100),
        "ann_vol_pct":      float(std * np.sqrt(TRADING_DAYS_PER_YEAR) * 100),
        "sharpe":           float(strat_ret.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR))
                            if std > 0 else 0.0,
        "max_drawdown_pct": float(((equity / equity.cummax()) - 1).min() * 100),
        "final_equity":     float(equity.iloc[-1]),
        "equity_curve":     equity,
    }


def buy_and_hold_equity(y_true: pd.Series) -> pd.Series:
    """100%-long SPY benchmark, equity curve (start = 1)."""
    mask = y_true.notna()
    return np.exp(y_true[mask].cumsum())


def pnl_summary_table(results: dict) -> pd.DataFrame:
    """
    `results` is {model: {split: simulate_pnl_result}}.
    Returns a flat DataFrame indexed by (model, split) with all numeric
    fields (drops the equity_curve series).
    """
    rows = []
    for model, splits in results.items():
        for split, r in splits.items():
            rows.append({
                "model":            model,
                "split":            split,
                "n_days":           r["n_days"],
                "active_pct":       r["active_pct"],
                "hit_rate":         r["hit_rate"],
                "total_return_pct": r["total_return_pct"],
                "ann_return_pct":   r["ann_return_pct"],
                "ann_vol_pct":      r["ann_vol_pct"],
                "sharpe":           r["sharpe"],
                "max_drawdown_pct": r["max_drawdown_pct"],
                "final_equity":     r["final_equity"],
            })
    return pd.DataFrame(rows).set_index(["model", "split"])


# ─── Plot ────────────────────────────────────────────────────────────────────

def plot_equity_curves(
    curves:       dict,
    buy_and_hold: pd.Series,
    title:        str,
    out_path:     str,
) -> None:
    """Overlays each model's equity curve against a buy-and-hold benchmark."""
    fig, ax = plt.subplots(figsize=(13, 6))

    ax.plot(
        buy_and_hold.index, buy_and_hold.values,
        linestyle="--", color=INK, alpha=0.85, linewidth=1.4,
        label="buy & hold SPY (benchmark)",
    )
    for name, eq in curves.items():
        if eq is None or len(eq) == 0:
            continue
        ax.plot(
            eq.index, eq.values,
            color=model_color(name), alpha=0.9, linewidth=1.4,
            label=name,
        )

    ax.axhline(1.0, color=GRAY, linewidth=0.6)
    ax.set_title(title)
    ax.set_ylabel("Equity (start = 1)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", ncol=2, fontsize=8)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")
