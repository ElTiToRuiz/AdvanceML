# compare_imputations.py
# ─────────────────────────────────────────────────────────────────────────────
# Block A — pick the best imputation method on LogReg + class_weight=balanced.
#
# Strategy (matches the planning doc):
#   1. Load full_dataset_with_nans (preserves the structural NaN pattern)
#   2. Build the chronological train/val/test split on the row index
#   3. For each method in IMPUTATION_METHODS, fit on TRAIN, transform both
#      splits, train LogReg(class_weight='balanced'), score val macro-F1
#   4. Add a "<winner>+mnar_indicator" run that appends 3 binary
#      missingness indicators on GLD/USO/UUP — see if they help
#   5. Save a CSV + bar chart, return the winner name
#
# class_weight='balanced' is applied during Block A on purpose, so the
# imputation comparison isn't blind on the rare classes (documented in the
# spec, §4.1).
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os
import warnings

import pandas as pd

from ..config import (
    IMPUTATION_METHODS,
    REPORT_PREPROCESSING_DIR,
    TRAIN_RATIO,
    VAL_RATIO,
)
from ..data.loader import load_with_nans
from ..evaluation.metrics import macro_f1
from ..evaluation.plots import bar_macro_f1
from ..models.logreg import LogRegClassifier
from ..preprocessing.imputation import add_mnar_indicators, build_imputer


warnings.filterwarnings("ignore")


# ── helpers ─────────────────────────────────────────────────────────────────


def _split_indices(n: int) -> tuple[int, int]:
    n_tr = int(n * TRAIN_RATIO)
    n_va = int(n * VAL_RATIO)
    return n_tr, n_tr + n_va


def _prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop the target and the price column from the feature matrix."""
    y = df["regime"].dropna()
    X = df.loc[y.index].drop(columns=["regime"])
    if "SPY_price" in X.columns:
        X = X.drop(columns=["SPY_price"])
    X = X.select_dtypes(include="number")
    return X, y


def _evaluate_imputation(method: str, X_tr, y_tr, X_va, y_va) -> float:
    imp = build_imputer(method).fit(X_tr)
    X_tr_i = imp.transform(X_tr)
    X_va_i = imp.transform(X_va)
    model = LogRegClassifier(class_weight="balanced").fit(X_tr_i, y_tr)
    pred = model.predict(X_va_i)
    return float(macro_f1(y_va, pred))


# ── Pipeline entry ──────────────────────────────────────────────────────────


def main() -> str:
    print("=" * 60)
    print("Activity 2 — Block A: imputation comparison")
    print("=" * 60)

    os.makedirs(REPORT_PREPROCESSING_DIR, exist_ok=True)

    df = load_with_nans()
    X, y = _prepare_xy(df)
    n_tr, n_va_end = _split_indices(len(X))

    X_tr, X_va = X.iloc[:n_tr], X.iloc[n_tr:n_va_end]
    y_tr, y_va = y.iloc[:n_tr], y.iloc[n_tr:n_va_end]

    rows = []
    for method in IMPUTATION_METHODS:
        score = _evaluate_imputation(method, X_tr, y_tr, X_va, y_va)
        rows.append({"method": method, "val_macro_f1": score})
        print(f"  {method:8s}  val macro-F1 = {score:.4f}")

    # Extra: winner + MNAR indicators
    best_so_far = max(rows, key=lambda r: r["val_macro_f1"])["method"]
    X_tr_ind = add_mnar_indicators(X_tr)
    X_va_ind = add_mnar_indicators(X_va)
    score_ind = _evaluate_imputation(best_so_far, X_tr_ind, y_tr, X_va_ind, y_va)
    rows.append({"method": f"{best_so_far}+mnar_indicator", "val_macro_f1": score_ind})
    print(f"  {best_so_far}+mnar_indicator  val macro-F1 = {score_ind:.4f}")

    df_results = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
    df_results.to_csv(os.path.join(REPORT_PREPROCESSING_DIR, "imputation_results.csv"), index=False)

    bar_macro_f1(
        df_results, by="method", score_col="val_macro_f1",
        title="Block A — val macro-F1 by imputation method",
        save_to=os.path.join(REPORT_PREPROCESSING_DIR, "imputation_macro_f1_bar.png"),
    )

    winner = df_results.iloc[0]["method"]
    print(f"\n  ► Winner: {winner}  ({df_results.iloc[0]['val_macro_f1']:.4f})")
    return winner


if __name__ == "__main__":
    main()
