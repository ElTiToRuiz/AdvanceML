# compare_imbalance.py
# ─────────────────────────────────────────────────────────────────────────────
# Block B — pick the best imbalance strategy.
#
# Uses the WINNING imputation from Block A (read from imputation_results.csv).
# Each strategy is applied on top of LogReg, validated with macro-F1.
#
# Strategy types are dispatched on `ImbalanceStrategy` flags:
#   - resampler ∈ {smote, adasyn, smote_enn, rus} → resample TRAIN only
#   - use_class_weight                            → pass class_weight='balanced'
#   - use_threshold                               → tune thresholds on val
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os
import warnings

import pandas as pd

from ..config import IMBALANCE_STRATEGIES, REPORT_PREPROCESSING_DIR, TRAIN_RATIO, VAL_RATIO
from ..data.loader import load_with_nans
from ..evaluation.metrics import macro_f1
from ..evaluation.plots import bar_macro_f1
from ..models.logreg import LogRegClassifier
from ..preprocessing.imbalance import (
    _resample, apply_thresholds, build_strategy, tune_thresholds,
)
from ..preprocessing.imputation import add_mnar_indicators, build_imputer


warnings.filterwarnings("ignore")


def _winner_imputation() -> str:
    df = pd.read_csv(os.path.join(REPORT_PREPROCESSING_DIR, "imputation_results.csv"))
    return df.sort_values("val_macro_f1", ascending=False).iloc[0]["method"]


def _prepare_split():
    df = load_with_nans()
    y = df["regime"].dropna()
    X = df.loc[y.index].drop(columns=["regime"])
    if "SPY_price" in X.columns:
        X = X.drop(columns=["SPY_price"])
    X = X.select_dtypes(include="number")
    n = len(X); n_tr = int(n * TRAIN_RATIO); n_va_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    return X.iloc[:n_tr], y.iloc[:n_tr], X.iloc[n_tr:n_va_end], y.iloc[n_tr:n_va_end]


def _impute_with(winner: str, X_tr, X_va):
    method = winner.replace("+mnar_indicator", "")
    use_indicators = winner.endswith("+mnar_indicator")
    if use_indicators:
        X_tr = add_mnar_indicators(X_tr)
        X_va = add_mnar_indicators(X_va)
    imp = build_imputer(method).fit(X_tr)
    return imp.transform(X_tr), imp.transform(X_va)


def _evaluate_strategy(name: str, X_tr_i, y_tr, X_va_i, y_va) -> float:
    strat = build_strategy(name)

    # Resampling on TRAIN only
    if strat.resampler is not None:
        X_fit, y_fit = _resample(strat.resampler, X_tr_i, y_tr)
    else:
        X_fit, y_fit = X_tr_i, y_tr

    # Class weighting at the model level
    cw = "balanced" if strat.use_class_weight else None
    model = LogRegClassifier(class_weight=cw).fit(X_fit, y_fit)

    # Threshold tuning is post-fit on val
    if strat.use_threshold:
        proba = model.predict_proba(X_va_i)
        shifts = tune_thresholds(y_va.values, proba, list(model.classes_))
        pred = apply_thresholds(proba, list(model.classes_), shifts)
    else:
        pred = model.predict(X_va_i)

    return float(macro_f1(y_va, pred))


def main() -> str:
    print("=" * 60)
    print("Activity 2 — Block B: imbalance comparison")
    print("=" * 60)

    os.makedirs(REPORT_PREPROCESSING_DIR, exist_ok=True)

    winner_imp = _winner_imputation()
    print(f"  Using winning imputation from Block A: {winner_imp}")

    X_tr, y_tr, X_va, y_va = _prepare_split()
    X_tr_i, X_va_i = _impute_with(winner_imp, X_tr, X_va)

    rows = []
    for name in IMBALANCE_STRATEGIES:
        score = _evaluate_strategy(name, X_tr_i, y_tr, X_va_i, y_va)
        rows.append({"strategy": name, "val_macro_f1": score})
        print(f"  {name:14s}  val macro-F1 = {score:.4f}")

    df = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
    df.to_csv(os.path.join(REPORT_PREPROCESSING_DIR, "imbalance_results.csv"), index=False)

    bar_macro_f1(
        df, by="strategy", score_col="val_macro_f1",
        title="Block B — val macro-F1 by imbalance strategy",
        save_to=os.path.join(REPORT_PREPROCESSING_DIR, "imbalance_macro_f1_bar.png"),
    )

    winner = df.iloc[0]["strategy"]
    print(f"\n  ► Winner: {winner}  ({df.iloc[0]['val_macro_f1']:.4f})")
    return winner


if __name__ == "__main__":
    main()
