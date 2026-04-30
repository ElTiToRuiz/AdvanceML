# tune_models.py
# ─────────────────────────────────────────────────────────────────────────────
# Block C — hyperparameter tune the 3 models on the (winner imputation,
# winner imbalance) combination from Blocks A + B.
#
# Search method per model (matches the planning doc):
#   - LogReg        → manual grid search over LOGREG_GRID
#   - RandomForest  → randomised search over RF_GRID, RF_RANDOM_ITER iter
#   - XGBoost       → Optuna (TPE + MedianPruner, XGB_OPTUNA_TRIALS trials)
#
# Selection uses a single (train, val) hold-out — no CV, no leakage.
#
# Outputs:
#   reports/activity2/models/logreg_grid.csv
#   reports/activity2/models/random_forest_random.csv
#   reports/activity2/models/xgboost_optuna.csv
#   reports/activity2/models/model_tuning_results.csv   (one row per model)
#   reports/activity2/models/tuning_history_xgboost.png Optuna trial history
#   reports/activity2/models/winner_bundle.joblib       Block D input
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import json
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.utils.class_weight import compute_sample_weight

from ..config import (
    LOGREG_GRID, RF_GRID, RF_RANDOM_ITER,
    REPORT_MODELS_DIR, REPORT_PREPROCESSING_DIR,
    TRAIN_RATIO, VAL_RATIO, RANDOM_SEED,
    XGB_OPTUNA_RANGES, XGB_OPTUNA_TRIALS,
)
from ..data.loader import load_with_nans
from ..evaluation.metrics import macro_f1
from ..models.logreg import LogRegClassifier
from ..models.random_forest import RandomForestModel
from ..models.xgboost_clf import XGBoostModel
from ..preprocessing.imbalance import _resample, build_strategy
from ..preprocessing.imputation import add_mnar_indicators, build_imputer


warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _winners() -> tuple[str, str]:
    imp = pd.read_csv(os.path.join(REPORT_PREPROCESSING_DIR, "imputation_results.csv"))
    imb = pd.read_csv(os.path.join(REPORT_PREPROCESSING_DIR, "imbalance_results.csv"))
    return (
        imp.sort_values("val_macro_f1", ascending=False).iloc[0]["method"],
        imb.sort_values("val_macro_f1", ascending=False).iloc[0]["strategy"],
    )


def _prepare(winner_imp: str, winner_imb: str):
    df = load_with_nans()
    y = df["regime"].dropna()
    X = df.loc[y.index].drop(columns=["regime"])
    if "SPY_price" in X.columns:
        X = X.drop(columns=["SPY_price"])
    X = X.select_dtypes(include="number")

    if winner_imp.endswith("+mnar_indicator"):
        X = add_mnar_indicators(X)
        method = winner_imp.replace("+mnar_indicator", "")
    else:
        method = winner_imp

    n = len(X); n_tr = int(n * TRAIN_RATIO); n_va_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    X_tr, X_va, X_te = X.iloc[:n_tr], X.iloc[n_tr:n_va_end], X.iloc[n_va_end:]
    y_tr, y_va, y_te = y.iloc[:n_tr], y.iloc[n_tr:n_va_end], y.iloc[n_va_end:]

    imp = build_imputer(method).fit(X_tr)
    X_tr_i, X_va_i, X_te_i = imp.transform(X_tr), imp.transform(X_va), imp.transform(X_te)

    strat = build_strategy(winner_imb)
    if strat.resampler is not None:
        X_fit, y_fit = _resample(strat.resampler, X_tr_i, y_tr)
    else:
        X_fit, y_fit = X_tr_i, y_tr

    return X_fit, y_fit, X_tr_i, y_tr, X_va_i, y_va, X_te_i, y_te, strat


# ── Search routines ──────────────────────────────────────────────────────────


def _tune_logreg(X_fit, y_fit, X_val, y_val, strat):
    cw = "balanced" if strat.use_class_weight else None
    rows = []
    best = (None, -np.inf, None)
    for C in LOGREG_GRID["C"]:
        for l1_ratio in LOGREG_GRID["l1_ratio"]:
            m = LogRegClassifier(C=C, l1_ratio=l1_ratio, class_weight=cw)
            m.fit(X_fit, y_fit)
            f1 = macro_f1(y_val, m.predict(X_val))
            rows.append({"C": C, "l1_ratio": l1_ratio, "val_macro_f1": f1})
            if f1 > best[1]:
                best = ({"C": C, "l1_ratio": l1_ratio, "class_weight": cw}, f1, m)
    pd.DataFrame(rows).to_csv(
        os.path.join(REPORT_MODELS_DIR, "logreg_grid.csv"), index=False,
    )
    return {"name": "logreg", "best_params": best[0], "val_macro_f1": best[1], "model": best[2]}


def _tune_random_forest(X_fit, y_fit, X_val, y_val, strat):
    rng = np.random.default_rng(RANDOM_SEED)
    cw = "balanced" if strat.use_class_weight else None
    rows = []
    best = (None, -np.inf, None)
    keys = list(RF_GRID.keys())
    for _ in range(RF_RANDOM_ITER):
        params = {k: RF_GRID[k][int(rng.integers(0, len(RF_GRID[k])))] for k in keys}
        m = RandomForestModel(**params, class_weight=cw)
        m.fit(X_fit, y_fit)
        f1 = macro_f1(y_val, m.predict(X_val))
        rows.append({**params, "val_macro_f1": f1})
        if f1 > best[1]:
            best = ({**params, "class_weight": cw}, f1, m)
    pd.DataFrame(rows).to_csv(
        os.path.join(REPORT_MODELS_DIR, "random_forest_random.csv"), index=False,
    )
    return {"name": "random_forest", "best_params": best[0], "val_macro_f1": best[1], "model": best[2]}


def _tune_xgboost(X_fit, y_fit, X_val, y_val, strat):
    history = []

    def objective(trial):
        params = {
            "learning_rate":    trial.suggest_float("learning_rate",    *XGB_OPTUNA_RANGES["learning_rate"]),
            "max_depth":        trial.suggest_int(  "max_depth",        *XGB_OPTUNA_RANGES["max_depth"]),
            "n_estimators":     trial.suggest_int(  "n_estimators",     *XGB_OPTUNA_RANGES["n_estimators"]),
            "subsample":        trial.suggest_float("subsample",        *XGB_OPTUNA_RANGES["subsample"]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *XGB_OPTUNA_RANGES["colsample_bytree"]),
            "reg_alpha":        trial.suggest_float("reg_alpha",        *XGB_OPTUNA_RANGES["reg_alpha"], log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda",       *XGB_OPTUNA_RANGES["reg_lambda"], log=True),
        }
        m = XGBoostModel(**params)
        sw = compute_sample_weight("balanced", y_fit) if strat.use_class_weight else None
        m.fit(X_fit, y_fit, sample_weight=sw)
        f1 = macro_f1(y_val, m.predict(X_val))
        history.append({**params, "val_macro_f1": f1, "trial": trial.number})
        return f1

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=RANDOM_SEED),
        pruner=MedianPruner(),
    )
    study.optimize(objective, n_trials=XGB_OPTUNA_TRIALS, show_progress_bar=False)

    pd.DataFrame(history).to_csv(
        os.path.join(REPORT_MODELS_DIR, "xgboost_optuna.csv"), index=False,
    )
    best_params = study.best_params
    sw = compute_sample_weight("balanced", y_fit) if strat.use_class_weight else None
    best_model = XGBoostModel(**best_params).fit(X_fit, y_fit, sample_weight=sw)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    df = pd.DataFrame(history)
    ax.plot(df["trial"], df["val_macro_f1"], "o-", linewidth=1, markersize=3)
    ax.axhline(study.best_value, color="red", linestyle="--", label=f"best = {study.best_value:.4f}")
    ax.set_xlabel("trial")
    ax.set_ylabel("val macro-F1")
    ax.set_title("XGBoost — Optuna trial history")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_MODELS_DIR, "tuning_history_xgboost.png"))
    plt.close(fig)

    return {
        "name": "xgboost",
        "best_params": best_params,
        "val_macro_f1": float(study.best_value),
        "model": best_model,
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("Activity 2 — Block C: hyperparameter tuning")
    print("=" * 60)
    os.makedirs(REPORT_MODELS_DIR, exist_ok=True)

    winner_imp, winner_imb = _winners()
    print(f"  Using imputation = {winner_imp}, imbalance = {winner_imb}")

    (X_fit, y_fit, X_tr_i, y_tr, X_va_i, y_va, X_te_i, y_te, strat) = _prepare(winner_imp, winner_imb)

    results = []
    for tuner in (_tune_logreg, _tune_random_forest, _tune_xgboost):
        r = tuner(X_fit, y_fit, X_va_i, y_va, strat)
        print(f"  {r['name']:15s}  val macro-F1 = {r['val_macro_f1']:.4f}  params = {r['best_params']}")
        results.append(r)

    df = pd.DataFrame([
        {"model": r["name"], "val_macro_f1": r["val_macro_f1"], "best_params": json.dumps(r["best_params"])}
        for r in results
    ]).sort_values("val_macro_f1", ascending=False)
    df.to_csv(os.path.join(REPORT_MODELS_DIR, "model_tuning_results.csv"), index=False)

    winner = df.iloc[0]["model"]
    print(f"\n  ► Block C winner: {winner}  ({df.iloc[0]['val_macro_f1']:.4f})")

    bundle = {
        "winner_name":    winner,
        "winner_model":   {r["name"]: r["model"] for r in results}[winner],
        "winner_imp":     winner_imp,
        "winner_imb":     winner_imb,
        "X_test":         X_te_i,
        "y_test":         y_te,
        "X_val":          X_va_i,
        "y_val":          y_va,
        "feature_names":  list(X_te_i.columns),
    }
    joblib.dump(bundle, os.path.join(REPORT_MODELS_DIR, "winner_bundle.joblib"))
    return winner


if __name__ == "__main__":
    main()
