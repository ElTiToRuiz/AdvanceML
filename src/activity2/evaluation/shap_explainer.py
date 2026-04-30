# shap_explainer.py
# ─────────────────────────────────────────────────────────────────────────────
# Tree SHAP wrapper for Activity 2.
#
# Tree SHAP is the variant the class notes recommend for tree ensembles:
# exact Shapley values, polynomial-time on tree depth, much cheaper
# than KernelSHAP. We expose:
#
#   explain(classifier, X_test, save_dir)  → dispatch on classifier type:
#       - tree models (RF, XGBoost) → Tree SHAP global importance + summary
#       - LogReg                    → fallback to standardised |coefficients|
#
# Output: shap_summary.png + shap_importance.csv in save_dir.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .style import apply_style, model_color


apply_style()


def explain_tree_model(
    classifier,                       # XGBoostModel or RandomForestModel
    X_test: pd.DataFrame,
    save_dir: str,
) -> pd.DataFrame:
    """
    Compute Tree SHAP for a tree ensemble + write summary plot.

    Returns the per-feature global importance (mean |SHAP|) per class
    as a DataFrame. shap.TreeExplainer takes either an sklearn-style
    estimator or an xgboost booster — the `booster` property on
    XGBoostModel exposes the right object.
    """
    os.makedirs(save_dir, exist_ok=True)
    booster = getattr(classifier, "booster", classifier._model)
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_test)

    # Multi-class SHAP value layout differs:
    #   - sklearn RF returns a list of per-class matrices (n_samples, n_features)
    #   - XGBoost returns either a list or a 3-D array
    if isinstance(shap_values, list):
        importances = np.stack([np.abs(s).mean(axis=0) for s in shap_values], axis=1)
        class_names = list(classifier.classes_)
    else:
        if shap_values.ndim == 3:
            importances = np.abs(shap_values).mean(axis=0)
            class_names = list(classifier.classes_)
        else:
            importances = np.abs(shap_values).mean(axis=0).reshape(-1, 1)
            class_names = ["overall"]

    df = pd.DataFrame(
        importances,
        index=X_test.columns,
        columns=class_names,
    ).sort_values(class_names[0], ascending=False)
    df.to_csv(os.path.join(save_dir, "shap_importance.csv"))

    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(df))))
    df.mean(axis=1).sort_values().plot(kind="barh", ax=ax, color=model_color(classifier.name))
    ax.set_xlabel("mean |SHAP|, averaged across classes")
    ax.set_title(f"Global feature importance — {classifier.name}")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "shap_summary.png"))
    plt.close(fig)

    return df


def explain_logreg(
    classifier,            # LogRegClassifier
    X_test: pd.DataFrame,
    save_dir: str,
) -> pd.DataFrame:
    """
    SHAP fallback for LogReg: |coefficient| on standardised features.
    Tree SHAP doesn't apply; LinearExplainer is overkill for a course
    deliverable. The proxy ships as `shap_summary.png` for parity.
    """
    os.makedirs(save_dir, exist_ok=True)
    coefs = classifier._model.coef_                  # (n_classes, n_features)
    importance = pd.DataFrame(
        np.abs(coefs).T,
        index=X_test.columns,
        columns=list(classifier.classes_),
    )
    importance.to_csv(os.path.join(save_dir, "shap_importance.csv"))

    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(importance))))
    importance.mean(axis=1).sort_values().plot(
        kind="barh", ax=ax, color=model_color(classifier.name)
    )
    ax.set_xlabel("|coefficient| on standardised features (proxy for SHAP)")
    ax.set_title(f"Feature importance — {classifier.name}")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "shap_summary.png"))
    plt.close(fig)
    return importance


def explain(classifier, X_test: pd.DataFrame, save_dir: str) -> pd.DataFrame:
    """Dispatch on the classifier type."""
    if classifier.name in {"random_forest", "xgboost"}:
        return explain_tree_model(classifier, X_test, save_dir)
    if classifier.name == "logreg":
        return explain_logreg(classifier, X_test, save_dir)
    raise ValueError(f"No explainer for {classifier.name}")
