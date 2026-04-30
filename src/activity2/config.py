"""
Central configuration for Activity 2 — Multi-class market regime classification.

Activity 2 reuses the financial domain of activity 1 but extends the universe with
'safe haven' (GLD), 'fear gauge' (^VIX), 'dollar' (UUP) and 'oil' (USO) assets.
The different inception dates of these ETFs create REAL missing data, which is
exactly what motivates the imputation comparison required by topic 1 of the
assignment (Imbalanced data, Imputation methods, Multi-class classification).
"""

# ── Asset universe ───────────────────────────────────────────────────────────
# Roles tell the EDA / modeling code how to interpret each asset.
TICKERS = {
    "SPY":  {"role": "equity",      "name": "SPDR S&P 500 ETF"},
    "IEF":  {"role": "bond_price",  "name": "iShares 7-10Y Treasury Bond ETF"},
    "^TNX": {"role": "bond_yield",  "name": "10-Year Treasury Yield"},
    "^VIX": {"role": "fear",        "name": "CBOE Volatility Index"},
    "GLD":  {"role": "safe_haven",  "name": "SPDR Gold Trust"},
    "UUP":  {"role": "dollar",      "name": "Invesco DB US Dollar Bullish ETF"},
    "USO":  {"role": "commodity",   "name": "United States Oil Fund"},
}

# Activity 1 starts in 2003. Several activity-2 assets launched later
# (GLD: 2004, USO: 2006, UUP: 2007). Keeping 2003 as start date intentionally
# preserves long missing periods so the imputation study has real data to work on.
START_DATE = "2003-01-01"
END_DATE   = "2026-03-01"

# ── Paths (separate from activity 1 to keep both pipelines independent) ──────
RAW_DIR       = "data/activity2/raw"
PROCESSED_DIR = "data/activity2/processed"
FIGURES_DIR   = "reports/activity2"

# ── Regime label thresholds ──────────────────────────────────────────────────
# Daily SPY log-return cutoffs (in % terms) used to bucket each day into one
# of four regimes. These are deliberately asymmetric: markets fall faster than
# they rise, and crashes are what we care about most.
REGIME_THRESHOLDS = {
    "crash":      -0.02,   # SPY return  <= -2.0%
    "correction": -0.005,  # -2.0%  <  ret <= -0.5%
    "normal":      0.005,  # -0.5%  <  ret <=  0.5%
    "rally":       None,   # ret > 0.5%
}

REGIME_ORDER = ["crash", "correction", "normal", "rally"]

# ── Split ratios (sequential, no shuffle — same convention as activity 1) ────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
RANDOM_SEED = 42


# ── Block A: imputation method list ──────────────────────────────────────────
IMPUTATION_METHODS = ["mean", "median", "ffill", "linear", "knn", "mice"]


# ── Block B: imbalance strategy list ─────────────────────────────────────────
IMBALANCE_STRATEGIES = [
    "untouched", "class_weight", "smote", "adasyn", "smote_enn", "rus", "threshold",
]


# ── Block C: model list and hyperparameter grids ────────────────────────────
MODEL_NAMES = ["logreg", "random_forest", "xgboost"]


# Note: sklearn 1.8 deprecated `penalty='l1'/'l2'` in favour of `l1_ratio`.
# l1_ratio = 0.0 → pure L2,  l1_ratio = 1.0 → pure L1.
LOGREG_GRID = {
    "C":         [0.01, 0.1, 1.0, 10.0],
    "l1_ratio":  [0.0, 1.0],
}


RF_GRID = {
    "n_estimators":     [200, 400, 800],
    "max_depth":        [None, 6, 10, 20],
    "min_samples_leaf": [1, 2, 5, 10],
    "max_features":     ["sqrt", "log2", 0.5],
}
RF_RANDOM_ITER = 50


XGB_OPTUNA_TRIALS = 200
XGB_OPTUNA_RANGES = {
    "learning_rate":    (0.01, 0.30),
    "max_depth":        (3, 10),
    "n_estimators":     (200, 800),
    "subsample":        (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "reg_alpha":        (1e-3, 10.0),     # log scale
    "reg_lambda":       (1e-3, 10.0),     # log scale
}


# ── Output paths under reports/activity2/ ───────────────────────────────────
REPORT_PREPROCESSING_DIR = "reports/activity2/preprocessing"
REPORT_MODELS_DIR        = "reports/activity2/models"
REPORT_FINAL_DIR         = "reports/activity2/final"
REPORT_SHAP_DIR          = "reports/activity2/shap"
