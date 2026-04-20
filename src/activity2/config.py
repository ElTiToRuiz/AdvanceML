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
