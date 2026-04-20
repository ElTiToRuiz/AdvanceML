# config.py
# ─────────────────────────────────────────────────────────────────────────────
# Central configuration for the Financial ML EDA project
# All scripts import from here — never hardcode these values elsewhere
# ─────────────────────────────────────────────────────────────────────────────

TICKERS = {
    "equity": "SPY",    # S&P 500 ETF — target variable
    "bond":   "IEF",    # 7-10Y Treasury Bond ETF — exogenous variable
    "yield":  "^TNX",   # 10-Year Treasury Yield — exogenous variable
}

START_DATE = "2003-01-01"   # SPY inception ~2003, IEF ~2002, TNX full history
END_DATE   = "2026-03-01"

RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
FIGURES_DIR   = "reports"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 0.15  (remainder)

RANDOM_SEED = 42