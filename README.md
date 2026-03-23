
## Author

**Igor Ruiz, Jon Vargas and Mikel Sanchez**
Advance Machine Learning

---

# Predictive Analysis & Explainability in Financial Markets

Machine Learning project analyzing the relationship between U.S. equities (SPY) and fixed income (IEF, ^TNX) using historical data from 2003 to March 2026.

---

## Project Structure

```
project/
├── config.py                  # central config: tickers, dates, paths, split ratios
├── requirements.txt           # all dependencies
├── src/
│   ├── download_data.py        # downloads raw data from Yahoo Finance
│   ├── clean_data.py           # cleans, engineers features, splits data
│   └── eda_plots.py            # generates all 7 EDA charts
├── data/
│   ├── raw/                   # created by 01 — original CSVs, never modify
│   └── processed/             # created by 02 — cleaned datasets
└── reports/                   # created by 03 — all output charts (PNG)
```

## Setup

**Create and Activate the Environment**

With pip:
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

With uv:
```bash
uv venv
source .venv/bin/activate       # Mac/Linux
.venv\Scripts\activate          # Windows
```

---

**Install dependencies**

With pip:
```bash
pip install -r requirements.txt
```

With uv:
```bash
uv add yfinance pandas numpy matplotlib scipy statsmodels
```

---

## How to Run

The scripts must be run **in order**. Each one depends on the output of the previous.

### Step 1 — Download raw data
```bash
python download_data.py
```
Downloads SPY, IEF and ^TNX from Yahoo Finance and saves them as CSVs to `data/raw/`. Requires internet connection.

### Step 2 — Clean and process
```bash
python clean_data.py
```
Merges the three assets on common trading dates, handles missing values, computes log returns and all derived features, and saves the train/validation/test split to `data/processed/`.

### Step 3 — Generate EDA charts
```bash
python eda_plots.py
```
Reads the cleaned dataset and generates all 7 figures to `reports/figures/`.

---

## Output Charts

| File | Description |
|------|-------------|
| `graph1_price_history.png` | Full price history of SPY, IEF and ^TNX (2003–2026) |
| `graph2_distributions.png` | Return distributions + Q-Q plots (fat tails) |
| `graph3_rolling_stats.png` | Rolling mean return and annualised volatility |
| `graph4_stationarity.png`  | Raw prices vs log returns — ADF stationarity test |
| `graph5_acf_pacf.png`      | ACF and PACF for SPY and IEF returns |
| `graph6_correlation.png`   | Dynamic SPY–IEF correlation over time |
| `graph7_split.png`         | Train / Validation / Test split visualisation |

---

## Assets

| Ticker | Name | Role |
|--------|------|------|
| SPY | SPDR S&P 500 ETF | Target variable |
| IEF | iShares 7-10Y Treasury Bond ETF | Exogenous variable |
| ^TNX | 10-Year Treasury Note Yield | Exogenous variable |

Data range: `2003-01-01` → `2026-03-01`

---

## Common Errors

**`ModuleNotFoundError: No module named 'statsmodels'`**
```bash
uv add statsmodels
# or
pip install statsmodels
```

**`FileNotFoundError: full_dataset.csv`**
You skipped a step. Run `01` and `02` before `03`.

**`No data returned for ticker`**
Check your internet connection and that the ticker symbol in `config.py` is correct.