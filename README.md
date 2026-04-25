# Predictive Analysis & Explainability in Financial Markets

**Authors:** Igor Ruiz · Jon Vargas · Mikel Sanchez
**Course:** Advanced Machine Learning · 2026

Two-activity project on machine learning for financial time series:

- **Activity 1 — Forecasting (regression).** Predict the next-day log-return of SPY (S&P 500 ETF) from 2003 to 2026, using the 10-year Treasury yield (^TNX) as exogenous signal. Compare four model families (baselines, SARIMAX, LSTM, Chronos-2) under six evaluation lenses.
- **Activity 2 — Classification (regimes).** Classify each trading day into `crash · correction · normal · rally`. Tackles imbalanced data, imputation, and multi-class evaluation.

> **Headline result of Activity 1:** no model beats the trivial "always-up" baseline on directional accuracy. This empirically reproduces the Efficient Market Hypothesis (Fama, Nobel 2013) for daily SPY returns and motivates the classification reframe of Activity 2.

---

## Project layout

```
project/
├── main.py                      # dispatcher: `python main.py 1` / `python main.py 2`
├── pyproject.toml               # uv-managed dependencies
├── uv.lock                      # locked dependency versions
│
├── src/
│   ├── activity1/               # ── Time-series forecasting (regression) ──
│   │   ├── config.py            # tickers, dates, paths, split ratios
│   │   ├── main.py              # orchestrates the full Activity-1 pipeline
│   │   │
│   │   ├── data/
│   │   │   ├── download.py      # yfinance → data/raw/
│   │   │   ├── clean.py         # merge + impute + features + sequential split
│   │   │   └── loader.py        # `Splits` container for train/val/test
│   │   │
│   │   ├── eda/
│   │   │   └── plots.py         # 7 EDA charts (price history, ACF/PACF, …)
│   │   │
│   │   ├── models/              # one file per model family, all share Forecaster API
│   │   │   ├── base.py          # abstract Forecaster (fit / predict / observe)
│   │   │   ├── baselines.py     # Naive · Seasonal · Drift/Mean · MA(20)
│   │   │   ├── sarimax.py       # SARIMAX(1,0,1) with TNX_chg exogenous
│   │   │   ├── lstm.py          # PyTorch LSTM (2 layers, 64 hidden, seq_len=20)
│   │   │   └── chronos.py       # Chronos-2 zero-shot + LoRA fine-tune
│   │   │
│   │   ├── evaluation/          # shared metrics + plots, model-agnostic
│   │   │   ├── metrics.py       # RMSE · MAE · DirAcc · Skill · IC · weighted_acc
│   │   │   ├── plots.py         # actual-vs-pred · threshold curves · IC bars
│   │   │   ├── backtesting.py   # long/short P&L simulation + equity curves
│   │   │   └── style.py         # shared matplotlib palette / rcParams
│   │   │
│   │   └── pipelines/           # entry-point scripts (one per experiment block)
│   │       ├── run_baselines.py
│   │       ├── run_sarimax.py
│   │       ├── run_lstm.py
│   │       ├── run_chronos.py
│   │       ├── tune_lstm.py             # grid search over LSTM hyperparameters
│   │       ├── finetune_chronos.py      # LoRA / full fine-tune of Chronos-2
│   │       ├── run_multi_horizon.py     # 1-day / 5-day / 30-day comparison
│   │       └── run_all.py               # unified comparison + leaderboard
│   │
│   └── activity2/               # ── Multi-class classification (work in progress) ──
│       ├── config.py            # asset universe + regime thresholds
│       ├── data/                # download, clean, missing-value handling
│       └── eda/
│
├── scripts/
│   ├── build_activity1_pptx.py  # generate the Activity 1 slide deck (PPTX)
│   └── build_activity2_doc.py   # generate the Activity 2 documentation (DOCX)
│
├── data/                        # (gitignored) raw + processed CSVs
└── reports/                     # (gitignored) all generated outputs
    └── activity1/
        ├── graph1_price_history.png … graph7_split.png    # EDA
        ├── models/                                         # model results
        │   ├── all_metrics.csv · pnl_summary.csv · …
        │   ├── all_test.png · pnl_test.png · ic_test.png · …
        └── presentation/
            ├── Activity1_Presentation.pptx                 # slide deck
            └── presentation.md                             # markdown source
```

---

## Setup

Requires **Python ≥ 3.13**. Dependency management uses [`uv`](https://github.com/astral-sh/uv).

```bash
# Create + activate the virtual environment
uv venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

# Install all dependencies from uv.lock
uv sync
```

Main dependencies:

| Library              | Purpose                              |
|----------------------|--------------------------------------|
| `yfinance`           | Download historical price data       |
| `pandas`, `numpy`    | Data wrangling                       |
| `scipy`, `statsmodels` | ADF test · ACF/PACF · SARIMAX      |
| `matplotlib`         | All plots                            |
| `torch`              | LSTM model                           |
| `chronos-forecasting` | Amazon Chronos-2 foundation model   |
| `python-pptx`        | Generate the PowerPoint deck         |

Optional for proper LoRA fine-tuning of Chronos-2:

```bash
uv add peft
```

(Without `peft`, the fine-tune pipeline falls back to full fine-tuning — slower but functionally equivalent.)

---

## How to run

### Full Activity 1 pipeline (one command)

```bash
python main.py 1
```

Runs in order: download → clean → EDA charts → unified model comparison.

### Or run each block individually

```bash
# Data
python -m src.activity1.data.download
python -m src.activity1.data.clean

# Exploratory analysis
python -m src.activity1.eda.plots

# Per-family pipelines
python -m src.activity1.pipelines.run_baselines
python -m src.activity1.pipelines.run_sarimax
python -m src.activity1.pipelines.run_lstm
python -m src.activity1.pipelines.run_chronos

# Advanced experiments
python -m src.activity1.pipelines.tune_lstm           # grid search (~3 min)
python -m src.activity1.pipelines.run_multi_horizon   # 1d / 5d / 30d
python -m src.activity1.pipelines.finetune_chronos    # ~15 min on CPU

# Final unified comparison + leaderboard
python -m src.activity1.pipelines.run_all
```

### Build the slide deck

```bash
uv run python scripts/build_activity1_pptx.py
# → reports/activity1/presentation/Activity1_Presentation.pptx
```

---

## Activity 1 — what we built

**Target:** daily log-return of SPY (5 800 trading days, Jan 2003 – Mar 2026).
**Exogenous:** 10-year Treasury yield change (^TNX). IEF was dropped — perfectly anti-correlated with ^TNX (r ≈ −0.95) and not the causal driver.
**Split:** sequential 70/15/15 = train (2003-2019) · val (2019-2022) · test (2022-2026). The test window covers the post-2022 Fed rate-hike regime.

### Forecaster API

Every model implements the same interface, enforcing leak-free evaluation at the code level:

```python
model.fit(train)            # learn parameters from train ONLY
val_pred  = model.predict(val)
model.observe(val)          # extend known history, no refit
test_pred = model.predict(test)
```

### Six evaluation lenses

| Lens                        | What it measures                                              |
|-----------------------------|---------------------------------------------------------------|
| RMSE / MAE                  | Standard regression metrics                                   |
| Directional Accuracy        | % days the sign of the prediction matches reality             |
| **Skill above drift**       | Dir. Accuracy − % up-days (separates skill from market drift) |
| Threshold curves            | Accuracy vs coverage when keeping only confident predictions  |
| **P&L (Sharpe / drawdown)** | Long/short backtest with equity curve                         |
| **Information Coefficient** | Pearson + Spearman correlation between prediction and reality |

### Key findings (test set)

| Model                | Dir.Acc. | Skill   | IC Pearson | Sharpe | Equity 3.5y |
|----------------------|----------|---------|------------|--------|-------------|
| **LSTM**             | 54.9 %   | −0.6 pp | **+0.10**  | **1.33** | **2.11 ×** |
| drift_mean           | 55.5 %   |  0.0 pp | −0.05      | 1.05   | 1.81 ×      |
| chronos2 (TNX)       | 55.2 %   | −0.4 pp | −0.07      | 0.34   | 1.21 ×      |
| ma_20                | 52.9 %   | −2.6 pp | −0.03      | 0.52   | 1.35 ×      |
| sarimax (TNX)        | 46.9 %   | −8.6 pp | −0.03      | −0.22  | 0.88 ×      |

The trivial "always-up" baseline scores 55.5 % directional accuracy because that is exactly the percentage of up-days in the test window. **No model achieves a statistically meaningful skill above this drift baseline** — empirical confirmation of the Efficient Market Hypothesis on daily horizons. The LSTM does extract signal in MAGNITUDE (IC 0.10 + Sharpe 1.33), which explains its superior P&L, but the directional skill is essentially zero across the board. SARIMAX learns a SPY–TNX relationship in 2003-2019 that breaks under the post-2022 Fed regime — confirmed by 3 independent regime-shift indicators across SARIMAX, LSTM-with-TNX, and fine-tuned Chronos-2.

---

## Activity 2 — work in progress

Multi-class classification of daily SPY into **crash · correction · normal · rally** based on log-return cutoffs (−2 %, −0.5 %, +0.5 %). Adds a wider asset universe (SPY · IEF · ^TNX · ^VIX · GLD · UUP · USO) whose different inception dates produce real missing data, motivating an imputation methods comparison.

Status: data pipeline + EDA implemented. Models pending.

---

## Common errors

**`ModuleNotFoundError: No module named 'X'`**
```bash
uv sync          # re-installs everything from uv.lock
```

**`FileNotFoundError: full_dataset.csv`**
You skipped a step. Run `python main.py 1` to do everything in order, or run `data.download` then `data.clean` before any pipeline.

**`No data returned for ticker`**
Yahoo Finance rate-limit or no internet. Wait a minute and retry.

**Chronos-2 fine-tune falls back to full fine-tuning**
Install `peft`:
```bash
uv add peft
```
