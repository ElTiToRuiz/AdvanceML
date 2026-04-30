# Predictive Analysis & Explainability in Financial Markets

**Authors:** Igor Ruiz · Jon Vargas · Mikel Sanchez
**Course:** Advanced Machine Learning · 2026

Two-activity project on machine learning for financial time series:

- **Activity 1 — Forecasting (regression).** Predict the next-day log-return of SPY (S&P 500 ETF) from 2003 to 2026, using the 10-year Treasury yield (^TNX) as exogenous signal. Compare four model families (baselines, SARIMAX, LSTM, Chronos-2) under six evaluation lenses.
- **Activity 2 — Classification of market regimes.** Classify each trading day into `crash · correction · normal · rally`. Tackles imbalanced data, imputation, and multi-class evaluation in five blocks (A → E).

> **Headlines.**
>
> 1. **Act1**: no model beats the trivial "always-up" baseline on directional accuracy. Empirical confirmation of the Efficient Market Hypothesis (Fama, Nobel 2013) at the daily horizon.
> 2. **Act2 (multi-class)**: macro-F1 lifts from 0.167 (always-most-common baseline) to **0.265** (XGBoost). Crash recall stays low at 4.8 %.
> 3. **Act2 — Block E (crash-focused)**: by reframing the loss around the rare class, crash recall jumps **4.8 % → 61.9 %** on test, but precision collapses to 3.4 % under the post-2022 regime. The "exit on crash signal" backtest reduces max drawdown from 18.8 % to 12.6 % — at the cost of underperforming buy-and-hold by 26 % over 3.5 years. **The EMH fingerprint holds: there is signal in crash detection, but not enough to be net profitable.**

---

## Project layout

```
project/
├── main.py                        # dispatcher: `python main.py 1` / `python main.py 2`
├── pyproject.toml                 # uv-managed dependencies
├── uv.lock                        # locked dependency versions
│
├── docs/
│   └── superpowers/
│       ├── specs/                 # design specs (one per phase)
│       └── plans/                 # implementation plans (one per phase)
│
├── src/
│   ├── activity1/                 # ── Time-series forecasting (regression) ──
│   │   ├── config.py
│   │   ├── main.py
│   │   ├── data/                  # download, clean, loader (Splits container)
│   │   ├── eda/                   # 7 EDA charts
│   │   ├── models/                # baselines, sarimax, lstm, chronos
│   │   ├── evaluation/            # metrics, plots, backtesting, style
│   │   └── pipelines/             # one script per experiment block + run_all
│   │
│   └── activity2/                 # ── Multi-class classification (5 blocks) ──
│       ├── config.py              # universe, regime cutoffs, hyperparam grids,
│       │                          # crash-focus settings (precision floor,
│       │                          # asymmetric cost weights, optuna budget)
│       ├── main.py                # data → EDA → A → B → C → D → E
│       │
│       ├── data/                  # download.py, clean.py, loader.py
│       ├── eda/                   # 11 EDA charts (imbalance / imputation themed)
│       │
│       ├── preprocessing/
│       │   ├── imputation.py      # mean/median/ffill/linear/KNN/MICE + MNAR indicator
│       │   └── imbalance.py       # SMOTE/ADASYN/SMOTE+ENN/RUS, class_weight,
│       │                          # threshold tuning
│       │
│       ├── models/
│       │   ├── base.py            # `Classifier` abstract base class
│       │   ├── baselines.py       # always-most-common, stratified-random
│       │   ├── logreg.py          # multinomial LR (sklearn 1.8 l1_ratio API)
│       │   ├── random_forest.py
│       │   └── xgboost_clf.py     # multi-class wrapper, frozen REGIME_ORDER labels
│       │
│       ├── evaluation/
│       │   ├── metrics.py         # macro-F1, MCC, G-Mean, balanced acc, per-class P/R
│       │   ├── plots.py           # confusion matrix, per-class bars, calibration
│       │   ├── shap_explainer.py  # Tree SHAP (with LogReg coefficient fallback)
│       │   ├── operational.py     # Block E helpers: calibration, threshold search,
│       │   │                      # PR overlay, backtest simulator
│       │   └── style.py           # shared rcParams + REGIME_COLORS + MODEL_COLORS
│       │
│       └── pipelines/
│           ├── compare_imputations.py    # Block A
│           ├── compare_imbalance.py      # Block B
│           ├── tune_models.py            # Block C (Grid + Random + Optuna)
│           ├── final_evaluation.py       # Block D (test + Tree SHAP)
│           ├── crash_focus.py            # Block E (calibration + asymmetric retune
│           │                             # + binary reframe + threshold + PR + backtest)
│           └── run_all.py                # A → B → C → D → E + manifest.json
│
├── data/                          # (gitignored at root) raw + processed CSVs
└── reports/                       # (gitignored at root) all generated outputs
    ├── activity1/                 # 7 EDA charts + per-model results + PPTX
    └── activity2/                 # 11 EDA charts + Block A-E results + manifest
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

| Library                | Purpose                                                     |
|------------------------|-------------------------------------------------------------|
| `yfinance`             | Download historical price data                              |
| `pandas`, `numpy`      | Data wrangling                                              |
| `scipy`, `statsmodels` | ADF test · ACF/PACF · SARIMAX                               |
| `matplotlib`           | All plots                                                   |
| `torch`                | LSTM forecaster (Activity 1)                                |
| `chronos-forecasting`  | Amazon Chronos-2 foundation model (Activity 1)              |
| `scikit-learn`         | LogReg, RF, KNN, MICE, calibration, sample weights          |
| `xgboost`              | Gradient-boosted classifier (Activity 2)                    |
| `imbalanced-learn`     | SMOTE, ADASYN, SMOTE+ENN, RandomUnderSampler                |
| `optuna`               | Bayesian hyperparameter search (TPE + MedianPruner)         |
| `shap`                 | Tree SHAP feature importance                                |
| `joblib`               | Sklearn-canonical model persistence                         |
| `python-pptx`          | Generate the PowerPoint deck                                |

Optional for proper LoRA fine-tuning of Chronos-2 (Activity 1):

```bash
uv add peft
```

---

## How to run

### Full pipeline for either activity

```bash
python main.py 1     # Activity 1 — download → clean → EDA → model comparison
python main.py 2     # Activity 2 — download → clean → EDA → A → B → C → D → E
```

### Activity 1 individual blocks

```bash
python -m src.activity1.data.download
python -m src.activity1.data.clean
python -m src.activity1.eda.plots

python -m src.activity1.pipelines.run_baselines
python -m src.activity1.pipelines.run_sarimax
python -m src.activity1.pipelines.run_lstm
python -m src.activity1.pipelines.run_chronos
python -m src.activity1.pipelines.tune_lstm           # grid search (~3 min)
python -m src.activity1.pipelines.run_multi_horizon   # 1d / 5d / 30d
python -m src.activity1.pipelines.finetune_chronos    # ~15 min on CPU
python -m src.activity1.pipelines.run_all             # leaderboard
```

### Activity 2 individual blocks

```bash
# Data + EDA
python -m src.activity2.data.download
python -m src.activity2.data.clean
python -m src.activity2.eda.plots

# Block A — pick best imputation method (LogReg + class_weight)
python -m src.activity2.pipelines.compare_imputations

# Block B — pick best imbalance strategy (uses Block A winner)
python -m src.activity2.pipelines.compare_imbalance

# Block C — tune LogReg / RF / XGBoost (uses A+B winners)
python -m src.activity2.pipelines.tune_models

# Block D — test set + Tree SHAP on the Block C winner
python -m src.activity2.pipelines.final_evaluation

# Block E — crash-focused: calibration + asymmetric retune + binary
#           + threshold + PR curve + backtest
python -m src.activity2.pipelines.crash_focus

# All five blocks + reproducibility manifest
python -m src.activity2.pipelines.run_all
```

Each Activity 2 pipeline reads the previous block's CSV winner directly,
so blocks run independently as long as the upstream artifacts exist
under `reports/activity2/`. `run_all.py` re-runs every block and writes
`manifest.json` (git commit + library versions + config hash).

### Build the slide deck

```bash
uv run python scripts/build_activity1_pptx.py
# → reports/activity1/presentation/Activity1_Presentation.pptx
```

The Activity 2 deck is pending.

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

## Activity 2 — what we built

Multi-class classification of next-day SPY into **crash · correction · normal · rally** based on log-return cutoffs (−2 %, −0.5 %, +0.5 %). The asset universe is widened with four extra macro signals: `^VIX` (fear), `GLD` (gold / safe-haven), `UUP` (US dollar), `USO` (oil). GLD/USO/UUP launched in 2004/2006/2007, producing real structural missing data that motivates the imputation comparison.

The deliverable is split into **five blocks (A → E)**. The first four implement the original planning document literally; Block E is an additive crash-focused extension that asks "what's the maximum recall on the rare class we can actually achieve?".

### Phase 2 — Blocks A → D (the original plan)

#### Block A: imputation comparison

Six methods compared on val macro-F1 with a fixed `LogReg(class_weight='balanced')`:

| Method               | Val macro-F1 |
|----------------------|--------------|
| linear (winner)      | 0.2781       |
| ffill                | 0.2767       |
| median               | 0.2608       |
| mean                 | 0.2588       |
| MICE                 | 0.2587       |
| KNN (k = 5)          | 0.2565       |
| linear + MNAR ind.   | 0.1737       |

The six imputers cluster within 2 pp of each other — on this dataset
the imputation choice barely matters. The MNAR-indicator variant
**hurts**: GLD/USO/UUP missingness is concentrated entirely in
2003-2007 (only train), so the indicator column is constant 0 in val
and acts as feature noise. A genuine empirical finding about MNAR
indicators on chronologically-split datasets.

#### Block B: imbalance comparison

Seven strategies on top of the winning imputation:

| Strategy                       | Val macro-F1 |
|--------------------------------|--------------|
| class_weight='balanced' (winner) | 0.2778     |
| SMOTE                           | 0.2740     |
| ADASYN                          | 0.2694     |
| RandomUnderSampler              | 0.2660     |
| threshold tuning                | 0.2085     |
| untouched                       | 0.2050     |
| SMOTE + ENN                     | **0.0965** |

SMOTE+ENN collapses — the ENN cleaning step removes too many
borderline points. Threshold tuning alone is weak because it starts
from an untouched model with near-zero mass on the rare classes.

#### Block C: model tuning

Three models with `(linear, class_weight='balanced')`:

| Model         | Search method            | Val macro-F1 |
|---------------|--------------------------|--------------|
| **XGBoost**   | Optuna (TPE, 200 trials) | **0.3558**   |
| Random Forest | Randomised (50 iter)     | 0.3318       |
| LogReg        | Grid                     | 0.2778       |

XGBoost picks up non-linear interactions LogReg cannot.

#### Block D: test set + Tree SHAP

| Metric                | Value   |
|-----------------------|---------|
| macro-F1              | 0.2653  |
| balanced accuracy     | 0.2690  |
| MCC                   | 0.0484  |
| G-Mean                | 0.1834  |
| accuracy              | 0.3993  |
| recall_crash          | **0.0476** |
| recall_correction     | 0.1272  |
| recall_normal         | 0.5779  |
| recall_rally          | 0.3232  |

Per-class precision/recall and the confusion matrix live in
`reports/activity2/final/`. Tree SHAP feature importance lives in
`reports/activity2/shap/`.

The val→test drop (0.356 → 0.265) is the same post-2022 regime shift
already documented in Activity 1. SHAP says the top features for
crash prediction are UUP_ret (dollar), today's SPY return, and USO_ret
(oil) — VIX is **not** the primary feature, contradicting the
folk-finance intuition.

### Phase 3 — Block E (crash-focused extension)

Macro-F1 weights all four classes equally, which dilutes what we
actually care about: detecting crashes ahead of time. Block E
reformulates the problem around `recall_crash` subject to a precision
floor (`PRECISION_FLOOR_CRASH = 0.10`), without modifying any Phase 2
artifact.

#### Five sub-analyses

1. **Probability calibration.** Platt scaling on top of the Phase 2
   XGBoost using val. The reliability diagram on test shifts the
   predicted-vs-empirical curve closer to the diagonal but doesn't
   eliminate over-confidence on the rare class.

2. **Asymmetric-cost re-tuning.** New Optuna search (100 trials)
   optimising `recall_crash` with `precision_crash >= 0.10` enforced
   via score-0 penalty. Sample weights: `crash 10×, correction 3×,
   rally 2×, normal 1×`. Val best score = **0.6863**.

3. **Threshold tuning + PR curve overlay** across the three crash
   detectors (calibrated, asymmetric, binary). The chosen operating
   point on val gets applied to test. The PR curves are saved at
   `reports/activity2/crash_focus/crash_pr_curve.png`.

4. **Binary reframe** (crash vs not-crash, `XGBClassifier` with
   `scale_pos_weight = n_neg/n_pos`). 100 Optuna trials, same
   precision-floor-constrained objective. Val best = **0.6863** —
   matches the asymmetric retune ceiling on val.

5. **Long-only "exit on crash signal" backtest** vs buy-and-hold over
   the 873-day test window.

#### Block E results — operating points on test

| Detector           | Threshold | Val recall | Val prec. | Test recall | Test prec. | Alarms |
|--------------------|-----------|------------|-----------|-------------|------------|--------|
| calibrated_xgb     | 0.061     | 0.471      | 0.10      | 0.190       | 0.017      | 234    |
| asymmetric_xgb     | 0.019     | 0.686      | 0.10      | 0.476       | 0.026      | 381    |
| **binary_xgb**     | 0.084     | 0.686      | 0.10      | **0.619**   | 0.034      | 384    |

**No detector held the precision floor on test** (best test precision = 3.4 %).
The fallback to "highest-recall detector" picked **binary_xgb**.
This is the post-2022 regime-shift cost — not a code bug.

#### Block E results — backtest (873 trading days, Sept 2022 → Mar 2026)

| | "Exit on signal" strategy | Buy & Hold |
|---|---|---|
| Final equity ($1 start) | 1.36 | **1.82** |
| Annualised return       | 9.2 % | **19.0 %** |
| Sharpe ratio            | 0.87 | **1.06** |
| **Max drawdown**        | **−12.6 %** | −18.8 % |
| Crashes evaded          | 8 of 21 | — |
| Crashes missed          | 13 of 21 | — |
| False alarms            | 375 of 383 signals | — |

**Reading.** The strategy avoids 8 of 21 real crashes and reduces max
drawdown by 6 percentage points relative to buy-and-hold. But 98 % of
its signals are false alarms, and the opportunity cost of being out of
the market on those normal-up days drags the final equity 26 % below
buy-and-hold. The signal has *some* predictive power — but not enough
to be net profitable. **The Activity 1 EMH conclusion holds: there is
exploitable information in macro stress signals, but the 1-day horizon
is too noisy for the precision required to convert it into alpha.**

### How the two activities relate

Activity 1 ran a regression on next-day SPY returns and could not
beat a constant "always-up" baseline. The interpretation was the EMH:
the daily-direction signal is essentially random walk + drift.

Activity 2 reformulated the same prediction problem as multi-class
classification of the *magnitude* of the move, with imbalanced classes
forcing the use of macro-F1 over plain accuracy. Phase 2 (Blocks A-D)
showed there *is* real signal — XGBoost lifts macro-F1 from 0.167 to
0.265 — but the rare-class recall stays low.

Block E then asked the operational question directly: maximise crash
recall under a precision floor. We pushed test crash recall from 4.8 %
to 61.9 % — a 13× improvement. The backtest then put a price on that
recall: 6 pp of drawdown reduction at the cost of 26 % less return.

The two activities tell one coherent story. Returns are essentially
unpredictable directionally (Act1). Volatility regimes are partly
predictable (Act2, Phase 2). Crashes can be detected better than
chance (Block E). But none of these signals is strong enough to beat
buy-and-hold after considering opportunity cost. **The market is
weakly inefficient, exactly inefficient enough to reward risk
management, not enough to reward pure timing.**

### Reproducibility

`reports/activity2/manifest.json` is written by `run_all.py` and
captures: ISO timestamp, `git rev-parse HEAD`, the SHA-256 of
`config.py`, and the installed versions of every ML library used
(scikit-learn, xgboost, imbalanced-learn, optuna, shap, pandas, numpy,
joblib). Identical commit + identical lock file → identical numbers.

---

## Common errors

**`ModuleNotFoundError: No module named 'X'`**
```bash
uv sync          # re-installs everything from uv.lock
```

**`FileNotFoundError: full_dataset.csv` (or `winner_bundle.joblib`)**
You skipped a step. Run `python main.py 1` or `python main.py 2` to do
everything in order, or run the upstream block(s) first. Each Activity 2
pipeline expects the previous block's CSV / joblib already exists.

**`No data returned for ticker`**
Yahoo Finance rate-limit or no internet. Wait a minute and retry.

**Chronos-2 fine-tune falls back to full fine-tuning** (Activity 1 only)
Install `peft`:
```bash
uv add peft
```

**Block E says `WARNING: no detector met precision floor on test`**
This is the documented outcome on the current test window (post-2022
regime). The pipeline falls back to the highest-recall detector and
reports it. See the §4.5 backtest summary for what the chosen
operating point actually does.
