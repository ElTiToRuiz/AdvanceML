# Predictive Analysis & Explainability in Financial Markets

**Authors:** Igor Ruiz · Jon Vargas · Mikel Sanchez
**Course:** Advanced Machine Learning · 2026

---

A two-activity research project on machine learning for daily SPY returns
(2003 – 2026). We test whether returns are predictable in two complementary
ways — regression on the magnitude (Activity 1) and classification of the
regime (Activity 2) — and we close with an honest economic backtest.

> **One-line conclusion.** The market is *weakly inefficient*: just inefficient enough to reward
> risk management, not enough to reward pure timing.

## Key results at a glance

| Question                                                          | Best score                          | Honest reading                                              |
|-------------------------------------------------------------------|-------------------------------------|-------------------------------------------------------------|
| **Act 1.** Will SPY go up tomorrow?                               | 54.9 % directional acc. (LSTM)      | Loses to "always-up" baseline (55.5 %). EMH confirmed.      |
| **Act 2 — Phase 2.** Which regime does tomorrow belong to?        | macro-F1 = **0.265** (XGBoost)      | +60 % over the 0.167 baseline — but crash recall only 4.8 %.|
| **Act 2 — Block E.** Can we detect crashes specifically?          | recall_crash = **61.9 %** on test   | 13 × lift, but precision drops to 3.4 % under post-2022 regime. |
| **Block E backtest.** Does the crash signal beat buy-and-hold?    | $1 → $1.36 vs. $1.82 buy-and-hold   | Drawdown −12.6 % vs −18.8 % — risk reduced, return given up.|

## Table of contents

1. [Quick start](#quick-start)
2. [Repository layout](#repository-layout)
3. [Activity 1 — forecasting](#activity-1--forecasting)
4. [Activity 2 — regime classification](#activity-2--regime-classification)
   - [Phase 2 · Blocks A → D](#phase-2--blocks-a--d-the-original-plan)
   - [Block E · crash-focused extension](#block-e--crash-focused-extension)
5. [How the two activities relate](#how-the-two-activities-relate)
6. [Design decisions](#design-decisions)
7. [Reproducibility](#reproducibility)
8. [Common errors](#common-errors)

---

## Quick start

Requires **Python ≥ 3.13** and [`uv`](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/ElTiToRuiz/AdvanceML.git
cd AdvanceML
uv venv && source .venv/bin/activate
uv sync

python main.py 1     # Activity 1 — forecasting       (~5 min)
python main.py 2     # Activity 2 — A → B → C → D → E (~10 min)
```

Each activity downloads its own data, builds its EDA charts, runs every
experiment, and writes the results to `reports/activity{1,2}/`. The full
pipeline is idempotent — running it twice on the same git commit produces
identical numbers.

### Run individual blocks

```bash
# Activity 1
python -m src.activity1.pipelines.run_baselines
python -m src.activity1.pipelines.run_sarimax
python -m src.activity1.pipelines.run_lstm
python -m src.activity1.pipelines.run_chronos
python -m src.activity1.pipelines.tune_lstm           # grid search (~3 min)
python -m src.activity1.pipelines.run_multi_horizon   # 1d / 5d / 30d
python -m src.activity1.pipelines.finetune_chronos    # ~15 min on CPU
python -m src.activity1.pipelines.run_all             # leaderboard

# Activity 2 — each block reads the previous winner from disk
python -m src.activity2.pipelines.compare_imputations  # Block A
python -m src.activity2.pipelines.compare_imbalance    # Block B
python -m src.activity2.pipelines.tune_models          # Block C
python -m src.activity2.pipelines.final_evaluation     # Block D + SHAP
python -m src.activity2.pipelines.crash_focus          # Block E
python -m src.activity2.pipelines.run_all              # all + manifest.json
```

### Build the slide decks

```bash
uv run python scripts/build_activity1_pptx.py   # → reports/activity1/presentation/
uv run python scripts/build_activity2_pptx.py   # → reports/activity2/presentation/
```

---

## Repository layout

```
project/
├── main.py                 # python main.py {1,2}
├── pyproject.toml · uv.lock
│
├── src/
│   ├── activity1/          # forecasting (regression)
│   │   ├── data/           # download · clean · loader
│   │   ├── eda/            # 7 charts
│   │   ├── models/         # baselines · sarimax · lstm · chronos
│   │   ├── evaluation/     # metrics · plots · backtesting · style
│   │   └── pipelines/      # one script per experiment + run_all
│   │
│   └── activity2/          # multi-class classification (5 blocks)
│       ├── data/           # download · clean · loader
│       ├── eda/            # 11 charts (imbalance/imputation themed)
│       ├── preprocessing/  # imputation · imbalance
│       ├── models/         # base (Classifier ABC) · baselines · logreg · rf · xgboost
│       ├── evaluation/     # metrics · plots · style · shap_explainer · operational
│       └── pipelines/      # compare_imputations · compare_imbalance ·
│                           # tune_models · final_evaluation · crash_focus · run_all
│
├── data/                   # (gitignored) raw + processed CSVs
└── reports/                # (gitignored) charts, metrics, manifest, .pptx
```

### Main libraries

| Library                | Purpose                                              |
|------------------------|------------------------------------------------------|
| `yfinance`             | Historical price data                                |
| `pandas`, `numpy`      | Data wrangling                                       |
| `scipy`, `statsmodels` | ADF · ACF/PACF · SARIMAX (Activity 1)                |
| `torch`                | LSTM forecaster                                      |
| `chronos-forecasting`  | Amazon Chronos-2 foundation model                    |
| `scikit-learn`         | LogReg · RF · KNN · MICE · calibration · class weights |
| `xgboost`              | Multi-class & binary gradient-boosted trees          |
| `imbalanced-learn`     | SMOTE · ADASYN · SMOTE+ENN · RandomUnderSampler      |
| `optuna`               | TPE-Bayesian hyperparameter search                   |
| `shap`                 | Tree SHAP feature importance                         |
| `joblib`               | Model persistence                                    |
| `python-pptx`          | PPTX deck generators                                 |

Optional for proper LoRA fine-tuning of Chronos-2 (Activity 1):
```bash
uv add peft
```

---

## Activity 1 — forecasting

**Target.** Daily log-return of SPY (5 800 trading days, Jan 2003 – Mar 2026).
**Exogenous.** 10-year Treasury yield change (^TNX). IEF was dropped — perfectly
anti-correlated with ^TNX (r ≈ −0.95) and not the causal driver.
**Split.** Sequential 70 / 15 / 15: train (2003-2019) · val (2019-2022) · test
(2022-2026). The test window covers the post-2022 Fed rate-hike regime.

### `Forecaster` API

Every model implements the same interface, enforcing leak-free evaluation
at the code level:

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

### Test-set leaderboard

| Model              | Dir. Acc. | Skill   | IC Pearson | Sharpe   | Equity 3.5 y |
|--------------------|-----------|---------|------------|----------|--------------|
| **LSTM**           | 54.9 %    | −0.6 pp | **+0.10**  | **1.33** | **2.11 ×**   |
| drift_mean         | 55.5 %    |  0.0 pp | −0.05      | 1.05     | 1.81 ×       |
| chronos2 (TNX)     | 55.2 %    | −0.4 pp | −0.07      | 0.34     | 1.21 ×       |
| ma_20              | 52.9 %    | −2.6 pp | −0.03      | 0.52     | 1.35 ×       |
| sarimax (TNX)      | 46.9 %    | −8.6 pp | −0.03      | −0.22    | 0.88 ×       |

The "always-up" baseline scores 55.5 % directional accuracy — exactly the
percentage of up-days in the test window. **No model achieves a
statistically meaningful skill above this drift baseline**, an empirical
confirmation of the Efficient Market Hypothesis on daily horizons. The
LSTM does extract signal in MAGNITUDE (IC 0.10, Sharpe 1.33), explaining
its superior P&L while directional skill stays near zero. SARIMAX learns
a SPY–TNX relationship in 2003-2019 that breaks under the post-2022 Fed
regime — confirmed by three independent regime-shift indicators across
SARIMAX, LSTM-with-TNX and fine-tuned Chronos-2.

---

## Activity 2 — regime classification

Multi-class classification of next-day SPY into **crash · correction · normal · rally**
based on log-return cutoffs (−2 %, −0.5 %, +0.5 %). The asset universe is
widened with four extra macro signals: `^VIX` (fear), `GLD` (gold safe-haven),
`UUP` (US dollar), `USO` (oil). GLD/USO/UUP launched in 2004/2006/2007,
producing real structural missing data that motivates the imputation comparison.

The deliverable is split into **five blocks (A → E)**. The first four
implement the original planning document literally; Block E is an additive
crash-focused extension that asks "what is the maximum recall on the rare
class we can actually achieve, and what is it worth in dollars?".

```
        ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
data →  │ Block A │ →  │ Block B │ →  │ Block C │ →  │ Block D │ →  │ Block E │
EDA →   │imputation│   │imbalance│    │  models │    │  test + │    │  crash  │
        │         │    │         │    │         │    │  SHAP   │    │ focused │
        └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘
        6 methods      7 strategies   3 families     macro-F1 +    calibrate +
        → linear       → class_weight → XGBoost      Tree SHAP     retune +
                                                                   binary +
                                                                   backtest
```

### Phase 2 — Blocks A → D (the original plan)

#### Block A — imputation comparison

Six methods compared on val macro-F1 with a fixed `LogReg(class_weight='balanced')`:

| Method                | Val macro-F1 |
|-----------------------|--------------|
| linear (winner)       | **0.2781**   |
| ffill                 | 0.2767       |
| median                | 0.2608       |
| mean                  | 0.2588       |
| MICE                  | 0.2587       |
| KNN (k = 5)           | 0.2565       |
| linear + MNAR ind.    | 0.1737       |

The six imputers cluster within 2 pp of each other — on this dataset the
imputation choice barely matters. **The MNAR-indicator variant hurts**:
GLD/USO/UUP missingness is concentrated entirely in 2003-2007 (only train),
so the indicator column is constant 0 in val and acts as feature noise. A
genuine empirical finding about MNAR indicators on chronologically-split
datasets.

#### Block B — imbalance comparison

Seven strategies on top of the winning imputation:

| Strategy                          | Val macro-F1 |
|-----------------------------------|--------------|
| `class_weight='balanced'` (winner)| **0.2778**   |
| SMOTE                             | 0.2740       |
| ADASYN                            | 0.2694       |
| RandomUnderSampler                | 0.2660       |
| threshold tuning                  | 0.2085       |
| untouched                         | 0.2050       |
| SMOTE + ENN                       | **0.0965**   |

SMOTE + ENN collapses — the ENN cleaning step removes too many borderline
points after synthetic generation. Threshold tuning alone is weak because
it starts from an untouched model with near-zero mass on the rare classes.

#### Block C — model tuning

Three models with `(linear, class_weight='balanced')`:

| Model         | Search method                  | Val macro-F1 |
|---------------|--------------------------------|--------------|
| **XGBoost**   | Optuna (TPE + MedianPruner, 200 trials) | **0.3558** |
| Random Forest | Randomised search (50 iter)    | 0.3318       |
| LogReg        | Grid search                    | 0.2778       |

XGBoost picks up non-linear interactions LogReg cannot.

#### Block D — test set + Tree SHAP

| Metric                | Value      |
|-----------------------|------------|
| macro-F1              | 0.2653     |
| balanced accuracy     | 0.2690     |
| MCC                   | 0.0484     |
| G-Mean                | 0.1834     |
| accuracy              | 0.3993     |
| recall_crash          | **0.0476** |
| recall_correction     | 0.1272     |
| recall_normal         | 0.5779     |
| recall_rally          | 0.3232     |

The val→test drop (0.356 → 0.265) is the same post-2022 regime shift
already documented in Activity 1. **Tree SHAP** says the top features for
crash prediction are UUP_ret (dollar), today's SPY return and USO_ret (oil)
— **VIX is not the primary feature**, contradicting the folk-finance
intuition.

### Block E — crash-focused extension

Macro-F1 weights all four classes equally, which dilutes what we actually
care about: detecting crashes ahead of time. Block E reformulates the
problem around `recall_crash` subject to a precision floor
(`PRECISION_FLOOR_CRASH = 0.10`), without modifying any Phase 2 artifact.

**Five sub-analyses:**

1. **Probability calibration.** Platt scaling on top of the Phase 2 XGBoost
   using val. The reliability diagram shifts the predicted-vs-empirical
   curve closer to the diagonal.

2. **Asymmetric-cost re-tuning.** New Optuna search (100 trials) optimising
   `recall_crash` with `precision_crash >= 0.10` enforced via score-0
   penalty. Sample weights: `crash 10×, correction 3×, rally 2×, normal 1×`.
   Val best score = **0.6863**.

3. **Threshold tuning + PR-curve overlay** across the three crash detectors
   (calibrated, asymmetric, binary). Saved to
   `reports/activity2/crash_focus/crash_pr_curve.png`.

4. **Binary reframe** (`crash` vs `not-crash`, `XGBClassifier` with
   `scale_pos_weight = n_neg / n_pos`). 100 Optuna trials, same
   precision-floor-constrained objective. Val best = **0.6863** — matches
   the asymmetric retune ceiling on val.

5. **Long-only "exit on crash signal" backtest** vs buy-and-hold over the
   873-day test window.

#### Operating points on test

| Detector            | Threshold | Val recall | Val prec. | Test recall | Test prec. | Alarms |
|---------------------|-----------|------------|-----------|-------------|------------|--------|
| calibrated_xgb      | 0.061     | 0.471      | 0.10      | 0.190       | 0.017      | 234    |
| asymmetric_xgb      | 0.019     | 0.686      | 0.10      | 0.476       | 0.026      | 381    |
| **binary_xgb**      | 0.084     | 0.686      | 0.10      | **0.619**   | 0.034      | 384    |

**No detector held the precision floor on test** (best test precision = 3.4 %).
The fallback to "highest-recall detector" picked **binary_xgb**. This is
the post-2022 regime-shift cost — not a code bug.

#### Backtest (873 trading days, Sept 2022 → Mar 2026)

|                              | "Exit on signal"   | Buy & hold     |
|------------------------------|--------------------|----------------|
| Final equity ($1 start)      | 1.36               | **1.82**       |
| Annualised return            | 9.2 %              | **19.0 %**     |
| Sharpe ratio                 | 0.87               | **1.06**       |
| **Max drawdown**             | **−12.6 %**        | −18.8 %        |
| Crashes evaded / missed      | 8 / 13             | —              |
| False alarms / signals       | 375 / 383 (98 %)   | —              |

The strategy avoids 8 of 21 real crashes and reduces max drawdown by 6
percentage points relative to buy-and-hold. But 98 % of its signals are
false alarms, and the opportunity cost on the normal-up days drags the
final equity 26 % below buy-and-hold. **The signal has predictive power,
but not enough to be net profitable — exactly the EMH fingerprint from
Activity 1.**

---

## How the two activities relate

| Layer                          | Question                                        | Outcome                                       |
|--------------------------------|-------------------------------------------------|-----------------------------------------------|
| **Act 1 — regression**         | Will SPY go up or down tomorrow?                | Random walk + drift. No model beats the constant. |
| **Act 2 — Phase 2**            | Which regime does tomorrow belong to?           | Real signal exists (macro-F1 + 60 %), but rare-class recall stays low. |
| **Act 2 — Block E**            | Specifically, can we detect crashes?            | Yes, 13 × better than chance — but precision drops sharply on test. |
| **Block E — backtest**         | Does the crash signal beat buy-and-hold?        | Drawdown reduced, but total return suffers more from false alarms. |

Three independent measurements converge on one conclusion: **the market is
weakly inefficient**. There is exploitable information in macro stress
signals, but the 1-day horizon is too noisy for the precision required to
convert it into alpha. The pricing of risk management (lower drawdown)
works; the pricing of pure timing (higher Sharpe) does not.

---

## Design decisions

A short list of methodological choices, each one defendable against the
rubric's "why have I implemented each decision/component that way?":

- **Chronological 70 / 15 / 15 split, no k-fold cross-validation.**
  k-fold would shuffle time and leak the future into the past on a
  time-series problem. Hyperparameter search uses
  `sklearn.model_selection.PredefinedSplit(train, val)` instead.
- **Stratification deliberately not used.** Stratifying by class would
  break the chronological order and is therefore the *exact* temporal
  leakage the class notes warn about.
- **Macro-F1 as primary multi-class metric, accuracy reported only for
  completeness.** With class proportions of 4 / 18 / 50 / 28 %, a constant
  predictor scores ~50 % accuracy with crash recall 0.
- **Tree SHAP rather than LIME or PDP for the tree-ensemble winner.**
  The class notes recommend Tree SHAP for tree models because it gives
  exact Shapley values in polynomial time. PDP would assume features are
  uncorrelated, which is false in finance (SPY and VIX are anti-correlated).
- **Block E adds analyses, never modifies Phase 2 artefacts.** macro-F1 is
  the metric of record; recall_crash is reported as a parallel,
  operationally relevant view. Phase 2 + Block E together is the deliverable.
- **Imputers fit on train only; resamplers act on train only; calibration
  fits on val.** This is the standard data-leakage prevention for the
  preprocessing-leakage failure mode in the class notes.
- **Asymmetric cost weights (crash 10 ×) instead of `class_weight='balanced'`
  in Block E.** Balanced weights treat one missed crash as equivalent to
  one missed rally — wrong for risk management. The 10 × weight encodes
  the operational asymmetry directly.
- **Single-author commits, modular package, manifest.json.** Reproducibility
  is treated as a deliverable: identical commit + identical lock file →
  identical numbers.

---

## Reproducibility

`reports/activity2/manifest.json` is written by `pipelines/run_all.py` and
captures: ISO timestamp, `git rev-parse HEAD`, the SHA-256 of `config.py`,
and the installed versions of every ML library used (`scikit-learn`,
`xgboost`, `imbalanced-learn`, `optuna`, `shap`, `pandas`, `numpy`,
`joblib`). Identical commit + identical lock file → identical numbers.

A representative output from a recent run:

```json
{
  "timestamp_utc":   "2026-04-30T...",
  "git_commit":      "ab16c9d...",
  "python":          "3.13.x",
  "config_sha256":   "...",
  "library_versions": {
    "sklearn": "1.8.0", "xgboost": "3.2.0", "imblearn": "0.14.1",
    "optuna":  "4.8.0", "shap":    "0.51.0", "pandas":  "3.0.1",
    "numpy":   "2.4.3", "joblib":  "1.5.3"
  }
}
```

---

## Common errors

**`ModuleNotFoundError: No module named 'X'`**
```bash
uv sync          # re-install everything from uv.lock
```

**`FileNotFoundError: full_dataset.csv` (or `winner_bundle.joblib`)**
You skipped a step. Run `python main.py 1` (or `2`) to do everything in
order, or run the upstream block(s) first. Each Activity 2 pipeline
expects the previous block's CSV / `joblib` to already exist under
`reports/activity2/`.

**`No data returned for ticker`**
Yahoo Finance rate-limit or no internet. Wait a minute and retry.

**Chronos-2 fine-tune falls back to full fine-tuning** (Activity 1 only)
```bash
uv add peft
```

**Block E says `WARNING: no detector met precision floor on test`**
Documented outcome on the current test window (post-2022 regime). The
pipeline falls back to the highest-recall detector and reports it
explicitly. See the §4.5 backtest summary for what the chosen operating
point actually does.
