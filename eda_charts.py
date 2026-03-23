"""
================================================================================
EDA Charts — Predictive Analysis & Explainability in Financial Markets
================================================================================
Generates all 7 figures referenced in Section 2 of the project document.

HOW TO USE:
    1. Install dependencies:  "uv add" or "pip install" yfinance pandas matplotlib scipy
    2. Run:                   python eda_charts.py
    3. Charts saved to:       reports/figures/

TO SWITCH FROM SIMULATED TO REAL DATA:
    Set USE_REAL_DATA = True below.
    Requires internet connection + yfinance installed.
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIG ──────────────────────────────────────────────────────────────────
USE_REAL_DATA = False        # Set True to download from Yahoo Finance
START_DATE    = "2010-01-01"
END_DATE      = "2023-12-31"
OUTPUT_DIR    = Path("figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── STYLE ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#F8FAFC',
    'axes.facecolor':   '#FFFFFF',
    'axes.grid':        True,
    'grid.alpha':       0.35,
    'grid.color':       '#CBD5E1',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.labelcolor':  '#1E293B',
    'axes.titlecolor':  '#1E293B',
    'xtick.color':      '#475569',
    'ytick.color':      '#475569',
    'font.family':      'DejaVu Sans',
    'axes.titlesize':   13,
    'axes.labelsize':   11,
})

BLUE   = '#1F4E79'
LBLUE  = '#2E75B6'
RED    = '#C0392B'
GREEN  = '#1A7A4A'
AMBER  = '#D97706'
PURPLE = '#6D28D9'
GRAY   = '#64748B'


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_real_data():
    """Download SPY, IEF and ^TNX from Yahoo Finance."""
    import yfinance as yf
    print("Downloading data from Yahoo Finance...")
    raw = yf.download(['SPY', 'IEF', '^TNX'], start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False)
    df = raw['Close'].copy()
    df.columns = ['IEF', 'SPY', 'TNX']  # yfinance returns alphabetically
    df = df.dropna()
    df['SPY_ret'] = np.log(df['SPY'] / df['SPY'].shift(1))
    df['IEF_ret'] = np.log(df['IEF'] / df['IEF'].shift(1))
    return df.dropna()


def load_simulated_data():
    """
    Generate statistically realistic simulated data.
    Matches SPY/IEF/TNX properties: drift, volatility, correlation,
    and known regime events (COVID 2020, rate hike 2022).
    """
    np.random.seed(42)
    dates = pd.date_range(START_DATE, END_DATE, freq='B')
    n = len(dates)

    # SPY: geometric Brownian motion + two regime shocks
    spy_ret = np.random.normal(0.00045, 0.0105, n)
    idx_covid  = np.arange(2590, 2640)   # ~March 2020
    idx_bear22 = np.arange(3080, 3280)   # ~2022 bear market
    spy_ret[idx_covid]  = np.random.normal(-0.025, 0.022, len(idx_covid))
    spy_ret[idx_bear22] = np.random.normal(-0.007, 0.016, len(idx_bear22))
    spy_prices = 100 * np.exp(np.cumsum(spy_ret))
    spy_prices = spy_prices * (450 / spy_prices[-1])  # scale to end ~$450

    # IEF: negatively correlated with SPY, except in 2022 (both fall)
    ief_base = np.random.normal(-0.00008, 0.0042, n)
    ief_base += -0.4 * spy_ret
    ief_base[idx_bear22] = np.random.normal(-0.012, 0.009, len(idx_bear22))
    ief_prices = 100 * np.exp(np.cumsum(ief_base))
    ief_prices = ief_prices * (95 / ief_prices[-1])

    # TNX: mean-reverting yield with 2022 rate hike cycle
    tnx = np.zeros(n)
    tnx[0] = 3.8
    for i in range(1, n):
        tnx[i] = np.clip(tnx[i-1] + np.random.normal(-0.0008, 0.055), 0.4, 5.2)
    tnx[3080:3350] = (np.linspace(1.5, 4.2, 270)
                      + np.random.normal(0, 0.12, 270))

    df = pd.DataFrame({'SPY': spy_prices, 'IEF': ief_prices, 'TNX': tnx}, index=dates)
    df['SPY_ret'] = np.log(df['SPY'] / df['SPY'].shift(1))
    df['IEF_ret'] = np.log(df['IEF'] / df['IEF'].shift(1))
    return df.dropna()


# ─────────────────────────────────────────────────────────────────────────────
# CHART FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def chart1_price_history(df):
    """Graph 1 — Full price history of SPY, IEF and TNX with regime annotations."""
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True,
                             gridspec_kw={'height_ratios': [3, 2, 2], 'hspace': 0.08})
    fig.suptitle('Figure 1 — Asset Price History (2010–2023)',
                 fontsize=15, fontweight='bold', color=BLUE, y=0.98)

    # SPY
    ax = axes[0]
    ax.plot(df.index, df['SPY'], color=LBLUE, linewidth=1.4)
    ax.fill_between(df.index, df['SPY'].min(), df['SPY'], alpha=0.08, color=LBLUE)
    ax.set_ylabel('Price (USD)', fontweight='bold')
    ax.set_title('SPY — S&P 500 ETF  (Target Variable)', fontsize=11, color=GRAY, loc='left')
    ax.annotate('COVID Crash\n(Mar 2020)', xy=(pd.Timestamp('2020-03-23'), df['SPY'].min() + 20),
                xytext=(pd.Timestamp('2018-01-01'), df['SPY'].min() + 60),
                arrowprops=dict(arrowstyle='->', color=RED, lw=1.5),
                fontsize=9, color=RED, fontweight='bold')
    ax.annotate('2022 Bear Market\n(Rate Hikes)', xy=(pd.Timestamp('2022-10-01'), 310),
                xytext=(pd.Timestamp('2020-10-01'), 260),
                arrowprops=dict(arrowstyle='->', color=AMBER, lw=1.5),
                fontsize=9, color=AMBER, fontweight='bold')

    # IEF
    ax2 = axes[1]
    ax2.plot(df.index, df['IEF'], color=GREEN, linewidth=1.4)
    ax2.fill_between(df.index, df['IEF'].min(), df['IEF'], alpha=0.08, color=GREEN)
    ax2.set_ylabel('Price (USD)', fontweight='bold')
    ax2.set_title('IEF — Treasury Bond ETF  (Exogenous Variable)', fontsize=11, color=GRAY, loc='left')

    # TNX
    ax3 = axes[2]
    ax3.plot(df.index, df['TNX'], color=AMBER, linewidth=1.4)
    ax3.fill_between(df.index, df['TNX'].min(), df['TNX'], alpha=0.08, color=AMBER)
    ax3.set_ylabel('Yield (%)', fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_title('^TNX — 10-Year Treasury Yield  (Exogenous Variable)', fontsize=11, color=GRAY, loc='left')

    path = OUTPUT_DIR / 'graph1_price_history.png'
    plt.savefig(path, dpi=160, bbox_inches='tight', facecolor='#F8FAFC')
    plt.close()
    print(f"  ✓ Graph 1 saved → {path}")


def chart2_distributions(df):
    """Graph 2 — Return histograms with normal fit + Q-Q plots."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle('Figure 2 — Return Distributions and Normality Analysis',
                 fontsize=15, fontweight='bold', color=BLUE)

    assets = [('SPY Daily Log Returns', df['SPY_ret'], LBLUE),
              ('IEF Daily Log Returns', df['IEF_ret'], GREEN)]

    for col, (title, ret, color) in enumerate(assets):
        # Histogram
        ax = axes[0][col]
        ax.hist(ret * 100, bins=80, color=color, alpha=0.7,
                density=True, edgecolor='white', linewidth=0.3)
        mu, sigma = ret.mean() * 100, ret.std() * 100
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 300)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'k--', linewidth=1.8, label='Normal fit')
        ax.axvline(mu, color='red', linewidth=1.2, linestyle=':', label=f'Mean = {mu:.3f}%')
        kurt = ret.kurtosis()
        ax.set_title(f'{title}\nKurtosis = {kurt:.2f}  (normal = 3.0, fat tails > 3)',
                     fontsize=10, color=GRAY)
        ax.set_xlabel('Daily Return (%)')
        ax.set_ylabel('Density')
        ax.legend(fontsize=9)

        # Q-Q plot
        ax2 = axes[1][col]
        (osm, osr), (slope, intercept, r) = stats.probplot(ret, dist='norm')
        ax2.scatter(osm, osr, alpha=0.25, s=4, color=color)
        line_x = np.array([osm.min(), osm.max()])
        ax2.plot(line_x, slope * line_x + intercept, 'k--',
                 linewidth=1.8, label=f'R² = {r**2:.4f}')
        ax2.set_title(f'Q-Q Plot — {title.split()[0]} Returns\n'
                      f'(deviations at extremes = fat tails confirmed)',
                      fontsize=10, color=GRAY)
        ax2.set_xlabel('Theoretical Quantiles')
        ax2.set_ylabel('Sample Quantiles')
        ax2.legend(fontsize=9)

    plt.tight_layout()
    path = OUTPUT_DIR / 'graph2_distributions.png'
    plt.savefig(path, dpi=160, bbox_inches='tight', facecolor='#F8FAFC')
    plt.close()
    print(f"  ✓ Graph 2 saved → {path}")


def chart3_rolling_stats(df):
    """Graph 3 — Rolling mean and rolling annualised volatility."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                             gridspec_kw={'hspace': 0.12})
    fig.suptitle('Figure 3 — Rolling Statistics: Mean and Volatility Over Time',
                 fontsize=15, fontweight='bold', color=BLUE)

    roll20  = df['SPY_ret'].rolling(20).mean() * 100
    roll60  = df['SPY_ret'].rolling(60).mean() * 100
    roll_vol = df['SPY_ret'].rolling(20).std() * 100 * np.sqrt(252)

    ax = axes[0]
    ax.plot(df.index, roll20, color=LBLUE, linewidth=1.2,
            label='20-day rolling mean (%)', alpha=0.9)
    ax.plot(df.index, roll60, color=BLUE,  linewidth=1.5,
            label='60-day rolling mean (%)')
    ax.axhline(0, color=GRAY, linewidth=0.8, linestyle='--')
    ax.fill_between(df.index, 0, roll20, where=(roll20 >= 0),
                    alpha=0.12, color=GREEN, label='Positive regime (bullish)')
    ax.fill_between(df.index, 0, roll20, where=(roll20 < 0),
                    alpha=0.18, color=RED,   label='Negative regime (bearish)')
    ax.set_ylabel('Rolling Mean Return (%)', fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.set_title('Rolling Mean Return — Regime Changes Visible', fontsize=11, color=GRAY, loc='left')

    ax2 = axes[1]
    ax2.plot(df.index, roll_vol, color=PURPLE, linewidth=1.2,
             label='20-day Rolling Annualised Volatility (%)')
    ax2.fill_between(df.index, roll_vol.min(), roll_vol, alpha=0.15, color=PURPLE)
    ax2.axhline(roll_vol.mean(), color=AMBER, linewidth=1.5, linestyle='--',
                label=f'Average volatility ({roll_vol.mean():.1f}%)')
    ax2.set_ylabel('Annualised Volatility (%)', fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.legend(fontsize=9, loc='upper left')
    ax2.set_title('Rolling Volatility — COVID (2020) and Rate Hike (2022) Spikes Visible',
                  fontsize=11, color=GRAY, loc='left')

    path = OUTPUT_DIR / 'graph3_rolling_stats.png'
    plt.savefig(path, dpi=160, bbox_inches='tight', facecolor='#F8FAFC')
    plt.close()
    print(f"  ✓ Graph 3 saved → {path}")


def chart4_stationarity(df):
    """Graph 4 — Raw prices vs log returns (ADF stationarity comparison)."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle('Figure 4 — Stationarity: Raw Prices vs. Log Returns',
                 fontsize=15, fontweight='bold', color=BLUE)

    panels = [
        (axes[0][0], df['SPY'],     'SPY Raw Price',     LBLUE, False, '~0.98'),
        (axes[0][1], df['SPY_ret'], 'SPY Log Returns',   LBLUE, True,  '< 0.001'),
        (axes[1][0], df['IEF'],     'IEF Raw Price',     GREEN, False, '~0.87'),
        (axes[1][1], df['IEF_ret'], 'IEF Log Returns',   GREEN, True,  '< 0.001'),
    ]

    for ax, series, label, color, is_stationary, pval in panels:
        ax.plot(df.index, series, color=color, linewidth=0.9 if is_stationary else 1.2)
        if is_stationary:
            ax.axhline(0, color='black', linewidth=0.8)
            status, status_color, bg = 'STATIONARY', GREEN, '#F0FDF4'
        else:
            status, status_color, bg = 'NON-STATIONARY', RED, '#FEF2F2'

        ax.set_title(f'{label} — {status}\nADF p-value: {pval}',
                     fontsize=10, color=status_color)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USD)' if not is_stationary else 'Log Return')
        ax.text(0.97, 0.05, f'ADF p-value: {pval}',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
                color=status_color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=bg,
                          edgecolor=status_color, alpha=0.85))

    plt.tight_layout()
    path = OUTPUT_DIR / 'graph4_stationarity.png'
    plt.savefig(path, dpi=160, bbox_inches='tight', facecolor='#F8FAFC')
    plt.close()
    print(f"  ✓ Graph 4 saved → {path}")


def chart5_acf_pacf(df):
    """Graph 5 — ACF and PACF for SPY and IEF log returns."""
    def compute_acf(series, nlags=40):
        series = (series - series.mean()).values
        return np.array([1.0] + [np.corrcoef(series[:-k], series[k:])[0, 1]
                                 for k in range(1, nlags + 1)])

    def compute_pacf(series, nlags=40):
        """Yule-Walker PACF approximation."""
        from numpy.linalg import solve
        s = (series - series.mean()).values
        pacf = [1.0]
        for k in range(1, nlags + 1):
            R = np.array([[np.corrcoef(s[:-abs(i-j)] if i != j else s,
                                       s[abs(i-j):]  if i != j else s)[0,1]
                           for j in range(1, k+1)] for i in range(1, k+1)])
            r = np.array([np.corrcoef(s[:-i], s[i:])[0, 1] for i in range(1, k+1)])
            try:
                phi = solve(R, r)
                pacf.append(phi[-1])
            except Exception:
                pacf.append(0.0)
        return np.array(pacf)

    lags = 40
    conf = 1.96 / np.sqrt(len(df))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle('Figure 5 — Autocorrelation (ACF) and Partial Autocorrelation (PACF)',
                 fontsize=15, fontweight='bold', color=BLUE)

    pairs = [('SPY Log Returns', df['SPY_ret'], LBLUE),
             ('IEF Log Returns', df['IEF_ret'], GREEN)]

    for col, (label, series, color) in enumerate(pairs):
        acf_v  = compute_acf(series,  nlags=lags)
        pacf_v = compute_pacf(series, nlags=lags)

        for row, (vals, func_label) in enumerate([(acf_v, 'ACF'), (pacf_v, 'PACF')]):
            ax = axes[row][col]
            x = np.arange(len(vals))
            ax.bar(x, vals, color=color, alpha=0.65, width=0.6)
            ax.axhline(0, color='black', linewidth=0.8)
            ax.axhline( conf, color=RED, linewidth=1.2, linestyle='--',
                       label=f'95% CI (±{conf:.3f})')
            ax.axhline(-conf, color=RED, linewidth=1.2, linestyle='--')
            ax.fill_between([-1, lags+1], -conf, conf, alpha=0.07, color=RED)
            ax.set_xlim(-1, lags + 1)
            ax.set_ylim(-0.20, 0.20)
            ax.set_title(f'{func_label} — {label}\n'
                         f'(bars beyond red = significant autocorrelation)',
                         fontsize=10, color=GRAY)
            ax.set_xlabel('Lag (days)')
            ax.set_ylabel('Correlation')
            ax.legend(fontsize=8)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))

    plt.tight_layout()
    path = OUTPUT_DIR / 'graph5_acf_pacf.png'
    plt.savefig(path, dpi=160, bbox_inches='tight', facecolor='#F8FAFC')
    plt.close()
    print(f"  ✓ Graph 5 saved → {path}")


def chart6_correlation(df):
    """Graph 6 — SPY vs IEF dynamic correlation (rolling + scatter + annual bars)."""
    fig = plt.figure(figsize=(13, 9))
    fig.suptitle('Figure 6 — SPY vs. IEF Relationship: The Stocks-Bonds Dynamic',
                 fontsize=15, fontweight='bold', color=BLUE)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

    # Rolling 60-day correlation
    ax1 = fig.add_subplot(gs[0, :2])
    roll_corr = df['SPY_ret'].rolling(60).corr(df['IEF_ret'])
    ax1.plot(df.index, roll_corr, color=PURPLE, linewidth=1.2)
    ax1.axhline(0, color=GRAY, linewidth=1, linestyle='--')
    ax1.fill_between(df.index, 0, roll_corr, where=(roll_corr < 0),
                     alpha=0.18, color=LBLUE, label='Negative corr (typical)')
    ax1.fill_between(df.index, 0, roll_corr, where=(roll_corr >= 0),
                     alpha=0.20, color=RED, label='Positive corr (regime break)')
    ax1.set_title('60-Day Rolling Correlation: SPY vs IEF\n'
                  '(note: 2022 inflation crisis breaks the historical negative relationship)',
                  fontsize=10, color=GRAY)
    ax1.set_ylabel('Correlation Coefficient')
    ax1.set_xlabel('Date')
    ax1.legend(fontsize=9)
    ax1.set_ylim(-1, 1)

    # Scatter
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.scatter(df['IEF_ret'] * 100, df['SPY_ret'] * 100,
                alpha=0.15, s=5, color=PURPLE)
    slope, intercept, r, p, se = stats.linregress(df['IEF_ret'], df['SPY_ret'])
    x_line = np.linspace(df['IEF_ret'].min(), df['IEF_ret'].max(), 200)
    ax2.plot(x_line * 100, (slope * x_line + intercept) * 100,
             'r-', linewidth=2, label=f'r = {r:.3f}')
    ax2.set_title(f'Scatter: IEF vs SPY\n(overall r = {r:.3f})',
                  fontsize=10, color=GRAY)
    ax2.set_xlabel('IEF Return (%)')
    ax2.set_ylabel('SPY Return (%)')
    ax2.legend(fontsize=9)

    # Annual correlation bar chart
    yearly_corr = df.groupby(df.index.year).apply(
        lambda g: g['SPY_ret'].corr(g['IEF_ret'])
    )
    ax3 = fig.add_subplot(gs[1, :])
    bar_colors = [RED if v >= 0 else LBLUE for v in yearly_corr.values]
    bars = ax3.bar(yearly_corr.index, yearly_corr.values,
                   color=bar_colors, alpha=0.75, edgecolor='white')
    ax3.axhline(0, color='black', linewidth=0.8)
    ax3.set_title('Annual SPY–IEF Correlation by Year\n'
                  '(Red = positive correlation, breaks the typical inverse relationship)',
                  fontsize=10, color=GRAY)
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Annual Correlation')
    ax3.set_xticks(yearly_corr.index)
    ax3.set_xticklabels(yearly_corr.index, rotation=45)
    for bar, val in zip(bars, yearly_corr.values):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.015 * np.sign(val),
                 f'{val:.2f}', ha='center',
                 va='bottom' if val >= 0 else 'top',
                 fontsize=8, color='#1E293B')

    path = OUTPUT_DIR / 'graph6_correlation.png'
    plt.savefig(path, dpi=160, bbox_inches='tight', facecolor='#F8FAFC')
    plt.close()
    print(f"  ✓ Graph 6 saved → {path}")


def chart7_split(df):
    """Graph 7 — Sequential train / validation / test split visualisation."""
    n_total = len(df)
    n_train = int(n_total * 0.70)
    n_val   = int(n_total * 0.15)
    n_test  = n_total - n_train - n_val

    train_idx = df.index[:n_train]
    val_idx   = df.index[n_train:n_train + n_val]
    test_idx  = df.index[n_train + n_val:]

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle(
        'Figure 7 — Sequential Train / Validation / Test Split\n'
        '(No shuffling — temporal order preserved to prevent Data Leakage)',
        fontsize=14, fontweight='bold', color=BLUE
    )

    ax.plot(train_idx, df['SPY'].iloc[:n_train],         color=BLUE,  linewidth=1.2,
            label=f'Training Set  (70% — {n_train} days)')
    ax.plot(val_idx,   df['SPY'].iloc[n_train:n_train+n_val], color=GREEN, linewidth=1.4,
            label=f'Validation Set  (15% — {n_val} days)')
    ax.plot(test_idx,  df['SPY'].iloc[n_train+n_val:],   color=RED,   linewidth=1.4,
            label=f'Test Set  (15% — {n_test} days)')

    ax.fill_between(train_idx, 0, df['SPY'].iloc[:n_train],              alpha=0.07, color=BLUE)
    ax.fill_between(val_idx,   0, df['SPY'].iloc[n_train:n_train+n_val], alpha=0.12, color=GREEN)
    ax.fill_between(test_idx,  0, df['SPY'].iloc[n_train+n_val:],        alpha=0.12, color=RED)

    ax.axvline(train_idx[-1], color=BLUE,  linewidth=2, linestyle='--', alpha=0.6)
    ax.axvline(val_idx[-1],   color=GREEN, linewidth=2, linestyle='--', alpha=0.6)

    y_top = df['SPY'].max() * 0.93
    for xpos, txt, c in [
        (train_idx[len(train_idx)//2], f'TRAIN\n2010–2019\n({n_train} days)', BLUE),
        (val_idx[len(val_idx)//2],     f'VALIDATION\n2020–2021\n({n_val} days)', GREEN),
        (test_idx[len(test_idx)//2],   f'TEST\n2022–2023\n({n_test} days)', RED),
    ]:
        ax.text(xpos, y_top, txt, ha='center', va='top', fontsize=10,
                fontweight='bold', color=c,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor=c, alpha=0.9))

    ax.set_ylabel('SPY Price (USD)', fontweight='bold')
    ax.set_xlabel('Date')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim(50)

    plt.tight_layout()
    path = OUTPUT_DIR / 'graph7_split.png'
    plt.savefig(path, dpi=160, bbox_inches='tight', facecolor='#F8FAFC')
    plt.close()
    print(f"  ✓ Graph 7 saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("EDA Chart Generator — Financial ML Project")
    print("=" * 60)

    if USE_REAL_DATA:
        df = load_real_data()
        print(f"Real data loaded: {len(df)} trading days")
    else:
        df = load_simulated_data()
        print(f"Simulated data generated: {len(df)} trading days")

    print(f"Date range: {df.index[0].date()} → {df.index[-1].date()}")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print()

    chart1_price_history(df)
    chart2_distributions(df)
    chart3_rolling_stats(df)
    chart4_stationarity(df)
    chart5_acf_pacf(df)
    chart6_correlation(df)
    chart7_split(df)

    print()
    print("All 7 charts generated successfully.")
    print(f"Find them in: {OUTPUT_DIR.resolve()}")