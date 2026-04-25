# plots.py
# ─────────────────────────────────────────────────────────────────────────────
# Generates the 7 EDA figures from the cleaned full_dataset.csv.
# Figures saved to reports/activity1/ as high-resolution PNGs.
#
# Run from project root:
#     python -m src.activity1.eda.plots
# ─────────────────────────────────────────────────────────────────────────────

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from scipy import stats
from statsmodels.tsa.stattools import adfuller, acf, pacf

from ..config import PROCESSED_DIR, FIGURES_DIR, TRAIN_RATIO, VAL_RATIO

os.makedirs(FIGURES_DIR, exist_ok=True)

# ─── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  '#F8FAFC',
    'axes.facecolor':    '#FFFFFF',
    'axes.grid':         True,
    'grid.alpha':        0.35,
    'grid.color':        '#CBD5E1',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.labelcolor':   '#1E293B',
    'axes.titlecolor':   '#1E293B',
    'xtick.color':       '#475569',
    'ytick.color':       '#475569',
    'font.family':       'DejaVu Sans',
    'axes.titlesize':    13,
    'axes.labelsize':    11,
    'legend.framealpha': 0.9,
    'legend.fontsize':   9,
})

BLUE   = '#1F4E79'
LBLUE  = '#2E75B6'
RED    = '#C0392B'
GREEN  = '#1A7A4A'
AMBER  = '#D97706'
PURPLE = '#6D28D9'
GRAY   = '#64748B'


# ─── Load data ────────────────────────────────────────────────────────────────

def load() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "full_dataset.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    print(f"Loaded full_dataset.csv  →  {len(df)} rows  "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    return df


def save_fig(fig, name: str) -> None:
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=160, bbox_inches='tight', facecolor='#F8FAFC')
    plt.close(fig)
    print(f"  ✓  {name}")


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 1 — Full price history
# ─────────────────────────────────────────────────────────────────────────────

def graph1_price_history(df: pd.DataFrame) -> None:
    """
    Three-panel chart showing the complete price/yield history of SPY, IEF
    and ^TNX from the earliest available date to 2026-03-01.
    Annotates major market events visible in the data.
    """
    fig, axes = plt.subplots(
        3, 1, figsize=(14, 11), sharex=True,
        gridspec_kw={'height_ratios': [3, 2, 2], 'hspace': 0.08}
    )
    fig.suptitle(
        f'Graph 1 — Asset Price & Yield History  '
        f'({df.index[0].year}–{df.index[-1].year})',
        fontsize=15, fontweight='bold', color=BLUE, y=0.99
    )

    # ── SPY ──────────────────────────────────────────
    ax = axes[0]
    ax.plot(df.index, df['SPY_price'], color=LBLUE, linewidth=1.3,
            label='SPY (S&P 500 ETF)')
    ax.fill_between(df.index, df['SPY_price'].min() * 0.95,
                    df['SPY_price'], alpha=0.07, color=LBLUE)
    ax.set_ylabel('Price (USD)', fontweight='bold')
    ax.set_title('SPY — S&P 500 ETF  |  Target Variable', fontsize=11,
                 color=GRAY, loc='left')
    ax.legend(loc='upper left')

    # Key event annotations
    events = [
        ('2008-10-10', 'GFC\n2008',      RED,   0.60),
        ('2020-03-23', 'COVID\n2020',     RED,   0.35),
        ('2022-10-12', '2022\nBear',      AMBER, 0.50),
    ]
    y_range = df['SPY_price'].max() - df['SPY_price'].min()
    for date_str, label, color, y_frac in events:
        ts = pd.Timestamp(date_str)
        if ts in df.index or df.index[df.index.searchsorted(ts) - 1]:
            yval = df['SPY_price'].min() + y_range * y_frac
            ax.axvline(ts, color=color, linewidth=1, linestyle=':', alpha=0.7)
            ax.text(ts, df['SPY_price'].max() * 0.98, label,
                    ha='center', va='top', fontsize=8,
                    color=color, fontweight='bold')

    # ── IEF ──────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(df.index, df['IEF_price'], color=GREEN, linewidth=1.3,
             label='IEF (7-10Y Treasury Bond ETF)')
    ax2.fill_between(df.index, df['IEF_price'].min() * 0.95,
                     df['IEF_price'], alpha=0.07, color=GREEN)
    ax2.set_ylabel('Price (USD)', fontweight='bold')
    ax2.set_title('IEF — Treasury Bond ETF  |  Exogenous Variable',
                  fontsize=11, color=GRAY, loc='left')
    ax2.legend(loc='upper left')

    # ── TNX yield ────────────────────────────────────
    ax3 = axes[2]
    ax3.plot(df.index, df['TNX_yield'], color=AMBER, linewidth=1.3,
             label='10-Year Treasury Yield (%)')
    ax3.fill_between(df.index, df['TNX_yield'].min() * 0.8,
                     df['TNX_yield'], alpha=0.08, color=AMBER)
    ax3.set_ylabel('Yield (%)', fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_title('^TNX — 10-Year Treasury Note Yield  |  Exogenous Variable',
                  fontsize=11, color=GRAY, loc='left')
    ax3.legend(loc='upper right')

    save_fig(fig, 'graph1_price_history.png')


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 2 — Return distributions + Q-Q plots
# ─────────────────────────────────────────────────────────────────────────────

def graph2_distributions(df: pd.DataFrame) -> None:
    """
    2×2 grid: histogram with normal-distribution overlay + Q-Q plot
    for SPY and IEF daily log returns.
    Highlights fat tails (excess kurtosis) relative to the normal distribution.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle('Graph 2 — Return Distributions and Normality Analysis',
                 fontsize=15, fontweight='bold', color=BLUE)

    assets = [
        ('SPY Daily Log Returns', df['SPY_ret'].dropna(), LBLUE),
        ('IEF Daily Log Returns', df['IEF_ret'].dropna(), GREEN),
    ]

    for col, (title, ret, color) in enumerate(assets):
        # ── Histogram ────────────────────────────────
        ax = axes[0][col]
        ret_pct = ret * 100
        ax.hist(ret_pct, bins=100, color=color, alpha=0.65,
                density=True, edgecolor='white', linewidth=0.2)

        mu, sigma = ret_pct.mean(), ret_pct.std()
        x = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 400)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'k--',
                linewidth=1.8, label='Normal fit')
        ax.axvline(mu, color='red', linewidth=1.2, linestyle=':',
                   label=f'Mean = {mu:.3f}%')

        kurt = ret.kurtosis()
        skew = ret.skew()
        ax.set_title(
            f'{title}\n'
            f'Kurtosis = {kurt:.2f}  |  Skewness = {skew:.2f}  '
            f'(Normal: kurt=3, skew=0)',
            fontsize=10, color=GRAY
        )
        ax.set_xlabel('Daily Return (%)')
        ax.set_ylabel('Density')
        ax.legend()

        # ── Q-Q plot ──────────────────────────────────
        ax2 = axes[1][col]
        (osm, osr), (slope, intercept, r) = stats.probplot(ret, dist='norm')
        ax2.scatter(osm, osr, alpha=0.20, s=4, color=color)
        line_x = np.array([osm.min(), osm.max()])
        ax2.plot(line_x, slope * line_x + intercept, 'k--',
                 linewidth=1.8, label=f'R² = {r**2:.4f}')
        ax2.set_title(
            f'Q-Q Plot — {title.split()[0]} Returns\n'
            f'Deviations at extremes confirm fat tails',
            fontsize=10, color=GRAY
        )
        ax2.set_xlabel('Theoretical Quantiles (Normal)')
        ax2.set_ylabel('Sample Quantiles')
        ax2.legend()

    plt.tight_layout()
    save_fig(fig, 'graph2_distributions.png')


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 3 — Rolling statistics
# ─────────────────────────────────────────────────────────────────────────────

def graph3_rolling_stats(df: pd.DataFrame) -> None:
    """
    Two panels:
    - Rolling 20-day and 60-day mean return with bull/bear regime shading
    - Rolling 20-day annualised volatility with average line
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                             gridspec_kw={'hspace': 0.10})
    fig.suptitle('Graph 3 — Rolling Statistics: Mean Return and Volatility Over Time',
                 fontsize=15, fontweight='bold', color=BLUE)

    ret = df['SPY_ret'].dropna()
    roll20  = ret.rolling(20).mean() * 100
    roll60  = ret.rolling(60).mean() * 100
    roll_vol = ret.rolling(20).std() * np.sqrt(252) * 100

    # ── Rolling mean ─────────────────────────────────
    ax = axes[0]
    ax.plot(df.index, roll20, color=LBLUE, linewidth=1.0, alpha=0.85,
            label='20-day rolling mean (%)')
    ax.plot(df.index, roll60, color=BLUE,  linewidth=1.5,
            label='60-day rolling mean (%)')
    ax.axhline(0, color=GRAY, linewidth=0.9, linestyle='--')
    ax.fill_between(df.index, 0, roll20,
                    where=(roll20 >= 0), alpha=0.12, color=GREEN,
                    label='Positive regime')
    ax.fill_between(df.index, 0, roll20,
                    where=(roll20 < 0),  alpha=0.18, color=RED,
                    label='Negative regime')
    ax.set_ylabel('Rolling Mean Return (%)', fontweight='bold')
    ax.legend(loc='upper left')
    ax.set_title('Rolling Mean Return — Regime Changes Visible', fontsize=11,
                 color=GRAY, loc='left')

    # ── Rolling volatility ────────────────────────────
    ax2 = axes[1]
    ax2.plot(df.index, roll_vol, color=PURPLE, linewidth=1.0,
             label='20-day Rolling Annualised Vol (%)')
    ax2.fill_between(df.index, 0, roll_vol, alpha=0.13, color=PURPLE)
    avg_vol = roll_vol.mean()
    ax2.axhline(avg_vol, color=AMBER, linewidth=1.5, linestyle='--',
                label=f'Average vol ({avg_vol:.1f}%)')
    ax2.set_ylabel('Annualised Volatility (%)', fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.legend(loc='upper right')
    ax2.set_title(
        'Rolling Volatility — GFC (2008), COVID (2020), Rate Hike (2022) Spikes Visible',
        fontsize=11, color=GRAY, loc='left'
    )

    save_fig(fig, 'graph3_rolling_stats.png')


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 4 — Stationarity: raw prices vs log returns
# ─────────────────────────────────────────────────────────────────────────────

def graph4_stationarity(df: pd.DataFrame) -> None:
    """
    2×2 grid comparing raw price (non-stationary) vs log returns (stationary)
    for both SPY and IEF. ADF test result annotated directly on each panel.
    """

    def adf_pvalue(series):
        try:
            result = adfuller(series.dropna(), autolag='AIC')
            return result[1]   # p-value
        except Exception:
            return float('nan')

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle('Graph 4 — Stationarity: Raw Prices vs. Log Returns',
                 fontsize=15, fontweight='bold', color=BLUE)

    panels = [
        (axes[0][0], df['SPY_price'], 'SPY Raw Price',   LBLUE, False),
        (axes[0][1], df['SPY_ret'],   'SPY Log Returns', LBLUE, True),
        (axes[1][0], df['IEF_price'], 'IEF Raw Price',   GREEN, False),
        (axes[1][1], df['IEF_ret'],   'IEF Log Returns', GREEN, True),
    ]

    for ax, series, label, color, is_stationary in panels:
        pval = adf_pvalue(series)
        series_clean = series.dropna()

        ax.plot(series_clean.index, series_clean, color=color,
                linewidth=0.8 if is_stationary else 1.2, alpha=0.9)

        if is_stationary:
            ax.axhline(0, color='black', linewidth=0.8)
            verdict, v_color, bg = 'STATIONARY ✓', GREEN, '#F0FDF4'
        else:
            verdict, v_color, bg = 'NON-STATIONARY ✗', RED, '#FEF2F2'

        ax.set_title(f'{label}  —  {verdict}', fontsize=10, color=v_color)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USD)' if not is_stationary else 'Log Return')

        pval_txt = f'ADF p-value: {pval:.4f}' if not np.isnan(pval) else 'ADF: N/A'
        ax.text(0.97, 0.05, pval_txt,
                transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
                color=v_color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=bg,
                          edgecolor=v_color, alpha=0.85))

    plt.tight_layout()
    save_fig(fig, 'graph4_stationarity.png')


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 5 — ACF and PACF
# ─────────────────────────────────────────────────────────────────────────────

def graph5_acf_pacf(df: pd.DataFrame) -> None:
    """
    2×2 grid of ACF and PACF for SPY and IEF log returns.
    Uses statsmodels for accurate computation.
    95% confidence bands highlighted in red.
    """

    nlags = 40
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        'Graph 5 — Autocorrelation (ACF) and Partial Autocorrelation (PACF)',
        fontsize=15, fontweight='bold', color=BLUE
    )

    assets = [
        ('SPY Log Returns', df['SPY_ret'].dropna(), LBLUE),
        ('IEF Log Returns', df['IEF_ret'].dropna(), GREEN),
    ]

    for col, (label, series, color) in enumerate(assets):
        acf_vals,  acf_ci  = acf( series, nlags=nlags, alpha=0.05)
        pacf_vals, pacf_ci = pacf(series, nlags=nlags, alpha=0.05, method='ywm')

        for row, (vals, ci, func_label) in enumerate([
            (acf_vals,  acf_ci,  'ACF'),
            (pacf_vals, pacf_ci, 'PACF'),
        ]):
            ax = axes[row][col]
            x = np.arange(len(vals))
            ax.bar(x, vals, color=color, alpha=0.60, width=0.6)
            ax.axhline(0, color='black', linewidth=0.8)

            # 95% confidence interval from statsmodels
            ci_upper =  ci[:, 1] - vals
            ci_lower = -ci[:, 0] + vals
            # Plot flat CI band using average half-width
            conf = np.mean(ci_upper[1:])
            ax.axhline( conf, color=RED, linewidth=1.2, linestyle='--',
                       label=f'95% CI (±{conf:.3f})')
            ax.axhline(-conf, color=RED, linewidth=1.2, linestyle='--')
            ax.fill_between([-1, nlags+1], -conf, conf, alpha=0.06, color=RED)

            ax.set_xlim(-1, nlags + 1)
            ax.set_ylim(-0.25, 0.25)
            ax.set_title(
                f'{func_label} — {label}\n'
                f'(bars beyond red = statistically significant at 95%)',
                fontsize=10, color=GRAY
            )
            ax.set_xlabel('Lag (days)')
            ax.set_ylabel('Correlation')
            ax.legend()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))

    plt.tight_layout()
    save_fig(fig, 'graph5_acf_pacf.png')


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 6 — SPY vs IEF dynamic correlation
# ─────────────────────────────────────────────────────────────────────────────

def graph6_correlation(df: pd.DataFrame) -> None:
    """
    Three-panel chart showing the dynamic relationship between stocks and bonds:
    - Rolling 60-day correlation over time (regime changes visible)
    - Scatter plot of daily returns (overall linear relationship)
    - Bar chart of annual correlation (year-by-year breakdown)
    """
    spy_ret = df['SPY_ret'].dropna()
    ief_ret = df['IEF_ret'].dropna()
    aligned = pd.concat([spy_ret, ief_ret], axis=1).dropna()

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        'Graph 6 — SPY vs. IEF: Dynamic Correlation (The Stocks-Bonds Relationship)',
        fontsize=15, fontweight='bold', color=BLUE
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    # ── Rolling correlation ───────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    roll_corr = aligned['SPY_ret'].rolling(60).corr(aligned['IEF_ret'])
    ax1.plot(aligned.index, roll_corr, color=PURPLE, linewidth=1.1)
    ax1.axhline(0, color=GRAY, linewidth=1, linestyle='--')
    ax1.fill_between(aligned.index, 0, roll_corr,
                     where=(roll_corr < 0), alpha=0.18, color=LBLUE,
                     label='Negative corr (typical: stocks ↑, bonds ↑)')
    ax1.fill_between(aligned.index, 0, roll_corr,
                     where=(roll_corr >= 0), alpha=0.22, color=RED,
                     label='Positive corr (regime break)')
    ax1.set_title(
        '60-Day Rolling Correlation: SPY vs IEF\n'
        '(2008 GFC, 2013 taper tantrum, 2022 rate hikes = regime breaks)',
        fontsize=10, color=GRAY
    )
    ax1.set_ylabel('Correlation')
    ax1.set_xlabel('Date')
    ax1.legend(fontsize=8, loc='lower left')
    ax1.set_ylim(-1, 1)

    # ── Scatter ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.scatter(aligned['IEF_ret'] * 100, aligned['SPY_ret'] * 100,
                alpha=0.12, s=4, color=PURPLE)
    slope, intercept, r, _, _ = stats.linregress(
        aligned['IEF_ret'], aligned['SPY_ret']
    )
    x_line = np.linspace(aligned['IEF_ret'].min(),
                         aligned['IEF_ret'].max(), 200)
    ax2.plot(x_line * 100, (slope * x_line + intercept) * 100,
             'r-', linewidth=2, label=f'r = {r:.3f}')
    ax2.set_title(f'Scatter: IEF vs SPY\n(overall r = {r:.3f})',
                  fontsize=10, color=GRAY)
    ax2.set_xlabel('IEF Return (%)')
    ax2.set_ylabel('SPY Return (%)')
    ax2.legend()

    # ── Annual correlation bar ────────────────────────
    ax3 = fig.add_subplot(gs[1, :])
    yearly = aligned.groupby(aligned.index.year).apply(
        lambda g: g['SPY_ret'].corr(g['IEF_ret'])
    )
    bar_colors = [RED if v >= 0 else LBLUE for v in yearly.values]
    bars = ax3.bar(yearly.index, yearly.values,
                   color=bar_colors, alpha=0.75, edgecolor='white')
    ax3.axhline(0, color='black', linewidth=0.8)
    ax3.set_title(
        'Annual SPY–IEF Correlation by Year\n'
        '(Red = positive correlation, breaks the typical inverse relationship)',
        fontsize=10, color=GRAY
    )
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Annual Correlation')
    ax3.set_xticks(yearly.index)
    ax3.set_xticklabels(yearly.index, rotation=45, ha='right')
    for bar, val in zip(bars, yearly.values):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015 * np.sign(val),
            f'{val:.2f}', ha='center',
            va='bottom' if val >= 0 else 'top',
            fontsize=7.5, color='#1E293B'
        )

    save_fig(fig, 'graph6_correlation.png')


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 7 — Train / Validation / Test split
# ─────────────────────────────────────────────────────────────────────────────

def graph7_split(df: pd.DataFrame) -> None:
    """
    SPY price curve coloured by split region (Train / Val / Test).
    Labels show exact date ranges and number of trading days per split.
    """
    n       = len(df)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    train_idx = df.index[:n_train]
    val_idx   = df.index[n_train : n_train + n_val]
    test_idx  = df.index[n_train + n_val :]

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(
        'Graph 7 — Sequential Train / Validation / Test Split\n'
        '(Temporal order strictly preserved — no shuffling)',
        fontsize=14, fontweight='bold', color=BLUE
    )

    spy = df['SPY_price']

    ax.plot(train_idx, spy.iloc[:n_train],
            color=BLUE,  linewidth=1.2, label=f'Training  ({len(train_idx)} days)')
    ax.plot(val_idx,   spy.iloc[n_train : n_train + n_val],
            color=GREEN, linewidth=1.4, label=f'Validation  ({len(val_idx)} days)')
    ax.plot(test_idx,  spy.iloc[n_train + n_val:],
            color=RED,   linewidth=1.4, label=f'Test  ({len(test_idx)} days)')

    ax.fill_between(train_idx, 0, spy.iloc[:n_train],
                    alpha=0.07, color=BLUE)
    ax.fill_between(val_idx,   0, spy.iloc[n_train : n_train + n_val],
                    alpha=0.12, color=GREEN)
    ax.fill_between(test_idx,  0, spy.iloc[n_train + n_val:],
                    alpha=0.12, color=RED)

    ax.axvline(train_idx[-1], color=BLUE,  linewidth=2, linestyle='--', alpha=0.6)
    ax.axvline(val_idx[-1],   color=GREEN, linewidth=2, linestyle='--', alpha=0.6)

    y_top = spy.max() * 0.94
    for idx, label, c in [
        (train_idx, f'TRAIN\n{train_idx[0].year}–{train_idx[-1].year}\n({len(train_idx)} days)', BLUE),
        (val_idx,   f'VALIDATION\n{val_idx[0].year}–{val_idx[-1].year}\n({len(val_idx)} days)',   GREEN),
        (test_idx,  f'TEST\n{test_idx[0].year}–{test_idx[-1].year}\n({len(test_idx)} days)',      RED),
    ]:
        ax.text(
            idx[len(idx) // 2], y_top, label,
            ha='center', va='top', fontsize=10, fontweight='bold', color=c,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor=c, alpha=0.90)
        )

    ax.set_ylabel('SPY Price (USD)', fontweight='bold')
    ax.set_xlabel('Date')
    ax.legend(loc='upper left')
    ax.set_ylim(0)

    save_fig(fig, 'graph7_split.png')


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("Step 3 — Generate EDA plots")
    print("=" * 55)
    print()

    df = load()
    print()

    graph1_price_history(df)
    graph2_distributions(df)
    graph3_rolling_stats(df)
    graph4_stationarity(df)
    graph5_acf_pacf(df)
    graph6_correlation(df)
    graph7_split(df)

    print()
    print(f"All 7 graphs saved to  {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
