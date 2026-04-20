"""
Activity 2 — Step 3: EDA charts tailored to the imbalance / imputation /
multi-class theme.

Charts are intentionally different from activity 1's (no ACF / PACF, no
stationarity tests, no rolling vol). Those belong to the time-series topic and
were already covered in activity 1.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .config import FIGURES_DIR, PROCESSED_DIR, REGIME_ORDER, TICKERS

os.makedirs(FIGURES_DIR, exist_ok=True)


# ── Visual style ─────────────────────────────────────────────────────────────

REGIME_COLORS = {
    "crash":      "#c0392b",
    "correction": "#e67e22",
    "normal":     "#7f8c8d",
    "rally":      "#27ae60",
}

ASSET_COLORS = {
    "SPY":  "#1f77b4",
    "IEF":  "#2ca02c",
    "^TNX": "#9467bd",
    "^VIX": "#d62728",
    "GLD":  "#bcbd22",
    "UUP":  "#17becf",
    "USO":  "#8c564b",
}


def _setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi":          120,
        "savefig.dpi":         150,
        "font.family":         "DejaVu Sans",
        "font.size":           10.5,
        "axes.titlesize":      12,
        "axes.titleweight":    "bold",
        "axes.labelsize":      10.5,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.grid":           True,
        "grid.alpha":          0.25,
        "grid.linestyle":      "--",
        "legend.frameon":      False,
        "legend.fontsize":     9.5,
        "xtick.labelsize":     9.5,
        "ytick.labelsize":     9.5,
        "axes.titlepad":       12,
        "figure.autolayout":   False,
    })


_setup_style()


# ── Loaders & utilities ──────────────────────────────────────────────────────

def _load_imputed() -> pd.DataFrame:
    return pd.read_csv(
        os.path.join(PROCESSED_DIR, "full_dataset_imputed.csv"),
        index_col=0, parse_dates=True,
    )


def _load_with_nans() -> pd.DataFrame:
    return pd.read_csv(
        os.path.join(PROCESSED_DIR, "full_dataset_with_nans.csv"),
        index_col=0, parse_dates=True,
    )


def _load_raw_prices() -> pd.DataFrame:
    """Load raw close prices (with NaNs) for every asset, aligned outer-join."""
    from .clean_data import _load_close, merge_outer
    return merge_outer([_load_close(t) for t in TICKERS])


def _save(fig, name: str) -> None:
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {path}")


# ── Chart 1. Multi-asset overview (normalised to 100 at first valid obs) ─────

def chart_universe_overview(prices: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))

    for tkr in TICKERS:
        s = prices[tkr].dropna()
        if s.empty:
            continue
        norm = (s / s.iloc[0]) * 100
        ax.plot(norm.index, norm.values, color=ASSET_COLORS[tkr],
                linewidth=1.1, label=tkr, alpha=0.9)

    ax.set_title("Graph 1. Extended asset universe rebased to 100 at each asset's first valid observation")
    ax.set_ylabel("Indexed level (start = 100)")
    ax.set_xlabel("Date")
    ax.set_yscale("log")
    ax.legend(loc="upper left", ncol=4)
    ax.grid(True, which="both", alpha=0.2)

    fig.tight_layout()
    _save(fig, "graph01_universe_overview.png")


# ── Chart 2. Class distribution (the imbalance) ──────────────────────────────

def chart_class_distribution(df: pd.DataFrame) -> None:
    counts = df["regime"].value_counts().reindex(REGIME_ORDER)
    pct = counts / counts.sum() * 100
    colors = [REGIME_COLORS[r] for r in REGIME_ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    bars = axes[0].bar(REGIME_ORDER, counts.values, color=colors,
                       edgecolor="white", linewidth=1.5)
    for b, c, p in zip(bars, counts.values, pct.values):
        axes[0].text(b.get_x() + b.get_width() / 2,
                     b.get_height() + counts.max() * 0.01,
                     f"{c:,}\n({p:.1f}%)", ha="center", va="bottom", fontsize=10.5)
    axes[0].set_title("Counts and percentages by regime")
    axes[0].set_ylabel("Number of trading days")
    axes[0].set_ylim(0, counts.max() * 1.20)

    axes[1].pie(counts.values, labels=REGIME_ORDER, colors=colors,
                autopct="%1.1f%%", startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[1].set_title("Proportional view")

    fig.suptitle(
        "Graph 2. The four-class target is naturally imbalanced. "
        "Crash days, the operationally critical class, account for less than 4 percent",
        fontsize=11.5, y=1.02,
    )
    fig.tight_layout()
    _save(fig, "graph02_class_distribution.png")


# ── Chart 3. SPY return distribution with regime cutoffs marked ──────────────

def chart_return_distribution(df: pd.DataFrame) -> None:
    spy_ret = df["SPY_ret"].dropna() * 100
    fig, ax = plt.subplots(figsize=(11, 4.8))

    ax.hist(spy_ret, bins=120, color="#34495e", edgecolor="white",
            alpha=0.85, linewidth=0.4)

    cutoffs = [(-2.0, "crash"), (-0.5, "correction"), (0.5, "normal")]
    for x, label in cutoffs:
        ax.axvline(x, color=REGIME_COLORS["crash" if x < -1 else "correction" if x < 0 else "rally"],
                   linestyle="--", linewidth=1.6,
                   label=f"cutoff at {x:+.1f}% ({label} boundary)")

    ax.set_title("Graph 3. Daily SPY log-return histogram with the regime boundaries used in this study")
    ax.set_xlabel("Daily log return (%)")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper left")
    ax.set_xlim(-8, 8)

    fig.tight_layout()
    _save(fig, "graph03_return_distribution_with_cutoffs.png")


# ── Chart 4. Missing-value heatmap (year × asset) ────────────────────────────

def chart_missing_heatmap(df_nan: pd.DataFrame) -> None:
    price_cols = [f"{t}_ret" for t in TICKERS]
    miss = df_nan[price_cols].isna()
    miss = miss.assign(year=df_nan.index.year)
    pct = miss.groupby("year").mean() * 100
    pct.columns = [c.replace("_ret", "") for c in pct.columns]

    fig, ax = plt.subplots(figsize=(12, 4.6))
    im = ax.imshow(pct.T.values, aspect="auto", cmap="Reds", vmin=0, vmax=100)

    ax.set_xticks(range(len(pct.index)))
    ax.set_xticklabels(pct.index, rotation=45, ha="right")
    ax.set_yticks(range(len(pct.columns)))
    ax.set_yticklabels(pct.columns)
    ax.set_xlabel("Year")
    ax.set_ylabel("Asset")
    ax.set_title("Graph 4. Percentage of missing daily observations per asset and per year")
    ax.grid(False)

    for i in range(pct.T.shape[0]):
        for j in range(pct.T.shape[1]):
            v = pct.T.values[i, j]
            if v > 0:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        color="white" if v > 50 else "black", fontsize=8.5)

    cbar = plt.colorbar(im, ax=ax, label="% missing", pad=0.01)
    cbar.outline.set_visible(False)

    fig.tight_layout()
    _save(fig, "graph04_missing_heatmap.png")


# ── Chart 5. Cross-asset return correlation ──────────────────────────────────

def chart_correlation(df: pd.DataFrame) -> None:
    cols = [f"{t}_ret" for t in TICKERS]
    corr = df[cols].corr()
    corr.columns = [c.replace("_ret", "") for c in corr.columns]
    corr.index = corr.columns

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.columns)
    ax.grid(False)

    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.5 else "black", fontsize=9.5)

    ax.set_title("Graph 5. Daily-return correlation matrix across the extended asset universe")
    cbar = plt.colorbar(im, ax=ax, label="Pearson correlation", pad=0.02)
    cbar.outline.set_visible(False)

    fig.tight_layout()
    _save(fig, "graph05_correlation.png")


# ── Chart 6. Today's signals split by tomorrow's regime (4 boxplots) ─────────

def chart_features_per_regime(df: pd.DataFrame) -> None:
    feature_cols = [c for c in ["VIX_ret", "GLD_ret", "USO_ret", "TNX_chg"] if c in df.columns]

    fig, axes = plt.subplots(1, len(feature_cols), figsize=(4.4 * len(feature_cols), 4.8))
    if len(feature_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, feature_cols):
        data = [df.loc[df["regime"] == r, col].dropna().values * 100 for r in REGIME_ORDER]
        bp = ax.boxplot(data, labels=REGIME_ORDER, patch_artist=True,
                        showfliers=False, widths=0.6,
                        medianprops={"color": "black", "linewidth": 1.4})
        for patch, r in zip(bp["boxes"], REGIME_ORDER):
            patch.set_facecolor(REGIME_COLORS[r])
            patch.set_alpha(0.78)
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.6)
        ax.set_title(col)
        ax.set_ylabel("Daily change (%)")
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle(
        "Graph 6. Today's exogenous signals grouped by tomorrow's regime, "
        "showing where each input separates the rare classes",
        fontsize=11.5, y=1.02,
    )
    fig.tight_layout()
    _save(fig, "graph06_features_per_regime.png")


# ── Chart 7. Regime timeline (SPY price + crash / rally markers) ─────────────

def chart_regime_timeline(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 4.5))

    spy_price = df["SPY_price"]
    ax.plot(spy_price.index, spy_price.values, color="#2c3e50",
            linewidth=0.9, zorder=3, label="SPY price")

    ymax = spy_price.max()
    ymin = spy_price.min()
    band = (ymax - ymin) * 0.045

    for r, col in [("crash", REGIME_COLORS["crash"]), ("rally", REGIME_COLORS["rally"])]:
        idx = df.index[df["regime"] == r]
        ax.vlines(idx, ymin - band, ymin, color=col, linewidth=0.45, alpha=0.75)

    legend_elems = [
        Line2D([0], [0], color="#2c3e50", linewidth=1.3, label="SPY price"),
        Patch(facecolor=REGIME_COLORS["crash"],  label="crash days"),
        Patch(facecolor=REGIME_COLORS["rally"],  label="rally days"),
    ]
    ax.legend(handles=legend_elems, loc="upper left")

    ax.set_title("Graph 7. SPY price across the full sample with crash and rally days marked along the bottom axis")
    ax.set_ylabel("SPY price (USD)")
    ax.set_xlabel("Date")
    ax.set_ylim(ymin - band * 1.5, ymax * 1.04)

    fig.tight_layout()
    _save(fig, "graph07_regime_timeline.png")


# ── Chart 8. VIX behaviour per regime + per-feature mean heatmap ─────────────

def chart_vix_per_regime(df: pd.DataFrame) -> None:
    vix_col = "VIX_ret" if "VIX_ret" in df.columns else "^VIX_ret"
    if vix_col not in df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    data = [df.loc[df["regime"] == r, vix_col].dropna() * 100 for r in REGIME_ORDER]
    bp = axes[0].boxplot(data, labels=REGIME_ORDER, patch_artist=True,
                         showfliers=False, widths=0.6,
                         medianprops={"color": "black", "linewidth": 1.4})
    for patch, r in zip(bp["boxes"], REGIME_ORDER):
        patch.set_facecolor(REGIME_COLORS[r])
        patch.set_alpha(0.78)
    axes[0].axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.6)
    axes[0].set_title("Today's VIX percentage change conditioned on tomorrow's regime")
    axes[0].set_ylabel("VIX daily change (%)")

    numeric = df.select_dtypes(include=[np.number])
    means = numeric.groupby(df["regime"]).mean().reindex(REGIME_ORDER)
    means_norm = (means - means.mean()) / (means.std() + 1e-12)

    im = axes[1].imshow(means_norm.values, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    axes[1].set_xticks(range(means_norm.shape[1]))
    axes[1].set_xticklabels(means_norm.columns, rotation=45, ha="right", fontsize=7.5)
    axes[1].set_yticks(range(len(REGIME_ORDER)))
    axes[1].set_yticklabels(REGIME_ORDER)
    axes[1].set_title("Per-regime mean of every feature, z-scored across regimes")
    axes[1].grid(False)
    cbar = plt.colorbar(im, ax=axes[1], label="z-score", pad=0.02)
    cbar.outline.set_visible(False)

    fig.suptitle(
        "Graph 8. The volatility index is the single strongest separator of the four classes",
        fontsize=11.5, y=1.02,
    )
    fig.tight_layout()
    _save(fig, "graph08_vix_per_regime.png")


# ── Chart 9. SPY vs VIX scatter coloured by regime (separability check) ──────

def chart_spy_vix_scatter(df: pd.DataFrame) -> None:
    vix_col = "VIX_ret" if "VIX_ret" in df.columns else "^VIX_ret"
    if vix_col not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(8.5, 6))

    for r in REGIME_ORDER:
        sub = df[df["regime"] == r].dropna(subset=["SPY_ret", vix_col])
        ax.scatter(sub["SPY_ret"] * 100, sub[vix_col] * 100,
                   s=10, alpha=0.45, color=REGIME_COLORS[r],
                   edgecolor="none", label=r)

    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_xlabel("SPY daily return today (%)")
    ax.set_ylabel("VIX daily change today (%)")
    ax.set_title("Graph 9. Today's SPY vs VIX coloured by tomorrow's regime, where the rare classes occupy distinct regions")
    ax.legend(loc="upper right")
    ax.set_xlim(-10, 10)

    fig.tight_layout()
    _save(fig, "graph09_spy_vix_scatter.png")


# ── Chart 10. Yearly crash frequency ─────────────────────────────────────────

def chart_yearly_crash_frequency(df: pd.DataFrame) -> None:
    yearly = df.groupby(df.index.year).apply(
        lambda x: (x["regime"] == "crash").mean() * 100
    )

    fig, ax = plt.subplots(figsize=(13, 4.5))
    bars = ax.bar(yearly.index, yearly.values,
                  color=REGIME_COLORS["crash"], edgecolor="white",
                  linewidth=1.2, alpha=0.92)

    for x, v in zip(yearly.index, yearly.values):
        if v >= 1:
            ax.text(x, v + 0.15, f"{v:.1f}%", ha="center", va="bottom", fontsize=8.5)

    ax.axhline(yearly.mean(), color="black", linewidth=1.0, linestyle="--",
               alpha=0.7, label=f"sample mean ({yearly.mean():.2f}%)")
    ax.set_title("Graph 10. Annual frequency of crash days, where crashes cluster around macro stress regimes")
    ax.set_ylabel("Crash days as % of trading days")
    ax.set_xlabel("Year")
    ax.legend(loc="upper right")

    fig.tight_layout()
    _save(fig, "graph10_yearly_crash_frequency.png")


# ── Chart 11. Class balance per chronological partition ──────────────────────

def chart_split_balance(df: pd.DataFrame) -> None:
    n = len(df)
    n_tr = int(n * 0.70)
    n_va = int(n * 0.15)
    parts = {
        "train": df.iloc[:n_tr],
        "val":   df.iloc[n_tr : n_tr + n_va],
        "test":  df.iloc[n_tr + n_va :],
    }
    pcts = pd.DataFrame({
        name: part["regime"].value_counts(normalize=True).reindex(REGIME_ORDER) * 100
        for name, part in parts.items()
    })

    fig, ax = plt.subplots(figsize=(9, 4.8))
    bottom = np.zeros(len(pcts.columns))
    for r in REGIME_ORDER:
        ax.bar(pcts.columns, pcts.loc[r].values, bottom=bottom,
               color=REGIME_COLORS[r], edgecolor="white", linewidth=1.5, label=r)
        for i, v in enumerate(pcts.loc[r].values):
            if v > 2:
                ax.text(i, bottom[i] + v / 2, f"{v:.1f}%",
                        ha="center", va="center", color="white", fontsize=10)
        bottom += pcts.loc[r].values

    ax.set_ylabel("% of partition")
    ax.set_title("Graph 11. Class distribution per chronological partition (train, val, test)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))

    fig.tight_layout()
    _save(fig, "graph11_split_balance.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Activity 2 — Step 3: EDA charts")
    print("=" * 60)

    df_imp = _load_imputed()
    df_nan = _load_with_nans()
    raw_prices = _load_raw_prices()

    chart_universe_overview(raw_prices)
    chart_class_distribution(df_imp)
    chart_return_distribution(df_imp)
    chart_missing_heatmap(df_nan)
    chart_correlation(df_imp)
    chart_features_per_regime(df_imp)
    chart_regime_timeline(df_imp)
    chart_vix_per_regime(df_imp)
    chart_spy_vix_scatter(df_imp)
    chart_yearly_crash_frequency(df_imp)
    chart_split_balance(df_imp)

    print(f"\n  All charts saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
