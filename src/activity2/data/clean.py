"""
Activity 2 — Step 2: align assets, build the multi-class regime label, and save.

Key differences from activity 1's cleaning:

1. Outer join (NOT inner): we deliberately KEEP the missing rows so the
   imputation comparison in the modeling phase has real NaN patterns to study.
2. The target is multi-class (regime), not binary (up/down).
3. Two processed datasets are saved:
   - full_dataset_with_nans.csv → for the imputation experiments
   - full_dataset_imputed.csv   → ffill-imputed baseline used by the EDA
"""

import os

import numpy as np
import pandas as pd

from ..config import (
    PROCESSED_DIR,
    RAW_DIR,
    REGIME_ORDER,
    REGIME_THRESHOLDS,
    TICKERS,
    TRAIN_RATIO,
    VAL_RATIO,
)

os.makedirs(PROCESSED_DIR, exist_ok=True)


# ── Load helpers ──────────────────────────────────────────────────────────────

def _load_close(ticker: str) -> pd.Series:
    safe = ticker.replace("^", "")
    path = os.path.join(RAW_DIR, f"{safe}.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    for col in ["Close", "Adj Close", "close", "adj close"]:
        if col in df.columns:
            s = df[col].squeeze()
            s.name = ticker
            return s
    raise KeyError(f"No close column found in {path}")


# ── Regime label ──────────────────────────────────────────────────────────────

def label_regime(spy_ret: pd.Series) -> pd.Series:
    """
    Bucket each daily SPY log-return into one of four market regimes.
    Returns a categorical series with the same index as `spy_ret`.
    """
    crash, correction, normal = (
        REGIME_THRESHOLDS["crash"],
        REGIME_THRESHOLDS["correction"],
        REGIME_THRESHOLDS["normal"],
    )
    labels = pd.Series(index=spy_ret.index, dtype="object")
    labels[spy_ret <= crash] = "crash"
    labels[(spy_ret > crash) & (spy_ret <= correction)] = "correction"
    labels[(spy_ret > correction) & (spy_ret <= normal)] = "normal"
    labels[spy_ret > normal] = "rally"
    return pd.Categorical(labels, categories=REGIME_ORDER, ordered=True)


# ── Pipeline ──────────────────────────────────────────────────────────────────

def merge_outer(series_list: list) -> pd.DataFrame:
    df = pd.concat(series_list, axis=1, join="outer").sort_index()
    df.index = pd.to_datetime(df.index)
    return df


def audit_missing(df: pd.DataFrame) -> pd.DataFrame:
    rep = pd.DataFrame(
        {
            "n_missing": df.isna().sum(),
            "pct_missing": df.isna().mean().mul(100).round(2),
            "first_obs": df.apply(lambda c: c.first_valid_index()),
            "last_obs": df.apply(lambda c: c.last_valid_index()),
        }
    )
    print("\n  Missing-value audit (outer-joined raw prices):")
    print(rep.to_string())
    return rep


def engineer_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns and the regime target. Keeps NaNs untouched."""
    out = pd.DataFrame(index=prices.index)
    out["SPY_price"] = prices["SPY"]

    # Log returns (NaN propagates wherever the price is missing — by design)
    # Strip the `^` from index tickers so column names stay clean (VIX_ret, TNX_ret)
    for tkr in TICKERS:
        out[f"{tkr.lstrip('^')}_ret"] = np.log(prices[tkr] / prices[tkr].shift(1))

    # Yield change for ^TNX (handled separately because it's not a price)
    out["TNX_chg"] = prices["^TNX"].diff()

    # Target: regime of NEXT day → shift(-1) so today's features predict tomorrow
    next_day_ret = out["SPY_ret"].shift(-1)
    out["regime"] = label_regime(next_day_ret)

    return out


def impute_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill then back-fill — this is ONLY the baseline used for the EDA.
    The modeling phase will compare ffill / linear / KNN / MICE explicitly.
    """
    return df.ffill().bfill()


def split_data(df: pd.DataFrame):
    n = len(df)
    n_tr = int(n * TRAIN_RATIO)
    n_va = int(n * VAL_RATIO)
    train = df.iloc[:n_tr]
    val   = df.iloc[n_tr : n_tr + n_va]
    test  = df.iloc[n_tr + n_va :]
    return train, val, test


def main() -> None:
    print("=" * 60)
    print("Activity 2 — Step 2: clean & label")
    print("=" * 60)

    # 1. Load
    print("\n  Loading raw CSVs ...")
    prices = merge_outer([_load_close(t) for t in TICKERS])
    print(
        f"  Combined index: {len(prices)} rows "
        f"({prices.index[0].date()} → {prices.index[-1].date()})"
    )

    # 2. Missing-value audit (BEFORE any imputation — for the report)
    audit = audit_missing(prices)
    audit.to_csv(os.path.join(PROCESSED_DIR, "missing_audit.csv"))

    # 3. Feature engineering on raw (NaN-preserving) prices
    feat_with_nans = engineer_features(prices)
    feat_with_nans.to_csv(os.path.join(PROCESSED_DIR, "full_dataset_with_nans.csv"))

    # 4. Baseline-imputed dataset (for EDA only)
    feat_imputed = impute_baseline(feat_with_nans.drop(columns=["regime"]))
    feat_imputed["regime"] = feat_with_nans["regime"]
    feat_imputed = feat_imputed.dropna(subset=["regime"])  # drop final-row target NaN
    feat_imputed.to_csv(os.path.join(PROCESSED_DIR, "full_dataset_imputed.csv"))

    # 5. Class distribution
    counts = feat_imputed["regime"].value_counts().reindex(REGIME_ORDER)
    pct = (counts / counts.sum() * 100).round(2)
    print("\n  Regime class distribution (next-day SPY):")
    for cls in REGIME_ORDER:
        print(f"    {cls:11s}  {counts[cls]:>5}  ({pct[cls]:>5.2f}%)")

    # 6. Sequential split (chronological — no shuffle)
    train, val, test = split_data(feat_imputed)
    train.to_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    val.to_csv(os.path.join(PROCESSED_DIR, "val.csv"))
    test.to_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    print(
        f"\n  Split  → train {len(train)}  ({train.index[0].date()}–{train.index[-1].date()})"
        f"\n          val   {len(val)}  ({val.index[0].date()}–{val.index[-1].date()})"
        f"\n          test  {len(test)}  ({test.index[0].date()}–{test.index[-1].date()})"
    )
    print(f"\n  Files saved to {PROCESSED_DIR}/")


if __name__ == "__main__":
    main()
