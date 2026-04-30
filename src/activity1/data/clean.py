# clean.py
# ─────────────────────────────────────────────────────────────────────────────
# Step 2 of the Activity-1 data pipeline.
#
# Loads the three raw CSVs, aligns them on a common trading calendar,
# handles missing values, computes log returns and derived features,
# performs the train/val/test split, and saves the cleaned dataset.
#
# Run from project root:
#     python -m src.activity1.data.clean
# ─────────────────────────────────────────────────────────────────────────────

import os
import numpy as np
import pandas as pd

from ..config import TICKERS, RAW_DIR, PROCESSED_DIR, TRAIN_RATIO, VAL_RATIO

os.makedirs(PROCESSED_DIR, exist_ok=True)


# ─── 1. Load raw CSVs ────────────────────────────────────────────────────────

def load_raw(ticker: str) -> pd.Series:
    """Load adjusted close price for one ticker from raw CSV."""
    safe = ticker.replace("^", "")
    path = os.path.join(RAW_DIR, f"{safe}.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    # yfinance column names vary slightly — find the close column
    close_col = None
    for candidate in ["Close", "Adj Close", "close", "adj close"]:
        if candidate in df.columns:
            close_col = candidate
            break
    if close_col is None:
        raise KeyError(f"No 'Close' column found in {path}. Columns: {list(df.columns)}")

    series = df[close_col].squeeze()
    series.name = ticker
    return series


# ─── 2. Merge on common dates ────────────────────────────────────────────────

def merge_series(series_list: list) -> pd.DataFrame:
    """Inner-join all series on their shared trading dates."""
    df = pd.concat(series_list, axis=1, join="inner")
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


# ─── 3. Missing value audit ──────────────────────────────────────────────────

def audit_missing(df: pd.DataFrame) -> None:
    """Print a quick missing-value report before any imputation."""
    print("\n  Missing value audit (raw merged data):")
    for col in df.columns:
        n_null = df[col].isna().sum()
        pct    = n_null / len(df) * 100
        print(f"    {col:6s}  →  {n_null} nulls  ({pct:.2f}%)")


# ─── 4. Imputation ───────────────────────────────────────────────────────────

def impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill only — standard market convention for non-trading days
    or data gaps (carry the last known price forward).

    NOTE: we deliberately do NOT use bfill. Backward-fill would impute
    early NaNs with FUTURE values, which is a look-ahead leak even if
    those rows would later be dropped by drop_na_rows(). Forward-fill is
    causal; any leading NaNs are handled downstream.
    """
    return df.ffill()


# ─── 5. Feature engineering ──────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all derived features used for modeling."""
    spy = df["SPY"]
    ief = df["IEF"]
    tnx = df["^TNX"]

    out = pd.DataFrame(index=df.index)

    # ── Raw prices ───────────────────────────────────
    out["SPY_price"] = spy
    out["IEF_price"] = ief
    out["TNX_yield"] = tnx

    # ── Log returns ──────────────────────────────────
    out["SPY_ret"] = np.log(spy / spy.shift(1))
    out["IEF_ret"] = np.log(ief / ief.shift(1))
    out["TNX_chg"] = tnx.diff()

    # ── Lagged SPY returns ───────────────────────────
    for lag in [1, 2, 3, 5, 10]:
        out[f"SPY_ret_lag{lag}"] = out["SPY_ret"].shift(lag)

    # ── Lagged bond features ─────────────────────────
    for lag in [1, 2]:
        out[f"IEF_ret_lag{lag}"] = out["IEF_ret"].shift(lag)
        out[f"TNX_chg_lag{lag}"] = out["TNX_chg"].shift(lag)

    # ── Rolling statistics (SPY) ─────────────────────
    for window in [5, 10, 20]:
        out[f"SPY_roll_mean_{window}"] = out["SPY_ret"].rolling(window).mean()
        out[f"SPY_roll_std_{window}"]  = out["SPY_ret"].rolling(window).std()

    # ── Rolling statistics (IEF) ─────────────────────
    for window in [5, 20]:
        out[f"IEF_roll_mean_{window}"] = out["IEF_ret"].rolling(window).mean()

    # ── Annualised rolling volatility ────────────────
    out["SPY_ann_vol_20"] = out["SPY_ret"].rolling(20).std() * np.sqrt(252) * 100

    # ── Day-of-week ──────────────────────────────────
    out["day_of_week"] = out.index.dayofweek

    # ── Target: next-day SPY return ──────────────────
    out["target"] = out["SPY_ret"].shift(-1)

    return out


# ─── 6. Drop NaN rows ────────────────────────────────────────────────────────

def drop_na_rows(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna()
    after  = len(df)
    print(f"\n  Rows dropped due to NaN (lagging/rolling warm-up): {before - after}")
    print(f"  Final dataset size: {after} trading days")
    return df


# ─── 7. Sequential train / val / test split ──────────────────────────────────

def split_data(df: pd.DataFrame):
    """Strictly sequential split — NO shuffling (leakage in time series)."""
    n       = len(df)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    train = df.iloc[:n_train]
    val   = df.iloc[n_train : n_train + n_val]
    test  = df.iloc[n_train + n_val :]

    print(f"\n  Train:      {len(train):>5} days  "
          f"({train.index[0].date()} → {train.index[-1].date()})")
    print(f"  Validation: {len(val):>5} days  "
          f"({val.index[0].date()} → {val.index[-1].date()})")
    print(f"  Test:       {len(test):>5} days  "
          f"({test.index[0].date()} → {test.index[-1].date()})")

    return train, val, test


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("Step 2 — Clean and preprocess data")
    print("=" * 55)

    print("\n  Loading raw CSVs ...")
    spy_s = load_raw("SPY")
    ief_s = load_raw("IEF")
    tnx_s = load_raw("^TNX")

    raw = merge_series([spy_s, ief_s, tnx_s])
    print(f"\n  Merged dataset: {len(raw)} common trading days "
          f"({raw.index[0].date()} → {raw.index[-1].date()})")

    audit_missing(raw)
    raw = impute(raw)
    print("\n  Imputation done (ffill → bfill). Remaining nulls:", raw.isna().sum().sum())

    feat = engineer_features(raw)
    feat = drop_na_rows(feat)

    print("\n  Splitting data sequentially ...")
    train, val, test = split_data(feat)

    feat.to_csv(os.path.join(PROCESSED_DIR, "full_dataset.csv"))
    train.to_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    val.to_csv(os.path.join(PROCESSED_DIR, "val.csv"))
    test.to_csv(os.path.join(PROCESSED_DIR, "test.csv"))

    print(f"\n  Saved to {PROCESSED_DIR}/")
    for name in ["full_dataset", "train", "val", "test"]:
        print(f"    {name}.csv")


if __name__ == "__main__":
    main()
