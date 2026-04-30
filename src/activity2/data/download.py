"""
Activity 2 — Step 1: download raw OHLCV data for the extended asset universe.

Each ticker is saved as its own CSV in data/activity2/raw/ so later steps can
reload them independently and the missing-value patterns are preserved exactly
as Yahoo Finance returns them.
"""

import os

import pandas as pd
import yfinance as yf

from ..config import END_DATE, RAW_DIR, START_DATE, TICKERS

os.makedirs(RAW_DIR, exist_ok=True)


def _safe_filename(ticker: str) -> str:
    """Yahoo Finance uses '^' for indices — strip it so the file is portable."""
    return ticker.replace("^", "")


def download_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def main() -> None:
    print("=" * 60)
    print("Activity 2 — Step 1: download raw data")
    print("=" * 60)

    for ticker, meta in TICKERS.items():
        df = download_ticker(ticker, START_DATE, END_DATE)
        path = os.path.join(RAW_DIR, f"{_safe_filename(ticker)}.csv")
        df.to_csv(path)
        print(
            f"  ✓ {ticker:5s} ({meta['role']:11s}) | "
            f"{len(df):>5} rows | "
            f"{df.index[0].date()} → {df.index[-1].date()} | {path}"
        )

    print(f"\nAll files saved to {RAW_DIR}/")


if __name__ == "__main__":
    main()
