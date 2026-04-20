# download_data.py
# ─────────────────────────────────────────────────────────────────────────────
# Downloads raw OHLCV data for SPY, IEF and ^TNX from Yahoo Finance
# and saves each as a separate CSV in data/raw/
#
# Run:  python download_data.py
# Requires: pip install yfinance pandas
# ─────────────────────────────────────────────────────────────────────────────

import os
import yfinance as yf
import pandas as pd
from .config import TICKERS, START_DATE, END_DATE, RAW_DIR

os.makedirs(RAW_DIR, exist_ok=True)


def download_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download full OHLCV history for a single ticker."""
    print(f"  Downloading {ticker}  ({start} → {end}) ...")
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,   # adjusts for splits & dividends automatically
        progress=False,
    )
    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check ticker or date range.")
    return df


def main():
    print("=" * 55)
    print("Step 1 — Download raw data from Yahoo Finance")
    print("=" * 55)

    for role, ticker in TICKERS.items():
        df = download_ticker(ticker, START_DATE, END_DATE)

        # Flatten MultiIndex columns if present (yfinance sometimes returns them)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        safe_name = ticker.replace("^", "")   # ^TNX → TNX for filename
        path = os.path.join(RAW_DIR, f"{safe_name}.csv")
        df.to_csv(path)

        print(f"  ✓  {ticker:6s} | {len(df):>5} trading days "
              f"| {df.index[0].date()} → {df.index[-1].date()} "
              f"| saved → {path}")

    print()
    print("All raw files saved to data/raw/")
    print("Next step: run  python 02_clean_data.py")


if __name__ == "__main__":
    main()
