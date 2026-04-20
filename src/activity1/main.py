"""
Activity 1 — EDA pipeline entry point.

Runs the three steps in order:
    1. download_data → fetches SPY, IEF, ^TNX from Yahoo Finance
    2. clean_data    → merges, imputes, engineers features, splits
    3. eda_plots     → generates the 7 EDA charts

Run from project root:
    python -m src.activity1.main
or via the root dispatcher:
    python main.py 1
"""

from . import download_data, clean_data, eda_plots


def main() -> None:
    download_data.main()
    clean_data.main()
    eda_plots.main()


if __name__ == "__main__":
    main()
