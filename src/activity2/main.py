"""
Activity 2 — pipeline entry point.

Runs the three steps in order:
    1. download_data → fetches SPY, IEF, ^TNX, ^VIX, GLD, UUP, USO from Yahoo Finance
    2. clean_data    → outer-merge, regime label, train/val/test split (NaNs preserved)
    3. eda_plots     → 7 charts tailored to imbalance / imputation / multi-class

Run from project root:
    python -m src.activity2.main
or via the root dispatcher:
    python main.py 2
"""

from . import clean_data, download_data, eda_plots


def main() -> None:
    download_data.main()
    clean_data.main()
    eda_plots.main()


if __name__ == "__main__":
    main()
