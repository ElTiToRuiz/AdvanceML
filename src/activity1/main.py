# main.py
# ─────────────────────────────────────────────────────────────────────────────
# Activity 1 orchestrator.
#
# Full pipeline in order:
#   1. data.download           → fetch SPY, IEF, ^TNX from Yahoo Finance
#   2. data.clean              → merge, impute, feature engineer, split
#   3. eda.plots               → generate the 7 EDA charts
#   4. pipelines.run_all       → fit + evaluate every model
#                                (baselines + SARIMAX + LSTM + Chronos-2)
#
# Run from project root:
#     python main.py 1
# or directly:
#     python -m src.activity1.main
# ─────────────────────────────────────────────────────────────────────────────

from .data import download, clean
from .eda  import plots as eda_plots
from .pipelines import run_all


def main() -> None:
    download.main()
    clean.main()
    eda_plots.main()
    run_all.main()


if __name__ == "__main__":
    main()
