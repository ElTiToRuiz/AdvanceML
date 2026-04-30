# main.py
# ─────────────────────────────────────────────────────────────────────────────
# Activity 2 orchestrator.
#
# Full pipeline in order:
#   1. data.download           → fetch SPY, IEF, ^TNX, ^VIX, GLD, UUP, USO
#   2. data.clean              → outer-merge, regime label, NaN-preserving
#                                + ffill-imputed datasets, sequential split
#   3. eda.plots               → 11 EDA charts
#   4. pipelines.run_all       → Block A → B → C → D
#
# Run from project root:
#     python main.py 2
# or directly:
#     python -m src.activity2.main
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
