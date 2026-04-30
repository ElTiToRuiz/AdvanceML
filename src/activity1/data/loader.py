# loader.py
# ─────────────────────────────────────────────────────────────────────────────
# Small helper to load the processed splits. Kept separate so every pipeline
# and notebook loads the data the same way and never touches the CSVs with
# custom pd.read_csv calls that could drift over time.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os
from dataclasses import dataclass

import pandas as pd

from ..config import PROCESSED_DIR


@dataclass(frozen=True)
class Splits:
    """Container for the three sequential splits + the full (concatenated) view."""
    full:  pd.DataFrame
    train: pd.DataFrame
    val:   pd.DataFrame
    test:  pd.DataFrame


def _read(name: str) -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, f"{name}.csv")
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_splits() -> Splits:
    """Load full / train / val / test into a single Splits container."""
    return Splits(
        full  = _read("full_dataset"),
        train = _read("train"),
        val   = _read("val"),
        test  = _read("test"),
    )
