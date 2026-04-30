# loader.py
# ─────────────────────────────────────────────────────────────────────────────
# Helper to load Activity-2 processed splits. Two flavours:
#
#   load_splits()                → ffill-imputed dataset (used by EDA + pipelines
#                                   that don't run their own imputation)
#   load_with_nans()             → outer-joined raw features WITH NaNs
#                                   (used by Block A so each imputation method
#                                    sees the real missing pattern)
#
# Mirrors src/activity1/data/loader.py.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os
from dataclasses import dataclass

import pandas as pd

from ..config import PROCESSED_DIR, REGIME_ORDER


@dataclass(frozen=True)
class Splits:
    """Container for the three sequential splits + the full concatenated view."""
    full:  pd.DataFrame
    train: pd.DataFrame
    val:   pd.DataFrame
    test:  pd.DataFrame


def _read(name: str) -> pd.DataFrame:
    df = pd.read_csv(
        os.path.join(PROCESSED_DIR, f"{name}.csv"),
        index_col=0,
        parse_dates=True,
    )
    if "regime" in df.columns:
        df["regime"] = pd.Categorical(df["regime"], categories=REGIME_ORDER, ordered=True)
    return df


def load_splits() -> Splits:
    """Load the ffill-imputed train/val/test splits."""
    return Splits(
        full  = _read("full_dataset_imputed"),
        train = _read("train"),
        val   = _read("val"),
        test  = _read("test"),
    )


def load_with_nans() -> pd.DataFrame:
    """Load the outer-joined feature dataset that PRESERVES the NaN pattern."""
    return _read("full_dataset_with_nans")
