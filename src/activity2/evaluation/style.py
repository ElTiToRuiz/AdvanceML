# style.py
# ─────────────────────────────────────────────────────────────────────────────
# Shared visual style for ALL Activity-2 plots. Identical palette and
# rcParams to Activity 1 so both decks look like one project.
# Adds REGIME_COLORS (4 classes) and MODEL_COLORS for the Activity-2
# model names.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import matplotlib.pyplot as plt


# ─── Palette (same as activity1) ─────────────────────────────────────────────

BLUE   = "#1F4E79"
LBLUE  = "#2E75B6"
RED    = "#C0392B"
GREEN  = "#1A7A4A"
LGREEN = "#16A34A"
AMBER  = "#D97706"
PURPLE = "#6D28D9"
GRAY   = "#64748B"
LGRAY  = "#CBD5E1"
INK    = "#1E293B"
PAPER  = "#F8FAFC"

PALETTE = [BLUE, LBLUE, GREEN, AMBER, PURPLE, RED, GRAY, LGREEN]


# ─── rcParams ────────────────────────────────────────────────────────────────

_RC = {
    "figure.facecolor":  PAPER,
    "axes.facecolor":    "#FFFFFF",
    "axes.grid":         True,
    "grid.alpha":        0.30,
    "grid.color":        LGRAY,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.labelcolor":   INK,
    "axes.titlecolor":   INK,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "xtick.color":       GRAY,
    "ytick.color":       GRAY,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "font.family":       "DejaVu Sans",
    "legend.framealpha": 0.92,
    "legend.fontsize":   9,
    "legend.edgecolor":  LGRAY,
    "lines.linewidth":   1.5,
    "savefig.facecolor": PAPER,
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
}


def apply_style() -> None:
    plt.rcParams.update(_RC)


# ─── Regime + model colour maps ──────────────────────────────────────────────

REGIME_COLORS = {
    "crash":      "#C0392B",
    "correction": "#E67E22",
    "normal":     "#7F8C8D",
    "rally":      "#27AE60",
}

MODEL_COLORS = {
    "baseline_most_common": LGRAY,
    "baseline_stratified":  GRAY,
    "logreg":               BLUE,
    "random_forest":        GREEN,
    "xgboost":              AMBER,
}


def regime_color(regime: str) -> str:
    return REGIME_COLORS.get(regime, GRAY)


def model_color(name: str) -> str:
    for key, color in MODEL_COLORS.items():
        if name.startswith(key):
            return color
    return GRAY
