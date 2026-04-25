# style.py
# ─────────────────────────────────────────────────────────────────────────────
# Shared visual style for ALL Activity-1 plots (EDA + model evaluation +
# backtesting). Centralised here so the whole project has a consistent
# look — same fonts, same palette, same grid — across every figure that
# ends up in the report or the slide deck.
#
# Usage:
#     from .style import apply_style, BLUE, GREEN, RED, ...
#     apply_style()        # call once at the top of any plotting module
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import matplotlib.pyplot as plt


# ─── Palette ─────────────────────────────────────────────────────────────────

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

# Default ordered palette for multi-series plots. Keeps colors
# consistent across model comparisons.
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
    "axes.labelweight":  "regular",
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
    """Apply the project-wide matplotlib style. Call once per module."""
    plt.rcParams.update(_RC)


def color_for(index: int) -> str:
    """Pick a palette colour by index (wraps around)."""
    return PALETTE[index % len(PALETTE)]


# ─── Helper: model-name → consistent colour ──────────────────────────────────

# Stable colour assignment so each model keeps the same colour across
# plots — easier to read across slides.
MODEL_COLORS = {
    "naive":              GRAY,
    "seasonal_naive_m5":  LGRAY,
    "drift_mean":         AMBER,
    "ma_20":              PURPLE,
    "sarimax":            RED,
    "lstm":               BLUE,
    "chronos2":           GREEN,
    "chronos2_ft":        LGREEN,
    "actual":             INK,
}


def model_color(name: str) -> str:
    """Return the canonical colour for a model name (prefix-matched)."""
    for key, color in MODEL_COLORS.items():
        if name.startswith(key):
            return color
    return GRAY
