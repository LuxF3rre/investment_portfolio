"""Catppuccin Macchiato color palette for Plotly charts."""

from typing import Final

# Primary chart colors (from .streamlit/config.toml Catppuccin Macchiato)
BLUE: Final = "#8aadf4"
RED: Final = "#ed8796"
TEAL: Final = "#8bd5ca"
SAPPHIRE: Final = "#7dc4e4"
GREEN: Final = "#a6da95"
OVERLAY1: Final = "#8087a2"

# Derived alpha variants for path fans / translucent traces
BLUE_ALPHA: Final = "rgba(138,173,244,0.15)"
GREEN_ALPHA: Final = "rgba(166,218,149,0.15)"
RED_ALPHA: Final = "rgba(237,135,150,0.15)"

# Plotly color scales — Catppuccin Macchiato aligned
# Diverging: correlation matrices (-1 → 0 → +1)
COLORSCALE_DIVERGING: Final = [
    [0.0, "#ed8796"],  # Red
    [0.5, "#494d64"],  # Surface1 (neutral)
    [1.0, "#a6da95"],  # Green
]

# Sequential: price surfaces, value heatmaps (low → high)
COLORSCALE_SEQUENTIAL: Final = [
    [0.0, "#181926"],  # Crust
    [0.33, "#7dc4e4"],  # Sapphire
    [0.67, "#c6a0f6"],  # Mauve
    [1.0, "#f4dbd6"],  # Rosewater
]
