"""Shared plotting helpers for SLAVV network visualizations."""

from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

NETWORK_COLOR_SCHEMES = {
    "energy": "RdBu_r",
    "depth": "Viridis",
    "strand_id": "Set3",
    "radius": "Plasma",
    "length": "Cividis",
    "random": "Set1",
}


def map_values_to_colors(values: np.ndarray, colorscale: str) -> list[str]:
    """Map numeric values to colors using a Plotly colorscale."""
    if len(values) == 0:
        return []

    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin:
        normalized = np.zeros_like(values, dtype=float)
    else:
        normalized = (values - vmin) / (vmax - vmin)

    return [px.colors.sample_colorscale(colorscale, float(v))[0] for v in normalized]


def add_colorbar(
    fig: go.Figure,
    values: np.ndarray,
    colorscale: str,
    title: str,
    is_3d: bool = False,
) -> None:
    """Add a Plotly colorbar for the provided numeric values."""
    if values is None or len(values) == 0:
        return

    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if vmin == vmax:
        vmax = vmin + 1.0

    marker = {
        "colorscale": colorscale,
        "cmin": vmin,
        "cmax": vmax,
        "color": [vmin],
        "showscale": True,
        "colorbar": {"title": title},
    }

    trace_kwargs = {
        "mode": "markers",
        "marker": marker,
        "showlegend": False,
        "hoverinfo": "none",
    }
    if is_3d:
        fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], **trace_kwargs))
    else:
        fig.add_trace(go.Scatter(x=[None], y=[None], **trace_kwargs))
