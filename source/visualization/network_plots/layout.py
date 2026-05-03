"""Pure layout helpers for SLAVV network plots."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

AXIS_NAMES = ("Y", "X", "Z")


def select_plot_axes(axis: int) -> tuple[int, int]:
    """Return the x/y axes for a projection that omits ``axis``."""
    if axis == 2:
        return 1, 0

    axes = [0, 1, 2]
    axes.remove(axis)
    return axes[0], axes[1]


def axis_labels(x_axis: int, y_axis: int) -> tuple[str, str]:
    """Return human-readable axis labels in microns."""
    return f"{AXIS_NAMES[x_axis]} (\u03bcm)", f"{AXIS_NAMES[y_axis]} (\u03bcm)"


def plot_2d_layout(projection_axis: int, x_label: str, y_label: str) -> dict[str, Any]:
    """Return the shared layout options for 2D network figures."""
    return {
        "title": f"2D Vascular Network (Projection along {AXIS_NAMES[projection_axis]})",
        "xaxis_title": x_label,
        "yaxis_title": y_label,
        "showlegend": True,
        "hovermode": "closest",
        "width": 800,
        "height": 600,
    }


def plot_slice_layout(
        center_in_microns: float,
        axis: int,
        x_label: str,
        y_label: str,
) -> dict[str, Any]:
    """Return the shared layout options for slice figures."""
    return {
        "title": f"Network Slice at {center_in_microns:.1f} \u03bcm along {AXIS_NAMES[axis]}",
        "xaxis_title": x_label,
        "yaxis_title": y_label,
        "showlegend": True,
        "hovermode": "closest",
        "width": 800,
        "height": 600,
    }


def plot_3d_layout(
        title: str,
        *,
        showlegend: bool,
        width: int = 800,
        height: int = 600,
        updatemenus: list[dict[str, Any]] | None = None,
        sliders: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return shared layout options for 3D network figures."""
    layout: dict[str, Any] = {
        "title": title,
        "scene": {
            "xaxis_title": "X (\u03bcm)",
            "yaxis_title": "Y (\u03bcm)",
            "zaxis_title": "Z (\u03bcm)",
            "aspectmode": "data",
        },
        "showlegend": showlegend,
        "width": width,
        "height": height,
    }
    if updatemenus is not None:
        layout["updatemenus"] = updatemenus
    if sliders is not None:
        layout["sliders"] = sliders
    return layout


def distribution_layout(
        title: str,
        xaxis_title: str,
        yaxis_title: str,
        *,
        width: int,
        height: int,
        showlegend: bool = False,
) -> dict[str, Any]:
    """Return shared layout options for 1D histogram/bar figures."""
    return {
        "title": title,
        "xaxis_title": xaxis_title,
        "yaxis_title": yaxis_title,
        "showlegend": showlegend,
        "width": width,
        "height": height,
    }


def summary_dashboard_layout() -> dict[str, Any]:
    """Return shared layout options for the processing summary dashboard."""
    return {
        "title": "SLAVV Processing Summary Dashboard",
        "showlegend": False,
        "height": 600,
        "width": 1000,
    }


def empty_figure(message: str) -> go.Figure:
    """Return a figure with a centered annotation for empty states."""
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False)
    return fig
