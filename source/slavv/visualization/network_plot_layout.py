"""Pure layout helpers for SLAVV network plots."""

from __future__ import annotations

from typing import Any

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
    return f"{AXIS_NAMES[x_axis]} (μm)", f"{AXIS_NAMES[y_axis]} (μm)"


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


def plot_slice_layout(center_in_microns: float, axis: int, x_label: str, y_label: str) -> dict[str, Any]:
    """Return the shared layout options for slice figures."""
    return {
        "title": f"Network Slice at {center_in_microns:.1f} μm along {AXIS_NAMES[axis]}",
        "xaxis_title": x_label,
        "yaxis_title": y_label,
        "showlegend": True,
        "hovermode": "closest",
        "width": 800,
        "height": 600,
    }
