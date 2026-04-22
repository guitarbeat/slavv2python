"""Public edge-selection facade for SLAVV."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from ._edge_selection.conflict_painting import (
    _choose_edges_matlab_style,
    _construct_structuring_element_offsets_matlab,
    _offset_coords_matlab,
)


def choose_edges_for_workflow(
    candidates: dict[str, Any],
    vertex_positions: np.ndarray,
    vertex_scales: np.ndarray,
    lumen_radius_pixels_axes: np.ndarray,
    image_shape: tuple[int, int, int],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Route edge cleanup through the maintained workflow-specific chooser."""
    result = _choose_edges_matlab_style(
        candidates,
        vertex_positions.astype(np.float32, copy=False),
        vertex_scales,
        lumen_radius_pixels_axes,
        image_shape,
        params,
    )
    return cast("dict[str, Any]", result)


__all__ = [
    "_choose_edges_matlab_style",
    "_construct_structuring_element_offsets_matlab",
    "_offset_coords_matlab",
    "choose_edges_for_workflow",
]
