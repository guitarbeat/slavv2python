"""Vertex extraction public package seam."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from typing_extensions import Literal

from .manager import VertexManager
from .painting import paint_vertex_center_image, paint_vertex_image

if TYPE_CHECKING:
    import numpy as np

    from slavv_python.engine.state import StageController
    from slavv_python.schema.results import EnergyResult, VertexSet

logger = logging.getLogger(__name__)


def extract_vertices(
    energy_data: EnergyResult,
    params: dict[str, Any] | None = None,
    *,
    stage_controller: StageController | None = None,
    **kwargs: Any,
) -> VertexSet:
    """Extract candidate vertices from the multi-scale energy field.

    This function identifies local energy extrema (minima or maxima depending on
    energy sign) and filters them using space-suppression kernels. It supports
    resumable execution via an optional StageController.

    Args:
        energy_data: The EnergyResult domain object from the energy stage.
        params: Authoritative configuration dictionary.
        stage_controller: Optional controller for resumable checkpointing.
        **kwargs: Parameter overrides that take precedence over `params`.

    Returns:
        A VertexSet containing the accepted coordinate positions and their scales.

    Note:
        The extraction logic typically uses a greedy paint-and-check algorithm
        (MATLAB-style) to ensure a minimum physical distance between vertices
        based on the detected local vessel radius.
    """
    # Build a consolidated parameters dictionary
    merged_params = {}
    if params is not None:
        merged_params.update(params)
    merged_params.update(kwargs)

    if stage_controller is not None:
        return VertexManager.run_resumable(energy_data, merged_params, stage_controller)
    return VertexManager.run(energy_data, merged_params)


def paint_vertices(
    vertices: VertexSet,
    image_shape: tuple[int, int, int],
    *,
    mode: Literal["body", "center"] = "body",
) -> np.ndarray:
    """Rasterize the Vertex Set into a 3D volume mask.

    Args:
        vertices: The VertexSet domain object.
        image_shape: The target 3D volume shape (Z, Y, X).
        mode: The painting strategy to employ:
            - "body": Renders ellipsoid regions matching the vertex scale and
              physical lumen radius.
            - "center": Renders single-voxel spikes at each vertex coordinate.

    Returns:
        A 3D numpy array (uint16 or bool) representing the vertex mask.

    Raises:
        ValueError: If an unknown painting mode is requested.
    """
    if mode == "body":
        return paint_vertex_image(
            vertices.positions,
            vertices.scales,
            vertices.radii_pixels,
            image_shape,
        )
    if mode == "center":
        return paint_vertex_center_image(
            vertices.positions,
            image_shape,
        )
    raise ValueError(f"Unknown painting mode: {mode}")


__all__ = [
    "VertexManager",
    "extract_vertices",
    "paint_vertex_center_image",
    "paint_vertex_image",
    "paint_vertices",
]
