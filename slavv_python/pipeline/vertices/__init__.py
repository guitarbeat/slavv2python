"""Vertex extraction public package seam."""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING
from typing_extensions import Literal
import numpy as np

from .manager import VertexManager
from .painting import paint_vertex_center_image, paint_vertex_image

if TYPE_CHECKING:
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
    """
    Extract a localized set of vertices from the 3D vascular energy volume.
    
    Supports:
      1. Pipeline execution: extract_vertices(energy, params)
      2. Direct default call: extract_vertices(energy)
      3. Clean overrides: extract_vertices(energy, space_strel_apothem=2)
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
    """
    Paint the extracted Vertex Set into a 3D volume mask for down-stream analysis.
    
    Args:
        vertices: The VertexSet domain object.
        image_shape: The target 3D volume shape.
        mode: The painting strategy:
            - "body": ellipsoid regions matching scales/lumen radii.
            - "center": single-voxel coordinate spikes.
    """
    if mode == "body":
        return paint_vertex_image(
            vertices.positions,
            vertices.scales,
            vertices.radii_pixels,
            image_shape,
        )
    elif mode == "center":
        return paint_vertex_center_image(
            vertices.positions,
            image_shape,
        )
    else:
        raise ValueError(f"Unknown painting mode: {mode}")


__all__ = [
    "VertexManager",
    "extract_vertices",
    "paint_vertices",
    "paint_vertex_center_image",
    "paint_vertex_image",
]
