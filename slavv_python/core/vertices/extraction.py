"""Vertex extraction workflow."""

from __future__ import annotations

import logging
from typing import Any

from slavv_python.core.edges.candidate_detection import (
    choose_vertices_matlab_style,
    crop_vertices_matlab_style,
    matlab_vertex_candidates,
)
from slavv_python.core.vertices.results import build_vertices_result, coerce_radius_axes, empty_vertices_result
from slavv_python.core.vertices.selection import sort_vertex_order

logger = logging.getLogger(__name__)


def extract_vertices(energy_data: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Extract vertices as local extrema in the energy field."""
    logger.info("Extracting vertices")

    energy = energy_data["energy"]
    scale_indices = energy_data["scale_indices"]
    lumen_radius_pixels = energy_data["lumen_radius_pixels"]
    lumen_radius_pixels_axes = coerce_radius_axes(
        lumen_radius_pixels,
        energy_data.get("lumen_radius_pixels_axes"),
    )
    energy_sign = energy_data.get("energy_sign", -1.0)
    lumen_radius_microns = energy_data["lumen_radius_microns"]

    energy_upper_bound = params.get("energy_upper_bound", 0.0)
    space_strel_apothem = params.get("space_strel_apothem", 1)
    length_dilation_ratio = params.get("length_dilation_ratio", 1.0)
    max_voxels_per_node = params.get("max_voxels_per_node", 6000)
    vertex_positions, vertex_scales, vertex_energies = matlab_vertex_candidates(
        energy,
        scale_indices,
        energy_sign,
        energy_upper_bound,
        space_strel_apothem,
        lumen_radius_pixels_axes[0],
        max_voxels_per_node,
    )

    vertex_positions, vertex_scales, vertex_energies = crop_vertices_matlab_style(
        vertex_positions,
        vertex_scales,
        vertex_energies,
        energy.shape,
        lumen_radius_pixels_axes,
        length_dilation_ratio,
    )

    if len(vertex_positions) == 0:
        logger.info("Extracted 0 vertices")
        return empty_vertices_result()

    sort_indices = sort_vertex_order(vertex_positions, vertex_energies, energy.shape, energy_sign)
    vertex_positions = vertex_positions[sort_indices]
    vertex_scales = vertex_scales[sort_indices]
    vertex_energies = vertex_energies[sort_indices]

    chosen_mask = choose_vertices_matlab_style(
        vertex_positions,
        vertex_scales,
        energy.shape,
        lumen_radius_pixels_axes,
        length_dilation_ratio,
    )
    vertex_positions = vertex_positions[chosen_mask]
    vertex_scales = vertex_scales[chosen_mask]
    vertex_energies = vertex_energies[chosen_mask]

    logger.info("Extracted %s vertices", len(vertex_positions))

    return build_vertices_result(
        vertex_positions,
        vertex_scales,
        vertex_energies,
        lumen_radius_pixels,
        lumen_radius_microns,
    )


__all__ = ["extract_vertices"]
