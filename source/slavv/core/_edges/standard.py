"""Standard non-resumable edge extraction."""

from __future__ import annotations

import logging
from typing import Any, Callable, cast

import numpy as np
from scipy.spatial import cKDTree

from ..edge_candidates import (
    _params_with_matlab_parity_edge_budget,
    _params_with_matlab_parity_watershed_candidate_mode,
)

logger = logging.getLogger(__name__)


def extract_edges(
    energy_data: dict[str, Any],
    vertices: dict[str, Any],
    params: dict[str, Any],
    *,
    empty_edges_result: Callable[[np.ndarray], dict[str, Any]],
    paint_vertex_center_image: Callable[[np.ndarray, tuple[int, ...]], np.ndarray],
    use_matlab_frontier_tracer: Callable[[dict[str, Any], dict[str, Any]], bool],
    generate_edge_candidates_matlab_frontier: Callable[..., dict[str, Any]],
    finalize_matlab_parity_candidates: Callable[..., dict[str, Any]],
    generate_edge_candidates: Callable[..., dict[str, Any]],
    choose_edges_for_workflow: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Extract edges by tracing from vertices through energy field."""
    logger.info("Extracting edges")

    energy = energy_data["energy"]
    vertex_positions = vertices["positions"]
    vertex_scales = vertices["scales"]
    lumen_radius_pixels = energy_data["lumen_radius_pixels"]
    lumen_radius_microns = energy_data["lumen_radius_microns"]
    energy_sign = energy_data.get("energy_sign", -1.0)

    microns_per_voxel = np.array(params.get("microns_per_voxel", [1.0, 1.0, 1.0]), dtype=float)
    scale_indices = energy_data.get("scale_indices")

    if len(vertex_positions) == 0:
        logger.info("Extracted 0 edges")
        return cast("dict[str, Any]", empty_edges_result(vertex_positions))

    lumen_radius_pixels_axes = energy_data["lumen_radius_pixels_axes"]
    logger.info("Creating vertex center lookup image...")
    vertex_center_image = paint_vertex_center_image(vertex_positions, energy.shape)
    logger.info("Vertex center lookup image created")

    vertex_positions_microns = vertex_positions * microns_per_voxel
    tree = cKDTree(vertex_positions_microns)
    max_vertex_radius = np.max(lumen_radius_microns) if len(lumen_radius_microns) > 0 else 0.0
    max_search_radius = max_vertex_radius * 5.0
    use_frontier = use_matlab_frontier_tracer(energy_data, params)
    params_for_workflow = params
    if use_frontier:
        params_for_workflow = _params_with_matlab_parity_edge_budget(
            params_for_workflow, edge_budget=2
        )
        params_for_workflow = _params_with_matlab_parity_watershed_candidate_mode(
            params_for_workflow,
            candidate_mode="remaining_origin_contacts",
        )
    if use_frontier:
        candidates = generate_edge_candidates_matlab_frontier(
            energy,
            scale_indices,
            vertex_positions,
            vertex_scales,
            lumen_radius_microns,
            microns_per_voxel,
            vertex_center_image,
            params_for_workflow,
        )
        candidates = finalize_matlab_parity_candidates(
            candidates,
            energy,
            scale_indices,
            vertex_positions,
            energy_sign,
            params_for_workflow,
            microns_per_voxel,
        )
    else:
        candidates = generate_edge_candidates(
            energy,
            scale_indices,
            vertex_positions,
            vertex_scales,
            lumen_radius_pixels,
            lumen_radius_microns,
            microns_per_voxel,
            vertex_center_image,
            tree,
            max_search_radius,
            params_for_workflow,
            energy_sign,
        )
    chosen = choose_edges_for_workflow(
        candidates,
        vertex_positions,
        vertex_scales,
        lumen_radius_pixels_axes,
        energy.shape,
        params_for_workflow,
    )
    logger.info(
        "Extracted %d chosen edges from %d traced candidates",
        len(chosen["traces"]),
        chosen["diagnostics"]["candidate_traced_edge_count"],
    )
    return chosen
