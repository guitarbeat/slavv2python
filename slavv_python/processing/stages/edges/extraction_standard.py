"""Standard non-resumable edge extraction."""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
from scipy.spatial import cKDTree

from slavv_python.processing.stages.edges.common import resolve_lumen_radius_pixels_axes
from slavv_python.schema.results import EdgeSet, EnergyResult, VertexSet

logger = logging.getLogger(__name__)


def extract_edges(
    energy_data: EnergyResult,
    vertices: VertexSet,
    params: dict[str, Any],
    *,
    empty_edges_result: Callable[[np.ndarray], dict[str, Any]],
    paint_vertex_center_image: Callable[[np.ndarray, tuple[int, ...]], np.ndarray],
    paint_vertex_image: Callable[[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]], np.ndarray],
    use_matlab_frontier_tracer: Callable[[EnergyResult, dict[str, Any]], bool],
    generate_edge_candidates_matlab_frontier: Callable[..., dict[str, Any]],
    finalize_matlab_parity_candidates: Callable[..., dict[str, Any]],
    generate_edge_candidates: Callable[..., dict[str, Any]],
    choose_edges_for_workflow: Callable[..., dict[str, Any]],
    add_vertices_to_edges_matlab_style: Callable[..., dict[str, Any]],
    finalize_edges_matlab_style: Callable[..., dict[str, Any]],
) -> EdgeSet:
    """Extract edges by tracing from vertices through energy field."""
    logger.info("Extracting edges")

    energy = energy_data.energy
    vertex_positions = vertices.positions
    vertex_scales = vertices.scales
    lumen_radius_microns = energy_data.lumen_radius_microns
    energy_sign = energy_data.extra.get("energy_sign", -1.0)

    microns_per_voxel = np.array(params.get("microns_per_voxel", [1.0, 1.0, 1.0]), dtype=float)
    scale_indices = energy_data.scale_indices

    if len(vertex_positions) == 0:
        logger.info("Extracted 0 edges")
        # For simplicity, we create from empty dict or just use EdgeSet.create
        empty_dict = empty_edges_result(vertex_positions)
        return EdgeSet.from_dict(empty_dict)

    lumen_radius_pixels_axes = resolve_lumen_radius_pixels_axes(
        energy_data,
        microns_per_voxel,
    )
    logger.info("Creating vertex center lookup image...")
    vertex_center_image = paint_vertex_center_image(vertex_positions, energy.shape)
    logger.info("Vertex center lookup image created")
    use_frontier = use_matlab_frontier_tracer(energy_data, params)
    vertex_image: np.ndarray | None = None
    if not use_frontier:
        logger.info("Creating painted vertex occupancy image...")
        vertex_image = paint_vertex_image(
            vertex_positions,
            vertex_scales,
            lumen_radius_pixels_axes,
            energy.shape,
        )
        logger.info("Painted vertex occupancy image created")

    vertex_positions_microns = vertex_positions * microns_per_voxel
    tree = cKDTree(vertex_positions_microns)
    max_vertex_radius = np.max(lumen_radius_microns) if len(lumen_radius_microns) > 0 else 0.0
    max_search_radius = max_vertex_radius * 5.0
    if use_frontier:
        candidates = generate_edge_candidates_matlab_frontier(
            energy,
            scale_indices,
            vertex_positions,
            vertex_scales,
            lumen_radius_microns,
            microns_per_voxel,
            vertex_center_image,
            params,
        )
        candidates = finalize_matlab_parity_candidates(
            candidates,
            energy,
            scale_indices,
            vertex_positions,
            energy_sign,
            params,
            microns_per_voxel,
        )
    else:
        candidates = generate_edge_candidates(
            energy=energy,
            scale_indices=scale_indices,
            vertex_positions=vertex_positions,
            vertex_scales=vertex_scales,
            lumen_radius_pixels=energy_data.lumen_radius_pixels,
            lumen_radius_microns=lumen_radius_microns,
            microns_per_voxel=microns_per_voxel,
            vertex_center_image=vertex_center_image,
            vertex_image=vertex_image,
            tree=tree,
            max_search_radius=max_search_radius,
            params=params,
            energy_sign=energy_sign,
        )
    chosen = choose_edges_for_workflow(
        candidates,
        vertex_positions,
        vertex_scales,
        lumen_radius_microns,
        lumen_radius_pixels_axes,
        energy.shape,
        params,
    )
    if use_frontier:
        chosen = add_vertices_to_edges_matlab_style(
            chosen,
            vertices.to_dict(), # Back to dict for legacy internal functions if needed, but let's see
            energy=energy,
            scale_indices=scale_indices,
            microns_per_voxel=microns_per_voxel,
            lumen_radius_microns=lumen_radius_microns,
            lumen_radius_pixels_axes=lumen_radius_pixels_axes,
            size_of_image=energy.shape,
            params=params,
        )
    chosen = finalize_edges_matlab_style(
        chosen,
        lumen_radius_microns=lumen_radius_microns,
        microns_per_voxel=microns_per_voxel,
        size_of_image=energy.shape,
    )
    chosen["lumen_radius_microns"] = np.asarray(lumen_radius_microns, dtype=np.float32).copy()
    logger.info(
        "Extracted %d chosen edges from %d traced candidates",
        len(chosen["traces"]),
        chosen["diagnostics"]["candidate_traced_edge_count"],
    )
    return EdgeSet.from_dict(chosen)
