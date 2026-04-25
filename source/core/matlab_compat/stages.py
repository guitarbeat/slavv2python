"""Thin MATLAB-named wrappers over the maintained Python implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

from source.analysis._geometry.trace_ops import get_edge_metric as _get_edge_metric
from source.core._edges.bridge_vertices import add_vertices_to_edges_matlab_style
from source.core.edge_selection import choose_edges_for_workflow
from source.core.edges import extract_edges
from source.core.energy import calculate_energy_field
from source.core.graph import (
    _matlab_get_network_v190,
    _matlab_get_strand_objects,
    _matlab_sort_network_v180,
)
from source.core.vertices import extract_vertices


def get_energy_v202(
    image: np.ndarray,
    params: dict[str, Any],
    *,
    get_chunking_lattice_func=None,
) -> dict[str, Any]:
    """Mirror MATLAB ``get_energy_V202`` through the maintained energy facade."""
    return calculate_energy_field(image, params, get_chunking_lattice_func)


def get_vertices_v200(
    energy_data: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Mirror MATLAB ``get_vertices_V200`` through the maintained vertex facade."""
    return extract_vertices(energy_data, params)


def get_edges_v300(
    energy_data: dict[str, Any],
    vertices: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Mirror MATLAB ``get_edges_V300`` through the maintained edge facade."""
    return extract_edges(energy_data, vertices, params)


def choose_edges_v200(
    candidates: dict[str, Any],
    vertex_positions: np.ndarray,
    vertex_scales: np.ndarray,
    lumen_radius_microns: np.ndarray,
    lumen_radius_pixels_axes: np.ndarray,
    image_shape: tuple[int, int, int],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Mirror MATLAB ``choose_edges_V200`` through the maintained chooser."""
    return choose_edges_for_workflow(
        candidates,
        vertex_positions,
        vertex_scales,
        lumen_radius_microns,
        lumen_radius_pixels_axes,
        image_shape,
        params,
    )


def add_vertices_to_edges(
    chosen_edges: dict[str, Any],
    vertices: dict[str, Any],
    *,
    energy: np.ndarray,
    scale_indices: np.ndarray | None,
    microns_per_voxel: np.ndarray,
    lumen_radius_microns: np.ndarray,
    lumen_radius_pixels_axes: np.ndarray,
    size_of_image: tuple[int, int, int],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Mirror MATLAB ``add_vertices_to_edges`` through the maintained bridge path."""
    return add_vertices_to_edges_matlab_style(
        chosen_edges,
        vertices,
        energy=energy,
        scale_indices=scale_indices,
        microns_per_voxel=microns_per_voxel,
        lumen_radius_microns=lumen_radius_microns,
        lumen_radius_pixels_axes=lumen_radius_pixels_axes,
        size_of_image=size_of_image,
        params=params,
    )


def get_network_v190(
    edge_connections: np.ndarray,
    n_vertices: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Mirror MATLAB ``get_network_V190`` topology decomposition."""
    return _matlab_get_network_v190(edge_connections, n_vertices)


def sort_network_v180(
    edge_connections: np.ndarray,
    end_vertices_in_strands: list[np.ndarray],
    edge_indices_in_strands: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Mirror MATLAB ``sort_network_V180`` edge ordering."""
    return _matlab_sort_network_v180(
        edge_connections,
        end_vertices_in_strands,
        edge_indices_in_strands,
    )


def get_strand_objects(
    edge_traces: list[np.ndarray],
    edge_scale_traces: list[np.ndarray],
    edge_energy_traces: list[np.ndarray],
    edge_indices_in_strands: list[np.ndarray],
    edge_backwards_in_strands: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Mirror MATLAB ``get_strand_objects`` over the maintained graph helpers."""
    return _matlab_get_strand_objects(
        edge_traces,
        edge_scale_traces,
        edge_energy_traces,
        edge_indices_in_strands,
        edge_backwards_in_strands,
    )


def get_edge_metric(
    trace: np.ndarray,
    energy: np.ndarray | None = None,
    *,
    method: str = "max_energy",
) -> float:
    """Mirror MATLAB ``get_edge_metric`` through the maintained trace metric helper."""
    return float(_get_edge_metric(trace, energy=energy, method=method))


__all__ = [
    "add_vertices_to_edges",
    "choose_edges_v200",
    "get_edge_metric",
    "get_edges_v300",
    "get_energy_v202",
    "get_network_v190",
    "get_strand_objects",
    "get_vertices_v200",
    "sort_network_v180",
]
