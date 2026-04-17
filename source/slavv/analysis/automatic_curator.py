from __future__ import annotations

import logging
from typing import Any

import numpy as np

try:
    from ..utils import calculate_path_length
except ImportError:  # pragma: no cover - fallback for direct execution
    from slavv.utils import calculate_path_length

logger = logging.getLogger(__name__)


def _merge_parameters(defaults: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    return {**defaults, **(overrides or {})}


def _within_boundary_mask(
    positions: np.ndarray, image_shape: tuple[int, int, int], boundary_margin: float
) -> np.ndarray:
    keep_mask = np.ones(len(positions), dtype=bool)
    for dim in range(3):
        keep_mask &= positions[:, dim] > boundary_margin
        keep_mask &= positions[:, dim] < image_shape[dim] - boundary_margin
    return keep_mask


def _has_local_contrast(
    pos: np.ndarray,
    radius: float,
    energy_field: np.ndarray,
    image_shape: tuple[int, int, int],
    contrast_threshold: float,
) -> bool:
    try:
        y, x, z = pos.astype(int)
        neighborhood_size = max(1, int(radius))
        y_min = max(0, y - neighborhood_size)
        y_max = min(image_shape[0], y + neighborhood_size + 1)
        x_min = max(0, x - neighborhood_size)
        x_max = min(image_shape[1], x + neighborhood_size + 1)
        z_min = max(0, z - neighborhood_size)
        z_max = min(image_shape[2], z + neighborhood_size + 1)
        local_energy = energy_field[y_min:y_max, x_min:x_max, z_min:z_max]
    except (IndexError, ValueError):
        return False
    return local_energy.size == 0 or np.std(local_energy) >= contrast_threshold


def _vertex_output_radii(vertices: dict[str, Any]) -> np.ndarray:
    return vertices.get("radii_microns", vertices.get("radii", []))


def _path_meets_length_requirement(trace: np.ndarray, min_length: float) -> bool:
    return len(trace) >= 2 and calculate_path_length(trace) >= min_length


def _path_meets_tortuosity_requirement(trace: np.ndarray, max_tortuosity: float) -> bool:
    if len(trace) < 2:
        return False
    edge_length = calculate_path_length(trace)
    euclidean_distance = np.linalg.norm(trace[-1] - trace[0])
    return euclidean_distance <= 0 or edge_length / euclidean_distance <= max_tortuosity


def _build_original_to_curated_index(vertices: dict[str, Any]) -> dict[int, int]:
    original_indices = vertices.get("original_indices")
    if original_indices is None:
        original_indices = np.arange(len(vertices["positions"]))
    return {int(orig_idx): i for i, orig_idx in enumerate(original_indices)}


def _endpoint_matches_vertex(
    trace_point: np.ndarray,
    vertex_index: Any,
    vertex_positions: np.ndarray,
    original_to_curated_idx: dict[int, int],
    max_connection_distance: float,
) -> bool:
    if vertex_index is None:
        return True
    if vertex_index not in original_to_curated_idx:
        return False
    vertex_pos = vertex_positions[original_to_curated_idx[vertex_index]]
    return np.linalg.norm(vertex_pos - trace_point) <= max_connection_distance


class AutomaticCurator:
    """Automatic curation using heuristic rules (no ML training required)."""

    def __init__(
        self,
        vertex_parameters: dict[str, Any] | None = None,
        edge_parameters: dict[str, Any] | None = None,
    ) -> None:
        self.vertex_parameters = vertex_parameters or {}
        self.edge_parameters = edge_parameters or {}

    def curate_vertices_automatic(
        self, vertices: dict[str, Any], energy_data: dict[str, Any], parameters: dict[str, Any]
    ) -> dict[str, Any]:
        logger.info("Performing automatic vertex curation")

        positions = vertices["positions"]
        energies = vertices["energies"]
        scales = vertices["scales"]
        radii = vertices.get("radii_pixels", vertices.get("radii", []))

        params = _merge_parameters(self.vertex_parameters, parameters)
        image_shape = energy_data.get("image_shape", (100, 100, 50))
        boundary_margin = params.get("boundary_margin", 5)
        keep_mask = energies < params.get("vertex_energy_threshold", -0.1)
        keep_mask &= radii > params.get("min_vertex_radius", 0.5)
        keep_mask &= _within_boundary_mask(positions, image_shape, boundary_margin)

        energy_field = energy_data["energy"]
        contrast_threshold = params.get("contrast_threshold", 0.1)

        for i, pos in enumerate(positions):
            if not keep_mask[i]:
                continue
            if not _has_local_contrast(
                pos,
                float(radii[i]),
                energy_field,
                image_shape,
                contrast_threshold,
            ):
                keep_mask[i] = False

        kept_indices = np.where(keep_mask)[0]
        output_radii = _vertex_output_radii(vertices)
        curated_vertices = {
            "positions": positions[kept_indices],
            "scales": scales[kept_indices],
            "energies": energies[kept_indices],
            "radii_pixels": radii[kept_indices],
            "radii_microns": output_radii[kept_indices],
            "radii": output_radii[kept_indices],
            "original_indices": kept_indices,
        }
        logger.info(f"Automatic vertex curation: {len(positions)} -> {len(kept_indices)} vertices")
        return curated_vertices

    def curate_edges_automatic(
        self, edges: dict[str, Any], vertices: dict[str, Any], parameters: dict[str, Any]
    ) -> dict[str, Any]:
        logger.info("Performing automatic edge curation")

        edge_traces = edges["traces"]
        edge_connections = edges["connections"]
        params = _merge_parameters(self.edge_parameters, parameters)
        keep_mask = np.ones(len(edge_traces), dtype=bool)

        min_length = params.get("min_edge_length", 2.0)
        for i, trace in enumerate(edge_traces):
            if not _path_meets_length_requirement(np.array(trace), min_length):
                keep_mask[i] = False

        max_tortuosity = params.get("max_edge_tortuosity", 3.0)
        for i, trace in enumerate(edge_traces):
            if keep_mask[i] and not _path_meets_tortuosity_requirement(
                np.array(trace), max_tortuosity
            ):
                keep_mask[i] = False

        vertex_positions = vertices["positions"]
        max_connection_distance = params.get("max_connection_distance", 5.0)
        original_to_curated_idx = _build_original_to_curated_index(vertices)

        for i, (trace, connection) in enumerate(zip(edge_traces, edge_connections)):
            if not keep_mask[i]:
                continue
            start_vertex, end_vertex = connection
            trace_arr = np.array(trace)
            if not _endpoint_matches_vertex(
                trace_arr[0],
                start_vertex,
                vertex_positions,
                original_to_curated_idx,
                max_connection_distance,
            ) or not _endpoint_matches_vertex(
                trace_arr[-1],
                end_vertex,
                vertex_positions,
                original_to_curated_idx,
                max_connection_distance,
            ):
                keep_mask[i] = False

        kept_indices = np.where(keep_mask)[0]
        curated_edges = {
            "traces": [edge_traces[i] for i in kept_indices],
            "connections": [edge_connections[i] for i in kept_indices],
            "original_indices": kept_indices,
            "vertex_positions": edges["vertex_positions"],
        }
        logger.info(f"Automatic edge curation: {len(edge_traces)} -> {len(kept_indices)} edges")
        return curated_edges
