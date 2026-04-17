from __future__ import annotations

import logging
from typing import Any

import numpy as np

try:
    from ..utils import calculate_path_length
except ImportError:  # pragma: no cover - fallback for direct execution
    from slavv.utils import calculate_path_length

logger = logging.getLogger(__name__)


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

        params = {**self.vertex_parameters, **(parameters or {})}
        keep_mask = np.ones(len(positions), dtype=bool)

        keep_mask &= energies < params.get("vertex_energy_threshold", -0.1)
        keep_mask &= radii > params.get("min_vertex_radius", 0.5)

        image_shape = energy_data.get("image_shape", (100, 100, 50))
        boundary_margin = params.get("boundary_margin", 5)
        for dim in range(3):
            keep_mask &= positions[:, dim] > boundary_margin
            keep_mask &= positions[:, dim] < image_shape[dim] - boundary_margin

        energy_field = energy_data["energy"]
        contrast_threshold = params.get("contrast_threshold", 0.1)

        for i, pos in enumerate(positions):
            if not keep_mask[i]:
                continue
            try:
                y, x, z = pos.astype(int)
                neighborhood_size = max(1, int(radii[i]))
                y_min = max(0, y - neighborhood_size)
                y_max = min(image_shape[0], y + neighborhood_size + 1)
                x_min = max(0, x - neighborhood_size)
                x_max = min(image_shape[1], x + neighborhood_size + 1)
                z_min = max(0, z - neighborhood_size)
                z_max = min(image_shape[2], z + neighborhood_size + 1)
                local_energy = energy_field[y_min:y_max, x_min:x_max, z_min:z_max]
                if local_energy.size > 0 and np.std(local_energy) < contrast_threshold:
                    keep_mask[i] = False
            except (IndexError, ValueError):
                keep_mask[i] = False

        kept_indices = np.where(keep_mask)[0]
        curated_vertices = {
            "positions": positions[kept_indices],
            "scales": scales[kept_indices],
            "energies": energies[kept_indices],
            "radii_pixels": radii[kept_indices],
            "radii_microns": vertices.get("radii_microns", vertices.get("radii", []))[kept_indices],
            "radii": vertices.get("radii_microns", vertices.get("radii", []))[kept_indices],
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
        params = {**self.edge_parameters, **(parameters or {})}
        keep_mask = np.ones(len(edge_traces), dtype=bool)

        min_length = params.get("min_edge_length", 2.0)
        for i, trace in enumerate(edge_traces):
            if len(trace) < 2:
                keep_mask[i] = False
                continue
            if calculate_path_length(np.array(trace)) < min_length:
                keep_mask[i] = False

        max_tortuosity = params.get("max_edge_tortuosity", 3.0)
        for i, trace in enumerate(edge_traces):
            if not keep_mask[i] or len(trace) < 2:
                continue
            trace = np.array(trace)
            edge_length = calculate_path_length(trace)
            euclidean_distance = np.linalg.norm(trace[-1] - trace[0])
            if euclidean_distance > 0 and edge_length / euclidean_distance > max_tortuosity:
                keep_mask[i] = False

        vertex_positions = vertices["positions"]
        max_connection_distance = params.get("max_connection_distance", 5.0)
        original_indices = vertices.get("original_indices")
        if original_indices is None:
            original_indices = np.arange(len(vertices["positions"]))
        original_to_curated_idx = {orig_idx: i for i, orig_idx in enumerate(original_indices)}

        for i, (trace, connection) in enumerate(zip(edge_traces, edge_connections)):
            if not keep_mask[i]:
                continue
            start_vertex, end_vertex = connection
            if start_vertex is not None:
                if start_vertex not in original_to_curated_idx:
                    keep_mask[i] = False
                    continue
                start_pos = vertex_positions[original_to_curated_idx[start_vertex]]
                if np.linalg.norm(start_pos - np.array(trace[0])) > max_connection_distance:
                    keep_mask[i] = False
                    continue
            if end_vertex is not None:
                if end_vertex not in original_to_curated_idx:
                    keep_mask[i] = False
                    continue
                end_pos = vertex_positions[original_to_curated_idx[end_vertex]]
                if np.linalg.norm(end_pos - np.array(trace[-1])) > max_connection_distance:
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
