from __future__ import annotations

import logging
from typing import Any

import numpy as np

try:
    from ..utils import calculate_path_length
except ImportError:  # pragma: no cover - fallback for direct execution
    from slavv.utils import calculate_path_length

logger = logging.getLogger(__name__)


class DrewsCurator:
    """
    Experimental curator based on legacy 'edge_curator_Drews.m'.

    This implements specific heuristic rules used in earlier versions of the pipeline
    for pruning edges based on tortuosity, min-length relative to radius, and flow properties.
    """

    def __init__(
        self,
        min_length_radius_ratio: float = 2.0,
        max_tortuosity: float = 3.5,
        max_endpoint_gap: float = 5.0,
    ):
        self.min_length_radius_ratio = float(min_length_radius_ratio)
        self.max_tortuosity = float(max_tortuosity)
        self.max_endpoint_gap = float(max_endpoint_gap)

    def curate(
        self,
        edges: dict[str, Any],
        vertices: dict[str, Any],
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        traces = edges.get("traces", [])
        connections = edges.get("connections", [])
        n_edges = len(traces)
        if n_edges == 0:
            return edges

        params = parameters or {}
        min_length_radius_ratio = float(
            params.get("min_length_radius_ratio", self.min_length_radius_ratio)
        )
        max_tortuosity = float(params.get("max_tortuosity", self.max_tortuosity))
        max_endpoint_gap = float(params.get("max_endpoint_gap", self.max_endpoint_gap))

        vertex_positions = np.asarray(vertices.get("positions", []), dtype=float)
        vertex_radii = np.asarray(
            vertices.get("radii_microns", vertices.get("radii_pixels", vertices.get("radii", []))),
            dtype=float,
        )

        keep_mask = np.ones(n_edges, dtype=bool)
        for i, trace in enumerate(traces):
            trace_arr = np.asarray(trace, dtype=float)
            if trace_arr.ndim != 2 or len(trace_arr) < 2:
                keep_mask[i] = False
                continue

            edge_length = calculate_path_length(trace_arr)
            euclidean = float(np.linalg.norm(trace_arr[-1] - trace_arr[0]))
            tortuosity = edge_length / (euclidean + 1e-10)
            if tortuosity > max_tortuosity:
                keep_mask[i] = False
                continue

            avg_radius = 0.0
            if i < len(connections):
                start_idx, end_idx = connections[i]
                endpoint_radii: list[float] = []
                for vidx in (start_idx, end_idx):
                    if isinstance(vidx, (int, np.integer)) and 0 <= int(vidx) < len(vertex_radii):
                        endpoint_radii.append(float(vertex_radii[int(vidx)]))
                if endpoint_radii:
                    avg_radius = float(np.mean(endpoint_radii))
            if avg_radius <= 0:
                avg_radius = 1.0

            length_radius_ratio = edge_length / (avg_radius + 1e-10)
            if length_radius_ratio < min_length_radius_ratio:
                keep_mask[i] = False
                continue

            if i < len(connections) and len(vertex_positions) > 0:
                start_idx, end_idx = connections[i]
                if (
                    isinstance(start_idx, (int, np.integer))
                    and 0 <= int(start_idx) < len(vertex_positions)
                    and np.linalg.norm(trace_arr[0] - vertex_positions[int(start_idx)])
                    > max_endpoint_gap
                ):
                    keep_mask[i] = False
                    continue
                if (
                    isinstance(end_idx, (int, np.integer))
                    and 0 <= int(end_idx) < len(vertex_positions)
                    and np.linalg.norm(trace_arr[-1] - vertex_positions[int(end_idx)])
                    > max_endpoint_gap
                ):
                    keep_mask[i] = False
                    continue

        keep_indices = np.flatnonzero(keep_mask)
        curated: dict[str, Any] = {}
        for key, value in edges.items():
            if key == "traces":
                curated[key] = [traces[idx] for idx in keep_indices]
            elif key == "connections":
                curated[key] = [connections[idx] for idx in keep_indices]
            elif isinstance(value, np.ndarray) and value.shape[:1] == (n_edges,):
                curated[key] = value[keep_indices]
            elif isinstance(value, list) and len(value) == n_edges:
                curated[key] = [value[idx] for idx in keep_indices]
            else:
                curated[key] = value
        curated["original_indices"] = keep_indices
        logger.info(f"Drews curation: {n_edges} -> {len(keep_indices)} edges")
        return curated
