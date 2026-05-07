from __future__ import annotations

import logging
from typing import Any

import numpy as np

try:
    from ..utils import calculate_path_length
except ImportError:  # pragma: no cover - fallback for direct execution
    from slavv_python.utils import calculate_path_length

logger = logging.getLogger(__name__)


def _resolve_curator_parameters(
    parameters: dict[str, Any] | None,
    min_length_radius_ratio: float,
    max_tortuosity: float,
    max_endpoint_gap: float,
) -> tuple[float, float, float]:
    params = parameters or {}
    return (
        float(params.get("min_length_radius_ratio", min_length_radius_ratio)),
        float(params.get("max_tortuosity", max_tortuosity)),
        float(params.get("max_endpoint_gap", max_endpoint_gap)),
    )


def _edge_endpoint_radii(
    connections: list[Any], edge_index: int, vertex_radii: np.ndarray
) -> list[float]:
    if edge_index >= len(connections):
        return []
    start_idx, end_idx = connections[edge_index]
    return [
        float(vertex_radii[int(vidx)])
        for vidx in (start_idx, end_idx)
        if isinstance(vidx, (int, np.integer)) and 0 <= int(vidx) < len(vertex_radii)
    ]


def _average_endpoint_radius(
    connections: list[Any], edge_index: int, vertex_radii: np.ndarray
) -> float:
    endpoint_radii = _edge_endpoint_radii(connections, edge_index, vertex_radii)
    return float(np.mean(endpoint_radii)) if endpoint_radii else 1.0


def _trace_passes_tortuosity(trace_arr: np.ndarray, max_tortuosity: float) -> tuple[bool, float]:
    edge_length = calculate_path_length(trace_arr)
    euclidean = float(np.linalg.norm(trace_arr[-1] - trace_arr[0]))
    tortuosity = edge_length / (euclidean + 1e-10)
    return tortuosity <= max_tortuosity, edge_length


def _endpoint_gap_ok(
    trace_point: np.ndarray,
    vertex_index: Any,
    vertex_positions: np.ndarray,
    max_endpoint_gap: float,
) -> bool:
    return not (
        isinstance(vertex_index, (int, np.integer))
        and 0 <= int(vertex_index) < len(vertex_positions)
        and np.linalg.norm(trace_point - vertex_positions[int(vertex_index)]) > max_endpoint_gap
    )


def _trace_passes_endpoint_gap(
    trace_arr: np.ndarray,
    connections: list[Any],
    edge_index: int,
    vertex_positions: np.ndarray,
    max_endpoint_gap: float,
) -> bool:
    if edge_index >= len(connections) or len(vertex_positions) == 0:
        return True
    start_idx, end_idx = connections[edge_index]
    return _endpoint_gap_ok(
        trace_arr[0], start_idx, vertex_positions, max_endpoint_gap
    ) and _endpoint_gap_ok(trace_arr[-1], end_idx, vertex_positions, max_endpoint_gap)


def _slice_edge_payload(
    edges: dict[str, Any],
    traces: list[Any],
    connections: list[Any],
    keep_indices: np.ndarray,
    n_edges: int,
) -> dict[str, Any]:
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
    return curated


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
        self.min_length_radius_ratio = min_length_radius_ratio
        self.max_tortuosity = max_tortuosity
        self.max_endpoint_gap = max_endpoint_gap

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

        min_length_radius_ratio, max_tortuosity, max_endpoint_gap = _resolve_curator_parameters(
            parameters,
            self.min_length_radius_ratio,
            self.max_tortuosity,
            self.max_endpoint_gap,
        )

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

            passes_tortuosity, edge_length = _trace_passes_tortuosity(trace_arr, max_tortuosity)
            if not passes_tortuosity:
                keep_mask[i] = False
                continue

            avg_radius = _average_endpoint_radius(connections, i, vertex_radii)
            length_radius_ratio = edge_length / (avg_radius + 1e-10)
            if length_radius_ratio < min_length_radius_ratio:
                keep_mask[i] = False
                continue

            if not _trace_passes_endpoint_gap(
                trace_arr, connections, i, vertex_positions, max_endpoint_gap
            ):
                keep_mask[i] = False

        keep_indices = np.flatnonzero(keep_mask)
        curated = _slice_edge_payload(edges, traces, connections, keep_indices, n_edges)
        logger.info(f"Drews curation: {n_edges} -> {len(keep_indices)} edges")
        return curated
