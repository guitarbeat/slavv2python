from __future__ import annotations

from typing import cast

import numpy as np

from ..pipeline.edges.payloads import _edge_metric_from_energy_trace, _trace_energy_series


def get_edge_metric(
    trace: np.ndarray,
    energy: np.ndarray | None = None,
    *,
    method: str = "max_energy",
) -> float:
    """Calculate a quality metric for an edge trace.

    Parameters
    ----------
    trace:
        (N, 3) or (N, 4) array of coordinates.
    energy:
        Optional energy field for sampling. Required for energy-based methods.
    method:
        Metric name: "max_energy" (default), "mean_energy", "min_energy", or "length".
    """
    if method == "length":
        from .vector_geometry import calculate_path_length

        return calculate_path_length(trace)

    if energy is None:
        return 0.0

    e_series = _trace_energy_series(trace, energy)
    if not e_series.size:
        return 0.0

    if method == "mean_energy":
        return float(np.mean(e_series))
    if method == "median_energy":
        return float(np.median(e_series))
    if method == "min_energy":
        return float(np.min(e_series))

    # Default to max_energy for everything else
    return _edge_metric_from_energy_trace(e_series)


def get_edges_for_vertex(connections: np.ndarray, vertex_idx: int) -> list[int]:
    """Return the indices of all edges connected to the specified vertex."""
    conn_arr = np.asarray(connections, dtype=int)
    if conn_arr.size == 0:
        return []
    mask = (conn_arr[:, 0] == vertex_idx) | (conn_arr[:, 1] == vertex_idx)
    return cast("list[int]", np.where(mask)[0].tolist())
