from __future__ import annotations

from typing import Any

import numpy as np

from ..processing.stages.edges.payloads import _edge_metric_from_energy_trace, _trace_energy_series


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
        Optional energy field for sampling. If None, returns 0.0.
    method:
        Metric name. Currently only "max_energy" is supported.
    """
    if energy is None:
        return 0.0
    if method != "max_energy":
        # Fallback for future methods
        pass

    e_series = _trace_energy_series(trace, energy)
    return _edge_metric_from_energy_trace(e_series)


def get_edges_for_vertex(connections: np.ndarray, vertex_idx: int) -> list[int]:
    """Return the indices of all edges connected to the specified vertex."""
    conn_arr = np.asarray(connections, dtype=int)
    if conn_arr.size == 0:
        return []
    mask = (conn_arr[:, 0] == vertex_idx) | (conn_arr[:, 1] == vertex_idx)
    return np.where(mask)[0].tolist()
