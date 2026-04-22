"""Shared helpers for maintained edge-candidate workflows."""

from __future__ import annotations

import numpy as np
from typing_extensions import TypeAlias

Int16Array: TypeAlias = "np.ndarray"
Int32Array: TypeAlias = "np.ndarray"
Int64Array: TypeAlias = "np.ndarray"
Float32Array: TypeAlias = "np.ndarray"
Float64Array: TypeAlias = "np.ndarray"
BoolArray: TypeAlias = "np.ndarray"


def _candidate_endpoint_pair_set(connections: np.ndarray) -> set[tuple[int, int]]:
    """Return orientation-independent endpoint pairs from a candidate payload."""
    pairs: set[tuple[int, int]] = set()
    normalized = np.asarray(connections, dtype=np.int32).reshape(-1, 2)
    for start_vertex, end_vertex in normalized:
        if int(start_vertex) < 0 or int(end_vertex) < 0:
            continue
        u, v = int(start_vertex), int(end_vertex)
        pairs.add((u, v) if u < v else (v, u))
    return pairs
