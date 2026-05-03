"""
Graph-level metrics for SLAVV.
"""

from __future__ import annotations

from typing import cast

import numpy as np


def _matlab_edge_metrics(energy_traces: list[np.ndarray]) -> np.ndarray:
    """Mirror MATLAB ``get_edge_metric(..., 'max')`` over a list of traces."""
    if not energy_traces:
        return cast("np.ndarray", np.zeros((0,), dtype=np.float32))
    metrics = np.asarray(
        [
            float(np.max(np.asarray(trace, dtype=np.float32)))
            if np.asarray(trace).size
            else float("nan")
            for trace in energy_traces
        ],
        dtype=np.float32,
    )
    metrics[np.isnan(metrics)] = np.float32(-1000.0)
    return cast("np.ndarray", metrics)
