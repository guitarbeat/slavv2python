"""Gradient and trace metric helpers."""

from __future__ import annotations

import numpy as np

from ..edge_payloads import (
    _clip_trace_indices,
    _edge_metric_from_energy_trace,
    _record_trace_diagnostics,
    _trace_energy_series,
    _trace_scale_series,
)
from ..energy import compute_gradient_impl


def compute_gradient(
    energy: np.ndarray, pos: np.ndarray, microns_per_voxel: np.ndarray
) -> np.ndarray:
    """Compute gradient at ``pos`` using central differences."""
    pos_int = np.round(pos).astype(np.int64)
    energy_arr = np.ascontiguousarray(energy, dtype=np.float64)
    mpv_arr = np.asarray(microns_per_voxel, dtype=np.float64)
    res: np.ndarray = compute_gradient_impl(energy_arr, pos_int, mpv_arr)
    return res


__all__ = [
    "_clip_trace_indices",
    "_edge_metric_from_energy_trace",
    "_record_trace_diagnostics",
    "_trace_energy_series",
    "_trace_scale_series",
    "compute_gradient",
]
