"""Gradient and trace metric helpers."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np

from .._edge_payloads import _empty_stop_reason_counts
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


def _clip_trace_indices(trace: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Convert a trace to clipped integer voxel indices."""
    clipped_coords = np.floor(np.asarray(trace, dtype=np.float32)[:, :3]).astype(
        np.int32,
        copy=False,
    )
    for axis in range(3):
        clipped_coords[:, axis] = np.clip(clipped_coords[:, axis], 0, shape[axis] - 1)
    return cast("np.ndarray", clipped_coords)


def _trace_scale_series(edge_trace: np.ndarray, scale_indices: np.ndarray | None) -> np.ndarray:
    """Sample projected scale indices along an edge trace."""
    if scale_indices is None:
        return np.zeros((len(edge_trace),), dtype=np.int16)
    idx = _clip_trace_indices(edge_trace, scale_indices.shape)
    scale_trace = scale_indices[idx[:, 0], idx[:, 1], idx[:, 2]].astype(
        np.int16,
        copy=False,
    )
    return cast("np.ndarray", scale_trace)


def _trace_energy_series(edge_trace: np.ndarray, energy: np.ndarray) -> np.ndarray:
    """Sample projected energy values along an edge trace."""
    idx = _clip_trace_indices(edge_trace, energy.shape)
    energy_trace = energy[idx[:, 0], idx[:, 1], idx[:, 2]].astype(
        np.float32,
        copy=False,
    )
    return cast("np.ndarray", energy_trace)


def _edge_metric_from_energy_trace(energy_trace: np.ndarray) -> float:
    """Match MATLAB's current edge quality metric: minimum max-energy is best."""
    arr = np.asarray(energy_trace, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    value = float(np.nanmax(arr))
    return -1000.0 if math.isnan(value) else value


def _record_trace_diagnostics(
    diagnostics: dict[str, Any],
    trace_metadata: dict[str, Any],
) -> None:
    """Accumulate per-trace terminal-resolution and stop-reason diagnostics."""
    if stop_reason := trace_metadata.get("stop_reason"):
        stop_reason_counts = diagnostics.setdefault(
            "stop_reason_counts", _empty_stop_reason_counts()
        )
        stop_reason_counts[stop_reason] = int(stop_reason_counts.get(stop_reason, 0)) + 1

    terminal_resolution = trace_metadata.get("terminal_resolution")
    if terminal_resolution == "direct_hit":
        diagnostics["terminal_direct_hit_count"] += 1
    elif terminal_resolution == "reverse_center_hit":
        diagnostics["terminal_reverse_center_hit_count"] += 1
    elif terminal_resolution == "reverse_volume_hit":
        diagnostics["terminal_reverse_volume_hit_count"] += 1
    elif terminal_resolution == "reverse_near_hit":
        diagnostics["terminal_reverse_near_hit_count"] += 1


__all__ = [
    "_clip_trace_indices",
    "_edge_metric_from_energy_trace",
    "_record_trace_diagnostics",
    "_trace_energy_series",
    "_trace_scale_series",
    "compute_gradient",
]
