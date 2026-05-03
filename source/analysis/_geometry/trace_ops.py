from __future__ import annotations

from typing import cast

import numpy as np
from scipy.ndimage import gaussian_filter1d


def get_edges_for_vertex(connections: np.ndarray, vertex_index: int) -> np.ndarray:
    """Return indices of edges incident to a given vertex."""
    connections = np.asarray(connections)
    if connections.size == 0:
        return np.empty((0,), dtype=int)
    mask = (connections[:, 0] == vertex_index) | (connections[:, 1] == vertex_index)
    return cast("np.ndarray", np.flatnonzero(mask))


def get_edge_metric(
        trace: np.ndarray,
        energy: np.ndarray | None = None,
        method: str = "mean_energy",
) -> float:
    """Compute a simple metric for a single edge trace."""
    arr = np.asarray(trace)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return 0.0
    if method == "length" or energy is None:
        diffs = np.diff(arr, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))
    coords = np.floor(arr).astype(int)
    coords[:, 0] = np.clip(coords[:, 0], 0, energy.shape[0] - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, energy.shape[1] - 1)
    coords[:, 2] = np.clip(coords[:, 2], 0, energy.shape[2] - 1)
    samples = energy[coords[:, 0], coords[:, 1], coords[:, 2]]
    if method == "mean_energy":
        return float(np.mean(samples))
    if method == "min_energy":
        return float(np.min(samples))
    if method == "max_energy":
        return float(np.max(samples))
    if method == "median_energy":
        return float(np.median(samples))
    return float(np.mean(samples))


def resample_vectors(trace: np.ndarray, step: float) -> np.ndarray:
    """Resample a polyline trace at approximately uniform arc-length spacing."""
    pts = np.asarray(trace, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2 or step <= 0:
        return cast("np.ndarray", pts.copy())
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arclen = np.concatenate([[0.0], np.cumsum(seg)])
    total = arclen[-1]
    if total == 0:
        return cast("np.ndarray", pts[[0]].copy())
    num = max(2, int(np.floor(total / step)) + 1)
    targets = np.linspace(0.0, total, num)
    out = np.empty((len(targets), pts.shape[1]), dtype=float)
    for dim in range(pts.shape[1]):
        out[:, dim] = np.interp(targets, arclen, pts[:, dim])
    return cast("np.ndarray", out)


def smooth_edge_traces(traces: list[np.ndarray], sigma: float = 1.0) -> list[np.ndarray]:
    """Smooth each polyline trace with a 1D Gaussian along its path."""
    smoothed: list[np.ndarray] = []
    for trace in traces:
        arr = np.asarray(trace, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 3:
            smoothed.append(arr.copy())
            continue
        out = np.empty_like(arr)
        for dim in range(arr.shape[1]):
            out[:, dim] = gaussian_filter1d(arr[:, dim], sigma=sigma, mode="nearest")
        smoothed.append(out)
    return smoothed


def subsample_vectors(trace: np.ndarray, step: int) -> np.ndarray:
    """Subsample a polyline by keeping every ``step``-th point."""
    arr = np.asarray(trace)
    if arr.ndim != 2 or step <= 1:
        return cast("np.ndarray", arr.copy())
    idx = np.arange(0, arr.shape[0], step, dtype=int)
    if idx[-1] != arr.shape[0] - 1:
        idx = np.r_[idx, arr.shape[0] - 1]
    return cast("np.ndarray", arr[idx])
