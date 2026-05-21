from __future__ import annotations

import numpy as np


def safe_normalize_rows(vectors: np.ndarray, *, eps: float = 0.0) -> np.ndarray:
    """Normalize each row of a 2D array while safely handling zero-norm rows."""
    vectors_arr = np.asarray(vectors, dtype=float)
    if vectors_arr.ndim != 2:
        raise ValueError("vectors must be a 2D array")
    if eps < 0:
        raise ValueError("eps must be non-negative")

    norms = np.linalg.norm(vectors_arr, axis=1, keepdims=True)
    if eps == 0.0:
        norms[norms == 0.0] = 1.0
    else:
        norms = np.maximum(norms, eps)
    return vectors_arr / norms


def angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    """Return the angle between two vectors in degrees."""
    a_arr = np.asarray(a, dtype=float).reshape(-1)
    b_arr = np.asarray(b, dtype=float).reshape(-1)
    if a_arr.shape != b_arr.shape:
        raise ValueError("a and b must have the same shape")

    norm_a = float(np.linalg.norm(a_arr))
    norm_b = float(np.linalg.norm(b_arr))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    cosang = float(np.clip(np.dot(a_arr, b_arr) / (norm_a * norm_b), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def calculate_path_length(path: np.ndarray) -> float:
    """Calculate the total length of a polyline path."""
    if len(path) < 2:
        return 0.0
    distances = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return float(np.sum(distances))


def scaled_positions(positions: np.ndarray, scale: list[float] | np.ndarray) -> np.ndarray:
    """Scale coordinate vectors by a per-axis scale vector."""
    positions_arr = np.asarray(positions, dtype=float)
    scale_arr = np.asarray(scale, dtype=float).reshape(-1)
    if positions_arr.ndim == 0:
        raise ValueError("positions must be at least 1D")
    if positions_arr.shape[-1] != scale_arr.shape[0]:
        raise ValueError("scale length must match positions coordinate dimension")
    return positions_arr * scale_arr


def resample_vectors(trace: np.ndarray, step: float = 1.0) -> np.ndarray:
    """Linearly resample a polyline path at regular intervals."""
    trace_arr = np.asarray(trace, dtype=float)
    if len(trace_arr) < 2:
        return trace_arr

    # Cumulative distance along the path
    diffs = np.diff(trace_arr, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dists = np.concatenate(([0.0], np.cumsum(dists)))

    total_len = float(cum_dists[-1])
    if total_len == 0:
        return trace_arr[0:1].copy()

    # Interpolation target distances
    new_dists = np.arange(0, total_len, step)
    if total_len > new_dists[-1]:
        new_dists = np.concatenate((new_dists, [total_len]))

    # Interpolate each coordinate axis
    resampled = np.empty((len(new_dists), trace_arr.shape[1]), dtype=float)
    for i in range(trace_arr.shape[1]):
        resampled[:, i] = np.interp(new_dists, cum_dists, trace_arr[:, i])

    return resampled


def smooth_edge_traces(traces: list[np.ndarray], sigma: float = 1.0) -> list[np.ndarray]:
    """Apply Gaussian smoothing to a set of polyline paths."""
    from scipy.ndimage import gaussian_filter1d

    smoothed = []
    for trace in traces:
        trace_arr = np.asarray(trace, dtype=float)
        if len(trace_arr) < 2:
            smoothed.append(trace_arr.copy())
            continue

        s_trace = np.empty_like(trace_arr)
        for i in range(trace_arr.shape[1]):
            s_trace[:, i] = gaussian_filter1d(trace_arr[:, i], sigma=sigma, mode="nearest")
        smoothed.append(s_trace)

    return smoothed
