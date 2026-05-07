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


def scaled_positions(positions: np.ndarray, scale: list[float] | np.ndarray) -> np.ndarray:
    """Scale coordinate vectors by a per-axis scale vector."""
    positions_arr = np.asarray(positions, dtype=float)
    scale_arr = np.asarray(scale, dtype=float).reshape(-1)
    if positions_arr.ndim == 0:
        raise ValueError("positions must be at least 1D")
    if positions_arr.shape[-1] != scale_arr.shape[0]:
        raise ValueError("scale length must match positions coordinate dimension")
    return positions_arr * scale_arr
