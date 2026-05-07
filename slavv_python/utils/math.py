"""
Mathematical helper functions for source.
"""

from __future__ import annotations

import numpy as np
import scipy.fft


def calculate_path_length(path: np.ndarray) -> float:
    """Calculate the total length of a polyline path."""
    if len(path) < 2:
        return 0.0
    distances = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return float(np.sum(distances))


def weighted_ks_test(
    sample1: np.ndarray,
    sample2: np.ndarray,
    weights1: np.ndarray | None = None,
    weights2: np.ndarray | None = None,
) -> float:
    """Compute the two-sample weighted Kolmogorov-Smirnov statistic."""
    sample1 = np.asarray(sample1)
    sample2 = np.asarray(sample2)
    w1 = (
        np.ones_like(sample1, dtype=float)
        if weights1 is None
        else np.asarray(weights1, dtype=float)
    )
    w2 = (
        np.ones_like(sample2, dtype=float)
        if weights2 is None
        else np.asarray(weights2, dtype=float)
    )

    if sample1.size == 0 or sample2.size == 0:
        return 0.0

    # Sort samples and weights together
    order1 = np.argsort(sample1)
    order2 = np.argsort(sample2)
    x = sample1[order1]
    y = sample2[order2]
    w1 = w1[order1]
    w2 = w2[order2]

    cdf1 = np.cumsum(w1) / np.sum(w1)
    cdf2 = np.cumsum(w2) / np.sum(w2)

    # Evaluate CDFs at all unique points from both samples
    all_values = np.concatenate([x, y])
    sorted_values = np.sort(all_values)
    idx1 = np.searchsorted(x, sorted_values, side="right")
    idx2 = np.searchsorted(y, sorted_values, side="right")
    cdf1_vals = np.concatenate([[0.0], cdf1])[idx1]
    cdf2_vals = np.concatenate([[0.0], cdf2])[idx2]

    return float(np.max(np.abs(cdf1_vals - cdf2_vals)))


def fourier_transform_even(image: np.ndarray) -> np.ndarray:
    """Pad an image to the next even dimensions and compute its N-dimensional FFT.

    Matches the logic of `fourier_transform_V2.m`.
    """
    arr = np.asarray(image)
    shape = np.array(arr.shape)
    next_even_shape = 2 * np.ceil((shape + 1) / 2).astype(int)

    pad_width = [(0, e - s) for s, e in zip(shape, next_even_shape)]
    padded_image = np.pad(arr, pad_width, mode="symmetric")

    return scipy.fft.fftn(padded_image)
