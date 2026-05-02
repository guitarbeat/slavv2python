"""Direction estimation helpers for edge tracing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.ndimage as ndi
from skimage import feature

if TYPE_CHECKING:
    from collections.abc import Callable


def generate_edge_directions(n_directions: int, seed: int | None = None) -> np.ndarray:
    """Generate uniformly distributed unit vectors on the sphere."""
    if n_directions <= 0:
        return np.empty((0, 3), dtype=np.float64)
    if n_directions == 1:
        return np.array([[0, 0, 1]], dtype=np.float64)

    rng = np.random.default_rng(seed)
    points = rng.standard_normal((n_directions, 3))
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (points / norms).astype(np.float64)


def estimate_vessel_directions(
    energy: np.ndarray,
    pos: np.ndarray,
    radius: float,
    microns_per_voxel: np.ndarray,
    fallback_direction_generator: Callable[[int, int | None], np.ndarray],
) -> np.ndarray:
    """Estimate vessel directions at a vertex via local Hessian analysis."""
    sigma = max(radius / 2.0, 1.0)
    center = np.round(pos).astype(int)
    r = int(max(1, np.ceil(sigma)))
    slices = tuple(slice(max(c - r, 0), min(c + r + 1, s)) for c, s in zip(center, energy.shape))
    patch = energy[slices]
    if patch.ndim != 3 or min(patch.shape) < 3:
        return fallback_direction_generator(2, seed=0)

    scale = microns_per_voxel / microns_per_voxel.min()
    if not np.allclose(scale, 1):
        patch = ndi.zoom(patch, scale, order=1, mode="nearest")

    try:
        raw_hessian = feature.hessian_matrix(
            patch,
            sigma=sigma,
            mode="nearest",
            order="rc",
            use_gaussian_derivatives=False,
        )
    except TypeError:
        raw_hessian = feature.hessian_matrix(
            patch,
            sigma=sigma,
            mode="nearest",
            order="rc",
        )
    hessian_elems = [h * (radius**2) for h in raw_hessian]
    patch_center_arr = np.array(patch.shape, dtype=np.int64) // 2
    patch_center = tuple(int(value) for value in patch_center_arr.tolist())
    Hxx, Hxy, Hxz, Hyy, Hyz, Hzz = [h[patch_center] for h in hessian_elems]
    H = np.array(
        [
            [Hxx, Hxy, Hxz],
            [Hxy, Hyy, Hyz],
            [Hxz, Hyz, Hzz],
        ]
    )
    try:
        w, v = np.linalg.eigh(H)
    except np.linalg.LinAlgError:
        return fallback_direction_generator(2, seed=0)
    if not np.all(np.isfinite(w)):
        return fallback_direction_generator(2, seed=0)

    w_abs = np.sort(np.abs(w))
    max_eig = w_abs[-1]
    if max_eig == 0 or (w_abs[1] - w_abs[0]) < 1e-6 * max_eig:
        return fallback_direction_generator(2, seed=0)

    direction = v[:, np.argmin(np.abs(w))]
    norm = np.linalg.norm(direction)
    if norm == 0 or not np.isfinite(norm):
        return fallback_direction_generator(2, seed=0)
    direction = direction / norm
    return np.stack((direction, -direction))


__all__ = [
    "estimate_vessel_directions",
    "generate_edge_directions",
]
