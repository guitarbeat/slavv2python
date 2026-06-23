"""Mathematical kernels for Energy stage computation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import DTypeLike


def compute_principal_energy(
    gradients: np.ndarray,
    curvatures: np.ndarray,
    *,
    energy_sign: float,
    batch_size: int = 256 * 1024,
    dtype: DTypeLike = np.float64,
) -> np.ndarray:
    """Calculate principal energy from Hessian and gradient components."""
    num_voxels = gradients.shape[0]
    if num_voxels == 0:
        return np.empty(0, dtype=dtype)

    principal_energies = np.empty((num_voxels, 3), dtype=dtype)
    g_full = gradients.astype(dtype, copy=False)
    c_full = curvatures.astype(dtype, copy=False)

    for start_idx in range(0, num_voxels, batch_size):
        end_idx = min(start_idx + batch_size, num_voxels)
        g_batch = g_full[start_idx:end_idx]
        c_batch = c_full[start_idx:end_idx]

        # Reconstruct (3, 3) Hessian: [dYY, dXX, dZZ, dYX, dXZ, dZY]
        h_batch = np.empty((g_batch.shape[0], 3, 3), dtype=dtype)
        h_batch[:, 0, 0] = c_batch[:, 0]
        h_batch[:, 1, 1] = c_batch[:, 1]
        h_batch[:, 2, 2] = c_batch[:, 2]
        h_batch[:, 0, 1] = h_batch[:, 1, 0] = c_batch[:, 3]
        h_batch[:, 1, 2] = h_batch[:, 2, 1] = c_batch[:, 4]
        h_batch[:, 0, 2] = h_batch[:, 2, 0] = c_batch[:, 5]

        w_batch, v_batch = np.linalg.eigh(h_batch)

        # MATLAB energy_filter_V200 uses eig's returned component order and
        # clips its third component; preserve the matching eigh order here.
        p_batch = np.einsum("ni,nij->nj", g_batch, v_batch)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            e_batch = w_batch * np.exp(-((p_batch / w_batch) ** 2) / 2.0)

        if energy_sign < 0:
            # MATLAB clips the third returned eig component when it is positive.
            e_batch[:, 2] = np.minimum(e_batch[:, 2], 0.0)
        else:
            e_batch[:, 2] = np.maximum(e_batch[:, 2], 0.0)

        principal_energies[start_idx:end_idx] = e_batch

    energy_sum = np.sum(principal_energies, axis=1)

    if energy_sign < 0:
        energy_sum[~np.isfinite(energy_sum)] = np.inf
        energy_sum[energy_sum >= 0] = np.inf
    else:
        energy_sum[~np.isfinite(energy_sum)] = -np.inf
        energy_sum[energy_sum <= 0] = -np.inf

    return energy_sum


__all__ = ["compute_principal_energy"]
