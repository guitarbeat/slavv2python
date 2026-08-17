"""Tiny synthetic helper vs original MATLAB ``get_energy_V202`` (isolation only).

Seeded noise volume — not crop, not a writer, not Energy unlock. Status stays
``blocked_float_path`` even if this fixture bit-matches.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from slavv_python.analytics.parity.proof.stretch import StretchStatus
from slavv_python.pipeline.energy.stretch_helper_body_isolation import (
    h5py_c_order_to_matlab_yxz,
)

SEED = 17
SHAPE_ZYX = (8, 8, 8)
ORIGINAL_HANDLE = "original_synthetic_8"
ENERGY_HANDLE = "energy_synthetic_8"
MATCHING_KERNEL = "3D gaussian conv annular pulse"

INTERPRET_SYNTHETIC_MATCH = (
    "Tiny synthetic helper vs original get_energy_V202 bit-matched. "
    "That does not unlock crop Energy; crop remains blocked_float_path."
)
INTERPRET_CLAMP_CLASS = (
    "Synthetic mismatches are helper nonnegative clamp vs original min-path; "
    "not the crop 1e-10 ULP class."
)
INTERPRET_TINY_ULP = (
    "Synthetic helper vs original still shows tiny-ULP residuals on simulated data."
)
INTERPRET_INCOMPLETE = "Synthetic original-vs-helper compare did not run (incomplete_infra)."


INTENSITY_UNIT = "unit"
INTENSITY_LOGNORMAL = "lognormal"


def seeded_volume_zyx(
    *,
    seed: int = SEED,
    shape_zyx: tuple[int, int, int] = SHAPE_ZYX,
    intensity: str = INTENSITY_UNIT,
) -> np.ndarray:
    """Seeded float64 volume in pipeline ZYX order."""
    rng = np.random.default_rng(int(seed))
    if intensity == INTENSITY_UNIT:
        return np.ascontiguousarray(rng.random(shape_zyx, dtype=np.float64))
    if intensity == INTENSITY_LOGNORMAL:
        return np.ascontiguousarray(
            np.exp(rng.normal(loc=0.0, scale=1.0, size=shape_zyx)).astype(np.float64)
        )
    raise ValueError(f"unknown intensity {intensity!r}")


def volume_zyx_to_matlab_yxz(image_zyx: np.ndarray) -> np.ndarray:
    """Pipeline ZYX -> MATLAB YXZ."""
    return np.transpose(np.asarray(image_zyx, dtype=np.float64), (1, 2, 0))


def matlab_yxz_to_h5py_c_order(volume_yxz: np.ndarray) -> np.ndarray:
    """Inverse of ``h5py_c_order_to_matlab_yxz`` for a 3D MATLAB ``[Y, X, Z]`` array."""
    arr = np.asarray(volume_yxz, dtype=np.float64)
    return np.transpose(arr, tuple(range(arr.ndim - 1, -1, -1)))


def energy_h5py_plane_to_zyx(plane: np.ndarray) -> np.ndarray:
    """MATLAB energy ``[Y, X, Z]`` via h5py reverse-all-axes -> pipeline ZYX."""
    yxz = h5py_c_order_to_matlab_yxz(np.asarray(plane, dtype=np.float64))
    return np.transpose(yxz, (2, 0, 1))


def classify_synthetic_compare(
    *,
    helper_energy: np.ndarray,
    original_energy: np.ndarray,
) -> dict[str, Any]:
    """Bit-compare helper vs original on the same ZYX grid. Not stretch success."""
    helper = np.asarray(helper_energy, dtype=np.float64)
    original = np.asarray(original_energy, dtype=np.float64)
    if helper.shape != original.shape:
        return {
            "result": "fail",
            "status_class": StretchStatus.INCOMPLETE_INFRA.value,
            "interpretation": INTERPRET_INCOMPLETE,
            "reason": f"shape mismatch helper={helper.shape} original={original.shape}",
            "isolation_only": True,
            "not_stretch_success": True,
            "stretch_complete": False,
        }
    n_total = int(helper.size)
    equal = helper == original
    n_bit = int(np.count_nonzero(equal))
    mismatch = ~equal
    n_mismatch = int(np.count_nonzero(mismatch))
    clamp_mask = mismatch & (original >= 0.0) & (helper == 0.0)
    n_clamp = int(np.count_nonzero(clamp_mask))
    residual = mismatch & ~clamp_mask
    finite = np.isfinite(helper) & np.isfinite(original) & residual
    abs_delta = np.abs(helper[finite] - original[finite]) if np.any(finite) else np.array([])
    max_abs = float(np.max(abs_delta)) if abs_delta.size else 0.0
    n_residual = int(np.count_nonzero(residual))
    if n_mismatch == 0:
        interpretation = INTERPRET_SYNTHETIC_MATCH
        result = "pass_fixture"
    elif n_residual == 0 and n_clamp == n_mismatch:
        interpretation = INTERPRET_CLAMP_CLASS
        result = "named_clamp"
    else:
        interpretation = INTERPRET_TINY_ULP
        result = "blocked_float_path"
    return {
        "result": result,
        "status_class": StretchStatus.BLOCKED_FLOAT_PATH.value,
        "interpretation": interpretation,
        "isolation_only": True,
        "not_stretch_success": True,
        "stretch_complete": False,
        "allclose_is_not_stretch_success": True,
        "total_voxels": n_total,
        "passed_voxels": n_bit,
        "failed_voxels": n_mismatch,
        "clamp_class_mismatches": n_clamp,
        "residual_mismatches": n_residual,
        "max_abs_delta": max_abs,
        "pass_rate": float(n_bit / n_total) if n_total else 0.0,
        "shape_zyx": [int(v) for v in helper.shape],
        "seed": SEED,
    }
