#!/usr/bin/env python
"""Tiny synthetic stretch helper vs original MATLAB get_energy_V202.

Isolation only — scratch dest, not crop unlock, not U5/U6, not a writer on
protected roots. Uses seeded 8^3 noise so the 100% parity loop can move
without the crop volume.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from slavv_python.analytics.parity.constants import STRETCH_CROP_DEST_NAME
from slavv_python.analytics.parity.proof.stretch import STATUS_FILENAME, StretchStatus
from slavv_python.pipeline.energy.config import _prepare_energy_config
from slavv_python.pipeline.energy.matlab_engine_backend import (
    MatlabEngineInfraError,
    default_vectorization_root,
    refuse_protected_stretch_energy_dest,
    resolve_matlab_root,
)
from slavv_python.pipeline.energy.matlab_engine_host import (
    MatlabEnginePy37Worker,
    resolve_python37_executable,
)
from slavv_python.pipeline.energy.matlab_get_energy_v202_chunked import (
    compute_exact_parity_energy_chunked,
)
from slavv_python.pipeline.energy.stretch_chunk_isolation import patch_stretch_status_extra
from slavv_python.pipeline.energy.stretch_synthetic_original_compare import (
    ENERGY_HANDLE,
    INTENSITY_LOGNORMAL,
    INTENSITY_UNIT,
    MATCHING_KERNEL,
    ORIGINAL_HANDLE,
    SEED,
    classify_synthetic_compare,
    energy_h5py_plane_to_zyx,
    matlab_yxz_to_h5py_c_order,
    seeded_volume_zyx,
    volume_zyx_to_matlab_yxz,
)
from slavv_python.utils.validation import validate_parameters

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = REPO_ROOT / "workspace" / "scratch" / "stretch_synthetic_original_8"
DEFAULT_STATUS = (
    REPO_ROOT / "workspace" / "runs" / "oracle_180709_E" / STRETCH_CROP_DEST_NAME / STATUS_FILENAME
)
DEFAULT_SCRATCH = REPO_ROOT / "workspace" / "scratch" / "stretch_synthetic_original_compare.json"

# Crop 180709_E optics only — not the crop volume. Isolation leftover after
# isotropic 1 µm noise volumes bit-matched helper vs original.
CROP_LIKE_OPTICS: dict[str, Any] = {
    "microns_per_voxel": [0.916, 0.916, 1.9968800000000002],
    "numerical_aperture": 0.95,
    "excitation_wavelength_in_microns": 0.95,
    "sample_index_of_refraction": 1.33,
    "approximating_PSF": True,
    "gaussian_to_ideal_ratio": 0.5,
    "spherical_to_annular_ratio": 0.5,
    "scales_per_octave": 6,
    "energy_sign": -1.0,
    "energy_projection_mode": "matlab",
}


def _incomplete(reason: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "result": "skip",
        "status_class": StretchStatus.INCOMPLETE_INFRA.value,
        "interpretation": "Synthetic original-vs-helper compare did not run (incomplete_infra).",
        "reason": reason,
        "isolation_only": True,
        "not_stretch_success": True,
        "stretch_complete": False,
    }
    payload.update(extra)
    return payload


def _write_original_hdf5(path: Path, image_zyx: np.ndarray) -> None:
    yxz = volume_zyx_to_matlab_yxz(image_zyx)
    c_order = matlab_yxz_to_h5py_c_order(yxz)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("d", data=np.ascontiguousarray(c_order))


def _load_original_energy_zyx(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        planes = np.asarray(handle["d"], dtype=np.float64)
    if planes.ndim != 4 or planes.shape[0] < 2:
        raise ValueError(f"unexpected MATLAB energy HDF5 shape {planes.shape} for {path}")
    return energy_h5py_plane_to_zyx(planes[1])


def run_synthetic_compare(
    *,
    dest: Path,
    radius_smallest: float = 1.5,
    radius_largest: float = 1.6,
    shape_zyx: tuple[int, int, int] = (8, 8, 8),
    max_voxels_per_node: float = 1e9,
    extra_params: dict[str, Any] | None = None,
    intensity: str = INTENSITY_UNIT,
) -> dict[str, Any]:
    refuse_protected_stretch_energy_dest(dest)
    python37 = resolve_python37_executable()
    if python37 is None:
        return _incomplete("isolated Python 3.7 stretch env missing")
    try:
        resolve_matlab_root()
    except MatlabEngineInfraError as exc:
        return _incomplete(str(exc))

    image = seeded_volume_zyx(shape_zyx=shape_zyx, intensity=intensity)
    dest.mkdir(parents=True, exist_ok=True)
    original_path = dest / ORIGINAL_HANDLE
    energy_path = dest / ENERGY_HANDLE
    _write_original_hdf5(original_path, image)
    if energy_path.exists():
        energy_path.unlink()

    shared = {
        "energy_method": "hessian",
        "energy_float_backend": "matlab_engine",
        "comparison_exact_network": True,
        "n_jobs": 1,
        "radius_of_smallest_vessel_in_microns": float(radius_smallest),
        "radius_of_largest_vessel_in_microns": float(radius_largest),
        "max_voxels_per_node_energy": float(max_voxels_per_node),
    }
    if extra_params:
        shared.update(extra_params)
    params = validate_parameters(shared)
    vectorization_root = default_vectorization_root()
    worker = MatlabEnginePy37Worker(python37, vectorization_root=vectorization_root)
    try:
        worker.start()
        params["_stretch_engine_float_body_bound"] = True
        params["_stretch_engine_session"] = worker
        config = _prepare_energy_config(image, params)
        helper_energy, helper_scales, _extra = compute_exact_parity_energy_chunked(image, config)
        del helper_scales, _extra
        radii = np.asarray(config["lumen_radius_microns"], dtype=np.float64)
        microns = np.asarray(config["microns_per_voxel"], dtype=np.float64)
        psf = np.asarray(config["pixels_per_sigma_PSF"], dtype=np.float64)
        elapsed = worker.get_energy_v202(
            matching_kernel_string=MATCHING_KERNEL,
            lumen_radius_in_microns_range=radii,
            vessel_wall=float(config.get("vessel_wall_thickness_in_microns", 0.0)),
            microns_per_voxel=np.array([microns[1], microns[2], microns[0]], dtype=np.float64),
            pixels_per_sigma_psf=np.array([psf[1], psf[2], psf[0]], dtype=np.float64),
            max_voxels_per_node=float(config["max_voxels"]),
            data_directory=dest,
            original_handle=ORIGINAL_HANDLE,
            energy_handle=ENERGY_HANDLE,
            gaussian_to_ideal_ratio=float(config.get("gaussian_to_ideal_ratio", 1.0)),
            spherical_to_annular_ratio=float(config.get("spherical_to_annular_ratio", 1.0)),
        )
    except MatlabEngineInfraError as exc:
        return _incomplete(str(exc))
    finally:
        worker.quit()

    if not energy_path.is_file():
        return _incomplete("original get_energy_V202 did not write energy HDF5")
    original_energy = _load_original_energy_zyx(energy_path)
    payload = classify_synthetic_compare(
        helper_energy=helper_energy,
        original_energy=original_energy,
    )
    payload["original_elapsed_sec"] = float(elapsed)
    payload["dest"] = str(dest)
    payload["seed"] = SEED
    payload["radius_smallest"] = float(radius_smallest)
    payload["radius_largest"] = float(radius_largest)
    payload["n_scales"] = int(np.asarray(radii).size)
    payload["max_voxels_per_node"] = float(max_voxels_per_node)
    payload["crop_like_optics"] = bool(extra_params)
    payload["intensity"] = str(intensity)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--scratch-json", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--status-json", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--radius-smallest", type=float, default=1.5)
    parser.add_argument("--radius-largest", type=float, default=1.6)
    parser.add_argument(
        "--shape",
        type=int,
        nargs=3,
        metavar=("Z", "Y", "X"),
        default=(8, 8, 8),
        help="Seeded volume shape in pipeline ZYX order.",
    )
    parser.add_argument("--max-voxels", type=float, default=1e9)
    parser.add_argument(
        "--crop-like-optics",
        action="store_true",
        help="Use crop 180709_E microns/PSF/ratio/scales_per_octave on seeded noise (not the crop volume).",
    )
    parser.add_argument(
        "--intensity",
        choices=(INTENSITY_UNIT, INTENSITY_LOGNORMAL),
        default=INTENSITY_UNIT,
    )
    parser.add_argument(
        "--status-extra-key",
        default="synthetic_original_compare",
        help="stretch_status.json extra key (keep 2-radius result distinct from multi-scale).",
    )
    args = parser.parse_args()
    extra = dict(CROP_LIKE_OPTICS) if args.crop_like_optics else None
    payload = run_synthetic_compare(
        dest=args.dest,
        radius_smallest=args.radius_smallest,
        radius_largest=args.radius_largest,
        shape_zyx=(int(args.shape[0]), int(args.shape[1]), int(args.shape[2])),
        max_voxels_per_node=float(args.max_voxels),
        extra_params=extra,
        intensity=str(args.intensity),
    )
    args.scratch_json.parent.mkdir(parents=True, exist_ok=True)
    args.scratch_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if (
        args.status_json.is_file()
        and payload.get("status_class") == StretchStatus.BLOCKED_FLOAT_PATH.value
    ):
        patch_stretch_status_extra(
            args.status_json,
            {str(args.status_extra_key): payload},
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
