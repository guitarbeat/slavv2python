#!/usr/bin/env python
"""Compare v2 Energy lattices/params vs original MATLAB get_energy_V202.

Isolation only — not Energy unlock, not U5/U6, not a writer. Writes scratch
JSON and patches dest stretch_status.json extra only. Status stays
blocked_float_path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.io import loadmat

from slavv_python.analytics.parity.constants import (
    EXPERIMENT_PARAMS_DIR,
    VALIDATED_PARAMS_PATH,
)
from slavv_python.analytics.parity.proof.stretch import STATUS_FILENAME, StretchStatus
from slavv_python.pipeline.energy.config import _prepare_energy_config
from slavv_python.pipeline.energy.matlab_engine_backend import (
    MatlabEngineInfraError,
    refuse_protected_stretch_energy_dest,
    resolve_matlab_root,
)
from slavv_python.pipeline.energy.matlab_engine_host import resolve_python37_executable
from slavv_python.pipeline.energy.matlab_get_energy_v202_chunked import (
    get_chunking_lattice_v190,
)
from slavv_python.pipeline.energy.stretch_chunk_isolation import patch_stretch_status_extra
from slavv_python.pipeline.energy.stretch_lattice_params_isolation import (
    EXPECTED_MATLAB_OCTAVE2_CHUNKS,
    INTERPRET_INCOMPLETE_INFRA,
    compare_param_fields,
    isolation_payload,
    matlab_derived_scales_per_octave,
    matlab_formula_lattices,
    matlab_h5_size_from_h5py_shape,
    python_lattices_from_config,
    record_at_octave,
    zyx_to_yxz,
)
from slavv_python.utils.validation import validate_parameters

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = REPO_ROOT / "workspace" / "runs" / "oracle_180709_E" / "crop_M_stretch_engine_v2"
DEFAULT_ORACLE = REPO_ROOT / "workspace" / "oracles" / "180709_E_crop_M_v2"
DEFAULT_SCRATCH = REPO_ROOT / "workspace" / "scratch" / "stretch_lattice_params_isolation.json"
ORIGINAL_HANDLE = "original_180709_E_crop_M"
SETTINGS_NAME = "energy_260624-105705.mat"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_dest_params(dest: Path) -> dict[str, Any]:
    candidates = (
        dest / VALIDATED_PARAMS_PATH,
        dest / EXPERIMENT_PARAMS_DIR / "validated_params.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return _load_json(candidate)
    raise FileNotFoundError(f"validated_params.json missing under {dest}")


def _load_energy_settings(settings_path: Path) -> dict[str, Any]:
    payload = loadmat(settings_path, squeeze_me=True, struct_as_record=False)
    radii = np.asarray(payload["lumen_radius_in_microns_range"], dtype=np.float64).reshape(-1)
    return {
        "radii": radii,
        "microns_per_voxel": np.asarray(payload["microns_per_voxel"], dtype=np.float64).reshape(-1),
        "pixels_per_sigma_psf": np.asarray(
            payload["pixels_per_sigma_PSF"], dtype=np.float64
        ).reshape(-1),
        "max_voxels": float(payload["max_voxels_per_node_energy"]),
        "gaussian_to_ideal_ratio": float(payload["gaussian_to_ideal_ratio"]),
        "spherical_to_annular_ratio": float(payload["spherical_to_annular_ratio"]),
        "scales_per_octave_derived": matlab_derived_scales_per_octave(radii),
    }


def _h5_matlab_size(path: Path) -> tuple[tuple[int, ...], tuple[int, ...]]:
    with h5py.File(path, "r") as handle:
        dataset = handle["d"] if "d" in handle else next(iter(handle.values()))
        h5py_shape = tuple(int(v) for v in dataset.shape)
    return h5py_shape, matlab_h5_size_from_h5py_shape(h5py_shape)


def _incomplete(reason: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "result": "skip",
        "status_class": StretchStatus.INCOMPLETE_INFRA.value,
        "interpretation": INTERPRET_INCOMPLETE_INFRA,
        "reason": reason,
        "isolation_only": True,
        "not_stretch_success": True,
        "stretch_complete": False,
    }
    payload.update(extra)
    return payload


def _confirm_lattice_with_engine(
    *,
    strel: np.ndarray,
    max_voxels: float,
    approx_size: np.ndarray,
) -> dict[str, Any]:
    """Optional MATLAB ``get_chunking_lattice_V190`` when formula octave 2 ≠ 726."""
    python37 = resolve_python37_executable()
    if python37 is None:
        return {"result": "skip", "status_class": StretchStatus.INCOMPLETE_INFRA.value}
    try:
        resolve_matlab_root()
    except MatlabEngineInfraError as exc:
        return {
            "result": "skip",
            "status_class": StretchStatus.INCOMPLETE_INFRA.value,
            "reason": str(exc),
        }
    python_dims, python_n = get_chunking_lattice_v190(strel, max_voxels, approx_size)
    return {
        "result": "python_port_only",
        "status_class": StretchStatus.INCOMPLETE_INFRA.value,
        "reason": "worker has no get_chunking_lattice_V190 op; Python port used",
        "python_port_dimensions": [int(v) for v in np.asarray(python_dims).tolist()],
        "python_port_chunks": int(python_n),
    }


def run_lattice_params_isolation(
    *,
    dest: Path,
    oracle_root: Path,
    scratch_out: Path,
) -> dict[str, Any]:
    refuse_protected_stretch_energy_dest(scratch_out)
    batch = oracle_root / "01_Input" / "matlab_results" / "batch_260624-105705"
    settings_path = batch / "settings" / SETTINGS_NAME
    original_path = batch / "data" / ORIGINAL_HANDLE
    energy_npy = dest / "02_Energy" / "best_energy.npy"
    try:
        params = validate_parameters(_load_dest_params(dest))
        oracle = _load_energy_settings(settings_path)
        h5py_shape, matlab_size = _h5_matlab_size(original_path)
        if not energy_npy.is_file():
            raise FileNotFoundError(f"v2 Energy npy missing: {energy_npy}")
        image_shape = tuple(int(v) for v in np.load(energy_npy, mmap_mode="r").shape)
        config = _prepare_energy_config(np.zeros(image_shape, dtype=np.float64), params)
    except (FileNotFoundError, ValueError, OSError, KeyError) as exc:
        return _incomplete(str(exc))

    dest_fields = {
        "lumen_radius_microns": np.asarray(config["lumen_radius_microns"], dtype=np.float64),
        "microns_per_voxel_raw": np.asarray(params.get("microns_per_voxel"), dtype=np.float64),
        "microns_per_voxel_working": np.asarray(config["microns_per_voxel"], dtype=np.float64),
        "pixels_per_sigma_PSF": np.asarray(config["pixels_per_sigma_PSF"], dtype=np.float64),
        "pixels_per_sigma_PSF_yxz": zyx_to_yxz(
            np.asarray(config["pixels_per_sigma_PSF"], dtype=np.float64)
        ),
        "max_voxels": float(config["max_voxels"]),
        "gaussian_to_ideal_ratio": float(config["gaussian_to_ideal_ratio"]),
        "spherical_to_annular_ratio": float(config["spherical_to_annular_ratio"]),
        "scales_per_octave": float(config["scales_per_octave"]),
    }
    param_fields = compare_param_fields(dest_fields, oracle)
    python_lattices = python_lattices_from_config(config)
    size_yxz = np.asarray(matlab_size[:3], dtype=float)
    matlab_lattices = matlab_formula_lattices(
        size_of_image_yxz=size_yxz,
        radii=np.asarray(oracle["radii"], dtype=np.float64),
        microns_yxz=np.asarray(oracle["microns_per_voxel"], dtype=float),
        max_voxels=float(oracle["max_voxels"]),
    )
    extra: dict[str, Any] = {
        "dest": str(dest),
        "oracle_root": str(oracle_root),
        "original_h5py_shape": list(h5py_shape),
        "original_matlab_size": list(matlab_size),
        "dest_image_shape_zyx": list(image_shape),
        "python_total_chunks": int(sum(record.number_of_chunks for record in python_lattices)),
        "matlab_total_chunks": int(sum(record.number_of_chunks for record in matlab_lattices)),
    }
    matlab_oct2 = record_at_octave(matlab_lattices, 2)
    if matlab_oct2 is None or matlab_oct2.number_of_chunks != EXPECTED_MATLAB_OCTAVE2_CHUNKS:
        strel = 1.0 / (
            np.asarray(oracle["microns_per_voxel"], dtype=float)
            * np.asarray(matlab_oct2.rf if matlab_oct2 is not None else (1, 1, 1), dtype=float)
        )
        approx = np.asarray(
            matlab_oct2.approx_size if matlab_oct2 is not None else size_yxz,
            dtype=float,
        )
        extra["engine_confirm"] = _confirm_lattice_with_engine(
            strel=strel,
            max_voxels=float(oracle["max_voxels"]),
            approx_size=approx,
        )
    return isolation_payload(
        param_fields=param_fields,
        python_lattices=python_lattices,
        matlab_lattices=matlab_lattices,
        extra=extra,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--scratch-out", type=Path, default=DEFAULT_SCRATCH)
    args = parser.parse_args(argv)
    payload = run_lattice_params_isolation(
        dest=args.dest,
        oracle_root=args.oracle_root,
        scratch_out=args.scratch_out,
    )
    args.scratch_out.parent.mkdir(parents=True, exist_ok=True)
    args.scratch_out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    status_path = args.dest / STATUS_FILENAME
    if (
        payload.get("status_class") == StretchStatus.BLOCKED_FLOAT_PATH.value
        and status_path.is_file()
    ):
        patch_stretch_status_extra(
            status_path,
            {
                "lattice_params_isolation": {
                    "interpretation": payload.get("interpretation"),
                    "isolation_only": True,
                    "not_stretch_success": True,
                    "params_equal": payload.get("params_equal"),
                    "params_core_equal": payload.get("params_core_equal"),
                    "lattices_match_by_rf": payload.get("lattices_match_by_rf"),
                    "octave2": payload.get("octave2"),
                    "scratch_json": str(args.scratch_out),
                    "status_class": StretchStatus.BLOCKED_FLOAT_PATH.value,
                    "stretch_complete": False,
                }
            },
        )
    print(
        json.dumps(
            {
                "result": payload.get("result"),
                "interpretation": payload.get("interpretation"),
                "octave2": payload.get("octave2"),
                "scratch": str(args.scratch_out),
            },
            indent=2,
        )
    )
    if payload.get("result") == "ok":
        return 0
    if payload.get("result") == "skip":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
