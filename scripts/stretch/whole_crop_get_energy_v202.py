#!/usr/bin/env python
"""E14: scratch-only Python-orchestrated MATLAB get_energy_V202 vs crop oracle.

Not a cheap probe: MATLAB get_energy_V202 is octave-chunked (726 chunks on
octave 2 of 6 on crop_M). Abort / time-box rather than wait overnight.
Isolation only — not a crop Energy unlock, not U5/U6, not stretch_complete.
Never overwrites canonical_full_v18, canonical_full_v16, crop_M_exact_v3, or crop_M_stretch_engine_v2.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat

from slavv_python.analytics.parity.constants import (
    CROP_ORIGINAL_HANDLE,
    STRETCH_CROP_ORACLE_ID,
)
from slavv_python.analytics.parity.proof.energy_ulp_proof import (
    EnergyFloatGateOptions,
    evaluate_energy_float_gate,
)
from slavv_python.pipeline.energy.matlab_engine_backend import (
    MatlabEngineInfraError,
    default_vectorization_root,
    refuse_matlab_only_energy_checkpoint_as_stretch_success,
    refuse_protected_stretch_energy_dest,
    resolve_matlab_root,
)
from slavv_python.pipeline.energy.matlab_engine_host import (
    MatlabEnginePy37Worker,
    resolve_python37_executable,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = REPO_ROOT / "workspace" / "scratch" / "e14_whole_crop_get_energy_v202"
ORACLE_BATCH = (
    REPO_ROOT
    / "workspace"
    / "oracles"
    / STRETCH_CROP_ORACLE_ID
    / "01_Input"
    / "matlab_results"
    / "batch_260624-105705"
)
ORIGINAL_HANDLE = CROP_ORIGINAL_HANDLE
ENERGY_HANDLE = "e14_energy_180709_E_crop_M"
MATCHING_KERNEL = "3D gaussian conv annular pulse"
CROP_VOXELS = 64 * 256 * 256
TIMEBOX_SEC = 3600


def _memory_check_crop() -> dict[str, float | int | bool]:
    energy_bytes = CROP_VOXELS * 8 * 2
    original_bytes = CROP_VOXELS * 2
    ok = energy_bytes + original_bytes < 512 * 1024 * 1024
    return {
        "crop_voxels": CROP_VOXELS,
        "approx_energy_bytes": energy_bytes,
        "ok": ok,
    }


def _load_energy_settings(settings_path: Path) -> dict[str, object]:
    payload = loadmat(settings_path, squeeze_me=True, struct_as_record=False)
    return {
        "lumen_radius_in_microns_range": np.asarray(
            payload["lumen_radius_in_microns_range"], dtype=np.float64
        ).reshape(-1),
        "microns_per_voxel": np.asarray(payload["microns_per_voxel"], dtype=np.float64).reshape(-1),
        "pixels_per_sigma_psf": np.asarray(
            payload["pixels_per_sigma_PSF"], dtype=np.float64
        ).reshape(-1),
        "max_voxels_per_node": float(payload["max_voxels_per_node_energy"]),
        "gaussian_to_ideal_ratio": float(payload["gaussian_to_ideal_ratio"]),
        "spherical_to_annular_ratio": float(payload["spherical_to_annular_ratio"]),
        "vessel_wall": 0.0,
    }


def _load_energy_hdf5(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        planes = np.asarray(handle["d"], dtype=np.float64)
    if planes.ndim != 4 or planes.shape[0] < 2:
        raise ValueError(f"unexpected MATLAB energy HDF5 shape {planes.shape} for {path}")
    return planes, planes[1], planes[0]


def _compare_to_oracle(oracle_path: Path, e14_path: Path) -> dict[str, object]:
    oracle_planes, oracle_energy, oracle_scales = _load_energy_hdf5(oracle_path)
    e14_planes, e14_energy, e14_scales = _load_energy_hdf5(e14_path)
    raw_equal = bool(
        oracle_planes.shape == e14_planes.shape and np.array_equal(oracle_planes, e14_planes)
    )
    gate = evaluate_energy_float_gate(
        oracle_energy,
        e14_energy,
        oracle_scales,
        e14_scales,
        options=EnergyFloatGateOptions(strict_floats=True, use_allclose=False),
    )
    return {
        "raw_hdf5_bit_identical": raw_equal,
        "oracle_shape": list(oracle_planes.shape),
        "e14_shape": list(e14_planes.shape),
        "energy_float_gate": gate,
        "isolation_only": True,
        "not_stretch_success": True,
        "allclose_is_not_stretch_success": True,
    }


def run_e14(*, dest: Path, timebox_sec: int = TIMEBOX_SEC) -> dict[str, object]:
    refuse_protected_stretch_energy_dest(dest)
    memory = _memory_check_crop()
    if not bool(memory["ok"]):
        return {
            "result": "deferred",
            "status_class": "incomplete_infra",
            "reason": "crop Energy memory check failed",
            "memory": memory,
        }
    original_src = ORACLE_BATCH / "data" / ORIGINAL_HANDLE
    oracle_energy = ORACLE_BATCH / "data" / "energy_260624-105705_180709_E_crop_M"
    settings_path = ORACLE_BATCH / "settings" / "energy_260624-105705.mat"
    if not original_src.is_file() or not oracle_energy.is_file() or not settings_path.is_file():
        return {
            "result": "skip",
            "status_class": "incomplete_infra",
            "reason": "crop original / oracle Energy / settings missing",
        }
    python37 = resolve_python37_executable()
    if python37 is None:
        return {
            "result": "skip",
            "status_class": "incomplete_infra",
            "reason": "isolated Python 3.7 stretch env missing",
        }
    try:
        resolve_matlab_root()
    except MatlabEngineInfraError as exc:
        return {"result": "skip", "status_class": "incomplete_infra", "reason": str(exc)}

    dest.mkdir(parents=True, exist_ok=True)
    original_dest = dest / ORIGINAL_HANDLE
    if not original_dest.is_file():
        shutil.copy2(original_src, original_dest)
    energy_out = dest / ENERGY_HANDLE
    if energy_out.exists():
        energy_out.unlink()

    params = _load_energy_settings(settings_path)
    started = dest / "e14_started.json"
    started.write_text(
        json.dumps({"started_at": time.time(), "timebox_sec": timebox_sec}, indent=2) + "\n",
        encoding="utf-8",
    )
    vectorization_root = default_vectorization_root()
    worker = MatlabEnginePy37Worker(python37, vectorization_root=vectorization_root)
    t0 = time.time()
    try:
        worker.start()
        elapsed = worker.get_energy_v202(
            matching_kernel_string=MATCHING_KERNEL,
            lumen_radius_in_microns_range=np.asarray(
                params["lumen_radius_in_microns_range"], dtype=np.float64
            ),
            vessel_wall=float(params["vessel_wall"]),
            microns_per_voxel=np.asarray(params["microns_per_voxel"], dtype=np.float64),
            pixels_per_sigma_psf=np.asarray(params["pixels_per_sigma_psf"], dtype=np.float64),
            max_voxels_per_node=float(params["max_voxels_per_node"]),
            data_directory=dest,
            original_handle=ORIGINAL_HANDLE,
            energy_handle=ENERGY_HANDLE,
            gaussian_to_ideal_ratio=float(params["gaussian_to_ideal_ratio"]),
            spherical_to_annular_ratio=float(params["spherical_to_annular_ratio"]),
        )
    except MatlabEngineInfraError as exc:
        return {
            "result": "skip",
            "status_class": "incomplete_infra",
            "reason": str(exc),
            "wall_sec": time.time() - t0,
        }
    finally:
        worker.quit()
    wall_sec = time.time() - t0
    if wall_sec > timebox_sec:
        return {
            "result": "deferred",
            "status_class": "incomplete_infra",
            "reason": f"E14 exceeded timebox {timebox_sec}s (wall {wall_sec:.1f}s)",
            "matlab_elapsed_sec": elapsed,
            "wall_sec": wall_sec,
        }
    if not energy_out.is_file():
        return {
            "result": "fail",
            "reason": "MATLAB get_energy_V202 did not write Energy HDF5",
            "matlab_elapsed_sec": elapsed,
            "wall_sec": wall_sec,
        }
    compare = _compare_to_oracle(oracle_energy, energy_out)
    gate = compare["energy_float_gate"]
    passed = bool(compare["raw_hdf5_bit_identical"]) and bool(gate.get("passed"))
    interpretation = (
        "MATLAB get_energy_V202 via engine matches oracle; residual is the chunked "
        "Python-orchestrated stretch_energy_chunk_v202 / 821-chunk lattice"
        if passed
        else "MATLAB get_energy_V202 via engine also mismatches oracle "
        "(oracle/params/orientation/batch provenance), not only the chunk helper"
    )
    return {
        "result": "pass" if passed else "fail",
        "interpretation": interpretation,
        "isolation_only": True,
        "not_stretch_success": True,
        "matlab_elapsed_sec": elapsed,
        "wall_sec": wall_sec,
        "memory": memory,
        "compare": compare,
        "dest": str(dest),
        "oracle_energy": str(oracle_energy),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--timebox-sec", type=int, default=TIMEBOX_SEC)
    parser.add_argument(
        "--claim-matlab-only-success",
        action="store_true",
        help="forbidden: MATLAB-only Energy is not stretch success (R6)",
    )
    args = parser.parse_args(argv)
    if args.claim_matlab_only_success:
        refuse_matlab_only_energy_checkpoint_as_stretch_success()
    payload = run_e14(dest=args.dest, timebox_sec=int(args.timebox_sec))
    out_path = Path(args.dest) / "e14_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"result": payload.get("result"), "path": str(out_path)}, indent=2))
    if payload.get("result") == "pass":
        return 0
    if payload.get("result") in {"skip", "deferred"}:
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
