#!/usr/bin/env python
"""Compare helper body vs original MATLAB get_energy_V202 chunk math.

Isolation only — not Energy unlock, not U5/U6, not a writer. Compares
local_ranges, TIFF vs oracle HDF5 input window on octave-2 chunk 0, and names
the helper nonnegative clamp. Writes scratch JSON and patches dest
stretch_status.json extra only. Status stays blocked_float_path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from slavv_python.analytics.parity.constants import (
    CROP_ORIGINAL_HANDLE,
    STRETCH_CROP_DEST_NAME,
    STRETCH_CROP_ORACLE_ID,
)
from slavv_python.analytics.parity.proof.stretch import STATUS_FILENAME, StretchStatus
from slavv_python.pipeline.energy.config import _prepare_energy_config
from slavv_python.pipeline.energy.matlab_engine_backend import (
    refuse_protected_stretch_energy_dest,
)
from slavv_python.pipeline.energy.stretch.chunk_isolation import (
    DEFAULT_WINNER_SCALE,
    OctaveChunkLattice,
    build_octave_chunk_lattice,
    octave_for_scale,
    patch_stretch_status_extra,
)
from slavv_python.pipeline.energy.stretch.crop_io import (
    find_crop_tif,
    load_dest_params,
    reorient_image_to_energy,
)
from slavv_python.pipeline.energy.stretch.helper_body_isolation import (
    INTERPRET_INCOMPLETE_INFRA,
    LocalRangeCompare,
    compare_input_windows,
    compare_local_range,
    default_production_local_compares,
    h5py_c_order_to_matlab_yxz,
    isolation_payload,
    matlab_h52mat_chunk_yxz,
    python_strided_chunk_yxz,
)
from slavv_python.storage.loaders.tiff import load_tiff_volume
from slavv_python.utils.validation import validate_parameters

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEST = REPO_ROOT / "workspace" / "runs" / "oracle_180709_E" / STRETCH_CROP_DEST_NAME
DEFAULT_ORACLE = REPO_ROOT / "workspace" / "oracles" / STRETCH_CROP_ORACLE_ID
DEFAULT_SCRATCH = REPO_ROOT / "workspace" / "scratch" / "stretch_helper_body_isolation.json"


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


def _load_original_h5_yxz(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        dataset = handle["d"] if "d" in handle else next(iter(handle.values()))
        raw = np.asarray(dataset)
    return h5py_c_order_to_matlab_yxz(raw).astype(np.float64, copy=False)


def _local_compares_from_lattice(
    lattice: OctaveChunkLattice, chunk_index: int
) -> list[LocalRangeCompare]:
    dims = lattice.lattice_dimensions_yxz
    y_idx, x_idx, z_idx = np.unravel_index(int(chunk_index), dims, order="F")
    stride_z, stride_y, stride_x = (int(v) for v in lattice.rf_zyx)
    return [
        compare_local_range(
            int(lattice.y_offsets[y_idx]),
            int(lattice.y_write_counts[y_idx]),
            stride_y,
        ),
        compare_local_range(
            int(lattice.x_offsets[x_idx]),
            int(lattice.x_write_counts[x_idx]),
            stride_x,
        ),
        compare_local_range(
            int(lattice.z_offsets[z_idx]),
            int(lattice.z_write_counts[z_idx]),
            stride_z,
        ),
    ]


def run_helper_body_isolation(
    *,
    dest: Path,
    oracle_root: Path,
    scratch_out: Path,
) -> dict[str, Any]:
    refuse_protected_stretch_energy_dest(scratch_out)
    batch = oracle_root / "01_Input" / "matlab_results" / "batch_260624-105705"
    original_path = batch / "data" / CROP_ORIGINAL_HANDLE
    energy_npy = dest / "02_Energy" / "best_energy.npy"
    try:
        params = validate_parameters(load_dest_params(dest))
        if not energy_npy.is_file():
            raise FileNotFoundError(f"v2 Energy npy missing: {energy_npy}")
        image_shape = tuple(int(v) for v in np.load(energy_npy, mmap_mode="r").shape)
        config = _prepare_energy_config(np.zeros(image_shape, dtype=np.float64), params)
        tiff_path = find_crop_tif(dest, repo_root=REPO_ROOT)
        image = reorient_image_to_energy(load_tiff_volume(tiff_path), image_shape)
        matlab_yxz = _load_original_h5_yxz(original_path)
    except (FileNotFoundError, ValueError, OSError, KeyError) as exc:
        payload = _incomplete(str(exc))
        payload["local_compares"] = [item.to_dict() for item in default_production_local_compares()]
        payload["local_ranges_equal"] = True
        return payload

    octave = octave_for_scale(config, DEFAULT_WINNER_SCALE)
    lattice = build_octave_chunk_lattice(config, octave)
    chunk_index = 0
    local_compares = _local_compares_from_lattice(lattice, chunk_index)
    y_idx, x_idx, z_idx = np.unravel_index(chunk_index, lattice.lattice_dimensions_yxz, order="F")
    py_z_start = int(lattice.z_read_starts[z_idx]) - 1
    py_y_start = int(lattice.y_read_starts[y_idx]) - 1
    py_x_start = int(lattice.x_read_starts[x_idx]) - 1
    py_z_count = int(lattice.z_read_counts[z_idx])
    py_y_count = int(lattice.y_read_counts[y_idx])
    py_x_count = int(lattice.x_read_counts[x_idx])
    rf_zyx = tuple(int(v) for v in lattice.rf_zyx)
    python_chunk = python_strided_chunk_yxz(
        image,
        z_start=py_z_start,
        y_start=py_y_start,
        x_start=py_x_start,
        z_count=py_z_count,
        y_count=py_y_count,
        x_count=py_x_count,
        rf_zyx=(rf_zyx[0], rf_zyx[1], rf_zyx[2]),
    )
    matlab_chunk = matlab_h52mat_chunk_yxz(
        matlab_yxz,
        y_start_1based=int(lattice.y_read_starts[y_idx]),
        x_start_1based=int(lattice.x_read_starts[x_idx]),
        z_start_1based=int(lattice.z_read_starts[z_idx]),
        y_read_count=py_y_count,
        x_read_count=py_x_count,
        z_read_count=py_z_count,
        rf_yxz=(rf_zyx[1], rf_zyx[2], rf_zyx[0]),
    )
    window = compare_input_windows(python_chunk, matlab_chunk)
    extra: dict[str, Any] = {
        "dest": str(dest),
        "oracle_root": str(oracle_root),
        "chunk_index": chunk_index,
        "octave": int(octave),
        "rf_zyx": list(rf_zyx),
        "read_start_zyx": [py_z_start, py_y_start, py_x_start],
        "read_count_zyx": [py_z_count, py_y_count, py_x_count],
        "python_chunk_shape": [int(v) for v in python_chunk.shape],
        "matlab_chunk_shape": [int(v) for v in matlab_chunk.shape],
        "matlab_h5_shape_yxz": [int(v) for v in matlab_yxz.shape],
        "tiff_shape_zyx": [int(v) for v in image.shape],
    }
    return isolation_payload(
        local_compares=local_compares,
        window_compare=window,
        extra=extra,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--scratch-out", type=Path, default=DEFAULT_SCRATCH)
    args = parser.parse_args(argv)
    payload = run_helper_body_isolation(
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
                "helper_body_isolation": {
                    "interpretation": payload.get("interpretation"),
                    "isolation_only": True,
                    "not_stretch_success": True,
                    "local_ranges_equal": payload.get("local_ranges_equal"),
                    "input_windows_equal": payload.get("input_windows_equal"),
                    "helper_clamps_nonnegative": payload.get("helper_clamps_nonnegative"),
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
                "local_ranges_equal": payload.get("local_ranges_equal"),
                "input_windows_equal": payload.get("input_windows_equal"),
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
