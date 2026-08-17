#!/usr/bin/env python
"""Re-run one v2 production crop Energy chunk (isolation, not unlock).

Maps the known mismatch voxel ZYX (13, 0, 0) / winner scale 43 onto the v2
chunk lattice and calls ``stretch_energy_chunk_v202`` for that chunk only.
Writes scratch JSON; patches dest ``stretch_status.json`` extra only.
Never overwrites Energy on protected dests. Never emits stretch_complete.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from slavv_python.analytics.parity.constants import (
    ANALYSIS_DIR,
    CHECKPOINTS_DIR,
    EXPERIMENT_PARAMS_DIR,
    EXPERIMENT_REFS_DIR,
    VALIDATED_PARAMS_PATH,
)
from slavv_python.analytics.parity.oracle.matlab_vector_loader import load_normalized_matlab_vectors
from slavv_python.analytics.parity.oracle.surfaces import load_oracle_surface
from slavv_python.analytics.parity.proof.stretch import (
    STATUS_FILENAME,
    StretchStatus,
    classify_stretch_energy_orientation,
)
from slavv_python.pipeline.energy.config import _prepare_energy_config
from slavv_python.pipeline.energy.matlab_engine_backend import (
    MatlabEngineInfraError,
    refuse_protected_stretch_energy_dest,
    resolve_matlab_root,
)
from slavv_python.pipeline.energy.matlab_engine_host import (
    resolve_python37_executable,
    stretch_engine_float_body_session,
)
from slavv_python.pipeline.energy.stretch_chunk_isolation import (
    DEFAULT_MISMATCH_VOXEL_ZYX,
    DEFAULT_WINNER_SCALE,
    INTERPRET_INCOMPLETE_INFRA,
    build_octave_chunk_lattice,
    chunk_index_for_voxel_zyx,
    compare_three_way,
    hit_to_dict,
    octave_owned_mask,
    patch_stretch_status_extra,
    run_stretch_chunk_v202,
)
from slavv_python.schema.results import EnergyResult
from slavv_python.storage.loaders.tiff import load_tiff_volume
from slavv_python.utils.validation import validate_parameters

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = REPO_ROOT / "workspace" / "runs" / "oracle_180709_E" / "crop_M_stretch_engine_v2"
DEFAULT_ORACLE = REPO_ROOT / "workspace" / "oracles" / "180709_E_crop_M_v2"
DEFAULT_SCRATCH = REPO_ROOT / "workspace" / "scratch" / "stretch_one_production_chunk.json"
CROP_TIF_NAME = "180709_E_crop_M.tif"
TIMEBOX_SEC = 600


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _voxel_and_scale_from_mismatch(path: Path | None) -> tuple[tuple[int, int, int], int]:
    if path is None or not path.is_file():
        return DEFAULT_MISMATCH_VOXEL_ZYX, DEFAULT_WINNER_SCALE
    payload = _load_json(path)
    context = payload.get("energy_context") or {}
    coord = context.get("coordinate_zyx")
    scale = context.get("python_winner_scale")
    if scale is None:
        scale = context.get("matlab_winner_scale")
    if coord is None:
        for field in payload.get("fields") or []:
            if str(field.get("field")) == "energy" and "first_coordinate" in field:
                coord = field["first_coordinate"]
                break
    voxel = (
        tuple(int(v) for v in coord)
        if isinstance(coord, (list, tuple)) and len(coord) == 3
        else DEFAULT_MISMATCH_VOXEL_ZYX
    )
    winner = int(scale) if scale is not None else DEFAULT_WINNER_SCALE
    return (int(voxel[0]), int(voxel[1]), int(voxel[2])), winner


def _load_dest_params(dest: Path) -> dict[str, Any]:
    candidates = (
        dest / VALIDATED_PARAMS_PATH,
        dest / EXPERIMENT_PARAMS_DIR / "validated_params.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return _load_json(candidate)
    raise FileNotFoundError(f"validated_params.json missing under {dest}")


def _find_crop_tif(dest: Path) -> Path:
    refs = dest / EXPERIMENT_REFS_DIR / CROP_TIF_NAME
    if refs.is_file():
        return refs
    datasets = REPO_ROOT / "workspace" / "datasets"
    if datasets.is_dir():
        matches = sorted(datasets.glob(f"*/01_Input/{CROP_TIF_NAME}"))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"crop TIFF missing: expected {refs} (not 01_Input) or workspace/datasets/"
    )


def _reorient_image_to_energy(image: np.ndarray, energy_shape: tuple[int, ...]) -> np.ndarray:
    if tuple(int(v) for v in image.shape) == tuple(int(v) for v in energy_shape):
        return image
    for perm in itertools.permutations((0, 1, 2)):
        reordered = tuple(int(image.shape[i]) for i in perm)
        if reordered == tuple(int(v) for v in energy_shape):
            return np.transpose(image, perm)
    raise ValueError(f"cannot reorient image {image.shape} to energy {energy_shape}")


def _load_dest_energy(dest: Path) -> tuple[np.ndarray, np.ndarray]:
    energy_npy = dest / "02_Energy" / "best_energy.npy"
    scale_npy = dest / "02_Energy" / "best_scale.npy"
    if energy_npy.is_file() and scale_npy.is_file():
        return (
            np.asarray(np.load(energy_npy), dtype=np.float64),
            np.asarray(np.load(scale_npy), dtype=np.int16),
        )
    checkpoint = dest / CHECKPOINTS_DIR / "checkpoint_energy.pkl"
    if checkpoint.is_file():
        result = EnergyResult.load(checkpoint)
        return (
            np.asarray(result.energy, dtype=np.float64),
            np.asarray(result.scale_indices, dtype=np.int16),
        )
    raise FileNotFoundError(f"v2 Energy missing under {dest} (npy or checkpoint_energy.pkl)")


def _load_oracle_energy(oracle_root: Path) -> tuple[np.ndarray, np.ndarray]:
    surface = load_oracle_surface(oracle_root)
    if surface.matlab_batch_dir is None:
        raise FileNotFoundError(f"oracle batch missing under {oracle_root}")
    payload = load_normalized_matlab_vectors(surface.matlab_batch_dir, ("energy",))["energy"]
    return (
        np.asarray(payload["energy"], dtype=np.float64),
        np.asarray(payload["scale_indices"], dtype=np.int16),
    )


def _local_voxel(
    voxel_zyx: tuple[int, int, int], write_start_zyx: tuple[int, int, int]
) -> tuple[int, int, int]:
    return (
        int(voxel_zyx[0] - write_start_zyx[0]),
        int(voxel_zyx[1] - write_start_zyx[1]),
        int(voxel_zyx[2] - write_start_zyx[2]),
    )


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


def run_one_production_chunk(
    *,
    dest: Path,
    oracle_root: Path,
    scratch_out: Path,
    timebox_sec: int = TIMEBOX_SEC,
    voxel_zyx: tuple[int, int, int] | None = None,
    winner_scale: int | None = None,
) -> dict[str, Any]:
    refuse_protected_stretch_energy_dest(scratch_out)
    mismatch_path = dest / ANALYSIS_DIR / "exact_mismatch_energy.json"
    default_voxel, default_scale = _voxel_and_scale_from_mismatch(mismatch_path)
    voxel = voxel_zyx or default_voxel
    scale = int(winner_scale) if winner_scale is not None else default_scale

    lattice_payload: dict[str, Any] = {
        "voxel_zyx": list(voxel),
        "winner_scale": scale,
        "dest": str(dest),
        "oracle_root": str(oracle_root),
    }
    try:
        dest_energy, dest_scales = _load_dest_energy(dest)
        oracle_energy, oracle_scales = _load_oracle_energy(oracle_root)
        params = _load_dest_params(dest)
        image = load_tiff_volume(_find_crop_tif(dest))
        image = _reorient_image_to_energy(image, dest_energy.shape)
        params = validate_parameters(params)
        config = _prepare_energy_config(image, params)
    except (FileNotFoundError, ValueError, OSError) as exc:
        payload = _incomplete(str(exc), lattice=lattice_payload)
        return payload

    orientation = classify_stretch_energy_orientation(
        energy_shape=tuple(int(v) for v in dest_energy.shape),
        oracle_shape=tuple(int(v) for v in oracle_energy.shape),
    )
    if orientation is not None:
        return _incomplete(orientation.reason, lattice=lattice_payload)

    try:
        lattice = build_octave_chunk_lattice(config, config["octave_at_scales"][scale])
        hit = chunk_index_for_voxel_zyx(config, voxel, winner_scale=scale, lattice=lattice)
    except (ValueError, IndexError) as exc:
        return _incomplete(str(exc), lattice=lattice_payload)
    lattice_payload.update(hit_to_dict(hit))
    slices = hit.write_slices_zyx
    v2_window = np.asarray(dest_energy[slices], dtype=np.float64)
    v2_scales_w = np.asarray(dest_scales[slices], dtype=np.int16)
    oracle_window = np.asarray(oracle_energy[slices], dtype=np.float64)
    v2_vs_oracle = compare_three_way(v2_window, v2_window, oracle_window)

    python37 = resolve_python37_executable()
    if python37 is None:
        payload = _incomplete(
            "isolated Python 3.7 stretch env missing",
            lattice=lattice_payload,
            v2_vs_oracle_window=v2_vs_oracle.to_dict(),
        )
        return payload
    try:
        resolve_matlab_root()
    except MatlabEngineInfraError as exc:
        return _incomplete(
            str(exc),
            lattice=lattice_payload,
            v2_vs_oracle_window=v2_vs_oracle.to_dict(),
        )

    engine_params = dict(params)
    engine_params["energy_float_backend"] = "matlab_engine"
    engine_params["energy_method"] = "hessian"
    engine_params["n_jobs"] = 1
    t0 = time.time()
    try:
        with stretch_engine_float_body_session(engine_params) as bound:
            bound_config = _prepare_energy_config(image, bound)
            rerun_e, rerun_s, _write = run_stretch_chunk_v202(
                image,
                bound_config,
                lattice,
                hit.chunk_index,
                bound_config.get("_stretch_engine_session"),
            )
    except MatlabEngineInfraError as exc:
        return _incomplete(
            str(exc),
            lattice=lattice_payload,
            v2_vs_oracle_window=v2_vs_oracle.to_dict(),
            wall_sec=time.time() - t0,
        )
    wall_sec = time.time() - t0
    if wall_sec > timebox_sec:
        return _incomplete(
            f"one-chunk exceeded timebox {timebox_sec}s (wall {wall_sec:.1f}s)",
            lattice=lattice_payload,
            wall_sec=wall_sec,
        )

    owned = octave_owned_mask(v2_scales_w, hit.scale_indices_at_octave)
    lz, ly, lx = _local_voxel(voxel, hit.write_start_zyx)
    voxel_rerun = np.asarray(rerun_e[lz, ly, lx], dtype=np.float64).reshape((1, 1, 1))
    voxel_v2 = np.asarray(v2_window[lz, ly, lx], dtype=np.float64).reshape((1, 1, 1))
    voxel_oracle = np.asarray(oracle_window[lz, ly, lx], dtype=np.float64).reshape((1, 1, 1))
    full_cmp = compare_three_way(rerun_e, v2_window, oracle_window)
    owned_cmp = (
        compare_three_way(rerun_e[owned], v2_window[owned], oracle_window[owned])
        if np.any(owned)
        else None
    )
    voxel_cmp = compare_three_way(voxel_rerun, voxel_v2, voxel_oracle)
    primary = owned_cmp if owned_cmp is not None else full_cmp
    return {
        "result": "ok",
        "status_class": StretchStatus.BLOCKED_FLOAT_PATH.value,
        "interpretation": primary.interpretation,
        "primary_surface": "octave_owned" if owned_cmp is not None else "full_write_window",
        "isolation_only": True,
        "not_stretch_success": True,
        "stretch_complete": False,
        "wall_sec": wall_sec,
        "lattice": lattice_payload,
        "full_write_window": full_cmp.to_dict(),
        "octave_owned": None if owned_cmp is None else owned_cmp.to_dict(),
        "mismatch_voxel": {
            **voxel_cmp.to_dict(),
            "voxel_zyx": list(voxel),
            "local_zyx": [lz, ly, lx],
            "v2_scale": int(v2_scales_w[lz, ly, lx]),
            "rerun_scale": int(rerun_s[lz, ly, lx]),
            "oracle_scale": int(oracle_scales[slices][lz, ly, lx]),
            "v2_energy": float(voxel_v2.reshape(-1)[0]),
            "rerun_energy": float(voxel_rerun.reshape(-1)[0]),
            "oracle_energy": float(voxel_oracle.reshape(-1)[0]),
        },
        "n_octave_owned": int(np.count_nonzero(owned)),
        "v2_vs_oracle_window": v2_vs_oracle.to_dict(),
    }


def _write_outputs(payload: dict[str, Any], *, dest: Path, scratch_out: Path) -> None:
    scratch_out.parent.mkdir(parents=True, exist_ok=True)
    scratch_out.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    extra = {
        "one_production_chunk": {
            "scratch_json": str(scratch_out),
            "result": payload.get("result"),
            "status_class": payload.get("status_class"),
            "interpretation": payload.get("interpretation"),
            "lattice": payload.get("lattice"),
            "isolation_only": True,
            "not_stretch_success": True,
            "stretch_complete": False,
        }
    }
    status_path = dest / STATUS_FILENAME
    if status_path.is_file():
        patch_stretch_status_extra(status_path, extra, require_blocked_float_path=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--scratch-out", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--timebox-sec", type=int, default=TIMEBOX_SEC)
    parser.add_argument("--voxel-z", type=int, default=None)
    parser.add_argument("--voxel-y", type=int, default=None)
    parser.add_argument("--voxel-x", type=int, default=None)
    parser.add_argument("--winner-scale", type=int, default=None)
    args = parser.parse_args(argv)
    voxel = None
    if args.voxel_z is not None or args.voxel_y is not None or args.voxel_x is not None:
        voxel = (
            int(args.voxel_z if args.voxel_z is not None else DEFAULT_MISMATCH_VOXEL_ZYX[0]),
            int(args.voxel_y if args.voxel_y is not None else DEFAULT_MISMATCH_VOXEL_ZYX[1]),
            int(args.voxel_x if args.voxel_x is not None else DEFAULT_MISMATCH_VOXEL_ZYX[2]),
        )
    payload = run_one_production_chunk(
        dest=args.dest,
        oracle_root=args.oracle_root,
        scratch_out=args.scratch_out,
        timebox_sec=int(args.timebox_sec),
        voxel_zyx=voxel,
        winner_scale=args.winner_scale,
    )
    _write_outputs(payload, dest=args.dest, scratch_out=args.scratch_out)
    print(
        json.dumps(
            {
                "result": payload.get("result"),
                "interpretation": payload.get("interpretation"),
                "scratch": str(args.scratch_out),
                "status_class": payload.get("status_class"),
            },
            indent=2,
        )
    )
    if payload.get("result") == "ok":
        return 0
    if payload.get("result") in {"skip", "deferred"}:
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
