"""Resumable energy execution helpers."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import numpy as np

from slavv_python.pipeline.energy import matlab_energy_filter_v200 as native_hessian
from slavv_python.pipeline.energy.chunking import (
    _compute_energy_scale as compute_energy_scale,
)
from slavv_python.pipeline.energy.chunking import (
    _energy_lattice as energy_lattice,
)
from slavv_python.pipeline.energy.chunking import (
    _energy_result_payload,
)
from slavv_python.pipeline.energy.chunking import (
    _open_energy_storage_array as open_energy_storage_array,
)
from slavv_python.pipeline.energy.chunking import (
    _project_scale_stack as project_scale_stack,
)
from slavv_python.pipeline.energy.chunking import (
    _remove_storage_path as remove_storage_path,
)
from slavv_python.pipeline.energy.chunking import (
    _select_energy_storage_format as select_energy_storage_format,
)
from slavv_python.pipeline.energy.config import (
    _prepare_energy_config as prepare_energy_config,
)
from slavv_python.pipeline.energy.provenance import energy_origin_for_method
from slavv_python.schema.results import EnergyResult

if TYPE_CHECKING:
    from pathlib import Path

    from slavv_python.engine.state import StageController


def _config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "params": {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in config.items()
                    if k not in {"image_shape", "image_dtype"}
                },
                "shape": list(config["image_shape"]),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _artifact_paths(
    stage_controller: StageController,
    *,
    storage_format: str,
) -> tuple[Path, Path, Path]:
    energy_suffix = ".zarr" if storage_format == "zarr" else ".npy"
    return (
        stage_controller.artifact_path(f"best_energy{energy_suffix}"),
        stage_controller.artifact_path(f"best_scale{energy_suffix}"),
        stage_controller.artifact_path(f"energy_4d{energy_suffix}"),
    )


def _clear_stale_artifacts(
    stage_controller: StageController,
    *,
    keep_paths: tuple[Path, Path, Path],
    remove_storage_path,
) -> None:
    for legacy_path in (
        stage_controller.artifact_path("best_energy.npy"),
        stage_controller.artifact_path("best_scale.npy"),
        stage_controller.artifact_path("energy_4d.npy"),
        stage_controller.artifact_path("best_energy.zarr"),
        stage_controller.artifact_path("best_scale.zarr"),
        stage_controller.artifact_path("energy_4d.zarr"),
    ):
        if legacy_path not in keep_paths:
            remove_storage_path(legacy_path)


def calculate_energy_field_resumable(
    image: np.ndarray,
    params: dict[str, Any],
    stage_controller: StageController,
    get_chunking_lattice_func=None,
) -> EnergyResult:
    """Compute energy with resumable chunk/scale units backed by persistent arrays."""
    if get_chunking_lattice_func is None:
        from slavv_python.utils import get_chunking_lattice

        get_chunking_lattice_func = get_chunking_lattice

    config = prepare_energy_config(image, params)

    if config.get("comparison_exact_network"):
        from slavv_python.pipeline.energy.matlab_get_energy_v202_chunked import (
            compute_exact_parity_energy_single_octave,
            get_chunking_lattice_v190,
        )

        config_hash = _config_hash(config)
        total_voxels = int(np.prod(image.shape))
        storage_format = select_energy_storage_format(config, total_voxels)
        energy_path, scale_path, energy4d_path = _artifact_paths(
            stage_controller,
            storage_format=storage_format,
        )
        _clear_stale_artifacts(
            stage_controller,
            keep_paths=(energy_path, scale_path, energy4d_path),
            remove_storage_path=remove_storage_path,
        )

        # Load state for resume detection
        state = stage_controller.load_state() or {}
        completed_octaves = state.get("completed_octaves", [])
        if state.get("config_hash") != config_hash:
            completed_octaves = []

        octave_at_scales = config["octave_at_scales"]
        octave_range = np.unique(octave_at_scales)

        energy_3d = np.zeros(image.shape, dtype=np.float64)
        scale_indices = np.full(image.shape, -1, dtype=np.int16)

        # Load from the last completed octave checkpoint if resuming
        if completed_octaves:
            # Filter to only valid octaves in the range
            completed_octaves = [int(o) for o in completed_octaves if o in octave_range]
            if completed_octaves:
                last_octave = max(completed_octaves)
                last_energy_path = stage_controller.artifact_path(f"octave_energy_{last_octave}.npy")
                last_scale_path = stage_controller.artifact_path(f"octave_scale_{last_octave}.npy")
                if last_energy_path.exists() and last_scale_path.exists():
                    try:
                        energy_3d = np.load(last_energy_path)
                        scale_indices = np.load(last_scale_path)
                    except Exception:
                        # Fallback if checkpoint files are corrupted
                        completed_octaves = []
                else:
                    completed_octaves = []

        microns_per_voxel = np.asarray(config["microns_per_voxel"], dtype=float)
        image_shape = np.asarray(image.shape, dtype=float)

        # Pre-compute total chunks for all octaves to provide correct progress estimation
        total_chunk_units = 0
        for planned_octave in octave_range:
            planned_scales = np.where(octave_at_scales == planned_octave)[0]
            if len(planned_scales) == 0:
                continue
            planned_rf = np.asarray(config["scale_resolution_factors"][planned_scales[0]], dtype=float)
            planned_shape = np.array([image_shape[1], image_shape[2], image_shape[0]], dtype=float)
            planned_rf_matlab = np.array([planned_rf[1], planned_rf[2], planned_rf[0]], dtype=float)
            planned_microns = microns_per_voxel * planned_rf
            _, planned_chunks = get_chunking_lattice_v190(
                1.0 / np.array([planned_microns[1], planned_microns[2], planned_microns[0]]),
                float(config["max_voxels"]),
                np.round(planned_shape / planned_rf_matlab),
            )
            total_chunk_units += int(planned_chunks)

        # Compute starting progress unit
        completed_chunk_units = 0
        for planned_octave in completed_octaves:
            planned_scales = np.where(octave_at_scales == planned_octave)[0]
            if len(planned_scales) == 0:
                continue
            planned_rf = np.asarray(config["scale_resolution_factors"][planned_scales[0]], dtype=float)
            planned_shape = np.array([image_shape[1], image_shape[2], image_shape[0]], dtype=float)
            planned_rf_matlab = np.array([planned_rf[1], planned_rf[2], planned_rf[0]], dtype=float)
            planned_microns = microns_per_voxel * planned_rf
            _, planned_chunks = get_chunking_lattice_v190(
                1.0 / np.array([planned_microns[1], planned_microns[2], planned_microns[0]]),
                float(config["max_voxels"]),
                np.round(planned_shape / planned_rf_matlab),
            )
            completed_chunk_units += int(planned_chunks)

        n_jobs = max(1, int(config.get("n_jobs", 2)))
        resumed = bool(completed_octaves)
        stage_controller.begin(
            detail="Computing exact-route octave-chunked energy",
            units_total=total_chunk_units,
            units_completed=completed_chunk_units,
            substage="exact_parity_chunks",
            resumed=resumed,
        )

        def _exact_progress(completed: int, total: int, octave: int, chunk_idx: int) -> None:
            detail = f"Exact Energy octave {octave}, chunk {chunk_idx + 1}"
            stage_controller.update(
                units_total=total_chunk_units,
                units_completed=completed,
                detail=detail,
                substage="exact_parity_chunks",
                resumed=resumed,
            )

        for current_octave in octave_range:
            if current_octave in completed_octaves:
                continue

            completed_chunk_units = compute_exact_parity_energy_single_octave(
                image=image,
                config=config,
                current_octave=current_octave,
                energy_3d=energy_3d,
                scale_indices=scale_indices,
                completed_chunk_units=completed_chunk_units,
                total_chunk_units=total_chunk_units,
                progress_callback=_exact_progress,
                n_jobs=n_jobs,
            )

            # Flush octave checkpoints
            oct_energy_path = stage_controller.artifact_path(f"octave_energy_{current_octave}.npy")
            oct_scale_path = stage_controller.artifact_path(f"octave_scale_{current_octave}.npy")
            np.save(oct_energy_path, energy_3d)
            np.save(oct_scale_path, scale_indices)

            completed_octaves.append(int(current_octave))
            stage_controller.save_state(
                {
                    "completed_octaves": completed_octaves,
                    "config_hash": config_hash,
                    "resumable": True,
                }
            )

        # Apply final post-processing clamp as in compute_exact_parity_energy_chunked
        energy_3d[energy_3d >= 0.0] = 0.0
        energy_3d[~np.isfinite(energy_3d)] = 0.0
        scale_indices[energy_3d >= 0.0] = -1

        best_energy = open_energy_storage_array(
            energy_path,
            mode="w",
            dtype=np.float64,
            shape=tuple(image.shape),
            fill_value=np.inf if config["energy_sign"] < 0 else -np.inf,
            storage_format=storage_format,
        )
        best_scale = open_energy_storage_array(
            scale_path,
            mode="w",
            dtype=np.int16,
            shape=tuple(image.shape),
            fill_value=0,
            storage_format=storage_format,
        )
        best_energy[...] = energy_3d
        best_scale[...] = scale_indices

        # Clean up temporary octave files
        for o in octave_range:
            oct_energy_path = stage_controller.artifact_path(f"octave_energy_{o}.npy")
            oct_scale_path = stage_controller.artifact_path(f"octave_scale_{o}.npy")
            if oct_energy_path.exists():
                oct_energy_path.unlink()
            if oct_scale_path.exists():
                oct_scale_path.unlink()

        stage_controller.remove_state()
        stage_controller.update(
            units_total=total_chunk_units,
            units_completed=total_chunk_units,
            detail="Exact-route energy field complete",
            substage="exact_parity_chunks",
        )
        return _energy_result_payload(config, image.shape, energy_3d, scale_indices, None)

    config_hash = _config_hash(config)

    total_voxels = int(np.prod(image.shape))
    storage_format = select_energy_storage_format(config, total_voxels)
    lattice = energy_lattice(
        image.shape,
        int(config["max_voxels"]),
        int(config["margin"]),
        get_chunking_lattice_func,
    )

    energy_path, scale_path, energy4d_path = _artifact_paths(
        stage_controller,
        storage_format=storage_format,
    )
    state = stage_controller.load_state()
    completed_units = set(state.get("completed_units", []))
    if state.get("config_hash") not in (None, config_hash):
        completed_units = set()
        for stale_path in (energy_path, scale_path, energy4d_path):
            remove_storage_path(stale_path)
    _clear_stale_artifacts(
        stage_controller,
        keep_paths=(energy_path, scale_path, energy4d_path),
        remove_storage_path=remove_storage_path,
    )

    n_scales = len(config["lumen_radius_microns"])
    total_units = len(lattice) * n_scales
    resumed = bool(completed_units)
    stage_controller.begin(
        detail="Computing resumable energy field",
        units_total=total_units,
        units_completed=len(completed_units),
        substage="scale_chunks",
        resumed=resumed,
    )

    best_energy = open_energy_storage_array(
        energy_path,
        mode="r+" if energy_path.exists() else "w",
        dtype=np.float32,
        shape=tuple(image.shape),
        fill_value=np.inf if config["energy_sign"] < 0 else -np.inf,
        storage_format=storage_format,
    )
    best_scale = open_energy_storage_array(
        scale_path,
        mode="r+" if scale_path.exists() else "w",
        dtype=np.int16,
        shape=tuple(image.shape),
        fill_value=0,
        storage_format=storage_format,
    )

    store_scale_stack = native_hessian.required_scale_stack(config)
    energy_4d = None
    if store_scale_stack:
        energy_4d = open_energy_storage_array(
            energy4d_path,
            mode="r+" if energy4d_path.exists() else "w",
            dtype=np.float32,
            shape=(*image.shape, n_scales),
            fill_value=0.0,
            storage_format=storage_format,
        )

    for chunk_idx, (chunk_slice, out_slice, inner_slice) in enumerate(lattice):
        chunk_img = image[chunk_slice]
        for scale_idx in range(n_scales):
            unit_id = f"{chunk_idx}:{scale_idx}"
            if unit_id in completed_units:
                continue

            energy_scale = compute_energy_scale(chunk_img, config, scale_idx)
            chunk_inner = energy_scale[inner_slice]
            target_view = best_energy[out_slice]
            mask = (
                chunk_inner < target_view
                if config["energy_sign"] < 0
                else chunk_inner > target_view
            )
            target_view[mask] = chunk_inner[mask]
            best_energy[out_slice] = target_view

            scale_view = best_scale[out_slice]
            scale_view[mask] = scale_idx
            best_scale[out_slice] = scale_view
            if energy_4d is not None:
                energy_4d[(*out_slice, scale_idx)] = chunk_inner

            completed_units.add(unit_id)
            state = {
                "config_hash": config_hash,
                "completed_units": sorted(completed_units),
                "total_units": total_units,
                "n_chunks": len(lattice),
                "n_scales": n_scales,
                "storage_format": storage_format,
            }
            stage_controller.save_state(state)
            stage_controller.update(
                units_total=total_units,
                units_completed=len(completed_units),
                detail=(
                    f"Energy volume tile {chunk_idx + 1}/{len(lattice)}, "
                    f"vessel scale {scale_idx + 1}/{n_scales}"
                ),
                substage="scale_chunks",
                resumed=resumed,
            )

    result_energy = np.asarray(best_energy)
    result_scale = np.asarray(best_scale)
    returned_energy_4d = None
    if energy_4d is not None:
        if str(config["energy_projection_mode"]) == "paper":
            result_energy, result_scale, returned_energy_4d = project_scale_stack(
                config,
                np.asarray(energy_4d),
            )
            best_energy[...] = result_energy
            best_scale[...] = result_scale
        elif bool(config["return_all_scales"]):
            returned_energy_4d = np.asarray(energy_4d)

    result = EnergyResult.create(
        energy=result_energy,
        scale_indices=result_scale,
        lumen_radius_microns=config["lumen_radius_microns"],
        lumen_radius_pixels=config["lumen_radius_pixels"],
        image_shape=image.shape,
        lumen_radius_pixels_axes=config["lumen_radius_pixels_axes"],
        pixels_per_sigma_PSF=config["pixels_per_sigma_PSF"],
        microns_per_sigma_PSF=config["microns_per_sigma_PSF"],
        energy_sign=config["energy_sign"],
        energy_origin=energy_origin_for_method(str(config["energy_method"])),
    )
    if returned_energy_4d is not None:
        result.extra["energy_4d"] = returned_energy_4d

    return result


__all__ = ["calculate_energy_field_resumable"]
