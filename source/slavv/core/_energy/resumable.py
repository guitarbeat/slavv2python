from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from slavv.runtime import StageController


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
    *,
    get_chunking_lattice_func,
    prepare_energy_config,
    select_energy_storage_format,
    energy_lattice,
    remove_storage_path,
    open_energy_storage_array,
    compute_energy_scale,
) -> dict[str, Any]:
    """Compute energy with resumable chunk/scale units backed by persistent arrays."""
    config = prepare_energy_config(image, params)
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

    energy_4d = None
    if config["return_all_scales"]:
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
            mask = chunk_inner < target_view if config["energy_sign"] < 0 else chunk_inner > target_view
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

    result = {
        "energy": np.asarray(best_energy),
        "scale_indices": np.asarray(best_scale),
        "lumen_radius_microns": config["lumen_radius_microns"],
        "lumen_radius_pixels": config["lumen_radius_pixels"],
        "lumen_radius_pixels_axes": config["lumen_radius_pixels_axes"],
        "pixels_per_sigma_PSF": config["pixels_per_sigma_PSF"],
        "microns_per_sigma_PSF": config["microns_per_sigma_PSF"],
        "energy_sign": config["energy_sign"],
        "image_shape": image.shape,
    }
    if energy_4d is not None:
        result["energy_4d"] = np.asarray(energy_4d)
    return result
