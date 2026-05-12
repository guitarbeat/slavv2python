"""Energy chunking and storage helpers."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from joblib import Parallel, delayed

from slavv_python.core.energy import storage as _energy_storage

from slavv_python.core.energy import backends as backends
from slavv_python.core.energy import hessian_response as native_hessian
from slavv_python.core.energy.provenance import energy_origin_for_method


def _select_energy_storage_format(config: dict[str, Any], total_voxels: int) -> str:
    """Choose the resumable energy array storage backend."""
    storage_format = _energy_storage.select_energy_storage_format(
        str(config.get("energy_storage_format", "auto")),
        total_voxels=total_voxels,
        max_voxels=int(config["max_voxels"]),
        require_zarr_backend=backends._require_zarr_backend,
    )
    return cast("str", storage_format)


def _remove_storage_path(path: Any) -> None:
    """Remove a file or directory-backed storage artifact."""
    _energy_storage.remove_storage_path(path)


def _open_energy_storage_array(
    path: Any,
    *,
    mode: str,
    dtype: Any,
    shape: tuple[int, ...],
    fill_value: float | int | None = None,
    storage_format: str,
) -> Any:
    """Open a resumable energy array in either NPY memmap or Zarr format."""
    return _energy_storage.open_energy_storage_array(
        path,
        mode=mode,
        dtype=dtype,
        shape=shape,
        fill_value=fill_value,
        storage_format=storage_format,
        require_zarr_backend=backends._require_zarr_backend,
    )


def _energy_lattice(
    image_shape: tuple[int, ...],
    max_voxels: int,
    margin: int,
    get_chunking_lattice_func,
) -> list[
    tuple[tuple[slice, slice, slice], tuple[slice, slice, slice], tuple[slice, slice, slice]]
]:
    total_voxels = int(np.prod(image_shape))
    if total_voxels > max_voxels:
        return cast(
            "list[tuple[tuple[slice, slice, slice], tuple[slice, slice, slice], tuple[slice, slice, slice]]]",
            get_chunking_lattice_func(image_shape, max_voxels, margin),
        )
    return [
        (
            (slice(0, image_shape[0]), slice(0, image_shape[1]), slice(0, image_shape[2])),
            (slice(0, image_shape[0]), slice(0, image_shape[1]), slice(0, image_shape[2])),
            (slice(0, image_shape[0]), slice(0, image_shape[1]), slice(0, image_shape[2])),
        )
    ]


def _energy_result_payload(
    config: dict[str, Any],
    image_shape: tuple[int, ...],
    energy_3d: np.ndarray,
    scale_indices: np.ndarray,
    energy_4d: np.ndarray | None = None,
) -> dict[str, Any]:
    result = {
        "energy": energy_3d,
        "scale_indices": scale_indices,
        "lumen_radius_microns": config["lumen_radius_microns"],
        "lumen_radius_pixels": config["lumen_radius_pixels"],
        "lumen_radius_pixels_axes": config["lumen_radius_pixels_axes"],
        "pixels_per_sigma_PSF": config["pixels_per_sigma_PSF"],
        "microns_per_sigma_PSF": config["microns_per_sigma_PSF"],
        "energy_sign": config["energy_sign"],
        "energy_origin": energy_origin_for_method(str(config["energy_method"])),
        "image_shape": image_shape,
    }
    if energy_4d is not None:
        result["energy_4d"] = energy_4d
    return result


def _best_energy_outputs(
    image_shape: tuple[int, ...],
    energy_sign: float,
    n_scales: int,
    return_all_scales: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    fill_value = np.inf if energy_sign < 0 else -np.inf
    energy_3d = np.full(image_shape, fill_value, dtype=np.float32)
    scale_indices: np.ndarray = np.zeros(image_shape, dtype=np.int16)
    energy_4d = np.zeros((*image_shape, n_scales), dtype=np.float32) if return_all_scales else None
    return energy_3d, scale_indices, energy_4d


def _returned_energy_4d(config: dict[str, Any], energy_4d: np.ndarray | None) -> np.ndarray | None:
    if bool(config["return_all_scales"]):
        return energy_4d
    return None


def _project_scale_stack(
    config: dict[str, Any],
    energy_4d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    energy_3d, scale_indices = native_hessian.project_energy_stack(
        energy_4d,
        energy_sign=float(config["energy_sign"]),
        projection_mode=str(config["energy_projection_mode"]),
        spherical_to_annular_ratio=float(config["spherical_to_annular_ratio"]),
    )
    return energy_3d, scale_indices, _returned_energy_4d(config, energy_4d)


def _update_best_energy(
    energy_3d: np.ndarray,
    scale_indices: np.ndarray,
    energy_scale: np.ndarray,
    scale_idx: int,
    energy_sign: float,
) -> None:
    mask = energy_scale < energy_3d if energy_sign < 0 else energy_scale > energy_3d
    energy_3d[mask] = energy_scale[mask]
    scale_indices[mask] = scale_idx


def _compute_energy_scale(image: np.ndarray, config: dict[str, Any], scale_idx: int) -> np.ndarray:
    """Compute a single-scale energy response for a chunk."""
    image = image.astype(np.float32, copy=False)
    energy_method = config["energy_method"]
    energy_sign = config["energy_sign"]
    if energy_method == "hessian":
        return native_hessian.compute_native_hessian_energy(image, config, scale_idx)

    sigma_scale = config["lumen_radius_microns"][scale_idx] / config["microns_per_voxel"]
    sigma_scale = sigma_scale / max(config["gaussian_to_ideal_ratio"], 1e-12)
    sigma_scale = np.asarray(sigma_scale, dtype=float)

    if config["approximating_PSF"]:
        sigma_object = np.sqrt(sigma_scale**2 + config["pixels_per_sigma_PSF"] ** 2)
    else:
        sigma_object = sigma_scale

    if energy_method in ("frangi", "sato"):
        sigma = float(config["lumen_radius_pixels"][scale_idx])
        if energy_method == "frangi":
            vesselness = backends.frangi(image, sigmas=[sigma], black_ridges=(energy_sign > 0))
        else:
            vesselness = backends.sato(image, sigmas=[sigma], black_ridges=(energy_sign > 0))
        return energy_sign * vesselness.astype(np.float32)  # type: ignore[no-any-return]

    if energy_method == "simpleitk_objectness":
        return backends._simpleitk_objectness_energy(
            image,
            float(config["lumen_radius_microns"][scale_idx]),
            np.asarray(config["microns_per_voxel"], dtype=float),
            float(energy_sign),
        )
    if energy_method == "cupy_hessian":
        if config["spherical_to_annular_ratio"] < 1.0:
            annular_scale = sigma_scale * 1.5
            if config["approximating_PSF"]:
                sigma_background = np.sqrt(annular_scale**2 + config["pixels_per_sigma_PSF"] ** 2)
            else:
                sigma_background = annular_scale
        else:
            sigma_background = None
        return backends._cupy_matlab_hessian_energy(
            image,
            sigma_object,
            sigma_background,
            float(config["spherical_to_annular_ratio"]),
            np.asarray(config["microns_per_voxel"], dtype=float),
            float(energy_sign),
        )

    raise ValueError(f"Unsupported energy_method: {energy_method!r}")


def _compute_direct_energy_outputs(
    image: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    n_scales = len(config["lumen_radius_microns"])
    n_jobs = int(config.get("n_jobs", 1))

    results = Parallel(n_jobs=n_jobs)(
        delayed(_compute_energy_scale)(image, config, scale_idx) for scale_idx in range(n_scales)
    )

    if native_hessian.required_scale_stack(config):
        energy_4d = np.stack(results, axis=3).astype(np.float32)
        return _project_scale_stack(config, energy_4d)

    energy_3d, scale_indices, energy_4d = _best_energy_outputs(
        image.shape,
        float(config["energy_sign"]),
        n_scales,
        bool(config["return_all_scales"]),
    )
    for scale_idx, energy_scale in enumerate(results):
        if energy_4d is not None:
            energy_4d[..., scale_idx] = energy_scale
        _update_best_energy(
            energy_3d,
            scale_indices,
            energy_scale,
            scale_idx,
            float(config["energy_sign"]),
        )
    return energy_3d, scale_indices, energy_4d


def _calculate_energy_field_chunked(
    image: np.ndarray,
    params: dict[str, Any],
    config: dict[str, Any],
    lattice,
    get_chunking_lattice_func,
    calculate_energy_field,
) -> dict[str, Any]:
    n_jobs = int(config.get("n_jobs", 1))

    def _worker(chunk_slice, out_slice, inner_slice, return_all_scales: bool):
        chunk_img = image[chunk_slice]
        sub_params = params.copy()
        sub_params["max_voxels_per_node_energy"] = chunk_img.size + 1
        sub_params["return_all_scales"] = return_all_scales
        sub_params["n_jobs"] = 1  # Disable nested parallelism
        chunk_data = calculate_energy_field(chunk_img, sub_params, get_chunking_lattice_func)
        return out_slice, inner_slice, chunk_data

    if native_hessian.required_scale_stack(config):
        n_scales = len(config["lumen_radius_microns"])
        energy_4d = np.zeros((*image.shape, n_scales), dtype=np.float32)

        results = Parallel(n_jobs=n_jobs)(
            delayed(_worker)(chunk_slice, out_slice, inner_slice, True)
            for chunk_slice, out_slice, inner_slice in lattice
        )

        for out_slice, inner_slice, chunk_data in results:
            energy_4d[(*out_slice, slice(None))] = chunk_data["energy_4d"][
                (*inner_slice, slice(None))
            ]

        energy_3d, scale_indices, returned_energy_4d = _project_scale_stack(config, energy_4d)
        return _energy_result_payload(
            config,
            image.shape,
            energy_3d,
            scale_indices,
            returned_energy_4d,
        )

    energy_3d = np.empty(image.shape, dtype=np.float32)
    scale_indices = np.empty(image.shape, dtype=np.int16)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_worker)(chunk_slice, out_slice, inner_slice, False)
        for chunk_slice, out_slice, inner_slice in lattice
    )

    for out_slice, inner_slice, chunk_data in results:
        energy_3d[out_slice] = chunk_data["energy"][inner_slice]
        scale_indices[out_slice] = chunk_data["scale_indices"][inner_slice]

    return _energy_result_payload(config, image.shape, energy_3d, scale_indices)


__all__ = [
    "_calculate_energy_field_chunked",
    "_compute_direct_energy_outputs",
    "_compute_energy_scale",
    "_energy_lattice",
    "_energy_result_payload",
    "_open_energy_storage_array",
    "_project_scale_stack",
    "_remove_storage_path",
    "_select_energy_storage_format",
]
