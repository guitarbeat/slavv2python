"""Energy chunking and storage helpers."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from joblib import Parallel, delayed

from slavv_python.processing.stages.energy import backends as backends
from slavv_python.processing.stages.energy import hessian_response as native_hessian
from slavv_python.processing.stages.energy import storage as _energy_storage
from slavv_python.processing.stages.energy.provenance import energy_origin_for_method
from slavv_python.schema.results import EnergyResult


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
) -> EnergyResult:
    result = EnergyResult.create(
        energy=energy_3d,
        scale_indices=scale_indices,
        lumen_radius_microns=config["lumen_radius_microns"],
        lumen_radius_pixels=config["lumen_radius_pixels"],
        image_shape=image_shape,
        lumen_radius_pixels_axes=config["lumen_radius_pixels_axes"],
        pixels_per_sigma_PSF=config["pixels_per_sigma_PSF"],
        microns_per_sigma_PSF=config["microns_per_sigma_PSF"],
        energy_sign=config["energy_sign"],
        energy_origin=energy_origin_for_method(str(config["energy_method"])),
    )
    if energy_4d is not None:
        result.extra["energy_4d"] = energy_4d
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
    mask: np.ndarray = energy_scale < energy_3d if energy_sign < 0 else energy_scale > energy_3d
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
    if config.get("comparison_exact_network"):
        return _compute_exact_parity_energy_chunked(image, config)

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
) -> EnergyResult:
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
            energy_4d[(*out_slice, slice(None))] = chunk_data.extra["energy_4d"][
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
        energy_3d[out_slice] = chunk_data.energy[inner_slice]
        scale_indices[out_slice] = chunk_data.scale_indices[inner_slice]

    return _energy_result_payload(config, image.shape, energy_3d, scale_indices)


def get_chunking_lattice_v190(
    strel_size_in_pixels: np.ndarray,
    max_voxels_per_node: int | float,
    size_of_image: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Replicate MATLAB get_chunking_lattice_V190."""
    target_voxel_per_chunk = float(max_voxels_per_node)
    target_chunk_characteristic_length = target_voxel_per_chunk ** (1.0 / 3.0)

    unit_volume_voxel_aspect_ratio = strel_size_in_pixels / (
        np.prod(strel_size_in_pixels) ** (1.0 / 3.0)
    )
    target_chunk_dimensions = target_chunk_characteristic_length * unit_volume_voxel_aspect_ratio

    chunk_lattice_dimensions = _matlab_uint16_cast(
        np.maximum(size_of_image / target_chunk_dimensions, 1.0)
    )

    number_of_chunks_in_lattice = int(np.prod(chunk_lattice_dimensions))
    return chunk_lattice_dimensions, number_of_chunks_in_lattice


def _matlab_uint16_cast(values: np.ndarray) -> np.ndarray:
    """Match MATLAB's positive double -> uint16 conversion."""
    rounded = np.floor(np.asarray(values, dtype=float) + 0.5)
    return np.clip(rounded, 0, np.iinfo(np.uint16).max).astype(np.uint16)


def get_starts_and_counts_v200(
    chunk_lattice_dimensions: np.ndarray,
    chunk_overlap_in_pixels: np.ndarray,
    size_of_image: np.ndarray,
    resolution_factors: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Replicate MATLAB get_starts_and_counts_V200 with unsigned 16-bit saturation arithmetic."""

    def get_borders(size, lat):
        pts = np.linspace(0, float(size), int(lat) + 1)
        return _matlab_uint16_cast(pts)

    y_writing_borders = get_borders(size_of_image[0], chunk_lattice_dimensions[0])
    x_writing_borders = get_borders(size_of_image[1], chunk_lattice_dimensions[1])
    z_writing_borders = get_borders(size_of_image[2], chunk_lattice_dimensions[2])

    def sat_sub(a, b):
        return np.clip(a.astype(np.int32) - b.astype(np.int32), 0, 65535).astype(np.uint16)

    def sat_add(a, b):
        return np.clip(a.astype(np.int32) + b.astype(np.int32), 0, 65535).astype(np.uint16)

    y_reading_starts = sat_sub(y_writing_borders[:-1], chunk_overlap_in_pixels[0]) + 1
    x_reading_starts = sat_sub(x_writing_borders[:-1], chunk_overlap_in_pixels[1]) + 1
    z_reading_starts = sat_sub(z_writing_borders[:-1], chunk_overlap_in_pixels[2]) + 1

    y_reading_ends = np.minimum(
        sat_add(y_writing_borders[1:], chunk_overlap_in_pixels[0]), int(size_of_image[0])
    )
    x_reading_ends = np.minimum(
        sat_add(x_writing_borders[1:], chunk_overlap_in_pixels[1]), int(size_of_image[1])
    )
    z_reading_ends = np.minimum(
        sat_add(z_writing_borders[1:], chunk_overlap_in_pixels[2]), int(size_of_image[2])
    )

    y_reading_counts = sat_sub(y_reading_ends, y_reading_starts - 1)
    x_reading_counts = sat_sub(x_reading_ends, x_reading_starts - 1)
    z_reading_counts = sat_sub(z_reading_ends, z_reading_starts - 1)

    def adjust_last_count(counts, rf):
        counts[-1] = 1 + int(rf) * int(np.floor((float(counts[-1]) - 1.0) / float(rf)))
        return counts

    y_reading_counts = adjust_last_count(y_reading_counts, resolution_factors[0])
    x_reading_counts = adjust_last_count(x_reading_counts, resolution_factors[1])
    z_reading_counts = adjust_last_count(z_reading_counts, resolution_factors[2])

    y_reading_starts[-1] = sat_sub(y_reading_ends[-1], y_reading_counts[-1] - 1)
    x_reading_starts[-1] = sat_sub(x_reading_ends[-1], x_reading_counts[-1] - 1)
    z_reading_starts[-1] = sat_sub(z_reading_ends[-1], z_reading_counts[-1] - 1)

    y_writing_starts = y_writing_borders[:-1].astype(np.int32) + 1
    x_writing_starts = x_writing_borders[:-1].astype(np.int32) + 1
    z_writing_starts = z_writing_borders[:-1].astype(np.int32) + 1

    y_writing_ends = y_writing_borders[1:].astype(np.int32)
    x_writing_ends = x_writing_borders[1:].astype(np.int32)
    z_writing_ends = z_writing_borders[1:].astype(np.int32)

    y_writing_counts = y_writing_ends - y_writing_starts + 1
    x_writing_counts = x_writing_ends - x_writing_starts + 1
    z_writing_counts = z_writing_ends - z_writing_starts + 1

    y_offsets = sat_sub(y_writing_starts, y_reading_starts)
    x_offsets = sat_sub(x_writing_starts, x_reading_starts)
    z_offsets = sat_sub(z_writing_starts, z_reading_starts)

    return (
        y_reading_starts.astype(float),
        x_reading_starts.astype(float),
        z_reading_starts.astype(float),
        y_reading_counts.astype(float),
        x_reading_counts.astype(float),
        z_reading_counts.astype(float),
        y_writing_starts.astype(float),
        x_writing_starts.astype(float),
        z_writing_starts.astype(float),
        y_writing_counts.astype(float),
        x_writing_counts.astype(float),
        z_writing_counts.astype(float),
        y_offsets.astype(float),
        x_offsets.astype(float),
        z_offsets.astype(float),
    )


def _interp3_matlab_linear_inf(
    volume: np.ndarray,
    coords: np.ndarray,
    *,
    cval: float = 0.0,
) -> np.ndarray:
    """Linear interp3-compatible interpolation that propagates positive-weight Inf."""
    y = coords[0]
    x = coords[1]
    z = coords[2]
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    z0 = np.floor(z).astype(np.int64)
    fy = y - y0
    fx = x - x0
    fz = z - z0

    out = np.zeros_like(y, dtype=np.float64)
    has_inf = np.zeros_like(y, dtype=bool)
    shape_y, shape_x, shape_z = volume.shape

    for dy in (0, 1):
        iy = y0 + dy
        wy = fy if dy else 1.0 - fy
        valid_y = (iy >= 0) & (iy < shape_y)
        for dx in (0, 1):
            ix = x0 + dx
            wx = fx if dx else 1.0 - fx
            valid_xy = valid_y & (ix >= 0) & (ix < shape_x)
            for dz in (0, 1):
                iz = z0 + dz
                wz = fz if dz else 1.0 - fz
                weight = wy * wx * wz
                positive_weight = weight > 0.0
                valid = valid_xy & (iz >= 0) & (iz < shape_z)

                values = np.full(out.shape, cval, dtype=np.float64)
                valid_positive = valid & positive_weight
                if np.any(valid_positive):
                    values[valid_positive] = volume[
                        iy[valid_positive],
                        ix[valid_positive],
                        iz[valid_positive],
                    ]
                    has_inf |= valid_positive & np.isposinf(values)
                    finite = valid_positive & np.isfinite(values)
                    out[finite] += weight[finite] * values[finite]

                invalid_positive = (~valid) & positive_weight
                if cval and np.any(invalid_positive):
                    out[invalid_positive] += weight[invalid_positive] * cval

    out[has_inf] = np.inf
    return out


def _matlab_zero_based_linspace(offset: int, stride: int, count: int) -> np.ndarray:
    """Return MATLAB ``linspace(1+offset/rf, ..., count)`` in zero-based coordinates."""
    if count <= 0:
        return np.empty(0, dtype=np.float64)
    x1 = 1.0 + float(offset % stride) / float(stride)
    x2 = x1 + float(count - 1) / float(stride)
    if count == 1:
        return np.array([x2 - 1.0], dtype=np.float64)
    i = np.arange(count, dtype=np.float64)
    y = ((count - 1 - i) * x1 + i * x2) / (count - 1) - 1.0
    return y


def _compute_exact_parity_energy_chunked(
    image: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Compute energy per scale using MATLAB-exact octave-chunked downsample + offset-mesh upsampling."""

    # Keep thread parallelism; process workers trigger joblib memmap tracker failures
    # on Windows during exact crop proof runs. Operators can tune n_jobs per run.
    n_jobs = max(1, int(config.get("n_jobs", 2)))
    image_shape = np.asarray(image.shape, dtype=float)

    energy_3d = np.zeros(image.shape, dtype=np.float64)
    scale_indices = np.full(image.shape, -1, dtype=np.int16)

    octave_at_scales = config["octave_at_scales"]
    octave_range = np.unique(octave_at_scales)

    microns_per_voxel = np.asarray(config["microns_per_voxel"], dtype=float)
    pixels_per_sigma_PSF = np.asarray(config["pixels_per_sigma_PSF"], dtype=float)
    lumen_radius_microns = np.asarray(config["lumen_radius_microns"], dtype=float)

    for current_octave in octave_range:
        scale_indices_at_octave = np.where(octave_at_scales == current_octave)[0]
        if len(scale_indices_at_octave) == 0:
            continue

        rf = np.asarray(config["scale_resolution_factors"][scale_indices_at_octave[0]], dtype=float)
        # get_starts_and_counts_V200 uses MATLAB axis order (Y, X, Z); working image is (Z, Y, X).
        matlab_image_shape = np.array(
            [image_shape[1], image_shape[2], image_shape[0]],
            dtype=float,
        )
        rf_matlab = np.array([rf[1], rf[2], rf[0]], dtype=float)

        largest_scale_idx = scale_indices_at_octave[-1]
        largest_radius_microns = lumen_radius_microns[largest_scale_idx]
        largest_pixels_per_radius = largest_radius_microns / microns_per_voxel

        approx_size = np.round(matlab_image_shape / rf_matlab)
        microns_per_pixel = microns_per_voxel * rf
        microns_per_pixel_matlab = np.array(
            [microns_per_pixel[1], microns_per_pixel[2], microns_per_pixel[0]],
            dtype=float,
        )
        voxel_aspect_ratio = 1.0 / microns_per_pixel_matlab

        chunk_lattice_dimensions, number_of_chunks = get_chunking_lattice_v190(
            voxel_aspect_ratio,
            float(config["max_voxels"]),
            approx_size,
        )

        chunk_overlap_vector = np.ceil(
            6.0 * np.sqrt(pixels_per_sigma_PSF**2 + largest_pixels_per_radius**2)
        ).astype(np.int32)
        chunk_overlap_matlab = chunk_overlap_vector[[1, 2, 0]]

        res_starts_counts = get_starts_and_counts_v200(
            chunk_lattice_dimensions,
            chunk_overlap_matlab,
            matlab_image_shape,
            rf_matlab,
        )

        y_read_starts = res_starts_counts[0]
        x_read_starts = res_starts_counts[1]
        z_read_starts = res_starts_counts[2]

        y_read_counts = res_starts_counts[3]
        x_read_counts = res_starts_counts[4]
        z_read_counts = res_starts_counts[5]

        y_write_starts = res_starts_counts[6]
        x_write_starts = res_starts_counts[7]
        z_write_starts = res_starts_counts[8]

        y_write_counts = res_starts_counts[9]
        x_write_counts = res_starts_counts[10]
        z_write_counts = res_starts_counts[11]

        y_offset = res_starts_counts[12]
        x_offset = res_starts_counts[13]
        z_offset = res_starts_counts[14]

        def _process_chunk(
            chunk_idx: int,
        ) -> tuple[int, tuple[slice, slice, slice, np.ndarray, np.ndarray]]:
            # Fortran unraveling matching MATLAB ind2sub on (Y, X, Z) lattice
            y_idx, x_idx, z_idx = np.unravel_index(chunk_idx, chunk_lattice_dimensions, order="F")

            py_z_start = int(z_read_starts[z_idx]) - 1
            py_y_start = int(y_read_starts[y_idx]) - 1
            py_x_start = int(x_read_starts[x_idx]) - 1

            py_z_count = int(z_read_counts[z_idx])
            py_y_count = int(y_read_counts[y_idx])
            py_x_count = int(x_read_counts[x_idx])

            stride_z = int(rf[0])
            stride_y = int(rf[1])
            stride_x = int(rf[2])

            original_chunk_zyx = image[
                py_z_start : py_z_start + py_z_count : stride_z,
                py_y_start : py_y_start + py_y_count : stride_y,
                py_x_start : py_x_start + py_x_count : stride_x,
            ]
            original_chunk = np.ascontiguousarray(original_chunk_zyx.transpose(1, 2, 0))

            padded_chunk = native_hessian._fourier_transform_input(original_chunk)
            chunk_dft = np.fft.fftn(padded_chunk.astype(np.float64, copy=False))
            pixel_freq_meshes = native_hessian._pixel_frequency_meshes(padded_chunk.shape)

            w_count_z = int(z_write_counts[z_idx])
            w_count_y = int(y_write_counts[y_idx])
            w_count_x = int(x_write_counts[x_idx])

            off_z = int(z_offset[z_idx])
            off_y = int(y_offset[y_idx])
            off_x = int(x_offset[x_idx])

            z_local = slice(
                int(np.floor(off_z / stride_z)),
                1 + int(np.ceil((off_z + w_count_z - 1) / stride_z)),
            )
            y_local = slice(
                int(np.floor(off_y / stride_y)),
                1 + int(np.ceil((off_y + w_count_y - 1) / stride_y)),
            )
            x_local = slice(
                int(np.floor(off_x / stride_x)),
                1 + int(np.ceil((off_x + w_count_x - 1) / stride_x)),
            )

            mesh_z = _matlab_zero_based_linspace(off_z, stride_z, w_count_z)
            mesh_y = _matlab_zero_based_linspace(off_y, stride_y, w_count_y)
            mesh_x = _matlab_zero_based_linspace(off_x, stride_x, w_count_x)

            mesh_coords = np.meshgrid(mesh_y, mesh_x, mesh_z, indexing="ij")
            coords_grid = np.stack(mesh_coords, axis=0)

            chunk_energy_4d = np.zeros(
                (w_count_y, w_count_x, w_count_z, len(scale_indices_at_octave)),
                dtype=np.float64,
            )

            for s_sub_idx, s_idx in enumerate(scale_indices_at_octave):
                radius_of_lumen_in_microns = lumen_radius_microns[s_idx]
                pixels_per_sigma_psf_at_oct = pixels_per_sigma_PSF / rf

                matching_kernel_dft, derivative_weights = native_hessian._matching_kernel_dft(
                    pixel_freq_meshes,
                    radius_of_lumen_in_microns=radius_of_lumen_in_microns,
                    microns_per_pixel=microns_per_pixel_matlab,
                    pixels_per_sigma_psf=pixels_per_sigma_psf_at_oct[[1, 2, 0]],
                    gaussian_to_ideal_ratio=float(config["gaussian_to_ideal_ratio"]),
                    spherical_to_annular_ratio=float(config["spherical_to_annular_ratio"]),
                )

                curvatures_kernels_dft, gradient_kernels_dft = (
                    native_hessian._derivative_kernels_dft(
                        pixel_freq_meshes,
                        derivative_weights,
                    )
                )

                filtered_chunk_dft = matching_kernel_dft * chunk_dft
                curvatures_chunk = np.empty(
                    (curvatures_kernels_dft.shape[0], *chunk_dft.shape),
                    dtype=np.float64,
                )
                for kernel_index, curvature_kernel_dft in enumerate(curvatures_kernels_dft):
                    curvatures_chunk[kernel_index] = native_hessian._ifftn_matlab_symmetric(
                        curvature_kernel_dft * filtered_chunk_dft,
                    )

                gradient_chunk = np.empty(
                    (gradient_kernels_dft.shape[0], *chunk_dft.shape),
                    dtype=np.float64,
                )
                for kernel_index, gradient_kernel_dft in enumerate(gradient_kernels_dft):
                    gradient_chunk[kernel_index] = native_hessian._ifftn_matlab_symmetric(
                        gradient_kernel_dft * filtered_chunk_dft,
                    )

                curvatures_chunk = curvatures_chunk[:, y_local, x_local, z_local]
                gradient_chunk = gradient_chunk[:, y_local, x_local, z_local]

                laplacian_chunk = curvatures_chunk[0] + curvatures_chunk[1] + curvatures_chunk[2]
                valid_voxels = laplacian_chunk < 0
                coarse_shape = curvatures_chunk.shape[1:4]
                coarse_energy = np.full(coarse_shape, np.inf, dtype=np.float64)

                if np.any(valid_voxels):
                    grad_valid = np.stack(
                        [
                            gradient_chunk[0][valid_voxels],
                            gradient_chunk[1][valid_voxels],
                            gradient_chunk[2][valid_voxels],
                        ],
                        axis=1,
                    )
                    hessian_valid = np.empty((grad_valid.shape[0], 3, 3), dtype=np.float64)
                    hessian_valid[:, 0, 0] = curvatures_chunk[0][valid_voxels]
                    hessian_valid[:, 0, 1] = curvatures_chunk[3][valid_voxels]
                    hessian_valid[:, 0, 2] = curvatures_chunk[5][valid_voxels]
                    hessian_valid[:, 1, 0] = curvatures_chunk[3][valid_voxels]
                    hessian_valid[:, 1, 1] = curvatures_chunk[1][valid_voxels]
                    hessian_valid[:, 1, 2] = curvatures_chunk[4][valid_voxels]
                    hessian_valid[:, 2, 0] = curvatures_chunk[5][valid_voxels]
                    hessian_valid[:, 2, 1] = curvatures_chunk[4][valid_voxels]
                    hessian_valid[:, 2, 2] = curvatures_chunk[2][valid_voxels]

                    principal_curvature_values, principal_curvature_vectors = np.linalg.eigh(
                        hessian_valid
                    )
                    principal_projections = np.einsum(
                        "ni,nij->nj",
                        grad_valid,
                        principal_curvature_vectors,
                    )
                    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                        principal_energy_values = principal_curvature_values * np.exp(
                            -((principal_projections / principal_curvature_values) ** 2) / 2.0
                        )
                    principal_energy_values[:, 2] = np.minimum(principal_energy_values[:, 2], 0.0)
                    energy_valid = np.sum(principal_energy_values, axis=1)
                    coarse_energy[valid_voxels] = energy_valid

                coarse_energy[~np.isfinite(coarse_energy)] = np.inf
                coarse_energy[coarse_energy >= 0] = np.inf

                upsampled = _interp3_matlab_linear_inf(coarse_energy, coords_grid)
                upsampled[(~np.isfinite(upsampled)) | (upsampled >= 0)] = 0.0
                chunk_energy_4d[..., s_sub_idx] = upsampled

            chunk_energy_min = np.min(chunk_energy_4d, axis=3).transpose(2, 0, 1)
            chunk_scale_min = np.argmin(chunk_energy_4d, axis=3).transpose(2, 0, 1)
            chunk_scale_min[chunk_energy_min >= 0.0] = -1

            prev_scales_count = int(np.sum(octave_at_scales < current_octave))
            valid_scale = chunk_scale_min >= 0
            chunk_scale_min = chunk_scale_min.astype(np.int16)
            chunk_scale_min[valid_scale] += prev_scales_count

            py_z_w_start = int(z_write_starts[z_idx]) - 1
            py_y_w_start = int(y_write_starts[y_idx]) - 1
            py_x_w_start = int(x_write_starts[x_idx]) - 1

            slice_z = slice(py_z_w_start, py_z_w_start + w_count_z)
            slice_y = slice(py_y_w_start, py_y_w_start + w_count_y)
            slice_x = slice(py_x_w_start, py_x_w_start + w_count_x)

            return chunk_idx, (slice_z, slice_y, slice_x, chunk_energy_min, chunk_scale_min)

        chunk_results = Parallel(n_jobs=n_jobs, prefer="threads", verbose=10)(
            delayed(_process_chunk)(c_idx) for c_idx in range(number_of_chunks)
        )

        for _, (slice_z, slice_y, slice_x, chunk_energy, chunk_scale) in chunk_results:
            master_energy = energy_3d[slice_z, slice_y, slice_x]
            is_better = chunk_energy < master_energy
            energy_3d[slice_z, slice_y, slice_x] = np.where(is_better, chunk_energy, master_energy)
            scale_indices[slice_z, slice_y, slice_x] = np.where(
                is_better, chunk_scale, scale_indices[slice_z, slice_y, slice_x]
            )

    energy_3d[energy_3d >= 0.0] = 0.0
    energy_3d[~np.isfinite(energy_3d)] = 0.0
    scale_indices[energy_3d >= 0.0] = -1
    return energy_3d.astype(np.float64, copy=False), scale_indices.astype(np.int16), None


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
    "get_chunking_lattice_v190",
    "get_starts_and_counts_v200",
    "_compute_exact_parity_energy_chunked",
]
