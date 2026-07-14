"""Preferred internal name for edge finalization helpers."""

from __future__ import annotations

from typing import Any, cast

import numpy as np


def _edge_trace_microns_per_voxel(microns_per_voxel: np.ndarray) -> np.ndarray:
    """Return voxel spacing aligned to internal edge traces.

    Run parameters preserve MATLAB's spatial convention, [Y, X, Z], while the
    Python exact-route edge traces are stored in array order, [Z, Y, X].
    """
    voxel_size = np.asarray(microns_per_voxel, dtype=np.float32).reshape(-1)
    if voxel_size.size >= 3:
        return cast("np.ndarray", voxel_size[[2, 0, 1]])
    return cast("np.ndarray", voxel_size)


def _matlab_uint_cast_positive(values: np.ndarray) -> np.ndarray:
    """Mirror MATLAB unsigned integer casts for non-negative crop coordinates."""
    arr = np.asarray(values, dtype=np.float64)
    return cast("np.ndarray", np.floor(arr + 0.5).astype(np.int64, copy=False))


def _matlab_resample_vectors_linf(
    edge_subscripts: np.ndarray,
) -> np.ndarray:
    """Mirror the active default ``resample_vectors`` L-infinity interpolation."""
    values = np.asarray(edge_subscripts, dtype=np.float64)
    if values.size == 0 or values.shape[0] <= 1:
        return values.copy()

    cumulative_lengths = np.concatenate(
        (
            np.zeros((1,), dtype=np.float64),
            np.cumsum(np.max(np.abs(np.diff(values[:, :3], axis=0)), axis=1)),
        )
    )
    total_length = float(cumulative_lengths[-1])
    if total_length == 0.0:
        return values[:1].copy()

    sample_lengths = np.linspace(
        0.0,
        total_length,
        num=int(np.ceil(total_length)) + 1,
        dtype=np.float64,
    )
    return cast(
        "np.ndarray",
        np.column_stack(
            [
                np.interp(sample_lengths, cumulative_lengths, values[:, dimension])
                for dimension in range(values.shape[1])
            ]
        ),
    )


def _matlab_precrop_resample_from_maps(
    edge_space_traces: list[np.ndarray],
    edge_scale_traces: list[np.ndarray],
    edge_energy_traces: list[np.ndarray],
    *,
    energy_map: np.ndarray,
    scale_indices: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Apply MATLAB's post-watershed resample + map-resample stage.

    ``get_edges_by_watershed`` stores Python traces in zero-based [Z, Y, X].
    ``vectorize_V200`` converts watershed linear locations to one-based [Y, X, Z],
    runs ``resample_vectors``, casts those resampled coordinates with ``uint64``,
    and then samples the energy/size maps.  The returned space traces remain in
    MATLAB's one-based [Y, X, Z] convention for the immediate smoothing/crop
    stage.
    """
    if len(edge_space_traces) == 0:
        return [], [], []

    energy_yxz = np.transpose(np.asarray(energy_map, dtype=np.float64), (1, 2, 0))
    scale_yxz = np.transpose(np.asarray(scale_indices, dtype=np.float64), (1, 2, 0))
    image_shape_yxz = np.asarray(energy_yxz.shape, dtype=np.int64)

    resampled_spaces: list[np.ndarray] = []
    resampled_scales: list[np.ndarray] = []
    resampled_energies: list[np.ndarray] = []

    for space_trace, scale_trace, energy_trace in zip(
        edge_space_traces,
        edge_scale_traces,
        edge_energy_traces,
    ):
        space_zyx = np.asarray(space_trace, dtype=np.float64).reshape(-1, 3)
        scale_zero = np.asarray(scale_trace, dtype=np.float64).reshape(-1)
        energies = np.asarray(energy_trace, dtype=np.float64).reshape(-1)
        if space_zyx.shape[0] == 0:
            resampled_spaces.append(np.zeros((0, 3), dtype=np.float64))
            resampled_scales.append(np.zeros((0,), dtype=np.float64))
            resampled_energies.append(np.zeros((0,), dtype=np.float64))
            continue

        space_yxz_one_based = space_zyx[:, [1, 2, 0]] + 1.0
        edge_quantities = np.column_stack((space_yxz_one_based, scale_zero + 1.0, energies))
        resampled = _matlab_resample_vectors_linf(edge_quantities)

        lookup_subscripts_one_based = _matlab_uint_cast_positive(resampled[:, :3])
        lookup_subscripts = lookup_subscripts_one_based - 1
        for axis, axis_size in enumerate(image_shape_yxz.tolist()):
            lookup_subscripts[:, axis] = np.clip(
                lookup_subscripts[:, axis],
                0,
                int(axis_size) - 1,
            )

        y = lookup_subscripts[:, 0]
        x = lookup_subscripts[:, 1]
        z = lookup_subscripts[:, 2]
        resampled_spaces.append(lookup_subscripts_one_based.astype(np.float64, copy=False))
        resampled_scales.append(scale_yxz[y, x, z].astype(np.float64, copy=False))
        resampled_energies.append(energy_yxz[y, x, z].astype(np.float64, copy=False))

    return resampled_spaces, resampled_scales, resampled_energies


def _matlab_edge_endpoint_energy(edge_energy_trace: np.ndarray) -> float:
    """Return MATLAB's endpoint energy normalizer for one chosen edge."""
    trace = np.asarray(edge_energy_trace, dtype=np.float32).reshape(-1)
    if trace.size == 0:
        return float("nan")
    endpoint_product = np.float32(trace[-1] * trace[0])
    with np.errstate(invalid="ignore"):
        endpoint_magnitude = np.float32(endpoint_product ** np.float32(0.5))
    return float(-endpoint_magnitude)


def normalize_edges_matlab_style(chosen_edges: dict[str, Any]) -> dict[str, Any]:
    """Apply MATLAB's post-choice edge-energy normalization formulas."""
    energy_traces = [
        np.asarray(trace, dtype=np.float32).copy()
        for trace in chosen_edges.get("energy_traces", [])
    ]
    if not energy_traces:
        return chosen_edges

    raw_energies = np.asarray(
        chosen_edges.get("energies", np.zeros((0,), dtype=np.float32)),
        dtype=np.float32,
    ).copy()
    endpoint_energies = np.asarray(
        [_matlab_edge_endpoint_energy(trace) for trace in energy_traces],
        dtype=np.float32,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        normalized_energy_traces = [
            (-trace / endpoint_energies[index]).astype(np.float32, copy=False)
            for index, trace in enumerate(energy_traces)
        ]
        normalized_energies = (-raw_energies / endpoint_energies[: len(raw_energies)]).astype(
            np.float32, copy=False
        )
    normalized_energies[np.isnan(normalized_energies)] = -np.inf

    chosen_edges["raw_energies"] = raw_energies
    chosen_edges["raw_energy_traces"] = energy_traces
    chosen_edges["edge_endpoint_energies"] = endpoint_energies
    chosen_edges["energies"] = normalized_energies
    chosen_edges["energy_traces"] = normalized_energy_traces
    return cast("dict[str, Any]", chosen_edges)


def _matlab_crop_edges_v200(
    edge_space_subscripts: list[np.ndarray],
    edge_scale_subscripts: list[np.ndarray],
    edge_energies: list[np.ndarray],
    *,
    lumen_radius_microns: np.ndarray,
    microns_per_voxel: np.ndarray,
    size_of_image: tuple[int, int, int],
    one_based_coordinates: bool = False,
) -> np.ndarray:
    """Mirror MATLAB ``crop_edges_V200`` on chosen edge trajectories."""
    n_edges = len(edge_space_subscripts)
    if n_edges == 0:
        return np.zeros((0,), dtype=bool)

    lumen_radii = np.asarray(lumen_radius_microns, dtype=np.float32).reshape(-1)
    voxel_size = np.asarray(microns_per_voxel, dtype=np.float32).reshape(1, 3)
    image_shape = np.asarray(size_of_image, dtype=np.int64).reshape(1, 3)
    excluded: np.ndarray = np.zeros((n_edges,), dtype=bool)

    for edge_index, (space_trace, scale_trace, _energy_trace) in enumerate(
        zip(edge_space_subscripts, edge_scale_subscripts, edge_energies)
    ):
        if len(space_trace) == 0:
            continue
        space_trace_arr = np.asarray(space_trace, dtype=np.float64)
        scale_trace_arr = np.asarray(scale_trace, dtype=np.float64).reshape(-1)
        if not (np.all(np.isfinite(space_trace_arr)) and np.all(np.isfinite(scale_trace_arr))):
            excluded[edge_index] = True
            continue
        space_trace_matrix = _matlab_uint_cast_positive(space_trace_arr)
        if not one_based_coordinates:
            # Python's legacy internal traces are zero-based.  MATLAB crops
            # one-based subscripts, so this is the zero-based equivalent of
            # ``uint16(space + 1) - 1``.
            space_trace_matrix = _matlab_uint_cast_positive(space_trace_arr + 1.0) - 1

        # Scale traces are stored zero-based in Python; MATLAB casts the
        # corresponding one-based labels with ``uint8`` before radius lookup.
        scale_trace_vector = _matlab_uint_cast_positive(scale_trace_arr + 1.0) - 1
        scale_trace_vector = np.clip(scale_trace_vector, 0, max(len(lumen_radii) - 1, 0))
        radii_in_pixels = _matlab_uint_cast_positive(
            lumen_radii[scale_trace_vector][:, None] / voxel_size,
        )

        subscript_maxs = space_trace_matrix[:, :3] + radii_in_pixels
        subscript_mins = space_trace_matrix[:, :3] - radii_in_pixels

        if one_based_coordinates:
            excluded[edge_index] = bool(
                np.any(subscript_maxs > image_shape) or np.any(subscript_mins < 1)
            )
        else:
            excluded[edge_index] = bool(
                np.any(subscript_maxs >= image_shape) or np.any(subscript_mins < 0)
            )

    return cast("np.ndarray", excluded)


def prefilter_edge_indices_for_cleanup_matlab_style(
    candidate_indices: list[int],
    traces: list[np.ndarray],
    scale_traces: list[np.ndarray],
    energy_traces: list[np.ndarray],
    *,
    lumen_radius_microns: np.ndarray,
    microns_per_voxel: np.ndarray,
    size_of_image: tuple[int, int, int],
    energy_map: np.ndarray | None = None,
    scale_indices: np.ndarray | None = None,
) -> tuple[list[int], int]:
    """Use MATLAB's pre-clean smoothing + crop logic to filter edge indices."""

    from slavv_python.pipeline.network import _matlab_smooth_edges_v2

    if not candidate_indices:
        return [], 0

    edge_space_traces = [
        np.asarray(traces[index], dtype=np.float64).copy() for index in candidate_indices
    ]
    edge_scale_traces = [
        np.asarray(scale_traces[index], dtype=np.float64).reshape(-1).copy()
        for index in candidate_indices
    ]
    edge_energy_traces = [
        np.asarray(energy_traces[index], dtype=np.float64).reshape(-1).copy()
        for index in candidate_indices
    ]

    sigma_edge_smoothing = float(np.sqrt(2.0) / 2.0)
    use_matlab_resample = energy_map is not None and scale_indices is not None
    if use_matlab_resample:
        edge_space_traces, edge_scale_traces, edge_energy_traces = (
            _matlab_precrop_resample_from_maps(
                edge_space_traces,
                edge_scale_traces,
                edge_energy_traces,
                energy_map=np.asarray(energy_map),
                scale_indices=np.asarray(scale_indices),
            )
        )
        trace_microns_per_voxel = np.asarray(microns_per_voxel, dtype=np.float32).reshape(3)
        crop_shape = (
            int(size_of_image[1]),
            int(size_of_image[2]),
            int(size_of_image[0]),
        )
    else:
        trace_microns_per_voxel = _edge_trace_microns_per_voxel(microns_per_voxel)
        crop_shape = size_of_image

    presmoothed_space, presmoothed_scales_matlab, presmoothed_energy = _matlab_smooth_edges_v2(
        edge_space_traces,
        edge_scale_traces,
        edge_energy_traces,
        sigma_edge_smoothing,
        np.asarray(lumen_radius_microns, dtype=np.float32),
        trace_microns_per_voxel,
    )
    excluded_edges = _matlab_crop_edges_v200(
        presmoothed_space,
        [scale_trace - np.float32(1.0) for scale_trace in presmoothed_scales_matlab],
        presmoothed_energy,
        lumen_radius_microns=np.asarray(lumen_radius_microns, dtype=np.float32),
        microns_per_voxel=trace_microns_per_voxel,
        size_of_image=crop_shape,
        one_based_coordinates=use_matlab_resample,
    )
    kept_indices = [
        index for keep, index in zip((~excluded_edges).tolist(), candidate_indices) if keep
    ]
    return kept_indices, int(np.sum(excluded_edges))


def _apply_edge_mask(chosen_edges: dict[str, Any], keep_mask: np.ndarray) -> dict[str, Any]:
    """Filter edge-aligned chosen-edge fields with a boolean mask."""
    keep = np.asarray(keep_mask, dtype=bool).reshape(-1)
    if keep.size == 0:
        return chosen_edges

    list_fields = (
        "traces",
        "scale_traces",
        "energy_traces",
        "raw_energy_traces",
        "connection_sources",
    )
    for field_name in list_fields:
        if field_name in chosen_edges:
            field_values = list(chosen_edges[field_name])
            chosen_edges[field_name] = [
                field_values[index] for index, include in enumerate(keep.tolist()) if include
            ]

    array_fields = (
        "connections",
        "energies",
        "raw_energies",
        "edge_endpoint_energies",
        "chosen_candidate_indices",
    )
    for field_name in array_fields:
        if field_name not in chosen_edges:
            continue
        field_values_arr = np.asarray(chosen_edges[field_name])
        if field_values_arr.shape[0] != keep.size:
            continue
        chosen_edges[field_name] = field_values_arr[keep]

    diagnostics = chosen_edges.get("diagnostics")
    if isinstance(diagnostics, dict):
        diagnostics["chosen_edge_count"] = int(np.sum(keep))
    return cast("dict[str, Any]", chosen_edges)


def finalize_edges_matlab_style(
    chosen_edges: dict[str, Any],
    *,
    lumen_radius_microns: np.ndarray,
    microns_per_voxel: np.ndarray,
    size_of_image: tuple[int, int, int],
) -> dict[str, Any]:
    """Apply MATLAB's final post-clean edge smoothing and normalization sequence."""

    from slavv_python.pipeline.network import _matlab_edge_metrics, _matlab_smooth_edges_v2

    edge_space_traces = [
        np.asarray(trace, dtype=np.float32).copy() for trace in chosen_edges.get("traces", [])
    ]
    if not edge_space_traces:
        return chosen_edges

    edge_scale_traces = [
        np.asarray(trace, dtype=np.float32).reshape(-1).copy()
        for trace in chosen_edges.get("scale_traces", [])
    ]
    edge_energy_traces = [
        np.asarray(trace, dtype=np.float32).reshape(-1).copy()
        for trace in chosen_edges.get("energy_traces", [])
    ]

    sigma_edge_smoothing = float(np.sqrt(2.0) / 2.0)
    trace_microns_per_voxel = _edge_trace_microns_per_voxel(microns_per_voxel)
    smoothed_space, smoothed_scales_matlab, smoothed_energy = _matlab_smooth_edges_v2(
        edge_space_traces,
        edge_scale_traces,
        edge_energy_traces,
        sigma_edge_smoothing,
        np.asarray(lumen_radius_microns, dtype=np.float32),
        trace_microns_per_voxel,
    )
    chosen_edges["traces"] = smoothed_space
    chosen_edges["scale_traces"] = [
        np.asarray(scale_trace - np.float32(1.0), dtype=np.float32)
        for scale_trace in smoothed_scales_matlab
    ]
    chosen_edges["energy_traces"] = smoothed_energy
    chosen_edges["energies"] = _matlab_edge_metrics(smoothed_energy)
    return normalize_edges_matlab_style(chosen_edges)


__all__ = [
    "_matlab_crop_edges_v200",
    "_matlab_edge_endpoint_energy",
    "finalize_edges_matlab_style",
    "normalize_edges_matlab_style",
    "prefilter_edge_indices_for_cleanup_matlab_style",
]
