"""Post-choice MATLAB edge math helpers."""

from __future__ import annotations

from typing import Any, cast

import numpy as np


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
        space_trace_matrix = np.rint(np.asarray(space_trace, dtype=np.float32)).astype(
            np.int64,
            copy=False,
        )
        scale_trace_vector = np.rint(np.asarray(scale_trace, dtype=np.float32).reshape(-1)).astype(
            np.int64,
            copy=False,
        )
        scale_trace_vector = np.clip(scale_trace_vector, 0, max(len(lumen_radii) - 1, 0))
        radii_in_pixels = np.rint(
            lumen_radii[scale_trace_vector][:, None] / voxel_size,
        ).astype(np.int64, copy=False)

        subscript_maxs = space_trace_matrix[:, :3] + radii_in_pixels
        subscript_mins = space_trace_matrix[:, :3] - radii_in_pixels

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
) -> tuple[list[int], int]:
    """Use MATLAB's pre-clean smoothing + crop logic to filter edge indices."""
    from ..graph import _matlab_smooth_edges_v2

    if not candidate_indices:
        return [], 0

    edge_space_traces = [
        np.asarray(traces[index], dtype=np.float32).copy() for index in candidate_indices
    ]
    edge_scale_traces = [
        np.asarray(scale_traces[index], dtype=np.float32).reshape(-1).copy()
        for index in candidate_indices
    ]
    edge_energy_traces = [
        np.asarray(energy_traces[index], dtype=np.float32).reshape(-1).copy()
        for index in candidate_indices
    ]

    sigma_edge_smoothing = float(np.sqrt(2.0) / 2.0)
    presmoothed_space, presmoothed_scales_matlab, presmoothed_energy = _matlab_smooth_edges_v2(
        edge_space_traces,
        edge_scale_traces,
        edge_energy_traces,
        sigma_edge_smoothing,
        np.asarray(lumen_radius_microns, dtype=np.float32),
        np.asarray(microns_per_voxel, dtype=np.float32),
    )
    excluded_edges = _matlab_crop_edges_v200(
        presmoothed_space,
        [scale_trace - np.float32(1.0) for scale_trace in presmoothed_scales_matlab],
        presmoothed_energy,
        lumen_radius_microns=np.asarray(lumen_radius_microns, dtype=np.float32),
        microns_per_voxel=np.asarray(microns_per_voxel, dtype=np.float32),
        size_of_image=size_of_image,
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
        field_values = np.asarray(chosen_edges[field_name])
        if field_values.shape[0] != keep.size:
            continue
        chosen_edges[field_name] = field_values[keep]

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
    from ..graph import _matlab_edge_metrics, _matlab_smooth_edges_v2

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
    smoothed_space, smoothed_scales_matlab, smoothed_energy = _matlab_smooth_edges_v2(
        edge_space_traces,
        edge_scale_traces,
        edge_energy_traces,
        sigma_edge_smoothing,
        np.asarray(lumen_radius_microns, dtype=np.float32),
        np.asarray(microns_per_voxel, dtype=np.float32),
    )
    chosen_edges["traces"] = smoothed_space
    chosen_edges["scale_traces"] = [
        np.asarray(scale_trace - np.float32(1.0), dtype=np.float32)
        for scale_trace in smoothed_scales_matlab
    ]
    chosen_edges["energy_traces"] = smoothed_energy
    chosen_edges["energies"] = _matlab_edge_metrics(smoothed_energy)
    return normalize_edges_matlab_style(chosen_edges)
