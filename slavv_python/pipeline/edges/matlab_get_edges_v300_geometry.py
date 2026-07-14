"""MATLAB port: frontier geometry helpers for ``get_edges_V300.m`` (Tracing Discovery)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from slavv_python.pipeline.edges.matlab_calculate_linear_strel_range import (
    build_matlab_local_strel_geometry,
)
from slavv_python.pipeline.edges.matlab_indexing import _matlab_watershed_min_candidate_energies

if TYPE_CHECKING:
    from slavv_python.pipeline.edges.edge_types import Float32Array, Int32Array


def _matlab_frontier_edge_budget(params: dict[str, Any]) -> int:
    """Return MATLAB's effective per-origin frontier edge budget."""
    requested_edges = int(params.get("number_of_edges_per_vertex", 4))
    return max(1, requested_edges)


def _matlab_frontier_offsets(
    strel_apothem: int,
    microns_per_voxel: Float32Array,
) -> tuple[Int32Array, Float32Array]:
    """Construct MATLAB-style cube-neighborhood offsets with Y-fastest ordering."""
    local_range: np.ndarray = np.arange(-strel_apothem, strel_apothem + 1, dtype=np.int32)
    offsets = np.array(
        [[d0, d1, d2] for d0 in local_range for d1 in local_range for d2 in local_range],
        dtype=np.int32,
    )
    relative_distances = offsets.astype(np.float64) * microns_per_voxel
    distances = np.sqrt(np.sum(relative_distances**2, axis=1))
    return cast("Int32Array", offsets), cast(
        "Float32Array", distances.astype(np.float64, copy=False)
    )


def _matlab_frontier_scale_offsets(
    scale_index: int,
    lumen_radius_microns: Float32Array,
    microns_per_voxel: Float32Array,
    *,
    step_size_per_origin_radius: float,
) -> tuple[Int32Array, Float32Array]:
    """Construct MATLAB's scale-dependent spherical strel plus 27-neighborhood box."""
    local_geometry = build_matlab_local_strel_geometry(
        scale_index,
        lumen_radius_microns,
        microns_per_voxel,
        step_size_per_origin_radius=step_size_per_origin_radius,
    )
    return (
        np.asarray(local_geometry["local_subscripts"], dtype=np.int32),
        np.asarray(local_geometry["distance_lut"], dtype=np.float64),
    )


def _matlab_frontier_size_tolerance(
    lumen_radius_microns: np.ndarray,
    *,
    radius_tolerance: float = 0.5,
) -> float:
    """Return MATLAB's scale-index tolerance from the released radius-tolerance constant."""
    radii = np.asarray(lumen_radius_microns, dtype=np.float64).reshape(-1)
    if radii.size < 2:
        return float("inf")
    positive_radii = radii[np.isfinite(radii) & (radii > 0)]
    if positive_radii.size < 2:
        return float("inf")
    size_ratio_per_index = float(positive_radii[1] / positive_radii[0])
    if not math.isfinite(size_ratio_per_index) or size_ratio_per_index <= 1.0:
        return float("inf")
    return float(math.log(1.0 + float(radius_tolerance)) / math.log(size_ratio_per_index))


def _matlab_frontier_adjusted_neighbor_energies(
    raw_energies: np.ndarray,
    *,
    neighbor_offsets: np.ndarray,
    neighbor_unit_vectors: np.ndarray | None = None,
    neighbor_r_over_R: np.ndarray,
    neighbor_scale_indices: np.ndarray | None,
    propagated_scale_index: int,
    current_d_over_r: float,
    origin_radius_microns: float,
    current_forward_unit: np.ndarray | None,
    microns_per_voxel: np.ndarray,
    lumen_radius_microns: np.ndarray,
    radius_tolerance: float = 0.5,
    distance_tolerance: float = 3.0,
) -> np.ndarray:
    """Apply MATLAB-style size, distance, and direction penalties to neighborhood energies."""
    adjusted = np.asarray(raw_energies, dtype=np.float64).copy()

    size_tolerance = _matlab_frontier_size_tolerance(
        lumen_radius_microns,
        radius_tolerance=radius_tolerance,
    )
    if neighbor_scale_indices is not None and math.isfinite(size_tolerance) and size_tolerance > 0:
        size_index_differences: np.ndarray = np.asarray(
            neighbor_scale_indices,
            dtype=np.float64,
        ) - float(propagated_scale_index)
        adjusted *= np.exp(-0.5 * np.square(size_index_differences / size_tolerance))

    local_r_over_R = np.asarray(neighbor_r_over_R, dtype=np.float64)
    local_distance_adjustment = (
        1.0 - np.cos(np.pi * np.minimum(1.0, (4.0 / 3.0) * local_r_over_R))
    ) / 2.0
    with np.errstate(invalid="ignore"):
        adjusted *= local_distance_adjustment

    safe_distance_tolerance = max(float(distance_tolerance), 1e-6)
    total_distance_adjustment = math.exp(
        -0.5 * ((3.0 * float(current_d_over_r) / safe_distance_tolerance) ** 2)
    )
    with np.errstate(invalid="ignore"):
        adjusted *= total_distance_adjustment

    if current_forward_unit is not None:
        forward = np.asarray(current_forward_unit, dtype=np.float64).reshape(3)
        forward_norm = float(np.linalg.norm(forward))
        if forward_norm > 1e-12:
            if neighbor_unit_vectors is None:
                neighbor_vectors: np.ndarray = (
                    np.asarray(neighbor_offsets, dtype=np.float64) * microns_per_voxel
                )
                neighbor_norms = np.linalg.norm(neighbor_vectors, axis=1)
                directional_alignment = np.zeros(
                    (len(neighbor_vectors),),
                    dtype=np.float64,
                )
                valid = neighbor_norms > 1e-12
                directional_alignment[valid] = (
                    np.sum(neighbor_vectors[valid] * forward, axis=1) / neighbor_norms[valid]
                )
            else:
                directional_alignment = np.sum(
                    np.asarray(neighbor_unit_vectors, dtype=np.float64) * forward,
                    axis=1,
                )
            directional_alignment[directional_alignment < 0.0] = 0.0
            with np.errstate(invalid="ignore"):
                adjusted *= directional_alignment

    adjusted = _matlab_watershed_min_candidate_energies(adjusted)
    return cast("np.ndarray", adjusted.astype(np.float64, copy=False))


def _matlab_frontier_directional_suppression_factors(
    neighbor_offsets: np.ndarray,
    *,
    selected_index: int,
    microns_per_voxel: np.ndarray,
) -> np.ndarray:
    """Return MATLAB's continuous same-direction suppression factors for a chosen seed."""
    neighbor_vectors: np.ndarray = (
        np.asarray(neighbor_offsets, dtype=np.float64) * microns_per_voxel
    )
    norms = np.linalg.norm(neighbor_vectors, axis=1)
    unit_vectors = np.zeros_like(neighbor_vectors, dtype=np.float64)
    valid = norms > 1e-12
    unit_vectors[valid] = neighbor_vectors[valid] / norms[valid, None]
    cosine_to_selected = np.sum(unit_vectors * unit_vectors[int(selected_index)], axis=1)
    suppression = (1.0 - cosine_to_selected) / 2.0
    return cast("np.ndarray", suppression.astype(np.float64, copy=False))


def _matlab_frontier_select_seed_moves(
    adjusted_neighbor_energies: np.ndarray,
    *,
    neighbor_offsets: np.ndarray,
    microns_per_voxel: np.ndarray,
    current_is_source: bool,
    edge_budget: int,
    current_branch_order: int,
) -> list[tuple[int, int]]:
    """Choose MATLAB-style seed moves from one strel using directional suppression."""
    if len(adjusted_neighbor_energies) == 0:
        return []

    working_energies = np.asarray(adjusted_neighbor_energies, dtype=np.float64).copy()
    seed_count = int(edge_budget) if current_is_source else 1
    selected_moves: list[tuple[int, int]] = []

    for seed_idx in range(1, max(1, seed_count) + 1):
        selected_index = int(np.argmin(working_energies))
        selected_energy = float(working_energies[selected_index])
        if not math.isfinite(selected_energy) or selected_energy >= 0.0:
            break
        selected_moves.append((selected_index, int(current_branch_order + seed_idx - 1)))
        working_energies *= _matlab_frontier_directional_suppression_factors(
            neighbor_offsets,
            selected_index=selected_index,
            microns_per_voxel=microns_per_voxel,
        )

    return selected_moves


def _matlab_frontier_insert_available_location(
    available_entries: list[tuple[float, int]],
    *,
    linear_index: int,
    energy: float,
) -> None:
    """Insert one MATLAB frontier location into worst-to-best energy order."""
    insert_at = len(available_entries)
    for idx, (existing_energy, _existing_linear_index) in enumerate(available_entries):
        if existing_energy < energy:
            insert_at = idx
            break
    available_entries.insert(insert_at, (float(energy), int(linear_index)))


def _matlab_frontier_pop_best_available_location(
    available_entries: list[tuple[float, int]],
    available_map: dict[int, float],
) -> tuple[float, int] | None:
    """Pop the MATLAB frontier's best currently valid available location."""
    while available_entries:
        candidate_energy, candidate_linear = available_entries.pop()
        if available_map.get(candidate_linear) == candidate_energy:
            return float(candidate_energy), int(candidate_linear)
    return None
