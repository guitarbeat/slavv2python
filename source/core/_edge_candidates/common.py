"""Shared helpers for edge-candidate generation and parity flows."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
from skimage.graph import route_through_array
from typing_extensions import TypeAlias

from .._energy.provenance import is_exact_compatible_energy_origin
from .._vertices.payloads import matlab_linear_indices as _matlab_linear_indices

Int16Array: TypeAlias = "np.ndarray"
Int32Array: TypeAlias = "np.ndarray"
Int64Array: TypeAlias = "np.ndarray"
Float32Array: TypeAlias = "np.ndarray"
Float64Array: TypeAlias = "np.ndarray"
BoolArray: TypeAlias = "np.ndarray"


def normalize_candidate_connection_sources(
        raw_sources: Any,
        candidate_connection_count: int,
        *,
        default_source: str = "unknown",
) -> list[str]:
    """Return a normalized per-connection source label list."""
    if candidate_connection_count <= 0:
        return []

    if isinstance(raw_sources, np.ndarray):
        source_values = np.asarray(raw_sources).reshape(-1).tolist()
    elif isinstance(raw_sources, (list, tuple)):
        source_values = list(raw_sources)
    else:
        source_values = []

    allowed_sources = {"frontier", "watershed", "geodesic", "fallback", "unknown"}
    default_label = default_source if default_source in allowed_sources else "unknown"
    normalized: list[str] = []
    for index in range(candidate_connection_count):
        if index < len(source_values):
            source_label = str(source_values[index]).strip().lower()
            normalized.append(source_label if source_label in allowed_sources else default_label)
            continue
        normalized.append(default_label)
    return normalized


def _use_matlab_frontier_tracer(energy_data: dict[str, Any], params: dict[str, Any]) -> bool:
    """Enable the MATLAB-style frontier tracer for exact-compatible energy reruns."""
    if not bool(params.get("comparison_exact_network", False)):
        return False
    return is_exact_compatible_energy_origin(energy_data.get("energy_origin"))


def _matlab_frontier_edge_budget(params: dict[str, Any]) -> int:
    """Return MATLAB's effective per-origin frontier edge budget."""
    requested_edges = int(params.get("number_of_edges_per_vertex", 4))
    if bool(params.get("comparison_exact_network", False)):
        return 2
    return max(1, requested_edges)


def _matlab_frontier_offsets(
        strel_apothem: int,
        microns_per_voxel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct MATLAB-style cube-neighborhood offsets with Y-fastest ordering."""
    local_range = np.arange(-strel_apothem, strel_apothem + 1, dtype=np.int32)
    offsets = np.array(
        [[y, x, z] for z in local_range for x in local_range for y in local_range],
        dtype=np.int32,
    )
    distances = np.sqrt(np.sum((offsets.astype(np.float64) * microns_per_voxel) ** 2, axis=1))
    return offsets, distances.astype(np.float32, copy=False)


def _matlab_frontier_scale_offsets(
        scale_index: int,
        lumen_radius_microns: np.ndarray,
        microns_per_voxel: np.ndarray,
        *,
        step_size_per_origin_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct MATLAB's scale-dependent spherical strel plus 27-neighborhood box."""
    local_geometry = _build_matlab_local_strel_geometry(
        scale_index,
        lumen_radius_microns,
        microns_per_voxel,
        step_size_per_origin_radius=step_size_per_origin_radius,
    )
    return (
        np.asarray(local_geometry["local_subscripts"], dtype=np.int32),
        np.asarray(local_geometry["distance_lut"], dtype=np.float32),
    )


def _build_matlab_local_strel_geometry(
        scale_index: int,
        lumen_radius_microns: np.ndarray,
        microns_per_voxel: np.ndarray,
        *,
        step_size_per_origin_radius: float,
) -> dict[str, np.ndarray]:
    """Port MATLAB ``calculate_linear_strel_range`` local geometry for one scale."""
    radii_microns = float(
        np.asarray(lumen_radius_microns, dtype=np.float64).reshape(-1)[int(scale_index)]
    ) * float(step_size_per_origin_radius)
    radii_pixels = np.maximum(radii_microns / np.asarray(microns_per_voxel, dtype=np.float64), 1.0)
    rounded_radii = np.rint(radii_pixels).astype(np.int32, copy=False)
    offsets: list[list[int]] = []
    for z in range(-int(rounded_radii[2]), int(rounded_radii[2]) + 1):
        for x in range(-int(rounded_radii[1]), int(rounded_radii[1]) + 1):
            for y in range(-int(rounded_radii[0]), int(rounded_radii[0]) + 1):
                linf_distance = max(abs(y), abs(x), abs(z))
                radial_l2_distance_squared = (
                        (float(y) / float(radii_pixels[0])) ** 2
                        + (float(x) / float(radii_pixels[1])) ** 2
                        + (float(z) / float(radii_pixels[2])) ** 2
                )
                if radial_l2_distance_squared <= 1.0 or linf_distance <= 1:
                    offsets.append([y, x, z])
    offsets_array: Int32Array = np.asarray(offsets, dtype=np.int32)
    relative_distances = offsets_array.astype(np.float64, copy=False) * np.asarray(
        microns_per_voxel,
        dtype=np.float64,
    )
    distance_lut = np.sqrt(np.sum(relative_distances ** 2, axis=1))
    unit_vectors = np.zeros_like(relative_distances, dtype=np.float64)
    valid = distance_lut > 1e-12
    unit_vectors[valid] = relative_distances[valid] / distance_lut[valid, None]
    safe_radius = max(
        float(np.asarray(lumen_radius_microns, dtype=np.float64).reshape(-1)[int(scale_index)]),
        1e-6,
    )
    r_over_r_lut = distance_lut / safe_radius
    return {
        "local_subscripts": offsets_array,
        "distance_lut": distance_lut.astype(np.float32, copy=False),
        "unit_vectors": unit_vectors.astype(np.float32, copy=False),
        "r_over_R": r_over_r_lut.astype(np.float32, copy=False),
    }


def _build_matlab_global_watershed_lut(
        scale_index: int,
        *,
        size_of_image: tuple[int, int, int],
        lumen_radius_microns: np.ndarray,
        microns_per_voxel: np.ndarray,
        step_size_per_origin_radius: float,
) -> dict[str, np.ndarray]:
    """Build MATLAB watershed LUT fields for one scale exactly enough for parity checks."""
    local_geometry = _build_matlab_local_strel_geometry(
        scale_index,
        lumen_radius_microns,
        microns_per_voxel,
        step_size_per_origin_radius=step_size_per_origin_radius,
    )
    local_subscripts = np.asarray(local_geometry["local_subscripts"], dtype=np.int32)
    cum_prod_image_dims = np.cumprod(np.asarray(size_of_image, dtype=np.int64))
    linear_offsets = (
            local_subscripts[:, 0].astype(np.int64, copy=False)
            + local_subscripts[:, 1].astype(np.int64, copy=False) * int(cum_prod_image_dims[0])
            + local_subscripts[:, 2].astype(np.int64, copy=False) * int(cum_prod_image_dims[1])
    )
    return {
        "linear_offsets": linear_offsets.astype(np.int64, copy=False),
        "local_subscripts": local_subscripts,
        "distance_lut": np.asarray(local_geometry["distance_lut"], dtype=np.float32),
        "r_over_R": np.asarray(local_geometry["r_over_R"], dtype=np.float32),
        "unit_vectors": np.asarray(local_geometry["unit_vectors"], dtype=np.float32),
    }


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
        neighbor_distances_microns: np.ndarray,
        neighbor_scale_indices: np.ndarray | None,
        propagated_scale_index: int,
        current_distance_microns: float,
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
        size_index_differences = np.asarray(
            neighbor_scale_indices,
            dtype=np.float64,
        ) - float(propagated_scale_index)
        adjusted *= np.exp(-0.5 * np.square(size_index_differences / size_tolerance))

    safe_origin_radius = max(float(origin_radius_microns), 1e-6)
    local_r_over_R = np.asarray(neighbor_distances_microns, dtype=np.float64) / safe_origin_radius
    local_distance_adjustment = (
                                        1.0 - np.cos(np.pi * np.minimum(1.0, (4.0 / 3.0) * local_r_over_R))
                                ) / 2.0
    with np.errstate(invalid="ignore"):
        adjusted *= local_distance_adjustment

    current_d_over_r = float(current_distance_microns) / safe_origin_radius
    safe_distance_tolerance = max(float(distance_tolerance), 1e-6)
    total_distance_adjustment = math.exp(
        -0.5 * ((3.0 * current_d_over_r / safe_distance_tolerance) ** 2)
    )
    with np.errstate(invalid="ignore"):
        adjusted *= total_distance_adjustment

    if current_forward_unit is not None:
        forward = np.asarray(current_forward_unit, dtype=np.float64).reshape(3)
        forward_norm = float(np.linalg.norm(forward))
        if forward_norm > 1e-12:
            forward = forward / forward_norm
            neighbor_vectors = np.asarray(neighbor_offsets, dtype=np.float64) * microns_per_voxel
            neighbor_norms = np.linalg.norm(neighbor_vectors, axis=1)
            directional_alignment: np.ndarray = np.zeros(
                (len(neighbor_vectors),),
                dtype=np.float64,
            )
            valid = neighbor_norms > 1e-12
            directional_alignment[valid] = (
                    np.sum(neighbor_vectors[valid] * forward, axis=1) / neighbor_norms[valid]
            )
            directional_alignment[directional_alignment < 0.0] = 0.0
            with np.errstate(invalid="ignore"):
                adjusted *= directional_alignment

    adjusted[~np.isfinite(adjusted)] = np.inf
    result: Float32Array = adjusted.astype(np.float32, copy=False)
    return cast("np.ndarray", result)


def _matlab_frontier_directional_suppression_factors(
        neighbor_offsets: np.ndarray,
        *,
        selected_index: int,
        microns_per_voxel: np.ndarray,
) -> np.ndarray:
    """Return MATLAB's continuous same-direction suppression factors for a chosen seed."""
    neighbor_vectors = np.asarray(neighbor_offsets, dtype=np.float64) * microns_per_voxel
    norms = np.linalg.norm(neighbor_vectors, axis=1)
    unit_vectors = np.zeros_like(neighbor_vectors, dtype=np.float64)
    valid = norms > 1e-12
    unit_vectors[valid] = neighbor_vectors[valid] / norms[valid, None]
    cosine_to_selected = np.sum(unit_vectors * unit_vectors[int(selected_index)], axis=1)
    suppression = (1.0 - cosine_to_selected) / 2.0
    result: Float32Array = suppression.astype(np.float32, copy=False)
    return cast("np.ndarray", result)


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


def _coord_to_matlab_linear_index(coord: np.ndarray, shape: tuple[int, int, int]) -> int:
    """Convert a 0-based ``(y, x, z)`` coordinate into MATLAB linear order."""
    y, x, z = (int(value) for value in coord[:3])
    return int(y + x * shape[0] + z * shape[0] * shape[1])


def _matlab_linear_index_to_coord(index: int, shape: tuple[int, int, int]) -> np.ndarray:
    """Convert a 0-based MATLAB linear index into a ``(y, x, z)`` coordinate."""
    xy_plane = shape[0] * shape[1]
    z = index // xy_plane
    pos_xy = index - z * xy_plane
    x = pos_xy // shape[0]
    y = pos_xy - x * shape[0]
    coord: Int32Array = np.array([y, x, z], dtype=np.int32)
    return cast("np.ndarray", coord)


def _path_coords_from_linear_indices(
        path_linear: list[int],
        shape: tuple[int, int, int],
) -> np.ndarray:
    """Convert a linear-index path into origin-to-terminal spatial coordinates."""
    coords = [_matlab_linear_index_to_coord(index, shape) for index in reversed(path_linear)]
    coord_array: Float32Array = np.asarray(coords, dtype=np.float32)
    return cast("np.ndarray", coord_array)


def _path_max_energy_from_linear_indices(
        path_linear: list[int],
        energy: np.ndarray,
        shape: tuple[int, int, int],
) -> float:
    """Return the maximum sampled energy along a linear-index path."""
    if not path_linear:
        return float("-inf")
    samples = []
    for index in path_linear:
        coord = _matlab_linear_index_to_coord(index, shape)
        samples.append(float(energy[coord[0], coord[1], coord[2]]))
    return max(samples, default=float("-inf"))


def _candidate_endpoint_pair_set(connections: np.ndarray) -> set[tuple[int, int]]:
    """Return the orientation-independent terminal endpoint pairs in a candidate payload."""
    pairs: set[tuple[int, int]] = set()
    normalized = np.asarray(connections, dtype=np.int32).reshape(-1, 2)
    for start_vertex, end_vertex in normalized:
        if int(start_vertex) < 0 or int(end_vertex) < 0:
            continue
        u, v = int(start_vertex), int(end_vertex)
        pairs.add((u, v) if u < v else (v, u))
    return pairs


def _vertex_center_linear_lookup(
        vertex_positions: np.ndarray,
        image_shape: tuple[int, int, int],
) -> dict[int, int]:
    """Map rounded vertex centers to their vertex indices."""
    if len(vertex_positions) == 0:
        return {}
    coords = np.rint(np.asarray(vertex_positions, dtype=np.float32)).astype(np.int32, copy=False)
    max_coord: Int32Array = np.asarray(image_shape, dtype=np.int32) - 1
    coords = np.clip(coords, 0, max_coord)
    linear_indices = _matlab_linear_indices(coords, image_shape)
    return {
        int(linear_index): int(vertex_index)
        for vertex_index, linear_index in enumerate(linear_indices)
    }


def _trace_local_geodesic_between_vertices(
        energy: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
        energy_sign: float,
        *,
        box_margin_voxels: int,
) -> np.ndarray | None:
    """Trace a local geodesic path between two vertices inside a bounded subvolume."""
    image_shape = energy.shape
    max_coord: Int32Array = np.asarray(image_shape, dtype=np.int32) - 1
    start_coord = np.clip(
        np.rint(np.asarray(start, dtype=np.float32)[:3]).astype(np.int32, copy=False),
        0,
        max_coord,
    )
    end_coord = np.clip(
        np.rint(np.asarray(end, dtype=np.float32)[:3]).astype(np.int32, copy=False),
        0,
        max_coord,
    )
    if np.array_equal(start_coord, end_coord):
        return None

    delta = np.abs(end_coord - start_coord)
    dynamic_margin = int(max(box_margin_voxels, 0) + math.ceil(float(np.max(delta)) * 0.25))
    lower = np.maximum(np.minimum(start_coord, end_coord) - dynamic_margin, 0)
    upper = np.minimum(np.maximum(start_coord, end_coord) + dynamic_margin + 1, image_shape)
    patch = np.asarray(
        energy[
            lower[0]: upper[0],
            lower[1]: upper[1],
            lower[2]: upper[2],
        ],
        dtype=np.float64,
    )
    if patch.size == 0:
        return None

    if energy_sign < 0:
        baseline = float(np.nanmin(patch))
        cost = patch - baseline + 1e-3
    else:
        baseline = float(np.nanmax(patch))
        cost = baseline - patch + 1e-3
    if not np.all(np.isfinite(cost)):
        return None

    local_start = tuple((start_coord - lower).tolist())
    local_end = tuple((end_coord - lower).tolist())
    try:
        local_coords, _weight = route_through_array(
            cost,
            local_start,
            local_end,
            fully_connected=True,
            geometric=True,
        )
    except (ValueError, RuntimeError):
        return None
    if len(local_coords) <= 1:
        return None

    global_coords = np.asarray(local_coords, dtype=np.int32) + lower
    deduped = [global_coords[0]]
    for coord in global_coords[1:]:
        if not np.array_equal(coord, deduped[-1]):
            deduped.append(coord)
    if len(deduped) <= 1:
        return None
    trace_coords: Float32Array = np.asarray(deduped, dtype=np.float32)
    return cast("np.ndarray", trace_coords)


def _candidate_incident_pair_counts(connections: np.ndarray) -> dict[int, int]:
    """Count unique incident endpoint pairs for each vertex."""
    counts: dict[int, int] = {}
    for start_vertex, end_vertex in _candidate_endpoint_pair_set(connections):
        counts[int(start_vertex)] = counts.get(int(start_vertex), 0) + 1
        counts[int(end_vertex)] = counts.get(int(end_vertex), 0) + 1
    return counts
