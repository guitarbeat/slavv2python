"""Edge candidate generation and parity helpers for SLAVV."""

from __future__ import annotations

import logging
import math
from heapq import heappop, heappush
from typing import Any, cast

import numpy as np
from scipy.spatial import cKDTree
from skimage.graph import route_through_array
from skimage.segmentation import watershed
from typing_extensions import TypeAlias

from .edge_primitives import (
    TraceMetadata,
    _clip_trace_indices,
    _edge_metric_from_energy_trace,
    _record_trace_diagnostics,
    _scalar_radius,
    _trace_energy_series,
    _trace_scale_series,
    estimate_vessel_directions,
    generate_edge_directions,
    trace_edge,
)
from .edge_selection import _empty_edge_diagnostics, _merge_edge_diagnostics
from .vertices import _matlab_linear_indices

logger = logging.getLogger(__name__)

Int16Array: TypeAlias = "np.ndarray"
Int32Array: TypeAlias = "np.ndarray"
Int64Array: TypeAlias = "np.ndarray"
Float32Array: TypeAlias = "np.ndarray"
Float64Array: TypeAlias = "np.ndarray"
BoolArray: TypeAlias = "np.ndarray"


def _use_matlab_frontier_tracer(energy_data: dict[str, Any], params: dict[str, Any]) -> bool:
    """Enable the parity-specific frontier tracer only for MATLAB-energy parity runs."""
    if not bool(params.get("comparison_exact_network", False)):
        return False
    return energy_data.get("energy_origin") == "matlab_batch_hdf5"


def _parity_watershed_candidate_mode(params: dict[str, Any]) -> str:
    """Return the MATLAB-parity watershed candidate strategy."""
    requested_mode = str(params.get("parity_watershed_candidate_mode", "all_contacts")).strip()
    normalized_mode = requested_mode.lower()
    allowed_modes = {"all_contacts", "remaining_origin_contacts", "legacy_supplement"}
    return normalized_mode if normalized_mode in allowed_modes else "all_contacts"


def _parity_watershed_metric_threshold_from_params(params: dict[str, Any]) -> float | None:
    """Return the optional parity-only watershed metric threshold."""
    threshold_raw = params.get("parity_watershed_metric_threshold")
    if threshold_raw in (None, ""):
        return None
    threshold_value = cast("Any", threshold_raw)
    return float(threshold_value)


def _parity_candidate_salvage_mode(
    params: dict[str, Any],
    candidate_mode: str,
) -> str:
    """Return the MATLAB-parity candidate salvage strategy."""
    requested_mode = str(params.get("parity_candidate_salvage_mode", "auto")).strip().lower()
    allowed_modes = {
        "auto",
        "none",
        "frontier_deficit_geodesic",
        "all_origins_geodesic",
    }
    normalized_mode = requested_mode if requested_mode in allowed_modes else "auto"
    if normalized_mode == "auto":
        return "none" if candidate_mode == "legacy_supplement" else "frontier_deficit_geodesic"
    return normalized_mode


def _matlab_frontier_edge_budget(params: dict[str, Any]) -> int:
    """Return MATLAB's per-origin frontier edge budget from get_edges_for_vertex.m."""
    requested_edges = int(params.get("number_of_edges_per_vertex", 4))
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
    return max(samples) if samples else float("-inf")


def _candidate_endpoint_pair_set(connections: np.ndarray) -> set[tuple[int, int]]:
    """Return the orientation-independent terminal endpoint pairs in a candidate payload."""
    pairs: set[tuple[int, int]] = set()
    connections = np.asarray(connections, dtype=np.int32).reshape(-1, 2)
    for start_vertex, end_vertex in connections:
        if int(start_vertex) < 0 or int(end_vertex) < 0:
            continue
        u, v = int(start_vertex), int(end_vertex)
        pair: tuple[int, int] = (u, v) if u < v else (v, u)
        pairs.add(pair)
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
            lower[0] : upper[0],
            lower[1] : upper[1],
            lower[2] : upper[2],
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


def _candidate_incident_pair_counts(
    connections: np.ndarray,
) -> dict[int, int]:
    """Count unique incident endpoint pairs for each vertex."""
    counts: dict[int, int] = {}
    for start_vertex, end_vertex in _candidate_endpoint_pair_set(connections):
        counts[int(start_vertex)] = counts.get(int(start_vertex), 0) + 1
        counts[int(end_vertex)] = counts.get(int(end_vertex), 0) + 1
    return counts


def _rasterize_trace_segment(
    start: np.ndarray,
    end: np.ndarray,
    image_shape: tuple[int, int, int],
) -> np.ndarray:
    """Rasterize a straight voxel segment between two points, preserving endpoints."""
    start_coord = np.rint(np.asarray(start, dtype=np.float32)[:3]).astype(np.int32, copy=False)
    end_coord = np.rint(np.asarray(end, dtype=np.float32)[:3]).astype(np.int32, copy=False)
    max_coord: Int32Array = np.asarray(image_shape, dtype=np.int32) - 1
    start_coord = np.clip(start_coord, 0, max_coord)
    end_coord = np.clip(end_coord, 0, max_coord)

    steps = int(np.max(np.abs(end_coord - start_coord)))
    if steps <= 0:
        single_coord: Float32Array = start_coord.reshape(1, 3).astype(np.float32, copy=False)
        return cast("np.ndarray", single_coord)

    coords = np.rint(np.linspace(start_coord, end_coord, num=steps + 1)).astype(np.int32)
    deduped = [coords[0]]
    for coord in coords[1:]:
        if not np.array_equal(coord, deduped[-1]):
            deduped.append(coord)
    segment_coords: Float32Array = np.asarray(deduped, dtype=np.float32)
    return cast("np.ndarray", segment_coords)


def _build_watershed_join_trace(
    start: np.ndarray,
    contact: np.ndarray,
    end: np.ndarray,
    image_shape: tuple[int, int, int],
) -> np.ndarray:
    """Construct a simple ordered trace that joins two vertices through a watershed contact."""
    start_half = _rasterize_trace_segment(start, contact, image_shape)
    end_half = _rasterize_trace_segment(contact, end, image_shape)
    if len(end_half) > 0 and len(start_half) > 0 and np.array_equal(start_half[-1], end_half[0]):
        end_half = end_half[1:]
    if len(end_half) == 0:
        return start_half
    joined_trace: Float32Array = np.vstack([start_half, end_half]).astype(np.float32, copy=False)
    return cast("np.ndarray", joined_trace)


def _best_watershed_contact_coords(
    labels: np.ndarray,
    energy: np.ndarray,
) -> dict[tuple[int, int], np.ndarray]:
    """Return the lowest-energy face contact voxel for each touching watershed pair."""
    best_contacts: dict[tuple[int, int], tuple[float, np.ndarray]] = {}
    shifts = (
        np.array((1, 0, 0), dtype=np.int32),
        np.array((0, 1, 0), dtype=np.int32),
        np.array((0, 0, 1), dtype=np.int32),
    )

    for shift in shifts:
        source_slices = tuple(slice(None, -int(delta)) if delta else slice(None) for delta in shift)
        target_slices = tuple(slice(int(delta), None) if delta else slice(None) for delta in shift)
        source_labels = labels[source_slices]
        target_labels = labels[target_slices]
        is_touching = (source_labels != target_labels) & (source_labels > 0) & (target_labels > 0)
        if not np.any(is_touching):
            continue

        source_coords = np.argwhere(is_touching).astype(np.int32, copy=False)
        target_coords = source_coords + shift
        source_pairs = source_labels[is_touching].astype(np.int32, copy=False) - 1
        target_pairs = target_labels[is_touching].astype(np.int32, copy=False) - 1
        pair_indices = np.stack([source_pairs, target_pairs], axis=1)
        pair_indices.sort(axis=1)

        source_energy = energy[source_slices][is_touching]
        target_energy = energy[target_slices][is_touching]
        prefer_target = target_energy < source_energy
        contact_coords = source_coords.copy()
        contact_coords[prefer_target] = target_coords[prefer_target]
        contact_energy = np.where(prefer_target, target_energy, source_energy).astype(
            np.float32,
            copy=False,
        )

        order = np.lexsort((contact_energy, pair_indices[:, 1], pair_indices[:, 0]))
        pair_indices = pair_indices[order]
        contact_coords = contact_coords[order]
        contact_energy = contact_energy[order]
        keep: np.ndarray = np.ones((len(pair_indices),), dtype=bool)
        keep[1:] = np.any(pair_indices[1:] != pair_indices[:-1], axis=1)

        for pair_array, coord, pair_energy in zip(
            pair_indices[keep],
            contact_coords[keep],
            contact_energy[keep],
        ):
            pair = (int(pair_array[0]), int(pair_array[1]))
            best = best_contacts.get(pair)
            if best is None or float(pair_energy) < best[0]:
                best_contacts[pair] = (float(pair_energy), coord.astype(np.int32, copy=False))

    return {pair: coord for pair, (_, coord) in best_contacts.items()}


def _supplement_matlab_frontier_candidates_with_watershed_joins(
    candidates: dict[str, Any],
    energy: np.ndarray,
    scale_indices: np.ndarray | None,
    vertex_positions: np.ndarray,
    energy_sign: float,
    max_edges_per_vertex: int = 4,
    enforce_frontier_reachability: bool = True,
    require_mutual_frontier_participation: bool = False,
    parity_watershed_metric_threshold: float | None = None,
) -> dict[str, Any]:
    """Add parity-only watershed contact candidates that the local frontier misses.

    Phase 2 gates (applied in order):
    1. Already-existing pair skip
    2. Short-trace rejection (trace length <= 1)
    3. Non-negative energy rejection (max energy along trace >= 0)
    4. Frontier reachability: at least one vertex in the pair must already
       have a frontier candidate to *any* vertex.
    5. Optional mutual frontier participation: both vertices in the pair
       must already participate in frontier candidates.
    6. Per-origin supplement cap: each seed origin can contribute at most
       ``max_edges_per_vertex`` supplement candidates.
    7. Optional parity-only metric threshold: reject watershed traces whose
       max sampled energy is weaker than the configured ceiling.
    """
    if len(vertex_positions) < 2:
        return candidates

    image_shape = energy.shape
    markers = np.zeros(image_shape, dtype=np.int32)
    idxs = np.floor(vertex_positions).astype(int)
    idxs = np.clip(idxs, 0, np.array(image_shape) - 1)
    markers[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = np.arange(1, len(vertex_positions) + 1)

    labels = watershed(-energy_sign * energy, markers)

    existing_pairs = _candidate_endpoint_pair_set(candidates.get("connections", np.zeros((0, 2))))
    contact_coords_by_pair = _best_watershed_contact_coords(labels, energy)
    endpoint_pair_degree_counts: dict[int, int] = {}
    for start_vertex, end_vertex in existing_pairs:
        endpoint_pair_degree_counts[int(start_vertex)] = (
            endpoint_pair_degree_counts.get(int(start_vertex), 0) + 1
        )
        endpoint_pair_degree_counts[int(end_vertex)] = (
            endpoint_pair_degree_counts.get(int(end_vertex), 0) + 1
        )

    # Phase 2: build frontier reachability set — vertices that participate in
    # at least one frontier candidate (before supplementation)
    frontier_vertices: set[int] = set()
    existing_connections = np.asarray(
        candidates.get("connections", np.zeros((0, 2))), dtype=np.int32
    ).reshape(-1, 2)
    for start_vertex, end_vertex in existing_connections:
        if int(start_vertex) >= 0:
            frontier_vertices.add(int(start_vertex))
        if int(end_vertex) >= 0:
            frontier_vertices.add(int(end_vertex))

    supplement_payload: dict[str, Any] = {
        "candidate_source": "watershed",
        "traces": [],
        "connections": [],
        "metrics": [],
        "energy_traces": [],
        "scale_traces": [],
        "origin_indices": [],
        "connection_sources": [],
        "diagnostics": {
            "watershed_join_supplement_count": 0,
            "watershed_per_origin_candidate_counts": {},
        },
    }

    # Diagnostic counters
    n_already_existing = 0
    n_short_trace = 0
    n_energy_rejected = 0
    n_reachability_rejected = 0
    n_mutual_frontier_rejected = 0
    n_cap_rejected = 0
    n_endpoint_degree_rejected = 0
    n_metric_threshold_rejected = 0
    n_accepted = 0
    n_total_watershed_pairs = len(contact_coords_by_pair)

    # Phase 2: per-origin supplement cap tracking
    origin_supplement_counts: dict[int, int] = {}

    for pair, contact_coord in sorted(contact_coords_by_pair.items()):
        if pair in existing_pairs:
            n_already_existing += 1
            continue

        # Phase 2 gate: frontier reachability — at least one endpoint must have
        # participated in a frontier candidate
        if (
            enforce_frontier_reachability
            and pair[0] not in frontier_vertices
            and pair[1] not in frontier_vertices
        ):
            n_reachability_rejected += 1
            continue

        if (
            enforce_frontier_reachability
            and require_mutual_frontier_participation
            and (pair[0] not in frontier_vertices or pair[1] not in frontier_vertices)
        ):
            n_mutual_frontier_rejected += 1
            continue

        if (
            endpoint_pair_degree_counts.get(pair[0], 0) >= max_edges_per_vertex
            or endpoint_pair_degree_counts.get(pair[1], 0) >= max_edges_per_vertex
        ):
            n_endpoint_degree_rejected += 1
            continue

        # Phase 2 gate: per-origin supplement cap
        seed_origin = pair[0]
        current_origin_count = origin_supplement_counts.get(seed_origin, 0)
        if current_origin_count >= max_edges_per_vertex:
            n_cap_rejected += 1
            continue

        trace = _build_watershed_join_trace(
            vertex_positions[pair[0]],
            contact_coord,
            vertex_positions[pair[1]],
            image_shape,
        )
        if len(trace) <= 1:
            n_short_trace += 1
            continue

        energy_trace = _trace_energy_series(trace, energy)
        energy_trace_array = np.asarray(energy_trace, dtype=np.float32)
        # Tighter parity gate: float comparison with nan handling.
        # MATLAB typically rejects traces that cross into non-negative energy
        # (background) when working with cost-based tracing.
        max_energy = float(np.nanmax(energy_trace_array))
        if energy_sign < 0:
            # For negative-is-foreground, a non-negative max energy means we hit
            # background.
            is_invalid = max_energy >= 0
        else:
            # For positive-is-foreground (less common in this repo's parity path),
            # we check the min instead, but this tracer is currently specialized
            # for the negative-sign imported MATLAB energy.
            min_energy = float(np.nanmin(energy_trace_array))
            is_invalid = min_energy <= 0

        if is_invalid:
            n_energy_rejected += 1
            continue

        if parity_watershed_metric_threshold is not None:
            if energy_sign < 0:
                fails_metric_threshold = max_energy > parity_watershed_metric_threshold
            else:
                fails_metric_threshold = min_energy < parity_watershed_metric_threshold
            if fails_metric_threshold:
                n_metric_threshold_rejected += 1
                continue

        scale_trace = _trace_scale_series(trace, scale_indices)
        supplement_payload["traces"].append(trace)
        supplement_payload["connections"].append([pair[0], pair[1]])
        supplement_payload["metrics"].append(_edge_metric_from_energy_trace(energy_trace))
        supplement_payload["energy_traces"].append(energy_trace)
        supplement_payload["scale_traces"].append(scale_trace)
        supplement_payload["origin_indices"].append(pair[0])
        supplement_payload["connection_sources"].append("watershed")
        supplement_payload["diagnostics"]["watershed_join_supplement_count"] += 1
        n_accepted += 1
        origin_supplement_counts[seed_origin] = current_origin_count + 1
        existing_pairs.add(pair)
        endpoint_pair_degree_counts[pair[0]] = endpoint_pair_degree_counts.get(pair[0], 0) + 1
        endpoint_pair_degree_counts[pair[1]] = endpoint_pair_degree_counts.get(pair[1], 0) + 1
        supplement_payload["diagnostics"]["watershed_per_origin_candidate_counts"][
            str(seed_origin)
        ] = int(origin_supplement_counts.get(seed_origin, 0))

    supplement_payload["diagnostics"]["watershed_total_pairs"] = n_total_watershed_pairs
    supplement_payload["diagnostics"]["watershed_already_existing"] = n_already_existing
    supplement_payload["diagnostics"]["watershed_short_trace_rejected"] = n_short_trace
    supplement_payload["diagnostics"]["watershed_energy_rejected"] = n_energy_rejected
    supplement_payload["diagnostics"]["watershed_reachability_rejected"] = n_reachability_rejected
    supplement_payload["diagnostics"]["watershed_mutual_frontier_rejected"] = (
        n_mutual_frontier_rejected
    )
    supplement_payload["diagnostics"]["watershed_cap_rejected"] = n_cap_rejected
    supplement_payload["diagnostics"]["watershed_endpoint_degree_rejected"] = (
        n_endpoint_degree_rejected
    )
    supplement_payload["diagnostics"]["watershed_metric_threshold_rejected"] = (
        n_metric_threshold_rejected
    )
    supplement_payload["diagnostics"]["watershed_accepted"] = n_accepted

    logger.info(
        "Watershed supplement: %d total pairs, %d already existing, "
        "%d reachability rejected, %d mutual-frontier rejected, "
        "%d endpoint-degree rejected, %d metric-threshold rejected, %d cap rejected, "
        "%d short-trace rejected, %d energy rejected, %d accepted",
        n_total_watershed_pairs,
        n_already_existing,
        n_reachability_rejected,
        n_mutual_frontier_rejected,
        n_endpoint_degree_rejected,
        n_metric_threshold_rejected,
        n_cap_rejected,
        n_short_trace,
        n_energy_rejected,
        n_accepted,
    )

    if supplement_payload["connections"]:
        _append_candidate_unit(candidates, supplement_payload)
    else:
        # Still merge the diagnostic counters even when no candidates were added
        _merge_edge_diagnostics(
            candidates.get("diagnostics", {}), supplement_payload["diagnostics"]
        )
    return candidates


def _augment_matlab_frontier_candidates_with_watershed_contacts(
    candidates: dict[str, Any],
    energy: np.ndarray,
    scale_indices: np.ndarray | None,
    vertex_positions: np.ndarray,
    energy_sign: float,
    *,
    max_edges_per_vertex: int = 4,
    candidate_mode: str = "all_contacts",
    parity_watershed_metric_threshold: float | None = None,
) -> dict[str, Any]:
    """Merge watershed-contact candidates into the MATLAB parity frontier payload.

    This candidate-stage path is intentionally more permissive than the legacy
    supplement. It feeds valid watershed-contact traces into the existing
    MATLAB-style chooser and lets the chooser resolve endpoint conflicts rather
    than pruning them up front.
    """
    if len(vertex_positions) < 2:
        return candidates

    image_shape = energy.shape
    markers = np.zeros(image_shape, dtype=np.int32)
    idxs = np.floor(vertex_positions).astype(int)
    idxs = np.clip(idxs, 0, np.array(image_shape) - 1)
    markers[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = np.arange(1, len(vertex_positions) + 1)

    labels = watershed(-energy_sign * energy, markers)
    existing_pairs = _candidate_endpoint_pair_set(candidates.get("connections", np.zeros((0, 2))))
    contact_coords_by_pair = _best_watershed_contact_coords(labels, energy)

    existing_connections = np.asarray(
        candidates.get("connections", np.zeros((0, 2), dtype=np.int32)),
        dtype=np.int32,
    ).reshape(-1, 2)
    connection_sources = _normalize_candidate_connection_sources(
        candidates.get("connection_sources"),
        len(existing_connections),
        default_source="frontier",
    )
    origin_indices = np.asarray(
        candidates.get("origin_indices", np.zeros((0,))), dtype=np.int32
    ).reshape(-1)
    frontier_origin_counts: dict[int, int] = {}
    for index, origin_index in enumerate(origin_indices):
        if index >= len(connection_sources) or connection_sources[index] != "frontier":
            continue
        frontier_origin_counts[int(origin_index)] = (
            frontier_origin_counts.get(int(origin_index), 0) + 1
        )

    candidate_rows: list[
        tuple[float, float, tuple[int, int], np.ndarray, np.ndarray, np.ndarray]
    ] = []
    n_already_existing = 0
    n_short_trace = 0
    n_energy_rejected = 0
    n_metric_threshold_rejected = 0
    n_total_watershed_pairs = len(contact_coords_by_pair)

    for pair, contact_coord in sorted(contact_coords_by_pair.items()):
        if pair in existing_pairs:
            n_already_existing += 1
            continue

        trace = _build_watershed_join_trace(
            vertex_positions[pair[0]],
            contact_coord,
            vertex_positions[pair[1]],
            image_shape,
        )
        if len(trace) <= 1:
            n_short_trace += 1
            continue

        energy_trace = _trace_energy_series(trace, energy)
        energy_trace_array = np.asarray(energy_trace, dtype=np.float32)
        max_energy = float(np.nanmax(energy_trace_array))
        if energy_sign < 0:
            is_invalid = max_energy >= 0
        else:
            min_energy = float(np.nanmin(energy_trace_array))
            is_invalid = min_energy <= 0
        if is_invalid:
            n_energy_rejected += 1
            continue

        if parity_watershed_metric_threshold is not None:
            if energy_sign < 0:
                fails_metric_threshold = max_energy > parity_watershed_metric_threshold
            else:
                min_energy = float(np.nanmin(energy_trace_array))
                fails_metric_threshold = min_energy < parity_watershed_metric_threshold
            if fails_metric_threshold:
                n_metric_threshold_rejected += 1
                continue

        scale_trace = _trace_scale_series(trace, scale_indices)
        metric = _edge_metric_from_energy_trace(energy_trace)
        endpoint_distance = float(
            np.linalg.norm(vertex_positions[pair[0]] - vertex_positions[pair[1]])
        )
        candidate_rows.append((metric, endpoint_distance, pair, trace, energy_trace, scale_trace))

    candidate_rows.sort(key=lambda row: (row[0], row[1]))

    supplement_payload: dict[str, Any] = {
        "candidate_source": "watershed",
        "traces": [],
        "connections": [],
        "metrics": [],
        "energy_traces": [],
        "scale_traces": [],
        "origin_indices": [],
        "connection_sources": [],
        "diagnostics": {
            "watershed_join_supplement_count": 0,
            "watershed_per_origin_candidate_counts": {},
            "watershed_total_pairs": n_total_watershed_pairs,
            "watershed_already_existing": n_already_existing,
            "watershed_short_trace_rejected": n_short_trace,
            "watershed_energy_rejected": n_energy_rejected,
            "watershed_metric_threshold_rejected": n_metric_threshold_rejected,
            "watershed_origin_budget_rejected": 0,
            "watershed_accepted": 0,
        },
    }
    origin_added_counts: dict[int, int] = {}
    for metric, _distance, pair, trace, energy_trace, scale_trace in candidate_rows:
        if candidate_mode == "remaining_origin_contacts":
            remaining_budget = max_edges_per_vertex - frontier_origin_counts.get(pair[0], 0)
            if remaining_budget <= 0 or origin_added_counts.get(pair[0], 0) >= remaining_budget:
                supplement_payload["diagnostics"]["watershed_origin_budget_rejected"] += 1
                continue

        supplement_payload["traces"].append(trace)
        supplement_payload["connections"].append([pair[0], pair[1]])
        supplement_payload["metrics"].append(metric)
        supplement_payload["energy_traces"].append(energy_trace)
        supplement_payload["scale_traces"].append(scale_trace)
        supplement_payload["origin_indices"].append(pair[0])
        supplement_payload["connection_sources"].append("watershed")
        supplement_payload["diagnostics"]["watershed_join_supplement_count"] += 1
        supplement_payload["diagnostics"]["watershed_accepted"] += 1
        origin_added_counts[pair[0]] = origin_added_counts.get(pair[0], 0) + 1
        supplement_payload["diagnostics"]["watershed_per_origin_candidate_counts"][str(pair[0])] = (
            origin_added_counts[pair[0]]
        )

    if supplement_payload["connections"]:
        _append_candidate_unit(candidates, supplement_payload)
    else:
        _merge_edge_diagnostics(
            candidates.get("diagnostics", {}), supplement_payload["diagnostics"]
        )
    return candidates


def _salvage_matlab_parity_candidates_with_local_geodesics(
    candidates: dict[str, Any],
    energy: np.ndarray,
    scale_indices: np.ndarray | None,
    vertex_positions: np.ndarray,
    energy_sign: float,
    microns_per_voxel: np.ndarray,
    params: dict[str, Any],
    *,
    salvage_mode: str,
    parity_metric_threshold: float | None,
) -> dict[str, Any]:
    """Recover parity candidates via bounded local geodesic searches."""
    if len(vertex_positions) < 2 or salvage_mode == "none":
        return candidates

    max_edges_per_vertex = int(params.get("number_of_edges_per_vertex", 4))
    k_nearest = max(1, int(params.get("parity_geodesic_salvage_k_nearest", 10)))
    box_margin_voxels = max(0, int(params.get("parity_geodesic_salvage_box_margin_voxels", 4)))
    max_path_ratio = float(params.get("parity_geodesic_salvage_max_path_ratio", 2.5))

    connections = np.asarray(
        candidates.get("connections", np.zeros((0, 2), dtype=np.int32)),
        dtype=np.int32,
    ).reshape(-1, 2)
    frontier_origin_counts: dict[int, int] = {}
    for origin_index, count in (
        candidates.get("diagnostics", {}).get("frontier_per_origin_candidate_counts", {}).items()
    ):
        try:
            frontier_origin_counts[int(origin_index)] = int(count)
        except (TypeError, ValueError):
            continue
    frontier_terminal_rejections: dict[int, int] = {}
    for origin_index, count in (
        candidates.get("diagnostics", {}).get("frontier_per_origin_terminal_rejections", {}).items()
    ):
        try:
            rejection_count = int(count)
        except (TypeError, ValueError):
            continue
        if rejection_count > 0:
            frontier_terminal_rejections[int(origin_index)] = rejection_count

    existing_pairs = _candidate_endpoint_pair_set(connections)
    incident_pair_counts = _candidate_incident_pair_counts(connections)
    vertex_positions_microns: Float32Array = np.asarray(
        vertex_positions, dtype=np.float32
    ) * np.asarray(
        microns_per_voxel,
        dtype=np.float32,
    )
    tree = cKDTree(vertex_positions_microns)
    query_k = min(len(vertex_positions), k_nearest + 1)
    if query_k <= 1:
        return candidates

    neighbor_distances, neighbor_indices = tree.query(vertex_positions_microns, k=query_k)
    neighbor_distances = np.asarray(neighbor_distances, dtype=np.float32)
    neighbor_indices = np.asarray(neighbor_indices, dtype=np.int32)
    if neighbor_indices.ndim == 1:
        neighbor_indices = neighbor_indices[:, np.newaxis]
        neighbor_distances = neighbor_distances[:, np.newaxis]

    vertex_linear_lookup = _vertex_center_linear_lookup(vertex_positions, energy.shape)
    accepted_rows: list[tuple[int, float, float, int, np.ndarray, np.ndarray, np.ndarray]] = []
    accepted_pairs: set[tuple[int, int]] = set()
    origin_added_counts: dict[int, int] = {}
    diagnostics: TraceMetadata = {
        "geodesic_join_supplement_count": 0,
        "geodesic_total_attempted_pairs": 0,
        "geodesic_existing_pair_skipped": 0,
        "geodesic_route_failed": 0,
        "geodesic_short_trace_rejected": 0,
        "geodesic_energy_rejected": 0,
        "geodesic_metric_threshold_rejected": 0,
        "geodesic_path_ratio_rejected": 0,
        "geodesic_vertex_crossing_rejected": 0,
        "geodesic_origin_budget_rejected": 0,
        "geodesic_endpoint_degree_rejected": 0,
        "geodesic_shared_neighborhood_endpoint_relaxed": 0,
        "geodesic_accepted": 0,
        "geodesic_per_origin_candidate_counts": {},
    }

    for origin_index in range(len(vertex_positions)):
        frontier_count = frontier_origin_counts.get(origin_index, 0)
        rejected_terminal_hits = frontier_terminal_rejections.get(origin_index, 0)
        shared_origin_overflow_enabled = (
            salvage_mode == "frontier_deficit_geodesic"
            and frontier_count >= max_edges_per_vertex
            and rejected_terminal_hits > 0
        )
        if salvage_mode == "frontier_deficit_geodesic" and (
            frontier_count >= max_edges_per_vertex and not shared_origin_overflow_enabled
        ):
            diagnostics["geodesic_origin_budget_rejected"] += 1
            continue

        if salvage_mode == "frontier_deficit_geodesic":
            if shared_origin_overflow_enabled:
                max_new_pairs = min(2, rejected_terminal_hits)
            else:
                max_new_pairs = max_edges_per_vertex - frontier_count
        else:
            max_new_pairs = max(1, min(max_edges_per_vertex, 2))
        if max_new_pairs <= 0:
            diagnostics["geodesic_origin_budget_rejected"] += 1
            continue

        origin_rows: list[tuple[float, float, int, np.ndarray, np.ndarray, np.ndarray]] = []
        for neighbor_distance, neighbor_index in zip(
            neighbor_distances[origin_index].tolist(),
            neighbor_indices[origin_index].tolist(),
        ):
            neighbor_index = int(neighbor_index)
            if neighbor_index < 0 or neighbor_index == origin_index:
                continue

            pair = (
                (origin_index, neighbor_index)
                if origin_index < neighbor_index
                else (neighbor_index, origin_index)
            )
            if pair in existing_pairs or pair in accepted_pairs:
                diagnostics["geodesic_existing_pair_skipped"] += 1
                continue
            relaxed_endpoint_cap = False
            blocked_by_endpoint_cap = False
            for endpoint in pair:
                if incident_pair_counts.get(endpoint, 0) < max_edges_per_vertex:
                    continue
                if endpoint == origin_index and shared_origin_overflow_enabled:
                    relaxed_endpoint_cap = True
                    continue
                blocked_by_endpoint_cap = True
                break
            if blocked_by_endpoint_cap:
                diagnostics["geodesic_endpoint_degree_rejected"] += 1
                continue
            if relaxed_endpoint_cap:
                diagnostics["geodesic_shared_neighborhood_endpoint_relaxed"] += 1

            diagnostics["geodesic_total_attempted_pairs"] += 1
            trace = _trace_local_geodesic_between_vertices(
                energy,
                vertex_positions[origin_index],
                vertex_positions[neighbor_index],
                energy_sign,
                box_margin_voxels=box_margin_voxels,
            )
            if trace is None:
                diagnostics["geodesic_route_failed"] += 1
                continue
            if len(trace) <= 1:
                diagnostics["geodesic_short_trace_rejected"] += 1
                continue

            if len(trace) > 2:
                trace_indices = _clip_trace_indices(trace[1:-1], energy.shape)
                trace_linear = _matlab_linear_indices(trace_indices, energy.shape)
                crossed_vertex = False
                for linear_index in trace_linear.tolist():
                    vertex_index = vertex_linear_lookup.get(int(linear_index))
                    if vertex_index is None:
                        continue
                    if vertex_index not in {origin_index, neighbor_index}:
                        crossed_vertex = True
                        break
                if crossed_vertex:
                    diagnostics["geodesic_vertex_crossing_rejected"] += 1
                    continue

            energy_trace = _trace_energy_series(trace, energy)
            energy_trace_array = np.asarray(energy_trace, dtype=np.float32)
            max_energy = float(np.nanmax(energy_trace_array))
            if energy_sign < 0:
                is_invalid = max_energy >= 0
            else:
                is_invalid = float(np.nanmin(energy_trace_array)) <= 0
            if is_invalid:
                diagnostics["geodesic_energy_rejected"] += 1
                continue

            if parity_metric_threshold is not None and max_energy > parity_metric_threshold:
                diagnostics["geodesic_metric_threshold_rejected"] += 1
                continue

            straight_distance = float(
                np.linalg.norm(
                    vertex_positions_microns[origin_index]
                    - vertex_positions_microns[neighbor_index]
                )
            )
            if len(trace) > 1:
                step_vectors = np.diff(np.asarray(trace, dtype=np.float32), axis=0)
                path_length = float(
                    np.linalg.norm(
                        step_vectors * np.asarray(microns_per_voxel, dtype=np.float32), axis=1
                    ).sum()
                )
            else:
                path_length = 0.0
            if straight_distance > 0 and path_length > straight_distance * max_path_ratio:
                diagnostics["geodesic_path_ratio_rejected"] += 1
                continue

            scale_trace = _trace_scale_series(trace, scale_indices)
            metric = _edge_metric_from_energy_trace(energy_trace)
            origin_rows.append(
                (
                    metric,
                    float(neighbor_distance),
                    neighbor_index,
                    trace,
                    energy_trace,
                    scale_trace,
                )
            )

        origin_rows.sort(key=lambda row: (row[0], row[1], row[2]))
        for metric, _distance, neighbor_index, trace, energy_trace, scale_trace in origin_rows:
            if origin_added_counts.get(origin_index, 0) >= max_new_pairs:
                break
            pair = (
                (origin_index, neighbor_index)
                if origin_index < neighbor_index
                else (neighbor_index, origin_index)
            )
            if pair in existing_pairs or pair in accepted_pairs:
                continue
            accepted_rows.append(
                (
                    origin_index,
                    metric,
                    float(_distance),
                    neighbor_index,
                    trace,
                    energy_trace,
                    scale_trace,
                )
            )
            accepted_pairs.add(pair)
            origin_added_counts[origin_index] = origin_added_counts.get(origin_index, 0) + 1
            incident_pair_counts[pair[0]] = incident_pair_counts.get(pair[0], 0) + 1
            incident_pair_counts[pair[1]] = incident_pair_counts.get(pair[1], 0) + 1
            diagnostics["geodesic_per_origin_candidate_counts"][str(origin_index)] = int(
                origin_added_counts[origin_index]
            )

    if not accepted_rows:
        _merge_edge_diagnostics(candidates.get("diagnostics", {}), diagnostics)
        return candidates

    supplement_payload: dict[str, Any] = {
        "candidate_source": "geodesic",
        "traces": [],
        "connections": [],
        "metrics": [],
        "energy_traces": [],
        "scale_traces": [],
        "origin_indices": [],
        "connection_sources": [],
        "diagnostics": diagnostics,
    }
    for (
        origin_index,
        metric,
        _distance,
        neighbor_index,
        trace,
        energy_trace,
        scale_trace,
    ) in accepted_rows:
        supplement_payload["traces"].append(trace)
        supplement_payload["connections"].append([origin_index, neighbor_index])
        supplement_payload["metrics"].append(metric)
        supplement_payload["energy_traces"].append(energy_trace)
        supplement_payload["scale_traces"].append(scale_trace)
        supplement_payload["origin_indices"].append(origin_index)
        supplement_payload["connection_sources"].append("geodesic")

    supplement_payload["diagnostics"]["geodesic_join_supplement_count"] = len(accepted_rows)
    supplement_payload["diagnostics"]["geodesic_accepted"] = len(accepted_rows)
    _append_candidate_unit(candidates, supplement_payload)
    return candidates


def _finalize_matlab_parity_candidates(
    candidates: dict[str, Any],
    energy: np.ndarray,
    scale_indices: np.ndarray | None,
    vertex_positions: np.ndarray,
    energy_sign: float,
    params: dict[str, Any],
    microns_per_voxel: np.ndarray | None = None,
) -> dict[str, Any]:
    """Finalize MATLAB-parity candidates with the configured watershed strategy."""
    candidate_mode = _parity_watershed_candidate_mode(params)
    watershed_metric_threshold = _parity_watershed_metric_threshold_from_params(params)

    if candidate_mode == "legacy_supplement":
        enforce_frontier_reachability_gate = bool(
            params.get("parity_frontier_reachability_gate", True)
        )
        require_mutual_frontier_participation = bool(
            params.get("parity_require_mutual_frontier_participation", True)
        )
        finalized = _supplement_matlab_frontier_candidates_with_watershed_joins(
            candidates,
            energy,
            scale_indices,
            vertex_positions,
            energy_sign,
            max_edges_per_vertex=int(params.get("number_of_edges_per_vertex", 4)),
            enforce_frontier_reachability=enforce_frontier_reachability_gate,
            require_mutual_frontier_participation=require_mutual_frontier_participation,
            parity_watershed_metric_threshold=watershed_metric_threshold,
        )
    else:
        finalized = _augment_matlab_frontier_candidates_with_watershed_contacts(
            candidates,
            energy,
            scale_indices,
            vertex_positions,
            energy_sign,
            max_edges_per_vertex=int(params.get("number_of_edges_per_vertex", 4)),
            candidate_mode=candidate_mode,
            parity_watershed_metric_threshold=watershed_metric_threshold,
        )

    salvage_mode = _parity_candidate_salvage_mode(params, candidate_mode)
    if salvage_mode == "none":
        return finalized

    microns_per_voxel_value = (
        np.asarray(microns_per_voxel, dtype=np.float32)
        if microns_per_voxel is not None
        else np.ones((3,), dtype=np.float32)
    )
    return _salvage_matlab_parity_candidates_with_local_geodesics(
        finalized,
        energy,
        scale_indices,
        vertex_positions,
        energy_sign,
        microns_per_voxel_value,
        params,
        salvage_mode=salvage_mode,
        parity_metric_threshold=watershed_metric_threshold,
    )


def _prune_frontier_indices_beyond_found_vertices(
    candidate_coords: np.ndarray,
    origin_position_microns: np.ndarray,
    displacement_vectors: list[np.ndarray],
    microns_per_voxel: np.ndarray,
) -> np.ndarray:
    """Remove frontier voxels that lie beyond an already-found terminal direction."""
    if len(candidate_coords) == 0 or not displacement_vectors:
        return candidate_coords

    vectors_from_origin = (
        candidate_coords.astype(np.float64) * microns_per_voxel - origin_position_microns
    )
    indices_beyond: np.ndarray = np.zeros((len(candidate_coords),), dtype=bool)
    for displacement in displacement_vectors:
        # MATLAB parity note: The threshold of 1.0 matches the normalized
        # dot-product gate in get_edges_for_vertex.m. 'displacement' has been
        # pre-normalized by ||d||^2 (not ||d||), so the dot product gives
        # (cos(theta) * ||v_from_origin||) / ||d|| which exceeds 1.0 when the
        # candidate voxel lies at least one ||d|| beyond the found vertex in
        # the same direction. This prevents the frontier from exploring beyond
        # already-resolved terminal vertices.
        indices_beyond |= np.sum(displacement * vectors_from_origin, axis=1) > 1.0
    kept_coords: Int32Array = candidate_coords[~indices_beyond]
    return cast("np.ndarray", kept_coords)


def _resolve_frontier_edge_connection(
    current_path_linear: list[int],
    terminal_vertex_idx: int,
    seed_origin_idx: int,
    edge_paths_linear: list[list[int]],
    edge_pairs: list[tuple[int, int]],
    pointer_index_map: dict[int, int],
    energy: np.ndarray,
    shape: tuple[int, int, int],
) -> tuple[int | None, int | None]:
    """Resolve MATLAB-style parent/child validity for a frontier-found terminal."""
    origin_idx, terminal_idx, _resolution_reason = _resolve_frontier_edge_connection_details(
        current_path_linear,
        terminal_vertex_idx,
        seed_origin_idx,
        edge_paths_linear,
        edge_pairs,
        pointer_index_map,
        energy,
        shape,
    )
    return origin_idx, terminal_idx


def _resolve_frontier_edge_connection_details(
    current_path_linear: list[int],
    terminal_vertex_idx: int,
    seed_origin_idx: int,
    edge_paths_linear: list[list[int]],
    edge_pairs: list[tuple[int, int]],
    pointer_index_map: dict[int, int],
    energy: np.ndarray,
    shape: tuple[int, int, int],
) -> tuple[int | None, int | None, str]:
    """Resolve MATLAB-style parent/child validity for a frontier-found terminal."""
    root_index = current_path_linear[-1]
    root_pointer = int(pointer_index_map.get(root_index, 0))
    parent_index = -root_pointer if root_pointer < 0 else 0

    if parent_index == 0:
        return seed_origin_idx, terminal_vertex_idx, "accepted_seed_origin"

    parent_path = edge_paths_linear[parent_index - 1]
    parent_pointers = {
        -int(pointer_index_map.get(index, 0))
        for index in parent_path
        if int(pointer_index_map.get(index, 0)) < 0
    }
    parent_pointers.discard(0)
    parent_pointers.discard(parent_index)
    if parent_pointers:
        return None, None, "rejected_parent_has_child"

    parent_terminal, parent_origin = edge_pairs[parent_index - 1]
    if parent_terminal < 0 or parent_origin < 0:
        return None, None, "rejected_parent_invalid"

    parent_energy = _path_max_energy_from_linear_indices(parent_path, energy, shape)
    child_energy = _path_max_energy_from_linear_indices(current_path_linear, energy, shape)
    # MATLAB parity note: This mirrors get_edges_for_vertex.m's "child is better
    # than parent" rejection. In MATLAB, when a child path has a better (lower,
    # more negative) energy than its parent, the child is considered to be
    # stealing the parent's best voxels, so the child is invalidated.
    # The strict <= comparison preserves MATLAB's exact behavior.
    if child_energy <= parent_energy:
        return None, None, "rejected_child_better_than_parent"

    if root_index not in parent_path:
        return None, None, "rejected_root_missing_from_parent"

    bifurcation_index = parent_path.index(root_index)
    parent_1 = parent_path[:bifurcation_index]
    parent_2 = parent_path[bifurcation_index + 1 :]
    half_candidates: list[tuple[int, float]] = [
        (
            parent_terminal,
            _path_max_energy_from_linear_indices(parent_1, energy, shape)
            if parent_1
            else float("-inf"),
        )
    ]
    if parent_2:
        half_candidates.append(
            (parent_origin, _path_max_energy_from_linear_indices(parent_2, energy, shape))
        )
    origin_vertex_idx = min(half_candidates, key=lambda item: item[1])[0]
    if origin_vertex_idx < 0:
        return None, None, "rejected_parent_origin_invalid"
    if origin_vertex_idx == parent_terminal:
        return origin_vertex_idx, terminal_vertex_idx, "accepted_parent_terminal_half"
    return origin_vertex_idx, terminal_vertex_idx, "accepted_parent_origin_half"


def _frontier_parent_child_outcome_from_reason(resolution_reason: str) -> str | None:
    """Map a frontier resolution reason onto parent/child lifecycle language."""
    if resolution_reason == "rejected_parent_has_child":
        return "parent_has_child"
    if resolution_reason == "rejected_child_better_than_parent":
        return "child_better_than_parent"
    if resolution_reason.startswith("accepted_parent_"):
        return "accepted_parent_child_resolution"
    return None


def _frontier_bifurcation_choice_from_reason(resolution_reason: str) -> str | None:
    """Map a frontier resolution reason onto the chosen parent half."""
    if resolution_reason == "accepted_parent_terminal_half":
        return "parent_terminal_half"
    if resolution_reason == "accepted_parent_origin_half":
        return "parent_origin_half"
    return None


def _build_frontier_lifecycle_event(
    *,
    seed_origin_idx: int,
    terminal_vertex_idx: int,
    origin_idx: int | None,
    terminal_idx: int | None,
    resolution_reason: str,
    terminal_hit_sequence: int,
    local_candidate_index: int | None = None,
) -> dict[str, Any]:
    """Create a serializable frontier lifecycle event entry."""
    emitted_endpoint_pair: list[int] | None = None
    if (
        origin_idx is not None
        and terminal_idx is not None
        and origin_idx >= 0
        and terminal_idx >= 0
    ):
        start_vertex, end_vertex = int(origin_idx), int(terminal_idx)
        emitted_endpoint_pair = (
            [start_vertex, end_vertex] if start_vertex < end_vertex else [end_vertex, start_vertex]
        )

    survived_candidate_manifest = emitted_endpoint_pair is not None
    return {
        "seed_origin_index": int(seed_origin_idx),
        "terminal_vertex_index": int(terminal_vertex_idx),
        "resolved_origin_index": None if origin_idx is None else int(origin_idx),
        "resolved_terminal_index": None if terminal_idx is None else int(terminal_idx),
        "emitted_endpoint_pair": emitted_endpoint_pair,
        "resolution_reason": str(resolution_reason),
        "rejection_reason": None if survived_candidate_manifest else str(resolution_reason),
        "parent_child_outcome": _frontier_parent_child_outcome_from_reason(str(resolution_reason)),
        "bifurcation_choice": _frontier_bifurcation_choice_from_reason(str(resolution_reason)),
        "survived_candidate_manifest": bool(survived_candidate_manifest),
        "origin_candidate_local_index": (
            None if local_candidate_index is None else int(local_candidate_index)
        ),
        "manifest_candidate_index": None,
        "chosen_final_edge": False,
        "terminal_hit_sequence": int(terminal_hit_sequence),
    }


def _trace_origin_edges_matlab_frontier(
    energy: np.ndarray,
    scale_indices: np.ndarray | None,
    vertex_positions: np.ndarray,
    vertex_scales: np.ndarray,
    lumen_radius_microns: np.ndarray,
    microns_per_voxel: np.ndarray,
    vertex_center_image: np.ndarray,
    origin_vertex_idx: int,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Trace a single origin using a MATLAB-style best-first voxel frontier."""
    shape = energy.shape
    # get_edges_for_vertex.m budgets both the trace count and the visited
    # frontier size from number_of_edges_per_vertex. The separate watershed
    # edge_number_tolerance heuristic does not cap this frontier tracer.
    max_edges_per_vertex = _matlab_frontier_edge_budget(params)
    max_length_ratio = float(params.get("max_edge_length_per_origin_radius", 60.0))
    strel_apothem = int(
        params.get(
            "space_strel_apothem_edges",
            params.get(
                "space_strel_apothem", max(1, round(params.get("step_size_per_origin_radius", 1.0)))
            ),
        )
    )
    offsets, offset_distances = _matlab_frontier_offsets(strel_apothem, microns_per_voxel)
    origin_coord = np.rint(vertex_positions[origin_vertex_idx]).astype(np.int32)
    origin_coord[0] = np.clip(origin_coord[0], 0, shape[0] - 1)
    origin_coord[1] = np.clip(origin_coord[1], 0, shape[1] - 1)
    origin_coord[2] = np.clip(origin_coord[2], 0, shape[2] - 1)
    origin_linear = _coord_to_matlab_linear_index(origin_coord, shape)
    origin_position_microns = origin_coord.astype(np.float64) * microns_per_voxel
    origin_scale = int(vertex_scales[origin_vertex_idx])
    origin_radius_microns = float(lumen_radius_microns[origin_scale])
    max_edge_length_microns = max_length_ratio * origin_radius_microns
    max_edge_length_voxels = int(np.round(max_edge_length_microns / np.min(microns_per_voxel))) + 1
    max_number_of_indices = max(1, max_edge_length_voxels * max_edges_per_vertex)

    diagnostics = _empty_edge_diagnostics()
    traces: list[np.ndarray] = []
    connections: list[list[int]] = []
    metrics: list[float] = []
    energy_traces: list[np.ndarray] = []
    scale_traces: list[np.ndarray] = []
    origin_indices: list[int] = []
    frontier_lifecycle_events: list[dict[str, Any]] = []
    edge_paths_linear: list[list[int]] = []
    edge_pairs: list[tuple[int, int]] = []
    displacement_vectors: list[np.ndarray] = []
    has_valid_terminal_edge = False
    previous_indices_visited: list[int] = []
    pointer_index_map: dict[int, int] = {origin_linear: 0}
    pointer_energy_map: dict[int, float] = {}
    # MATLAB seeds the origin distance map at 1 before any expansion. Matching
    # that off-by-one budget keeps the frontier's max-length cutoff aligned with
    # get_edges_for_vertex.m.
    distance_map: dict[int, float] = {origin_linear: 1.0}
    available_map: dict[int, float] = {}
    available_heap: list[tuple[float, int]] = []

    # MATLAB does not even enter the edge-search loop when the origin lies too
    # close to the border for the current structuring element.
    if np.any(origin_coord < strel_apothem) or np.any(
        origin_coord >= (np.asarray(shape, dtype=np.int32) - strel_apothem)
    ):
        diagnostics["stop_reason_counts"]["bounds"] += 1
        return {
            "origin_index": origin_vertex_idx,
            "candidate_source": "frontier",
            "traces": traces,
            "connections": connections,
            "metrics": metrics,
            "energy_traces": energy_traces,
            "scale_traces": scale_traces,
            "origin_indices": [origin_vertex_idx] * len(traces),
            "connection_sources": ["frontier"] * len(traces),
            "diagnostics": diagnostics,
        }

    current_linear = origin_linear

    while (
        len(edge_paths_linear) < max_edges_per_vertex
        and len(previous_indices_visited) < max_number_of_indices
    ):
        current_coord = _matlab_linear_index_to_coord(current_linear, shape)
        current_energy = float(energy[current_coord[0], current_coord[1], current_coord[2]])
        terminal_vertex_idx = (
            int(vertex_center_image[current_coord[0], current_coord[1], current_coord[2]]) - 1
        )
        if terminal_vertex_idx == origin_vertex_idx:
            terminal_vertex_idx = -1

        previous_indices_visited.append(current_linear)
        current_visit_order = len(previous_indices_visited)
        pointer_energy_map[current_linear] = float("-inf")

        neighbor_coords: Int32Array = np.asarray(current_coord + offsets, dtype=np.int32)
        valid_mask = (
            (neighbor_coords[:, 0] >= 0)
            & (neighbor_coords[:, 0] < shape[0])
            & (neighbor_coords[:, 1] >= 0)
            & (neighbor_coords[:, 1] < shape[1])
            & (neighbor_coords[:, 2] >= 0)
            & (neighbor_coords[:, 2] < shape[2])
        )
        neighbor_coords = np.asarray(neighbor_coords[valid_mask], dtype=np.int32)
        neighbor_distances = offset_distances[valid_mask]
        new_coords: list[np.ndarray] = []
        new_distances: list[float] = []
        for coord_row, distance in zip(neighbor_coords, neighbor_distances):
            coord: Int32Array = np.asarray(coord_row, dtype=np.int32)
            linear_index = _coord_to_matlab_linear_index(coord, shape)
            if pointer_energy_map.get(linear_index, 0.0) > current_energy:
                pointer_index_map[linear_index] = current_visit_order
                pointer_energy_map[linear_index] = current_energy
                distance_map[linear_index] = distance_map[current_linear] + float(distance)
                new_coords.append(coord.astype(np.int32, copy=False))
                new_distances.append(float(distance_map[linear_index]))

        new_coords_array: Int32Array = np.zeros((0, 3), dtype=np.int32)
        if new_coords:
            new_coords_array = np.asarray(new_coords, dtype=np.int32)
            new_distances_array = np.asarray(new_distances, dtype=np.float32)
            diagnostics["stop_reason_counts"]["length_limit"] += int(
                np.sum(new_distances_array >= max_edge_length_microns)
            )
            within_length: BoolArray = new_distances_array < max_edge_length_microns
            new_coords_array = new_coords_array[within_length]
            if len(new_coords_array) and has_valid_terminal_edge:
                new_coords_array = _prune_frontier_indices_beyond_found_vertices(
                    new_coords_array,
                    origin_position_microns,
                    displacement_vectors,
                    microns_per_voxel,
                )

        if terminal_vertex_idx >= 0:
            diagnostics["stop_reason_counts"]["terminal_frontier_hit"] += 1
            diagnostics.setdefault("frontier_per_origin_terminal_hits", {})
            diagnostics["frontier_per_origin_terminal_hits"][str(origin_vertex_idx)] = (
                int(diagnostics["frontier_per_origin_terminal_hits"].get(str(origin_vertex_idx), 0))
                + 1
            )
            terminal_hit_sequence = int(
                diagnostics["frontier_per_origin_terminal_hits"].get(str(origin_vertex_idx), 0)
            )
            path_linear = [current_linear]
            tracing_linear = current_linear
            while int(pointer_index_map.get(tracing_linear, 0)) > 0:
                tracing_linear = previous_indices_visited[
                    int(pointer_index_map[tracing_linear]) - 1
                ]
                path_linear.append(tracing_linear)

            origin_idx, terminal_idx, resolution_reason = _resolve_frontier_edge_connection_details(
                path_linear,
                terminal_vertex_idx,
                origin_vertex_idx,
                edge_paths_linear,
                edge_pairs,
                pointer_index_map,
                energy,
                shape,
            )
            diagnostics.setdefault("frontier_terminal_resolution_counts", {})
            diagnostics["frontier_terminal_resolution_counts"][resolution_reason] = (
                int(diagnostics["frontier_terminal_resolution_counts"].get(resolution_reason, 0))
                + 1
            )

            if origin_idx is not None and terminal_idx is not None:
                diagnostics.setdefault("frontier_per_origin_terminal_accepts", {})
                diagnostics["frontier_per_origin_terminal_accepts"][str(origin_vertex_idx)] = (
                    int(
                        diagnostics["frontier_per_origin_terminal_accepts"].get(
                            str(origin_vertex_idx), 0
                        )
                    )
                    + 1
                )
                path_record_index = len(edge_paths_linear) + 1
                for path_index in path_linear[:-1]:
                    pointer_index_map[path_index] = -path_record_index

                edge_paths_linear.append(path_linear)
                edge_pairs.append((int(terminal_idx), int(origin_idx)))

                current_position = current_coord.astype(np.float64) * microns_per_voxel
                displacement = current_position - origin_position_microns
                displacement_norm_sq = float(np.sum(displacement**2))
                if displacement_norm_sq > 0:
                    displacement_vectors.append(displacement / displacement_norm_sq)

                has_valid_terminal_edge = True
                edge_trace = _path_coords_from_linear_indices(path_linear, shape)
                energy_trace = _trace_energy_series(edge_trace, energy)
                scale_trace = _trace_scale_series(edge_trace, scale_indices)
                traces.append(edge_trace)
                connections.append([int(origin_idx), int(terminal_idx)])
                metrics.append(_edge_metric_from_energy_trace(energy_trace))
                energy_traces.append(energy_trace)
                scale_traces.append(scale_trace)
                origin_indices.append(origin_vertex_idx)
                diagnostics["terminal_direct_hit_count"] += 1
                frontier_lifecycle_events.append(
                    _build_frontier_lifecycle_event(
                        seed_origin_idx=origin_vertex_idx,
                        terminal_vertex_idx=terminal_vertex_idx,
                        origin_idx=int(origin_idx),
                        terminal_idx=int(terminal_idx),
                        resolution_reason=resolution_reason,
                        terminal_hit_sequence=terminal_hit_sequence,
                        local_candidate_index=len(connections) - 1,
                    )
                )
            else:
                diagnostics.setdefault("frontier_per_origin_terminal_rejections", {})
                diagnostics["frontier_per_origin_terminal_rejections"][str(origin_vertex_idx)] = (
                    int(
                        diagnostics["frontier_per_origin_terminal_rejections"].get(
                            str(origin_vertex_idx), 0
                        )
                    )
                    + 1
                )
                frontier_lifecycle_events.append(
                    _build_frontier_lifecycle_event(
                        seed_origin_idx=origin_vertex_idx,
                        terminal_vertex_idx=terminal_vertex_idx,
                        origin_idx=None,
                        terminal_idx=None,
                        resolution_reason=resolution_reason,
                        terminal_hit_sequence=terminal_hit_sequence,
                    )
                )
                for coord in new_coords_array:
                    linear_index = _coord_to_matlab_linear_index(coord, shape)
                    available_energy = float(energy[coord[0], coord[1], coord[2]])
                    available_map[linear_index] = available_energy
                    heappush(available_heap, (available_energy, linear_index))
            if origin_idx is not None and terminal_idx is not None and len(new_coords_array):
                # MATLAB clears the newly exposed frontier voxels after any terminal
                # hit that survives parent/child resolution instead of leaving
                # them queued for later expansion.
                for coord in new_coords_array:
                    linear_index = _coord_to_matlab_linear_index(coord, shape)
                    available_map.pop(linear_index, None)
        else:
            for coord in new_coords_array:
                linear_index = _coord_to_matlab_linear_index(coord, shape)
                available_energy = float(energy[coord[0], coord[1], coord[2]])
                available_map[linear_index] = available_energy
                # MATLAB parity note: the tiebreaker for equal-energy frontier
                # voxels is the linear index, which corresponds to MATLAB's
                # column-major order. This matches get_edges_for_vertex.m.
                heappush(available_heap, (available_energy, linear_index))

        available_map.pop(current_linear, None)
        next_current = None
        stopped_on_nonnegative = False
        while available_heap:
            candidate_energy, candidate_linear = heappop(available_heap)
            if available_map.get(candidate_linear) != candidate_energy:
                continue
            if candidate_energy >= 0:
                available_map.pop(candidate_linear, None)
                diagnostics["stop_reason_counts"]["frontier_exhausted_nonnegative"] += 1
                available_heap.clear()
                stopped_on_nonnegative = True
                next_current = None
                break
            next_current = int(candidate_linear)
            break

        if next_current is None:
            if not available_map and not stopped_on_nonnegative:
                diagnostics["stop_reason_counts"]["frontier_exhausted_nonnegative"] += 1
            break

        current_linear = next_current

    return {
        "origin_index": origin_vertex_idx,
        "candidate_source": "frontier",
        "traces": traces,
        "connections": connections,
        "metrics": metrics,
        "energy_traces": energy_traces,
        "scale_traces": scale_traces,
        "origin_indices": [origin_vertex_idx] * len(traces),
        "connection_sources": ["frontier"] * len(traces),
        "frontier_lifecycle_events": frontier_lifecycle_events,
        "diagnostics": diagnostics,
    }


def _append_candidate_unit(target: dict[str, Any], unit_payload: dict[str, Any]) -> None:
    """Append a per-origin candidate payload into the aggregate candidate manifest."""
    unit_traces = [np.asarray(trace, dtype=np.float32) for trace in unit_payload["traces"]]
    unit_connections = np.asarray(unit_payload["connections"], dtype=np.int32).reshape(-1, 2)
    unit_metrics = np.asarray(unit_payload["metrics"], dtype=np.float32).reshape(-1)
    unit_origin_indices = np.asarray(
        unit_payload.get("origin_indices", []), dtype=np.int32
    ).reshape(-1)
    unit_connection_sources = _normalize_candidate_connection_sources(
        unit_payload.get("connection_sources"),
        len(unit_connections),
        default_source=str(unit_payload.get("candidate_source", "unknown")),
    )
    target.setdefault("frontier_lifecycle_events", [])
    base_candidate_index = len(target["traces"])
    emitted_frontier_count = 0
    for raw_event in unit_payload.get("frontier_lifecycle_events", []):
        if not isinstance(raw_event, dict):
            continue
        event = dict(raw_event)
        if event.get("survived_candidate_manifest"):
            event["manifest_candidate_index"] = base_candidate_index + emitted_frontier_count
            emitted_frontier_count += 1
        else:
            event["manifest_candidate_index"] = None
        target["frontier_lifecycle_events"].append(event)

    target["traces"].extend(unit_traces)
    target["energy_traces"].extend(
        np.asarray(trace, dtype=np.float32) for trace in unit_payload["energy_traces"]
    )
    target["scale_traces"].extend(
        np.asarray(trace, dtype=np.int16) for trace in unit_payload["scale_traces"]
    )

    if unit_connections.size:
        target["connections"] = (
            unit_connections
            if target["connections"].size == 0
            else np.vstack([target["connections"], unit_connections])
        )
        target["metrics"] = np.concatenate([target["metrics"], unit_metrics])
        target["origin_indices"] = np.concatenate([target["origin_indices"], unit_origin_indices])
        target.setdefault("connection_sources", []).extend(unit_connection_sources)

    _merge_edge_diagnostics(
        cast("dict[str, Any]", target["diagnostics"]),
        cast("dict[str, Any]", unit_payload.get("diagnostics", {})),
    )


def _build_frontier_candidate_lifecycle(
    candidates: dict[str, Any],
    chosen_candidate_indices: np.ndarray | list[int] | None = None,
) -> dict[str, Any]:
    """Build a JSON-friendly frontier lifecycle artifact for shared-neighborhood audits."""
    raw_events = candidates.get("frontier_lifecycle_events", [])
    chosen_indices = {
        int(index)
        for index in np.asarray(
            chosen_candidate_indices if chosen_candidate_indices is not None else [], dtype=np.int32
        ).reshape(-1)
    }
    events: list[dict[str, Any]] = []
    per_origin_summary: dict[int, dict[str, Any]] = {}

    for raw_event in raw_events:
        if not isinstance(raw_event, dict):
            continue
        event = dict(raw_event)
        seed_origin_index = int(event.get("seed_origin_index", -1))
        manifest_candidate_index_raw = event.get("manifest_candidate_index")
        manifest_candidate_index = (
            int(manifest_candidate_index_raw)
            if manifest_candidate_index_raw not in (None, "")
            else None
        )
        chosen_final_edge = (
            manifest_candidate_index is not None and manifest_candidate_index in chosen_indices
        )
        event["seed_origin_index"] = seed_origin_index
        event["terminal_vertex_index"] = int(event.get("terminal_vertex_index", -1))
        event["terminal_hit_sequence"] = int(event.get("terminal_hit_sequence", 0))
        event["survived_candidate_manifest"] = bool(event.get("survived_candidate_manifest", False))
        event["manifest_candidate_index"] = manifest_candidate_index
        event["chosen_final_edge"] = bool(chosen_final_edge)
        events.append(event)

        summary = per_origin_summary.setdefault(
            seed_origin_index,
            {
                "origin_index": seed_origin_index,
                "terminal_hit_count": 0,
                "emitted_candidate_count": 0,
                "rejected_terminal_count": 0,
                "chosen_final_edge_count": 0,
                "resolution_counts": {},
                "emitted_endpoint_pair_samples": [],
                "rejection_reason_samples": [],
            },
        )
        summary["terminal_hit_count"] += 1
        resolution_reason = str(event.get("resolution_reason", "unknown"))
        resolution_counts = cast("dict[str, int]", summary["resolution_counts"])
        resolution_counts[resolution_reason] = int(resolution_counts.get(resolution_reason, 0)) + 1
        if event["survived_candidate_manifest"]:
            summary["emitted_candidate_count"] += 1
            if chosen_final_edge:
                summary["chosen_final_edge_count"] += 1
            endpoint_pair = event.get("emitted_endpoint_pair")
            if (
                isinstance(endpoint_pair, list)
                and endpoint_pair not in summary["emitted_endpoint_pair_samples"]
                and len(summary["emitted_endpoint_pair_samples"]) < 3
            ):
                summary["emitted_endpoint_pair_samples"].append(endpoint_pair)
        else:
            summary["rejected_terminal_count"] += 1
            rejection_reason = event.get("rejection_reason")
            if (
                isinstance(rejection_reason, str)
                and rejection_reason
                and rejection_reason not in summary["rejection_reason_samples"]
                and len(summary["rejection_reason_samples"]) < 3
            ):
                summary["rejection_reason_samples"].append(rejection_reason)

    per_origin_payload = [
        per_origin_summary[origin_index] for origin_index in sorted(per_origin_summary)
    ]
    per_origin_payload.sort(
        key=lambda item: (
            -int(item["terminal_hit_count"]),
            -int(item["rejected_terminal_count"]),
            int(item["origin_index"]),
        )
    )
    return {
        "schema_version": 1,
        "frontier_terminal_hit_event_count": len(events),
        "frontier_terminal_accept_event_count": len(
            [event for event in events if event.get("survived_candidate_manifest")]
        ),
        "frontier_terminal_reject_event_count": len(
            [event for event in events if not event.get("survived_candidate_manifest")]
        ),
        "events": events,
        "per_origin_summary": per_origin_payload,
        "top_origin_summaries": per_origin_payload[:5],
    }


def _normalize_candidate_origin_counts(raw_counts: dict[Any, Any] | None) -> dict[str, int]:
    """Return a JSON-safe mapping from origin index to candidate count."""
    normalized: dict[str, int] = {}
    if not raw_counts:
        return normalized

    for key, value in raw_counts.items():
        try:
            normalized[str(int(key))] = int(value)
        except (TypeError, ValueError):
            continue
    return normalized


def _normalize_candidate_count_map(raw_counts: dict[Any, Any] | None) -> dict[str, int]:
    """Return a JSON-safe additive counter map with string keys."""
    normalized: dict[str, int] = {}
    if not raw_counts:
        return normalized

    for key, value in raw_counts.items():
        key_text = str(key).strip()
        if not key_text:
            continue
        try:
            normalized[key_text] = int(value)
        except (TypeError, ValueError):
            continue
    return normalized


def _normalize_candidate_connection_sources(
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


def _build_edge_candidate_audit(
    candidates: dict[str, Any],
    vertex_count: int,
    use_frontier_tracer: bool,
    frontier_origin_counts: dict[int, int] | None = None,
    supplement_origin_counts: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Build a stable, JSON-serializable summary of edge-candidate provenance."""
    connections = np.asarray(
        candidates.get("connections", np.zeros((0, 2), dtype=np.int32)), dtype=np.int32
    )
    if connections.ndim == 1 and connections.size > 0:
        connections = connections.reshape(-1, 2)
    candidate_connection_count = int(connections.shape[0]) if connections.size else 0

    origin_indices = np.asarray(
        candidates.get("origin_indices", np.zeros((0,), dtype=np.int32)), dtype=np.int32
    )
    origin_indices = origin_indices.reshape(-1)
    if origin_indices.size != candidate_connection_count:
        origin_indices = np.zeros((candidate_connection_count,), dtype=np.int32)

    connection_sources = _normalize_candidate_connection_sources(
        candidates.get("connection_sources"),
        candidate_connection_count,
        default_source=str(candidates.get("candidate_source", "unknown")),
    )

    total_origin_counts: dict[int, int] = {}
    total_origin_pairs: dict[int, set[tuple[int, int]]] = {}
    source_pair_sets: dict[str, set[tuple[int, int]]] = {
        "frontier": set(),
        "watershed": set(),
        "geodesic": set(),
        "fallback": set(),
    }
    pair_sources: dict[tuple[int, int], set[str]] = {}
    source_origin_pair_sets: dict[str, dict[int, set[tuple[int, int]]]] = {
        "frontier": {},
        "watershed": {},
        "geodesic": {},
        "fallback": {},
    }
    source_origin_sets: dict[str, set[int]] = {
        "frontier": set(),
        "watershed": set(),
        "geodesic": set(),
        "fallback": set(),
    }
    for index, origin_index in enumerate(origin_indices):
        origin_index_int = int(origin_index)
        if origin_index_int < 0:
            continue
        total_origin_counts[origin_index_int] = total_origin_counts.get(origin_index_int, 0) + 1

        source_label = connection_sources[index] if index < len(connection_sources) else "unknown"
        if source_label in source_origin_sets:
            source_origin_sets[source_label].add(origin_index_int)

        if index >= len(connections):
            continue
        start_vertex, end_vertex = (int(value) for value in connections[index][:2])
        if start_vertex < 0 or end_vertex < 0:
            continue
        endpoint_pair = (
            (start_vertex, end_vertex) if start_vertex < end_vertex else (end_vertex, start_vertex)
        )
        total_origin_pairs.setdefault(origin_index_int, set()).add(endpoint_pair)
        if source_label in source_pair_sets:
            pair_sources.setdefault(endpoint_pair, set()).add(source_label)
            source_pair_sets[source_label].add(endpoint_pair)
            source_origin_pair_sets[source_label].setdefault(origin_index_int, set()).add(
                endpoint_pair
            )

    frontier_origin_counts = {
        int(origin): int(count) for origin, count in (frontier_origin_counts or {}).items()
    }
    supplement_origin_counts = {
        int(origin): int(count) for origin, count in (supplement_origin_counts or {}).items()
    }
    diag = candidates.get("diagnostics", {})
    geodesic_origin_counts = _normalize_candidate_origin_counts(
        diag.get("geodesic_per_origin_candidate_counts") if isinstance(diag, dict) else None
    )
    geodesic_origin_counts_int = {
        int(origin): int(count) for origin, count in geodesic_origin_counts.items()
    }
    frontier_terminal_hits = _normalize_candidate_origin_counts(
        diag.get("frontier_per_origin_terminal_hits") if isinstance(diag, dict) else None
    )
    frontier_terminal_accepts = _normalize_candidate_origin_counts(
        diag.get("frontier_per_origin_terminal_accepts") if isinstance(diag, dict) else None
    )
    frontier_terminal_rejections = _normalize_candidate_origin_counts(
        diag.get("frontier_per_origin_terminal_rejections") if isinstance(diag, dict) else None
    )
    frontier_terminal_hits_int = {
        int(origin): int(count) for origin, count in frontier_terminal_hits.items()
    }
    frontier_terminal_accepts_int = {
        int(origin): int(count) for origin, count in frontier_terminal_accepts.items()
    }
    frontier_terminal_rejections_int = {
        int(origin): int(count) for origin, count in frontier_terminal_rejections.items()
    }
    frontier_connection_count = (
        sum(frontier_origin_counts.values())
        if frontier_origin_counts
        else len([source for source in connection_sources if source == "frontier"])
    )
    supplement_connection_count = (
        sum(supplement_origin_counts.values())
        if supplement_origin_counts
        else len([source for source in connection_sources if source == "watershed"])
    )
    geodesic_connection_count = len(
        [source for source in connection_sources if source == "geodesic"]
    )
    fallback_connection_count = max(
        0,
        candidate_connection_count
        - frontier_connection_count
        - supplement_connection_count
        - geodesic_connection_count,
    )
    frontier_origin_count = (
        len(frontier_origin_counts)
        if frontier_origin_counts
        else len(source_origin_sets["frontier"])
    )
    supplement_origin_count = (
        len(supplement_origin_counts)
        if supplement_origin_counts
        else len(source_origin_sets["watershed"])
    )
    geodesic_origin_count = (
        len(geodesic_origin_counts_int)
        if geodesic_origin_counts_int
        else len(source_origin_sets["geodesic"])
    )
    fallback_origin_count = max(
        0,
        len(total_origin_counts)
        - len(
            source_origin_sets["frontier"]
            | source_origin_sets["watershed"]
            | source_origin_sets["geodesic"]
        ),
    )

    per_origin_payload: list[dict[str, Any]] = []
    all_origins = (
        set(total_origin_counts.keys())
        | set(frontier_origin_counts.keys())
        | set(supplement_origin_counts.keys())
        | set(geodesic_origin_counts_int.keys())
        | set(frontier_terminal_hits_int.keys())
        | set(frontier_terminal_accepts_int.keys())
        | set(frontier_terminal_rejections_int.keys())
    )
    for origin_index in sorted(all_origins):
        frontier_count = int(frontier_origin_counts.get(origin_index, 0))
        supplement_count = int(supplement_origin_counts.get(origin_index, 0))
        geodesic_count = int(geodesic_origin_counts_int.get(origin_index, 0))
        frontier_terminal_hit_count = int(frontier_terminal_hits_int.get(origin_index, 0))
        frontier_terminal_accept_count = int(frontier_terminal_accepts_int.get(origin_index, 0))
        frontier_terminal_rejection_count = int(
            frontier_terminal_rejections_int.get(origin_index, 0)
        )
        total_count = int(total_origin_counts.get(origin_index, 0))
        fallback_count = max(0, total_count - frontier_count - supplement_count - geodesic_count)
        candidate_pairs = total_origin_pairs.get(origin_index, set())
        frontier_pairs = source_origin_pair_sets["frontier"].get(origin_index, set())
        watershed_pairs = source_origin_pair_sets["watershed"].get(origin_index, set())
        geodesic_pairs = source_origin_pair_sets["geodesic"].get(origin_index, set())
        fallback_pairs = source_origin_pair_sets["fallback"].get(origin_index, set())
        per_origin_payload.append(
            {
                "origin_index": origin_index,
                "frontier_candidate_count": frontier_count,
                "watershed_candidate_count": supplement_count,
                "geodesic_candidate_count": geodesic_count,
                "fallback_candidate_count": fallback_count,
                "frontier_terminal_hit_count": frontier_terminal_hit_count,
                "frontier_terminal_accept_count": frontier_terminal_accept_count,
                "frontier_terminal_rejection_count": frontier_terminal_rejection_count,
                "candidate_connection_count": total_count,
                "candidate_endpoint_pair_count": len(candidate_pairs),
                "candidate_endpoint_pair_samples": sorted(candidate_pairs)[:3],
                "frontier_endpoint_pair_count": len(frontier_pairs),
                "frontier_endpoint_pair_samples": sorted(frontier_pairs)[:3],
                "watershed_endpoint_pair_count": len(watershed_pairs),
                "watershed_endpoint_pair_samples": sorted(watershed_pairs)[:3],
                "geodesic_endpoint_pair_count": len(geodesic_pairs),
                "geodesic_endpoint_pair_samples": sorted(geodesic_pairs)[:3],
                "fallback_endpoint_pair_count": len(fallback_pairs),
                "fallback_endpoint_pair_samples": sorted(fallback_pairs)[:3],
            }
        )
    candidate_diagnostics: dict[str, int] = {
        "candidate_traced_edge_count": int(diag.get("candidate_traced_edge_count", 0)),
        "terminal_edge_count": int(diag.get("terminal_edge_count", 0)),
        "chosen_edge_count": int(diag.get("chosen_edge_count", 0)),
        "watershed_join_supplement_count": int(diag.get("watershed_join_supplement_count", 0)),
        "watershed_endpoint_degree_rejected": int(
            diag.get("watershed_endpoint_degree_rejected", 0)
        ),
        "watershed_total_pairs": int(diag.get("watershed_total_pairs", 0)),
        "watershed_already_existing": int(diag.get("watershed_already_existing", 0)),
        "watershed_short_trace_rejected": int(diag.get("watershed_short_trace_rejected", 0)),
        "watershed_energy_rejected": int(diag.get("watershed_energy_rejected", 0)),
        "watershed_metric_threshold_rejected": int(
            diag.get("watershed_metric_threshold_rejected", 0)
        ),
        "watershed_reachability_rejected": int(diag.get("watershed_reachability_rejected", 0)),
        "watershed_mutual_frontier_rejected": int(
            diag.get("watershed_mutual_frontier_rejected", 0)
        ),
        "watershed_cap_rejected": int(diag.get("watershed_cap_rejected", 0)),
        "watershed_accepted": int(diag.get("watershed_accepted", 0)),
        "geodesic_join_supplement_count": int(diag.get("geodesic_join_supplement_count", 0)),
        "geodesic_route_failed": int(diag.get("geodesic_route_failed", 0)),
        "geodesic_energy_rejected": int(diag.get("geodesic_energy_rejected", 0)),
        "geodesic_metric_threshold_rejected": int(
            diag.get("geodesic_metric_threshold_rejected", 0)
        ),
        "geodesic_path_ratio_rejected": int(diag.get("geodesic_path_ratio_rejected", 0)),
        "geodesic_vertex_crossing_rejected": int(diag.get("geodesic_vertex_crossing_rejected", 0)),
        "geodesic_endpoint_degree_rejected": int(diag.get("geodesic_endpoint_degree_rejected", 0)),
        "geodesic_origin_budget_rejected": int(diag.get("geodesic_origin_budget_rejected", 0)),
        "geodesic_shared_neighborhood_endpoint_relaxed": int(
            diag.get("geodesic_shared_neighborhood_endpoint_relaxed", 0)
        ),
        "geodesic_accepted": int(diag.get("geodesic_accepted", 0)),
        "frontier_origins_with_candidates": int(diag.get("frontier_origins_with_candidates", 0)),
        "frontier_origins_without_candidates": int(
            diag.get("frontier_origins_without_candidates", 0)
        ),
    }

    fallback_source_total = {
        "candidate_connection_count": fallback_connection_count,
        "candidate_origin_count": fallback_origin_count,
        "candidate_endpoint_pair_count": len(source_pair_sets["fallback"]),
        "candidate_endpoint_pair_samples": sorted(source_pair_sets["fallback"])[:5],
    }
    frontier_only_pairs = sorted(
        pair for pair, sources in pair_sources.items() if sources == {"frontier"}
    )
    watershed_only_pairs = sorted(
        pair for pair, sources in pair_sources.items() if sources == {"watershed"}
    )
    fallback_only_pairs = sorted(
        pair for pair, sources in pair_sources.items() if sources == {"fallback"}
    )
    multi_source_pairs = sorted(pair for pair, sources in pair_sources.items() if len(sources) > 1)

    return {
        "schema_version": 1,
        "vertex_count": int(vertex_count),
        "use_frontier_tracer": bool(use_frontier_tracer),
        "candidate_traces": len(candidates.get("traces", [])),
        "candidate_connection_count": candidate_connection_count,
        "candidate_origin_count": len(total_origin_counts),
        "source_breakdown": {
            "frontier": {
                "candidate_connection_count": frontier_connection_count,
                "candidate_origin_count": frontier_origin_count,
                "candidate_endpoint_pair_count": len(source_pair_sets["frontier"]),
                "candidate_endpoint_pair_samples": sorted(source_pair_sets["frontier"])[:5],
            },
            "watershed": {
                "candidate_connection_count": supplement_connection_count,
                "candidate_origin_count": supplement_origin_count,
                "candidate_endpoint_pair_count": len(source_pair_sets["watershed"]),
                "candidate_endpoint_pair_samples": sorted(source_pair_sets["watershed"])[:5],
            },
            "geodesic": {
                "candidate_connection_count": geodesic_connection_count,
                "candidate_origin_count": geodesic_origin_count,
                "candidate_endpoint_pair_count": len(source_pair_sets["geodesic"]),
                "candidate_endpoint_pair_samples": sorted(source_pair_sets["geodesic"])[:5],
            },
            "fallback": fallback_source_total,
        },
        "frontier_per_origin_candidate_counts": frontier_origin_counts,
        "frontier_per_origin_terminal_hits": frontier_terminal_hits,
        "frontier_per_origin_terminal_accepts": frontier_terminal_accepts,
        "frontier_per_origin_terminal_rejections": frontier_terminal_rejections,
        "watershed_per_origin_candidate_counts": _normalize_candidate_origin_counts(
            diag.get("watershed_per_origin_candidate_counts")
        ),
        "geodesic_per_origin_candidate_counts": geodesic_origin_counts,
        "frontier_terminal_resolution_counts": _normalize_candidate_count_map(
            diag.get("frontier_terminal_resolution_counts")
        ),
        "pair_source_breakdown": {
            "frontier_only_pair_count": len(frontier_only_pairs),
            "watershed_only_pair_count": len(watershed_only_pairs),
            "geodesic_only_pair_count": len(
                [pair for pair, sources in pair_sources.items() if sources == {"geodesic"}]
            ),
            "fallback_only_pair_count": len(fallback_only_pairs),
            "multi_source_pair_count": len(multi_source_pairs),
            "frontier_only_endpoint_pair_samples": frontier_only_pairs[:5],
            "watershed_only_endpoint_pair_samples": watershed_only_pairs[:5],
            "geodesic_only_endpoint_pair_samples": [
                pair for pair, sources in pair_sources.items() if sources == {"geodesic"}
            ][:5],
            "fallback_only_endpoint_pair_samples": fallback_only_pairs[:5],
        },
        "per_origin_summary": per_origin_payload,
        "diagnostic_counters": candidate_diagnostics,
    }


def _generate_edge_candidates_matlab_frontier(
    energy: np.ndarray,
    scale_indices: np.ndarray | None,
    vertex_positions: np.ndarray,
    vertex_scales: np.ndarray,
    lumen_radius_microns: np.ndarray,
    microns_per_voxel: np.ndarray,
    vertex_center_image: np.ndarray,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Generate edge candidates using the MATLAB-style best-first frontier search."""
    candidates: dict[str, Any] = {
        "traces": [],
        "connections": np.zeros((0, 2), dtype=np.int32),
        "metrics": np.zeros((0,), dtype=np.float32),
        "energy_traces": [],
        "scale_traces": [],
        "origin_indices": np.zeros((0,), dtype=np.int32),
        "connection_sources": [],
        "diagnostics": _empty_edge_diagnostics(),
    }
    per_origin_candidate_counts: dict[int, int] = {}
    for origin_vertex_idx in range(len(vertex_positions)):
        unit_payload = _trace_origin_edges_matlab_frontier(
            energy,
            scale_indices,
            vertex_positions,
            vertex_scales,
            lumen_radius_microns,
            microns_per_voxel,
            vertex_center_image,
            origin_vertex_idx,
            params,
        )
        n_unit_traces = len(unit_payload.get("traces", []))
        if n_unit_traces > 0:
            per_origin_candidate_counts[origin_vertex_idx] = n_unit_traces
        _append_candidate_unit(candidates, unit_payload)

    # Phase 1 per-origin summary diagnostics
    candidates["diagnostics"]["frontier_origins_with_candidates"] = len(per_origin_candidate_counts)
    candidates["diagnostics"]["frontier_origins_without_candidates"] = len(vertex_positions) - len(
        per_origin_candidate_counts
    )
    candidates["diagnostics"]["frontier_per_origin_candidate_counts"] = per_origin_candidate_counts
    logger.info(
        "Frontier candidates: %d origins produced candidates, %d did not",
        len(per_origin_candidate_counts),
        len(vertex_positions) - len(per_origin_candidate_counts),
    )
    return candidates


def _generate_edge_candidates(
    energy: np.ndarray,
    scale_indices: np.ndarray | None,
    vertex_positions: np.ndarray,
    vertex_scales: np.ndarray,
    lumen_radius_pixels: np.ndarray,
    lumen_radius_microns: np.ndarray,
    microns_per_voxel: np.ndarray,
    vertex_center_image: np.ndarray | None,
    tree: cKDTree,
    max_search_radius: float,
    params: dict[str, Any],
    energy_sign: float,
) -> dict[str, Any]:
    """Generate directed edge candidates without final dedupe or degree pruning."""
    max_edges_per_vertex = params.get("number_of_edges_per_vertex", 4)
    step_size_ratio = params.get("step_size_per_origin_radius", 1.0)
    max_edge_energy = params.get("max_edge_energy", 0.0)
    max_length_ratio = params.get("max_edge_length_per_origin_radius", 60.0)
    discrete_tracing = params.get("discrete_tracing", False)
    direction_method = params.get("direction_method", "hessian")

    traces: list[np.ndarray] = []
    connections: list[list[int]] = []
    metrics: list[float] = []
    energy_traces: list[np.ndarray] = []
    scale_traces: list[np.ndarray] = []
    origin_indices: list[int] = []
    connection_sources: list[str] = []
    diagnostics = _empty_edge_diagnostics()

    energy_prepared = np.ascontiguousarray(energy, dtype=np.float64)
    mpv_prepared = np.asarray(microns_per_voxel, dtype=np.float64)

    for vertex_idx, (start_pos, start_scale) in enumerate(zip(vertex_positions, vertex_scales)):
        start_radius = _scalar_radius(lumen_radius_pixels[start_scale])
        step_size = start_radius * step_size_ratio
        max_length = start_radius * max_length_ratio
        max_steps = max(1, int(np.ceil(max_length / max(step_size, 1e-12))))

        if direction_method == "hessian":
            directions = estimate_vessel_directions(
                energy, start_pos, start_radius, microns_per_voxel
            )
            if directions.shape[0] < max_edges_per_vertex:
                extra = generate_edge_directions(
                    max_edges_per_vertex - directions.shape[0], seed=vertex_idx
                )
                directions = np.vstack([directions, extra])
            else:
                directions = directions[:max_edges_per_vertex]
        else:
            directions = generate_edge_directions(max_edges_per_vertex, seed=vertex_idx)

        for direction in directions:
            trace_result = trace_edge(
                energy_prepared,
                start_pos,
                direction,
                step_size,
                max_edge_energy,
                vertex_positions,
                vertex_scales,
                lumen_radius_pixels,
                lumen_radius_microns,
                max_steps,
                mpv_prepared,
                energy_sign,
                discrete_steps=discrete_tracing,
                vertex_center_image=vertex_center_image,
                tree=tree,
                max_search_radius=max_search_radius,
                origin_vertex_idx=vertex_idx,
                return_metadata=True,
            )
            edge_trace, trace_metadata = cast(
                "tuple[list[np.ndarray], TraceMetadata]",
                trace_result,
            )
            if len(edge_trace) <= 1:
                continue

            edge_arr = np.asarray(edge_trace, dtype=np.float32)
            terminal_vertex = trace_metadata["terminal_vertex"]
            energy_trace = _trace_energy_series(edge_arr, energy)
            scale_trace = _trace_scale_series(edge_arr, scale_indices)
            _record_trace_diagnostics(diagnostics, trace_metadata)

            traces.append(edge_arr)
            connections.append([vertex_idx, terminal_vertex if terminal_vertex is not None else -1])
            metrics.append(_edge_metric_from_energy_trace(energy_trace))
            energy_traces.append(energy_trace)
            scale_traces.append(scale_trace)
            origin_indices.append(vertex_idx)
            connection_sources.append("fallback")

    return {
        "traces": traces,
        "connections": np.asarray(connections, dtype=np.int32).reshape(-1, 2),
        "metrics": np.asarray(metrics, dtype=np.float32),
        "energy_traces": energy_traces,
        "scale_traces": scale_traces,
        "origin_indices": np.asarray(origin_indices, dtype=np.int32),
        "candidate_source": "fallback",
        "connection_sources": connection_sources,
        "diagnostics": diagnostics,
    }
