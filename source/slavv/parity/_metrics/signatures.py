"""Signature helpers for parity metrics."""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_position_array(positions: Any) -> np.ndarray:
    """Normalize position payloads into a 2D numpy array."""
    array = np.asarray(positions)
    if array.size == 0:
        return np.array([])
    return array.reshape(1, -1) if array.ndim == 1 else array


def _round_positions(positions: Any) -> np.ndarray:
    """Convert coordinate payloads to rounded integer voxel positions."""
    array = _as_position_array(positions)
    if array.size == 0:
        return np.empty((0, 3), dtype=np.int32)
    return np.rint(array[:, :3]).astype(np.int32, copy=False)


def _extract_vertex_scales(payload: dict[str, Any]) -> np.ndarray:
    """Extract 0-based scale indices from vertex payloads."""
    for key in ("scales", "scale_indices"):
        value = payload.get(key)
        if value is not None:
            return np.rint(np.asarray(value)).astype(np.int32, copy=False).reshape(-1)

    positions = _as_position_array(payload.get("positions", np.array([])))
    if positions.size > 0 and positions.shape[1] >= 4:
        return np.rint(positions[:, 3]).astype(np.int32, copy=False).reshape(-1)
    return np.empty((0,), dtype=np.int32)


def _extract_vertex_energies(payload: dict[str, Any]) -> np.ndarray:
    """Extract per-vertex energies when available."""
    value = payload.get("energies")
    if value is None:
        return np.empty((0,), dtype=np.float32)
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _vertex_signatures(
    payload: dict[str, Any],
) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    """Build exact-match signatures for vertices."""
    coords = _round_positions(payload.get("positions", np.array([])))
    scales = _extract_vertex_scales(payload)
    energies = _extract_vertex_energies(payload)

    coords_scales: list[tuple[Any, ...]] = []
    coords_scales_energies: list[tuple[Any, ...]] = []
    for index, coord in enumerate(coords):
        coord_tuple = tuple(int(value) for value in coord.tolist())
        scale = int(scales[index]) if index < len(scales) else None
        coords_scales.append((coord_tuple, scale))
        energy = None
        if index < len(energies):
            energy = round(float(energies[index]), 6)
        coords_scales_energies.append((coord_tuple, scale, energy))
    return coords_scales, coords_scales_energies


def _sample_counter_diff(counter_a: Counter, counter_b: Counter, limit: int = 3) -> list[Any]:
    """Return a few mismatch samples present in `counter_a` but not `counter_b`."""
    samples = []
    for item, count in (counter_a - counter_b).items():
        for _ in range(count):
            samples.append(item)
            if len(samples) >= limit:
                return samples
    return samples


def _sample_set_diff(items_a: set[Any], items_b: set[Any], limit: int = 3) -> list[Any]:
    """Return deterministic samples present in `items_a` but not `items_b`."""
    return sorted(items_a - items_b)[:limit]


def _canonical_trace(trace: Any) -> tuple[tuple[int, int, int], ...]:
    """Canonicalize a voxel trace independent of traversal direction."""
    coords = _round_positions(trace)
    if coords.size == 0:
        return ()
    forward = tuple(tuple(int(v) for v in row.tolist()) for row in coords)
    reverse = tuple(tuple(int(v) for v in row.tolist()) for row in coords[::-1])
    return min(forward, reverse)


def _edge_signatures(
    payload: dict[str, Any],
    include_trace: bool,
    include_energy: bool,
) -> list[tuple[Any, ...]]:
    """Build exact-match signatures for chosen edges."""
    connections = np.asarray(payload.get("connections", np.array([])))
    if connections.size == 0:
        connections = np.empty((0, 2), dtype=np.int32)
    if connections.ndim == 1:
        connections = connections.reshape(1, -1)
    traces = payload.get("traces", [])
    energies = np.asarray(payload.get("energies", np.array([])), dtype=np.float32).reshape(-1)

    signatures: list[tuple[Any, ...]] = []
    n_items = max(len(traces), len(connections))
    for index in range(n_items):
        if index < len(connections):
            connection = [int(value) for value in np.asarray(connections[index]).tolist()]
        else:
            connection = [-1, -1]
        if len(connection) >= 2 and connection[0] >= 0 and connection[1] >= 0:
            connection_signature = tuple(sorted(connection[:2]))
        else:
            connection_signature = tuple(connection[:2])
        trace_signature = ()
        if include_trace and index < len(traces):
            trace_signature = _canonical_trace(traces[index])
        energy = None
        if include_energy and index < len(energies):
            energy = round(float(energies[index]), 6)
        signatures.append((connection_signature, trace_signature, energy))
    return signatures


def _edge_endpoint_signatures(payload: dict[str, Any]) -> list[tuple[int, int]]:
    """Build orientation-independent endpoint signatures for final edges."""
    connections = np.asarray(payload.get("connections", np.array([])))
    if connections.size == 0:
        return []
    if connections.ndim == 1:
        connections = connections.reshape(1, -1)
    signatures = []
    for connection in connections:
        pair = [int(value) for value in np.asarray(connection).tolist()[:2]]
        if len(pair) < 2:
            continue
        signatures.append(tuple(sorted(pair)))
    return signatures


def _edge_endpoint_pair_set(payload: dict[str, Any]) -> set[tuple[int, int]]:
    """Build the unique orientation-independent endpoint pairs in an edge payload."""
    return set(_edge_endpoint_signatures(payload))


def _normalize_candidate_connection_sources(
    raw_sources: Any,
    candidate_connection_count: int,
) -> list[str]:
    """Return normalized connection-source labels when candidate provenance is available."""
    if candidate_connection_count <= 0:
        return []
    if isinstance(raw_sources, np.ndarray):
        source_values = np.asarray(raw_sources).reshape(-1).tolist()
    elif isinstance(raw_sources, (list, tuple)):
        source_values = list(raw_sources)
    else:
        return []
    if len(source_values) != candidate_connection_count:
        return []
    allowed_sources = {"frontier", "watershed", "geodesic", "fallback"}
    normalized: list[str] = []
    for value in source_values:
        source_label = str(value).strip().lower()
        normalized.append(source_label if source_label in allowed_sources else "fallback")
    return normalized


def _candidate_endpoint_pair_details(
    payload: dict[str, Any],
) -> tuple[
    dict[int, set[tuple[int, int]]],
    dict[int, dict[str, set[tuple[int, int]]]],
    dict[str, set[tuple[int, int]]],
    dict[tuple[int, int], set[str]],
]:
    """Group candidate endpoint pairs by seed origin and provenance source."""
    connections = np.asarray(payload.get("connections", np.array([])))
    if connections.size == 0:
        return (
            {},
            {},
            {"frontier": set(), "watershed": set(), "geodesic": set(), "fallback": set()},
            {},
        )
    if connections.ndim == 1:
        connections = connections.reshape(1, -1)

    origins = np.asarray(payload.get("origin_indices", np.array([])), dtype=np.int32).reshape(-1)
    if origins.size != len(connections):
        origins = connections[:, 0].astype(np.int32, copy=False)
    connection_sources = _normalize_candidate_connection_sources(
        payload.get("connection_sources"),
        len(connections),
    )

    pairs_by_seed_origin: dict[int, set[tuple[int, int]]] = {}
    source_pairs_by_seed_origin: dict[int, dict[str, set[tuple[int, int]]]] = {}
    pairs_by_source: dict[str, set[tuple[int, int]]] = {
        "frontier": set(),
        "watershed": set(),
        "geodesic": set(),
        "fallback": set(),
    }
    pair_sources: dict[tuple[int, int], set[str]] = {}

    for index, connection in enumerate(connections):
        pair = [int(value) for value in np.asarray(connection).tolist()[:2]]
        if len(pair) < 2 or pair[0] < 0 or pair[1] < 0:
            continue
        endpoint_pair = tuple(sorted(pair))
        origin = int(origins[index]) if index < len(origins) else int(endpoint_pair[0])
        pairs_by_seed_origin.setdefault(origin, set()).add(endpoint_pair)
        if index >= len(connection_sources):
            continue
        source_label = connection_sources[index]
        pairs_by_source[source_label].add(endpoint_pair)
        pair_sources.setdefault(endpoint_pair, set()).add(source_label)
        source_pairs_by_seed_origin.setdefault(origin, {}).setdefault(source_label, set()).add(
            endpoint_pair
        )

    return pairs_by_seed_origin, source_pairs_by_seed_origin, pairs_by_source, pair_sources


def _incident_endpoint_pairs_by_vertex(
    endpoint_pairs: set[tuple[int, int]],
) -> dict[int, set[tuple[int, int]]]:
    """Group endpoint pairs by each incident vertex."""
    grouped: dict[int, set[tuple[int, int]]] = {}
    for pair in endpoint_pairs:
        start_vertex, end_vertex = (int(value) for value in pair)
        grouped.setdefault(start_vertex, set()).add(pair)
        grouped.setdefault(end_vertex, set()).add(pair)
    return grouped


def _candidate_endpoint_pairs_by_source(
    payload: dict[str, Any],
) -> dict[str, set[tuple[int, int]]]:
    """Group unique candidate endpoint pairs by their recorded source label."""
    _, _, grouped, _ = _candidate_endpoint_pair_details(payload)
    return {label: pairs for label, pairs in grouped.items() if pairs}


def _strand_signatures(payload: dict[str, Any]) -> list[tuple[int, ...]]:
    """Build orientation-independent strand signatures."""
    strands = payload.get("strands_to_vertices")
    if strands is None:
        strands = payload.get("strands", [])

    signatures: list[tuple[int, ...]] = []
    for strand in strands:
        strand_array = np.asarray(strand).reshape(-1)
        if strand_array.size == 0:
            continue
        try:
            strand_tuple = tuple(int(value) for value in strand_array.tolist())
        except (TypeError, ValueError):
            strand_tuple = tuple(str(value) for value in strand_array.tolist())
        reverse = strand_tuple[::-1]
        signatures.append(min(strand_tuple, reverse))
    return signatures
