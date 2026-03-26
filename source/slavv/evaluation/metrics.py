"""
Comparison metrics for SLAVV validation.

This module contains functions to compare vertices, edges, and network statistics
between MATLAB and Python implementations.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from scipy import stats
from scipy.spatial import cKDTree


def _coerce_count(value: Any) -> int | None:
    """Convert a count-like value into an integer when possible."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return 0
        if value.size == 1:
            return int(np.asarray(value).item())
        return int(value.shape[0])
    if isinstance(value, (list, tuple, set)):
        return len(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _count_items(value: Any) -> int:
    """Count rows/items for numpy arrays and sequence-like containers."""
    if value is None:
        return 0
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return 0
        if value.ndim == 0:
            return 1
        return int(value.shape[0])
    try:
        return len(value)
    except TypeError:
        return 1


def _resolve_count(explicit: Any, inferred: int) -> int:
    """Prefer explicit counts when present, otherwise fall back to inferred ones."""
    explicit_count = _coerce_count(explicit)
    if explicit_count is not None and (explicit_count > 0 or inferred == 0):
        return explicit_count
    return inferred


def _infer_vertices_count(vertices: dict[str, Any]) -> int:
    """Infer vertex count from payload structure when `count` is absent."""
    if not isinstance(vertices, dict):
        return 0
    return _resolve_count(vertices.get("count"), _count_items(vertices.get("positions")))


def _infer_edges_count(edges: dict[str, Any]) -> int:
    """Infer edge count from connections or traces when `count` is absent."""
    if not isinstance(edges, dict):
        return 0
    inferred = _count_items(edges.get("connections"))
    if inferred == 0:
        inferred = _count_items(edges.get("traces"))
    return _resolve_count(edges.get("count"), inferred)


def _infer_strand_count(network: dict[str, Any]) -> int:
    """Infer strand count from network topology when explicit counts are absent."""
    if not isinstance(network, dict):
        return 0
    return _resolve_count(network.get("strand_count"), _count_items(network.get("strands")))


def _as_position_array(positions: Any) -> np.ndarray:
    """Normalize position payloads into a 2D numpy array."""
    array = np.asarray(positions)
    if array.size == 0:
        return np.array([])
    if array.ndim == 1:
        return array.reshape(1, -1)
    return array


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


def match_vertices(
    matlab_positions: np.ndarray, python_positions: np.ndarray, distance_threshold: float = 3.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match vertices between MATLAB and Python using nearest neighbors."""
    matlab_positions = _as_position_array(matlab_positions)
    python_positions = _as_position_array(python_positions)

    if matlab_positions.size == 0 or python_positions.size == 0:
        return np.array([]), np.array([]), np.array([])

    # Use only spatial coordinates (first 3 columns)
    matlab_xyz = matlab_positions[:, :3]
    python_xyz = python_positions[:, :3]

    # Build KD-tree for fast nearest neighbor search
    tree = cKDTree(python_xyz)

    # Find nearest Python vertex for each MATLAB vertex
    distances, python_indices = tree.query(matlab_xyz)

    # Filter by distance threshold
    valid_matches = distances < distance_threshold

    matlab_indices = np.arange(len(matlab_xyz))[valid_matches]
    matched_python_indices = python_indices[valid_matches]
    matched_distances = distances[valid_matches]

    return matlab_indices, matched_python_indices, matched_distances


def compare_vertices(matlab_verts: dict[str, Any], python_verts: dict[str, Any]) -> dict[str, Any]:
    """Compare vertex information between MATLAB and Python."""
    comparison = {
        "matlab_count": _infer_vertices_count(matlab_verts),
        "python_count": _infer_vertices_count(python_verts),
        "count_difference": 0,
        "count_percent_difference": 0.0,
        "position_rmse": None,
        "matched_vertices": 0,
        "unmatched_matlab": 0,
        "unmatched_python": 0,
        "radius_correlation": None,
        "radius_stats": {},
        "exact_positions_scales_match": False,
        "exact_positions_scales_energies_match": False,
        "matlab_only_samples": [],
        "python_only_samples": [],
    }

    matlab_count = comparison["matlab_count"]
    python_count = comparison["python_count"]

    if matlab_count > 0 or python_count > 0:
        comparison["count_difference"] = abs(matlab_count - python_count)
        avg_count = (matlab_count + python_count) / 2.0
        if avg_count > 0:
            comparison["count_percent_difference"] = (
                comparison["count_difference"] / avg_count
            ) * 100.0

    # Match vertices if both have data
    matlab_positions = _as_position_array(matlab_verts.get("positions", np.array([])))
    python_positions = _as_position_array(python_verts.get("positions", np.array([])))

    if matlab_positions.size > 0 and python_positions.size > 0:
        matlab_idx, python_idx, distances = match_vertices(matlab_positions, python_positions)

        comparison["matched_vertices"] = len(matlab_idx)
        comparison["unmatched_matlab"] = matlab_count - len(matlab_idx)
        comparison["unmatched_python"] = python_count - len(python_idx)

        if len(distances) > 0:
            comparison["position_rmse"] = float(np.sqrt(np.mean(distances**2)))
            comparison["position_mean_distance"] = float(np.mean(distances))
            comparison["position_median_distance"] = float(np.median(distances))
            comparison["position_95th_percentile"] = float(np.percentile(distances, 95))

        # Compare radii for matched vertices
        matlab_radii = np.asarray(matlab_verts.get("radii", np.array([])))
        python_radii = np.asarray(python_verts.get("radii", np.array([])))

        if len(matlab_idx) > 0 and matlab_radii.size > 0 and python_radii.size > 0:
            matched_matlab_radii = matlab_radii[matlab_idx]
            matched_python_radii = python_radii[python_idx]

            # Compute correlation
            if len(matched_matlab_radii) > 1:
                pearson_r, pearson_p = stats.pearsonr(matched_matlab_radii, matched_python_radii)
                spearman_r, spearman_p = stats.spearmanr(matched_matlab_radii, matched_python_radii)

                comparison["radius_correlation"] = {
                    "pearson_r": float(pearson_r),
                    "pearson_p": float(pearson_p),
                    "spearman_r": float(spearman_r),
                    "spearman_p": float(spearman_p),
                }

            # Radius statistics
            comparison["radius_stats"] = {
                "matlab_mean": float(np.mean(matched_matlab_radii)),
                "matlab_std": float(np.std(matched_matlab_radii)),
                "python_mean": float(np.mean(matched_python_radii)),
                "python_std": float(np.std(matched_python_radii)),
                "mean_difference": float(np.mean(matched_matlab_radii - matched_python_radii)),
                "rmse": float(np.sqrt(np.mean((matched_matlab_radii - matched_python_radii) ** 2))),
            }

    coords_scales_matlab, coords_scales_energy_matlab = _vertex_signatures(matlab_verts)
    coords_scales_python, coords_scales_energy_python = _vertex_signatures(python_verts)
    coords_scales_counter_matlab = Counter(coords_scales_matlab)
    coords_scales_counter_python = Counter(coords_scales_python)
    coords_scales_energy_counter_matlab = Counter(coords_scales_energy_matlab)
    coords_scales_energy_counter_python = Counter(coords_scales_energy_python)

    comparison["exact_positions_scales_match"] = (
        coords_scales_counter_matlab == coords_scales_counter_python
    )
    comparison["exact_positions_scales_energies_match"] = (
        coords_scales_energy_counter_matlab == coords_scales_energy_counter_python
    )
    comparison["matlab_only_samples"] = _sample_counter_diff(
        coords_scales_counter_matlab, coords_scales_counter_python
    )
    comparison["python_only_samples"] = _sample_counter_diff(
        coords_scales_counter_python, coords_scales_counter_matlab
    )

    return comparison


def compare_edges(matlab_edges: dict[str, Any], python_edges: dict[str, Any]) -> dict[str, Any]:
    """Compare edge information between MATLAB and Python."""
    comparison = {
        "matlab_count": _infer_edges_count(matlab_edges),
        "python_count": _infer_edges_count(python_edges),
        "count_difference": 0,
        "count_percent_difference": 0.0,
        "total_length": {},
        "exact_match": False,
        "exact_endpoint_pairs_match": False,
        "exact_trace_match": False,
        "matlab_only_samples": [],
        "python_only_samples": [],
        "endpoint_pair_matlab_only_samples": [],
        "endpoint_pair_python_only_samples": [],
        "diagnostics": {
            "matlab": matlab_edges.get("diagnostics", {}),
            "python": python_edges.get("diagnostics", {}),
        },
    }

    matlab_count = comparison["matlab_count"]
    python_count = comparison["python_count"]

    if matlab_count > 0 or python_count > 0:
        comparison["count_difference"] = abs(matlab_count - python_count)
        avg_count = (matlab_count + python_count) / 2.0
        if avg_count > 0:
            comparison["count_percent_difference"] = (
                comparison["count_difference"] / avg_count
            ) * 100.0

    # Compare total lengths if available
    matlab_total_length = matlab_edges.get("total_length", 0.0)
    if matlab_total_length > 0:
        comparison["total_length"]["matlab"] = float(matlab_total_length)

    # Calculate Python edge lengths
    python_traces = python_edges.get("traces", [])
    if _count_items(python_traces) > 0:
        python_total_length = 0.0
        for trace in python_traces:
            trace_array = np.asarray(trace)
            if trace_array.size > 0 and trace_array.ndim == 2 and trace_array.shape[0] > 1:
                diffs = np.diff(trace_array[:, :3], axis=0)
                lengths = np.sqrt(np.sum(diffs**2, axis=1))
                python_total_length += np.sum(lengths)

        comparison["total_length"]["python"] = float(python_total_length)

        if matlab_total_length > 0 and python_total_length > 0:
            comparison["total_length"]["difference"] = float(
                abs(matlab_total_length - python_total_length)
            )
            comparison["total_length"]["percent_difference"] = float(
                (
                    comparison["total_length"]["difference"]
                    / ((matlab_total_length + python_total_length) / 2.0)
                )
                * 100.0
            )

    include_trace = (
        _count_items(matlab_edges.get("traces")) > 0
        and _count_items(python_edges.get("traces")) > 0
    )
    include_energy = (
        _count_items(matlab_edges.get("energies")) > 0
        and _count_items(python_edges.get("energies")) > 0
    )
    matlab_counter = Counter(_edge_signatures(matlab_edges, include_trace, include_energy))
    python_counter = Counter(_edge_signatures(python_edges, include_trace, include_energy))
    comparison["exact_match"] = matlab_counter == python_counter
    comparison["exact_trace_match"] = comparison["exact_match"] if include_trace else False
    comparison["matlab_only_samples"] = _sample_counter_diff(matlab_counter, python_counter)
    comparison["python_only_samples"] = _sample_counter_diff(python_counter, matlab_counter)

    matlab_endpoint_counter = Counter(_edge_endpoint_signatures(matlab_edges))
    python_endpoint_counter = Counter(_edge_endpoint_signatures(python_edges))
    comparison["exact_endpoint_pairs_match"] = matlab_endpoint_counter == python_endpoint_counter
    comparison["endpoint_pair_matlab_only_samples"] = _sample_counter_diff(
        matlab_endpoint_counter, python_endpoint_counter
    )
    comparison["endpoint_pair_python_only_samples"] = _sample_counter_diff(
        python_endpoint_counter, matlab_endpoint_counter
    )

    return comparison


def compare_networks(
    matlab_network: dict[str, Any],
    python_network: dict[str, Any],
    matlab_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare network-level statistics."""
    if (
        matlab_stats is None
        and "strands_to_vertices" not in matlab_network
        and "strands" not in matlab_network
    ):
        matlab_stats = matlab_network
        matlab_network = {}

    comparison = {
        "matlab_strand_count": _resolve_count(
            (matlab_stats or {}).get("strand_count"),
            _count_items((matlab_network or {}).get("strands_to_vertices")),
        ),
        "python_strand_count": _infer_strand_count(python_network),
        "exact_match": False,
        "matlab_only_samples": [],
        "python_only_samples": [],
    }

    matlab_count = comparison["matlab_strand_count"]
    python_count = comparison["python_strand_count"]

    if matlab_count > 0 or python_count > 0:
        comparison["strand_count_difference"] = abs(matlab_count - python_count)
        avg_count = (matlab_count + python_count) / 2.0
        if avg_count > 0:
            comparison["strand_count_percent_difference"] = (
                comparison["strand_count_difference"] / avg_count
            ) * 100.0

    matlab_counter = Counter(_strand_signatures(matlab_network or {}))
    python_counter = Counter(_strand_signatures(python_network))
    comparison["exact_match"] = matlab_counter == python_counter
    comparison["matlab_only_samples"] = _sample_counter_diff(matlab_counter, python_counter)
    comparison["python_only_samples"] = _sample_counter_diff(python_counter, matlab_counter)

    return comparison


def compare_results(
    matlab_results: dict[str, Any],
    python_results: dict[str, Any],
    matlab_parsed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Compare MATLAB and Python vectorization results.

    Args:
        matlab_results: Results from MATLAB execution (includes timing, paths, etc.)
        python_results: Results from Python execution (includes timing and processed data)
        matlab_parsed: Optional parsed MATLAB data from .mat files

    Returns:
        Comprehensive comparison dictionary
    """
    python_data = python_results.get("results") or {}
    matlab_vertices_count = _resolve_count(
        matlab_results.get("vertices_count"),
        _infer_vertices_count((matlab_parsed or {}).get("vertices", {})),
    )
    matlab_edges_count = _resolve_count(
        matlab_results.get("edges_count"),
        _infer_edges_count((matlab_parsed or {}).get("edges", {})),
    )
    matlab_strands_count = _resolve_count(
        matlab_results.get("strand_count"),
        _resolve_count(
            matlab_results.get("network_strands_count"),
            _resolve_count(
                (matlab_parsed or {}).get("network_stats", {}).get("strand_count"),
                _infer_strand_count((matlab_parsed or {}).get("network", {})),
            ),
        ),
    )
    python_vertices_count = _resolve_count(
        python_results.get("vertices_count"),
        _infer_vertices_count(python_data.get("vertices", {})),
    )
    python_edges_count = _resolve_count(
        python_results.get("edges_count"),
        _infer_edges_count(python_data.get("edges", {})),
    )
    python_strands_count = _resolve_count(
        python_results.get("network_strands_count"),
        _infer_strand_count(python_data.get("network", {})),
    )

    comparison = {
        "matlab": {
            "success": matlab_results.get("success", False),
            "elapsed_time": matlab_results.get("elapsed_time", 0.0),
            "output_dir": matlab_results.get("output_dir", ""),
            "vertices_count": matlab_vertices_count,
            "edges_count": matlab_edges_count,
            "strand_count": matlab_strands_count,
        },
        "python": {
            "success": python_results.get("success", False),
            "elapsed_time": python_results.get("elapsed_time", 0.0),
            "output_dir": python_results.get("output_dir", ""),
            "vertices_count": python_vertices_count,
            "edges_count": python_edges_count,
            "network_strands_count": python_strands_count,
            "comparison_mode": python_results.get("comparison_mode", {}),
        },
        "performance": {},
    }

    # Performance comparison
    matlab_time = matlab_results.get("elapsed_time", 0.0)
    python_time = python_results.get("elapsed_time", 0.0)

    if matlab_time > 0 and python_time > 0:
        speedup = matlab_time / python_time
        comparison["performance"] = {
            "matlab_time_seconds": matlab_time,
            "python_time_seconds": python_time,
            "speedup": speedup,
            "faster": "Python" if speedup > 1.0 else "MATLAB",
        }

    # Detailed comparison if parsed MATLAB data is available
    if matlab_parsed and python_data:
        # Compare vertices
        if "vertices" in matlab_parsed and "vertices" in python_data:
            comparison["vertices"] = compare_vertices(
                matlab_parsed["vertices"], python_data["vertices"]
            )

        # Compare edges
        if "edges" in matlab_parsed and "edges" in python_data:
            comparison["edges"] = compare_edges(matlab_parsed["edges"], python_data["edges"])

        # Compare networks
        if "network" in matlab_parsed and "network" in python_data:
            comparison["network"] = compare_networks(
                matlab_parsed["network"],
                python_data["network"],
                matlab_parsed.get("network_stats"),
            )

    comparison["parity_gate"] = {
        "vertices_exact": comparison.get("vertices", {}).get("exact_positions_scales_match", False),
        "edges_exact": comparison.get("edges", {}).get("exact_match", False),
        "strands_exact": comparison.get("network", {}).get("exact_match", False),
    }
    comparison["parity_gate"]["passed"] = all(comparison["parity_gate"].values())

    return comparison
