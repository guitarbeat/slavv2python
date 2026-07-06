"""
Edge Candidate Generation Engine.

This module provides the primary entrypoints for discovering potential vascular
connections (candidates) before they are validated and filtered into a final network.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from joblib import Parallel, delayed

from .matlab_get_edges_by_watershed import (
    _generate_edge_candidates_matlab_global_watershed as execute_watershed_engine,
)
from .payloads import (
    _edge_metric_from_energy_trace,
    _empty_edge_diagnostics,
    _merge_edge_diagnostics,
    _record_trace_diagnostics,
)
from .radius_utils import _scalar_radius
from .trace_directions import estimate_vessel_directions, generate_edge_directions
from .trace_metrics import _trace_energy_series, _trace_scale_series
from .tracing import trace_edge

if TYPE_CHECKING:
    from scipy.spatial import cKDTree

    from .edge_types import Float32Array, Int16Array, Int32Array
    from .tracing import TraceMetadata

logger = logging.getLogger(__name__)

# --- SPATIAL ALIGNMENT HELPERS ---


# --- CORE GENERATION STRATEGIES ---


def generate_watershed_candidates(
    energy: Float32Array,
    scale_indices: Int16Array | None,
    vertex_positions: Float32Array,
    vertex_scales: Int32Array,
    lumen_radius_microns: Float32Array,
    microns_per_voxel: Float32Array,
    vertex_center_image: Float32Array,
    params: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Generates candidates using a global shared-state watershed search.

    This is the high-precision route that perfectly mirrors MATLAB's
    deterministic frontier insertion.
    """
    # The watershed engine performs its OWN [Z, Y, X] -> [Y, X, Z] reorientation
    # internally (np.transpose(energy, (1, 2, 0))) and permutes microns_per_voxel
    # with [[1, 2, 0]]. We must therefore pass raw physical-order inputs straight
    # through. Pre-aligning here previously caused a DOUBLE transpose ([Z,Y,X] ->
    # [Y,X,Z] -> [X,Z,Y]), which scrambled the spatial grid: the watershed's
    # vertex_index_map agreed with MATLAB on only ~1% of voxels under the implied
    # Y<->X swap, vs ~63% with the correct single transpose. (The double-transpose's
    # higher raw edge-PAIR overlap was a coincidental graph artifact, not spatial
    # parity.) The engine returns traces already in physical [Z, Y, X] order, so no
    # post-hoc restore is applied.
    #
    # microns_per_voxel arrives in MATLAB axis order [dy, dx, dz]; the engine expects
    # physical [dz, dy, dx] (it re-permutes to [dy, dx, dz] with [[1, 2, 0]]).
    mpv_engine: np.ndarray = np.asarray(microns_per_voxel, dtype=np.float64).copy()
    if len(mpv_engine) >= 3:
        mpv_engine = mpv_engine[[2, 0, 1]]

    candidates = execute_watershed_engine(
        cast("Float32Array", energy),
        cast("Int16Array", scale_indices),
        cast("Float32Array", vertex_positions),
        vertex_scales,
        lumen_radius_microns,
        mpv_engine,
        cast("np.ndarray", vertex_center_image),
        params,
        **kwargs,
    )

    return cast("dict[str, Any]", candidates)


def generate_directional_candidates(
    energy: Float32Array,
    scale_indices: Int16Array | None,
    vertex_positions: Float32Array,
    vertex_scales: Int32Array,
    lumen_radius_pixels: Float32Array,
    lumen_radius_microns: Float32Array,
    microns_per_voxel: Float32Array,
    tree: cKDTree,
    params: dict[str, Any],
    energy_sign: float = -1.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Generates candidates using multi-threaded parallel tracing from origin seeds.

    This is the standard performance-oriented route for public workflows.
    """
    n_jobs = int(params.get("n_jobs", 1))

    # Energy prepared once for shared memory
    energy_prepared = np.ascontiguousarray(energy, dtype=np.float64)
    mpv_prepared = np.asarray(microns_per_voxel, dtype=np.float64)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_trace_vertex_unit)(
            vertex_idx=v_idx,
            start_pos=pos,
            start_scale=int(scale),
            energy=energy_prepared,
            scale_indices=scale_indices,
            vertex_positions=vertex_positions,
            vertex_scales=vertex_scales,
            radii_pixels=lumen_radius_pixels,
            radii_microns=lumen_radius_microns,
            mpv=mpv_prepared,
            tree=tree,
            params=params,
            sign=energy_sign,
            energy_prepared=energy_prepared,
            mpv_prepared=mpv_prepared,
            **kwargs,
        )
        for v_idx, (pos, scale) in enumerate(zip(vertex_positions, vertex_scales))
    )

    return _assemble_parallel_results(results)


# --- INTERNAL EXECUTION UNITS ---


def _trace_vertex_unit(
    vertex_idx: int,
    start_pos: Float32Array,
    start_scale: int,
    energy: Float32Array,
    scale_indices: Int16Array | None,
    vertex_positions: Float32Array,
    vertex_scales: Int32Array,
    radii_pixels: Float32Array,
    radii_microns: Float32Array,
    mpv: Float32Array,
    tree: cKDTree,
    params: dict[str, Any],
    sign: float,
    energy_prepared: np.ndarray,
    mpv_prepared: np.ndarray,
    **kwargs: Any,
) -> tuple:
    """Atomic unit of work for tracing one vertex; designed for thread-safety."""
    start_radius = _scalar_radius(radii_pixels[start_scale])

    # Configuration
    step_size_ratio = params.get("step_size_per_origin_radius", 1.0)
    max_length_ratio = params.get("max_edge_length_per_origin_radius", 60.0)
    max_edge_energy = params.get("max_edge_energy", 0.0)
    discrete_tracing = params.get("discrete_tracing", False)

    # Extract max_search_radius from kwargs if present, otherwise default to 5.0
    # We remove it from kwargs to avoid "multiple values for keyword argument" error
    # when calling facade.trace_edge below.
    local_kwargs = dict(kwargs)
    max_search_radius = local_kwargs.pop("max_search_radius", 5.0)

    step_size = start_radius * step_size_ratio
    max_length = start_radius * max_length_ratio
    max_steps = max(1, int(np.ceil(max_length / max(step_size, 1e-12))))

    # 1. Determine Tracing Vectors
    directions = _get_seed_directions(
        vertex_idx, start_pos, start_radius, energy, mpv, params
    )

    # 2. Trace Paths
    traces: list[np.ndarray] = []
    connections: list[list[int]] = []
    metrics: list[float] = []
    energy_traces: list[np.ndarray] = []
    scale_traces: list[np.ndarray] = []
    origin_indices: list[int] = []
    connection_sources: list[str] = []
    unit_diagnostics = _empty_edge_diagnostics()

    for direction in directions:
        trace_result = trace_edge(
            energy_prepared,
            start_pos,
            direction,
            step_size,
            max_edge_energy,
            vertex_positions,
            vertex_scales,
            radii_pixels,
            radii_microns,
            max_steps,
            mpv_prepared,
            sign,
            discrete_steps=discrete_tracing,
            tree=tree,
            max_search_radius=max_search_radius,
            origin_vertex_idx=vertex_idx,
            return_metadata=True,
            **local_kwargs,
        )
        edge_trace, trace_metadata = cast("tuple[np.ndarray, TraceMetadata]", trace_result)

        if len(edge_trace) <= 1:
            continue

        edge_arr = np.asarray(edge_trace, dtype=np.float64)
        terminal_vertex = trace_metadata["terminal_vertex"]

        e_series = _trace_energy_series(edge_arr, energy)
        s_series = _trace_scale_series(edge_arr, scale_indices)
        _record_trace_diagnostics(unit_diagnostics, trace_metadata)

        traces.append(edge_arr)
        connections.append([vertex_idx, terminal_vertex if terminal_vertex is not None else -1])
        metrics.append(_edge_metric_from_energy_trace(e_series))
        energy_traces.append(e_series)
        scale_traces.append(s_series)
        origin_indices.append(vertex_idx)
        connection_sources.append("fallback")

    return (
        traces,
        connections,
        metrics,
        energy_traces,
        scale_traces,
        origin_indices,
        connection_sources,
        unit_diagnostics,
    )


def _get_seed_directions(
    v_idx: int,
    pos: np.ndarray,
    r: float,
    energy: np.ndarray,
    mpv: np.ndarray,
    params: dict[str, Any],
) -> np.ndarray:
    """Decides seed directions based on Hessian response or random distribution."""
    method = params.get("direction_method", "hessian")
    limit = params.get("number_of_edges_per_vertex", 4)

    if method == "hessian":
        dirs = estimate_vessel_directions(energy, pos, r, mpv, generate_edge_directions)
        if len(dirs) < limit:
            extra = generate_edge_directions(limit - len(dirs), seed=v_idx)
            dirs = np.vstack([dirs, extra])
        return cast("np.ndarray", dirs[:limit])

    return cast("np.ndarray", generate_edge_directions(limit, seed=v_idx))


def _assemble_parallel_results(results: list[tuple]) -> dict[str, Any]:
    """Flattens parallel job outputs into a structured candidate dictionary."""
    candidates: dict[str, Any] = {
        "traces": [],
        "connections": [],
        "metrics": [],
        "energy_traces": [],
        "scale_traces": [],
        "origin_indices": [],
        "candidate_source": "fallback",
        "diagnostics": _empty_edge_diagnostics(),
    }

    for t, c, m, et, st, oi, _cs, diag in results:
        candidates["traces"].extend(t)
        candidates["connections"].extend(c)
        candidates["metrics"].extend(m)
        candidates["energy_traces"].extend(et)
        candidates["scale_traces"].extend(st)
        candidates["origin_indices"].extend(oi)
        _merge_edge_diagnostics(candidates["diagnostics"], diag)

    candidates["connections"] = np.asarray(candidates["connections"], dtype=np.int32).reshape(-1, 2)
    candidates["metrics"] = np.asarray(candidates["metrics"], dtype=np.float64)
    candidates["origin_indices"] = np.asarray(candidates["origin_indices"], dtype=np.int32)

    return candidates


def sort_candidates_by_quality(
    candidates: dict[str, Any] | Any,
    energy: np.ndarray | None = None,
    scale_indices: np.ndarray | None = None,
    vertex_positions: np.ndarray | None = None,
    sign: float = -1.0,
    params: dict[str, Any] | None = None,
    microns_per_voxel: np.ndarray | None = None,
    **kwargs: Any,
) -> dict[str, Any] | Any:
    """Ranks candidates by energy metric (ascending) to align with MATLAB priority."""
    from slavv_python.pipeline.edges.discovery import CandidateManifest

    del energy, scale_indices, vertex_positions, sign, params, microns_per_voxel, kwargs
    metrics = np.asarray(
        candidates.metrics if isinstance(candidates, CandidateManifest) else candidates.get("metrics", []),
        dtype=np.float64,
    )
    if metrics.size == 0:
        return candidates

    sort_idx = np.argsort(metrics, kind="stable")
    if isinstance(candidates, CandidateManifest):
        return candidates.reordered(sort_idx)

    from slavv_python.pipeline.edges.candidate_payload import _reorder_candidate_payload

    return _reorder_candidate_payload(candidates, sort_idx)
