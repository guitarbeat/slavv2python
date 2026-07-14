"""Offline probe: replay Python edge selection funnel on the crop and localize
where MATLAB pairs (present as Python candidates) are dropped per stage.

Mirrors slavv_python/pipeline/edges/selection.py:_choose_edges_matlab_style
exactly (exact route => conflict painting disabled) but records the surviving
undirected pair set after each cleanup stage and its overlap with the MATLAB
oracle pair set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from slavv_python.analytics.parity.constants import NORMALIZED_DIR
from slavv_python.analytics.parity.oracle.surfaces import validate_exact_proof_source_surface
from slavv_python.analytics.parity.proof.coordinator import (
    load_exact_energy_result,
    load_exact_vertex_set,
)
from slavv_python.engine.state import load_json_dict
from slavv_python.pipeline.edges.cleanup import (
    break_graph_cycles,
    prune_orphan_edges,
    remove_excess_vertex_degrees,
)
from slavv_python.pipeline.edges.finalize import (
    _matlab_precrop_resample_from_maps,
    prefilter_edge_indices_for_cleanup_matlab_style,
)
from slavv_python.pipeline.edges.selection_payloads import (
    initialize_edge_selection_diagnostics,
    prepare_candidate_indices_for_cleanup,
)
from slavv_python.pipeline.energy.matlab_get_energy_v202_chunked import (
    get_chunking_lattice_v190,
    get_starts_and_counts_v200,
)
from slavv_python.utils.safe_unpickle import safe_load


def _pair_set(connections: np.ndarray) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    arr = np.asarray(connections, dtype=np.int64).reshape(-1, 2)
    for u, v in arr:
        if u < 0 or v < 0 or u == v:
            continue
        pairs.add((u, v) if u < v else (v, u))
    return pairs


def _load_oracle_pairs(oracle_root: Path) -> set[tuple[int, int]]:
    payload = safe_load(oracle_root / NORMALIZED_DIR / "oracle" / "edges.pkl")
    return _pair_set(np.asarray(payload.get("connections", np.zeros((0, 2))), dtype=np.int64))


def _degree_counts(pairs: set[tuple[int, int]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for start_vertex, end_vertex in pairs:
        counts[start_vertex] = counts.get(start_vertex, 0) + 1
        counts[end_vertex] = counts.get(end_vertex, 0) + 1
    return counts


def _matlab_chunk_eligible_candidate_indices(
    *,
    connections: np.ndarray,
    vertex_positions_zyx: np.ndarray,
    vertex_scales_zero_based: np.ndarray,
    lumen_radius_microns: np.ndarray,
    microns_per_voxel_yxz: np.ndarray,
    image_shape_zyx: tuple[int, int, int],
) -> list[int]:
    """Return candidates that could be emitted by MATLAB ``get_edges_V300`` chunks.

    MATLAB runs watershed per reading chunk, then keeps an edge only if both
    endpoints were in that chunk's reading window and at least one endpoint was
    in the chunk's writing window. Coordinates here are converted from Python's
    internal zero-based [Z, Y, X] vertices into MATLAB one-based [Y, X, Z].
    """
    size_of_image_yxz = np.asarray(
        [image_shape_zyx[1], image_shape_zyx[2], image_shape_zyx[0]], dtype=np.float64
    )
    mpv_yxz = np.asarray(microns_per_voxel_yxz, dtype=np.float64).reshape(3)
    strel_size_in_pixels = np.asarray(lumen_radius_microns, dtype=np.float64)[0] / mpv_yxz
    chunk_lattice_dimensions, _ = get_chunking_lattice_v190(
        strel_size_in_pixels,
        1e8,
        size_of_image_yxz,
    )

    (
        _,
        _,
        _,
        _,
        _,
        _,
        y_writing_starts,
        x_writing_starts,
        z_writing_starts,
        y_writing_counts,
        x_writing_counts,
        z_writing_counts,
        _,
        _,
        _,
    ) = get_starts_and_counts_v200(
        chunk_lattice_dimensions,
        np.zeros(3, dtype=np.uint16),
        size_of_image_yxz,
        np.ones(3, dtype=np.uint16),
    )

    scales_one_based = np.asarray(vertex_scales_zero_based, dtype=np.float64).reshape(-1) + 1.0
    radius_x = np.arange(1, len(lumen_radius_microns) + 1, dtype=np.float64)
    vertex_radii = np.exp(
        np.interp(scales_one_based, radius_x, np.log(np.asarray(lumen_radius_microns)))
    )
    vertex_distance_tolerance_pixels = np.floor(
        2.0 * 3.0 * vertex_radii[:, None] / mpv_yxz[None, :] + 0.5
    )
    chunk_overlap_pixels = 3.0 * np.max(vertex_distance_tolerance_pixels, axis=0)

    (
        y_reading_starts,
        x_reading_starts,
        z_reading_starts,
        y_reading_counts,
        x_reading_counts,
        z_reading_counts,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = get_starts_and_counts_v200(
        chunk_lattice_dimensions,
        chunk_overlap_pixels.astype(np.uint16),
        size_of_image_yxz,
        np.ones(3, dtype=np.uint16),
    )

    vertices_yxz = np.asarray(vertex_positions_zyx, dtype=np.float64)[:, [1, 2, 0]] + 1.0
    y, x, z = vertices_yxz[:, 0], vertices_yxz[:, 1], vertices_yxz[:, 2]
    start_vertices = connections[:, 0].astype(np.int64)
    end_vertices = connections[:, 1].astype(np.int64)
    valid_connections = (
        (start_vertices >= 0)
        & (end_vertices >= 0)
        & (start_vertices < len(vertices_yxz))
        & (end_vertices < len(vertices_yxz))
    )
    safe_start_vertices = np.where(valid_connections, start_vertices, 0)
    safe_end_vertices = np.where(valid_connections, end_vertices, 0)

    eligible = np.zeros(len(connections), dtype=bool)
    ny, nx, nz = (int(value) for value in chunk_lattice_dimensions)
    for y_chunk in range(ny):
        y_write_lo = y_writing_starts[y_chunk]
        y_write_hi = y_writing_starts[y_chunk] + y_writing_counts[y_chunk] - 1
        y_read_lo = y_reading_starts[y_chunk]
        y_read_hi = y_reading_starts[y_chunk] + y_reading_counts[y_chunk] - 1
        y_in_write = (y >= y_write_lo) & (y <= y_write_hi)
        y_in_read = (y > y_read_lo) & (y < y_read_hi)
        for x_chunk in range(nx):
            x_write_lo = x_writing_starts[x_chunk]
            x_write_hi = x_writing_starts[x_chunk] + x_writing_counts[x_chunk] - 1
            x_read_lo = x_reading_starts[x_chunk]
            x_read_hi = x_reading_starts[x_chunk] + x_reading_counts[x_chunk] - 1
            xy_in_write = y_in_write & (x >= x_write_lo) & (x <= x_write_hi)
            xy_in_read = y_in_read & (x > x_read_lo) & (x < x_read_hi)
            for z_chunk in range(nz):
                z_write_lo = z_writing_starts[z_chunk]
                z_write_hi = z_writing_starts[z_chunk] + z_writing_counts[z_chunk] - 1
                z_read_lo = z_reading_starts[z_chunk]
                z_read_hi = z_reading_starts[z_chunk] + z_reading_counts[z_chunk] - 1
                in_write = xy_in_write & (z >= z_write_lo) & (z <= z_write_hi)
                in_read = xy_in_read & (z > z_read_lo) & (z < z_read_hi)
                pair_in_read = in_read[safe_start_vertices] & in_read[safe_end_vertices]
                pair_in_write = in_write[safe_start_vertices] | in_write[safe_end_vertices]
                eligible |= valid_connections & pair_in_read & pair_in_write

    return np.flatnonzero(eligible).astype(np.int32).tolist()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--oracle-root", type=Path, required=True)
    ap.add_argument("--sample-size", type=int, default=20)
    ap.add_argument(
        "--apply-matlab-chunk-eligibility",
        action="store_true",
        help=(
            "Diagnostic only: prefilter candidates to edges that MATLAB get_edges_V300 "
            "could emit from at least one reading/writing chunk."
        ),
    )
    args = ap.parse_args(argv)

    source = validate_exact_proof_source_surface(args.run_dir)
    energy = load_exact_energy_result(source)
    vertices = load_exact_vertex_set(source, energy)
    params = load_json_dict(source.validated_params_path) or {}

    cands = safe_load(args.run_dir / "04_Edges" / "candidates.pkl")
    connections = np.asarray(cands["connections"], dtype=np.int32).reshape(-1, 2)
    traces = cands["traces"]
    metrics = np.asarray(cands["metrics"], dtype=np.float32).reshape(-1)
    energy_traces = cands["energy_traces"]
    scale_traces = cands["scale_traces"]
    origin_indices = np.asarray(cands.get("origin_indices", []), dtype=np.int32).reshape(-1)
    matlab_global = bool(cands.get("matlab_global_watershed_exact", False))
    reject_nonneg = not matlab_global

    shape = energy.energy.shape
    microns_per_voxel = np.asarray(
        params.get("microns_per_voxel", [1.0, 1.0, 1.0]), dtype=np.float32
    )
    lumen_radius_microns = np.asarray(energy.lumen_radius_microns, dtype=np.float32)

    matlab_pairs = _load_oracle_pairs(args.oracle_root)
    python_cand_pairs = _pair_set(connections)
    matlab_candidate_pairs = matlab_pairs & python_cand_pairs
    pair_to_candidate_indices: dict[tuple[int, int], list[int]] = {}
    for index, (start_vertex, end_vertex) in enumerate(connections):
        if start_vertex < 0 or end_vertex < 0 or start_vertex == end_vertex:
            continue
        pair = (
            (int(start_vertex), int(end_vertex))
            if int(start_vertex) < int(end_vertex)
            else (int(end_vertex), int(start_vertex))
        )
        pair_to_candidate_indices.setdefault(pair, []).append(index)

    diagnostics = initialize_edge_selection_diagnostics(cands, connections, traces)
    selection_traces = traces
    selection_energy_traces = energy_traces
    selection_metrics = metrics
    resampled_spaces_yxz_one_based, _, resampled_energies = (
        _matlab_precrop_resample_from_maps(
            [np.asarray(trace, dtype=np.float64) for trace in traces],
            [np.asarray(trace, dtype=np.float64) for trace in scale_traces],
            [np.asarray(trace, dtype=np.float64) for trace in energy_traces],
            energy_map=energy.energy,
            scale_indices=energy.scale_indices,
        )
    )
    selection_traces = [
        np.asarray(space_trace[:, [2, 0, 1]] - 1.0, dtype=np.float64)
        for space_trace in resampled_spaces_yxz_one_based
    ]
    selection_energy_traces = resampled_energies
    selection_metrics = np.asarray(
        [
            np.max(energy_trace) if len(energy_trace) else -1000.0
            for energy_trace in selection_energy_traces
        ],
        dtype=np.float64,
    )
    selection_metrics[np.isnan(selection_metrics)] = -1000.0
    previous_pairs: set[tuple[int, int]] | None = None
    previous_indices: list[int] | None = None
    displacement_summaries: dict[str, dict[str, object]] = {}
    stage_indices_by_label: dict[str, list[int]] = {}

    def _pair_for_index(index: int) -> tuple[int, int] | None:
        start_vertex, end_vertex = (int(connections[index, 0]), int(connections[index, 1]))
        if start_vertex < 0 or end_vertex < 0 or start_vertex == end_vertex:
            return None
        return (
            (start_vertex, end_vertex) if start_vertex < end_vertex else (end_vertex, start_vertex)
        )

    def _index_detail(index: int, row_index: int | None) -> dict[str, object]:
        return {
            "candidate_index": int(index),
            "row_index": None if row_index is None else int(row_index),
            "trace_len": len(selection_traces[index]),
            "metric": float(selection_metrics[index]),
            "connection": [int(connections[index, 0]), int(connections[index, 1])],
        }

    def _same_vertex_surviving_extras(
        pair: tuple[int, int],
        surviving_indices: list[int],
        *,
        limit: int = 8,
    ) -> list[dict[str, object]]:
        details: list[dict[str, object]] = []
        pair_vertices = set(pair)
        for row_index, index in enumerate(surviving_indices):
            candidate_pair = _pair_for_index(index)
            if candidate_pair is None or candidate_pair in matlab_pairs:
                continue
            if pair_vertices.isdisjoint(candidate_pair):
                continue
            details.append(_index_detail(index, row_index))
            if len(details) >= limit:
                break
        return details

    def _displacement_summary(
        lost_pairs: list[tuple[int, int]],
        surviving_indices: list[int],
        previous_row_lookup: dict[int, int],
    ) -> dict[str, object]:
        """Aggregate whether surviving extras are positioned to displace lost MATLAB pairs."""
        surviving_extra_details: list[tuple[int, tuple[int, int], int, float]] = []
        for row_index, index in enumerate(surviving_indices):
            candidate_pair = _pair_for_index(index)
            if candidate_pair is None or candidate_pair in matlab_pairs:
                continue
            surviving_extra_details.append(
                (index, candidate_pair, row_index, float(selection_metrics[index]))
            )

        lost_with_incident_extra = 0
        lost_with_earlier_incident_extra = 0
        lost_with_better_metric_incident_extra = 0
        incident_extra_counts: list[int] = []
        sample_pairs: list[dict[str, object]] = []

        for pair in lost_pairs:
            pair_vertices = set(pair)
            lost_candidate_indices = pair_to_candidate_indices.get(pair, [])
            lost_rows = [
                previous_row_lookup[index]
                for index in lost_candidate_indices
                if index in previous_row_lookup
            ]
            lost_metrics = [float(selection_metrics[index]) for index in lost_candidate_indices]
            incident_extras = [
                (index, extra_pair, row_index, metric)
                for index, extra_pair, row_index, metric in surviving_extra_details
                if not pair_vertices.isdisjoint(extra_pair)
            ]
            incident_extra_counts.append(len(incident_extras))
            if not incident_extras:
                continue

            lost_with_incident_extra += 1
            if lost_rows and any(row_index < min(lost_rows) for _, _, row_index, _ in incident_extras):
                lost_with_earlier_incident_extra += 1
            if lost_metrics and any(metric < min(lost_metrics) for _, _, _, metric in incident_extras):
                lost_with_better_metric_incident_extra += 1
            if len(sample_pairs) < max(0, args.sample_size):
                sample_pairs.append(
                    {
                        "lost_pair": [int(pair[0]), int(pair[1])],
                        "lost_rows": [int(value) for value in lost_rows],
                        "lost_metrics": lost_metrics,
                        "incident_extra_count": len(incident_extras),
                        "incident_extra_sample": [
                            {
                                "candidate_index": int(index),
                                "row_index": int(row_index),
                                "pair": [int(extra_pair[0]), int(extra_pair[1])],
                                "metric": metric,
                            }
                            for index, extra_pair, row_index, metric in incident_extras[:5]
                        ],
                    }
                )

        return {
            "lost_matlab_pair_count": len(lost_pairs),
            "lost_with_incident_surviving_extra": lost_with_incident_extra,
            "lost_with_earlier_incident_surviving_extra": lost_with_earlier_incident_extra,
            "lost_with_better_metric_incident_extra": lost_with_better_metric_incident_extra,
            "total_incident_surviving_extra_links": int(sum(incident_extra_counts)),
            "max_incident_surviving_extras_for_lost_pair": int(max(incident_extra_counts, default=0)),
            "sample": sample_pairs,
        }

    def _extra_surface_summary(final_indices: list[int]) -> dict[str, object]:
        final_pairs = _pair_set(connections[final_indices])
        candidate_extra_pairs = python_cand_pairs - matlab_pairs
        final_extra_pairs = final_pairs - matlab_pairs
        final_missing_pairs = matlab_pairs - final_pairs
        matlab_degree = _degree_counts(matlab_pairs)
        candidate_degree = _degree_counts(python_cand_pairs)
        final_degree = _degree_counts(final_pairs)

        pair_to_indices: dict[tuple[int, int], list[int]] = {}
        for index, (start_vertex, end_vertex) in enumerate(connections):
            if start_vertex < 0 or end_vertex < 0 or start_vertex == end_vertex:
                continue
            pair = (
                (int(start_vertex), int(end_vertex))
                if int(start_vertex) < int(end_vertex)
                else (int(end_vertex), int(start_vertex))
            )
            pair_to_indices.setdefault(pair, []).append(index)

        def origin_for_index(index: int) -> int | None:
            if 0 <= index < len(origin_indices):
                return int(origin_indices[index])
            return None

        def top_origins_for_pairs(pairs: set[tuple[int, int]], limit: int = 12) -> list[dict[str, object]]:
            counts: dict[int, int] = {}
            for pair in pairs:
                for index in pair_to_indices.get(pair, []):
                    origin = origin_for_index(index)
                    if origin is None:
                        continue
                    counts[origin] = counts.get(origin, 0) + 1
            rows = []
            for origin, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[
                :limit
            ]:
                position = (
                    vertices.positions[origin].astype(float).tolist()
                    if 0 <= origin < len(vertices.positions)
                    else None
                )
                boundary_distance = None
                if position is not None:
                    coords = np.asarray(position, dtype=np.float64)
                    shape_arr = np.asarray(shape, dtype=np.float64) - 1.0
                    boundary_distance = float(np.min(np.minimum(coords, shape_arr - coords)))
                rows.append(
                    {
                        "origin": int(origin),
                        "extra_pair_count": int(count),
                        "matlab_final_degree": int(matlab_degree.get(origin, 0)),
                        "python_candidate_degree": int(candidate_degree.get(origin, 0)),
                        "python_final_degree": int(final_degree.get(origin, 0)),
                        "position_zyx": position,
                        "boundary_distance": boundary_distance,
                    }
                )
            return rows

        missing_vertices = {vertex for pair in final_missing_pairs for vertex in pair}
        extra_vertices = {vertex for pair in final_extra_pairs for vertex in pair}
        over_degree_vertices = [
            vertex
            for vertex, degree in final_degree.items()
            if degree > matlab_degree.get(vertex, 0)
        ]
        top_over_degree_vertices = [
            {
                "vertex": int(vertex),
                "python_final_degree": int(final_degree.get(vertex, 0)),
                "matlab_final_degree": int(matlab_degree.get(vertex, 0)),
                "final_extra_incident": int(
                    sum(1 for pair in final_extra_pairs if vertex in pair)
                ),
                "missing_incident": int(
                    sum(1 for pair in final_missing_pairs if vertex in pair)
                ),
                "position_zyx": vertices.positions[vertex].astype(float).tolist()
                if 0 <= vertex < len(vertices.positions)
                else None,
            }
            for vertex in sorted(
                over_degree_vertices,
                key=lambda item: (
                    -(final_degree.get(item, 0) - matlab_degree.get(item, 0)),
                    item,
                ),
            )[:12]
        ]

        return {
            "candidate_extra_pair_count": len(candidate_extra_pairs),
            "final_extra_pair_count": len(final_extra_pairs),
            "final_missing_pair_count": len(final_missing_pairs),
            "vertices_with_missing_and_extra": len(missing_vertices & extra_vertices),
            "vertices_over_matlab_final_degree": len(over_degree_vertices),
            "top_candidate_extra_origins": top_origins_for_pairs(candidate_extra_pairs),
            "top_final_extra_origins": top_origins_for_pairs(final_extra_pairs),
            "top_over_degree_vertices": top_over_degree_vertices,
        }

    def report(label: str, idx_list: list[int]) -> set[tuple[int, int]]:
        nonlocal previous_indices, previous_pairs
        stage_indices_by_label[label] = list(idx_list)
        surv = _pair_set(connections[idx_list])
        ov_all = surv & matlab_pairs
        ov_gen = surv & matlab_candidate_pairs
        print(
            f"{label:34s} n_edges={len(idx_list):6d} "
            f"pairs={len(surv):6d} ov_matlab={len(ov_all):6d} "
            f"ov_gen={len(ov_gen):6d}"
        )
        if previous_pairs is not None:
            lost_from_previous = sorted((previous_pairs & matlab_pairs) - surv)
            if lost_from_previous:
                print(
                    "   lost_from_previous_sample:",
                    json.dumps(
                        [[int(start), int(end)] for start, end in lost_from_previous][
                            : max(0, args.sample_size)
                        ]
                    ),
                )
                if label in {
                    "1. after crop",
                    "4. after degree-excess",
                    "6. FINAL",
                }:
                    previous_row_lookup: dict[int, int] = {
                        index: row_index
                        for row_index, index in enumerate(previous_indices or [])
                    }
                    displacement_summaries[label] = _displacement_summary(
                        lost_from_previous,
                        idx_list,
                        previous_row_lookup,
                    )
                    print(
                        "   displacement_summary:",
                        json.dumps(displacement_summaries[label]),
                    )
                    lost_detail_samples = []
                    for pair in lost_from_previous[: max(0, args.sample_size)]:
                        previous_candidate_details = [
                            _index_detail(index, previous_row_lookup.get(index))
                            for index in pair_to_candidate_indices.get(pair, [])[:5]
                        ]
                        lost_detail_samples.append(
                            {
                                "pair": [int(start) for start in pair],
                                "previous_candidates": previous_candidate_details,
                                "surviving_same_vertex_extras": _same_vertex_surviving_extras(
                                    pair, idx_list
                                ),
                            }
                        )
                    print("   lost_pair_rank_detail_sample:", json.dumps(lost_detail_samples))
        extra_vs_matlab = sorted(surv - matlab_pairs)
        if extra_vs_matlab:
            print(
                "   extra_vs_matlab_sample:",
                json.dumps(
                    [[int(start), int(end)] for start, end in extra_vs_matlab][
                        : max(0, args.sample_size)
                    ]
                ),
            )
        previous_pairs = surv
        previous_indices = list(idx_list)
        return surv

    print(f"MATLAB total pairs         : {len(matlab_pairs)}")
    print(f"Python candidate pairs     : {len(python_cand_pairs)}")
    print(f"MATLAB candidate overlap   : {len(matlab_candidate_pairs)}")
    print("-" * 92)

    active_indices = list(range(len(connections)))
    report("0. candidates", active_indices)

    if args.apply_matlab_chunk_eligibility:
        active_indices = _matlab_chunk_eligible_candidate_indices(
            connections=connections,
            vertex_positions_zyx=vertices.positions,
            vertex_scales_zero_based=vertices.scales,
            lumen_radius_microns=lumen_radius_microns,
            microns_per_voxel_yxz=microns_per_voxel,
            image_shape_zyx=shape,
        )
        print(f"   [chunk eligibility dropped {len(connections) - len(active_indices)}]")
        report("0b. after chunk eligibility", active_indices)

    kept_after_crop, cropped = prefilter_edge_indices_for_cleanup_matlab_style(
        active_indices,
        traces,
        scale_traces,
        energy_traces,
        lumen_radius_microns=lumen_radius_microns,
        microns_per_voxel=microns_per_voxel,
        size_of_image=shape,
        energy_map=energy.energy,
        scale_indices=energy.scale_indices,
    )
    print(f"   [crop dropped {cropped}]")
    report("1. after crop", kept_after_crop)

    filtered = prepare_candidate_indices_for_cleanup(
        connections,
        selection_metrics,
        selection_energy_traces,
        diagnostics,
        subset_indices=kept_after_crop,
        reject_nonnegative_energy_edges=reject_nonneg,
    )
    report("2. after dedup/choose-best", filtered)

    # exact route: no conflict painting -> all filtered chosen
    chosen = filtered
    report("3. chosen (no conflict paint)", chosen)

    keep_degree = remove_excess_vertex_degrees(
        connections[chosen],
        selection_metrics[chosen],
        max(1, int(params.get("number_of_edges_per_vertex", 4))),
    )
    print(f"   [degree pruned {int(np.sum(~keep_degree))}]")
    after_degree = [i for keep, i in zip(keep_degree.tolist(), chosen) if keep]
    report("4. after degree-excess", after_degree)

    keep_orphans = prune_orphan_edges(
        [selection_traces[i] for i in after_degree],
        shape,
        vertices.positions,
    )
    print(f"   [orphan pruned {int(np.sum(~keep_orphans))}]")
    after_orphan = [i for keep, i in zip(keep_orphans.tolist(), after_degree) if keep]
    report("5. after orphan", after_orphan)

    keep_cycles = break_graph_cycles(connections[after_orphan])
    print(f"   [cycle pruned {int(np.sum(~keep_cycles))}]")
    final = [i for keep, i in zip(keep_cycles.tolist(), after_orphan) if keep]
    report("6. FINAL", final)

    print("-" * 92)
    print("extra surface summary:", json.dumps(_extra_surface_summary(final), default=str))
    print("displacement summaries:", json.dumps(displacement_summaries, default=str))
    print("diag counters:", json.dumps(diagnostics, default=str)[:800])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
