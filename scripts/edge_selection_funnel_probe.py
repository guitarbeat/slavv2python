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
    prefilter_edge_indices_for_cleanup_matlab_style,
)
from slavv_python.pipeline.edges.candidate_manifest import normalize_candidate_connection_sources
from slavv_python.pipeline.edges.selection_payloads import (
    initialize_edge_selection_diagnostics,
    prepare_candidate_indices_for_cleanup,
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--oracle-root", type=Path, required=True)
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
    connection_sources = normalize_candidate_connection_sources(
        cands.get("connection_sources"), len(connections)
    )
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

    diagnostics = initialize_edge_selection_diagnostics(cands, connections, traces)

    def report(label: str, idx_list: list[int]) -> None:
        surv = _pair_set(connections[idx_list])
        ov_all = surv & matlab_pairs
        ov_gen = surv & matlab_candidate_pairs
        print(
            f"{label:34s} n_edges={len(idx_list):6d} "
            f"pairs={len(surv):6d} ov_matlab={len(ov_all):6d} "
            f"ov_gen={len(ov_gen):6d}"
        )

    print(f"MATLAB total pairs         : {len(matlab_pairs)}")
    print(f"Python candidate pairs     : {len(python_cand_pairs)}")
    print(f"MATLAB candidate overlap   : {len(matlab_candidate_pairs)}")
    print("-" * 92)

    report("0. candidates", list(range(len(connections))))

    kept_after_crop, cropped = prefilter_edge_indices_for_cleanup_matlab_style(
        list(range(len(traces))),
        traces,
        scale_traces,
        energy_traces,
        lumen_radius_microns=lumen_radius_microns,
        microns_per_voxel=microns_per_voxel,
        size_of_image=shape,
    )
    print(f"   [crop dropped {cropped}]")
    report("1. after crop", kept_after_crop)

    filtered = prepare_candidate_indices_for_cleanup(
        connections,
        metrics,
        energy_traces,
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
        metrics[chosen],
        max(1, int(params.get("number_of_edges_per_vertex", 4))),
    )
    print(f"   [degree pruned {int(np.sum(~keep_degree))}]")
    after_degree = [i for keep, i in zip(keep_degree.tolist(), chosen) if keep]
    report("4. after degree-excess", after_degree)

    keep_orphans = prune_orphan_edges(
        [traces[i] for i in after_degree], shape, vertices.positions
    )
    print(f"   [orphan pruned {int(np.sum(~keep_orphans))}]")
    after_orphan = [i for keep, i in zip(keep_orphans.tolist(), after_degree) if keep]
    report("5. after orphan", after_orphan)

    keep_cycles = break_graph_cycles(connections[after_orphan])
    print(f"   [cycle pruned {int(np.sum(~keep_cycles))}]")
    final = [i for keep, i in zip(keep_cycles.tolist(), after_orphan) if keep]
    report("6. FINAL", final)

    print("-" * 92)
    print("diag counters:", json.dumps(diagnostics, default=str)[:800])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
