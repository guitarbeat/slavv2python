#!/usr/bin/env python
"""Detailed Phase 2 Performance Profiling for Edges (Discovery vs Selection).

Profiles the internal hot-spots of Watershed Edge Discovery and Edge Selection
using cProfile, measuring call counts, cumulative time, and percentage breakdown.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import time
from pathlib import Path
from typing import Any

import joblib

from slavv_python.engine.state.io import load_json_dict
from slavv_python.pipeline.edges.manager import EdgeManager
from slavv_python.pipeline.edges.selection_workflow import select_and_finalize_edge_set
from slavv_python.schema.results import EnergyResult, VertexSet

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CROP_DIR = REPO_ROOT / "workspace" / "runs" / "oracle_180709_E" / "crop_M_r2024b"
DEFAULT_OUT_DIR = REPO_ROOT / "workspace" / "reports" / "phase2_profiling"


def profile_edges(
    run_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = run_dir / "02_Output" / "python_results" / "checkpoints"
    params_path = run_dir / "99_Metadata" / "validated_params.json"

    energy_path = checkpoints_dir / "checkpoint_energy.pkl"
    vertices_path = checkpoints_dir / "checkpoint_vertices.pkl"

    print(f"Loading checkpoints from {run_dir.name}...")
    energy_cp = joblib.load(energy_path)
    vertices_cp = joblib.load(vertices_path)
    params = load_json_dict(params_path) or {}

    energy_data = EnergyResult.from_dict(energy_cp)
    vertices = VertexSet.from_dict(vertices_cp)

    print(f"Energy shape: {energy_data.energy.shape}, Vertices count: {len(vertices.positions)}")

    # 1. Profile Discovery
    print("Profiling Watershed Discovery under cProfile...")
    prof_discovery = cProfile.Profile()
    t0_disc = time.perf_counter()
    prof_discovery.enable()
    candidates = EdgeManager.discover_candidates(energy_data, vertices, params)
    prof_discovery.disable()
    t1_disc = time.perf_counter()
    disc_time = t1_disc - t0_disc
    print(f"Discovery completed in {disc_time:.2f}s ({len(candidates.traces)} candidates generated)")

    s_disc = io.StringIO()
    ps_disc = pstats.Stats(prof_discovery, stream=s_disc).sort_stats("cumulative")
    ps_disc.print_stats(35)
    discovery_stats_text = s_disc.getvalue()

    # Extract top functions from discovery
    top_discovery_funcs = []
    for func, (_cc, nc, tt, ct, _) in sorted(ps_disc.stats.items(), key=lambda item: item[1][3], reverse=True)[:25]:
        filename, line, func_name = func
        top_discovery_funcs.append({
            "function": func_name,
            "filename": Path(filename).name,
            "line": line,
            "call_count": nc,
            "total_time_seconds": round(tt, 4),
            "cumulative_time_seconds": round(ct, 4),
            "pct_of_stage": round(100.0 * ct / max(disc_time, 1e-6), 2),
        })

    # 2. Profile Selection
    print("Profiling Edge Selection & Finalization under cProfile...")
    prof_selection = cProfile.Profile()
    t0_sel = time.perf_counter()
    prof_selection.enable()
    edge_set = select_and_finalize_edge_set(
        candidates,
        energy_data=energy_data,
        vertices=vertices,
        params=params,
    )
    prof_selection.disable()
    t1_sel = time.perf_counter()
    sel_time = t1_sel - t0_sel
    print(f"Selection completed in {sel_time:.2f}s ({len(edge_set.connections)} edges chosen)")

    s_sel = io.StringIO()
    ps_sel = pstats.Stats(prof_selection, stream=s_sel).sort_stats("cumulative")
    ps_sel.print_stats(25)
    selection_stats_text = s_sel.getvalue()

    # Extract top functions from selection
    top_selection_funcs = []
    for func, (_cc, nc, tt, ct, _) in sorted(ps_sel.stats.items(), key=lambda item: item[1][3], reverse=True)[:20]:
        filename, line, func_name = func
        top_selection_funcs.append({
            "function": func_name,
            "filename": Path(filename).name,
            "line": line,
            "call_count": nc,
            "total_time_seconds": round(tt, 4),
            "cumulative_time_seconds": round(ct, 4),
            "pct_of_stage": round(100.0 * ct / max(sel_time, 1e-6), 2),
        })

    # Aggregate category times in Discovery
    category_breakdown = {
        "numba_geometric_adjustments": 0.0,
        "numba_voxel_claiming": 0.0,
        "frontier_priority_queue": 0.0,
        "strel_indexing_and_bounds": 0.0,
        "candidate_trace_assembly": 0.0,
        "other": 0.0,
    }

    for item in top_discovery_funcs:
        fn = item["function"]
        tt = item["total_time_seconds"]
        if "adjusted_neighbor_energies" in fn or "directional_suppression" in fn:
            category_breakdown["numba_geometric_adjustments"] += tt
        elif "claim_unowned_strel" in fn:
            category_breakdown["numba_voxel_claiming"] += tt
        elif "heap" in fn or "pop" in fn or "push" in fn or "frontier" in fn:
            category_breakdown["frontier_priority_queue"] += tt
        elif "strel" in fn or "linear_index" in fn or "coord" in fn:
            category_breakdown["strel_indexing_and_bounds"] += tt
        elif "trace" in fn or "assemble" in fn:
            category_breakdown["candidate_trace_assembly"] += tt
        else:
            category_breakdown["other"] += tt

    report_payload = {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "volume_shape": list(energy_data.energy.shape),
        "vertex_count": len(vertices.positions),
        "candidate_count": len(candidates.traces),
        "edge_count": len(edge_set.connections),
        "spans": {
            "discovery_seconds": round(disc_time, 4),
            "selection_seconds": round(sel_time, 4),
            "total_edges_seconds": round(disc_time + sel_time, 4),
            "discovery_pct": round(100.0 * disc_time / (disc_time + sel_time), 2),
            "selection_pct": round(100.0 * sel_time / (disc_time + sel_time), 2),
        },
        "discovery_category_breakdown": {
            k: round(v, 4) for k, v in category_breakdown.items()
        },
        "top_discovery_functions": top_discovery_funcs,
        "top_selection_functions": top_selection_funcs,
    }

    json_path = out_dir / "edges_detailed_profile.json"
    json_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    print(f"Detailed profiling JSON written to {json_path}")

    txt_path = out_dir / "cprofile_discovery_stats.txt"
    txt_path.write_text(discovery_stats_text, encoding="utf-8")

    txt_sel_path = out_dir / "cprofile_selection_stats.txt"
    txt_sel_path.write_text(selection_stats_text, encoding="utf-8")

    return report_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_CROP_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    payload = profile_edges(args.run_dir, args.out_dir)
    print("\n=== Phase 2 Edges Profiling Summary ===")
    print(f"Total Edges Time: {payload['spans']['total_edges_seconds']:.2f}s")
    print(f"  - Discovery: {payload['spans']['discovery_seconds']:.2f}s ({payload['spans']['discovery_pct']}%)")
    print(f"  - Selection: {payload['spans']['selection_seconds']:.2f}s ({payload['spans']['selection_pct']}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
