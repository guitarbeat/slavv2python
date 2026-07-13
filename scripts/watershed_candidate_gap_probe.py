"""Offline probe: compare MATLAB oracle edge pairs against Python watershed candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from slavv_python.analytics.parity.constants import NORMALIZED_DIR
from slavv_python.analytics.parity.oracle.surfaces import validate_exact_proof_source_surface
from slavv_python.analytics.parity.proof.coordinator import (
    load_exact_energy_result,
    load_exact_vertex_set,
)
from slavv_python.engine.state import load_json_dict
from slavv_python.pipeline.edges.candidate_generation import generate_watershed_candidates
from slavv_python.pipeline.edges.execution_tracing import JsonExecutionTracer
from slavv_python.pipeline.vertices.painting import paint_vertex_center_image
from slavv_python.utils.safe_unpickle import safe_load


def _regenerate_watershed_candidates(
    *,
    run_dir: Path,
    params: dict[str, Any],
    tracer: JsonExecutionTracer | None = None,
) -> dict[str, Any]:
    """Regenerate candidates using the same production path as ``EdgeManager``."""
    source_surface = validate_exact_proof_source_surface(run_dir)
    energy = load_exact_energy_result(source_surface)
    vertices = load_exact_vertex_set(source_surface, energy)
    run_params = dict(params)
    run_params.setdefault("comparison_exact_network", True)
    vertex_center_image = paint_vertex_center_image(vertices.positions, energy.energy.shape)
    microns_per_voxel = np.asarray(
        run_params.get("microns_per_voxel", [1.0, 1.0, 1.0]),
        dtype=np.float64,
    )
    kwargs: dict[str, Any] = {}
    if tracer is not None:
        kwargs["tracer"] = tracer
    return generate_watershed_candidates(
        energy.energy,
        energy.scale_indices,
        vertices.positions,
        vertices.scales,
        energy.lumen_radius_microns,
        microns_per_voxel,
        vertex_center_image,
        run_params,
        **kwargs,
    )


def _endpoint_pair_set(connections: np.ndarray) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for start_vertex, end_vertex in np.asarray(connections, dtype=np.int64).reshape(-1, 2):
        u, v = int(start_vertex), int(end_vertex)
        if u < 0 or v < 0:
            continue
        pairs.add((u, v) if u < v else (v, u))
    return pairs


def _load_oracle_edge_connections(oracle_root: Path) -> np.ndarray:
    oracle_edges_path = oracle_root / NORMALIZED_DIR / "oracle" / "edges.pkl"
    if not oracle_edges_path.is_file():
        raise FileNotFoundError(f"Missing normalized oracle edges artifact: {oracle_edges_path}")
    payload = joblib.load(oracle_edges_path)
    return np.asarray(payload.get("connections", np.zeros((0, 2))), dtype=np.int64)


def _load_python_candidate_connections(run_dir: Path) -> np.ndarray:
    candidates_path = run_dir / "04_Edges" / "candidates.pkl"
    if not candidates_path.is_file():
        raise FileNotFoundError(f"Missing candidate checkpoint: {candidates_path}")
    payload = safe_load(candidates_path)
    return np.asarray(payload.get("connections", np.zeros((0, 2))), dtype=np.int64)


def _summarize_gap(
    matlab_pairs: set[tuple[int, int]],
    python_pairs: set[tuple[int, int]],
) -> dict[str, Any]:
    missing = matlab_pairs - python_pairs
    extra = python_pairs - matlab_pairs
    overlap = matlab_pairs & python_pairs
    return {
        "matlab_pair_count": len(matlab_pairs),
        "python_candidate_pair_count": len(python_pairs),
        "overlap_pair_count": len(overlap),
        "overlap_fraction_of_matlab": (
            float(len(overlap) / len(matlab_pairs)) if matlab_pairs else 1.0
        ),
        "generation_gap_count": len(missing),
        "generation_gap_fraction_of_matlab": (
            float(len(missing) / len(matlab_pairs)) if matlab_pairs else 0.0
        ),
        "extra_candidate_count": len(extra),
        "sample_missing_pairs": sorted(missing)[:20],
        "sample_extra_pairs": sorted(extra)[:20],
    }


def _count_trace_events(trace_path: Path, event_name: str) -> int:
    if not trace_path.is_file():
        return 0
    count = 0
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("event") == event_name:
            count += 1
    return count


def _trace_missing_pairs(
    *,
    run_dir: Path,
    oracle_root: Path,
    trace_path: Path,
    sample_size: int,
) -> dict[str, Any]:
    source_surface = validate_exact_proof_source_surface(run_dir)
    matlab_pairs = _endpoint_pair_set(_load_oracle_edge_connections(oracle_root))
    python_pairs = _endpoint_pair_set(_load_python_candidate_connections(run_dir))
    missing_pairs = sorted(matlab_pairs - python_pairs)[:sample_size]

    params = load_json_dict(source_surface.validated_params_path) or {}

    tracer = JsonExecutionTracer(trace_path)
    payload = _regenerate_watershed_candidates(
        run_dir=run_dir,
        params=params,
        tracer=tracer,
    )
    traced_pairs = _endpoint_pair_set(np.asarray(payload["connections"], dtype=np.int64))
    traced_missing = sorted(matlab_pairs - traced_pairs)[:sample_size]
    traced_overlap = matlab_pairs & traced_pairs

    return {
        "missing_from_run_checkpoint": missing_pairs,
        "missing_after_live_trace": traced_missing,
        "live_overlap_pair_count": len(traced_overlap),
        "live_overlap_fraction_of_matlab": (
            float(len(traced_overlap) / len(matlab_pairs)) if matlab_pairs else 1.0
        ),
        "live_generation_gap_count": len(matlab_pairs - traced_pairs),
        "live_extra_candidate_count": len(traced_pairs - matlab_pairs),
        "live_trace_pair_count": len(traced_pairs),
        "join_events_in_trace": _count_trace_events(trace_path, "join"),
        "join_skipped_events_in_trace": _count_trace_events(trace_path, "join_skipped"),
        "trace_path": str(trace_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument(
        "--trace-missing",
        action="store_true",
        help="Re-run watershed generation with JSONL tracer on a sample of missing pairs.",
    )
    parser.add_argument(
        "--trace-path",
        type=Path,
        default=Path("workspace/scratch/watershed_gap_trace.jsonl"),
    )
    parser.add_argument("--sample-size", type=int, default=5)
    args = parser.parse_args(argv)

    matlab_pairs = _endpoint_pair_set(_load_oracle_edge_connections(args.oracle_root))
    python_pairs = _endpoint_pair_set(_load_python_candidate_connections(args.run_dir))
    summary = _summarize_gap(matlab_pairs, python_pairs)
    print(json.dumps(summary, indent=2))

    if args.trace_missing:
        trace_summary = _trace_missing_pairs(
            run_dir=args.run_dir,
            oracle_root=args.oracle_root,
            trace_path=args.trace_path,
            sample_size=args.sample_size,
        )
        print(json.dumps(trace_summary, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
