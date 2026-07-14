"""Complete the crop edge selection in-process (no watershed re-run) and persist
the chosen edges, mirroring slavv_python/pipeline/edges/manager.py:_run_tracing
selection/bridge/finalize path. Used to validate the crop-truncation fix on the
existing fresh candidates.pkl without re-running the slow watershed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from slavv_python.analytics.parity.constants import NORMALIZED_DIR
from slavv_python.analytics.parity.oracle.surfaces import validate_exact_proof_source_surface
from slavv_python.analytics.parity.proof.coordinator import (
    load_exact_energy_result,
    load_exact_vertex_set,
)
from slavv_python.engine.state import load_json_dict
from slavv_python.pipeline.edges.bridge_insertion import add_vertices_to_edges_matlab_style
from slavv_python.pipeline.edges.discovery import (
    PipelinePolicy,
    _use_matlab_frontier_tracer,
    candidate_as_payload,
    resolve_lumen_radius_pixels_axes,
)
from slavv_python.pipeline.edges.finalize import finalize_edges_matlab_style
from slavv_python.pipeline.edges.selection import choose_edges_for_workflow
from slavv_python.utils.safe_unpickle import safe_load


def _pair_set(connections: np.ndarray) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for u, v in np.asarray(connections, dtype=np.int64).reshape(-1, 2):
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

    microns_per_voxel = np.asarray(
        params.get("microns_per_voxel", [1.0, 1.0, 1.0]), dtype=np.float64
    )
    policy = PipelinePolicy.from_params(params)
    lumen_radius_pixels_axes = resolve_lumen_radius_pixels_axes(energy, microns_per_voxel, policy)

    candidates = safe_load(args.run_dir / "04_Edges" / "candidates.pkl")
    matlab_pairs = _load_oracle_pairs(args.oracle_root)

    chosen = choose_edges_for_workflow(
        candidate_as_payload(candidates),
        vertices.positions,
        vertices.scales,
        energy.lumen_radius_microns,
        lumen_radius_pixels_axes,
        energy.energy.shape,
        params,
        energy_map=energy.energy,
        scale_indices=energy.scale_indices,
    )

    use_frontier = _use_matlab_frontier_tracer(energy.to_dict(), params)
    if use_frontier:
        chosen = add_vertices_to_edges_matlab_style(
            chosen,
            vertices.to_dict(),
            energy=energy,
            scale_indices=energy.scale_indices,
            microns_per_voxel=microns_per_voxel,
            lumen_radius_microns=energy.lumen_radius_microns,
            lumen_radius_pixels_axes=lumen_radius_pixels_axes,
            size_of_image=energy.energy.shape,
            params=params,
        )

    chosen = finalize_edges_matlab_style(
        chosen,
        lumen_radius_microns=energy.lumen_radius_microns,
        microns_per_voxel=microns_per_voxel,
        size_of_image=energy.energy.shape,
    )

    connections = np.asarray(chosen.get("connections", np.zeros((0, 2))), dtype=np.int32)
    chosen_pair_count = len(_pair_set(connections))
    overlap = len(_pair_set(connections) & matlab_pairs)

    out_path = args.run_dir / "04_Edges" / "chosen_edges.pkl"
    joblib.dump(chosen, out_path)
    print(f"Persisted {out_path}")
    print(f"chosen connections={len(connections)} pairs={chosen_pair_count} ov_matlab={overlap}")

    manifest = {
        "artifacts": {
            "candidate_audit.json": str(args.run_dir / "04_Edges" / "candidate_audit.json"),
            "candidates.pkl": str(args.run_dir / "04_Edges" / "candidates.pkl"),
            "chosen_edges.pkl": str(out_path),
        },
        "checkpoint": str(
            args.run_dir / "02_Output" / "python_results" / "checkpoints" / "checkpoint_edges.pkl"
        ),
        "completed_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "stage": "edges",
    }
    (args.run_dir / "04_Edges" / "stage_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
