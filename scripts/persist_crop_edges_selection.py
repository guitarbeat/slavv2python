"""Complete crop edge selection without re-running Watershed Discovery.

Uses ``select_and_finalize_edge_set`` (same post-discovery path as EdgeManager)
on an existing ``candidates.pkl``.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
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
from slavv_python.pipeline.edges.selection_workflow import select_and_finalize_edge_set
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

    candidates = safe_load(args.run_dir / "04_Edges" / "candidates.pkl")
    matlab_pairs = _load_oracle_pairs(args.oracle_root)

    edge_set = select_and_finalize_edge_set(candidates, energy, vertices, params)
    chosen = edge_set.to_dict()

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
        "completed_at": datetime.now(UTC).isoformat(),
        "stage": "edges",
    }
    (args.run_dir / "04_Edges" / "stage_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
