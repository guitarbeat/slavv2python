"""Emission-order probe for the residual edge-pair / Network one-strand swap.

Root cause (see investigation): the crop swap ``(4043, 6281)`` vs ``(4212, 6281)``
and the full-volume Network swap ``(26444, 38584)`` vs ``(34897, 38584)`` are
*perfect resampled-metric ties* at one vertex. Cleanup breaks the tie by
candidate **generation (emission) order**, and Python emits the MATLAB-kept
edge *later* than MATLAB does. This probe extracts the per-vertex edge-emission
order from both sides and pinpoints the swap.

Python emission order is read directly from ``candidates.pkl`` (the watershed
appends edges in emission order in
``matlab_get_edges_by_watershed._matlab_global_watershed_assemble_results``).
MATLAB emission order is read from a fresh ``raw_watershed_candidates.mat``
(``edges2vertices`` is appended in emission order in MATLAB's
``get_edges_by_watershed.m``).

Usage:
    # Python-only (fast):
    python scripts/edge_emission_order_probe.py \
        --run-dir workspace/runs/oracle_180709_E/crop_M_exact_v3 \
        --target-vertices 6281

    # With a fresh MATLAB raw candidates .mat:
    python scripts/edge_emission_order_probe.py \
        --run-dir workspace/runs/oracle_180709_E/crop_M_exact_v3 \
        --target-vertices 6281 \
        --matlab-raw-candidates workspace/scratch/matlab_edge_dump/raw_watershed_candidates.mat
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from slavv_python.utils.safe_unpickle import safe_load


def _undirected(a: int, b: int) -> tuple[int, int]:
    return (int(min(a, b)), int(max(a, b)))


def _python_emission_order(
    run_dir: Path, target: int
) -> tuple[list[int], list[tuple[int, int]]]:
    candidates = safe_load(run_dir / "04_Edges" / "candidates.pkl")
    connections = np.asarray(candidates["connections"], dtype=np.int32).reshape(-1, 2)
    rows = [int(i) for i, (a, b) in enumerate(connections) if a == target or b == target]
    pairs = [_undirected(int(connections[i, 0]), int(connections[i, 1])) for i in rows]
    return rows, pairs


def _matlab_emission_order(mat_path: Path, target: int) -> tuple[list[int], list[tuple[int, int]]]:
    import h5py

    with h5py.File(mat_path, "r") as f:
        e2v = np.asarray(f["edges2vertices"][:], dtype=np.int64).reshape(-1, 2)
    v1 = target + 1  # MATLAB is 1-based
    rows = [int(i) for i in range(e2v.shape[0]) if e2v[i, 0] == v1 or e2v[i, 1] == v1]
    pairs = [_undirected(int(e2v[i, 0]) - 1, int(e2v[i, 1]) - 1) for i in rows]
    return rows, pairs


def _diff_orders(
    py_pairs: list[tuple[int, int]], mat_pairs: list[tuple[int, int]]
) -> dict:
    py_set, mat_set = set(py_pairs), set(mat_pairs)
    common = [p for p in py_pairs if p in mat_set]
    mat_common_order = [p for p in mat_pairs if p in py_set]
    # Find first position where the common-pair order diverges.
    first_div = None
    for idx, (p, mp) in enumerate(zip(common, mat_common_order)):
        if p != mp:
            first_div = idx
            break
    return {
        "python_pair_count": len(py_pairs),
        "matlab_pair_count": len(mat_pairs),
        "pair_set_equal": bool(py_set == mat_set),
        "python_only_pairs": sorted(py_set - mat_set),
        "matlab_only_pairs": sorted(mat_set - py_set),
        "common_pair_order_python": common,
        "common_pair_order_matlab": mat_common_order,
        "first_divergence_position": first_div,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--target-vertices", type=int, nargs="+", required=True)
    parser.add_argument("--matlab-raw-candidates", type=Path, default=None)
    args = parser.parse_args(argv)

    report: dict[str, Any] = {"run_dir": str(args.run_dir)}
    for target in args.target_vertices:
        py_rows, py_pairs = _python_emission_order(args.run_dir, target)
        entry: dict[str, Any] = {
            "python_emission_rows": py_rows,
            "python_emission_pairs": py_pairs,
        }
        if args.matlab_raw_candidates is not None and args.matlab_raw_candidates.exists():
            mat_rows, mat_pairs = _matlab_emission_order(args.matlab_raw_candidates, target)
            entry["matlab_emission_rows"] = mat_rows
            entry["matlab_emission_pairs"] = mat_pairs
            entry["diff"] = _diff_orders(py_pairs, mat_pairs)
        report[str(target)] = entry

    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
