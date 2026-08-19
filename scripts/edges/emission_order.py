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

from slavv_python.analytics.parity.experiments import (
    ArtifactClass,
    compare_same_class_pair_sets,
    load_edge_artifact,
)


def _incident_pairs(
    connections: np.ndarray, target: int
) -> tuple[list[int], list[tuple[int, int]]]:
    rows: list[int] = []
    pairs: list[tuple[int, int]] = []
    for index, (start, end) in enumerate(np.asarray(connections, dtype=np.int64).reshape(-1, 2)):
        if int(start) != target and int(end) != target:
            continue
        rows.append(int(index))
        left, right = (int(start), int(end)) if int(start) < int(end) else (int(end), int(start))
        pairs.append((left, right))
    return rows, pairs


def _python_emission_order(run_dir: Path, target: int) -> tuple[list[int], list[tuple[int, int]]]:
    artifact = load_edge_artifact(run_dir / "04_Edges" / "candidates.pkl")
    return _incident_pairs(artifact.connections, target)


def _matlab_emission_order(mat_path: Path, target: int) -> tuple[list[int], list[tuple[int, int]]]:
    artifact = load_edge_artifact(mat_path)
    return _incident_pairs(artifact.connections, target)


def _diff_orders(
    py_pairs: list[tuple[int, int]], mat_pairs: list[tuple[int, int]]
) -> dict[str, Any]:
    compare = compare_same_class_pair_sets(
        set(py_pairs),
        set(mat_pairs),
        left_class=ArtifactClass.RAW_CANDIDATE_SET,
        right_class=ArtifactClass.RAW_CANDIDATE_SET,
    )
    py_set, mat_set = set(py_pairs), set(mat_pairs)
    common = [pair for pair in py_pairs if pair in mat_set]
    mat_common_order = [pair for pair in mat_pairs if pair in py_set]
    first_div = None
    for idx, (python_pair, matlab_pair) in enumerate(zip(common, mat_common_order, strict=False)):
        if python_pair != matlab_pair:
            first_div = idx
            break
    return {
        "python_pair_count": compare.n_left,
        "matlab_pair_count": compare.n_right,
        "pair_set_equal": compare.n_only_left == 0 and compare.n_only_right == 0,
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
