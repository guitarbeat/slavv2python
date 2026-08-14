"""E5: MATLAB Edge Set → Python Network isolation (crop-first).

Feeds normalized MATLAB edges + vertices into Python Network construction and
compares strand endpoint-pair multisets against the MATLAB network oracle.

**R2 / AE4 non-claim:** Isolation pass confirms Edge-Set residual class.
It is **not** Phase 1 closure. Closure still requires evaluated Network
ADR 0012 on a fresh full claim root.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from slavv_python.engine.state import load_json_dict
from slavv_python.pipeline.network.manager import NetworkManager
from slavv_python.schema.results import EdgeSet, VertexSet
from slavv_python.utils.safe_unpickle import safe_load

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORACLE_ROOT = REPO_ROOT / "workspace/oracles/180709_E_crop_M_v2"
DEFAULT_RUN_DIR = REPO_ROOT / "workspace/runs/oracle_180709_E/crop_M_exact_v3"

R2_NON_CLAIM = (
    "Isolation confirmed Edge-Set residual class when it passes; "
    "Phase 1 still open until evaluated Network ADR 0012 on a fresh claim root. "
    "Isolation ≠ Phase 1 closure."
)

RESULT_BLOCKED = "blocked"
RESULT_PASS = "pass"
RESULT_FAIL = "fail"


def _strand_endpoint_pair_multiset(strands: list[Any]) -> Counter[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for strand in strands:
        flat = np.asarray(strand).ravel()
        if flat.size == 0:
            pairs.append((-1, -1))
            continue
        lo, hi = sorted((int(flat[0]), int(flat[-1])))
        pairs.append((lo, hi))
    return Counter(pairs)


def _normalized_oracle_path(oracle_root: Path, stage: str) -> Path:
    return oracle_root / "03_Analysis" / "normalized" / "oracle" / f"{stage}.pkl"


def _load_params(run_dir: Path | None, oracle_root: Path) -> dict[str, Any]:
    candidates: list[Path] = []
    if run_dir is not None:
        candidates.extend(
            [
                run_dir / "99_Metadata" / "validated_params.json",
                run_dir / "99_Metadata" / "params.json",
            ]
        )
    candidates.append(oracle_root / "99_Metadata" / "validated_params.json")
    for path in candidates:
        if path.is_file():
            loaded = load_json_dict(path) or {}
            if loaded:
                return loaded
    return {
        "microns_per_voxel": [1.0, 1.0, 1.0],
        "comparison_exact_network": True,
    }


def run_e5_isolation(
    *,
    oracle_root: Path,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    edges_path = _normalized_oracle_path(oracle_root, "edges")
    vertices_path = _normalized_oracle_path(oracle_root, "vertices")
    network_path = _normalized_oracle_path(oracle_root, "network")
    missing = [str(p) for p in (edges_path, vertices_path, network_path) if not p.is_file()]
    if missing:
        return {
            "status": RESULT_BLOCKED,
            "reason": f"isolation artifacts absent: {missing}",
            "r2_non_claim": R2_NON_CLAIM,
        }

    edges = EdgeSet.from_dict(safe_load(edges_path))
    vertices = VertexSet.from_dict(safe_load(vertices_path))
    matlab_network = safe_load(network_path)
    if not isinstance(matlab_network, dict):
        return {
            "status": RESULT_BLOCKED,
            "reason": f"MATLAB network payload is not a mapping: {network_path}",
            "r2_non_claim": R2_NON_CLAIM,
        }

    params = _load_params(run_dir, oracle_root)
    python_network = NetworkManager.run(edges, vertices, params).to_dict()

    matlab_pairs = _strand_endpoint_pair_multiset(list(matlab_network.get("strands", [])))
    python_pairs = _strand_endpoint_pair_multiset(list(python_network.get("strands", [])))
    only_mat = matlab_pairs - python_pairs
    only_py = python_pairs - matlab_pairs
    passed = not only_mat and not only_py

    message = (
        "Isolation multiset match on this surface."
        if passed
        else (
            "E5 hypothesis falsified on this surface: Network diverges with a "
            "matched MATLAB Edge Set. Portfolio scope still forbids a Network "
            "rewrite; escalate carefully or treat as residual-class refinement."
        )
    )

    return {
        "status": RESULT_PASS if passed else RESULT_FAIL,
        "passed": passed,
        "message": message,
        "n_matlab_strands": int(sum(matlab_pairs.values())),
        "n_python_strands": int(sum(python_pairs.values())),
        "n_only_matlab": int(sum(only_mat.values())),
        "n_only_python": int(sum(only_py.values())),
        "oracle_root": str(oracle_root),
        "run_dir": str(run_dir) if run_dir is not None else None,
        "r2_non_claim": R2_NON_CLAIM,
        "ae4": (
            "Isolation pass is not Phase 1 closure; evaluated Network ADR 0012 "
            "on a fresh claim root remains required."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ORACLE_ROOT)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Optional crop/claim run dir for params (default: crop_M_exact_v3)",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    print("E5 MATLAB-edge Network isolation (crop-first)")
    print(f"R2 / AE4 non-claim: {R2_NON_CLAIM}")

    result = run_e5_isolation(oracle_root=args.oracle_root, run_dir=args.run_dir)
    result["completed_at"] = datetime.now(UTC).isoformat()
    print(json.dumps(result, indent=2, default=str))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    status = result.get("status")
    if status == RESULT_BLOCKED:
        return 2
    if status == RESULT_PASS:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
