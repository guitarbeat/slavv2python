"""E4: Full no-writer Edge Selection with claimed-map hub ranking adapter.

Re-selects an existing full-volume ``candidates.pkl`` via
``select_and_finalize_edge_set`` without Watershed Discovery / writer.

Stored Python traces may still sample the original energy field. This script
applies a **script-side claimed-map ranking adapter** for the residual hub
pairs from ONE TRUTH so ``sort_edges`` (raw max) ranks the oracle partner
ahead of the extra pair.

**R2 non-claim:** Not Certification. Does not close Network ADR 0012 until a
fresh claim root's evaluated Network proof passes.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from slavv_python.analytics.parity.experiments import (
    ArtifactClass,
    compare_same_class_pair_sets,
    load_edge_artifact,
)
from slavv_python.analytics.parity.oracle.surfaces import validate_exact_proof_source_surface
from slavv_python.analytics.parity.proof.coordinator import (
    load_exact_energy_result,
    load_exact_vertex_set,
)
from slavv_python.engine.state import load_json_dict
from slavv_python.pipeline.edges.selection_workflow import select_and_finalize_edge_set
from slavv_python.utils.safe_unpickle import safe_load

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "workspace/runs/oracle_180709_E/canonical_full_v16"
DEFAULT_ORACLE_ROOT = REPO_ROOT / "workspace/oracles/180709_E_full_v2"

# ONE TRUTH residual hub (1-based MATLAB vertex indices as stored in connections).
RESIDUAL_EXTRA_PAIR = (26444, 38584)
RESIDUAL_ORACLE_PAIR = (34897, 38584)
CLAIMED_MAX_EXTRA = 0.0
CLAIMED_MAX_ORACLE = -0.239

R2_NON_CLAIM = (
    "Not Certification until a fresh claim root's evaluated Network ADR 0012 "
    "proof passes. E4 is a full no-writer residual falsifier only."
)

RESULT_BLOCKED = "blocked"
RESULT_PASS = "pass"
RESULT_FAIL = "fail"


def _undirected(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def _pair_index_map(connections: np.ndarray) -> dict[tuple[int, int], list[int]]:
    mapping: dict[tuple[int, int], list[int]] = {}
    for index, (a, b) in enumerate(np.asarray(connections, dtype=np.int64).tolist()):
        key = _undirected(int(a), int(b))
        mapping.setdefault(key, []).append(index)
    return mapping


def apply_claimed_map_hub_ranking_adapter(
    candidates: dict[str, Any],
    *,
    extra_pair: tuple[int, int] = RESIDUAL_EXTRA_PAIR,
    oracle_pair: tuple[int, int] = RESIDUAL_ORACLE_PAIR,
    claimed_max_extra: float = CLAIMED_MAX_EXTRA,
    claimed_max_oracle: float = CLAIMED_MAX_ORACLE,
) -> dict[str, Any]:
    """Override residual-hub energy traces so raw-max ranking matches claimed map.

    Returns a deep-copied candidate payload. Raises ``LookupError`` when either
    hub pair is absent (adapter unavailable on this lineage → E4 blocked).
    """
    adapted = copy.deepcopy(candidates)
    connections = np.asarray(adapted["connections"], dtype=np.int64)
    energy_traces = list(adapted["energy_traces"])
    by_pair = _pair_index_map(connections)

    extra_key = _undirected(*extra_pair)
    oracle_key = _undirected(*oracle_pair)
    if extra_key not in by_pair:
        raise LookupError(f"residual extra pair {extra_key} absent from candidates")
    if oracle_key not in by_pair:
        raise LookupError(f"residual oracle pair {oracle_key} absent from candidates")

    for index in by_pair[extra_key]:
        energy_traces[index] = np.asarray([claimed_max_extra, claimed_max_extra], dtype=np.float64)
    for index in by_pair[oracle_key]:
        energy_traces[index] = np.asarray(
            [claimed_max_oracle, claimed_max_oracle], dtype=np.float64
        )

    adapted["energy_traces"] = energy_traces
    adapted["e4_claimed_map_adapter"] = {
        "extra_pair": list(extra_key),
        "oracle_pair": list(oracle_key),
        "claimed_max_extra": claimed_max_extra,
        "claimed_max_oracle": claimed_max_oracle,
    }
    return adapted


def undirected_pair_set(connections: np.ndarray) -> set[tuple[int, int]]:
    return {_undirected(int(a), int(b)) for a, b in np.asarray(connections).tolist()}


def evaluate_hub_outcome(
    connections: np.ndarray,
    *,
    extra_pair: tuple[int, int] = RESIDUAL_EXTRA_PAIR,
    oracle_pair: tuple[int, int] = RESIDUAL_ORACLE_PAIR,
) -> dict[str, Any]:
    pairs = undirected_pair_set(connections)
    extra_key = _undirected(*extra_pair)
    oracle_key = _undirected(*oracle_pair)
    keeps_oracle = oracle_key in pairs
    drops_extra = extra_key not in pairs
    passed = keeps_oracle and drops_extra
    return {
        "keeps_oracle_partner": keeps_oracle,
        "drops_residual_extra": drops_extra,
        "passed": passed,
        "status": RESULT_PASS if passed else RESULT_FAIL,
        "extra_pair": list(extra_key),
        "oracle_pair": list(oracle_key),
        "r2_non_claim": R2_NON_CLAIM,
    }


def run_e4_reselection(
    *,
    run_dir: Path,
    oracle_root: Path | None = None,
    apply_adapter: bool = True,
) -> dict[str, Any]:
    """Load candidates, optionally adapt hub ranking, finalize Edge Set."""
    candidates_path = run_dir / "04_Edges" / "candidates.pkl"
    if not candidates_path.is_file():
        return {
            "status": RESULT_BLOCKED,
            "reason": f"candidates absent: {candidates_path}",
            "r2_non_claim": R2_NON_CLAIM,
        }

    source = validate_exact_proof_source_surface(run_dir)
    energy = load_exact_energy_result(source)
    vertices = load_exact_vertex_set(source, energy)
    params = load_json_dict(source.validated_params_path) or {}
    candidates = safe_load(candidates_path)
    if not isinstance(candidates, dict):
        return {
            "status": RESULT_BLOCKED,
            "reason": f"candidates payload is not a mapping: {candidates_path}",
            "r2_non_claim": R2_NON_CLAIM,
        }

    if apply_adapter:
        try:
            candidates = apply_claimed_map_hub_ranking_adapter(candidates)
        except LookupError as exc:
            return {
                "status": RESULT_BLOCKED,
                "reason": f"claimed-map adapter unavailable: {exc}",
                "r2_non_claim": R2_NON_CLAIM,
            }

    edge_set = select_and_finalize_edge_set(candidates, energy, vertices, params)
    chosen = edge_set.to_dict()
    connections = np.asarray(chosen.get("connections", np.zeros((0, 2))), dtype=np.int32)
    hub = evaluate_hub_outcome(connections)

    result: dict[str, Any] = {
        **hub,
        "run_dir": str(run_dir),
        "n_connections": int(connections.shape[0]),
        "adapter_applied": bool(apply_adapter),
        "writer_invoked": False,
    }
    if oracle_root is not None:
        oracle_edges = oracle_root / "03_Analysis" / "normalized" / "oracle" / "edges.pkl"
        if oracle_edges.is_file():
            pair_compare = compare_same_class_pair_sets(
                connections,
                load_edge_artifact(oracle_edges).connections,
                left_class=ArtifactClass.EDGE_SET,
                right_class=ArtifactClass.EDGE_SET,
            )
            result["oracle_edge_set_overlap"] = {
                "n_left": pair_compare.n_left,
                "n_right": pair_compare.n_right,
                "n_intersection": pair_compare.n_intersection,
                "n_only_left": pair_compare.n_only_left,
                "n_only_right": pair_compare.n_only_right,
            }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Claim/run root with 04_Edges/candidates.pkl (default: canonical_full_v16)",
    )
    parser.add_argument(
        "--oracle-root",
        type=Path,
        default=DEFAULT_ORACLE_ROOT,
        help="Oracle root for optional final↔final pair overlap diagnostics",
    )
    parser.add_argument(
        "--no-adapter",
        action="store_true",
        help="Skip claimed-map hub adapter (does not satisfy R8 alone)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write the result JSON",
    )
    args = parser.parse_args(argv)

    print("E4 full no-writer re-selection")
    print(f"R2 non-claim: {R2_NON_CLAIM}")
    print("Writer refused: this script never launches Watershed Discovery.")

    result = run_e4_reselection(
        run_dir=args.run_dir,
        oracle_root=args.oracle_root,
        apply_adapter=not args.no_adapter,
    )
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
