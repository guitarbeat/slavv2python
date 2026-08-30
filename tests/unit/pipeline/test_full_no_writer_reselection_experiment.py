"""E4 portfolio tests: claimed-map adapter + no-writer reselection (skip when blocked)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from slavv_python.pipeline.edges.cleanup import remove_excess_vertex_degrees
from slavv_python.pipeline.edges.selection_payloads import matlab_sort_edge_indices_by_raw_max
from slavv_python.utils.safe_unpickle import safe_load

_REPO = Path(__file__).resolve().parents[3]
_CLAIM_CANDIDATES = (
    _REPO / "workspace/runs/oracle_180709_E/canonical_full_v16/04_Edges/candidates.pkl"
)
_SCRIPT = _REPO / "scripts" / "persist_full_edges_selection.py"


def _load_e4_script() -> Any:
    name = "persist_full_edges_selection"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load E4 script: {_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


e4 = _load_e4_script()


@pytest.mark.unit
def test_e4_r2_non_claim_language() -> None:
    assert "Not Certification" in e4.R2_NON_CLAIM
    assert "ADR 0012" in e4.R2_NON_CLAIM


@pytest.mark.unit
def test_e4_claimed_map_adapter_ranks_oracle_ahead_of_extra() -> None:
    """Toy hub: adapter makes claimed-map ranking drop the residual extra."""
    extra, oracle, hub = (
        e4.RESIDUAL_EXTRA_PAIR[0],
        e4.RESIDUAL_ORACLE_PAIR[0],
        e4.RESIDUAL_EXTRA_PAIR[1],
    )
    candidates = {
        "connections": np.array([[extra, hub], [oracle, hub]], dtype=np.int32),
        "energy_traces": [
            np.array([-19.2, -9.24, -15.3], dtype=np.float64),
            np.array([-16.4, -7.73, -15.3], dtype=np.float64),
        ],
    }
    adapted = e4.apply_claimed_map_hub_ranking_adapter(candidates)
    ranked = matlab_sort_edge_indices_by_raw_max(adapted["energy_traces"], [0, 1])
    assert ranked == [1, 0], "oracle (claimed -0.239) before extra (claimed 0.0)"
    keep = remove_excess_vertex_degrees(adapted["connections"][ranked], np.zeros(2), max_degree=1)
    kept = e4.undirected_pair_set(adapted["connections"][ranked][np.asarray(keep, dtype=bool)])
    assert (min(oracle, hub), max(oracle, hub)) in {(min(a, b), max(a, b)) for a, b in kept}
    assert (min(extra, hub), max(extra, hub)) not in {(min(a, b), max(a, b)) for a, b in kept}


@pytest.mark.unit
def test_e4_adapter_missing_pair_raises_lookup() -> None:
    candidates = {
        "connections": np.array([[1, 2]], dtype=np.int32),
        "energy_traces": [np.array([-1.0], dtype=np.float64)],
    }
    with pytest.raises(LookupError, match="absent"):
        e4.apply_claimed_map_hub_ranking_adapter(candidates)


@pytest.mark.unit
def test_e4_evaluate_hub_outcome_pass_fail() -> None:
    oracle = np.array([list(e4.RESIDUAL_ORACLE_PAIR)], dtype=np.int32)
    both = np.array([list(e4.RESIDUAL_EXTRA_PAIR), list(e4.RESIDUAL_ORACLE_PAIR)], dtype=np.int32)
    assert e4.evaluate_hub_outcome(oracle)["status"] == "pass"
    assert e4.evaluate_hub_outcome(both)["status"] == "fail"


@pytest.mark.unit
def test_e4_missing_candidates_blocked_not_falsified(tmp_path: Path) -> None:
    result = e4.run_e4_reselection(run_dir=tmp_path / "missing_run")
    assert result["status"] == "blocked"
    assert "Not Certification" in result["r2_non_claim"]


@pytest.mark.unit
@pytest.mark.skipif(
    not _CLAIM_CANDIDATES.is_file(),
    reason="E4 blocked: claim candidates absent (KTD4 — not falsified)",
)
def test_e4_claim_candidates_contain_residual_hub_pairs() -> None:
    """Fast artifact probe: adapter can locate ONE TRUTH hub pairs (no writer)."""
    candidates = safe_load(_CLAIM_CANDIDATES)
    adapted = e4.apply_claimed_map_hub_ranking_adapter(candidates)
    extra = tuple(adapted["e4_claimed_map_adapter"]["extra_pair"])
    oracle = tuple(adapted["e4_claimed_map_adapter"]["oracle_pair"])
    assert set(extra) == set(e4.RESIDUAL_EXTRA_PAIR)
    assert set(oracle) == set(e4.RESIDUAL_ORACLE_PAIR)
