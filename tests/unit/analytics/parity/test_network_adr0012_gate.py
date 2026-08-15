from __future__ import annotations

import numpy as np
import pytest

from slavv_python.analytics.parity.proof.artifact_comparator import (
    _compare_network_stage,
    compare_exact_artifacts,
)


@pytest.mark.unit
def test_network_stage_emits_evaluated_adr0012_gate_on_pass() -> None:
    strands = [[[0, 1], [1, 2]], [[2, 3]]]
    payload = {
        "strands": strands,
        "bifurcations": np.asarray([1], dtype=np.int32),
    }
    mismatch, gate = _compare_network_stage(payload, payload)
    assert mismatch is None
    assert gate["adr0012_evaluated"] is True
    assert gate["passed"] is True
    assert gate["n_matlab_strand_pairs"] == gate["n_python_strand_pairs"] == 2


@pytest.mark.unit
def test_network_stage_emits_evaluated_adr0012_gate_on_fail() -> None:
    matlab = {
        "strands": [[[0, 1]]],
        "bifurcations": np.asarray([], dtype=np.int32),
    }
    python = {
        "strands": [[[0, 2]]],
        "bifurcations": np.asarray([], dtype=np.int32),
    }
    mismatch, gate = _compare_network_stage(matlab, python)
    assert mismatch is not None
    assert mismatch["mismatch_type"] == "strand endpoint-pair multiset mismatch"
    assert gate["adr0012_evaluated"] is True
    assert gate["passed"] is False


@pytest.mark.unit
def test_compare_exact_artifacts_includes_network_adr0012_gate() -> None:
    strands = [[[0, 1]]]
    artifacts = {
        "network": {
            "strands": strands,
            "bifurcations": np.asarray([], dtype=np.int32),
            "strand_subscripts": [],
            "strand_energies": [],
            "strand_radii": [],
            "edge_indices_in_strands": [],
        }
    }
    # Fill required EXACT_STAGE_FIELDS keys loosely via compare path — use
    # matching matlab/python payloads that include only what compare reads.
    report = compare_exact_artifacts(
        matlab_artifacts=artifacts,
        python_artifacts=artifacts,
        stages=["network"],
    )
    assert report["passed"] is True
    gate = report["network_adr0012_gate"]
    assert gate["adr0012_evaluated"] is True
    assert gate["passed"] is True
