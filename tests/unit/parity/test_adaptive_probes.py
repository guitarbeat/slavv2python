from __future__ import annotations

import json

import numpy as np
import pytest

from slavv_python.analytics.parity.adaptive_probes import (
    build_energy_probe_payload,
    compare_probe_jsonl,
    ensure_rerun_allowed,
    record_hypothesis,
)


def test_adaptive_probe_groups_without_coordinate_table():
    matlab_energy = np.zeros((2, 2, 1), dtype=np.float64)
    python_energy = matlab_energy.copy()
    matlab_scales = np.zeros((2, 2, 1), dtype=np.int16)
    python_scales = matlab_scales.copy()
    python_energy[0, 0, 0] = -1.0
    python_energy[1, 1, 0] = -3.0
    python_scales[1, 1, 0] = 2

    payload = build_energy_probe_payload(
        matlab_energy, python_energy, matlab_scales, python_scales, provenance={"run": "x"}
    )

    assert payload["mismatch_count"] == 2
    assert payload["group_count"] == 2
    assert len(payload["probe_requests"]) == 4
    assert "coordinates" not in payload["groups"][0]


def test_probe_jsonl_comparison_reports_field_difference(tmp_path):
    matlab = tmp_path / "matlab.jsonl"
    python = tmp_path / "python.jsonl"
    matlab.write_text(json.dumps({"request_id": "r1", "energy": -1.0}) + "\n", encoding="utf-8")
    python.write_text(json.dumps({"request_id": "r1", "energy": -2.0}) + "\n", encoding="utf-8")

    report = compare_probe_jsonl(matlab, python)

    assert report["passed"] is False
    assert report["differences"][0]["fields"] == ["energy"]


def test_hypothesis_ledger_blocks_third_mathematical_attempt(tmp_path):
    proof = tmp_path / "proof.json"
    proof.write_text(json.dumps({"passed": False}), encoding="utf-8")
    common = {
        "proof_report": proof,
        "first_failing_field": "energy.energy",
        "probe_request_id": "g0_first",
        "hypothesis": "mesh endpoint differs",
        "expected_field": "energy.energy",
        "kind": "mathematical",
    }
    record_hypothesis(tmp_path, **common)
    record_hypothesis(tmp_path, **common)

    with pytest.raises(RuntimeError, match="Two failed"):
        record_hypothesis(tmp_path, **common)

    with pytest.raises(RuntimeError, match="rerun blocked"):
        ensure_rerun_allowed(tmp_path, stage="energy")

    ensure_rerun_allowed(tmp_path, stage="vertices")
