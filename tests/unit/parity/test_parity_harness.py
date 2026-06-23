"""Unit tests for shared MATLAB-Python parity harness helpers."""

from __future__ import annotations

import json

import numpy as np
import pytest

from slavv_python.analytics.parity.adaptive_probes import compare_probe_jsonl
from tests.support.parity_harness import (
    assert_bit_parity_energy,
    assert_oracle_energy_parity,
    compare_matlab_python_probe_jsonl,
    float64_ulp_distance,
    probe_result_from_voxel_probe,
    probe_result_to_jsonl_record,
    write_probe_jsonl,
)


def test_assert_bit_parity_energy_accepts_identical_float64():
    assert_bit_parity_energy(-13.52067537392248, -13.52067537392248)


def test_assert_bit_parity_energy_rejects_different_values():
    with pytest.raises(AssertionError, match="bit-parity mismatch"):
        assert_bit_parity_energy(-1.0, -2.0)


def test_assert_oracle_energy_parity_allows_bounded_ulp_drift():
    expected = -13.52067537392248
    actual = np.float64(expected)
    for _ in range(4):
        actual = np.nextafter(actual, np.float64(0.0))
    assert float64_ulp_distance(actual, expected) == 4
    assert_oracle_energy_parity(actual, expected)


def test_probe_result_jsonl_round_trip(tmp_path):
    probe = {
        "consolidated_octave": 3,
        "chunk_idx": 0,
        "octave_winner": {"global_scale": 54, "upsampled_energy": -13.52067537392248},
    }
    result = probe_result_from_voxel_probe(
        probe,
        request_id="corner_12_0_0",
        coordinate_zyx=(12, 0, 0),
    )
    record = probe_result_to_jsonl_record(result)
    assert record["winner_scale"] == 54
    assert np.float64(record["winner_energy"]) == np.float64(-13.52067537392248)

    python_path = write_probe_jsonl(tmp_path / "python.jsonl", [result])
    matlab_path = tmp_path / "matlab.jsonl"
    matlab_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    report = compare_matlab_python_probe_jsonl(matlab_path, python_path)
    assert report["passed"] is True


def test_compare_jsonl_reports_winner_energy_mismatch(tmp_path):
    matlab = tmp_path / "matlab.jsonl"
    python = tmp_path / "python.jsonl"
    base = {
        "request_id": "r1",
        "coordinate_zyx": [0, 0, 0],
        "octave": 1,
        "chunk_index": 0,
        "winner_scale": 10,
    }
    matlab.write_text(
        json.dumps({**base, "winner_energy": -1.0}) + "\n",
        encoding="utf-8",
    )
    python.write_text(
        json.dumps({**base, "winner_energy": -2.0}) + "\n",
        encoding="utf-8",
    )
    report = compare_probe_jsonl(matlab, python)
    assert report["passed"] is False
    assert "winner_energy" in report["differences"][0]["fields"]
