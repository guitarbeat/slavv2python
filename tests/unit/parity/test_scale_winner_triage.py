"""Unit tests for crop Energy scale-winner batch triage."""

from __future__ import annotations

import json

import pytest

from tests.support.parity_harness import select_mismatch_group_requests
from tests.support.scale_winner_triage import _probe_difference, compare_batch_reports, main

pytestmark = [pytest.mark.unit, pytest.mark.parity]


def _probe() -> dict:
    return {
        "voxel_zyx": [1, 2, 3],
        "consolidated_octave": 2,
        "rf_matlab_yxz": [3, 3, 1],
        "chunk_lattice_dimensions_yxz": [3, 3, 2],
        "chunk_idx": 1,
        "write_index_yxz": [1, 0, 0],
        "write_window_zyx": {"starts": [0, 0, 0], "counts": [16, 32, 32]},
        "offsets_yxz": [0, 0, 0],
        "strided_read_shape_yxz": [32, 32, 16],
        "padded_shape_yxz": [34, 34, 18],
        "coarse_local_slices_yxz": [[0, 12], [0, 12], [0, 12]],
        "mesh_at_voxel": {"mesh_y": 1.0, "mesh_x": 2.0, "mesh_z": 3.0},
        "per_scale": [{"global_scale": 10, "upsampled_energy": -2.0}],
        "octave_winner": {"global_scale": 10, "upsampled_energy": -2.0},
    }


def test_probe_difference_classifies_lattice_before_energy():
    matlab = _probe()
    matlab["chunk_idx"] = 2
    difference = _probe_difference(_probe(), matlab)
    assert difference is not None
    assert difference["stage"] == "chunk_lattice"


def test_probe_difference_classifies_per_scale_energy_with_float_details():
    matlab = _probe()
    matlab["per_scale"][0]["upsampled_energy"] = -1.0
    difference = _probe_difference(_probe(), matlab)
    assert difference is not None
    assert difference["stage"] == "per_scale_energy"
    assert difference["python_hex"].startswith("0x")
    assert difference["ulp_distance"] > 0


def test_batch_comparator_reports_missing_matlab_response():
    comparison = compare_batch_reports(
        {"records": [{"request_id": "r1", "probe": _probe()}]},
        {"records": []},
    )
    assert comparison["failed"] == 1
    assert comparison["results"][0]["failure_class"] == "missing_matlab_response"


def test_batch_comparator_classifies_matching_winner_ulp_drift():
    python_probe = _probe()
    matlab_probe = _probe()
    matlab_probe["per_scale"][0]["upsampled_energy"] = -2.0 + 1e-12
    comparison = compare_batch_reports(
        {"records": [{"request_id": "r1", "probe": python_probe}]},
        {"records": [{"request_id": "r1", "probe": matlab_probe}]},
    )
    assert comparison["failed"] == 1
    assert comparison["classifications"]["matching_winner_ulp_drift"] == 1
    assert comparison["results"][0]["failure_class"] == "matching_winner_ulp_drift"


def test_request_selection_is_sorted_and_excludes_invalid_scales(tmp_path, monkeypatch):
    probe_requests = tmp_path / "requests.json"
    probe_requests.write_text(
        json.dumps(
            {
                "groups": [
                    {
                        "matlab_scale": 2,
                        "python_scale": 3,
                        "mismatch_count": 3,
                        "first_coordinate_zyx": [1, 1, 1],
                        "max_delta_coordinate_zyx": [2, 2, 2],
                        "boundary_class": "interior",
                    },
                    {
                        "matlab_scale": 1,
                        "python_scale": 2,
                        "mismatch_count": 9,
                        "first_coordinate_zyx": [3, 3, 3],
                        "max_delta_coordinate_zyx": [4, 4, 4],
                        "boundary_class": "boundary",
                    },
                    {
                        "matlab_scale": -1,
                        "python_scale": 2,
                        "mismatch_count": 99,
                        "first_coordinate_zyx": [0, 0, 0],
                        "max_delta_coordinate_zyx": [0, 0, 0],
                        "boundary_class": "boundary",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "tests.support.parity_harness.load_crop_image_and_config",
        lambda: (
            None,
            {
                "octave_at_scales": [1, 1, 2],
                "scale_resolution_factors": [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
            },
        ),
    )
    selected = select_mismatch_group_requests(probe_requests, top_groups=2, coordinates_per_group=2)
    assert [request["matlab_scale"] for request in selected["requests"]] == [1, 1, 2, 2]
    assert selected["requests"][0]["rf_matlab_yxz"] == [1, 1, 1]


def test_cli_can_reuse_existing_python_report(tmp_path, monkeypatch):
    requests_path = tmp_path / "requests.json"
    requests_path.write_text(json.dumps({"requests": []}), encoding="utf-8")
    python_report_path = tmp_path / "existing_python.json"
    python_report_path.write_text(
        json.dumps({"version": 1, "records": [{"request_id": "r1", "probe": _probe()}]}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "triage"

    def fake_run_matlab_batch(_requests_path, output_path, _matlab_exe=None, **_kwargs):
        output_path.write_text(
            json.dumps({"version": 1, "records": [{"request_id": "r1", "probe": _probe()}]}),
            encoding="utf-8",
        )

    def fail_if_recomputed(_requests_path):
        raise AssertionError("python probes should have been reused")

    monkeypatch.setattr("tests.support.scale_winner_triage.run_matlab_batch", fake_run_matlab_batch)
    monkeypatch.setattr(
        "tests.support.scale_winner_triage.build_python_batch_report", fail_if_recomputed
    )

    exit_code = main(
        [
            "--requests",
            str(requests_path),
            "--output-dir",
            str(output_dir),
            "--python-report",
            str(python_report_path),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "python_batch_probes.json").is_file()
    assert (output_dir / "scale_winner_triage.json").is_file()
