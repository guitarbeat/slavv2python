"""Unit tests for cross-octave Energy reduction probes."""

from __future__ import annotations

import json

from tests.support.cross_octave_reduction_probe import (
    build_cross_octave_requests,
    compare_cross_octave_reports,
)


def _probe(request_id: str, octave: int, scale: int, energy: float) -> dict:
    return {
        "request_id": request_id,
        "probe": {
            "consolidated_octave": octave,
            "octave_winner": {"global_scale": scale, "upsampled_energy": energy},
        },
    }


def test_build_cross_octave_requests_expands_unique_voxels(tmp_path, monkeypatch):
    source_path = tmp_path / "scale_winner_requests.json"
    source_path.write_text(
        json.dumps(
            {
                "requests": [
                    {
                        "request_id": "g00_0",
                        "voxel_zyx": [1, 2, 3],
                        "matlab_scale": 4,
                        "stored_python_scale": 5,
                        "mismatch_count": 9,
                        "boundary_class": "interior",
                    },
                    {
                        "request_id": "duplicate",
                        "voxel_zyx": [1, 2, 3],
                        "matlab_scale": 4,
                        "stored_python_scale": 5,
                        "mismatch_count": 8,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "tests.support.cross_octave_reduction_probe.load_crop_image_and_config",
        lambda: (
            None,
            {
                "octave_at_scales": [1, 1, 2],
                "scale_resolution_factors": [[1, 1, 1], [1, 1, 1], [2, 3, 4]],
            },
        ),
    )

    expanded = build_cross_octave_requests(source_path)

    assert len(expanded["voxels"]) == 1
    assert [request["request_id"] for request in expanded["requests"]] == [
        "g00_0__oct1",
        "g00_0__oct2",
    ]
    assert expanded["requests"][1]["rf_matlab_yxz"] == [3, 4, 2]


def test_compare_cross_octave_reports_classifies_reduction_mismatch():
    requests = {
        "voxels": [
            {
                "parent_request_id": "g00_0",
                "voxel_zyx": [1, 2, 3],
                "stored_matlab_scale": 10,
                "stored_python_scale": 20,
            }
        ],
        "requests": [
            {"request_id": "g00_0__oct1", "parent_request_id": "g00_0"},
            {"request_id": "g00_0__oct2", "parent_request_id": "g00_0"},
        ],
    }
    python_report = {
        "records": [
            _probe("g00_0__oct1", 1, 10, -1.0),
            _probe("g00_0__oct2", 2, 20, -2.0),
        ]
    }
    matlab_report = {
        "records": [
            _probe("g00_0__oct1", 1, 10, -3.0),
            _probe("g00_0__oct2", 2, 20, -2.0),
        ]
    }

    comparison = compare_cross_octave_reports(requests, python_report, matlab_report)

    assert comparison["classifications"] == {"cross_octave_reduction": 1}
    result = comparison["results"][0]
    assert result["python_reduction"]["global_scale"] == 20
    assert result["matlab_reduction"]["global_scale"] == 10


def test_compare_cross_octave_reports_classifies_stored_python_path():
    requests = {
        "voxels": [
            {
                "parent_request_id": "g00_0",
                "voxel_zyx": [1, 2, 3],
                "stored_matlab_scale": 10,
                "stored_python_scale": 20,
            }
        ],
        "requests": [{"request_id": "g00_0__oct1", "parent_request_id": "g00_0"}],
    }
    report = {"records": [_probe("g00_0__oct1", 1, 10, -1.0)]}

    comparison = compare_cross_octave_reports(requests, report, report)

    assert comparison["classifications"] == {"python_stored_state_path": 1}
