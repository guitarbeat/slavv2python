from __future__ import annotations

import numpy as np
from source.core._edge_candidates.common import _build_matlab_global_watershed_lut
from source.io.matlab_fail_fast import (
    DEBUG_MAP_FIELDS,
    build_candidate_coverage_report,
    build_candidate_snapshot_payload,
    compare_lut_fixture_payload,
)


def _lut_fixture_payload() -> dict[str, object]:
    lut = _build_matlab_global_watershed_lut(
        0,
        size_of_image=(5, 5, 5),
        lumen_radius_microns=np.array([1.0], dtype=np.float32),
        microns_per_voxel=np.ones((3,), dtype=np.float32),
        step_size_per_origin_radius=1.0,
    )
    return {
        "size_of_image": [5, 5, 5],
        "microns_per_voxel": [1.0, 1.0, 1.0],
        "lumen_radius_microns": [1.0],
        "scales": {
            "0": {
                "linear_offsets": np.asarray(lut["linear_offsets"], dtype=np.int64).tolist(),
                "local_subscripts": np.asarray(lut["local_subscripts"], dtype=np.int32).tolist(),
                "r_over_R": np.asarray(lut["r_over_R"], dtype=np.float32).tolist(),
                "unit_vectors": np.asarray(lut["unit_vectors"], dtype=np.float32).tolist(),
            }
        },
    }


def test_compare_lut_fixture_payload_passes_on_exact_match():
    report = compare_lut_fixture_payload(
        _lut_fixture_payload(),
        size_of_image=(5, 5, 5),
        microns_per_voxel=np.ones((3,), dtype=np.float32),
        lumen_radius_microns=np.array([1.0], dtype=np.float32),
    )

    assert report["passed"] is True
    assert report["first_failure"] is None


def test_compare_lut_fixture_payload_reports_ordering_mismatch():
    fixture = _lut_fixture_payload()
    scale_payload = fixture["scales"]["0"]
    scale_payload["linear_offsets"][0], scale_payload["linear_offsets"][1] = (
        scale_payload["linear_offsets"][1],
        scale_payload["linear_offsets"][0],
    )
    scale_payload["local_subscripts"][0], scale_payload["local_subscripts"][1] = (
        scale_payload["local_subscripts"][1],
        scale_payload["local_subscripts"][0],
    )

    report = compare_lut_fixture_payload(
        fixture,
        size_of_image=(5, 5, 5),
        microns_per_voxel=np.ones((3,), dtype=np.float32),
        lumen_radius_microns=np.array([1.0], dtype=np.float32),
    )

    assert report["passed"] is False
    assert report["first_failure"]["mismatch_type"] == "ordering mismatch"


def test_build_candidate_snapshot_payload_excludes_debug_maps_by_default():
    candidates = {
        "connections": np.array([[0, 1]], dtype=np.int32),
        "traces": [np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)],
        "scale_traces": [np.array([0.0, 0.0], dtype=np.float32)],
        "energy_traces": [np.array([-2.0, -1.0], dtype=np.float32)],
        "metrics": np.array([-1.0], dtype=np.float32),
        "origin_indices": np.array([0], dtype=np.int32),
        "connection_sources": ["global_watershed"],
        "diagnostics": {"candidate_traced_edge_count": 1},
        "candidate_source": "global_watershed",
        "matlab_global_watershed_exact": True,
        "energy_map": np.ones((2, 2, 2), dtype=np.float32),
        "pointer_map": np.ones((2, 2, 2), dtype=np.float32),
    }

    snapshot = build_candidate_snapshot_payload(candidates)

    assert snapshot["matlab_global_watershed_exact"] is True
    assert snapshot["candidate_source"] == "global_watershed"
    for field_name in DEBUG_MAP_FIELDS:
        assert field_name not in snapshot


def test_build_candidate_coverage_report_counts_matched_missing_and_extra_pairs():
    matlab_edges_payload = {
        "connections": np.array([[0, 1], [1, 2]], dtype=np.int64),
    }
    candidate_payload = {
        "connections": np.array([[0, 1], [2, 3]], dtype=np.int32),
    }

    report = build_candidate_coverage_report(matlab_edges_payload, candidate_payload)

    assert report["passed"] is False
    assert report["matched_pair_count"] == 1
    assert report["missing_pair_count"] == 1
    assert report["extra_pair_count"] == 1
    assert report["missing_pair_samples"] == [[1, 2]]
    assert report["extra_pair_samples"] == [[2, 3]]
