"""Fast tests for the seeded MATLAB/Python random-component harness."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from tests.support.random_component_parity import (
    FIXTURE_PATH,
    CorpusCase,
    _compare_values,
    _matching_kernel_reference,
    _query_coordinates,
    compare_references,
    format_hessian_advisory_summary,
    load_manifest,
    load_matching_reference,
    load_matlab_reference,
    materialize_corpus,
    print_hessian_advisory_summary,
    verify_matlab_prerequisites,
    write_case_reports,
)

pytestmark = [pytest.mark.unit, pytest.mark.parity]


def test_manifest_is_fixed_six_case_corpus_with_128_linspace_contexts():
    manifest = load_manifest()
    assert manifest["version"] == 1
    assert len(manifest["cases"]) == 6
    assert manifest["linspace_context_count"] == 128


def test_matching_reference_exposes_iso_and_aniso_spacing_keys():
    payload = load_matching_reference()
    assert payload["version"] == 1
    assert set(payload["kernels"]) == {"1,1,1", "1,1,2"}


def test_matching_reference_kernel_shape_matches_corpus_padding():
    manifest = load_manifest()
    energy = manifest["energy"]
    iso_spacing = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
    matching, weights = _matching_kernel_reference(iso_spacing, (34, 34, 18), energy)
    assert matching.shape == (34, 34, 18)
    assert weights.tolist() == [1.0, 1.0, 1.0]


def test_manifest_rejects_invalid_versions(tmp_path):
    invalid = tmp_path / "invalid_manifest.json"
    invalid.write_text(
        json.dumps({"version": 2, "cases": [], "linspace_context_count": 128}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="version 1"):
        load_manifest(invalid)


def test_query_coordinates_use_integer_half_integer_and_boundary_probes():
    manifest = load_manifest()
    for raw_case in manifest["cases"]:
        case = CorpusCase(
            raw_case["id"],
            int(raw_case["seed"]),
            tuple(int(v) for v in raw_case["shape_zyx"]),
            tuple(float(v) for v in raw_case["microns_per_voxel_zyx"]),
        )
        queries = _query_coordinates(case, int(manifest["query_seed"]))
        assert len(queries) == 16
        interior = queries[:12]
        boundaries = queries[12:]
        for y, x, z in interior:
            twice = (2.0 * y, 2.0 * x, 2.0 * z)
            assert all(value.is_integer() for value in twice)
        assert boundaries[0] == [0.0, 0.0, 0.0]
        assert boundaries[1] == [31.0, 31.0, 15.0]


def test_materialized_corpus_is_byte_deterministic(tmp_path):
    first = materialize_corpus(tmp_path / "first")
    second = materialize_corpus(tmp_path / "second")
    first_manifest = json.loads(first.read_text(encoding="utf-8"))
    second_manifest = json.loads(second.read_text(encoding="utf-8"))

    assert first_manifest["linspace_contexts"] == second_manifest["linspace_contexts"]
    for first_case, second_case in zip(
        first_manifest["cases"], second_manifest["cases"], strict=True
    ):
        assert first_case["sha256"] == second_case["sha256"]
        assert first_case["seed"] == second_case["seed"]
        assert first_case["query_yxz"] == second_case["query_yxz"]
        first_tif = (tmp_path / "first" / "inputs" / f"{first_case['id']}.tif").read_bytes()
        second_tif = (tmp_path / "second" / "inputs" / f"{second_case['id']}.tif").read_bytes()
        assert first_tif == second_tif


def test_malformed_matlab_output_is_rejected(tmp_path):
    malformed = tmp_path / "matlab_reference.mat"
    malformed.write_text("not a MATLAB MAT file", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed MATLAB"):
        load_matlab_reference(malformed)


def test_missing_matlab_executable_fails_before_comparison():
    with pytest.raises(FileNotFoundError, match="MATLAB R2019a executable unavailable"):
        verify_matlab_prerequisites("C:\\missing\\matlab.exe")


def test_comparator_reports_float_hex_and_ulp_for_deliberate_mismatch():
    difference = _compare_values("energy.samples[0].laplacian", 1.0, 2.0)
    assert difference is not None
    assert difference["python_hex"] == "0x3ff0000000000000"
    assert difference["matlab_hex"] == "0x4000000000000000"
    assert difference["ulp_distance"] > 0


def test_comparator_reports_first_mismatch_from_matlab_reference():
    python = {
        "linspace": [{"offset": 0, "stride": 1, "count": 1, "local_start": 0, "values": [0.0]}]
        * 128,
        "cases": [],
    }
    matlab = {
        "linspace": [SimpleNamespace(offset=0, stride=1, count=1, local_start=0, values=[1.0])]
        * 128,
        "cases": [],
    }
    report = compare_references(python, matlab, manifest=load_manifest(FIXTURE_PATH))
    assert report["passed"] is False
    assert report["first_difference"]["path"] == "linspace[0].values[0]"
    assert report["first_difference"]["component"] == "linspace"
    assert report["linspace"]["passed"] is False


def test_case_reports_include_seed_and_component_context(tmp_path):
    python = {
        "linspace": [{"offset": 0, "stride": 1, "count": 1, "local_start": 0, "values": [0.0]}]
        * 128,
        "cases": [
            {
                "case_id": "noise_iso_01",
                "query_yxz": [[0.0, 0.0, 0.0]],
                "interpolation": [0.0],
                "energy": {"padded_shape_yxz": [32, 32, 32], "samples": []},
            }
        ],
    }
    matlab = {
        "linspace": [SimpleNamespace(offset=0, stride=1, count=1, local_start=0, values=[0.0])]
        * 128,
        "cases": [
            SimpleNamespace(
                case_id="noise_iso_01",
                interpolation=[1.0],
                padded_shape_yxz=[32, 32, 32],
                samples=[],
            )
        ],
    }
    report = compare_references(python, matlab, manifest=load_manifest(FIXTURE_PATH))
    write_case_reports(tmp_path, report)
    case_report = json.loads(
        (tmp_path / "reports" / "noise_iso_01.json").read_text(encoding="utf-8")
    )
    assert case_report["seed"] == 1001
    assert case_report["passed"] is False
    assert case_report["first_difference"]["component"] == "interp3"
    assert case_report["first_difference"]["operands"]["query_yxz"] == [0.0, 0.0, 0.0]


def test_compare_references_include_hessian_diagnostics_without_failing_gate():
    python = {
        "linspace": [{"offset": 0, "stride": 1, "count": 1, "local_start": 0, "values": [0.0]}]
        * 128,
        "cases": [
            {
                "case_id": "noise_iso_01",
                "query_yxz": [[0.0, 0.0, 0.0]],
                "interpolation": [0.0],
                "energy": {
                    "padded_shape_yxz": [34, 34, 18],
                    "samples": [
                        {
                            "coordinate_yxz": [0, 0, 0],
                            "curvatures": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            "gradient": [0.0, 0.0, 0.0],
                            "laplacian": 1.0,
                            "valid": False,
                            "energy": float("inf"),
                        }
                    ],
                },
            }
        ],
    }
    matlab = {
        "linspace": [SimpleNamespace(offset=0, stride=1, count=1, local_start=0, values=[0.0])]
        * 128,
        "cases": [
            SimpleNamespace(
                case_id="noise_iso_01",
                interpolation=[0.0],
                padded_shape_yxz=[34, 34, 18],
                samples=[
                    SimpleNamespace(
                        coordinate_yxz=[0, 0, 0],
                        curvatures=[2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        gradient=[0.0, 0.0, 0.0],
                        laplacian=2.0,
                        valid=False,
                        energy=float("inf"),
                    )
                ],
            )
        ],
    }
    report = compare_references(python, matlab, manifest=load_manifest(FIXTURE_PATH))
    assert report["passed"] is True
    assert report["hessian_diagnostics"]["max_ulp_distance"] > 0
    assert report["cases"][0]["hessian_diagnostics"]["mismatch_count"] > 0


def test_format_hessian_advisory_summary_includes_case_rows():
    report = {
        "passed": True,
        "hessian_diagnostics": {
            "max_ulp_distance": 12,
            "worst_case_id": "noise_iso_01",
            "worst_mismatch": {
                "component": "energy.curvatures",
                "path": "cases[noise_iso_01].energy.samples[0].curvatures[0]",
                "coordinate_yxz": [0, 0, 0],
            },
            "cases": [
                {"case_id": "noise_iso_01", "mismatch_count": 3, "max_ulp_distance": 12},
            ],
        },
    }
    summary = format_hessian_advisory_summary(report)
    assert "max_ulp_distance: 12" in summary
    assert "case noise_iso_01: mismatches=3 max_ulp=12" in summary


def test_print_hessian_advisory_summary_reads_saved_report(tmp_path):
    report_path = tmp_path / "random_component_parity_report.json"
    report_path.write_text(
        json.dumps(
            {
                "passed": True,
                "hessian_diagnostics": {
                    "max_ulp_distance": 4,
                    "worst_case_id": "noise_iso_02",
                    "worst_mismatch": None,
                    "cases": [],
                },
            }
        ),
        encoding="utf-8",
    )
    print_hessian_advisory_summary(report_path)
