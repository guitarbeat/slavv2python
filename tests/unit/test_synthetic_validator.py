"""Unit Tests for Synthetic Input Behavioral Validator (Milestone M3).

Verifies SyntheticFixtureFactory, TestResultItem data models, 5-stage test suites,
SyntheticValidatorEngine, and CLI interface on isolated synthetic 3D inputs.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

_REPO_FOR_TOOLS = Path(__file__).resolve().parents[2]
_SYNTHETIC_VALIDATOR_TOOL = (
    _REPO_FOR_TOOLS
    / "workspace"
    / "experiments"
    / "matlab2python_audit"
    / "tools"
    / "synthetic_validator.py"
)
if not _SYNTHETIC_VALIDATOR_TOOL.is_file():
    pytest.skip(
        "matlab2python audit tools absent (workspace/ is gitignored)",
        allow_module_level=True,
    )

from workspace.experiments.matlab2python_audit.tools.synthetic_validator import (
    TAXONOMY_BENIGN_OPTIMIZATION,
    TAXONOMY_FILTERED_AUXILIARY,
    TAXONOMY_GENUINE_DIVERGENCE,
    TAXONOMY_VERIFIED_PARITY,
    VALIDATION_MODE,
    SyntheticFixtureFactory,
    SyntheticValidatorEngine,
    TestResultItem,
    main,
    make_result,
    parse_args,
    run_edges_tests,
    run_energy_tests,
    run_network_tests,
    run_preprocessing_tests,
    run_vertices_tests,
    sanitize_json_value,
)

from tests.e2e_audit.test_matlab2python_audit_e2e import validate_results_schema

# ============================================================================
# Test 1: Synthetic Fixture Factory
# ============================================================================


class TestSyntheticFixtureFactory:
    """Verify deterministic generation and geometric properties of synthetic fixtures."""

    def test_create_volume_fixture(self) -> None:
        """Verify 3D volume fixture dimensions, Gaussian noise, and axial banding."""
        fixture = SyntheticFixtureFactory.create_volume_fixture(shape=(32, 32, 16))
        assert fixture["shape"] == (32, 32, 16)
        assert fixture["volume"].shape == (32, 32, 16)
        assert fixture["banded_volume"].shape == (32, 32, 16)
        assert fixture["volume"].flags.f_contiguous
        assert "microns_per_voxel" in fixture
        assert fixture["bandpass_window"] == 4.0

        # Verify cylinder vessel presence at (y=16, x=16)
        center_val = fixture["volume"][16, 16, 8]
        corner_val = fixture["volume"][0, 0, 8]
        assert center_val > corner_val + 200.0

    def test_create_hessian_cylinder_fixture(self) -> None:
        """Verify 3D Hessian cylinder and analytical quadratic scalar field."""
        fixture = SyntheticFixtureFactory.create_hessian_cylinder_fixture(shape=(32, 32, 32))
        assert fixture["shape"] == (32, 32, 32)
        assert fixture["cylinder_volume"].shape == (32, 32, 32)
        assert fixture["quadratic_field"].shape == (32, 32, 32)
        assert fixture["radius_of_lumen_in_microns"] == 3.0
        assert fixture["gaussian_to_ideal_ratio"] == 0.5
        assert fixture["spherical_to_annular_ratio"] == 0.0

        # Verify quadratic field values: E(y, x, z) = 2y^2 + 3x^2 + 5z^2
        q = fixture["quadratic_field"]
        assert np.isclose(q[0, 0, 0], 0.0)
        assert np.isclose(q[2, 3, 4], 2.0 * (2**2) + 3.0 * (3**2) + 5.0 * (4**2))

    def test_create_vertex_extrema_fixture(self) -> None:
        """Verify seed vertex locations, scales, radii, and energy assignments."""
        fixture = SyntheticFixtureFactory.create_vertex_extrema_fixture(shape=(32, 32, 32))
        assert fixture["shape"] == (32, 32, 32)
        assert len(fixture["vertex_positions"]) == 4
        assert len(fixture["vertex_scales"]) == 4
        assert len(fixture["vertex_energies"]) == 4
        assert len(fixture["lumen_radius_microns"]) == 6
        assert fixture["vertex_energies"][0] == -50.0

    def test_create_catchment_basin_fixture(self) -> None:
        """Verify dual catchment basins and saddle valley energy landscape."""
        fixture = SyntheticFixtureFactory.create_catchment_basin_fixture(shape=(32, 32, 16))
        assert fixture["shape"] == (32, 32, 16)
        energy = fixture["energy"]
        well0 = energy[10, 10, 8]
        well1 = energy[10, 22, 8]
        saddle = energy[10, 16, 8]
        # Both wells must be deeper minima than the connecting saddle point
        assert well0 < saddle
        assert well1 < saddle
        assert saddle < 0.0

    def test_create_graph_topology_fixture(self) -> None:
        """Verify multi-primitive graph topology (chains, bifurcation, cycle, hair, isolated)."""
        fixture = SyntheticFixtureFactory.create_graph_topology_fixture()
        assert fixture["n_vertices"] == 11
        assert len(fixture["connections"]) == 9
        assert len(fixture["edge_traces"]) == 9
        assert len(fixture["edge_scale_traces"]) == 9
        assert len(fixture["edge_energy_traces"]) == 9
        assert fixture["helix_coords"].shape == (50, 3)


# ============================================================================
# Test 2: Taxonomy & Data Models
# ============================================================================


class TestTaxonomyAndDataModels:
    """Verify data model serialization and 4-tier taxonomy assignment."""

    def test_sanitize_json_value_numpy_types(self) -> None:
        """Verify recursive conversion of numpy data types to standard Python primitives."""
        raw = {
            "bool_val": np.bool_(True),
            "int_val": np.int64(42),
            "float_val": np.float64(3.14159),
            "array_val": np.array([1, 2, 3]),
            "nested": {
                "sub_bool": np.bool_(False),
                "sub_list": [np.int32(10), np.float32(2.5)],
            },
        }
        sanitized = sanitize_json_value(raw)
        serialized = json.dumps(sanitized)
        decoded = json.loads(serialized)

        assert decoded["bool_val"] is True
        assert decoded["int_val"] == 42
        assert np.isclose(decoded["float_val"], 3.14159)
        assert decoded["array_val"] == [1, 2, 3]
        assert decoded["nested"]["sub_bool"] is False
        assert decoded["nested"]["sub_list"] == [10, 2.5]

    def test_make_result_passed_and_failed(self) -> None:
        """Verify make_result verdict assignment based on pass/fail status."""
        passed_res = make_result(
            stage="energy",
            test_name="test_passed",
            target_modules=["mod_a.py", "mod_b.py"],
            passed=True,
            max_diff=0.0,
            classification=TAXONOMY_VERIFIED_PARITY,
            details={"metric": 100},
        )
        assert passed_res.passed is True
        assert passed_res.divergence_detected is False
        assert passed_res.classification == TAXONOMY_VERIFIED_PARITY

        failed_res = make_result(
            stage="energy",
            test_name="test_failed",
            target_modules=["mod_a.py", "mod_b.py"],
            passed=False,
            max_diff=1.5,
            classification=TAXONOMY_VERIFIED_PARITY,
            details={"error": "math mismatch"},
        )
        assert failed_res.passed is False
        assert failed_res.divergence_detected is True
        assert failed_res.classification == TAXONOMY_GENUINE_DIVERGENCE

    def test_test_result_item_to_dict_schema(self) -> None:
        """Verify TestResultItem dictionary export conforms to required schema keys."""
        item = TestResultItem(
            stage="vertices",
            test_name="test_item",
            target_modules=["vertices/detection.py"],
            passed=True,
            max_diff=0.0,
            divergence_detected=False,
            classification=TAXONOMY_BENIGN_OPTIMIZATION,
            details={"key": np.int64(7)},
        )
        d = item.to_dict()
        assert d["stage"] == "vertices"
        assert d["test_name"] == "test_item"
        assert d["target_modules"] == ["vertices/detection.py"]
        assert d["passed"] is True
        assert d["max_diff"] == 0.0
        assert d["divergence_detected"] is False
        assert d["classification"] == TAXONOMY_BENIGN_OPTIMIZATION
        assert d["details"]["key"] == 7


# ============================================================================
# Test 3: Stage Test Suites
# ============================================================================


class TestStageTestSuites:
    """Verify execution of test suites across all 5 individual pipeline stages."""

    def test_run_preprocessing_tests(self) -> None:
        """Verify Preprocessing stage test suite executes and passes all test cases."""
        results = run_preprocessing_tests(tolerance=1e-5)
        assert len(results) == 3
        for r in results:
            assert r.passed, f"Preprocessing test {r.test_name} failed: {r.details}"
            assert r.stage == "preprocessing"

        names = [r.test_name for r in results]
        assert "test_preprocessing_axial_band_removal" in names
        assert "test_preprocessing_dynamic_range" in names
        assert "test_preprocessing_legacy_script_classification" in names

    def test_run_energy_tests(self) -> None:
        """Verify Energy stage test suite executes and passes all 6 test cases."""
        results = run_energy_tests(tolerance=1e-5)
        assert len(results) == 6
        for r in results:
            assert r.passed, f"Energy test {r.test_name} failed: {r.details}"
            assert r.stage == "energy"

        names = [r.test_name for r in results]
        assert "test_energy_kernel_dft_parity" in names
        assert "test_energy_hessian_derivatives" in names
        assert "test_energy_principal_decomposition" in names
        assert "test_energy_spatial_gradients" in names
        assert "test_energy_backup_and_experimental_kernels" in names
        assert "test_energy_vessel_directions_mapping_resolution" in names

    def test_run_vertices_tests(self) -> None:
        """Verify Vertices stage test suite executes and passes all 4 test cases."""
        results = run_vertices_tests(tolerance=1e-5)
        assert len(results) == 4
        for r in results:
            assert r.passed, f"Vertices test {r.test_name} failed: {r.details}"
            assert r.stage == "vertices"

        names = [r.test_name for r in results]
        assert "test_vertices_sphere_painting" in names
        assert "test_vertices_terminal_resolution" in names
        assert "test_vertices_fix_strand_mismatch_classification" in names
        assert "test_vertices_dijkstra_trace_classification" in names

    def test_run_edges_tests(self) -> None:
        """Verify Edges stage test suite executes and passes all 4 test cases."""
        results = run_edges_tests(tolerance=1e-5)
        assert len(results) == 4
        for r in results:
            assert r.passed, f"Edges test {r.test_name} failed: {r.details}"
            assert r.stage == "edges"

        names = [r.test_name for r in results]
        assert "test_edges_watershed_catchment_connectivity" in names
        assert "test_edges_candidate_metric_and_sorting" in names
        assert "test_edges_bridge_insertion_logic" in names
        assert "test_edges_get_edges_v203_classification" in names

    def test_run_network_tests(self) -> None:
        """Verify Network stage test suite executes and passes all 5 test cases."""
        results = run_network_tests(tolerance=1e-5)
        assert len(results) == 5
        for r in results:
            assert r.passed, f"Network test {r.test_name} failed: {r.details}"
            assert r.stage == "network"

        names = [r.test_name for r in results]
        assert "test_network_get_network_v190_parity" in names
        assert "test_network_hair_pruning" in names
        assert "test_network_cycle_pruning" in names
        assert "test_network_tangents_and_smoothing" in names
        assert "test_network_combine_strands_classification" in names


# ============================================================================
# Test 4: Synthetic Validator Engine
# ============================================================================


class TestSyntheticValidatorEngine:
    """Verify SyntheticValidatorEngine orchestration, aggregation, and schema compliance."""

    def test_engine_run_all_and_summary(self) -> None:
        """Verify complete validation execution across all 22 test cases."""
        engine = SyntheticValidatorEngine(tolerance=1e-5)
        results = engine.run_all(stage_filter="all")

        summary = results["summary"]
        assert results["metadata"]["validation_mode"] == "production_probe"
        assert summary["total_tests"] == 22
        assert summary["passed_tests"] == 22
        assert summary["failed_tests"] == 0
        assert summary["divergences_detected"] == 0

        # Verify per-stage counts
        assert summary["by_stage"]["preprocessing"]["total"] == 3
        assert summary["by_stage"]["energy"]["total"] == 6
        assert summary["by_stage"]["vertices"]["total"] == 4
        assert summary["by_stage"]["edges"]["total"] == 4
        assert summary["by_stage"]["network"]["total"] == 5

        # Verify taxonomy breakdown
        by_class = summary["by_classification"]
        assert by_class[TAXONOMY_VERIFIED_PARITY] >= 10
        assert by_class[TAXONOMY_BENIGN_OPTIMIZATION] >= 3
        assert by_class[TAXONOMY_FILTERED_AUXILIARY] >= 4
        assert by_class[TAXONOMY_GENUINE_DIVERGENCE] == 0

    def test_engine_matrix_coverage_from_discrepancy_rows(self, tmp_path: Path) -> None:
        """Matrix DISCREPANCY_DETECTED rows drive coverage accounting (not dual-run)."""
        matrix = {
            "comparisons": [
                {
                    "matlab_module": "energy_filter_V200.py",
                    "matlab_file": "energy_filter_V200.py",
                    "python_target": "slavv_python/pipeline/energy/matlab_energy_filter_v200.py",
                    "stage": "energy",
                    "severity": "MEDIUM",
                    "audit_verdict": "DISCREPANCY_DETECTED",
                },
                {
                    "matlab_module": "orphan_unprobed.py",
                    "matlab_file": "orphan_unprobed.py",
                    "python_target": "slavv_python/pipeline/never/probed.py",
                    "stage": "energy",
                    "severity": "MEDIUM",
                    "audit_verdict": "DISCREPANCY_DETECTED",
                },
                {
                    "matlab_module": "ok.py",
                    "audit_verdict": "VERIFIED_PARITY",
                },
            ]
        }
        matrix_path = tmp_path / "ast_comparison_matrix.json"
        matrix_path.write_text(json.dumps(matrix), encoding="utf-8")

        engine = SyntheticValidatorEngine(matrix_path=matrix_path, tolerance=1e-5)
        results = engine.run_all(stage_filter="energy")
        coverage = results["summary"]["matrix_coverage"]
        assert results["metadata"]["validation_mode"] == "production_probe"
        assert coverage["discrepancy_modules_total"] == 2
        assert coverage["covered_count"] == 1
        assert coverage["uncovered_count"] == 1
        assert coverage["covered"][0]["matlab_module"] == "energy_filter_V200.py"
        assert coverage["uncovered"][0]["matlab_module"] == "orphan_unprobed.py"
        assert coverage["covered"][0]["covering_probes"]

    def test_engine_stage_filtering(self) -> None:
        """Verify stage filter isolates execution to requested stage only."""
        engine = SyntheticValidatorEngine(tolerance=1e-5)
        for stage in ["preprocessing", "energy", "vertices", "edges", "network"]:
            stage_res = engine.run_all(stage_filter=stage)
            assert (
                len(stage_res["test_results"]) == stage_res["summary"]["by_stage"][stage]["total"]
            )
            assert all(r["stage"] == stage for r in stage_res["test_results"])

    def test_engine_schema_contract_compliance(self) -> None:
        """Verify emitted validation results conform to PROJECT.md interface contract."""
        engine = SyntheticValidatorEngine(tolerance=1e-5)
        results = engine.run_all(stage_filter="all")
        valid, errors = validate_results_schema(results)
        assert valid, f"Validation results schema failed contract validation: {errors}"


# ============================================================================
# Test 5: CLI Interface
# ============================================================================


class TestSyntheticValidatorCLI:
    """Verify CLI argument parsing and main execution entrypoint."""

    def test_parse_args_defaults(self) -> None:
        """Verify default CLI arguments."""
        args = parse_args([])
        assert args.stage == "all"
        assert np.isclose(args.tolerance, 1e-5)
        assert args.verbose is False
        assert "ast_comparison_matrix.json" in args.matrix
        assert "validation_results.json" in args.output

    def test_cli_main_execution(self) -> None:
        """Verify main() CLI executes cleanly and writes valid JSON to destination file."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            exit_code = main(["--output", tmp_path, "--stage", "all", "--tolerance", "1e-5"])
            assert exit_code == 0
            assert os.path.exists(tmp_path)

            with open(tmp_path, encoding="utf-8") as f:
                saved_data = json.load(f)

            valid, errors = validate_results_schema(saved_data)
            assert valid, f"CLI output failed schema contract: {errors}"
            assert saved_data["summary"]["total_tests"] == 22
            assert saved_data["summary"]["passed_tests"] == 22
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


# ============================================================================
# Portfolio E6-E8 honesty (plan 2026-08-14-001)
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = REPO_ROOT / "workspace" / "experiments" / "matlab2python_audit"
_LIVE_MATRIX = AUDIT_ROOT / "ast_diffs" / "ast_comparison_matrix.json"
_LIVE_RESULTS = AUDIT_ROOT / "synthetic_tests" / "validation_results.json"


@pytest.mark.unit
def test_e6_validation_mode_constant_is_production_probe() -> None:
    """E6: VALIDATION_MODE / engine metadata agree on production-only probes."""
    assert VALIDATION_MODE == "production_probe"
    engine = SyntheticValidatorEngine(tolerance=1e-5)
    results = engine.run_all(stage_filter="preprocessing")
    assert results["metadata"]["validation_mode"] == "production_probe"


@pytest.mark.unit
@pytest.mark.skipif(
    not (_LIVE_MATRIX.is_file() and _LIVE_RESULTS.is_file()),
    reason="E7 blocked: live audit matrix/results absent",
)
def test_e7_live_matrix_discrepancy_coverage_is_complete() -> None:
    """E7: every DISCREPANCY_DETECTED flag has probe coverage (13/13)."""
    results = json.loads(_LIVE_RESULTS.read_text(encoding="utf-8"))
    coverage = results["summary"]["matrix_coverage"]
    assert coverage["discrepancy_modules_total"] == 13
    assert coverage["covered_count"] == 13
    assert coverage["uncovered_count"] == 0
    assert coverage["uncovered"] == []


@pytest.mark.unit
def test_e8_static_only_failure_path_is_not_auto_genuine_without_probe() -> None:
    """E8: make_result only promotes GENUINE on a failing probe, not AST alone.

    Static AST branch-count mismatches must not classify GENUINE without a
    failing synthetic/oracle differential (characterization of make_result).
    """
    passed = make_result(
        stage="energy",
        test_name="static_ast_branch_count_only",
        target_modules=["energy_filter_V200.py"],
        passed=True,
        max_diff=0.0,
        classification=TAXONOMY_VERIFIED_PARITY,
        details={"static_branch_delta": 3},
    )
    assert passed.classification == TAXONOMY_VERIFIED_PARITY
    assert passed.divergence_detected is False

    failed = make_result(
        stage="energy",
        test_name="behavioral_fail",
        target_modules=["energy_filter_V200.py"],
        passed=False,
        max_diff=1.0,
        classification=TAXONOMY_VERIFIED_PARITY,
        details={"error": "probe mismatch"},
    )
    assert failed.classification == TAXONOMY_GENUINE_DIVERGENCE


@pytest.mark.unit
@pytest.mark.skipif(not _LIVE_RESULTS.is_file(), reason="E8 blocked: live results absent")
def test_e8_live_results_have_zero_genuine_divergences() -> None:
    results = json.loads(_LIVE_RESULTS.read_text(encoding="utf-8"))
    by_class = results["summary"]["by_classification"]
    assert by_class.get(TAXONOMY_GENUINE_DIVERGENCE, 0) == 0
    assert results["metadata"]["validation_mode"] == "production_probe"
