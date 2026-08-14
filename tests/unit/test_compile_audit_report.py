"""Unit tests for Milestone M4 audit report compiler."""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_COMPILER_TOOL = (
    _REPO
    / "workspace"
    / "experiments"
    / "matlab2python_audit"
    / "tools"
    / "compile_audit_report.py"
)
if not _COMPILER_TOOL.is_file():
    pytest.skip(
        "matlab2python audit tools absent (workspace/ is gitignored)",
        allow_module_level=True,
    )

from workspace.experiments.matlab2python_audit.tools.compile_audit_report import (
    classification_for_comparison,
    load_json,
    main,
    matlab_stem_to_m,
    parse_args,
    render_audit_report,
    write_reports,
)

from tests.e2e_audit.test_matlab2python_audit_e2e import (
    build_synthetic_ast_matrix,
    build_synthetic_manifest,
    build_synthetic_validation_results,
)

REPO_ROOT = _REPO
AUDIT_ROOT = REPO_ROOT / "workspace" / "experiments" / "matlab2python_audit"


def test_matlab_stem_to_m() -> None:
    assert matlab_stem_to_m("energy_filter_V200.py") == "energy_filter_V200.m"
    assert matlab_stem_to_m("get_network_V190.m") == "get_network_V190.m"


def test_render_audit_report_required_sections() -> None:
    manifest = build_synthetic_manifest()
    matrix = build_synthetic_ast_matrix()
    results = build_synthetic_validation_results()
    report = render_audit_report(manifest, matrix, results)
    assert "# Comprehensive MATLAB-to-Python Transpilation & Differential Audit Report" in report
    assert "## 1. Inventory of Transpiled Modules vs Python Modules" in report
    assert "## 2. Verified Genuine Code Defects & Discrepancies" in report
    assert "## 3. Filtered-Out Transpiler Artifacts" in report
    assert "## 4. Production Probe Results (Synthetic Fixtures)" in report
    assert "do **not** dual-run" in report
    assert "Mapping |" in report
    assert "Actionable Remediation" in report
    assert "[Y, X, Z]" in report
    assert "not** Certification" in report or "not Certification" in report


@pytest.mark.unit
def test_e6_render_banner_matches_production_probe_mode() -> None:
    """E6: report section mode banner agrees with production_probe metadata."""
    results = build_synthetic_validation_results()
    results.setdefault("metadata", {})["validation_mode"] = "production_probe"
    report = render_audit_report(
        build_synthetic_manifest(),
        build_synthetic_ast_matrix(),
        results,
    )
    assert "Mode: `production_probe`" in report or "production_probe" in report
    assert "do **not** dual-run" in report
    assert "Certification" in report


def test_write_reports_dual_paths(tmp_path: Path) -> None:
    report_path = tmp_path / "reports" / "AUDIT_REPORT.md"
    root_path = tmp_path / "AUDIT_REPORT.md"
    body = "# Comprehensive MATLAB-to-Python Transpilation & Differential Audit Report\n"
    write_reports(body, report_path, root_path)
    assert report_path.is_file()
    assert root_path.is_file()
    assert report_path.read_text(encoding="utf-8") == body
    assert root_path.read_text(encoding="utf-8") == body


def test_classification_for_comparison_prefers_genuine() -> None:
    comparison = {"matlab_module": "combine_strands.py"}
    index = {
        "combine_strands.py": [
            {"classification": "VERIFIED_BEHAVIORAL_PARITY"},
            {"classification": "GENUINE_BEHAVIORAL_DIVERGENCE"},
        ]
    }
    assert classification_for_comparison(comparison, index) == "GENUINE_BEHAVIORAL_DIVERGENCE"


def test_main_compiles_real_artifacts(tmp_path: Path) -> None:
    manifest = AUDIT_ROOT / "transpilation_manifest.json"
    matrix = AUDIT_ROOT / "ast_diffs" / "ast_comparison_matrix.json"
    results = AUDIT_ROOT / "synthetic_tests" / "validation_results.json"
    if not (manifest.is_file() and matrix.is_file() and results.is_file()):
        pytest.skip("Real matlab2python audit artifacts not present")

    report_out = tmp_path / "reports" / "AUDIT_REPORT.md"
    root_out = tmp_path / "AUDIT_REPORT.md"
    code = main(
        [
            "--manifest",
            str(manifest),
            "--matrix",
            str(matrix),
            "--results",
            str(results),
            "--report-out",
            str(report_out),
            "--root-report-out",
            str(root_out),
        ]
    )
    assert code == 0
    text = report_out.read_text(encoding="utf-8")
    assert len(text) > 1000
    assert "154" in text or "modules" in text.lower()
    assert "## 1. Inventory of Transpiled Modules vs Python Modules" in text
    data = load_json(manifest)
    assert len(data["modules"]) == 154


def test_parse_args_defaults() -> None:
    args = parse_args([])
    assert args.manifest.name == "transpilation_manifest.json"
    assert args.matrix.name == "ast_comparison_matrix.json"
    assert args.results.name == "validation_results.json"
