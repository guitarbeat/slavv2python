"""Focused tests for staged run layout support in evaluation tools."""

import json
from pathlib import Path

from slavv.evaluation.management import generate_manifest, list_runs, load_run_info, resolve_run_layout
from slavv.evaluation.reporting import generate_summary


def test_resolve_run_layout_prefers_staged_directories(tmp_path: Path):
    run_dir = tmp_path / "20260210_101213_manual_run"
    (run_dir / "01_Input" / "matlab_results").mkdir(parents=True)
    (run_dir / "02_Output" / "python_results").mkdir(parents=True)
    (run_dir / "03_Analysis").mkdir(parents=True)

    layout = resolve_run_layout(run_dir)

    assert layout["matlab_dir"] == run_dir / "01_Input" / "matlab_results"
    assert layout["python_dir"] == run_dir / "02_Output" / "python_results"
    assert layout["analysis_dir"] == run_dir / "03_Analysis"
    assert layout["report_file"] == run_dir / "03_Analysis" / "comparison_report.json"


def test_load_run_info_reads_report_from_analysis_stage(tmp_path: Path):
    run_dir = tmp_path / "20260209_173550_full_run"
    (run_dir / "01_Input" / "matlab_results").mkdir(parents=True)
    (run_dir / "02_Output" / "python_results").mkdir(parents=True)
    analysis_dir = run_dir / "03_Analysis"
    analysis_dir.mkdir(parents=True)
    report = {
        "matlab": {"elapsed_time": 20.0, "vertices_count": 10},
        "python": {"elapsed_time": 10.0, "vertices_count": 12},
        "performance": {"speedup": 2.0},
    }
    (analysis_dir / "comparison_report.json").write_text(json.dumps(report), encoding="utf-8")

    info = load_run_info(run_dir)

    assert info["has_matlab"] is True
    assert info["has_python"] is True
    assert info["has_report"] is True
    assert info["speedup"] == 2.0


def test_list_runs_returns_run_roots_for_staged_layout(tmp_path: Path):
    run_with_report = tmp_path / "20260209_173134_full_run"
    (run_with_report / "03_Analysis").mkdir(parents=True)
    (run_with_report / "03_Analysis" / "comparison_report.json").write_text("{}", encoding="utf-8")

    run_with_python = tmp_path / "20260209_173027_full_run"
    (run_with_python / "02_Output" / "python_results").mkdir(parents=True)

    names = [run["name"] for run in list_runs(tmp_path)]

    assert "20260209_173134_full_run" in names
    assert "20260209_173027_full_run" in names
    assert "03_Analysis" not in names
    assert "02_Output" not in names


def test_generate_summary_uses_staged_result_paths(tmp_path: Path):
    run_dir = tmp_path / "20260210_100526_full_run"
    matlab_dir = run_dir / "01_Input" / "matlab_results"
    python_dir = run_dir / "02_Output" / "python_results"
    analysis_dir = run_dir / "03_Analysis"
    matlab_dir.mkdir(parents=True)
    python_dir.mkdir(parents=True)
    analysis_dir.mkdir(parents=True)

    # Ensure directories are non-empty so status checks mark them as present.
    (matlab_dir / "batch_260210-100526").mkdir(parents=True)
    (python_dir / "checkpoints").mkdir(parents=True)
    (analysis_dir / "comparison_report.json").write_text("{}", encoding="utf-8")

    output_file = analysis_dir / "summary.txt"
    generate_summary(run_dir, output_file)
    summary = output_file.read_text(encoding="utf-8")

    assert "MATLAB results: Present" in summary
    assert "Python results: Present" in summary


def test_generate_summary_normalizes_staged_run_name(tmp_path: Path):
    run_dir = tmp_path / "20260210_100526_full_run"
    analysis_dir = run_dir / "03_Analysis"
    (run_dir / "01_Input" / "matlab_results").mkdir(parents=True)
    (run_dir / "02_Output" / "python_results").mkdir(parents=True)
    analysis_dir.mkdir(parents=True)
    (analysis_dir / "comparison_report.json").write_text("{}", encoding="utf-8")

    output_file = analysis_dir / "summary.txt"
    generate_summary(analysis_dir, output_file)
    summary = output_file.read_text(encoding="utf-8")

    assert "Run: 20260210_100526_full_run" in summary
    assert "Run: 03_Analysis" not in summary
    assert "Date: 2026-02-10" in summary


def test_generate_manifest_normalizes_staged_run_root(tmp_path: Path):
    run_dir = tmp_path / "20260210_100526_full_run"
    analysis_dir = run_dir / "03_Analysis"
    metadata_dir = run_dir / "99_Metadata"
    (run_dir / "01_Input" / "matlab_results").mkdir(parents=True)
    python_dir = run_dir / "02_Output" / "python_results"
    python_dir.mkdir(parents=True)
    analysis_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)
    (python_dir / "network.json").write_text("{}", encoding="utf-8")

    content = generate_manifest(analysis_dir, metadata_dir / "run_manifest.md")

    assert content.splitlines()[0] == "# SLAVV Comparison Run: 20260210_100526_full_run"
    assert "`02_Output/python_results/network.json`" in content
