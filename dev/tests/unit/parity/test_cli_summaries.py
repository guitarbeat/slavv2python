"""Unit tests for CLI summary formatting."""

from __future__ import annotations

from pathlib import Path

from slavv.parity.cli_summaries import (
    format_missing_artifact_explanation,
    format_reuse_eligibility_summary,
    generate_next_action_commands,
    generate_reuse_commands,
)
from slavv.parity.workflow_assessment import LoopAssessmentReport


def test_format_reuse_eligibility_summary_basic():
    """Test basic summary formatting with minimal report."""
    report = LoopAssessmentReport(
        run_root="/path/to/run",
        requested_loop="full_comparison",
        verdict="reuse_ready",
        safe_to_reuse=True,
        safe_to_analyze_only=False,
        requires_fresh_matlab=False,
    )

    run_root = Path("/path/to/run")
    input_file = Path("/path/to/input.tif")

    summary = format_reuse_eligibility_summary(report, run_root=run_root, input_file=input_file)

    assert "WORKFLOW REUSE ELIGIBILITY SUMMARY" in summary
    assert "Verdict: Reuse Ready" in summary
    assert "Safe to reuse: Yes" in summary
    assert "Safe to analyze only: No" in summary
    assert "Requires fresh MATLAB: No" in summary


def test_format_reuse_eligibility_summary_with_missing_artifacts():
    """Test summary formatting with missing artifacts."""
    report = LoopAssessmentReport(
        run_root="/path/to/run",
        requested_loop="skip_matlab_edges",
        verdict="blocked",
        safe_to_reuse=False,
        missing_artifacts=["MATLAB edges checkpoint", "Python network.json"],
        artifact_explanations={
            "MATLAB edges checkpoint": "Required for stage-isolated network gate validation",
            "Python network.json": "Required for analysis-only comparison",
        },
    )

    run_root = Path("/path/to/run")
    input_file = Path("/path/to/input.tif")

    summary = format_reuse_eligibility_summary(report, run_root=run_root, input_file=input_file)

    assert "Missing Artifacts:" in summary
    assert "MATLAB edges checkpoint" in summary
    assert "Required for stage-isolated network gate validation" in summary
    assert "Python network.json" in summary
    assert "Required for analysis-only comparison" in summary


def test_format_reuse_eligibility_summary_with_commands():
    """Test summary formatting with reuse commands."""
    report = LoopAssessmentReport(
        run_root="/path/to/run",
        requested_loop="skip_matlab_edges",
        verdict="reuse_ready",
        safe_to_reuse=True,
        reuse_commands=[
            "python dev/scripts/cli/compare_matlab_python.py --skip-matlab --resume-latest",
        ],
        next_action_commands=[
            "python dev/scripts/cli/compare_matlab_python.py --input data/volume.tif",
        ],
        recommended_action="Reuse this run root for the imported-MATLAB parity rerun.",
    )

    run_root = Path("/path/to/run")
    input_file = Path("/path/to/input.tif")

    summary = format_reuse_eligibility_summary(report, run_root=run_root, input_file=input_file)

    assert "Available Workflow Loops:" in summary
    assert "--skip-matlab --resume-latest" in summary
    assert "Next Action Commands:" in summary
    assert "--input data/volume.tif" in summary
    assert "Recommended Next Action:" in summary
    assert "Reuse this run root" in summary


def test_generate_reuse_commands_analysis_only(tmp_path):
    """Test command generation for analysis-only comparison."""
    # Create mock directory structure
    run_root = tmp_path / "run"
    matlab_dir = run_root / "01_Input" / "matlab_results"
    python_dir = run_root / "02_Output" / "python_results"
    matlab_dir.mkdir(parents=True)
    python_dir.mkdir(parents=True)

    report = LoopAssessmentReport(
        run_root=str(run_root),
        requested_loop="standalone_analysis",
        verdict="analysis_ready",
        safe_to_analyze_only=True,
    )

    input_file = tmp_path / "input.tif"
    commands = generate_reuse_commands(report, run_root=run_root, input_file=input_file)

    assert len(commands) >= 1
    assert "--standalone-matlab-dir" in commands[0]
    assert "--standalone-python-dir" in commands[0]


def test_generate_reuse_commands_imported_matlab(tmp_path):
    """Test command generation for imported-MATLAB edge rerun."""
    run_root = tmp_path / "run"
    run_root.mkdir()

    report = LoopAssessmentReport(
        run_root=str(run_root),
        requested_loop="skip_matlab_edges",
        verdict="reuse_ready",
        safe_to_reuse=True,
        artifact_checks={"matlab_batch_present": True},
    )

    input_file = tmp_path / "input.tif"
    commands = generate_reuse_commands(report, run_root=run_root, input_file=input_file)

    # Should have both edges and network commands
    assert len(commands) >= 2
    edges_cmd = [cmd for cmd in commands if "--python-parity-rerun-from edges" in cmd]
    network_cmd = [cmd for cmd in commands if "--python-parity-rerun-from network" in cmd]
    assert len(edges_cmd) == 1
    assert len(network_cmd) == 1
    assert "--skip-matlab" in edges_cmd[0]
    assert "--skip-matlab" in network_cmd[0]


def test_generate_reuse_commands_no_matlab_batch():
    """Test command generation when MATLAB batch is not present."""
    report = LoopAssessmentReport(
        run_root="/path/to/run",
        requested_loop="skip_matlab_edges",
        verdict="fresh_matlab_required",
        safe_to_reuse=False,
        artifact_checks={"matlab_batch_present": False},
    )

    run_root = Path("/path/to/run")
    input_file = Path("/path/to/input.tif")
    commands = generate_reuse_commands(report, run_root=run_root, input_file=input_file)

    # Should not generate skip-matlab commands without MATLAB batch
    skip_matlab_cmds = [cmd for cmd in commands if "--skip-matlab" in cmd]
    assert not skip_matlab_cmds


def test_format_missing_artifact_explanation():
    """Test missing artifact explanation formatting."""
    explanation = format_missing_artifact_explanation(
        "MATLAB edges checkpoint",
        required_for="stage-isolated network gate validation",
    )

    assert "MATLAB edges checkpoint" in explanation
    assert "Required for" in explanation
    assert "stage-isolated network gate validation" in explanation


def test_generate_next_action_commands_fresh_matlab():
    """Test next action command generation for fresh MATLAB run."""
    report = LoopAssessmentReport(
        run_root="/path/to/run",
        requested_loop="full_comparison",
        verdict="fresh_matlab_required",
        requires_fresh_matlab=True,
    )

    run_root = Path("/path/to/run")
    input_file = Path("/path/to/input.tif")
    commands = generate_next_action_commands(report, run_root=run_root, input_file=input_file)

    assert len(commands) >= 1
    assert str(input_file) in commands[0]
    assert str(run_root) in commands[0]
    assert "compare_matlab_python.py" in commands[0]


def test_generate_next_action_commands_blocked():
    """Test next action command generation for blocked workflow."""
    report = LoopAssessmentReport(
        run_root="/path/to/run",
        requested_loop="skip_matlab_edges",
        verdict="blocked",
        input_compatible=False,
    )

    run_root = Path("/path/to/run")
    input_file = Path("/path/to/input.tif")
    commands = generate_next_action_commands(report, run_root=run_root, input_file=input_file)

    assert len(commands) >= 1
    assert "<new_run_root>" in commands[0]
    assert str(input_file) in commands[0]


def test_generate_next_action_commands_analysis_ready(tmp_path):
    """Test next action command generation for analysis-ready workflow."""
    # Create mock directory structure
    run_root = tmp_path / "run"
    matlab_dir = run_root / "01_Input" / "matlab_results"
    python_dir = run_root / "02_Output" / "python_results"
    matlab_dir.mkdir(parents=True)
    python_dir.mkdir(parents=True)

    report = LoopAssessmentReport(
        run_root=str(run_root),
        requested_loop="standalone_analysis",
        verdict="analysis_ready",
        safe_to_analyze_only=True,
    )

    input_file = tmp_path / "input.tif"
    commands = generate_next_action_commands(report, run_root=run_root, input_file=input_file)

    assert len(commands) >= 1
    assert "--standalone-matlab-dir" in commands[0]
    assert "--standalone-python-dir" in commands[0]
