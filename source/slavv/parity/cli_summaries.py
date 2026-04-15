"""CLI summary formatting for parity workflow reuse eligibility."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from .workflow_assessment import LoopAssessmentReport


def format_reuse_eligibility_summary(
    report: LoopAssessmentReport,
    *,
    run_root: Path,
    input_file: Path,
) -> str:
    """
    Generate human-readable CLI summary from loop assessment.

    Returns formatted text with:
    - Reuse eligibility status
    - Safe workflow loops with specific commands
    - Missing artifacts with explanations
    - Recommended next action

    Args:
        report: Loop assessment report with workflow decision surface
        run_root: Path to the staged run root
        input_file: Path to the input volume file

    Returns:
        Formatted CLI summary text
    """
    lines = []
    lines.append("=" * 80)
    lines.append("WORKFLOW REUSE ELIGIBILITY SUMMARY")
    lines.append("=" * 80)
    lines.append("")

    # Verdict and status
    verdict_label = report.verdict.replace("_", " ").title()
    lines.append(f"Verdict: {verdict_label}")
    lines.append(f"Run Root: {run_root}")
    lines.append(f"Requested Loop: {report.requested_loop}")
    lines.append("")

    # Eligibility flags
    lines.append("Eligibility:")
    lines.append(f"  - Safe to reuse: {'Yes' if report.safe_to_reuse else 'No'}")
    lines.append(f"  - Safe to analyze only: {'Yes' if report.safe_to_analyze_only else 'No'}")
    lines.append(f"  - Requires fresh MATLAB: {'Yes' if report.requires_fresh_matlab else 'No'}")
    lines.append("")

    # Compatibility status
    if report.compatibility_reason:
        lines.append("Compatibility:")
        lines.append(f"  - Input compatible: {'Yes' if report.input_compatible else 'No'}")
        lines.append(f"  - Params compatible: {'Yes' if report.params_compatible else 'No'}")
        if not report.input_compatible or not report.params_compatible:
            lines.append(f"  - Reason: {report.compatibility_reason}")
        lines.append("")

    # Artifact status
    if report.artifact_reason:
        lines.append("Artifacts:")
        lines.append(
            f"  - Required artifacts present: {'Yes' if report.has_required_artifacts else 'No'}"
        )
        lines.append(f"  - Status: {report.artifact_reason}")
        lines.append("")

    # Missing artifacts with explanations
    if report.missing_artifacts:
        lines.append("Missing Artifacts:")
        for artifact in report.missing_artifacts:
            explanation = report.artifact_explanations.get(artifact, "")
            if explanation:
                lines.append(f"  - {artifact}: {explanation}")
            else:
                lines.append(f"  - {artifact}")
        lines.append("")

    # Reasons and warnings
    if report.reasons:
        lines.append("Assessment Reasons:")
        for reason in report.reasons:
            lines.append(f"  - {reason}")
        lines.append("")

    if report.warnings:
        lines.append("Warnings:")
        for warning in report.warnings:
            lines.append(f"  - {warning}")
        lines.append("")

    # Available reuse commands
    if report.reuse_commands:
        lines.append("Available Workflow Loops:")
        for cmd in report.reuse_commands:
            lines.append(f"  {cmd}")
        lines.append("")

    # Recommended next action
    if report.recommended_action:
        lines.append("Recommended Next Action:")
        lines.append(f"  {report.recommended_action}")
        lines.append("")

    # Next action commands
    if report.next_action_commands:
        lines.append("Next Action Commands:")
        for cmd in report.next_action_commands:
            lines.append(f"  {cmd}")
        lines.append("")

    lines.append("=" * 80)

    return "\n".join(lines)


def generate_reuse_commands(
    report: LoopAssessmentReport,
    *,
    run_root: Path,
    input_file: Path,
) -> list[str]:
    """
    Generate specific rerun commands based on workflow state.

    Returns commands for:
    - Analysis-only comparison (--standalone-matlab-dir / --standalone-python-dir)
    - Imported-MATLAB edge rerun (--skip-matlab --python-parity-rerun-from edges)
    - Stage-isolated network gate (--skip-matlab --python-parity-rerun-from network)

    Args:
        report: Loop assessment report with workflow decision surface
        run_root: Path to the staged run root
        input_file: Path to the input volume file

    Returns:
        List of executable command strings
    """
    commands = []

    # Analysis-only comparison
    if report.safe_to_analyze_only:
        from .run_layout import resolve_run_layout

        layout = resolve_run_layout(run_root)
        matlab_dir = layout["matlab_dir"]
        python_dir = layout["python_dir"]

        if matlab_dir.exists() and python_dir.exists():
            cmd = (
                f"python dev/scripts/cli/compare_matlab_python.py "
                f"--standalone-matlab-dir {matlab_dir} "
                f"--standalone-python-dir {python_dir} "
                f"--output-dir {run_root}"
            )
            commands.append(cmd)

    # Imported-MATLAB edge rerun
    if report.safe_to_reuse and report.artifact_checks.get("matlab_batch_present", False):
        cmd = (
            f"python dev/scripts/cli/compare_matlab_python.py "
            f"--input {input_file} "
            f"--output-dir {run_root} "
            f"--skip-matlab "
            f"--python-parity-rerun-from edges"
        )
        commands.append(cmd)

    # Stage-isolated network gate
    if report.safe_to_reuse and report.artifact_checks.get("matlab_batch_present", False):
        cmd = (
            f"python dev/scripts/cli/compare_matlab_python.py "
            f"--input {input_file} "
            f"--output-dir {run_root} "
            f"--skip-matlab "
            f"--python-parity-rerun-from network"
        )
        commands.append(cmd)

    return commands


def format_missing_artifact_explanation(
    artifact_name: str,
    *,
    required_for: str,
) -> str:
    """
    Format explanation for a missing artifact.

    Args:
        artifact_name: Name of the missing artifact
        required_for: Description of what the artifact is required for

    Returns:
        Formatted explanation string
    """
    return f"{artifact_name}: Required for {required_for}"


def generate_next_action_commands(
    report: LoopAssessmentReport,
    *,
    run_root: Path,
    input_file: Path,
) -> list[str]:
    """
    Generate next action command based on workflow state.

    Args:
        report: Loop assessment report with workflow decision surface
        run_root: Path to the staged run root
        input_file: Path to the input volume file

    Returns:
        List of recommended next action commands
    """
    commands = []

    # Fresh MATLAB run required
    if report.requires_fresh_matlab:
        cmd = (
            f"python dev/scripts/cli/compare_matlab_python.py "
            f"--input {input_file} "
            f"--output-dir {run_root}"
        )
        commands.append(cmd)

    # Blocked - suggest fresh run root
    if report.verdict == "blocked" and not report.input_compatible:
        # Suggest creating a new run root
        cmd = (
            f"python dev/scripts/cli/compare_matlab_python.py "
            f"--input {input_file} "
            f"--output-dir <new_run_root>"
        )
        commands.append(cmd)

    # Analysis ready - suggest analysis command
    if report.verdict == "analysis_ready" and report.safe_to_analyze_only:
        from .run_layout import resolve_run_layout

        layout = resolve_run_layout(run_root)
        matlab_dir = layout["matlab_dir"]
        python_dir = layout["python_dir"]

        if matlab_dir.exists() and python_dir.exists():
            cmd = (
                f"python dev/scripts/cli/compare_matlab_python.py "
                f"--standalone-matlab-dir {matlab_dir} "
                f"--standalone-python-dir {python_dir} "
                f"--output-dir {run_root}"
            )
            commands.append(cmd)

    return commands
