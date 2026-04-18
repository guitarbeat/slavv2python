"""Comparison report and artifact persistence helpers."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from .._persistence import (
    _json_default,
    _json_default_with_string_fallback,
    write_json_file,
    write_kv_tsv,
)
from ..preflight import OutputRootPreflightReport, persist_output_preflight
from ..workflow_assessment import (
    LoopAssessmentReport,
    MatlabHealthCheckReport,
    persist_loop_assessment,
    persist_matlab_health_check,
)

_comparison_report_default = _json_default_with_string_fallback


def _build_serializable_comparison_report(comparison: dict[str, Any]) -> dict[str, Any]:
    """Normalize comparison output into the persisted report surface."""
    return {
        "matlab": copy.deepcopy(comparison["matlab"]),
        "python": copy.deepcopy(comparison["python"]),
        "performance": comparison["performance"],
        "vertices": comparison.get("vertices", {}),
        "edges": comparison.get("edges", {}),
        "network": comparison.get("network", {}),
        "parity_gate": comparison.get("parity_gate", {}),
    }


def _comparison_count(
    comparison: dict[str, Any],
    side: str,
    top_level_key: str,
    section: str,
    nested_key: str,
) -> int:
    """Read a side-specific count with nested fallbacks."""
    value = comparison.get(side, {}).get(top_level_key)
    if value is not None:
        return int(value)
    nested = comparison.get(section, {}).get(nested_key, 0)
    return int(nested)


def _build_comparison_quick_view(comparison: dict[str, Any]) -> dict[str, Any]:
    """Build a compact scalar metrics surface for quick diff comparisons."""
    matlab_vertices = _comparison_count(
        comparison, "matlab", "vertices_count", "vertices", "matlab_count"
    )
    python_vertices = _comparison_count(
        comparison, "python", "vertices_count", "vertices", "python_count"
    )
    matlab_edges = _comparison_count(comparison, "matlab", "edges_count", "edges", "matlab_count")
    python_edges = _comparison_count(comparison, "python", "edges_count", "edges", "python_count")
    matlab_strands = _comparison_count(
        comparison, "matlab", "strand_count", "network", "matlab_strand_count"
    )
    python_strands = _comparison_count(
        comparison, "python", "network_strands_count", "network", "python_strand_count"
    )

    return {
        "edges_diff": python_edges - matlab_edges,
        "edges_exact": bool(comparison.get("edges", {}).get("exact_match", False)),
        "edges_matlab": matlab_edges,
        "edges_python": python_edges,
        "matlab_elapsed_seconds": float(
            comparison.get("matlab", {}).get("elapsed_time", 0.0) or 0.0
        ),
        "network_strands_diff": python_strands - matlab_strands,
        "network_strands_exact": bool(comparison.get("network", {}).get("exact_match", False)),
        "network_strands_matlab": matlab_strands,
        "network_strands_python": python_strands,
        "parity_gate_passed": bool(comparison.get("parity_gate", {}).get("passed", False)),
        "python_elapsed_seconds": float(
            comparison.get("python", {}).get("elapsed_time", 0.0) or 0.0
        ),
        "python_vs_matlab_time_delta_seconds": float(
            (comparison.get("python", {}).get("elapsed_time", 0.0) or 0.0)
            - (comparison.get("matlab", {}).get("elapsed_time", 0.0) or 0.0)
        ),
        "speedup": float(comparison.get("performance", {}).get("speedup", 0.0) or 0.0),
        "vertices_diff": python_vertices - matlab_vertices,
        "vertices_exact": bool(comparison.get("vertices", {}).get("exact_match", False)),
        "vertices_matlab": matlab_vertices,
        "vertices_python": python_vertices,
    }


def _write_comparison_quick_view(
    comparison: dict[str, Any], analysis_dir: Path
) -> tuple[Path, Path]:
    """Write compact sidecar artifacts that are easy to diff across runs."""
    quick_view = _build_comparison_quick_view(comparison)
    quick_json = analysis_dir / "comparison_quick_view.json"
    quick_tsv = analysis_dir / "comparison_quick_view.tsv"
    write_json_file(quick_json, quick_view)
    write_kv_tsv(quick_tsv, quick_view)

    return quick_json, quick_tsv


def _write_comparison_report(comparison: dict[str, Any], report_file: Path) -> Path:
    """Persist the normalized comparison report JSON."""
    write_json_file(
        report_file,
        _build_serializable_comparison_report(comparison),
        default=_json_default_with_string_fallback,
    )
    print(f"\nComparison report saved to: {report_file}")
    return report_file


def _write_normalized_params_file(metadata_dir: Path, params: dict[str, Any]) -> Path:
    """Persist the normalized comparison parameters for both MATLAB and Python runs."""
    metadata_dir.mkdir(parents=True, exist_ok=True)
    params_path = metadata_dir / "comparison_params.normalized.json"
    write_json_file(params_path, params, default=_json_default)
    return params_path


def _persist_preflight_report(
    report: OutputRootPreflightReport,
    metadata_dir: Path,
) -> Path | None:
    """Best-effort persistence for output-root preflight metadata."""
    try:
        return persist_output_preflight(report, metadata_dir)
    except OSError as exc:
        print(f"Warning: Could not persist output preflight report: {exc}")
        return None


def _persist_loop_assessment_report(
    report: LoopAssessmentReport,
    metadata_dir: Path,
) -> Path | None:
    """Best-effort persistence for workflow-loop assessments."""
    try:
        return persist_loop_assessment(report, metadata_dir)
    except OSError as exc:
        print(f"Warning: Could not persist loop assessment report: {exc}")
        return None


def _persist_matlab_status_report(
    report: Any,
    metadata_dir: Path,
) -> Path | None:
    """Best-effort persistence for normalized MATLAB resume metadata."""
    from ..matlab_status import persist_matlab_status

    try:
        return persist_matlab_status(report, metadata_dir)
    except OSError as exc:
        print(f"Warning: Could not persist MATLAB status report: {exc}")
        return None


def _persist_matlab_failure_report(
    report: Any,
    metadata_dir: Path,
) -> Path | None:
    """Best-effort persistence for MATLAB failure summaries."""
    from ..matlab_status import persist_matlab_failure_summary

    try:
        return persist_matlab_failure_summary(report, metadata_dir)
    except OSError as exc:
        print(f"Warning: Could not persist MATLAB failure summary: {exc}")
        return None


def _persist_matlab_health_check_report(
    report: MatlabHealthCheckReport,
    metadata_dir: Path,
) -> Path | None:
    """Best-effort persistence for MATLAB health-check metadata."""
    try:
        return persist_matlab_health_check(report, metadata_dir)
    except OSError as exc:
        print(f"Warning: Could not persist MATLAB health check report: {exc}")
        return None


def _record_loop_assessment_task(
    comparison_context: Any,
    report: LoopAssessmentReport,
    report_path: Path | None,
) -> None:
    """Mirror the workflow decision into the shared run snapshot."""
    task_status = "completed"
    from ..workflow_assessment import LOOP_BLOCKED, LOOP_FRESH_MATLAB_REQUIRED

    if report.verdict == LOOP_BLOCKED:
        task_status = "failed"
    elif report.verdict == LOOP_FRESH_MATLAB_REQUIRED:
        task_status = "running"

    artifacts = {
        "requested_loop": report.requested_loop,
        "verdict": report.verdict,
        "safe_to_reuse": str(report.safe_to_reuse).lower(),
        "safe_to_analyze_only": str(report.safe_to_analyze_only).lower(),
        "requires_fresh_matlab": str(report.requires_fresh_matlab).lower(),
    }
    if report_path is not None:
        artifacts["report"] = str(report_path)

    from ..workflow_assessment import summarize_loop_assessment

    comparison_context.update_optional_task(
        "workflow_assessment",
        status=task_status,
        detail=summarize_loop_assessment(report),
        artifacts=artifacts,
    )


def _record_preflight_task(
    comparison_context: Any,
    report: OutputRootPreflightReport,
    report_path: Path | None,
) -> None:
    """Mirror the preflight decision into the shared run snapshot."""
    artifacts = {
        "output_root": report.resolved_output_root or report.output_root,
        "preflight_status": report.preflight_status,
    }
    if report_path is not None:
        artifacts["report"] = str(report_path)
    if report.free_space_gb is not None:
        artifacts["free_space_gb"] = f"{report.free_space_gb:.2f}"

    from ..preflight import summarize_output_preflight

    comparison_context.update_optional_task(
        "output_preflight",
        status="completed" if report.allows_launch else "failed",
        detail=summarize_output_preflight(report),
        artifacts=artifacts,
    )


def _record_matlab_health_check_task(
    comparison_context: Any,
    report: MatlabHealthCheckReport,
    report_path: Path | None,
) -> None:
    """Mirror MATLAB health-check results into the shared run snapshot."""
    artifacts = {
        "matlab_path": report.matlab_path,
        "exit_code": str(report.exit_code if report.exit_code is not None else ""),
        "timed_out": str(report.timed_out).lower(),
    }
    if report_path is not None:
        artifacts["report"] = str(report_path)

    comparison_context.update_optional_task(
        "matlab_health_check",
        status="completed" if report.success else "failed",
        detail=_summarize_matlab_health_check(report),
        artifacts=artifacts,
    )


def _record_matlab_status_task(
    comparison_context: Any,
    report: Any,
    report_path: Path | None,
    failure_summary_path: Path | None = None,
    *,
    python_force_rerun_from: str | None = None,
    matlab_launch_skipped_reason: str | None = None,
    matlab_reuse_mode: str | None = None,
) -> None:
    """Mirror normalized MATLAB resume semantics into the shared run snapshot."""
    from ..matlab_status import summarize_matlab_status

    task_status = "failed" if report.failure_summary else "completed"
    if report.stale_running_snapshot_suspected and not report.failure_summary:
        task_status = "failed"

    artifacts = {
        "resume_mode": report.matlab_resume_mode,
        "last_completed_stage": report.matlab_last_completed_stage or "(none)",
        "next_stage": report.matlab_next_stage or "(none)",
        "rerun_prediction": report.matlab_rerun_prediction,
        "batch_folder": report.matlab_batch_folder or "",
        "resume_state_file": report.matlab_resume_state_file,
        "log_file": report.matlab_log_file,
    }
    if report_path is not None:
        artifacts["report"] = str(report_path)
    if failure_summary_path is not None:
        artifacts["failure_summary_file"] = str(failure_summary_path)
    if python_force_rerun_from is not None:
        artifacts["python_force_rerun_from"] = python_force_rerun_from
    if matlab_launch_skipped_reason is not None:
        artifacts["matlab_launch"] = "skipped"
        artifacts["matlab_launch_skip_reason"] = matlab_launch_skipped_reason
    if matlab_reuse_mode is not None:
        artifacts["matlab_reuse_mode"] = matlab_reuse_mode

    comparison_context.update_optional_task(
        "matlab_status",
        status=task_status,
        detail=summarize_matlab_status(report),
        artifacts=artifacts,
    )


def _summarize_matlab_health_check(report: MatlabHealthCheckReport) -> str:
    from ..workflow_assessment import summarize_matlab_health_check

    return summarize_matlab_health_check(report)
