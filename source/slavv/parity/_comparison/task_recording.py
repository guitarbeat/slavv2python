"""Run-snapshot task recording and progress formatting helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...matlab_status import MatlabStatusReport, summarize_matlab_status
from ...preflight import OutputRootPreflightReport, summarize_output_preflight
from ...utils import format_time
from ...workflow_assessment import (
    LOOP_BLOCKED,
    LOOP_FRESH_MATLAB_REQUIRED,
    LoopAssessmentReport,
    MatlabHealthCheckReport,
    summarize_loop_assessment,
    summarize_matlab_health_check,
)

if TYPE_CHECKING:
    from pathlib import Path

    from ...runtime import ProgressEvent, RunContext


def _format_progress_event_message(event: ProgressEvent) -> str:
    """Render progress events into concise, human-readable console output."""
    stage_snapshot = event.snapshot.stages.get(event.stage)
    stage_label = event.stage.replace("_", " ").title()
    detail = (event.detail or "").strip()
    pieces = [f"{stage_label} {event.stage_progress * 100:.1f}%"]

    if detail:
        pieces.append(detail)

    if stage_snapshot is not None and stage_snapshot.units_total > 0:
        pieces.append(
            f"{stage_snapshot.units_completed:,}/{stage_snapshot.units_total:,} work units"
        )
        if stage_snapshot.eta_seconds is not None:
            pieces.append(f"ETA {format_time(stage_snapshot.eta_seconds)}")

    if event.resumed:
        pieces.append("resumed")

    return " | ".join(pieces)


def _record_loop_assessment_task(
    comparison_context: RunContext,
    report: LoopAssessmentReport,
    report_path: Path | None,
) -> None:
    """Mirror the workflow decision into the shared run snapshot."""
    task_status = "completed"
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

    comparison_context.update_optional_task(
        "workflow_assessment",
        status=task_status,
        detail=summarize_loop_assessment(report),
        artifacts=artifacts,
    )


def _record_preflight_task(
    comparison_context: RunContext,
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
    comparison_context.update_optional_task(
        "output_preflight",
        status="completed" if report.allows_launch else "failed",
        detail=summarize_output_preflight(report),
        artifacts=artifacts,
    )


def _record_matlab_health_check_task(
    comparison_context: RunContext,
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
        detail=summarize_matlab_health_check(report),
        artifacts=artifacts,
    )


def _record_matlab_status_task(
    comparison_context: RunContext,
    report: MatlabStatusReport,
    report_path: Path | None,
    failure_summary_path: Path | None = None,
    *,
    python_force_rerun_from: str | None = None,
    matlab_launch_skipped_reason: str | None = None,
    matlab_reuse_mode: str | None = None,
) -> None:
    """Mirror normalized MATLAB resume semantics into the shared run snapshot."""
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
