"""MATLAB health-check workflow helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from slavv.runtime import RunContext

if TYPE_CHECKING:
    from pathlib import Path


def run_matlab_health_check_workflow(
    *,
    output_dir: Path,
    matlab_path: str,
    project_root: Path,
    resolve_run_layout_fn,
    assess_loop_request_fn,
    persist_loop_assessment_report_fn,
    record_loop_assessment_task_fn,
    evaluate_output_root_preflight_for_workflow_fn,
    persist_preflight_report_fn,
    summarize_output_preflight_fn,
    record_preflight_task_fn,
    generate_post_comparison_artifacts_fn,
    run_matlab_health_check_fn,
    persist_matlab_health_check_report_fn,
    record_matlab_health_check_task_fn,
    summarize_matlab_health_check_fn,
    generate_summary_fn,
    generate_manifest_fn,
) -> int:
    """Run a lightweight MATLAB health check and persist staged metadata."""
    layout = resolve_run_layout_fn(output_dir)
    run_root = layout["run_root"]
    analysis_dir = layout["analysis_dir"]
    metadata_dir = layout["metadata_dir"]
    analysis_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    comparison_context = RunContext(
        run_dir=run_root,
        target_stage="preflight",
        provenance={"source": "matlab-health-check"},
    )
    loop_assessment = assess_loop_request_fn(run_root, loop_kind="validate_only")
    loop_assessment_path = persist_loop_assessment_report_fn(loop_assessment, metadata_dir)
    record_loop_assessment_task_fn(comparison_context, loop_assessment, loop_assessment_path)

    preflight_report = evaluate_output_root_preflight_for_workflow_fn(run_root, metadata_dir)
    preflight_report_path = persist_preflight_report_fn(preflight_report, metadata_dir)

    print("\n" + "=" * 60)
    print("Output Root Preflight")
    print("=" * 60)
    print(summarize_output_preflight_fn(preflight_report))
    record_preflight_task_fn(comparison_context, preflight_report, preflight_report_path)

    if not preflight_report.allows_launch:
        comparison_context.mark_run_status(
            "failed",
            current_stage="preflight",
            detail=summarize_output_preflight_fn(preflight_report),
        )
        generate_post_comparison_artifacts_fn(
            run_dir=run_root,
            analysis_dir=analysis_dir,
            metadata_dir=metadata_dir,
            comparison_context=comparison_context,
            include_summary=False,
            announce_manifest=True,
            generate_summary_fn=generate_summary_fn,
            generate_manifest_fn=generate_manifest_fn,
        )
        print(f"Recommended action: {preflight_report.recommended_action}")
        return 1

    health_report = run_matlab_health_check_fn(
        output_root=run_root,
        matlab_path=matlab_path,
        project_root=project_root,
    )
    health_report_path = persist_matlab_health_check_report_fn(health_report, metadata_dir)
    record_matlab_health_check_task_fn(comparison_context, health_report, health_report_path)

    print("\n" + "=" * 60)
    print("MATLAB Health Check")
    print("=" * 60)
    print(summarize_matlab_health_check_fn(health_report))

    comparison_context.mark_run_status(
        "completed" if health_report.success else "failed",
        current_stage="matlab_health_check",
        detail=summarize_matlab_health_check_fn(health_report),
    )
    generate_post_comparison_artifacts_fn(
        run_dir=run_root,
        analysis_dir=analysis_dir,
        metadata_dir=metadata_dir,
        comparison_context=comparison_context,
        include_summary=False,
        announce_manifest=True,
        generate_summary_fn=generate_summary_fn,
        generate_manifest_fn=generate_manifest_fn,
    )
    return 0 if health_report.success else 1
