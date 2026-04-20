"""
Comparison execution module.

This module handles the execution of MATLAB and Python vectorization pipelines
for comparison purposes.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

from slavv.core import SLAVVProcessor
from slavv.io import export_pipeline_results, load_tiff_volume
from slavv.io.matlab_bridge import import_matlab_batch, load_matlab_batch_params
from slavv.io.matlab_parser import load_matlab_batch_results
from slavv.runtime import RunContext
from slavv.utils import get_matlab_info, get_system_info
from slavv.visualization import NetworkVisualizer

from ._comparison.analysis import (
    _generate_post_comparison_artifacts,
    _run_comparison_analysis,
)
from ._comparison.artifacts import (
    _build_comparison_quick_view,
    _build_serializable_comparison_report,
    _comparison_count,
    _comparison_report_default,
    _json_default,
    _persist_loop_assessment_report,
    _persist_matlab_failure_report,
    _persist_matlab_health_check_report,
    _persist_matlab_status_report,
    _persist_preflight_report,
    _write_comparison_quick_view,
    _write_comparison_report,
    _write_normalized_params_file,
)
from ._comparison.config import (
    discover_matlab_artifacts as _discover_matlab_artifacts_impl,
)
from ._comparison.config import load_parameters as _load_parameters_impl
from ._comparison.health_check import (
    run_matlab_health_check_workflow as _run_matlab_health_check_workflow_impl,
)
from ._comparison.matlab_runner import run_matlab_vectorization as _run_matlab_vectorization_impl
from ._comparison.python_runner import run_python_vectorization as _run_python_vectorization_impl
from ._comparison.python_sources import (
    _load_python_candidate_audit,
    _load_python_candidate_edges,
    _load_python_candidate_lifecycle,
    _load_python_results_from_source,
    _resolve_python_energy_source,
)
from ._comparison.reuse import _print_reuse_guidance, _resolve_python_parity_import_plan
from ._comparison.standalone import (
    run_standalone_comparison as _run_standalone_comparison_impl,
)
from ._comparison.task_recording import (
    _format_progress_event_message,
    _record_loop_assessment_task,
    _record_matlab_health_check_task,
    _record_matlab_status_task,
    _record_preflight_task,
)
from .matlab_status import (
    MatlabStatusReport,
    summarize_matlab_status,
)
from .matlab_status import (
    inspect_matlab_status as _inspect_matlab_status_direct,
)
from .metrics import build_shared_neighborhood_audit, compare_results
from .preflight import (
    OutputRootPreflightReport,
    summarize_output_preflight,
)
from .preflight import (
    evaluate_output_root_preflight as _evaluate_output_root_preflight_direct,
)
from .reporting import generate_summary
from .run_layout import generate_manifest, resolve_run_layout
from .workflow_assessment import (
    LOOP_ANALYSIS_READY,
    LOOP_REUSE_READY,
    assess_loop_request,
    determine_loop_kind,
    evaluate_output_root_preflight_cached,
    inspect_matlab_status_cached,
    run_matlab_health_check,
    summarize_matlab_health_check,
)

__all__ = [
    "_build_comparison_quick_view",
    "_build_serializable_comparison_report",
    "_comparison_count",
    "_comparison_report_default",
    "_format_progress_event_message",
    "_json_default",
    "_load_python_candidate_audit",
    "_load_python_candidate_edges",
    "_load_python_candidate_lifecycle",
    "_load_python_results_from_source",
    "_persist_loop_assessment_report",
    "_persist_matlab_failure_report",
    "_persist_matlab_health_check_report",
    "_persist_matlab_status_report",
    "_persist_preflight_report",
    "_record_loop_assessment_task",
    "_record_matlab_health_check_task",
    "_record_matlab_status_task",
    "_record_preflight_task",
    "_resolve_python_energy_source",
    "_write_comparison_quick_view",
    "_write_comparison_report",
    "_write_normalized_params_file",
    "discover_matlab_artifacts",
    "evaluate_output_root_preflight",
    "inspect_matlab_status",
    "load_parameters",
    "orchestrate_comparison",
    "run_matlab_health_check_workflow",
    "run_matlab_vectorization",
    "run_python_vectorization",
    "run_standalone_comparison",
]

_PYTHON_RESULT_SOURCE_CHOICES = {
    "auto",
    "checkpoints-only",
    "export-json-only",
    "network-json-only",
}


def evaluate_output_root_preflight(output_root: str | Path) -> OutputRootPreflightReport:
    """Backward-compatible public preflight hook for callers and tests."""
    return _evaluate_output_root_preflight_direct(output_root)


def inspect_matlab_status(
    output_directory: str | Path,
    *,
    input_file: str | None = None,
) -> MatlabStatusReport:
    """Backward-compatible public MATLAB status hook for callers and tests."""
    return _inspect_matlab_status_direct(output_directory, input_file=input_file)


_DEFAULT_EVALUATE_OUTPUT_ROOT_PREFLIGHT_HOOK = evaluate_output_root_preflight
_DEFAULT_INSPECT_MATLAB_STATUS_HOOK = inspect_matlab_status


def _evaluate_output_root_preflight_for_workflow(
    output_root: str | Path,
    metadata_dir: Path,
) -> OutputRootPreflightReport:
    """Use cached preflight by default while preserving the monkeypatch surface."""
    preflight_hook = globals()["evaluate_output_root_preflight"]
    if preflight_hook is not _DEFAULT_EVALUATE_OUTPUT_ROOT_PREFLIGHT_HOOK:
        return preflight_hook(output_root)
    return evaluate_output_root_preflight_cached(output_root, metadata_dir)


def _inspect_matlab_status_for_workflow(
    output_directory: str | Path,
    metadata_dir: Path,
    *,
    input_file: str | None = None,
) -> MatlabStatusReport:
    """Use cached MATLAB status inspection by default while preserving tests."""
    status_hook = globals()["inspect_matlab_status"]
    if status_hook is not _DEFAULT_INSPECT_MATLAB_STATUS_HOOK:
        return status_hook(output_directory, input_file=input_file)
    return inspect_matlab_status_cached(output_directory, metadata_dir, input_file=input_file)


def _finalize_run_status_before_post_artifacts(
    comparison_context: RunContext,
    *,
    matlab_results: dict[str, Any] | None,
    python_results: dict[str, Any] | None,
) -> None:
    """Persist a terminal snapshot state before archive maintenance inspects the run."""
    terminal_statuses = {"completed", "completed_target", "failed", "resume_blocked"}
    if str(comparison_context.snapshot.status or "").strip().lower() in terminal_statuses:
        return

    if matlab_results is not None and python_results is not None:
        matlab_success = bool(matlab_results.get("success"))
        python_success = bool(python_results.get("success"))
        if matlab_success and python_success:
            comparison_context.mark_run_status(
                "completed",
                current_stage="comparison_analysis",
                detail="Comparison workflow completed successfully.",
            )
            return

        if not matlab_success:
            matlab_status = matlab_results.get("matlab_status") or {}
            detail = str(
                matlab_status.get("failure_summary")
                or matlab_results.get("error")
                or "MATLAB workflow failed."
            )
            comparison_context.mark_run_status("failed", current_stage="matlab", detail=detail)
            return

        detail = str(python_results.get("error") or "Python workflow failed.")
        comparison_context.mark_run_status("failed", current_stage="python", detail=detail)
        return

    if matlab_results is not None:
        if bool(matlab_results.get("success")):
            comparison_context.mark_run_status(
                "completed",
                current_stage="matlab",
                detail="MATLAB workflow completed successfully.",
            )
            return
        matlab_status = matlab_results.get("matlab_status") or {}
        detail = str(
            matlab_status.get("failure_summary")
            or matlab_results.get("error")
            or "MATLAB workflow failed."
        )
        comparison_context.mark_run_status("failed", current_stage="matlab", detail=detail)
        return

    if python_results is not None:
        if bool(python_results.get("success")):
            comparison_context.mark_run_status(
                "completed",
                current_stage="python",
                detail="Python workflow completed successfully.",
            )
            return
        detail = str(python_results.get("error") or "Python workflow failed.")
        comparison_context.mark_run_status("failed", current_stage="python", detail=detail)


def _bootstrap_existing_matlab_batch_for_python_parity(
    *,
    matlab_output: Path,
    python_output: Path,
    input_file: str,
    params_for_python: dict[str, Any],
    comparison_context: RunContext | None,
    metadata_dir: Path,
    python_parity_rerun_from: str = "edges",
) -> tuple[dict[str, Any] | None, str | None]:
    """Reuse an existing MATLAB batch to seed a skip-MATLAB Python parity rerun."""
    matlab_status_report = _inspect_matlab_status_for_workflow(
        matlab_output,
        metadata_dir,
        input_file=input_file,
    )
    matlab_status_report_path = _persist_matlab_status_report(matlab_status_report, metadata_dir)
    if comparison_context is not None:
        _record_matlab_status_task(
            comparison_context,
            matlab_status_report,
            matlab_status_report_path,
        )

    print("\n" + "=" * 60)
    print("MATLAB Reuse Semantics")
    print("=" * 60)
    print(summarize_matlab_status(matlab_status_report))

    batch_folder = matlab_status_report.matlab_batch_folder
    if not batch_folder:
        print(
            "Warning: --skip-matlab requested but no reusable MATLAB batch was found in the staged input surface."
        )
        return None, None

    matlab_results = {
        "success": True,
        "elapsed_time": 0.0,
        "output_dir": str(matlab_output),
        "batch_folder": batch_folder,
        "matlab_status": matlab_status_report.to_dict(),
    }

    checkpoint_dir = python_output / "checkpoints"
    python_force_rerun_from: str | None = None
    import_stages, requested_rerun_from = _resolve_python_parity_import_plan(
        python_parity_rerun_from
    )
    if comparison_context is not None:
        comparison_context.update_optional_task(
            "matlab_import",
            status="running",
            detail=(
                "Importing existing MATLAB "
                + ", ".join(import_stages)
                + f" for a skip-MATLAB parity rerun from {requested_rerun_from}"
            ),
            artifacts={
                "batch_folder": batch_folder,
                "python_parity_rerun_from": requested_rerun_from,
            },
        )

    try:
        imported = import_matlab_batch(
            batch_folder,
            checkpoint_dir,
            stages=import_stages,
        )
    except Exception as exc:
        if comparison_context is not None:
            comparison_context.update_optional_task(
                "matlab_import",
                status="failed",
                detail=f"Existing MATLAB checkpoint import failed: {exc}",
                artifacts={"batch_folder": batch_folder},
            )
        print(f"Warning: Could not import existing MATLAB checkpoints: {exc}")
        return matlab_results, None

    imported_stages = set(imported)
    if set(import_stages).issubset(imported_stages):
        python_force_rerun_from = requested_rerun_from

    try:
        params_for_python |= load_matlab_batch_params(batch_folder)
    except Exception as exc:
        if comparison_context is not None:
            comparison_context.update_optional_task(
                "matlab_import",
                status="completed",
                detail=f"Imported existing MATLAB checkpoints; settings overlay unavailable: {exc}",
                artifacts=dict(imported),
            )
    else:
        if comparison_context is not None:
            comparison_context.update_optional_task(
                "matlab_import",
                status="completed",
                detail=(
                    "Imported existing MATLAB checkpoints for skip-MATLAB parity rerun "
                    f"from {requested_rerun_from}"
                ),
                artifacts={
                    **dict(imported),
                    "python_parity_rerun_from": requested_rerun_from,
                    **(
                        {"python_force_rerun_from": python_force_rerun_from}
                        if python_force_rerun_from is not None
                        else {}
                    ),
                },
            )
    if comparison_context is not None:
        _record_matlab_status_task(
            comparison_context,
            matlab_status_report,
            matlab_status_report_path,
            python_force_rerun_from=python_force_rerun_from,
        )
    return matlab_results, python_force_rerun_from


def load_parameters(params_file: str | None = None) -> dict[str, Any]:
    """Load parameters from JSON file or use defaults."""
    return _load_parameters_impl(params_file)


def discover_matlab_artifacts(output_dir: str | Path) -> dict[str, Any]:
    """Discover the newest MATLAB batch folder and key output artifacts."""
    return _discover_matlab_artifacts_impl(output_dir)


def run_matlab_vectorization(
    input_file: str,
    output_dir: str,
    matlab_path: str,
    project_root: Path,
    batch_script: str | None = None,
    params_file: str | None = None,
) -> dict[str, Any]:
    """Run MATLAB vectorization via CLI."""
    return _run_matlab_vectorization_impl(
        input_file,
        output_dir,
        matlab_path,
        project_root,
        batch_script=batch_script,
        params_file=params_file,
        discover_matlab_artifacts=discover_matlab_artifacts,
        get_matlab_info_fn=get_matlab_info,
        get_system_info_fn=get_system_info,
    )


def run_python_vectorization(
    input_file: str,
    output_dir: str,
    params: dict[str, Any],
    run_dir: str | None = None,
    force_rerun_from: str | None = None,
    minimal_exports: bool = False,
) -> dict[str, Any]:
    """Run Python vectorization."""
    return _run_python_vectorization_impl(
        input_file,
        output_dir,
        params,
        run_dir=run_dir,
        force_rerun_from=force_rerun_from,
        minimal_exports=minimal_exports,
        get_system_info_fn=get_system_info,
        load_tiff_volume_fn=load_tiff_volume,
        processor_factory=SLAVVProcessor,
        format_progress_event_message_fn=_format_progress_event_message,
        resolve_python_energy_source_fn=_resolve_python_energy_source,
        export_pipeline_results_fn=export_pipeline_results,
        visualizer_factory=NetworkVisualizer,
        load_python_candidate_edges_fn=_load_python_candidate_edges,
        load_python_candidate_audit_fn=_load_python_candidate_audit,
        load_python_candidate_lifecycle_fn=_load_python_candidate_lifecycle,
    )


def orchestrate_comparison(
    input_file: str,
    output_dir: Path,
    matlab_path: str,
    project_root: Path,
    params: dict[str, Any],
    skip_matlab: bool = False,
    skip_python: bool = False,
    validate_only: bool = False,
    minimal_exports: bool = False,
    comparison_depth: str = "deep",
    python_result_source: str = "auto",
    python_parity_rerun_from: str = "edges",
) -> int:
    """Run full comparison workflow."""
    layout = resolve_run_layout(output_dir)
    run_root = layout["run_root"]
    matlab_output = layout["matlab_dir"]
    python_output = layout["python_dir"]
    analysis_dir = layout["analysis_dir"]
    metadata_dir = run_root / "99_Metadata"
    import_stages, requested_python_rerun_from = _resolve_python_parity_import_plan(
        python_parity_rerun_from
    )
    params_for_python = copy.deepcopy(params)
    params_for_python["comparison_exact_network"] = True
    params_for_python["python_parity_rerun_from"] = requested_python_rerun_from
    loop_kind = determine_loop_kind(
        standalone_mode=False,
        validate_only=validate_only,
        skip_matlab=skip_matlab,
        skip_python=skip_python,
        python_parity_rerun_from=requested_python_rerun_from,
    )
    comparison_context = RunContext(
        run_dir=run_root,
        target_stage="network",
        provenance={"source": "comparison", "input_file": input_file},
    )
    loop_assessment = assess_loop_request(
        run_root,
        loop_kind=loop_kind,
        input_path=Path(input_file) if input_file else None,
        params=params_for_python,
    )
    loop_assessment_path = _persist_loop_assessment_report(loop_assessment, metadata_dir)
    _record_loop_assessment_task(comparison_context, loop_assessment, loop_assessment_path)
    python_force_rerun_from: str | None = None
    preflight_report: OutputRootPreflightReport | None = None
    matlab_status_report: MatlabStatusReport | None = None
    auto_matlab_reuse_mode: str | None = None

    if validate_only:
        preflight_report = _evaluate_output_root_preflight_for_workflow(run_root, metadata_dir)
        preflight_report_path = _persist_preflight_report(preflight_report, metadata_dir)
        print("\n" + "=" * 60)
        print("Output Root Preflight")
        print("=" * 60)
        print(summarize_output_preflight(preflight_report))
        for warning in preflight_report.warnings:
            print(f"WARNING: {warning}")
        for error in preflight_report.errors:
            print(f"ERROR: {error}")

        _record_preflight_task(comparison_context, preflight_report, preflight_report_path)
        if not preflight_report.allows_launch:
            comparison_context.mark_run_status(
                "failed",
                current_stage="preflight",
                detail=summarize_output_preflight(preflight_report),
            )
            print(f"Recommended action: {preflight_report.recommended_action}")
            return 1

        print("\nValidation-only mode enabled.")
        print("Preflight checks passed; skipping MATLAB/Python execution.")
        comparison_context.mark_run_status(
            "completed",
            current_stage="preflight",
            detail="Validation-only mode completed successfully.",
        )
        _generate_post_comparison_artifacts(
            run_dir=run_root,
            analysis_dir=analysis_dir,
            metadata_dir=metadata_dir,
            comparison_context=comparison_context,
            include_summary=False,
            announce_manifest=True,
            generate_summary_fn=generate_summary,
            generate_manifest_fn=generate_manifest,
        )

        # Display reuse eligibility summary after validation
        _print_reuse_guidance(
            run_root,
            input_file=input_file,
            params=params_for_python,
            loop_kind=loop_kind,
        )

        return 0

    if not skip_matlab:
        preflight_report = _evaluate_output_root_preflight_for_workflow(run_root, metadata_dir)
        preflight_report_path = _persist_preflight_report(preflight_report, metadata_dir)

        print("\n" + "=" * 60)
        print("Output Root Preflight")
        print("=" * 60)
        print(summarize_output_preflight(preflight_report))
        for warning in preflight_report.warnings:
            print(f"WARNING: {warning}")
        for error in preflight_report.errors:
            print(f"ERROR: {error}")

        if preflight_report.allows_launch or preflight_report_path is not None:
            _record_preflight_task(comparison_context, preflight_report, preflight_report_path)

        if not preflight_report.allows_launch:
            if comparison_context is not None:
                comparison_context.mark_run_status(
                    "failed",
                    current_stage="preflight",
                    detail=summarize_output_preflight(preflight_report),
                )
                _generate_post_comparison_artifacts(
                    run_dir=run_root,
                    analysis_dir=analysis_dir,
                    metadata_dir=metadata_dir,
                    comparison_context=comparison_context,
                    include_summary=False,
                    announce_manifest=True,
                    generate_summary_fn=generate_summary,
                    generate_manifest_fn=generate_manifest,
                )
            print(f"Recommended action: {preflight_report.recommended_action}")
            return 1

    analysis_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    normalized_params_file = _write_normalized_params_file(metadata_dir, params_for_python)
    if preflight_report is not None:
        _record_preflight_task(
            comparison_context, preflight_report, metadata_dir / "output_preflight.json"
        )

    if not skip_matlab:
        matlab_status_report = _inspect_matlab_status_for_workflow(
            matlab_output,
            metadata_dir,
            input_file=input_file,
        )
        matlab_status_report_path = _persist_matlab_status_report(
            matlab_status_report, metadata_dir
        )
        _record_matlab_status_task(
            comparison_context,
            matlab_status_report,
            matlab_status_report_path,
        )
        print("\n" + "=" * 60)
        print("MATLAB Rerun Semantics")
        print("=" * 60)
        print(summarize_matlab_status(matlab_status_report))

        completed_matlab_batch = (
            bool(matlab_status_report.matlab_batch_complete)
            or matlab_status_report.matlab_resume_mode == "complete-noop"
        ) and bool(matlab_status_report.matlab_batch_folder)
        if completed_matlab_batch:
            if loop_assessment.verdict == LOOP_ANALYSIS_READY and not skip_python:
                auto_matlab_reuse_mode = "analysis-only"
            elif loop_assessment.verdict == LOOP_REUSE_READY:
                auto_matlab_reuse_mode = "python-rerun"

        if auto_matlab_reuse_mode is not None:
            comparison_context.update_optional_task(
                "matlab_pipeline",
                status="completed",
                detail=(
                    "MATLAB launch skipped due to completed reusable batch "
                    f"({auto_matlab_reuse_mode})."
                ),
                artifacts={
                    "launch": "skipped",
                    "skip_reason": "completed_reusable_batch",
                    "reuse_mode": auto_matlab_reuse_mode,
                    "batch_folder": matlab_status_report.matlab_batch_folder or "",
                },
            )
            _record_matlab_status_task(
                comparison_context,
                matlab_status_report,
                matlab_status_report_path,
                matlab_launch_skipped_reason="completed_reusable_batch",
                matlab_reuse_mode=auto_matlab_reuse_mode,
            )

    effective_skip_matlab = skip_matlab or auto_matlab_reuse_mode is not None
    analysis_only_reuse = auto_matlab_reuse_mode == "analysis-only"
    effective_skip_python = skip_python or analysis_only_reuse

    # Run MATLAB
    matlab_results = None
    if not effective_skip_matlab:
        os.makedirs(matlab_output, exist_ok=True)
        comparison_context.update_optional_task(
            "matlab_pipeline",
            status="running",
            detail="Running MATLAB wrapper",
            artifacts={"params_file": str(normalized_params_file)},
        )
        matlab_results = run_matlab_vectorization(
            input_file,
            str(matlab_output),
            matlab_path,
            project_root,
            params_file=str(normalized_params_file),
        )
        matlab_status_report = _inspect_matlab_status_for_workflow(
            matlab_output,
            metadata_dir,
            input_file=input_file,
        )
        matlab_status_report_path = _persist_matlab_status_report(
            matlab_status_report, metadata_dir
        )
        failure_summary_path = _persist_matlab_failure_report(matlab_status_report, metadata_dir)
        _record_matlab_status_task(
            comparison_context,
            matlab_status_report,
            matlab_status_report_path,
            failure_summary_path,
        )
        matlab_results["matlab_status"] = matlab_status_report.to_dict()
        if failure_summary_path is not None:
            matlab_results["failure_summary_file"] = str(failure_summary_path)
        comparison_context.update_optional_task(
            "matlab_pipeline",
            status="completed" if matlab_results.get("success") else "failed",
            detail=(
                "MATLAB wrapper finished"
                if matlab_results.get("success")
                else (
                    matlab_status_report.failure_summary
                    or matlab_results.get("error", "MATLAB wrapper failed")
                )
            ),
            artifacts=(
                {
                    "batch_folder": matlab_results["batch_folder"],
                    "params_file": str(normalized_params_file),
                    "log_file": matlab_results.get("log_file", ""),
                }
                if matlab_results.get("batch_folder")
                else {
                    "params_file": str(normalized_params_file),
                    "log_file": matlab_results.get("log_file", ""),
                }
            ),
        )
        if not matlab_results.get("success"):
            comparison_context.mark_run_status(
                "failed",
                current_stage="matlab",
                detail=(
                    matlab_status_report.failure_summary
                    or matlab_results.get("error", "MATLAB wrapper failed")
                ),
            )
        batch_folder = matlab_results.get("batch_folder")
        if matlab_results.get("success") and batch_folder:
            checkpoint_dir = python_output / "checkpoints"
            comparison_context.update_optional_task(
                "matlab_import",
                status="running",
                detail="Importing MATLAB energy and vertices into Python checkpoints",
                artifacts={"batch_folder": batch_folder},
            )
            try:
                imported = import_matlab_batch(
                    batch_folder,
                    checkpoint_dir,
                    stages=import_stages,
                )
            except Exception as exc:
                comparison_context.update_optional_task(
                    "matlab_import",
                    status="failed",
                    detail=f"MATLAB checkpoint import failed: {exc}",
                    artifacts={"batch_folder": batch_folder},
                )
            else:
                imported_stages = set(imported)
                if set(import_stages).issubset(imported_stages):
                    python_force_rerun_from = requested_python_rerun_from
                try:
                    params_for_python |= load_matlab_batch_params(batch_folder)
                except Exception as exc:
                    comparison_context.update_optional_task(
                        "matlab_import",
                        status="completed",
                        detail=f"Imported MATLAB checkpoints; settings overlay unavailable: {exc}",
                        artifacts=dict(imported),
                    )
                else:
                    comparison_context.update_optional_task(
                        "matlab_import",
                        status="completed",
                        detail=(
                            "Imported MATLAB checkpoints for Python parity rerun "
                            f"from {requested_python_rerun_from}"
                        ),
                        artifacts={
                            **dict(imported),
                            "python_parity_rerun_from": requested_python_rerun_from,
                            **(
                                {"python_force_rerun_from": python_force_rerun_from}
                                if python_force_rerun_from is not None
                                else {}
                            ),
                        },
                    )
                _record_matlab_status_task(
                    comparison_context,
                    matlab_status_report,
                    matlab_status_report_path,
                    failure_summary_path,
                    python_force_rerun_from=python_force_rerun_from,
                )
    elif auto_matlab_reuse_mode == "analysis-only" and matlab_status_report is not None:
        matlab_results = {
            "success": True,
            "elapsed_time": 0.0,
            "output_dir": str(matlab_output),
            "batch_folder": matlab_status_report.matlab_batch_folder,
            "matlab_status": matlab_status_report.to_dict(),
        }
    else:
        if auto_matlab_reuse_mode == "python-rerun":
            print("\nSkipping MATLAB execution (completed reusable batch detected)")
        else:
            print("\nSkipping MATLAB execution (--skip-matlab)")
        matlab_results, python_force_rerun_from = (
            _bootstrap_existing_matlab_batch_for_python_parity(
                matlab_output=matlab_output,
                python_output=python_output,
                input_file=input_file,
                params_for_python=params_for_python,
                comparison_context=comparison_context,
                metadata_dir=metadata_dir,
                python_parity_rerun_from=requested_python_rerun_from,
            )
        )

    # Run Python
    python_results = None
    if not effective_skip_python:
        os.makedirs(python_output, exist_ok=True)

        # Check if this is a network gate execution (network rerun with MATLAB artifacts present)
        is_network_gate_candidate = (
            python_force_rerun_from == "network" and requested_python_rerun_from == "network"
        )

        # Only attempt network gate if we're rerunning from network
        # The network gate will validate artifacts and fall back if needed
        if is_network_gate_candidate:
            from .network_gate import validate_network_gate_artifacts

            # Quick validation check to see if network gate is possible
            validation = validate_network_gate_artifacts(run_root)

            if validation.validation_passed:
                # Stage-isolated network gate workflow with full artifacts
                from .network_gate import execute_stage_isolated_network_gate

                print("\n" + "=" * 60)
                print("Stage-Isolated Network Gate Validation")
                print("=" * 60)
                print("Network gate validation PASSED:")
                print("  âœ“ MATLAB edges artifact present")
                print("  âœ“ MATLAB vertices artifact present")
                print("  âœ“ MATLAB energy artifact present")
                print(f"  Edges fingerprint: {validation.matlab_edges_fingerprint[:16]}...")
                print(f"  Vertices fingerprint: {validation.matlab_vertices_fingerprint[:16]}...")

                comparison_context.update_optional_task(
                    "network_gate_validation",
                    status="completed",
                    detail="Network gate validation passed",
                    artifacts={
                        "matlab_edges_fingerprint": validation.matlab_edges_fingerprint,
                        "matlab_vertices_fingerprint": validation.matlab_vertices_fingerprint,
                    },
                )

                # Execute stage-isolated network gate
                print("\n" + "=" * 60)
                print("Stage-Isolated Network Gate Execution")
                print("=" * 60)

                comparison_context.update_optional_task(
                    "network_gate_execution",
                    status="running",
                    detail="Executing stage-isolated network gate",
                )

                execution = execute_stage_isolated_network_gate(
                    run_root,
                    input_file=Path(input_file),
                    params=params_for_python,
                )

                if execution.execution_errors:
                    print("\nNetwork gate execution encountered errors:")
                    for error in execution.execution_errors:
                        print(f"  - {error}")

                    comparison_context.update_optional_task(
                        "network_gate_execution",
                        status="failed",
                        detail="Network gate execution failed",
                        artifacts={
                            "elapsed_seconds": execution.elapsed_seconds,
                            "execution_errors": execution.execution_errors,
                        },
                    )
                    comparison_context.mark_run_status(
                        "failed",
                        current_stage="network_gate_execution",
                        detail=f"Network gate execution failed: {execution.execution_errors[0]}",
                    )
                    return 1

                # Report parity status
                print(
                    f"\nNetwork gate execution completed in {execution.elapsed_seconds:.2f} seconds"
                )
                print("\nParity Status:")
                print(f"  Vertices: {'âœ“ PASS' if execution.vertices_match else 'âœ— FAIL'}")
                print(f"  Edges:    {'âœ“ PASS' if execution.edges_match else 'âœ— FAIL'}")
                print(f"  Strands:  {'âœ“ PASS' if execution.strands_match else 'âœ— FAIL'}")
                print(
                    f"  Overall:  {'âœ“ PARITY ACHIEVED' if execution.parity_achieved else 'âœ— PARITY NOT ACHIEVED'}"
                )

                if execution.proof_artifact_json_path:
                    print("\nProof Artifact:")
                    print(f"  JSON: {execution.proof_artifact_json_path}")
                    if execution.proof_index_path:
                        print(f"  Index: {execution.proof_index_path}")

                comparison_context.update_optional_task(
                    "network_gate_execution",
                    status="completed",
                    detail=(
                        "Network gate execution completed - parity achieved"
                        if execution.parity_achieved
                        else "Network gate execution completed - parity not achieved"
                    ),
                    artifacts={
                        "elapsed_seconds": execution.elapsed_seconds,
                        "parity_achieved": execution.parity_achieved,
                        "vertices_match": execution.vertices_match,
                        "edges_match": execution.edges_match,
                        "strands_match": execution.strands_match,
                        "python_network_fingerprint": execution.python_network_fingerprint,
                        "proof_artifact_json_path": execution.proof_artifact_json_path,
                        "proof_index_path": execution.proof_index_path,
                    },
                )

                # Mark overall run status
                if execution.parity_achieved:
                    comparison_context.mark_run_status(
                        "completed",
                        current_stage="network_gate",
                        detail="Stage-isolated network gate achieved exact parity",
                    )
                else:
                    comparison_context.mark_run_status(
                        "completed",
                        current_stage="network_gate",
                        detail="Stage-isolated network gate completed but parity not achieved",
                    )

                # Network gate execution already persists metadata and loads results
                # We don't need to run the standard Python pipeline
                python_results = {
                    "success": True,
                    "elapsed_time": execution.elapsed_seconds,
                    "output_dir": str(python_output),
                    "network_gate_execution": True,
                    "parity_achieved": execution.parity_achieved,
                }
            else:
                # Network gate artifacts not available, fall back to standard Python pipeline
                print(
                    "\nNote: Network gate artifacts not available, running standard Python pipeline"
                )
                comparison_context.update_optional_task(
                    "python_pipeline",
                    status="running",
                    detail=(
                        f"Running Python pipeline from {python_force_rerun_from}"
                        if python_force_rerun_from is not None
                        else "Running Python pipeline"
                    ),
                )
                python_kwargs: dict[str, Any] = {
                    "run_dir": str(run_root),
                    "force_rerun_from": python_force_rerun_from,
                }
                if minimal_exports:
                    python_kwargs["minimal_exports"] = True
                python_results = run_python_vectorization(
                    input_file,
                    str(python_output),
                    params_for_python,
                    **python_kwargs,
                )
                comparison_context.update_optional_task(
                    "python_pipeline",
                    status="completed" if python_results.get("success") else "failed",
                    detail="Python pipeline finished",
                    artifacts={
                        "output_dir": str(python_output),
                        **(
                            {"force_rerun_from": python_force_rerun_from}
                            if python_force_rerun_from is not None
                            else {}
                        ),
                    },
                )
        else:
            # Standard Python pipeline execution
            comparison_context.update_optional_task(
                "python_pipeline",
                status="running",
                detail=(
                    f"Running Python pipeline from {python_force_rerun_from}"
                    if python_force_rerun_from is not None
                    else "Running Python pipeline"
                ),
            )
            python_kwargs: dict[str, Any] = {
                "run_dir": str(run_root),
                "force_rerun_from": python_force_rerun_from,
            }
            if minimal_exports:
                python_kwargs["minimal_exports"] = True
            python_results = run_python_vectorization(
                input_file,
                str(python_output),
                params_for_python,
                **python_kwargs,
            )
            comparison_context.update_optional_task(
                "python_pipeline",
                status="completed" if python_results.get("success") else "failed",
                detail="Python pipeline finished",
                artifacts={
                    "output_dir": str(python_output),
                    **(
                        {"force_rerun_from": python_force_rerun_from}
                        if python_force_rerun_from is not None
                        else {}
                    ),
                },
            )
    elif effective_skip_python and python_output.exists():
        # Even if we skip execution, we can still load existing results for comparison
        if analysis_only_reuse:
            print(
                "\nReusing existing Python outputs for analysis-only comparison; "
                f"loading results from {python_output}"
            )
        else:
            print(
                f"\nSkipping Python execution (--skip-python); attempting to load results from {python_output}"
            )
        try:
            python_results = _load_python_results_from_source(python_output, python_result_source)
        except Exception as e:
            print(f"Warning: Could not load existing Python results: {e}")
            python_results = None
    else:
        print("\nSkipping Python execution (--skip-python)")

    # Compare results
    if matlab_results and python_results:
        _run_comparison_analysis(
            matlab_results=matlab_results,
            python_results=python_results,
            run_root=run_root,
            analysis_dir=analysis_dir,
            metadata_dir=metadata_dir,
            comparison_depth=comparison_depth,
            comparison_context=comparison_context,
            load_matlab_batch_results_fn=load_matlab_batch_results,
            compare_results_fn=compare_results,
            build_shared_neighborhood_audit_fn=build_shared_neighborhood_audit,
        )

    _finalize_run_status_before_post_artifacts(
        comparison_context,
        matlab_results=matlab_results,
        python_results=python_results,
    )

    _generate_post_comparison_artifacts(
        run_dir=run_root,
        analysis_dir=analysis_dir,
        metadata_dir=metadata_dir,
        comparison_context=comparison_context,
        announce_manifest=True,
        generate_summary_fn=generate_summary,
        generate_manifest_fn=generate_manifest,
    )

    # Print final summary status
    if matlab_results and python_results:
        success = matlab_results.get("success") and python_results.get("success")
        if success:
            _print_reuse_guidance(
                run_root,
                input_file=input_file,
                params=params_for_python,
                loop_kind=loop_kind,
            )
        return 0 if success else 1
    return 0


def run_matlab_health_check_workflow(
    *,
    output_dir: Path,
    matlab_path: str,
    project_root: Path,
) -> int:
    """Run a lightweight MATLAB health check and persist staged metadata."""
    return _run_matlab_health_check_workflow_impl(
        output_dir=output_dir,
        matlab_path=matlab_path,
        project_root=project_root,
        resolve_run_layout_fn=resolve_run_layout,
        assess_loop_request_fn=assess_loop_request,
        persist_loop_assessment_report_fn=_persist_loop_assessment_report,
        record_loop_assessment_task_fn=_record_loop_assessment_task,
        evaluate_output_root_preflight_for_workflow_fn=_evaluate_output_root_preflight_for_workflow,
        persist_preflight_report_fn=_persist_preflight_report,
        summarize_output_preflight_fn=summarize_output_preflight,
        record_preflight_task_fn=_record_preflight_task,
        generate_post_comparison_artifacts_fn=_generate_post_comparison_artifacts,
        run_matlab_health_check_fn=run_matlab_health_check,
        persist_matlab_health_check_report_fn=_persist_matlab_health_check_report,
        record_matlab_health_check_task_fn=_record_matlab_health_check_task,
        summarize_matlab_health_check_fn=summarize_matlab_health_check,
        generate_summary_fn=generate_summary,
        generate_manifest_fn=generate_manifest,
    )


def run_standalone_comparison(
    matlab_dir: Path,
    python_dir: Path,
    output_dir: Path,
    project_root: Path,
    *,
    python_result_source: str = "auto",
    comparison_depth: str = "deep",
) -> int:
    """Run comparison on existing result directories."""
    return _run_standalone_comparison_impl(
        matlab_dir,
        python_dir,
        output_dir,
        python_result_source=python_result_source,
        comparison_depth=comparison_depth,
        resolve_run_layout_fn=resolve_run_layout,
        assess_loop_request_fn=assess_loop_request,
        persist_loop_assessment_report_fn=_persist_loop_assessment_report,
        load_python_results_from_source_fn=_load_python_results_from_source,
        run_comparison_analysis_fn=_run_comparison_analysis,
        generate_post_comparison_artifacts_fn=_generate_post_comparison_artifacts,
        print_reuse_guidance_fn=_print_reuse_guidance,
        load_matlab_batch_results_fn=load_matlab_batch_results,
        compare_results_fn=compare_results,
        build_shared_neighborhood_audit_fn=build_shared_neighborhood_audit,
        generate_summary_fn=generate_summary,
        generate_manifest_fn=generate_manifest,
    )
