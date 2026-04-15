"""
Comparison execution module.

This module handles the execution of MATLAB and Python vectorization pipelines
for comparison purposes.
"""

from __future__ import annotations

import copy
import glob
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from slavv.core import SLAVVProcessor
from slavv.io import export_pipeline_results, load_tiff_volume
from slavv.io.matlab_bridge import import_matlab_batch, load_matlab_batch_params
from slavv.io.matlab_parser import load_matlab_batch_results
from slavv.runtime import ProgressEvent, RunContext
from slavv.utils import format_time, get_matlab_info, get_system_info
from slavv.utils.safe_unpickle import safe_load
from slavv.visualization import NetworkVisualizer

from .matlab_status import (
    MatlabStatusReport,
    persist_matlab_failure_summary,
    persist_matlab_status,
    summarize_matlab_status,
)
from .matlab_status import (
    inspect_matlab_status as _inspect_matlab_status_direct,
)
from .metrics import build_shared_neighborhood_audit, compare_results
from .preflight import (
    OutputRootPreflightReport,
    persist_output_preflight,
    summarize_output_preflight,
)
from .preflight import (
    evaluate_output_root_preflight as _evaluate_output_root_preflight_direct,
)
from .reporting import generate_summary
from .run_layout import generate_manifest, resolve_run_layout
from .workflow_assessment import (
    LOOP_ANALYSIS_READY,
    LOOP_BLOCKED,
    LOOP_FRESH_MATLAB_REQUIRED,
    LOOP_REUSE_READY,
    LoopAssessmentReport,
    MatlabHealthCheckReport,
    assess_loop_request,
    determine_loop_kind,
    evaluate_output_root_preflight_cached,
    inspect_matlab_status_cached,
    persist_loop_assessment,
    persist_matlab_health_check,
    run_matlab_health_check,
    summarize_loop_assessment,
    summarize_matlab_health_check,
)

_PYTHON_RESULT_SOURCE_CHOICES = {
    "auto",
    "checkpoints-only",
    "export-json-only",
    "network-json-only",
}
_COMPARISON_DEPTH_CHOICES = {"shallow", "deep"}


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


def _json_default(value: Any) -> Any:
    """Serialize numpy-heavy comparison parameters to JSON."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _comparison_report_default(value: Any) -> Any:
    """Serialize comparison-report payloads with a string fallback."""
    try:
        return _json_default(value)
    except TypeError:
        return str(value)


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


def _write_comparison_report(comparison: dict[str, Any], report_file: Path) -> Path:
    """Persist the normalized comparison report JSON."""
    with open(report_file, "w", encoding="utf-8") as handle:
        json.dump(
            _build_serializable_comparison_report(comparison),
            handle,
            indent=2,
            default=_comparison_report_default,
        )
    print(f"\nComparison report saved to: {report_file}")
    return report_file


def _run_comparison_analysis(
    *,
    matlab_results: dict[str, Any],
    python_results: dict[str, Any],
    analysis_dir: Path,
    comparison_depth: str,
    comparison_context: RunContext | None = None,
) -> None:
    """Run the shared MATLAB-vs-Python analysis path and persist the report."""
    if comparison_context is not None:
        comparison_context.update_optional_task(
            "comparison_analysis",
            status="running",
            detail="Comparing MATLAB and Python results",
        )

    matlab_parsed = None
    if matlab_results.get("success") and matlab_results.get("batch_folder"):
        print("\nLoading MATLAB output data...")
        if _should_load_deep_matlab_results(comparison_depth):
            try:
                matlab_parsed = load_matlab_batch_results(matlab_results["batch_folder"])
                print(f"Successfully loaded MATLAB data from {matlab_results['batch_folder']}")
            except Exception as e:
                print(f"Warning: Could not load MATLAB output data: {e}")
                print("Comparison will proceed with basic metrics only.")
        else:
            print("Shallow comparison mode enabled; skipping full MATLAB batch parse.")

    comparison = compare_results(matlab_results, python_results, matlab_parsed)
    python_data = python_results.get("results") or {}
    shared_neighborhood_audit = None
    if matlab_parsed and "edges" in comparison:
        shared_neighborhood_audit = build_shared_neighborhood_audit(
            matlab_parsed.get("edges", {}),
            python_data.get("edges", {}),
            python_results.get("candidate_edges") or python_data.get("candidate_edges"),
            comparison["edges"],
            python_results.get("candidate_audit") or python_data.get("candidate_audit"),
            python_results.get("candidate_lifecycle") or python_data.get("candidate_lifecycle"),
        )
    if shared_neighborhood_audit is not None:
        comparison.setdefault("edges", {}).setdefault("diagnostics", {})[
            "shared_neighborhood_audit"
        ] = shared_neighborhood_audit
        (analysis_dir / "shared_neighborhood_audit.json").write_text(
            json.dumps(shared_neighborhood_audit, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    report_file = _write_comparison_report(comparison, analysis_dir / "comparison_report.json")

    if comparison_context is not None:
        comparison_context.update_optional_task(
            "comparison_analysis",
            status="completed",
            detail="Comparison report generated",
            artifacts={
                "comparison_report": str(report_file),
                "comparison_depth": comparison_depth,
            },
        )


def _generate_post_comparison_artifacts(
    *,
    run_dir: Path,
    analysis_dir: Path,
    metadata_dir: Path,
    comparison_context: RunContext | None = None,
    include_summary: bool = True,
    include_manifest: bool = True,
    announce_manifest: bool = False,
) -> None:
    """Best-effort generation of summary and manifest sidecar files."""
    if include_summary:
        try:
            summary_file = analysis_dir / "summary.txt"
            generate_summary(run_dir, summary_file)
        except Exception as exc:
            print(f"Note: Could not auto-generate summary: {exc}")

    if include_manifest:
        try:
            manifest_file = metadata_dir / "run_manifest.md"
            generate_manifest(run_dir, manifest_file)
            if announce_manifest:
                print(f"Manifest generated: {manifest_file}")
            if comparison_context is not None:
                comparison_context.update_optional_task(
                    "manifest",
                    status="completed",
                    detail="Comparison manifest written",
                    artifacts={"manifest": str(manifest_file)},
                )
        except Exception as exc:
            print(f"Note: Could not auto-generate manifest: {exc}")


def _resolve_python_energy_source(energy_data: dict[str, Any] | None) -> str:
    """Infer the effective energy source label for comparison reporting."""
    if not energy_data:
        return "native_python"
    return str(
        energy_data.get("energy_source") or energy_data.get("energy_origin") or "native_python"
    )


def _should_load_deep_matlab_results(comparison_depth: str) -> bool:
    """Return whether comparison should parse the full MATLAB batch surface."""
    normalized = comparison_depth.strip().lower()
    if normalized not in _COMPARISON_DEPTH_CHOICES:
        raise ValueError(
            f"Unsupported comparison depth '{comparison_depth}'. "
            f"Expected one of: {sorted(_COMPARISON_DEPTH_CHOICES)}"
        )
    return normalized == "deep"


def _print_reuse_guidance(
    run_root: Path,
    *,
    input_file: str | None = None,
    params: dict[str, Any] | None = None,
    loop_kind: str | None = None,
) -> None:
    """Print reuse eligibility summary using CLI summaries module."""
    from .cli_summaries import format_reuse_eligibility_summary, generate_reuse_commands
    from .workflow_assessment import assess_loop_request

    # Assess the current run root for reuse eligibility
    assessment = assess_loop_request(
        run_root,
        loop_kind=loop_kind or "skip_matlab_edges",
        input_path=Path(input_file) if input_file else None,
        params=params,
    )

    # Generate reuse commands and populate the report
    if input_file:
        assessment.reuse_commands = generate_reuse_commands(
            assessment,
            run_root=run_root,
            input_file=Path(input_file),
        )

    # Display the formatted summary
    summary = format_reuse_eligibility_summary(
        assessment,
        run_root=run_root,
        input_file=Path(input_file) if input_file else Path(""),
    )
    print("\n" + summary)


def _resolve_python_parity_import_plan(
    python_parity_rerun_from: str,
) -> tuple[list[str], str]:
    """Return the MATLAB stages needed to seed a Python parity rerun."""
    normalized = str(python_parity_rerun_from or "edges").strip().lower()
    if normalized == "network":
        return ["energy", "vertices", "edges"], "network"
    return ["energy", "vertices"], "edges"


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
        params_for_python.update(load_matlab_batch_params(batch_folder))
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


def _write_normalized_params_file(metadata_dir: Path, params: dict[str, Any]) -> Path:
    """Persist the normalized comparison parameters for both MATLAB and Python runs."""
    metadata_dir.mkdir(parents=True, exist_ok=True)
    params_path = metadata_dir / "comparison_params.normalized.json"
    with open(params_path, "w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=2, default=_json_default)
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


def _persist_matlab_status_report(
    report: MatlabStatusReport,
    metadata_dir: Path,
) -> Path | None:
    """Best-effort persistence for normalized MATLAB status metadata."""
    try:
        return persist_matlab_status(report, metadata_dir)
    except OSError as exc:
        print(f"Warning: Could not persist MATLAB status report: {exc}")
        return None


def _persist_matlab_failure_report(
    report: MatlabStatusReport,
    metadata_dir: Path,
) -> Path | None:
    """Best-effort persistence for MATLAB failure summaries."""
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


def load_parameters(params_file: str | None = None) -> dict[str, Any]:
    """Load parameters from JSON file or use defaults."""
    if params_file and os.path.exists(params_file):
        with open(params_file) as f:
            params = json.load(f)
    else:
        # Use Python defaults
        params = {
            "microns_per_voxel": [1.0, 1.0, 1.0],
            "radius_of_smallest_vessel_in_microns": 1.5,
            "radius_of_largest_vessel_in_microns": 50.0,
            "approximating_PSF": True,
            "excitation_wavelength_in_microns": 1.3,
            "numerical_aperture": 0.95,
            "sample_index_of_refraction": 1.33,
            "scales_per_octave": 1.5,
            "gaussian_to_ideal_ratio": 1.0,
            "spherical_to_annular_ratio": 1.0,
            "max_voxels_per_node_energy": 1e5,
        }

    # Convert lists to numpy arrays where needed
    if "microns_per_voxel" in params:
        params["microns_per_voxel"] = np.array(params["microns_per_voxel"])

    return params


def discover_matlab_artifacts(output_dir: str | Path) -> dict[str, Any]:
    """Discover the newest MATLAB batch folder and key output artifacts."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return {}

    batch_folders = sorted(
        path for path in output_path.iterdir() if path.is_dir() and path.name.startswith("batch_")
    )
    if not batch_folders:
        return {}

    batch_folder = batch_folders[-1]
    artifacts: dict[str, Any] = {"batch_folder": str(batch_folder)}

    vectors_dir = batch_folder / "vectors"
    if vectors_dir.exists():
        artifacts["vectors_dir"] = str(vectors_dir)
        network_files = sorted(vectors_dir.glob("network_*.mat"))
        if network_files:
            artifacts["network_mat"] = str(network_files[-1])

    return artifacts


def _run_command_with_timeout(
    cmd: list[str] | str, cwd: Path, timeout_seconds: int, *, shell: bool = False
) -> tuple[int, str, str, bool]:
    """Run a command and tear down the full process tree on timeout."""
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=shell,
    )

    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        return process.returncode or 0, stdout, stderr, False
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/f", "/t", "/pid", str(process.pid)],
                capture_output=True,
                text=True,
                check=False,
            )
        else:
            process.kill()

        stdout, stderr = process.communicate()
        returncode = process.returncode if process.returncode is not None else -9
        return returncode, stdout, stderr, True


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


def run_matlab_vectorization(
    input_file: str,
    output_dir: str,
    matlab_path: str,
    project_root: Path,
    batch_script: str | None = None,
    params_file: str | None = None,
) -> dict[str, Any]:
    """Run MATLAB vectorization via CLI."""
    print("\n" + "=" * 60)
    print("Running MATLAB Implementation")
    print("=" * 60)

    # Validate input file path
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    input_file = os.path.abspath(input_file)
    output_dir = os.path.abspath(output_dir)
    if params_file is not None:
        if not os.path.exists(params_file):
            raise FileNotFoundError(f"Parameters file not found: {params_file}")
        params_file = os.path.abspath(params_file)

    # Define script paths with absolute resolution.
    # Prefer dev/scripts (current layout), fallback to scripts (legacy layout).
    if batch_script is None:
        script_name = "run_matlab_cli.bat" if os.name == "nt" else "run_matlab_cli.sh"
        candidates = [
            (project_root / "dev" / "scripts" / "cli" / script_name).resolve(),
            (project_root / "scripts" / "cli" / script_name).resolve(),
        ]
        for candidate in candidates:
            if candidate.exists():
                batch_script = str(candidate)
                break
        if batch_script is None:
            raise FileNotFoundError(
                f"Could not find MATLAB CLI script '{script_name}' in any known location: {candidates}"
            )
    else:
        # Use provided script directly, but validate existence first
        if not os.path.exists(batch_script):
            raise FileNotFoundError(f"Custom batch script not found: {batch_script}")
        batch_script = os.path.abspath(batch_script)

    # Ensure executable permissions on POSIX
    if os.name != "nt" and not os.access(batch_script, os.X_OK):
        try:
            current_permissions = os.stat(batch_script).st_mode
            os.chmod(batch_script, current_permissions | 0o111)
        except Exception as e:
            print(f"Warning: Could not set executable permission on {batch_script}: {e}")

    # Build command list
    if os.name == "nt" and batch_script.lower().endswith((".bat", ".cmd")):
        script_args = [batch_script, input_file, output_dir, matlab_path]
        if params_file is not None:
            script_args.append(params_file)
        cmd: list[str] | str = subprocess.list2cmdline(script_args)
        use_shell = True
    else:
        cmd = [batch_script, input_file, output_dir, matlab_path]
        if params_file is not None:
            cmd.append(params_file)
        use_shell = False
    log_file = os.path.join(output_dir, "matlab_run.log")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    rendered_command = cmd if isinstance(cmd, str) else " ".join(cmd)
    print(f"Command: {rendered_command}")
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    if params_file is not None:
        print(f"Parameters file: {params_file}")

    # Capture system info
    system_info = get_system_info()
    matlab_info = get_matlab_info(matlab_path)
    system_info["matlab"] = matlab_info

    start_time = time.time()
    timeout_seconds = 3600

    returncode, stdout, stderr, timed_out = _run_command_with_timeout(
        cmd, project_root, timeout_seconds, shell=use_shell
    )
    elapsed_time = time.time() - start_time
    artifacts = discover_matlab_artifacts(output_dir)

    if timed_out:
        print(f"\nMATLAB execution timed out after {elapsed_time:.2f} seconds")
        print("STDOUT:")
        print(stdout)
        print("STDERR:")
        print(stderr)

        matlab_results = {
            "success": False,
            "elapsed_time": elapsed_time,
            "error": f"TimeoutExpired after {timeout_seconds} seconds",
            "stdout": stdout,
            "stderr": stderr,
            "system_info": system_info,
            "output_dir": output_dir,
            "params_file": params_file or "",
            "log_file": log_file,
        }
        matlab_results.update(artifacts)
        return matlab_results

    if returncode != 0:
        print(f"\nMATLAB execution failed after {elapsed_time:.2f} seconds")
        print(f"Exit code: {returncode}")
        print("STDOUT:")
        print(stdout)
        print("STDERR:")
        print(stderr)

        matlab_results = {
            "success": False,
            "elapsed_time": elapsed_time,
            "error": f"MATLAB exited with code {returncode}",
            "stdout": stdout,
            "stderr": stderr,
            "system_info": system_info,
            "output_dir": output_dir,
            "params_file": params_file or "",
            "log_file": log_file,
        }
        matlab_results.update(artifacts)
        return matlab_results

    print(f"\nMATLAB execution completed in {elapsed_time:.2f} seconds")
    print("STDOUT:")
    print(stdout)

    if stderr:
        print("STDERR:")
        print(stderr)

    matlab_results = {
        "success": True,
        "elapsed_time": elapsed_time,
        "output_dir": output_dir,
        "params_file": params_file or "",
        "stdout": stdout,
        "stderr": stderr,
        "system_info": system_info,
        "log_file": log_file,
    }
    matlab_results.update(artifacts)
    return matlab_results


def run_python_vectorization(
    input_file: str,
    output_dir: str,
    params: dict[str, Any],
    run_dir: str | None = None,
    force_rerun_from: str | None = None,
    minimal_exports: bool = False,
) -> dict[str, Any]:
    """Run Python vectorization."""
    print("\n" + "=" * 60)
    print("Running Python Implementation")
    print("=" * 60)

    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")

    # Capture system info
    system_info = get_system_info()

    # Load image
    print("Loading image...")
    image = load_tiff_volume(input_file)
    print(f"Image shape: {image.shape}, dtype: {image.dtype}")

    # Initialize processor
    processor = SLAVVProcessor()
    params_for_run = copy.deepcopy(params)
    params_for_run["comparison_exact_network"] = True

    # Run pipeline
    print("Running pipeline...")
    start_time = time.time()

    def progress_callback(frac, stage):
        stage_descriptions = {
            "start": "Initializing pipeline...",
            "preprocess": "Preprocessing finished. Calculating multi-scale energy field (this step takes 20-30 minutes)...",
            "energy": "Energy calculation complete. Extracting vertices...",
            "vertices": "Vertices extracted. Tracing edges...",
            "edges": "Edges traced. Building final network...",
            "network": "Network assembly complete!",
        }
        msg = stage_descriptions.get(stage, f"completed {stage}")
        print(f"  Progress: {frac * 100:.1f}% - {msg}")

    last_message = ""

    def event_callback(event):
        nonlocal last_message
        # Use carriage return \r to avoid spamming the console with millions of lines,
        # overwriting the current progress line instead of flooding the console.
        message = _format_progress_event_message(event)
        if message and message != last_message:
            print(f"      -> {message:<120}", end="\r", flush=True)
            last_message = message

    try:
        results = processor.process_image(
            image,
            params_for_run,
            progress_callback=progress_callback,
            event_callback=event_callback,
            run_dir=run_dir,
            checkpoint_dir=os.path.join(output_dir, "checkpoints") if run_dir is None else None,
            force_rerun_from=force_rerun_from,
        )
        print()  # Add a final newline to clear the \r string

        elapsed_time = time.time() - start_time

        print(f"\nPython execution completed in {elapsed_time:.2f} seconds")

        # Extract statistics
        python_results = {
            "success": True,
            "elapsed_time": elapsed_time,
            "output_dir": output_dir,
            "vertices_count": len(results["vertices"]["positions"]) if "vertices" in results else 0,
            "edges_count": len(results["edges"]["traces"]) if "edges" in results else 0,
            "network_strands_count": len(results["network"]["strands"])
            if "network" in results
            else 0,
            "results": results,
            "system_info": system_info,
            "comparison_mode": {
                "network_cleanup": "bypass_python_specific_cleanup",
                "energy_source": _resolve_python_energy_source(results.get("energy_data")),
            },
        }

        # Export results
        print("Exporting results...")
        export_pipeline_results(results, output_dir, base_name="python_comparison")

        if minimal_exports:
            print("Minimal export mode enabled; skipping VMV/CASX/CSV/JSON extra exports.")
            python_results["exports"] = {"profile": "minimal"}
        else:
            # Export VMV and CASX formats for visualization
            print("Exporting VMV and CASX formats...")
            try:
                visualizer = NetworkVisualizer()
                vmv_path = os.path.join(output_dir, "network.vmv")
                casx_path = os.path.join(output_dir, "network.casx")
                csv_base = os.path.join(output_dir, "network")
                json_path = os.path.join(output_dir, "network.json")

                visualizer.export_network_data(results, vmv_path, format="vmv")
                print(f"  VMV export: {vmv_path}")

                visualizer.export_network_data(results, casx_path, format="casx")
                print(f"  CASX export: {casx_path}")

                visualizer.export_network_data(results, csv_base, format="csv")
                print(f"  CSV export: {csv_base}_vertices.csv, {csv_base}_edges.csv")

                visualizer.export_network_data(results, json_path, format="json")
                print(f"  JSON export: {json_path}")

                python_results["exports"] = {
                    "profile": "full",
                    "vmv": vmv_path,
                    "casx": casx_path,
                    "csv": csv_base,
                    "json": json_path,
                }
            except Exception as e:
                print(f"  Warning: Export failed: {e}")
                import traceback

                traceback.print_exc()

        candidate_edges = _load_python_candidate_edges(Path(output_dir))
        if candidate_edges is not None:
            results["candidate_edges"] = candidate_edges
            python_results["candidate_edges"] = candidate_edges
        candidate_audit = _load_python_candidate_audit(Path(output_dir))
        if candidate_audit is not None:
            results["candidate_audit"] = candidate_audit
            python_results["candidate_audit"] = candidate_audit
        candidate_lifecycle = _load_python_candidate_lifecycle(Path(output_dir))
        if candidate_lifecycle is not None:
            results["candidate_lifecycle"] = candidate_lifecycle
            python_results["candidate_lifecycle"] = candidate_lifecycle

        return python_results

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\nPython execution failed after {elapsed_time:.2f} seconds")
        import traceback

        traceback.print_exc()

        return {
            "success": False,
            "elapsed_time": elapsed_time,
            "error": str(e),
            "system_info": system_info,
        }


def _load_python_results_from_checkpoints(python_root: Path) -> dict[str, Any] | None:
    """Prefer stage checkpoints over exported JSON when reconstructing comparison inputs."""
    checkpoint_dir = python_root / "checkpoints"
    energy_path = checkpoint_dir / "checkpoint_energy.pkl"
    vertices_path = checkpoint_dir / "checkpoint_vertices.pkl"
    edges_path = checkpoint_dir / "checkpoint_edges.pkl"
    network_path = checkpoint_dir / "checkpoint_network.pkl"
    if not (vertices_path.exists() and edges_path.exists() and network_path.exists()):
        return None

    try:
        energy_data = safe_load(energy_path) if energy_path.exists() else None
        vertices = safe_load(vertices_path)
        edges = safe_load(edges_path)
        network = safe_load(network_path)
    except Exception:
        return None

    results = {
        "energy_data": energy_data,
        "vertices": vertices,
        "edges": edges,
        "network": network,
    }
    candidate_edges = _load_python_candidate_edges(python_root)
    if candidate_edges is not None:
        results["candidate_edges"] = candidate_edges
    candidate_audit = _load_python_candidate_audit(python_root)
    if candidate_audit is not None:
        results["candidate_audit"] = candidate_audit
    candidate_lifecycle = _load_python_candidate_lifecycle(python_root)
    if candidate_lifecycle is not None:
        results["candidate_lifecycle"] = candidate_lifecycle
    return results


def _load_python_candidate_edges(python_root: Path) -> dict[str, Any] | None:
    """Load the persisted pre-cleanup candidate edge manifest when available."""
    candidate_path = python_root / "stages" / "edges" / "candidates.pkl"
    if not candidate_path.exists():
        return None
    try:
        return safe_load(candidate_path)
    except Exception:
        return None


def _load_python_candidate_audit(python_root: Path) -> dict[str, Any] | None:
    """Load the persisted candidate provenance audit when available."""
    candidate_audit_path = python_root / "stages" / "edges" / "candidate_audit.json"
    if not candidate_audit_path.exists():
        return None
    try:
        with open(candidate_audit_path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _load_python_candidate_lifecycle(python_root: Path) -> dict[str, Any] | None:
    """Load the persisted frontier lifecycle artifact when available."""
    candidate_lifecycle_path = python_root / "stages" / "edges" / "candidate_lifecycle.json"
    if not candidate_lifecycle_path.exists():
        return None
    try:
        with open(candidate_lifecycle_path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


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
                    params_for_python.update(load_matlab_batch_params(batch_folder))
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
            analysis_dir=analysis_dir,
            comparison_depth=comparison_depth,
            comparison_context=comparison_context,
        )

    _generate_post_comparison_artifacts(
        run_dir=run_root,
        analysis_dir=analysis_dir,
        metadata_dir=metadata_dir,
        comparison_context=comparison_context,
        announce_manifest=True,
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


def _load_python_results_from_source(
    python_root: Path,
    python_result_source: str,
) -> dict[str, Any]:
    """Load Python comparison data from the requested source."""
    normalized_source = python_result_source.strip().lower()
    if normalized_source not in _PYTHON_RESULT_SOURCE_CHOICES:
        raise ValueError(
            f"Unsupported python result source '{python_result_source}'. "
            f"Expected one of: {sorted(_PYTHON_RESULT_SOURCE_CHOICES)}"
        )

    source_order: list[str]
    if normalized_source == "auto":
        source_order = ["checkpoints-only", "export-json-only", "network-json-only"]
    else:
        source_order = [normalized_source]

    result_payload = {"success": True, "output_dir": str(python_root), "elapsed_time": 0.0}
    source_errors: list[str] = []

    for source_name in source_order:
        if source_name == "checkpoints-only":
            checkpoint_results = _load_python_results_from_checkpoints(python_root)
            if checkpoint_results is None:
                source_errors.append("checkpoints unavailable")
                continue

            print(f"Loading Python results from checkpoints in: {python_root / 'checkpoints'}")
            result_payload["results"] = checkpoint_results
            result_payload["vertices_count"] = len(
                checkpoint_results.get("vertices", {}).get("positions", [])
            )
            result_payload["edges_count"] = len(
                checkpoint_results.get("edges", {}).get("traces", [])
            )
            result_payload["network_strands_count"] = len(
                checkpoint_results.get("network", {}).get("strands", [])
            )
            result_payload["comparison_mode"] = {
                "result_source": "checkpoints",
                "energy_source": _resolve_python_energy_source(
                    checkpoint_results.get("energy_data")
                ),
            }
            return result_payload

        if source_name == "export-json-only":
            json_files = [
                path
                for path in glob.glob(str(python_root / "python_comparison_*.json"))
                if not path.endswith("_parameters.json")
            ]
            if not json_files:
                source_errors.append("export_json unavailable")
                continue

            latest_json = sorted(json_files)[-1]
            print(f"Loading Python results from: {latest_json}")
            try:
                with open(latest_json) as f:
                    loaded_data = json.load(f)

                if "vertices" in loaded_data and "positions" in loaded_data["vertices"]:
                    loaded_data["vertices"]["positions"] = np.array(
                        loaded_data["vertices"]["positions"]
                    )
                    if "radii" in loaded_data["vertices"]:
                        loaded_data["vertices"]["radii"] = np.array(
                            loaded_data["vertices"]["radii"]
                        )

                if "edges" in loaded_data and "traces" in loaded_data["edges"]:
                    loaded_data["edges"]["traces"] = [
                        np.array(t) for t in loaded_data["edges"]["traces"]
                    ]

                result_payload["results"] = loaded_data
                result_payload["vertices_count"] = len(
                    loaded_data.get("vertices", {}).get("positions", [])
                )
                result_payload["edges_count"] = len(loaded_data.get("edges", {}).get("traces", []))
                result_payload["network_strands_count"] = len(
                    loaded_data.get("network", {}).get("strands", [])
                )
                result_payload["comparison_mode"] = {
                    "result_source": "export_json",
                    "energy_source": "native_python",
                }
                return result_payload
            except Exception as e:
                result_payload["success"] = False
                source_errors.append(f"export_json error: {e}")
                continue

        if source_name == "network-json-only":
            network_json_paths = glob.glob(str(python_root / "network.json"))
            if not network_json_paths:
                source_errors.append("network_json unavailable")
                continue

            network_json = network_json_paths[0]
            print(f"Loading Python results from fallback: {network_json}")
            try:
                with open(network_json) as f:
                    net_data = json.load(f)

                loaded_data = {
                    "vertices": net_data.get("vertices", {}),
                    "edges": net_data.get("edges", {}),
                    "network": net_data.get("network", {}),
                }

                if "positions" in loaded_data["vertices"]:
                    loaded_data["vertices"]["positions"] = np.array(
                        loaded_data["vertices"]["positions"]
                    )

                result_payload["results"] = loaded_data
                result_payload["vertices_count"] = len(
                    loaded_data.get("vertices", {}).get("positions", [])
                )
                if "connections" in loaded_data.get("edges", {}):
                    result_payload["edges_count"] = len(loaded_data["edges"]["connections"])
                else:
                    result_payload["edges_count"] = len(
                        loaded_data.get("edges", {}).get("traces", [])
                    )
                result_payload["network_strands_count"] = len(
                    loaded_data.get("network", {}).get("strands", [])
                )
                result_payload["comparison_mode"] = {
                    "result_source": "network_json",
                    "energy_source": "native_python",
                }
                return result_payload
            except Exception as e:
                result_payload["success"] = False
                source_errors.append(f"network_json error: {e}")
                continue

    result_payload["success"] = False
    result_payload["error"] = (
        "; ".join(source_errors) if source_errors else "No result files found."
    )
    return result_payload


def run_matlab_health_check_workflow(
    *,
    output_dir: Path,
    matlab_path: str,
    project_root: Path,
) -> int:
    """Run a lightweight MATLAB health check and persist staged metadata."""
    layout = resolve_run_layout(output_dir)
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
    loop_assessment = assess_loop_request(run_root, loop_kind="validate_only")
    loop_assessment_path = _persist_loop_assessment_report(loop_assessment, metadata_dir)
    _record_loop_assessment_task(comparison_context, loop_assessment, loop_assessment_path)

    preflight_report = _evaluate_output_root_preflight_for_workflow(run_root, metadata_dir)
    preflight_report_path = _persist_preflight_report(preflight_report, metadata_dir)

    print("\n" + "=" * 60)
    print("Output Root Preflight")
    print("=" * 60)
    print(summarize_output_preflight(preflight_report))
    _record_preflight_task(comparison_context, preflight_report, preflight_report_path)

    if not preflight_report.allows_launch:
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
        )
        print(f"Recommended action: {preflight_report.recommended_action}")
        return 1

    health_report = run_matlab_health_check(
        output_root=run_root,
        matlab_path=matlab_path,
        project_root=project_root,
    )
    health_report_path = _persist_matlab_health_check_report(health_report, metadata_dir)
    _record_matlab_health_check_task(comparison_context, health_report, health_report_path)

    print("\n" + "=" * 60)
    print("MATLAB Health Check")
    print("=" * 60)
    print(summarize_matlab_health_check(health_report))

    comparison_context.mark_run_status(
        "completed" if health_report.success else "failed",
        current_stage="matlab_health_check",
        detail=summarize_matlab_health_check(health_report),
    )
    _generate_post_comparison_artifacts(
        run_dir=run_root,
        analysis_dir=analysis_dir,
        metadata_dir=metadata_dir,
        comparison_context=comparison_context,
        include_summary=False,
        announce_manifest=True,
    )
    return 0 if health_report.success else 1


def run_standalone_comparison(
    matlab_dir: Path,
    python_dir: Path,
    output_dir: Path,
    project_root: Path,
    *,
    python_result_source: str = "auto",
    comparison_depth: str = "deep",
) -> int:
    """
    Run comparison on existing result directories.

    Args:
        matlab_dir: Directory containing MATLAB results (e.g., .../TIMESTAMP_matlab_run)
        python_dir: Directory containing Python results (e.g., .../TIMESTAMP_python_run)
        output_dir: Directory to save comparison results
        project_root: Root of the repository

    Returns:
        0 if successful, 1 otherwise
    """
    print("\n" + "=" * 60)
    print("Running Standalone Comparison")
    print("=" * 60)
    print(f"MATLAB Dir: {matlab_dir}")
    print(f"Python Dir: {python_dir}")
    print(f"Output Dir: {output_dir}")

    layout = resolve_run_layout(output_dir)
    run_root = layout["run_root"]
    analysis_dir = layout["analysis_dir"]
    metadata_dir = layout["metadata_dir"]
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    loop_assessment = assess_loop_request(
        run_root,
        loop_kind="standalone_analysis",
        standalone_matlab_dir=matlab_dir,
        standalone_python_dir=python_dir,
    )
    _persist_loop_assessment_report(loop_assessment, metadata_dir)

    # 1. Reconstruct MATLAB results dict
    matlab_results = {
        "success": True,  # Assume success if we are selecting it
        "output_dir": str(matlab_dir),
        "elapsed_time": 0.0,  # Unknown
    }

    # Find batch folder
    matlab_layout = resolve_run_layout(matlab_dir)
    matlab_root = matlab_layout["matlab_dir"]
    batch_folders = [
        d for d in os.listdir(matlab_root) if (matlab_root / d).is_dir() and d.startswith("batch_")
    ]

    if batch_folders:
        batch_folder = str(matlab_root / sorted(batch_folders)[-1])
        matlab_results["batch_folder"] = batch_folder
        print(f"Found MATLAB batch folder: {batch_folder}")
    else:
        print("Warning: No MATLAB batch_* folder found.")

    python_layout = resolve_run_layout(python_dir)
    python_root = python_layout["python_dir"]
    python_results = _load_python_results_from_source(python_root, python_result_source)
    if not python_results.get("success"):
        print(f"Error loading Python results: {python_results.get('error', 'unknown error')}")
        return 1

    _run_comparison_analysis(
        matlab_results=matlab_results,
        python_results=python_results,
        analysis_dir=analysis_dir,
        comparison_depth=comparison_depth,
    )
    _generate_post_comparison_artifacts(
        run_dir=run_root,
        analysis_dir=analysis_dir,
        metadata_dir=metadata_dir,
    )

    _print_reuse_guidance(
        run_root,
        input_file=None,
        params=None,
        loop_kind="standalone_analysis",
    )
    return 0
