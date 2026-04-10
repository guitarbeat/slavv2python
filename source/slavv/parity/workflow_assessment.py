"""Shared workflow assessment and cached inspection helpers for parity tooling."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from slavv.runtime import load_run_snapshot

from .matlab_status import MatlabStatusReport, inspect_matlab_status, load_matlab_status
from .preflight import (
    DEFAULT_REQUIRED_FREE_SPACE_GB,
    OutputRootPreflightReport,
    evaluate_output_root_preflight,
    load_output_preflight,
)

PREFLIGHT_CACHE_VALID_SECONDS = 5 * 60
LOOP_ANALYSIS_READY = "analysis_ready"
LOOP_BLOCKED = "blocked"
LOOP_FRESH_MATLAB_REQUIRED = "fresh_matlab_required"
LOOP_REUSE_READY = "reuse_ready"


@dataclass
class LoopAssessmentReport:
    """Decision surface describing what the requested workflow can do safely."""

    run_root: str
    requested_loop: str
    verdict: str = LOOP_BLOCKED
    safe_to_reuse: bool = False
    safe_to_analyze_only: bool = False
    requires_fresh_matlab: bool = False
    input_compatible: bool = False
    params_compatible: bool = False
    has_required_artifacts: bool = False
    compatibility_reason: str = ""
    artifact_reason: str = ""
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommended_action: str = ""
    authoritative_files: dict[str, str] = field(default_factory=dict)
    artifact_checks: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass
class MatlabHealthCheckReport:
    """Persisted result of a lightweight MATLAB launch probe."""

    output_root: str
    resolved_output_root: str = ""
    matlab_path: str = ""
    checked_at: str = ""
    success: bool = False
    timed_out: bool = False
    exit_code: int | None = None
    elapsed_seconds: float = 0.0
    command: list[str] = field(default_factory=list)
    stdout_tail: list[str] = field(default_factory=list)
    stderr_tail: list[str] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def determine_loop_kind(
    *,
    standalone_mode: bool,
    validate_only: bool,
    skip_matlab: bool,
    skip_python: bool,
    python_parity_rerun_from: str = "edges",
) -> str:
    """Normalize CLI-style workflow flags into one loop identifier."""
    if standalone_mode:
        return "standalone_analysis"
    if validate_only:
        return "validate_only"
    if skip_matlab and not skip_python:
        normalized = str(python_parity_rerun_from or "edges").strip().lower()
        return "skip_matlab_network" if normalized == "network" else "skip_matlab_edges"
    if skip_python and not skip_matlab:
        return "skip_python_matlab_only"
    return "full_comparison"


def persist_loop_assessment(report: LoopAssessmentReport, metadata_dir: Path) -> Path:
    """Persist the normalized loop assessment report."""
    metadata_dir.mkdir(parents=True, exist_ok=True)
    report_path = metadata_dir / "loop_assessment.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2, sort_keys=True)
    return report_path


def load_loop_assessment(path_or_dir: Path) -> dict[str, Any] | None:
    """Load a persisted loop-assessment report."""
    candidate = path_or_dir / "loop_assessment.json" if path_or_dir.is_dir() else path_or_dir
    if not candidate.exists():
        return None
    with open(candidate, encoding="utf-8") as handle:
        return json.load(handle)


def summarize_loop_assessment(report: LoopAssessmentReport) -> str:
    """Render a compact human-facing workflow decision summary."""
    verdict_label = report.verdict.replace("_", " ").title()
    if report.recommended_action:
        return f"Workflow decision: {verdict_label}. {report.recommended_action}"
    if report.reasons:
        return f"Workflow decision: {verdict_label}. {report.reasons[0]}"
    return f"Workflow decision: {verdict_label}."


def assess_loop_request(
    run_root: Path,
    *,
    loop_kind: str,
    input_path: Path | None = None,
    params: dict[str, Any] | None = None,
    standalone_matlab_dir: Path | None = None,
    standalone_python_dir: Path | None = None,
) -> LoopAssessmentReport:
    """Assess whether a run root can satisfy the requested loop safely."""
    from .run_layout import resolve_run_layout

    layout = resolve_run_layout(run_root)
    metadata_dir = layout["metadata_dir"]
    report = LoopAssessmentReport(
        run_root=str(layout["run_root"]),
        requested_loop=loop_kind,
        authoritative_files={
            "run_root": str(layout["run_root"]),
            "metadata_dir": str(metadata_dir),
            "run_snapshot": str(layout["run_root"] / "99_Metadata" / "run_snapshot.json"),
            "comparison_params": str(metadata_dir / "comparison_params.normalized.json"),
        },
    )

    has_matlab_batch = _has_reusable_matlab_batch(layout["matlab_dir"])
    has_python_checkpoints = _has_python_checkpoint_surface(layout["python_dir"])
    has_python_results = _has_python_results_surface(layout["python_dir"])
    has_analysis_artifacts = _has_analysis_surface(layout["analysis_dir"])
    has_matlab_results = _has_matlab_results_surface(layout["matlab_dir"])
    report.artifact_checks = {
        "matlab_batch_present": has_matlab_batch,
        "python_checkpoints_present": has_python_checkpoints,
        "python_results_present": has_python_results,
        "analysis_artifacts_present": has_analysis_artifacts,
        "matlab_results_present": has_matlab_results,
    }
    report.safe_to_analyze_only = has_analysis_artifacts or (
        has_matlab_results and has_python_results
    )

    compatibility_ok, compatibility_reason, params_ok = _assess_recorded_compatibility(
        layout["run_root"],
        input_path=input_path,
        params=params,
        loop_kind=loop_kind,
    )
    report.input_compatible = compatibility_ok
    report.params_compatible = params_ok
    report.compatibility_reason = compatibility_reason

    if not compatibility_ok:
        report.verdict = LOOP_BLOCKED
        report.reasons.append(compatibility_reason)
        report.recommended_action = "Use a compatible staged run root or create a fresh run root."
        return report

    if loop_kind == "standalone_analysis":
        has_standalone_inputs = bool(
            standalone_matlab_dir is not None
            and standalone_matlab_dir.exists()
            and standalone_python_dir is not None
            and standalone_python_dir.exists()
        )
        report.has_required_artifacts = has_standalone_inputs
        report.artifact_reason = (
            "standalone MATLAB and Python inputs are available"
            if has_standalone_inputs
            else "standalone MATLAB and Python inputs are required"
        )
        report.safe_to_analyze_only = has_standalone_inputs
        if has_standalone_inputs:
            report.verdict = LOOP_ANALYSIS_READY
            report.recommended_action = "Analyze the existing MATLAB and Python result directories."
        else:
            report.verdict = LOOP_BLOCKED
            report.reasons.append(report.artifact_reason)
            report.recommended_action = "Provide both standalone result directories."
        return report

    if loop_kind == "validate_only":
        report.verdict = LOOP_ANALYSIS_READY
        report.has_required_artifacts = True
        report.artifact_reason = "validate-only mode needs no staged artifacts"
        report.safe_to_analyze_only = True
        report.recommended_action = "Run output-root preflight and inspect the persisted metadata."
        return report

    if loop_kind in {"skip_matlab_edges", "skip_matlab_network"}:
        has_skip_matlab_surface = has_matlab_batch or has_python_checkpoints
        report.has_required_artifacts = has_skip_matlab_surface
        report.artifact_reason = (
            "reusable MATLAB batch artifacts or Python checkpoints are available"
            if has_skip_matlab_surface
            else (
                "missing reusable Python checkpoints or MATLAB batch artifacts "
                "for a skip-matlab parity rerun"
            )
        )
        report.requires_fresh_matlab = not has_matlab_batch
        if has_skip_matlab_surface:
            report.verdict = LOOP_REUSE_READY
            report.safe_to_reuse = True
            if has_matlab_batch:
                report.recommended_action = (
                    "Reuse this run root for the imported-MATLAB parity rerun."
                )
            else:
                report.recommended_action = "Reuse this run root for a Python-side rerun; refresh MATLAB if imported parity is needed."
        else:
            report.verdict = LOOP_FRESH_MATLAB_REQUIRED
            report.reasons.append(report.artifact_reason)
            report.recommended_action = (
                "Create or refresh a MATLAB batch before relying on skip-matlab parity reuse."
            )
        return report

    if loop_kind in {"skip_python_matlab_only", "full_comparison"}:
        report.has_required_artifacts = True
        report.artifact_reason = "this loop can run against the selected run root"
        report.verdict = LOOP_FRESH_MATLAB_REQUIRED
        report.requires_fresh_matlab = True
        report.recommended_action = "Launch a fresh MATLAB run in this staged run root."
        return report

    report.verdict = LOOP_BLOCKED
    report.reasons.append(f"unsupported workflow loop '{loop_kind}'")
    report.recommended_action = "Use a supported parity workflow loop."
    return report


def evaluate_output_root_preflight_cached(
    output_root: str | Path,
    metadata_dir: Path,
    *,
    required_free_space_gb: float = DEFAULT_REQUIRED_FREE_SPACE_GB,
    cache_valid_seconds: int = PREFLIGHT_CACHE_VALID_SECONDS,
) -> OutputRootPreflightReport:
    """Reuse a recent matching preflight report when it is still fresh."""
    resolved_output_root = Path(output_root).expanduser()
    try:
        resolved_output_root = resolved_output_root.resolve(strict=False)
    except OSError:
        resolved_output_root = resolved_output_root.absolute()

    cached_payload = load_output_preflight(metadata_dir)
    cached_report = _hydrate_dataclass(OutputRootPreflightReport, cached_payload)
    if cached_report is not None:
        cached_created_at = _parse_iso_datetime(cached_report.cache_created_at)
        cache_is_fresh = cached_created_at is not None and datetime.now(
            timezone.utc
        ) - cached_created_at <= timedelta(seconds=cache_valid_seconds)
        same_output_root = str(resolved_output_root) == str(
            cached_report.resolved_output_root or cached_report.output_root
        )
        same_required_space = float(cached_report.required_space_gb) == float(
            required_free_space_gb
        )
        if cache_is_fresh and same_output_root and same_required_space:
            cached_report.cache_used = True
            cached_report.cache_valid_for_seconds = cache_valid_seconds
            return cached_report

    report = evaluate_output_root_preflight(
        resolved_output_root,
        required_free_space_gb=required_free_space_gb,
    )
    report.cache_used = False
    report.cache_created_at = _utc_now_iso()
    report.cache_valid_for_seconds = cache_valid_seconds
    return report


def inspect_matlab_status_cached(
    output_directory: str | Path,
    metadata_dir: Path,
    *,
    input_file: str | Path | None = None,
    log_tail_lines: int = 20,
    stale_running_seconds: int = 15 * 60,
) -> MatlabStatusReport:
    """Reuse cached MATLAB-status metadata when authoritative file mtimes match."""
    cached_payload = load_matlab_status(metadata_dir)
    cached_report = _hydrate_dataclass(MatlabStatusReport, cached_payload)
    if cached_report is not None and _matlab_status_cache_matches(
        cached_report,
        input_file=input_file,
    ):
        cached_report.cache_used = True
        return cached_report

    report = inspect_matlab_status(
        output_directory,
        input_file=input_file,
        log_tail_lines=log_tail_lines,
        stale_running_seconds=stale_running_seconds,
    )
    report.cache_used = False
    report.cache_created_at = _utc_now_iso()
    report.matlab_resume_state_mtime = _safe_stat_mtime(Path(report.matlab_resume_state_file))
    report.matlab_log_mtime = _safe_stat_mtime(Path(report.matlab_log_file))
    report.matlab_batch_folder_mtime = _safe_stat_mtime(Path(report.matlab_batch_folder))
    return report


def persist_matlab_health_check(report: MatlabHealthCheckReport, metadata_dir: Path) -> Path:
    """Persist a MATLAB health-check report."""
    metadata_dir.mkdir(parents=True, exist_ok=True)
    report_path = metadata_dir / "matlab_health_check.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2, sort_keys=True)
    return report_path


def load_matlab_health_check(path_or_dir: Path) -> dict[str, Any] | None:
    """Load a persisted MATLAB health-check report."""
    candidate = path_or_dir / "matlab_health_check.json" if path_or_dir.is_dir() else path_or_dir
    if not candidate.exists():
        return None
    with open(candidate, encoding="utf-8") as handle:
        return json.load(handle)


def summarize_matlab_health_check(report: MatlabHealthCheckReport) -> str:
    """Create a compact summary of the MATLAB health-check outcome."""
    if report.success:
        return f"MATLAB health check passed in {report.elapsed_seconds:.1f}s."
    if report.timed_out:
        return f"MATLAB health check timed out after {report.elapsed_seconds:.1f}s."
    if report.message:
        return f"MATLAB health check failed: {report.message}"
    return "MATLAB health check failed."


def run_matlab_health_check(
    *,
    output_root: Path,
    matlab_path: str,
    project_root: Path,
    timeout_seconds: int = 60,
) -> MatlabHealthCheckReport:
    """Run a lightweight direct MATLAB launch probe using the repo's batch style."""
    resolved_output_root = output_root.expanduser()
    try:
        resolved_output_root = resolved_output_root.resolve(strict=False)
    except OSError:
        resolved_output_root = resolved_output_root.absolute()

    matlab_script = "disp('slavv-matlab-health-check'); exit"
    command = [matlab_path]
    if os.name == "nt":
        command.append("-wait")
    command.extend(["-batch", matlab_script])

    started_at = time.time()
    returncode, stdout, stderr, timed_out = _run_command_with_timeout(
        command,
        project_root,
        timeout_seconds=timeout_seconds,
    )
    elapsed_seconds = time.time() - started_at
    success = not timed_out and returncode == 0
    message = "MATLAB launch probe succeeded."
    if timed_out:
        message = f"MATLAB launch probe timed out after {timeout_seconds} seconds."
    elif returncode != 0:
        message = f"MATLAB exited with code {returncode} during the launch probe."

    return MatlabHealthCheckReport(
        output_root=str(output_root),
        resolved_output_root=str(resolved_output_root),
        matlab_path=matlab_path,
        checked_at=_utc_now_iso(),
        success=success,
        timed_out=timed_out,
        exit_code=returncode,
        elapsed_seconds=elapsed_seconds,
        command=command,
        stdout_tail=_tail_text(stdout),
        stderr_tail=_tail_text(stderr),
        message=message,
    )


def _assess_recorded_compatibility(
    run_root: Path,
    *,
    input_path: Path | None,
    params: dict[str, Any] | None,
    loop_kind: str,
) -> tuple[bool, str, bool]:
    """Return compatibility status for the recorded run snapshot and params."""
    if loop_kind in {"standalone_analysis", "validate_only"}:
        return True, "no compatibility gate required for this loop", True

    snapshot = load_run_snapshot(run_root)
    if snapshot is None:
        if loop_kind in {"skip_matlab_edges", "skip_matlab_network"}:
            return False, "missing run snapshot for reusable imported-MATLAB parity rerun", False
        return True, "no recorded run snapshot yet", True

    recorded_input = str(snapshot.provenance.get("input_file", "") or "").strip()
    if (
        input_path is not None
        and recorded_input
        and _normalize_compare_path(recorded_input) != _normalize_compare_path(input_path)
    ):
        return (
            False,
            f"recorded input '{recorded_input}' does not match '{input_path}'",
            True,
        )

    if params is None:
        return True, "input provenance matches", True

    recorded_params = _load_recorded_comparison_params(run_root)
    if recorded_params is None:
        if loop_kind in {"skip_matlab_edges", "skip_matlab_network"}:
            return False, "missing normalized comparison params for reusable parity rerun", False
        return True, "no recorded comparison params yet", True

    if _normalize_value(recorded_params) != _normalize_value(params):
        return False, "recorded comparison parameters do not match the current request", False
    return True, "input provenance matches", True


def _load_recorded_comparison_params(run_root: Path) -> dict[str, Any] | None:
    params_path = run_root / "99_Metadata" / "comparison_params.normalized.json"
    if not params_path.exists():
        return None
    with open(params_path, encoding="utf-8") as handle:
        params = json.load(handle)
    params["comparison_exact_network"] = True
    params.setdefault("python_parity_rerun_from", "edges")
    return params


def _normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_value(subvalue) for key, subvalue in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return value.tolist()
        except TypeError:
            pass
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _normalize_compare_path(path_value: str | Path | None) -> str:
    if path_value in (None, ""):
        return ""
    path_text = str(path_value).replace("/", os.sep).replace("\\", os.sep)
    try:
        normalized = str(Path(path_text).resolve(strict=False))
    except OSError:
        normalized = path_text
    return normalized.lower() if os.name == "nt" else normalized


def _has_reusable_matlab_batch(matlab_dir: Path) -> bool:
    if not matlab_dir.exists():
        return False
    return any(child.is_dir() and child.name.startswith("batch_") for child in matlab_dir.iterdir())


def _has_python_checkpoint_surface(python_dir: Path) -> bool:
    checkpoint_dir = python_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return False
    return any(checkpoint_dir.glob("checkpoint_*.pkl"))


def _has_python_results_surface(python_dir: Path) -> bool:
    if not python_dir.exists():
        return False
    if (python_dir / "network.json").exists():
        return True
    return _has_python_checkpoint_surface(python_dir)


def _has_analysis_surface(analysis_dir: Path) -> bool:
    return (analysis_dir / "comparison_report.json").exists() or (
        analysis_dir / "summary.txt"
    ).exists()


def _has_matlab_results_surface(matlab_dir: Path) -> bool:
    return _has_reusable_matlab_batch(matlab_dir)


def _hydrate_dataclass(cls: type[Any], payload: dict[str, Any] | None) -> Any | None:
    if not isinstance(payload, dict):
        return None
    allowed_keys = set(getattr(cls, "__dataclass_fields__", {}).keys())
    kwargs = {key: value for key, value in payload.items() if key in allowed_keys}
    try:
        return cls(**kwargs)
    except TypeError:
        return None


def _parse_iso_datetime(value: str) -> datetime | None:
    if not value:
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_stat_mtime(path: Path) -> float | None:
    if not path or not str(path).strip():
        return None
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _matlab_status_cache_matches(
    report: MatlabStatusReport,
    *,
    input_file: str | Path | None,
) -> bool:
    if input_file is not None and _normalize_compare_path(
        report.input_file
    ) != _normalize_compare_path(input_file):
        return False
    if report.matlab_resume_state_mtime != _safe_stat_mtime(Path(report.matlab_resume_state_file)):
        return False
    if report.matlab_log_mtime != _safe_stat_mtime(Path(report.matlab_log_file)):
        return False
    return report.matlab_batch_folder_mtime == _safe_stat_mtime(Path(report.matlab_batch_folder))


def _run_command_with_timeout(
    cmd: list[str],
    cwd: Path,
    *,
    timeout_seconds: int,
) -> tuple[int, str, str, bool]:
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False,
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


def _tail_text(value: str, *, max_lines: int = 10) -> list[str]:
    return [line for line in value.splitlines()[-max_lines:] if line.strip()]
