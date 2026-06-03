"""Shared run-monitoring service for CLI and TUI run operations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

from slavv_python.engine.state import RunSnapshot, load_run_snapshot
from slavv_python.engine.state.status import target_stage_progress


@dataclass(frozen=True)
class PidStatus:
    """Observed state for a PID file associated with a run."""

    path: Path
    pid: int | None
    state: str
    command_line: str = ""


@dataclass(frozen=True)
class ProofStatus:
    """Summary of parity proof artifacts for a run."""

    path: Path
    exists: bool
    passed: bool | None = None
    first_failing_stage: str = ""
    first_failing_field: str = ""


@dataclass(frozen=True)
class ArtifactStatus:
    """Presence check for a run artifact."""

    label: str
    path: Path
    exists: bool


@dataclass(frozen=True)
class RunMonitorView:
    """Decision-ready monitoring view for a structured run directory."""

    run_dir: Path
    snapshot_path: Path
    snapshot: RunSnapshot | None
    effective_status: str
    status_reason: str
    pid_statuses: tuple[PidStatus, ...] = ()
    proof_statuses: tuple[ProofStatus, ...] = ()
    artifact_statuses: tuple[ArtifactStatus, ...] = ()
    log_paths: tuple[Path, ...] = ()
    errors: tuple[str, ...] = ()


def snapshot_path_for_run(run_dir: Path) -> Path:
    """Return the structured snapshot path for a run directory."""
    return run_dir / "99_Metadata" / "run_snapshot.json"


def _repo_root_from_run(run_dir: Path) -> Path:
    for parent in (run_dir, *run_dir.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _pid_file_candidates(run_dir: Path) -> list[Path]:
    repo_root = _repo_root_from_run(run_dir)
    candidates = [
        run_dir / "run.pid",
        run_dir / "99_Metadata" / "run.pid",
        repo_root / "workspace" / "scratch" / "crop_energy_rerun_latest.pid",
    ]
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def _process_command_line(pid: int) -> str | None:
    try:
        process = psutil.Process(pid)
        return " ".join(process.cmdline())
    except (psutil.Error, OSError):
        return None


def _inspect_pid_file(path: Path, run_dir: Path) -> PidStatus | None:
    if not path.is_file():
        return None
    raw_pid = path.read_text(encoding="utf-8").strip()
    try:
        pid = int(raw_pid)
    except ValueError:
        return PidStatus(path=path, pid=None, state="invalid")

    command_line = _process_command_line(pid)
    if command_line is None:
        return PidStatus(path=path, pid=pid, state="dead")

    normalized_run = str(run_dir.resolve()).lower()
    command_lower = command_line.lower()
    if normalized_run in command_lower or run_dir.name.lower() in command_lower:
        return PidStatus(path=path, pid=pid, state="alive", command_line=command_line)
    return PidStatus(path=path, pid=pid, state="unrelated", command_line=command_line)


def _load_pid_statuses(run_dir: Path) -> tuple[PidStatus, ...]:
    statuses = []
    for candidate in _pid_file_candidates(run_dir):
        status = _inspect_pid_file(candidate, run_dir)
        if status is not None:
            statuses.append(status)
    return tuple(statuses)


def _load_proof_status(path: Path) -> ProofStatus:
    payload = _read_json(path)
    if payload is None:
        return ProofStatus(path=path, exists=path.is_file())
    return ProofStatus(
        path=path,
        exists=True,
        passed=bool(payload.get("passed")),
        first_failing_stage=str(payload.get("first_failing_stage") or ""),
        first_failing_field=str(payload.get("first_failing_field_path") or ""),
    )


def _load_proof_statuses(run_dir: Path) -> tuple[ProofStatus, ...]:
    analysis = run_dir / "03_Analysis"
    return tuple(
        _load_proof_status(path)
        for path in (
            analysis / "exact_proof_energy.json",
            analysis / "exact_proof.json",
        )
        if path.is_file()
    )


def _artifact_checks(run_dir: Path) -> tuple[ArtifactStatus, ...]:
    paths = {
        "energy": run_dir / "02_Energy" / "best_energy.npy",
        "scale": run_dir / "02_Energy" / "best_scale.npy",
        "vertices": run_dir / "03_Vertices",
        "edges": run_dir / "04_Edges",
        "network": run_dir / "05_Network",
    }
    return tuple(
        ArtifactStatus(label=label, path=path, exists=path.exists())
        for label, path in paths.items()
    )


def _log_candidates(run_dir: Path) -> tuple[Path, ...]:
    repo_root = _repo_root_from_run(run_dir)
    candidates = [
        repo_root / "workspace" / "scratch" / "crop_energy_rerun_latest.err.log",
        repo_root / "workspace" / "scratch" / "crop_energy_rerun_latest.out.log",
    ]
    return tuple(path for path in candidates if path.is_file())


def _effective_status(snapshot: RunSnapshot | None, pid_statuses: tuple[PidStatus, ...]) -> tuple[str, str]:
    alive = [pid for pid in pid_statuses if pid.state == "alive"]
    dead = [pid for pid in pid_statuses if pid.state == "dead"]
    if snapshot is None:
        return "missing-snapshot", "No run snapshot found."
    if alive:
        return "running", f"PID {alive[0].pid} is alive."
    if snapshot.status == "running" or any(stage.status == "running" for stage in snapshot.stages.values()):
        if dead:
            return "stale-running-snapshot", f"Snapshot is running but PID {dead[0].pid} is dead."
        return "stale-running-snapshot", "Snapshot is running but no live PID was found."
    return snapshot.status, "Snapshot status."


def load_run_monitor_view(run_dir: str | Path) -> RunMonitorView:
    """Load a consolidated monitoring view for a structured run directory."""
    root = Path(run_dir).expanduser().resolve()
    snapshot_path = snapshot_path_for_run(root)
    snapshot = load_run_snapshot(root)
    pid_statuses = _load_pid_statuses(root)
    status, reason = _effective_status(snapshot, pid_statuses)
    errors: list[str] = []
    if snapshot is None:
        errors.append(f"Missing snapshot: {snapshot_path}")
    return RunMonitorView(
        run_dir=root,
        snapshot_path=snapshot_path,
        snapshot=snapshot,
        effective_status=status,
        status_reason=reason,
        pid_statuses=pid_statuses,
        proof_statuses=_load_proof_statuses(root),
        artifact_statuses=_artifact_checks(root),
        log_paths=_log_candidates(root),
        errors=tuple(errors),
    )


def render_monitor_lines(view: RunMonitorView) -> list[str]:
    """Render a human-readable status summary for CLI and legacy scripts."""
    lines = [
        f"Run directory: {view.run_dir}",
        f"Effective status: {view.effective_status} ({view.status_reason})",
        f"Snapshot: {view.snapshot_path}",
    ]
    snapshot = view.snapshot
    if snapshot is not None:
        lines.extend(
            [
                f"Run ID: {snapshot.run_id}",
                f"Snapshot status: {snapshot.status}",
                f"Target stage: {snapshot.target_stage}",
                f"Current stage: {snapshot.current_stage or '(idle)'}",
                f"Current detail: {snapshot.current_detail or '(none)'}",
                f"Overall progress: {snapshot.overall_progress * 100:.1f}%",
                f"Target progress: {target_stage_progress(snapshot) * 100:.1f}%",
            ]
        )
        lines.append("")
        lines.append("Stages:")
        for name, stage in snapshot.stages.items():
            units = f" units={stage.units_completed}/{stage.units_total}" if stage.units_total else ""
            detail = f" - {stage.detail}" if stage.detail else ""
            lines.append(f"  - {name}: {stage.status} {stage.progress * 100:.1f}%{units}{detail}")

    if view.pid_statuses:
        lines.extend(("", "PID files:"))
        for pid in view.pid_statuses:
            pid_text = "unknown" if pid.pid is None else str(pid.pid)
            lines.append(f"  - {pid.path}: pid={pid_text} state={pid.state}")

    if view.proof_statuses:
        lines.extend(("", "Proofs:"))
        for proof in view.proof_statuses:
            if proof.passed is None:
                status = "unreadable"
            else:
                status = "passed" if proof.passed else "failed"
            failure = ""
            if proof.first_failing_stage or proof.first_failing_field:
                field = proof.first_failing_field
                if proof.first_failing_stage and field.startswith(f"{proof.first_failing_stage}."):
                    failure = f" first_failure={field}"
                else:
                    failure = f" first_failure={proof.first_failing_stage}.{field}"
            lines.append(f"  - {proof.path.name}: {status}{failure}")

    lines.extend(("", "Artifacts:"))
    for artifact in view.artifact_statuses:
        mark = "present" if artifact.exists else "missing"
        lines.append(f"  - {artifact.label}: {mark} ({artifact.path})")

    if view.log_paths:
        lines.extend(("", "Logs:"))
        lines.extend(f"  - {path}" for path in view.log_paths)

    if view.errors:
        lines.extend(("", "Errors:"))
        lines.extend(f"  - {error}" for error in view.errors)

    return lines


__all__ = [
    "ArtifactStatus",
    "PidStatus",
    "ProofStatus",
    "RunMonitorView",
    "load_run_monitor_view",
    "render_monitor_lines",
    "snapshot_path_for_run",
]
