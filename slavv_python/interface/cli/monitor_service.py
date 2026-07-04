"""Shared run-monitoring service for CLI and TUI run operations."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

from slavv_python.engine.constants import TRACKED_RUN_STAGES
from slavv_python.engine.state import RunSnapshot, load_run_snapshot
from slavv_python.engine.state.progress import calculate_overall_progress, preprocess_complete
from slavv_python.engine.state.status import target_stage_progress

_MISSING: Any = object()

UNRESPONSIVE_AFTER_SECONDS = 15 * 60
ENERGY_STAGE = "energy"
_ENERGY_DONE_RE = re.compile(r"Done\s+(\d+)\s+tasks.*?elapsed:\s+([0-9.]+)\s*(min|s)")
_LOG_TAIL_BYTES = 64 * 1024


@dataclass(frozen=True)
class PidStatus:
    """Observed state for a PID file associated with a run."""

    path: Path
    pid: int | None
    state: str
    command_line: str = ""
    source: str = "legacy"


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
class EnergyProgress:
    """Determinate energy-chunk progress fused from the snapshot and joblib log.

    ``durable_units_completed`` is the authoritative octave-quantized cursor from
    the run snapshot; ``live_units_completed`` folds in the joblib ``Done N tasks``
    count so a parallel (``n_jobs>1``) octave advances instead of looking stalled.
    """

    units_total: int
    durable_units_completed: int
    live_units_completed: int
    chunks_in_batch: int
    per_chunk_seconds: float | None
    eta_seconds: float | None
    is_live: bool
    log_path: Path | None

    @property
    def fraction(self) -> float:
        """Return the live completion fraction clamped to ``[0, 1]``."""
        if self.units_total <= 0:
            return 0.0
        return max(0.0, min(1.0, self.live_units_completed / self.units_total))


@dataclass(frozen=True)
class StageRow:
    """Per-stage summary row for compact monitor rendering."""

    name: str
    status: str
    progress: float
    units_total: int
    units_completed: int
    detail: str

    @property
    def units_label(self) -> str:
        """Return ``completed/total`` units, or an em dash when not tracked."""
        return f"{self.units_completed}/{self.units_total}" if self.units_total else "—"


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
    metadata_dir = run_dir / "99_Metadata"
    candidates = [
        run_dir / "run.pid",
        metadata_dir / "run.pid",
        metadata_dir / "parity_job.pid",
        repo_root / "workspace" / "scratch" / "crop_energy_rerun_latest.pid",
    ]
    if metadata_dir.is_dir():
        candidates.extend(sorted(metadata_dir.glob("*.pid")))
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
        return PidStatus(path=path, pid=None, state="invalid", source="legacy")

    command_line = _process_command_line(pid)
    if command_line is None:
        return PidStatus(path=path, pid=pid, state="dead", source="legacy")

    normalized_run = str(run_dir.resolve()).lower()
    command_lower = command_line.lower()
    if normalized_run in command_lower or run_dir.name.lower() in command_lower:
        return PidStatus(
            path=path, pid=pid, state="alive", command_line=command_line, source="legacy"
        )
    return PidStatus(
        path=path, pid=pid, state="unrelated", command_line=command_line, source="legacy"
    )


def _load_pid_statuses(run_dir: Path) -> tuple[PidStatus, ...]:
    statuses: list[PidStatus] = []
    metadata_dir = run_dir / "99_Metadata"
    lease_path = metadata_dir / "writer_lease.json"
    lease = _read_json(lease_path)
    if lease is not None:
        try:
            lease_pid = int(lease["pid"])
        except (KeyError, TypeError, ValueError):
            statuses.append(PidStatus(lease_path, None, "invalid", source="lease"))
        else:
            command_line = _process_command_line(lease_pid)
            state = "alive" if command_line is not None else "dead"
            statuses.append(
                PidStatus(lease_path, lease_pid, state, command_line or "", source="lease")
            )

    try:
        from slavv_python.analytics.parity.runs.job_registry import JobRegistry

        record = JobRegistry().get_job_by_run_dir(run_dir)
    except (OSError, ValueError):
        record = None
    if record is not None and record.status == "running":
        command_line = _process_command_line(record.pid)
        statuses.append(
            PidStatus(
                metadata_dir / "job_registry.jsonl",
                record.pid,
                "alive" if command_line is not None else "dead",
                command_line or record.command,
                source="registry",
            )
        )
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
    metadata_dir = run_dir / "99_Metadata"
    candidates = [
        metadata_dir / "parity_job.err.log",
        metadata_dir / "parity_job.out.log",
        repo_root / "workspace" / "scratch" / "crop_energy_rerun_latest.err.log",
        repo_root / "workspace" / "scratch" / "crop_energy_rerun_latest.out.log",
    ]
    if metadata_dir.is_dir():
        candidates.extend(sorted(metadata_dir.glob("*.log")))
    unique: list[Path] = []
    for candidate in candidates:
        if candidate.is_file() and candidate not in unique:
            unique.append(candidate)
    return tuple(unique)


def _parity_job_terminal_status(run_dir: Path) -> tuple[str, str] | None:
    try:
        from slavv_python.analytics.parity.runs.parity_job_lifecycle import (
            TERMINAL_STATUSES,
            load_parity_job_manifest,
        )
    except ImportError:
        return None
    manifest = load_parity_job_manifest(run_dir)
    if manifest is None:
        return None
    status = str(manifest.get("status") or "")
    if status not in TERMINAL_STATUSES:
        return None
    reason = str(manifest.get("reason") or f"Parity job ended with status {status}.")
    exit_code = manifest.get("exit_code")
    if exit_code is not None:
        reason = f"{reason} exit_code={exit_code}"
    return status, reason


def _effective_status(
    snapshot: RunSnapshot | None,
    pid_statuses: tuple[PidStatus, ...],
    *,
    run_dir: Path | None = None,
) -> tuple[str, str]:
    if run_dir is not None:
        terminal = _parity_job_terminal_status(run_dir)
        if terminal is not None:
            return terminal

    alive = [pid for pid in pid_statuses if pid.state == "alive"]
    dead = [pid for pid in pid_statuses if pid.state == "dead"]
    if snapshot is None:
        return "missing-snapshot", "No run snapshot found."
    live_lease = [pid for pid in alive if pid.source == "lease"]
    live_registry = [pid for pid in alive if pid.source == "registry"]
    if live_lease and live_registry and live_lease[0].pid != live_registry[0].pid:
        return (
            "conflicting-writers",
            f"Lease PID {live_lease[0].pid} conflicts with registry PID {live_registry[0].pid}.",
        )
    if alive:
        owner = next((pid for pid in alive if pid.source == "lease"), alive[0])
        if snapshot is not None and _snapshot_is_stale(snapshot):
            return (
                "unresponsive",
                f"{owner.source} PID {owner.pid} is alive but snapshot progress is stale.",
            )
        return "running", f"{owner.source} PID {owner.pid} is alive."
    if snapshot.status == "running" or any(
        stage.status == "running" for stage in snapshot.stages.values()
    ):
        if dead:
            return "stale-running-snapshot", f"Snapshot is running but PID {dead[0].pid} is dead."
        return "stale-running-snapshot", "Snapshot is running but no live PID was found."
    return snapshot.status, "Snapshot status."


def _snapshot_is_stale(snapshot: RunSnapshot) -> bool:
    """Return whether a live writer has stopped updating durable stage progress."""
    updated_at = snapshot.updated_at
    if not updated_at:
        return False
    try:
        timestamp = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - timestamp).total_seconds() > UNRESPONSIVE_AFTER_SECONDS


def load_run_monitor_view(run_dir: str | Path) -> RunMonitorView:
    """Load a consolidated monitoring view for a structured run directory."""
    root = Path(run_dir).expanduser().resolve()
    snapshot_path = snapshot_path_for_run(root)
    snapshot = load_run_snapshot(root)
    pid_statuses = _load_pid_statuses(root)
    status, reason = _effective_status(snapshot, pid_statuses, run_dir=root)
    if status == "stale-running-snapshot":
        from slavv_python.analytics.parity.runs.parity_job_lifecycle import (
            reconcile_interrupted_run,
        )

        reconcile_interrupted_run(root, reason=reason)
        status, reason = _effective_status(snapshot, pid_statuses, run_dir=root)
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


def _iso_to_epoch(value: str | None) -> float | None:
    """Convert an ISO-8601 timestamp to a UTC epoch, or ``None`` if unparseable."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _read_log_tail(path: Path, max_bytes: int = _LOG_TAIL_BYTES) -> str:
    """Return the trailing ``max_bytes`` of a log file as text (empty on error)."""
    try:
        with path.open("rb") as handle:
            if path.stat().st_size > max_bytes:
                handle.seek(-max_bytes, os.SEEK_END)
            data = handle.read()
    except OSError:
        return ""
    return data.decode("utf-8", errors="ignore")


def _joblib_done_points(text: str) -> list[tuple[int, float]]:
    """Return ``(tasks_done, elapsed_seconds)`` points from joblib ``Done`` lines."""
    points: list[tuple[int, float]] = []
    for match in _ENERGY_DONE_RE.finditer(text):
        value = float(match.group(2))
        seconds = value * 60.0 if match.group(3) == "min" else value
        points.append((int(match.group(1)), seconds))
    return points


def _latest_joblib_log(
    log_paths: tuple[Path, ...],
) -> tuple[Path, list[tuple[int, float]], float] | None:
    """Return the newest log with joblib progress, its points, and its mtime."""
    best: tuple[Path, list[tuple[int, float]], float] | None = None
    for path in log_paths:
        points = _joblib_done_points(_read_log_tail(path))
        if not points:
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if best is None or mtime > best[2]:
            best = (path, points, mtime)
    return best


def _trailing_batch(points: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Return the trailing run of monotonically increasing task counts.

    joblib resets its ``Done N`` counter for every octave's ``Parallel`` call, so
    only the final increasing run reflects the batch that is in flight right now.
    """
    if not points:
        return []
    start = len(points) - 1
    while start > 0 and points[start - 1][0] < points[start][0]:
        start -= 1
    return points[start:]


def compute_energy_progress(view: RunMonitorView) -> EnergyProgress | None:
    """Return determinate energy-chunk progress for the active energy stage.

    The durable snapshot cursor only advances at octave boundaries under
    ``n_jobs>1``, so the joblib ``Done N tasks`` log is the live leading
    indicator during parallel compute. The log is trusted only while it is
    fresher than the durable stage checkpoint, which keeps the octave-boundary
    window from double-counting the just-merged batch.
    """
    snapshot = view.snapshot
    if snapshot is None or snapshot.current_stage != ENERGY_STAGE:
        return None
    stage = snapshot.stages.get(ENERGY_STAGE)
    if stage is None or stage.units_total <= 0:
        return None

    units_total = int(stage.units_total)
    durable = max(0, min(int(stage.units_completed), units_total))
    live = durable
    chunks_in_batch = 0
    per_chunk_seconds: float | None = None
    eta_seconds: float | None = None
    log_path: Path | None = None

    latest = _latest_joblib_log(view.log_paths)
    if latest is not None:
        log_path, points, log_epoch = latest
        batch = _trailing_batch(points)
        chunks_in_batch = batch[-1][0] if batch else 0
        stage_epoch = _iso_to_epoch(stage.updated_at) or _iso_to_epoch(snapshot.updated_at)
        log_is_fresh = stage_epoch is None or log_epoch >= stage_epoch
        if log_is_fresh and chunks_in_batch > 0:
            live = min(durable + chunks_in_batch, units_total)
        if len(batch) >= 2:
            (n0, t0), (n1, t1) = batch[0], batch[-1]
            if n1 > n0 and t1 > t0:
                per_chunk_seconds = (t1 - t0) / (n1 - n0)
                eta_seconds = max(0, units_total - live) * per_chunk_seconds

    live = max(live, durable)
    return EnergyProgress(
        units_total=units_total,
        durable_units_completed=durable,
        live_units_completed=live,
        chunks_in_batch=chunks_in_batch,
        per_chunk_seconds=per_chunk_seconds,
        eta_seconds=eta_seconds,
        is_live=live > durable,
        log_path=log_path,
    )


def live_overall_progress(snapshot: RunSnapshot, energy: EnergyProgress) -> float:
    """Return overall pipeline progress with the live energy fraction substituted.

    Reuses the canonical stage-weighted formula so the primary bar advances during
    parallel energy compute instead of tracking only the octave-quantized cursor.
    """
    stages = dict(snapshot.stages)
    energy_stage = stages.get(ENERGY_STAGE)
    if energy_stage is not None:
        stages[ENERGY_STAGE] = replace(energy_stage, progress=energy.fraction)
    done = preprocess_complete(stages, snapshot=snapshot)
    return float(calculate_overall_progress(stages, preprocess_done=done))


def format_duration(seconds: float) -> str:
    """Format a rough duration/ETA using the largest sensible unit."""
    if seconds >= 3600:
        return f"~{seconds / 3600:.1f}h"
    if seconds >= 60:
        return f"~{seconds / 60:.1f}m"
    return f"~{seconds:.0f}s"


_STATUS_STYLE_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("fail", "error", "block", "conflict"), "red"),
    (("interrupt", "unresponsive", "stale", "missing"), "yellow"),
    (("running",), "green"),
    (("complete",), "cyan"),
    (("pending",), "grey62"),
)


def status_style(status: str) -> str:
    """Map an effective/stage status to a Rich color for monitor rendering."""
    text = status.lower()
    for keywords, style in _STATUS_STYLE_RULES:
        if any(keyword in text for keyword in keywords):
            return style
    return "white"


def build_stage_rows(snapshot: RunSnapshot) -> list[StageRow]:
    """Return one :class:`StageRow` per tracked pipeline stage, in run order."""
    rows: list[StageRow] = []
    for name in TRACKED_RUN_STAGES:
        stage = snapshot.stages.get(name)
        if stage is None:
            rows.append(
                StageRow(
                    name=name,
                    status="pending",
                    progress=0.0,
                    units_total=0,
                    units_completed=0,
                    detail="",
                )
            )
            continue
        rows.append(
            StageRow(
                name=name,
                status=stage.status,
                progress=float(stage.progress),
                units_total=int(stage.units_total),
                units_completed=int(stage.units_completed),
                detail=stage.detail,
            )
        )
    return rows


def _newest_log(log_paths: tuple[Path, ...]) -> Path | None:
    """Return the most recently modified log among ``log_paths``."""
    best: tuple[Path, float] | None = None
    for path in log_paths:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if best is None or mtime > best[1]:
            best = (path, mtime)
    return best[0] if best else None


def tail_log_lines(view: RunMonitorView, *, max_lines: int = 20) -> tuple[str | None, list[str]]:
    """Return ``(log_name, last_lines)`` from the newest run log, if any."""
    log = _newest_log(view.log_paths)
    if log is None:
        return None, []
    lines = [line for line in _read_log_tail(log).splitlines() if line.strip()]
    return log.name, lines[-max_lines:]


def format_energy_progress_line(energy: EnergyProgress) -> str:
    """Render a single-line summary of live energy-chunk progress."""
    suffix = " (live from log)" if energy.is_live else ""
    summary = (
        f"Energy chunks{suffix}: {energy.live_units_completed}/{energy.units_total}"
        f" ({energy.fraction * 100:.1f}%)"
    )
    extras: list[str] = []
    if energy.per_chunk_seconds:
        extras.append(f"~{energy.per_chunk_seconds:.1f}s/chunk")
    if energy.eta_seconds is not None:
        extras.append(f"ETA {format_duration(energy.eta_seconds)}")
    if energy.is_live and energy.log_path is not None:
        extras.append(f"src {energy.log_path.name}")
    if extras:
        summary += " | " + " | ".join(extras)
    return summary


def render_monitor_lines(
    view: RunMonitorView, *, energy: EnergyProgress | None = _MISSING
) -> list[str]:
    """Render a human-readable status summary for CLI and legacy scripts.

    ``energy`` may be a precomputed :class:`EnergyProgress` (or ``None``) to avoid
    re-parsing the joblib log; when omitted it is computed from ``view``.
    """
    energy_progress = compute_energy_progress(view) if energy is _MISSING else energy
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
        for row in build_stage_rows(snapshot):
            marker = ">" if row.name == snapshot.current_stage else "-"
            units = f" units={row.units_label}" if row.units_total else ""
            detail = f" - {row.detail}" if row.detail else ""
            lines.append(
                f"  {marker} {row.name}: {row.status} {row.progress * 100:.1f}%{units}{detail}"
            )

        if energy_progress is not None:
            lines.extend(("", format_energy_progress_line(energy_progress)))
            live_overall = live_overall_progress(snapshot, energy_progress)
            if live_overall > snapshot.overall_progress + 1e-9:
                lines.append(f"Overall progress (live): {live_overall * 100:.1f}%")

    if view.pid_statuses:
        lines.extend(("", "PID files:"))
        for pid in view.pid_statuses:
            pid_text = "unknown" if pid.pid is None else str(pid.pid)
            lines.append(f"  - {pid.path}: pid={pid_text} state={pid.state} source={pid.source}")

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
    "EnergyProgress",
    "PidStatus",
    "ProofStatus",
    "RunMonitorView",
    "StageRow",
    "build_stage_rows",
    "compute_energy_progress",
    "format_duration",
    "format_energy_progress_line",
    "live_overall_progress",
    "load_run_monitor_view",
    "render_monitor_lines",
    "snapshot_path_for_run",
    "status_style",
    "tail_log_lines",
]
