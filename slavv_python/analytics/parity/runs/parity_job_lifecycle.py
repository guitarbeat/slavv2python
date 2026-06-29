"""Run-local parity job manifest lifecycle and stale-writer reconciliation."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003
from typing import Any

from slavv_python.analytics.parity.constants import PARITY_JOB_MANIFEST_PATH
from slavv_python.analytics.parity.runs.writer_lease import finalize_writer_lease, load_writer_lease
from slavv_python.analytics.parity.utils import now_iso, write_json_with_hash
from slavv_python.engine.state import load_json_dict

TERMINAL_STATUSES = frozenset({"succeeded", "failed", "interrupted"})
ACTIVE_STATUSES = frozenset({"launched", "running"})


def parity_job_manifest_path(run_dir: Path) -> Path:
    """Return the canonical parity job manifest path for a run."""
    return run_dir.expanduser().resolve() / PARITY_JOB_MANIFEST_PATH


def load_parity_job_manifest(run_dir: Path) -> dict[str, Any] | None:
    """Load the run-local parity job manifest when present."""
    payload = load_json_dict(parity_job_manifest_path(run_dir))
    return payload if isinstance(payload, dict) else None


def update_parity_job_manifest(run_dir: Path, **updates: Any) -> dict[str, Any]:
    """Merge updates into the run-local parity job manifest."""
    root = run_dir.expanduser().resolve()
    manifest = load_parity_job_manifest(root) or {
        "kind": "parity_exact_run_job",
        "dest_run_root": str(root),
    }
    manifest.update(updates)
    write_json_with_hash(parity_job_manifest_path(root), manifest)
    return manifest


def mark_parity_job_running(
    run_dir: Path,
    *,
    pid: int,
    command: str | list[str],
    stage: str,
) -> dict[str, Any]:
    """Record that a foreground or detached writer has started."""
    command_text = command if isinstance(command, str) else " ".join(command)
    existing = load_parity_job_manifest(run_dir) or {}
    started_at = existing.get("started_at") or existing.get("launched_at") or now_iso()
    return update_parity_job_manifest(
        run_dir,
        status="running",
        pid=int(pid),
        stage=stage,
        command=command_text,
        started_at=started_at,
        ended_at=None,
        exit_code=None,
        reason=None,
    )


def finalize_parity_job(
    run_dir: Path,
    *,
    status: str,
    exit_code: int | None = None,
    reason: str = "",
) -> dict[str, Any]:
    """Persist terminal parity job metadata for a completed writer."""
    if status not in TERMINAL_STATUSES:
        raise ValueError(f"unsupported terminal parity job status: {status}")
    updates: dict[str, Any] = {
        "status": status,
        "ended_at": now_iso(),
        "exit_code": exit_code,
    }
    if reason:
        updates["reason"] = reason
    return update_parity_job_manifest(run_dir, **updates)


def _sync_registry_interrupted(run_dir: Path, *, reason: str) -> None:
    try:
        from slavv_python.analytics.parity.runs.job_registry import JobRegistry

        registry = JobRegistry()
        record = registry.get_job_by_run_dir(run_dir)
        if record is not None and record.status == "running":
            registry.update_job(
                record.job_id,
                status="interrupted",
                completed_at=now_iso(),
                metadata={"reason": reason},
            )
    except (OSError, ValueError):
        return


def reconcile_interrupted_run(run_dir: Path, *, reason: str) -> dict[str, Any] | None:
    """Persist interrupted terminal metadata without rewriting pipeline checkpoints."""
    root = run_dir.expanduser().resolve()
    manifest = load_parity_job_manifest(root)
    if manifest is not None and manifest.get("status") in TERMINAL_STATUSES:
        return manifest

    lease = load_writer_lease(root)
    if lease is not None and lease.get("status") == "running":
        finalize_writer_lease(root, status="interrupted")

    _sync_registry_interrupted(root, reason=reason)
    return finalize_parity_job(
        root,
        status="interrupted",
        exit_code=None,
        reason=reason,
    )


__all__ = [
    "ACTIVE_STATUSES",
    "TERMINAL_STATUSES",
    "finalize_parity_job",
    "load_parity_job_manifest",
    "mark_parity_job_running",
    "parity_job_manifest_path",
    "reconcile_interrupted_run",
    "update_parity_job_manifest",
]
