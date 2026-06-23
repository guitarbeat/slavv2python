"""Authoritative run-local ownership records for parity writers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from slavv_python.engine.state import load_json_dict

from .constants import WRITER_LEASE_PATH
from .utils import now_iso, resolve_python_commit, write_json_with_hash


def writer_lease_path(run_dir: Path) -> Path:
    """Return the canonical writer-lease path for a parity run."""
    return run_dir.expanduser().resolve() / WRITER_LEASE_PATH


def load_writer_lease(run_dir: Path) -> dict[str, Any] | None:
    """Load a lease written by a hardened parity command, if present."""
    payload = load_json_dict(writer_lease_path(run_dir))
    return payload if isinstance(payload, dict) else None


def write_writer_lease(
    run_dir: Path,
    *,
    pid: int,
    command: str,
    stage: str,
    status: str,
    run_id: str | None = None,
    source_commit: str | None = None,
    previous: dict[str, Any] | None = None,
    ended_at: str | None = None,
) -> dict[str, Any]:
    """Atomically replace the run-local ownership record and its hash sidecar."""
    root = run_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    existing = previous or load_writer_lease(root) or {}
    started_at = existing.get("started_at") if existing.get("pid") == pid else None
    terminal = status not in {"running", "launched"}
    payload: dict[str, Any] = {
        "schema_version": 1,
        "pid": int(pid),
        "command": command,
        "stage": stage,
        "status": status,
        "run_id": run_id or existing.get("run_id"),
        "started_at": started_at or now_iso(),
        "updated_at": now_iso(),
        "source_commit": source_commit or resolve_python_commit(_repo_root(root)),
    }
    if terminal:
        payload["ended_at"] = ended_at or now_iso()
    write_json_with_hash(writer_lease_path(root), payload)
    return payload


def claim_writer_lease(run_dir: Path, *, command: str, stage: str) -> dict[str, Any]:
    """Record the current process as the only writer for a run."""
    return write_writer_lease(
        run_dir,
        pid=os.getpid(),
        command=command,
        stage=stage,
        status="running",
    )


def finalize_writer_lease(run_dir: Path, *, status: str, stage: str | None = None) -> None:
    """Record terminal writer state without deleting ownership provenance."""
    lease = load_writer_lease(run_dir)
    if lease is None:
        return
    write_writer_lease(
        run_dir,
        pid=int(lease.get("pid", os.getpid())),
        command=str(lease.get("command", "")),
        stage=stage or str(lease.get("stage", "all")),
        status=status,
        run_id=lease.get("run_id"),
        previous=lease,
    )


def _repo_root(path: Path) -> Path:
    for parent in (path, *path.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


__all__ = [
    "claim_writer_lease",
    "finalize_writer_lease",
    "load_writer_lease",
    "write_writer_lease",
    "writer_lease_path",
]
