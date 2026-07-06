"""In-process parity writer session: lease, job manifest, and registry as one transaction."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

from slavv_python.analytics.parity.runs.job_registry import JobRegistry
from slavv_python.analytics.parity.runs.parity_job_lifecycle import (
    finalize_parity_job,
    mark_parity_job_running,
)
from slavv_python.analytics.parity.runs.process_utils import (
    ensure_monitor_daemon_running,
    is_process_alive,
    is_python_process,
    kill_process_tree,
)
from slavv_python.analytics.parity.runs.writer_lease import (
    claim_writer_lease,
    finalize_writer_lease,
    load_writer_lease,
)


def register_monitor_job(
    *,
    run_dir: Path,
    oracle_root: Path | None,
    stage: str,
    command: str,
    pid: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Register a parity job for monitoring and ensure the monitor daemon is running."""
    registry = JobRegistry()
    oracle_root_path = oracle_root if oracle_root is not None else Path("unknown")
    job_id = registry.register_job(
        pid=pid if pid is not None else os.getpid(),
        run_dir=run_dir,
        oracle_root=oracle_root_path,
        stage=stage,
        command=command,
        metadata=metadata,
    )
    ensure_monitor_daemon_running()
    print(f"Job registered for monitoring (ID: {job_id})")
    return job_id


def reconcile_stale_writer_lease(
    dest_run_root: Path,
    *,
    force_kill: bool = False,
) -> None:
    """Clear or reject an active writer lease left by a prior process."""
    existing_lease = load_writer_lease(dest_run_root)
    if existing_lease and existing_lease.get("status") == "running":
        lease_pid = int(existing_lease.get("pid", -1))
        if lease_pid != os.getpid() and not is_process_alive(lease_pid):
            finalize_writer_lease(dest_run_root, status="interrupted")
            finalize_parity_job(
                dest_run_root,
                status="interrupted",
                exit_code=None,
                reason=f"Writer lease PID {lease_pid} is no longer alive.",
            )
        elif (
            lease_pid != os.getpid()
            and is_process_alive(lease_pid)
            and is_python_process(lease_pid)
        ):
            if not force_kill:
                raise RuntimeError(
                    f"Run directory has active writer lease (PID {lease_pid}). "
                    "Use --force-kill to replace it."
                )
            print(f"Terminating writer-lease PID {lease_pid}...")
            kill_process_tree(lease_pid)


def reconcile_registry_writer_conflict(
    dest_run_root: Path,
    *,
    force_kill: bool = False,
) -> None:
    """Reject or terminate a live writer tracked in the global job registry."""
    registry = JobRegistry()
    active_job = registry.get_job_by_run_dir(dest_run_root)
    if active_job and is_process_alive(active_job.pid) and is_python_process(active_job.pid):
        if not force_kill:
            raise RuntimeError(
                f"Run directory has active writer (PID {active_job.pid}).\n"
                f"Job started: {active_job.started_at}\n"
                f"Use --force-kill to terminate, or wait for completion.\n"
                f"Check status: slavv jobs list"
            )
        print(f"Terminating active writer PID {active_job.pid}...")
        kill_process_tree(active_job.pid)
        registry.update_job(active_job.job_id, status="killed")


@contextmanager
def resume_writer_session(
    dest_run_root: Path,
    *,
    command: str,
    stage: str,
    argv: list[str] | None = None,
    monitor: bool = False,
    force_kill: bool = False,
    oracle_root: Path | None = None,
    stop_after: str | None = None,
    registry_metadata: dict[str, Any] | None = None,
) -> Iterator[None]:
    """Claim lease, manifest, and optional registry entry; finalize all on exit."""
    argv = list(sys.argv) if argv is None else argv
    reconcile_stale_writer_lease(dest_run_root, force_kill=force_kill)
    if monitor:
        reconcile_registry_writer_conflict(dest_run_root, force_kill=force_kill)

    job_id: str | None = None
    registry: JobRegistry | None = None
    if monitor:
        registry = JobRegistry()
        job_id = register_monitor_job(
            run_dir=dest_run_root,
            oracle_root=oracle_root,
            stage=stage,
            command=command,
            metadata=registry_metadata,
        )

    claim_writer_lease(dest_run_root, command=command, stage=stage)
    mark_parity_job_running(
        dest_run_root,
        pid=os.getpid(),
        command=argv,
        stage=stage,
    )

    try:
        yield
        if monitor and job_id is not None and registry is not None:
            registry.update_job(
                job_id,
                status="succeeded",
                completed_at=datetime.now().isoformat(),
                exit_code=0,
            )
        finalize_writer_lease(
            dest_run_root,
            status="completed",
            stage=stop_after or "all",
        )
        finalize_parity_job(dest_run_root, status="succeeded", exit_code=0)
    except Exception as exc:
        if monitor and job_id is not None and registry is not None:
            registry.update_job(
                job_id,
                status="failed",
                completed_at=datetime.now().isoformat(),
                exit_code=1,
                metadata={"error": str(exc)},
            )
        finalize_writer_lease(dest_run_root, status="failed")
        finalize_parity_job(
            dest_run_root,
            status="failed",
            exit_code=1,
            reason=str(exc),
        )
        raise


__all__ = [
    "reconcile_registry_writer_conflict",
    "reconcile_stale_writer_lease",
    "register_monitor_job",
    "resume_writer_session",
]
