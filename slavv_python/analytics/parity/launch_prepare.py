"""Preflight and foreground launch probes before detached exact-route writers."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from slavv_python.analytics.parity.process_utils import (
    is_process_alive,
    is_python_process,
)

from .jobs import build_resume_exact_run_command
from .proofs import run_exact_preflight


class LaunchPreparationError(RuntimeError):
    """Raised when a run root is not safe to launch against."""


def reconcile_run_before_launch(run_dir: Path) -> str:
    """Reconcile stale writer metadata and return the effective monitor status."""
    from slavv_python.interface.cli.monitor_service import load_run_monitor_view

    view = load_run_monitor_view(run_dir)
    if view.effective_status == "running":
        alive = next((pid for pid in view.pid_statuses if pid.state == "alive"), None)
        pid_text = alive.pid if alive is not None else "unknown"
        raise LaunchPreparationError(
            f"Run directory already has an active writer ({view.status_reason}; PID {pid_text})."
        )
    return view.effective_status


def run_launch_preflight(
    *,
    dest_run_root: Path,
    oracle_root: Path | None,
    dataset_root: Path | None,
    memory_safety_fraction: float,
    force: bool,
) -> dict[str, Any]:
    """Run the exact-route preflight gate; raise when it fails."""
    report, _json_path, _text_path = run_exact_preflight(
        source_run_root=dest_run_root,
        dest_run_root=dest_run_root,
        oracle_root=oracle_root,
        dataset_root=dataset_root,
        memory_safety_fraction=memory_safety_fraction,
        force=force,
    )
    if not report.get("passed"):
        raise LaunchPreparationError(
            "preflight-exact failed; fix blockers before launching a long writer."
        )
    return report


def build_foreground_probe_command(
    *,
    dest_run_root: Path,
    oracle_root: Path | None = None,
    dataset_root: Path | None = None,
    force_rerun_from: str | None = None,
    memory_safety_fraction: float | None = None,
    force: bool = False,
    n_jobs: int | None = None,
    python_executable: Path | None = None,
) -> list[str]:
    """Build a short foreground diagnostic command (preprocess-only) for launch health."""
    return build_resume_exact_run_command(
        dest_run_root=dest_run_root,
        oracle_root=oracle_root,
        dataset_root=dataset_root,
        stop_after="preprocess",
        force_rerun_from=force_rerun_from,
        memory_safety_fraction=memory_safety_fraction,
        force=force,
        skip_preflight=True,
        n_jobs=n_jobs,
        python_executable=python_executable,
    )


def run_foreground_launch_probe(
    command: list[str],
    *,
    cwd: Path,
) -> int:
    """Execute the foreground diagnostic command and return its exit code."""
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        check=False,
    )
    return int(completed.returncode)


def prepare_detached_exact_run_launch(
    *,
    dest_run_root: Path,
    oracle_root: Path | None,
    dataset_root: Path | None,
    stop_after: str | None,
    force_rerun_from: str | None,
    memory_safety_fraction: float | None,
    force: bool,
    skip_preflight: bool,
    skip_foreground_probe: bool,
    n_jobs: int | None,
    python_executable: Path | None = None,
) -> tuple[list[str], list[str]]:
    """Reconcile stale state, preflight, and foreground probe before detach."""
    root = dest_run_root.expanduser().resolve()
    repo_root = _repo_root_from_path(root)
    reconcile_run_before_launch(root)

    if not skip_preflight:
        run_launch_preflight(
            dest_run_root=root,
            oracle_root=oracle_root.expanduser().resolve() if oracle_root else None,
            dataset_root=dataset_root.expanduser().resolve() if dataset_root else None,
            memory_safety_fraction=float(memory_safety_fraction or 0.8),
            force=force,
        )

    foreground_command = build_foreground_probe_command(
        dest_run_root=root,
        oracle_root=oracle_root,
        dataset_root=dataset_root,
        force_rerun_from=force_rerun_from,
        memory_safety_fraction=memory_safety_fraction,
        force=force,
        n_jobs=n_jobs,
        python_executable=python_executable,
    )
    if not skip_foreground_probe:
        exit_code = run_foreground_launch_probe(foreground_command, cwd=repo_root)
        if exit_code != 0:
            raise LaunchPreparationError(
                "Foreground launch probe failed with exit code "
                f"{exit_code}. Re-run the same command in a foreground shell "
                f"to capture the traceback: {' '.join(foreground_command)}"
            )

    detached_command = build_resume_exact_run_command(
        dest_run_root=root,
        oracle_root=oracle_root.expanduser().resolve() if oracle_root else None,
        dataset_root=dataset_root.expanduser().resolve() if dataset_root else None,
        stop_after=stop_after,
        force_rerun_from=force_rerun_from,
        memory_safety_fraction=memory_safety_fraction,
        force=force,
        skip_preflight=True,
        n_jobs=n_jobs,
        python_executable=python_executable,
    )
    return detached_command, foreground_command


def assert_no_conflicting_registry_writer(
    dest_run_root: Path,
    *,
    force_kill: bool,
) -> None:
    """Reject launch when the global registry still tracks a live writer."""
    from slavv_python.analytics.parity.job_registry import JobRegistry
    from slavv_python.analytics.parity.process_utils import kill_process_tree

    registry = JobRegistry()
    active_job = registry.get_job_by_run_dir(dest_run_root)
    if (
        active_job
        and active_job.status == "running"
        and is_process_alive(active_job.pid)
        and is_python_process(active_job.pid)
    ):
        if not force_kill:
            raise LaunchPreparationError(
                f"Registry still tracks active writer PID {active_job.pid}. "
                "Use --force-kill or wait for completion."
            )
        kill_process_tree(active_job.pid)
        registry.update_job(
            active_job.job_id,
            status="interrupted",
            completed_at=_now_iso(),
            exit_code=None,
            metadata={"reason": "terminated before relaunch"},
        )


def _repo_root_from_path(path: Path) -> Path:
    for parent in (path, *path.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def _now_iso() -> str:
    from .utils import now_iso

    return now_iso()


__all__ = [
    "LaunchPreparationError",
    "assert_no_conflicting_registry_writer",
    "build_foreground_probe_command",
    "prepare_detached_exact_run_launch",
    "reconcile_run_before_launch",
    "run_foreground_launch_probe",
    "run_launch_preflight",
]
