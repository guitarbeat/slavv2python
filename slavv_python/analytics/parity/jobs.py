"""Detached parity job launcher for long exact-route runs."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from slavv_python.engine.state import atomic_write_text

from .constants import (
    PARITY_JOB_MANIFEST_PATH,
    PARITY_JOB_PID_PATH,
    PARITY_JOB_STDERR_PATH,
    PARITY_JOB_STDOUT_PATH,
)
from .utils import now_iso, resolve_python_commit, write_json_with_hash


def _repo_root_from_path(path: Path) -> Path:
    for parent in (path, *path.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def _append_option(command: list[str], flag: str, value: str | int | float | None) -> None:
    if value is None:
        return
    command.extend([flag, str(value)])


def build_resume_exact_run_command(
    *,
    dest_run_root: Path,
    oracle_root: Path | None = None,
    dataset_root: Path | None = None,
    stop_after: str | None = None,
    force_rerun_from: str | None = None,
    memory_safety_fraction: float | None = None,
    force: bool = False,
    skip_preflight: bool = False,
    n_jobs: int | None = None,
    python_executable: Path | None = None,
    script_path: Path | None = None,
) -> list[str]:
    """Build the command used by detached exact-route resume jobs."""
    repo_root = _repo_root_from_path(dest_run_root.resolve())
    resolved_python = python_executable or Path(sys.executable)
    resolved_script = script_path or repo_root / "scripts" / "cli" / "parity_experiment.py"

    command = [
        str(resolved_python),
        str(resolved_script),
        "resume-exact-run",
        "--dest-run-root",
        str(dest_run_root),
    ]
    _append_option(command, "--dataset-root", str(dataset_root) if dataset_root else None)
    _append_option(command, "--oracle-root", str(oracle_root) if oracle_root else None)
    _append_option(command, "--stop-after", stop_after)
    _append_option(command, "--force-rerun-from", force_rerun_from)
    _append_option(command, "--memory-safety-fraction", memory_safety_fraction)
    _append_option(command, "--n-jobs", n_jobs)
    if force:
        command.append("--force")
    if skip_preflight:
        command.append("--skip-preflight")
    return command


def launch_exact_run_job(
    *,
    dest_run_root: Path,
    oracle_root: Path | None = None,
    dataset_root: Path | None = None,
    stop_after: str | None = None,
    force_rerun_from: str | None = None,
    memory_safety_fraction: float | None = None,
    force: bool = False,
    skip_preflight: bool = False,
    n_jobs: int | None = None,
    python_executable: Path | None = None,
    script_path: Path | None = None,
) -> dict[str, Any]:
    """Start a long exact-route resume in an OS-owned background process."""
    root = dest_run_root.expanduser().resolve()
    metadata_dir = root / "99_Metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = root / PARITY_JOB_STDOUT_PATH
    stderr_path = root / PARITY_JOB_STDERR_PATH
    pid_path = root / PARITY_JOB_PID_PATH
    manifest_path = root / PARITY_JOB_MANIFEST_PATH
    repo_root = _repo_root_from_path(root)

    command = build_resume_exact_run_command(
        dest_run_root=root,
        oracle_root=oracle_root.expanduser().resolve() if oracle_root else None,
        dataset_root=dataset_root.expanduser().resolve() if dataset_root else None,
        stop_after=stop_after,
        force_rerun_from=force_rerun_from,
        memory_safety_fraction=memory_safety_fraction,
        force=force,
        skip_preflight=skip_preflight,
        n_jobs=n_jobs,
        python_executable=python_executable,
        script_path=script_path,
    )

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

    with stdout_path.open("ab") as stdout_handle, stderr_path.open("ab") as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=str(repo_root),
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            close_fds=True,
            creationflags=creationflags,
        )

    atomic_write_text(pid_path, f"{process.pid}\n")
    manifest: dict[str, Any] = {
        "kind": "parity_exact_run_job",
        "status": "launched",
        "pid": process.pid,
        "command": command,
        "cwd": str(repo_root),
        "dest_run_root": str(root),
        "oracle_root": str(oracle_root.expanduser().resolve()) if oracle_root else None,
        "dataset_root": str(dataset_root.expanduser().resolve()) if dataset_root else None,
        "stop_after": stop_after,
        "force_rerun_from": force_rerun_from,
        "n_jobs": n_jobs,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "pid_file": str(pid_path),
        "python_commit": resolve_python_commit(repo_root),
        "launched_at": now_iso(),
    }
    write_json_with_hash(manifest_path, manifest)
    return manifest


__all__ = [
    "build_resume_exact_run_command",
    "launch_exact_run_job",
]
