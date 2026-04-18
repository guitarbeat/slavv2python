"""MATLAB CLI execution helpers for parity comparison."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any


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


def _escape_matlab_string(value: str) -> str:
    """Escape a Python path for a MATLAB single-quoted string literal."""
    return value.replace("\\", "/").replace("'", "''")


def _resolve_matlab_wrapper_script(
    project_root: Path, batch_script: str | None
) -> tuple[Path, Path]:
    """Resolve the MATLAB wrapper script and the upstream MATLAB checkout root."""
    if batch_script is None:
        candidates = [
            (project_root / "dev" / "scripts" / "cli" / "run_matlab_vectorization.m").resolve(),
            (project_root / "scripts" / "cli" / "run_matlab_vectorization.m").resolve(),
        ]
        for candidate in candidates:
            if candidate.exists():
                wrapper_script = candidate
                break
        else:
            raise FileNotFoundError(
                "Could not find MATLAB wrapper script 'run_matlab_vectorization.m' "
                f"in any known location: {candidates}"
            )
    else:
        wrapper_script = Path(batch_script).resolve()
        if not wrapper_script.exists():
            raise FileNotFoundError(f"Custom MATLAB wrapper script not found: {wrapper_script}")

    vectorization_candidates = [
        (project_root / "external" / "Vectorization-Public").resolve(),
        (project_root / "external" / "Vectorization-Public" / "source").resolve().parent,
    ]
    for candidate in vectorization_candidates:
        if candidate.exists():
            return wrapper_script, candidate

    raise FileNotFoundError(
        "Could not find upstream MATLAB checkout 'external/Vectorization-Public'."
    )


def _build_matlab_cli_command(
    *,
    matlab_path: str,
    vectorization_dir: Path,
    wrapper_script: Path,
    input_file: str,
    output_dir: str,
    params_file: str | None,
) -> list[str]:
    """Build a direct MATLAB CLI invocation without shell-specific wrapper scripts."""
    escaped_vectorization_dir = _escape_matlab_string(str(vectorization_dir))
    escaped_wrapper_dir = _escape_matlab_string(str(wrapper_script.parent))
    escaped_input = _escape_matlab_string(input_file)
    escaped_output = _escape_matlab_string(output_dir)

    matlab_script = (
        f"cd('{escaped_vectorization_dir}'); "
        f"addpath('{escaped_wrapper_dir}'); "
        f"run_matlab_vectorization('{escaped_input}', '{escaped_output}'"
    )
    if params_file is not None:
        matlab_script += f", '{_escape_matlab_string(params_file)}'"
    matlab_script += "); exit"

    cmd = [matlab_path]
    if os.name == "nt":
        cmd.append("-wait")
    cmd.extend(["-batch", matlab_script])
    return cmd


def _write_matlab_log_prelude(
    *,
    log_file: Path,
    input_file: str,
    output_dir: str,
    matlab_path: str,
    rendered_command: str,
    params_file: str | None,
) -> None:
    """Create the MATLAB log file with stable preamble metadata."""
    warning = ""
    if "onedrive" in output_dir.lower():
        warning = (
            "WARNING: Output directory appears to be under OneDrive sync; "
            "prefer a local non-synced drive for MATLAB outputs."
        )

    with log_file.open("w", encoding="utf-8") as handle:
        handle.write("MATLAB CLI Run Log\n")
        handle.write("===================\n")
        handle.write(f"Input file: {input_file}\n")
        handle.write(f"Output directory: {output_dir}\n")
        handle.write(f"MATLAB path: {matlab_path}\n")
        if params_file is not None:
            handle.write(f"Parameters file: {params_file}\n")
        handle.write(f"Command: {rendered_command}\n")
        handle.write(f"Start time: {time.ctime()}\n\n")
        if warning:
            handle.write(f"{warning}\n\n")


def _append_matlab_log_output(log_file: Path, stdout: str, stderr: str) -> None:
    """Append captured MATLAB stdout/stderr to the canonical run log."""
    with log_file.open("a", encoding="utf-8") as handle:
        if stdout:
            handle.write(stdout)
            if not stdout.endswith("\n"):
                handle.write("\n")
        if stderr:
            handle.write(stderr)
            if not stderr.endswith("\n"):
                handle.write("\n")


def run_matlab_vectorization(
    input_file: str,
    output_dir: str,
    matlab_path: str,
    project_root: Path,
    batch_script: str | None = None,
    params_file: str | None = None,
    *,
    discover_matlab_artifacts,
    get_matlab_info_fn,
    get_system_info_fn,
) -> dict[str, Any]:
    """Run MATLAB vectorization via CLI."""
    print("\n" + "=" * 60)
    print("Running MATLAB Implementation")
    print("=" * 60)

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    input_file = os.path.abspath(input_file)
    output_dir = os.path.abspath(output_dir)
    if params_file is not None:
        if not os.path.exists(params_file):
            raise FileNotFoundError(f"Parameters file not found: {params_file}")
        params_file = os.path.abspath(params_file)

    wrapper_script, vectorization_dir = _resolve_matlab_wrapper_script(project_root, batch_script)
    os.makedirs(output_dir, exist_ok=True)
    log_file = Path(output_dir) / "matlab_run.log"
    cmd = _build_matlab_cli_command(
        matlab_path=matlab_path,
        vectorization_dir=vectorization_dir,
        wrapper_script=wrapper_script,
        input_file=input_file,
        output_dir=output_dir,
        params_file=params_file,
    )
    rendered_command = subprocess.list2cmdline(cmd)
    _write_matlab_log_prelude(
        log_file=log_file,
        input_file=input_file,
        output_dir=output_dir,
        matlab_path=matlab_path,
        rendered_command=rendered_command,
        params_file=params_file,
    )
    print(f"Command: {rendered_command}")
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    if params_file is not None:
        print(f"Parameters file: {params_file}")

    system_info = get_system_info_fn()
    matlab_info = get_matlab_info_fn(matlab_path)
    system_info["matlab"] = matlab_info

    start_time = time.time()
    timeout_seconds = 3600
    returncode, stdout, stderr, timed_out = _run_command_with_timeout(
        cmd, vectorization_dir, timeout_seconds, shell=False
    )
    _append_matlab_log_output(log_file, stdout, stderr)
    elapsed_time = time.time() - start_time
    artifacts = discover_matlab_artifacts(output_dir)

    if timed_out:
        matlab_results = {
            "success": False,
            "elapsed_time": elapsed_time,
            "error": f"TimeoutExpired after {timeout_seconds} seconds",
            "stdout": stdout,
            "stderr": stderr,
            "system_info": system_info,
            "output_dir": output_dir,
            "params_file": params_file or "",
            "log_file": str(log_file),
        }
        matlab_results |= artifacts
        return matlab_results

    if returncode != 0:
        matlab_results = {
            "success": False,
            "elapsed_time": elapsed_time,
            "error": f"MATLAB exited with code {returncode}",
            "stdout": stdout,
            "stderr": stderr,
            "system_info": system_info,
            "output_dir": output_dir,
            "params_file": params_file or "",
            "log_file": str(log_file),
        }
        matlab_results |= artifacts
        return matlab_results

    matlab_results = {
        "success": True,
        "elapsed_time": elapsed_time,
        "output_dir": output_dir,
        "params_file": params_file or "",
        "stdout": stdout,
        "stderr": stderr,
        "system_info": system_info,
        "log_file": str(log_file),
    }
    return matlab_results | artifacts
