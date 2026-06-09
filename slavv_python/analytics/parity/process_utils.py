"""Process management utilities for parity job monitoring."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore

logger = logging.getLogger(__name__)


def is_process_alive(pid: int) -> bool:
    """
    Check if a process with given PID is alive.

    Args:
        pid: Process ID to check

    Returns:
        True if process exists and is running
    """
    if psutil is None:
        # Fallback: check if PID exists (less reliable)
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    try:
        proc = psutil.Process(pid)
        return proc.is_running()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def get_process_info(pid: int) -> Optional[Dict[str, any]]:
    """
    Get information about a process.

    Args:
        pid: Process ID

    Returns:
        Dict with keys: name, cmdline, create_time, or None if process doesn't exist
    """
    if psutil is None:
        logger.warning("psutil not available, cannot get process info")
        return None

    try:
        proc = psutil.Process(pid)
        return {
            "name": proc.name(),
            "cmdline": proc.cmdline(),
            "create_time": proc.create_time(),
            "status": proc.status(),
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        logger.debug(f"Cannot get process info for PID {pid}: {e}")
        return None


def is_python_process(pid: int) -> bool:
    """
    Check if a process is a Python interpreter.

    This helps prevent PID reuse false positives.

    Args:
        pid: Process ID

    Returns:
        True if process name contains 'python'
    """
    info = get_process_info(pid)
    if info is None:
        return False

    name = info.get("name", "").lower()
    return "python" in name


def kill_process_tree(pid: int, timeout: int = 5) -> bool:
    """
    Terminate a process and all its children.

    Args:
        pid: Root process ID
        timeout: Seconds to wait for graceful termination

    Returns:
        True if process was terminated successfully
    """
    if psutil is None:
        logger.warning("psutil not available, cannot kill process tree")
        # Fallback: try simple kill
        try:
            os.kill(pid, 15)  # SIGTERM
            return True
        except (OSError, ProcessLookupError):
            return False

    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)

        # Terminate children first
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass

        # Terminate parent
        parent.terminate()

        # Wait for termination
        gone, alive = psutil.wait_procs([parent] + children, timeout=timeout)

        # Force kill any remaining processes
        for proc in alive:
            try:
                proc.kill()
            except psutil.NoSuchProcess:
                pass

        logger.info(f"Terminated process tree rooted at PID {pid}")
        return True

    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        logger.error(f"Failed to kill process tree {pid}: {e}")
        return False


def ensure_monitor_daemon_running() -> bool:
    """
    Ensure the monitor daemon is running. Start it if not.

    Returns:
        True if daemon is running or was started successfully
    """
    from slavv_python.analytics.parity.monitor_daemon import MonitorDaemon

    daemon = MonitorDaemon()

    if daemon.is_running():
        logger.debug("Monitor daemon already running")
        return True

    logger.info("Starting monitor daemon...")
    success = daemon.start()

    if success:
        logger.info("Monitor daemon started successfully")
    else:
        logger.error("Failed to start monitor daemon")

    return success


def get_python_executable() -> str:
    """
    Get the current Python executable path.

    Returns:
        Path to Python interpreter
    """
    return sys.executable


def get_daemon_pid_file() -> Path:
    """
    Get path to daemon PID file.

    Returns:
        Path to monitor_daemon.pid
    """
    workspace = Path("workspace/scratch")
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace / "monitor_daemon.pid"


def read_daemon_pid() -> Optional[int]:
    """
    Read daemon PID from file.

    Returns:
        PID if file exists and contains valid integer, None otherwise
    """
    pid_file = get_daemon_pid_file()
    if not pid_file.exists():
        return None

    try:
        return int(pid_file.read_text().strip())
    except (ValueError, OSError) as e:
        logger.warning(f"Failed to read daemon PID: {e}")
        return None


def write_daemon_pid(pid: int) -> None:
    """
    Write daemon PID to file.

    Args:
        pid: Process ID to write
    """
    pid_file = get_daemon_pid_file()
    pid_file.write_text(str(pid))


def clear_daemon_pid() -> None:
    """Remove daemon PID file."""
    pid_file = get_daemon_pid_file()
    if pid_file.exists():
        pid_file.unlink()
