"""Host file-manager helpers for Streamlit run-location panels."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def file_manager_action_label() -> str | None:
    """Return the OS-specific reveal button label, or None when unsupported."""
    if sys.platform == "darwin":
        return "Reveal in Finder"
    if sys.platform == "win32":
        return "Open in Explorer"
    return None


def reveal_run_directory(run_dir: str | Path) -> None:
    """Open or reveal a validated run directory in the host file manager."""
    path = Path(run_dir).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise ValueError("The run directory is unavailable.")
    if sys.platform == "darwin":
        subprocess.Popen(["open", "-R", str(path)], close_fds=True)
        return
    if sys.platform == "win32":
        subprocess.Popen(["explorer.exe", str(path)], close_fds=True)
        return
    raise OSError("Opening the run folder is not supported on this platform.")


__all__ = ["file_manager_action_label", "reveal_run_directory"]
