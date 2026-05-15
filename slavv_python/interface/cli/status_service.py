"""Resumable run status reporting for the SLAVV CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from slavv_python.engine.state import RunSnapshot


def build_run_status_lines(snapshot: RunSnapshot) -> list[str]:
    """Format a multi-line status report from a run snapshot."""
    from slavv_python.engine.state import build_status_lines

    return build_status_lines(snapshot)


__all__ = ["build_run_status_lines"]
