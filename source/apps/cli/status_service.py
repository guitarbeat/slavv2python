"""Resumable run status reporting for the SLAVV CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...runtime import RunSnapshot


def build_run_status_lines(snapshot: RunSnapshot) -> list[str]:
    """Format a multi-line status report from a run snapshot."""
    from ...runtime import build_status_lines

    return build_status_lines(snapshot)


__all__ = ["build_run_status_lines"]
