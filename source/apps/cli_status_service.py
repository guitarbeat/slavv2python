"""Status-loading helpers for the SLAVV CLI."""

from __future__ import annotations


def load_status_snapshot(
        run_dir: str,
        *,
        snapshot_loader,
):
    """Load a run snapshot from the structured run directory."""
    return snapshot_loader(run_dir)


def build_status_output_lines(snapshot, *, status_line_builder) -> list[str]:
    """Build printable status lines from a loaded snapshot."""
    return list(status_line_builder(snapshot))


__all__ = ["build_status_output_lines", "load_status_snapshot"]
