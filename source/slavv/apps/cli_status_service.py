"""Status-loading helpers for the SLAVV CLI."""

from __future__ import annotations


def load_status_snapshot(
    run_dir: str,
    *,
    snapshot_loader,
    legacy_snapshot_loader,
):
    """Load a run snapshot, falling back to legacy checkpoint metadata."""
    snapshot = snapshot_loader(run_dir)
    if snapshot is None:
        snapshot = legacy_snapshot_loader(run_dir)
    return snapshot


def build_status_output_lines(snapshot, *, status_line_builder) -> list[str]:
    """Build printable status lines from a loaded snapshot."""
    return list(status_line_builder(snapshot))


__all__ = ["build_status_output_lines", "load_status_snapshot"]
